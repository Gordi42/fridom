"""FV-vs-nodal step fusion guards (perf_guard_plan §1.5 gap E, §4.1, §5.3).

The finite-volume C-grid and its point-value (``family="nodal"``)
sibling run bitwise-identical trajectories (bench_step.py's
``nh_flat_*``/``*_nodal`` pairs). Their step time being at parity rests
on the ``flux_diff @ reconstruct`` composition fusing all the way down
to the nodal finite-difference kernels; a lowering change could quietly
unfuse the FV default path without any bitwise test noticing. These are
the deterministic non-timing guards for that (techniques #4/#5 of the
plan): compile the full model chunk for both families on the *same*
backend, in the *same* process, and compare compiled-HLO opcode
Counters.

Two regimes, measured 2026-07-18 (jax 0.10.2):

* **Periodic (uniform rows)** — FV and nodal lower to *byte-identical*
  HLO, so their per-op Counters are exactly equal on 1 and on 4 CPU
  devices. The guard asserts full Counter equality; any divergence is
  an unfusion to review, not physics.
* **Walled-x / mapped (boundary-special rows)** — one-sided
  reconstruction, Outer/Inner shapes and zero-padded wall fluxes break
  pattern uniformity, so FV carries a small op-count overhead over
  nodal (owner ruling §5.3: ratchet the gap, do not equalize it). The
  guard gates the FV/nodal op-count ratio against a committed baseline
  with a +10% relative band. This pins a compiler artifact; the failure
  message routes a jax-upgrade drift to the regen knob.
"""
from __future__ import annotations

import collections
import json
import os
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.model import _compile_chunk
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

#: tiny cube: the cost here is the pressure-solve compile, so keep the
#: grid minimal (the differential is size-independent in kind).
N = 16
TWO_PI = 2.0 * np.pi

#: opcode-definition regex, matching test_reblock_step_collectives.py:
#: async GPU spellings and the CPU decompositions all fall out the same
#: way on both families, so the differential stays backend-agnostic.
_OPCODE = re.compile(r"= \S+ ([a-z][a-z0-9-]*)\(")

#: committed FV/nodal op-count ratchet, keyed "case|backend|devices".
_BASELINE = Path(__file__).parent / "fv_ratchet_baseline.json"

#: relative slack above the recorded ratio before the ratchet trips.
_RATCHET_TOL = 0.10

#: walled/mapped ratchet cases (the boundary-special FV rows).
_RATCHET_CASES = {
    "walled_x": {"periodic_x": False, "periodic_z": True, "mapped": False},
    "mapped": {"periodic_x": True, "periodic_z": False, "mapped": True},
}


# ================================================================
#  Model construction (mirrors benchmarks/model/bench_step.py at 16^3)
# ================================================================
def _make_model(*, mapped=False, periodic_x=True, periodic_z=False,
                advection=False, family=None):
    """Linear/advective f-plane nonhydro model, FV or nodal sibling."""
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, TWO_PI),
                                        periodic=periodic_x, name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, TWO_PI), periodic=True,
                                        name="y")
    mz = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                        periodic=periodic_z, name="z")
    mapping = None
    if mapped:
        mapping = fr.spatial.CoordinateMapping(
            maps={"zp": lambda z, H: z * H},
            params={"H": lambda x: 1.0 + 0.2 * jnp.sin(x)})
    grid = fr.spatial.Grid((mx, my, mz), mapping=mapping)
    dt = 0.25 * TWO_PI / N if advection else 0.02
    return nh.Model(
        grid=grid,
        core=nh.Core(aspect_ratio=0.5, family=family, pressure_iterations=8),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=1.0),
        advection=advection,
        chunk_size=1)


def _opcounts(model) -> collections.Counter:
    """Compiled-HLO opcode Counter of one model step (chunk of 1)."""
    text = _compile_chunk(model._artifacts.record, model._carry,
                          model._stepper, 1).as_text()
    return collections.Counter(_OPCODE.findall(text))


def _counter_mismatch(fv: collections.Counter,
                      nodal: collections.Counter) -> str:
    """Per-op FV/nodal differences, for a readable assertion message."""
    keys = sorted(set(fv) | set(nodal))
    diffs = {k: (fv.get(k, 0), nodal.get(k, 0))
             for k in keys if fv.get(k, 0) != nodal.get(k, 0)}
    return (
        "periodic FV lost fusion parity with the nodal sibling "
        f"(op: fv/nodal): {diffs}. The periodic FV step must lower to "
        "the same kernels as nodal; a divergence here is an unfused "
        "default path to investigate (perf_guard_plan §1.5 gap E), not "
        "physics -- the trajectories stay bitwise-identical.")


# ================================================================
#  E1 -- uniform-parity: periodic FV == nodal, opcode for opcode
# ================================================================
@pytest.mark.parametrize("advection", [
    pytest.param(False, id="linear"),
    pytest.param(True, id="advective"),
])
def test_periodic_fv_matches_nodal_opcounts(advection):
    # the periodic (uniform-row) case: FV and nodal are the same
    # compiled program, so every opcode count matches. Device-count
    # robust -- the equality is a differential, so it holds on 1 and on
    # 4 devices (verified 2026-07-18), hence unmarked.
    fv_model = _make_model(periodic_x=True, periodic_z=True,
                           advection=advection, family=None)
    nodal_model = _make_model(periodic_x=True, periodic_z=True,
                              advection=advection, family="nodal")
    # not a vacuous self-comparison: the default resolves to FV, the
    # sibling is explicitly nodal
    assert fv_model.grid.default_family == "fv"
    assert nodal_model.grid.default_family == "nodal"
    fv = _opcounts(fv_model)
    nodal = _opcounts(nodal_model)
    assert fv, "no opcodes counted -- the chunk did not compile"
    assert fv == nodal, _counter_mismatch(fv, nodal)


# ================================================================
#  E2 -- walled/mapped ratchet against a committed baseline (§5.3)
# ================================================================
def _load_baseline() -> dict:
    if _BASELINE.exists():
        return json.loads(_BASELINE.read_text())
    return {}


def _save_baseline(data: dict) -> None:
    _BASELINE.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize("case", list(_RATCHET_CASES),
                         ids=list(_RATCHET_CASES))
def test_walled_mapped_fv_ratchet(case):
    # the boundary-special rows carry a small FV overhead over nodal
    # (accepted, ratcheted -- owner ruling §5.3). Gate the FV/nodal
    # op-count ratio against the committed baseline with a +10% band.
    kw = _RATCHET_CASES[case]
    fv = sum(_opcounts(_make_model(**kw, family=None)).values())
    nodal = sum(_opcounts(_make_model(**kw, family="nodal")).values())
    ratio = fv / nodal

    key = f"{case}|{jax.default_backend()}|{jax.device_count()}"
    baseline = _load_baseline()
    if os.environ.get("FRIDOM_REGEN_FV_RATCHET"):
        # deliberate re-baseline (FRIDOM_REGEN_HLO_GOLDEN precedent):
        # record this backend/device_count and review the JSON diff.
        baseline[key] = {"fv_ops": fv, "nodal_ops": nodal,
                         "ratio": ratio}
        _save_baseline(baseline)
        baseline = _load_baseline()

    if key not in baseline:
        # the CPU single-device entry is the committed CI gate; any
        # other backend/device_count is opt-in via regen.
        pytest.skip(
            f"no FV ratchet baseline for {key!r} -- the CPU "
            "single-device entry is the committed CI gate. Add one on "
            "this backend/device_count with FRIDOM_REGEN_FV_RATCHET=1.")

    base_ratio = baseline[key]["ratio"]
    limit = base_ratio * (1.0 + _RATCHET_TOL)
    assert ratio <= limit, (
        f"FV/nodal op-count ratio for {case!r} on {key!r} rose to "
        f"{ratio:.4f} (baseline {base_ratio:.4f}, +{_RATCHET_TOL:.0%} "
        f"limit {limit:.4f}). This gate PINS A COMPILER ARTIFACT: a "
        "jax/jaxlib upgrade can legitimately move the op counts for "
        "reasons unrelated to fridom. Re-measure and regenerate the "
        "baseline (FRIDOM_REGEN_FV_RATCHET=1) before attributing this "
        "failure to your change -- see "
        "design/plans/active/perf_guard_plan.md §1.5.")
