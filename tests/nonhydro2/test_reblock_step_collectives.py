"""Whole-step collective regression: a walled sharded axis is not slow.

The staggered face velocity on a walled axis has ``n_cells - 1`` DOFs.
When that axis is the sharded one, the reblock gate used to exclude the
divisible-deficit leg (``n = cells * P - 1``), so the hot loop's
true-frame excursions (AB3 ``_weighted``, the coriolis lifts) took the
global slice/concat reblock, which GSPMD lowers to replicate + rescatter
collectives on every step (the measured 2.3x 4-GPU slowdown on walled-x).

Admitting the deficit leg to the shard-local plan collapses the whole
x-walled step's collectives onto the fully-periodic baseline. This test
compiles both configs in one process and asserts the collective totals
are equal -- the end-to-end guard that the deficit leg never regresses
back to the global path. Counted as opcode definition lines and summed
across the collective family, so it is backend-agnostic (on CPU
all-to-all decomposes into collective-permute; the same summation is
applied to both configs).
"""
from __future__ import annotations

import collections
import re

import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.model import _compile_chunk
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

#: small, divisible by the forced-4 mesh so the sharded axis blocks
#: evenly (the deficit leg is n = cells * P - 1); linear, one-step chunk
#: -- kept minimal because the cost here is the pressure-solve compile.
N = 8
TWO_PI = 2.0 * np.pi

#: collective opcode roots; async GPU spellings (``-start``/``-done``)
#: and the CPU ``collective-permute`` decomposition all match by prefix,
#: so the same summation is backend-agnostic when applied to both configs.
_COLLECTIVE_ROOTS = ("all-to-all", "collective-permute", "all-gather",
                     "all-reduce", "reduce-scatter")


def _make_model(walled: tuple[str, ...]):
    """Build a linear f-plane nonhydro model with the named axes walled."""
    meshes = tuple(
        fr.spatial.meshes.IntervalMesh(
            N, (0.0, 1.0) if name in walled else (0.0, TWO_PI),
            periodic=(name not in walled), name=name)
        for name in ("x", "y", "z"))
    return nh.Model(
        grid=fr.spatial.Grid(meshes),
        core=nh.Core(aspect_ratio=0.5),
        time_stepper=AdamBashforth(0.02, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=1.0),
        advection=False,
        chunk_size=1)


def _collective_total(walled: tuple[str, ...]) -> int:
    """Sum the collective-family opcode definition lines of one step."""
    model = _make_model(walled)
    text = _compile_chunk(model._artifacts.record, model._carry,
                          model._stepper, 1).as_text()
    ops = collections.Counter(
        re.findall(r"= \S+ ([a-z][a-z0-9-]*)\(", text))
    return sum(count for op, count in ops.items()
               if any(root in op for root in _COLLECTIVE_ROOTS))


@pytest.mark.multi_device
def test_x_walled_step_collectives_equal_periodic():
    # x is the sharded axis (default_layout shards axis 0), so a walled
    # x makes the face velocity u the n_cells - 1 deficit leg. With the
    # leg admitted to the shard-local plan, the whole x-walled step's
    # collective total must equal the fully-periodic baseline (before
    # the fix it was ~7x larger: the per-excursion replicate/rescatter).
    periodic = _collective_total(())
    x_walled = _collective_total(("x",))
    assert x_walled == periodic


@pytest.mark.multi_device
def test_xyz_walled_step_collectives_match_off_axis_wall():
    # xyz-walled walls the sharded axis (x) too; its collective total
    # must match a wall that is OFF the sharded axis (z-walled) -- i.e.
    # walling the sharded axis adds no collectives over walling a
    # non-sharded one. Guards the same deficit-leg collapse from the
    # walled-vs-walled side (both carry the trig-transform solve, so the
    # only difference is whether the wall sits on the sharded axis).
    z_walled = _collective_total(("z",))
    xyz_walled = _collective_total(("x", "y", "z"))
    assert xyz_walled == z_walled
