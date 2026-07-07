"""The wave-4 gate: end-to-end oracles over a module-only toy model.

Tracer diffusion on a tiny periodic grid — one module declaring a
PROGNOSTIC field ``c`` with a ``@fr.term`` computing diffusion via
the grid's ``diff`` operator chain, a module-provided parameter
``kappa``, and ``AdamBashforth(order=2)``. NO fake core. The
mandated oracles:

1. treedef stability across ``advance``;
2. the compile counter: repeated ``advance`` and a kappa
   ``update_parameters`` sweep compile NOTHING new;
3. ``advance(2); advance(3)`` == ``advance(5)`` bitwise, on the
   shared jit-cache entry;
4. snapshot round trip bitwise + fingerprint-mismatch refusal;
5. the S5 NaN abort (PanicError at the boundary, exact first-bad
   iteration, debug_nan replay);
6. physics sanity: monotone variance decay, mass conservation.

Single-device only — the orchestrator runs the 4-device sweep.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import dtype_real, jaxify
from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.io.streams import SnapshotMismatchError
from fridom.framework2.model.declarations import FieldDeclaration
from fridom.framework2.model.model import Model, chunk_cache_size
from fridom.framework2.model.module import Module
from fridom.framework2.model.parameters import ParameterDeclaration
from fridom.framework2.model.results import PanicError
from fridom.framework2.model.space_patterns import Collocated
from fridom.framework2.model.terms import term
from fridom.framework2.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)

N = 16
DT = 1e-3
KAPPA = 5e-3


# ================================================================
#  The toy model: tracer diffusion (API-sketch style)
# ================================================================
@partial(jaxify, dynamic=("kappa",))
class TracerDiffusion(Module):

    """dc/dt = kappa * d2c/dx2 via the diff operator chain.

    ``kappa`` is a dynamic leaf provided as ``tracer.kappa`` — the
    term reads its OWN live leaf (the owner-read; sweeps ride
    ``update_parameters``). The traced-scalar scaling is spelled on
    the raw data, so the module declares its stencil halo and is
    halo-trace exempt (V-N2); the declared width covers the two
    chained first differences.
    """

    def __init__(self, kappa=KAPPA):
        self.kappa = jnp.asarray(kappa, dtype=dtype_real())

    extra_halo = HaloSpec({"x": 2})

    field_declarations = (
        FieldDeclaration("c", space=Collocated(),
                         long_name="Tracer"),)
    parameter_declarations = (
        ParameterDeclaration("tracer.kappa", attr="kappa",
                             units="m^2/s"),)

    @term(advances=("c",))
    def diffuse(self, state, _ctx):
        flux = state["c"].diff("x")      # center -> staggered
        lap = flux.diff("x")             # staggered -> center
        return {"c": lap.with_data(self.kappa * lap.data)}


# ================================================================
#  Fixtures and helpers
# ================================================================
def make_grid():
    return Grid((IntervalMesh(N, (0.0, 1.0), periodic=True,
                              name="x"),))


def make_model(grid=None, *, dt=DT, order=2, kappa=KAPPA,
               **kwargs):
    if grid is None:
        grid = make_grid()
    return Model(grid=grid, modules=(TracerDiffusion(kappa),),
                 time_stepper=AdamBashforth(dt, order=order),
                 **kwargs)


def tracer_ic():
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    return np.sin(2 * np.pi * x) + 1.0


def c_data(model):
    return np.asarray(model.state["c"].data)


# ================================================================
#  Oracle 1 — treedef stability
# ================================================================
def test_carry_treedef_identical_before_and_after_advance():
    model = make_model()
    model.set_fields(c=tracer_ic())
    before = jax.tree_util.tree_structure(model._carry)
    model.advance(5)
    after = jax.tree_util.tree_structure(model._carry)
    assert before == after


# ================================================================
#  Oracle 2 — the compile counter
# ================================================================
def test_second_advance_compiles_nothing(compile_counter):
    model = make_model()
    model.set_fields(c=tracer_ic())
    model.advance(5)                     # warm every code path
    compile_counter.reset()
    model.advance(5)
    assert compile_counter.count == 0


def test_kappa_sweep_compiles_nothing(compile_counter):
    model = make_model()
    model.set_fields(c=tracer_ic())
    # warm the sweep path once (update + rewarm + advance + the
    # copy-on-read state view)
    model.update_parameters({"tracer.kappa": 2 * KAPPA})
    model.advance(5)
    c_data(model)
    compile_counter.reset()
    reference = chunk_cache_size()
    results = []
    for kappa in (1e-3, 3e-3, 7e-3):
        model.update_parameters({"tracer.kappa": kappa})
        model.advance(5)
        results.append(c_data(model))
    assert compile_counter.count == 0
    assert chunk_cache_size() == reference
    # the sweep really changed the physics (live leaves, one trace)
    assert not np.array_equal(results[0], results[1])


# ================================================================
#  Oracle 3 — repeated advance == one uninterrupted run
# ================================================================
def test_split_advance_is_bitwise_and_shares_the_cache(
        compile_counter):
    grid = make_grid()
    whole = make_model(grid)
    split = make_model(grid)             # identical re-assembly
    whole.set_fields(c=tracer_ic())
    split.set_fields(c=tracer_ic())
    whole.advance(5)
    compile_counter.reset()
    reference = chunk_cache_size()
    split.advance(2)
    split.advance(3)
    # the same record and carry structure: the SAME cache entry
    assert compile_counter.count == 0
    assert chunk_cache_size() == reference
    assert np.array_equal(c_data(whole), c_data(split))
    assert int(whole.clock.it) == int(split.clock.it)
    assert float(whole.clock.elapsed) == float(
        split.clock.elapsed)


# ================================================================
#  Oracle 4 — snapshot round trip
# ================================================================
def test_snapshot_roundtrip_is_bitwise(tmp_path):
    grid = make_grid()
    model = make_model(grid)
    model.set_fields(c=tracer_ic())
    model.advance(3)
    model.snapshot(tmp_path / "snap")
    model.advance(4)                     # the uninterrupted run
    reference = c_data(model)
    resumed = make_model(grid)           # same record, same cache
    resumed.load_snapshot(tmp_path / "snap")
    assert int(resumed.clock.it) == 3
    resumed.advance(4)
    assert np.array_equal(reference, c_data(resumed))
    assert float(model.clock.elapsed) == float(
        resumed.clock.elapsed)


def test_snapshot_refuses_a_different_assembly(tmp_path):
    model = make_model()
    model.advance(2)
    model.snapshot(tmp_path / "snap")
    variant = make_model(order=3)        # different stepper statics
    with pytest.raises(SnapshotMismatchError,
                       match="stepper statics"):
        variant.load_snapshot(tmp_path / "snap")


# ================================================================
#  Oracle 5 — the NaN abort and the debug replay
# ================================================================
def test_nan_abort_at_the_boundary_with_exact_iteration():
    model = make_model(chunk_size=8)
    model.set_fields(c=tracer_ic())
    # a wildly unstable kappa: the explosion overflows to inf
    # after a deterministic number of steps
    model.update_parameters({"tracer.kappa": 1e6})
    with pytest.raises(PanicError) as err:
        model.advance(128, debug_nan=True)
    first_bad = err.value.first_bad_it
    boundary = err.value.partial.steps_done
    assert first_bad is not None
    # the abort fires at the chunk boundary AFTER the bad step
    assert boundary % 8 == 0
    assert boundary - 8 < first_bad <= boundary
    # the debug replay pinpoints the exact same first-bad step
    assert model.replay_nan() == first_bad
    # entry guard: a panicked carry refuses to advance
    assert model.panicked
    with pytest.raises(PanicError):
        model.advance(1)
    # reset is a sanctioned resume path
    model.reset()
    assert not model.panicked
    model.set_fields(c=tracer_ic())
    model.update_parameters({"tracer.kappa": KAPPA})
    model.advance(8)


def test_nan_injected_state_flags_step_one():
    model = make_model()
    model.set_fields(c=np.full(N, np.nan))
    with pytest.raises(PanicError) as err:
        model.advance(3)
    assert err.value.first_bad_it == 1
    assert err.value.partial.steps_done == 1


# ================================================================
#  Oracle 6 — physics sanity
# ================================================================
def test_diffusion_decays_the_variance_monotonically():
    model = make_model()
    model.set_fields(c=tracer_ic())
    variances = []
    for _ in range(30):
        model.advance(1)
        c = c_data(model)
        variances.append(float(((c - c.mean()) ** 2).sum()))
    diffs = np.diff(np.asarray(variances))
    assert (diffs < 0.0).all()           # strictly monotone decay


def test_total_tracer_mass_is_conserved():
    model = make_model()
    model.set_fields(c=tracer_ic())
    mass0 = float(c_data(model).sum())
    model.advance(50)
    mass1 = float(c_data(model).sum())
    # the periodic FD laplacian telescopes: conservation to fp
    # roundoff of the accumulated sums
    assert mass1 == pytest.approx(mass0, rel=1e-12, abs=1e-12)
