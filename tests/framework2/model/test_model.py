"""Tests for the Model host surface (model/model.py).

Covers construction (assembly step 8: carry allocation, AUX
defaults through the one re-materialization path, stepper init,
clock, panic ledger), the host read surface (copy-on-read state,
ParameterView, module lookup, DiagnosticsNamespace laziness), the
lifecycle mutators and their exact panic-flag semantics
(set_fields / set_state / set_aux / update_parameters / reset —
the three-operation matrix of 02_rules), and snapshot/load through
the io store (bitwise leaves, fingerprint diff, dt checks).
"""
from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import dtype_real, jaxify
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.model.io.snapshots import Snapshots, read_manifest
from fridom.model.io.streams import (
    IOCollisionError,
    SnapshotMismatchError,
)
from fridom.model.io.triggers import every
from fridom.model.declarations import (
    FieldDeclaration,
    Lifecycle,
)
from fridom.model.errors import (
    AssemblyError,
    MissingParameterError,
)
from fridom.model.model import Model, ModelState
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration
from fridom.model.results import AdvanceResult, PanicError
from fridom.spatial.space_patterns import (
    Collocated,
    Profile,
    Staggered,
)
from fridom.model.terms import term
from fridom.model.time_dependent import Ramp
from fridom.model.time_steppers.adam_bashforth import (
    ABState,
    AdamBashforth,
)

DT = 1e-3
N = 8


# ================================================================
#  Toy modules (API-sketch style, on the real fr.Module)
# ================================================================
@partial(jaxify, dynamic=("nu",))
class Core(Module):

    """Two-field PROGNOSTIC core plus bound package diagnostics."""

    def __init__(self, nu=1e-3):
        self.nu = jnp.asarray(nu, dtype=dtype_real())

    field_declarations = (
        FieldDeclaration("u", space=Staggered("x"),
                         long_name="Velocity"),
        FieldDeclaration("b", space=Collocated(),
                         long_name="Buoyancy"),
    )
    parameter_declarations = (
        ParameterDeclaration("core.nu", attr="nu", units="1"),)

    # duck-typed package-diagnostics channel (model.md open q. 6)
    diagnostics: ClassVar = {
        "b_sum": lambda state, _params: state["b"].data.sum(),
        "needs_ghost": lambda _state, params: params["ghost.p"],
    }

    @term(advances=("u",))
    def pressure_force(self, state, _ctx):
        return {"u": state["b"].diff("x")}

    @term(advances=("b",))
    def restoring(self, state, _ctx):
        return {"b": state["u"].to(state["b"].function_space)
                * (-1.0)}


@partial(jaxify, dynamic=("n2",))
class Background(Module):

    """Owner-derived AUX (bg), consented AUX (q), consented DIAG."""

    def __init__(self, n2=1e-5):
        self.n2 = jnp.asarray(n2, dtype=dtype_real())

    def _make_bg(self, grid, space):
        data = jnp.full(space.shape, 2.0 * self.n2)
        return grid.create_field(space, data=data, name="bg")

    field_declarations = (
        FieldDeclaration("bg", space=Profile("x"),
                         lifecycle=Lifecycle.AUXILIARY,
                         default=_make_bg, units="1"),
        FieldDeclaration("q", space=Collocated(),
                         lifecycle=Lifecycle.AUXILIARY,
                         default=1.5, host_writable=True),
        FieldDeclaration("acc", space=Collocated(),
                         lifecycle=Lifecycle.DIAGNOSTIC,
                         host_writable=True),
    )
    parameter_declarations = (
        ParameterDeclaration("background.n2", attr="n2",
                             units="1/s^2"),)


@partial(jaxify, dynamic=("value",))
class Provider(Module):

    """Pure parameter provider (scalar or Ramp leaf)."""

    def __init__(self, value=1.0):
        self.value = (value if isinstance(value, Ramp)
                      else jnp.asarray(value, dtype=dtype_real()))

    parameter_declarations = (
        ParameterDeclaration("toy.value", attr="value", units="1"),)


# ================================================================
#  Fixtures and helpers
# ================================================================
def make_grid():
    return Grid((IntervalMesh(N, (0.0, 1.0), periodic=True,
                              name="x"),))


def make_model(grid=None, modules=None, stepper=None, **kwargs):
    if grid is None:
        grid = make_grid()
    if modules is None:
        modules = (Core(), Background())
    if stepper is None:
        stepper = AdamBashforth(DT, order=2)
    return Model(grid=grid, modules=modules, time_stepper=stepper,
                 **kwargs)


@pytest.fixture
def model():
    return make_model()


def ic(shift=0.0):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    return np.sin(2 * np.pi * x) + shift


def panic_the_model(model):
    """Drive the model into a real S5 panic (NaN in the state)."""
    bad = np.full(N, np.nan)
    model.set_fields(b=bad)
    with pytest.raises(PanicError):
        model.advance(1)
    assert model.panicked


# ================================================================
#  Construction — assembly + step 8 (carry allocation)
# ================================================================
def test_allocation_prognostic_and_diagnostic_zero(model):
    carry = model.carry
    assert isinstance(carry, ModelState)
    assert (carry.state["u"].data == 0.0).all()
    assert (carry.state["b"].data == 0.0).all()
    assert (carry.state["acc"].data == 0.0).all()


def test_allocation_aux_defaults_via_remat_path(model):
    carry = model.carry
    # bg: the owner-method default with assembly-time leaves
    assert np.allclose(np.asarray(carry.state["bg"].data), 2e-5)
    # q: constant default; host-writable defaults are
    # initialization-only but DO run at allocation
    assert (np.asarray(carry.state["q"].data) == 1.5).all()


def test_allocation_stepper_state_and_clock(model):
    carry = model.carry
    assert isinstance(carry.stepper_state, ABState)
    assert len(carry.stepper_state.history) == 2
    assert int(carry.stepper_state.warmup) == 0
    assert float(carry.clock.elapsed) == 0.0
    assert int(carry.clock.it) == 0
    assert not bool(carry.panic.flag)
    assert not model.panicked


def test_io_rejects_snapshots(tmp_path):
    snap = Snapshots(tmp_path / "snaps", trigger=every(steps=1))
    with pytest.raises(IOCollisionError, match="run-config only"):
        make_model(io=(snap,))


def test_chunk_size_validation():
    with pytest.raises(ValueError, match="chunk_size"):
        make_model(chunk_size=0)


def test_repr_is_the_report_header(model):
    assert repr(model) == model.report.header
    assert "AdamBashforth" in repr(model)


# ================================================================
#  Read surface
# ================================================================
def test_state_is_copy_on_read(model):
    view = model.state
    assert view is not model._carry.state
    assert view["u"] is not model._carry.state["u"]
    # the copy survives a later advance (donation-safe)
    model.set_fields(b=ic())
    view = model.state
    before = np.asarray(view["b"].data)
    model.advance(2)
    assert np.array_equal(np.asarray(view["b"].data), before)


def test_clock_and_grid_and_name_reads():
    grid = make_grid()
    model = make_model(grid=grid, name="host-reads")
    assert model.grid is grid
    assert model.name == "host-reads"
    model.advance(3)
    assert int(model.clock.it) == 3
    assert float(model.clock.elapsed) == pytest.approx(3 * DT)


def test_parameters_view_live_and_hinted(model):
    params = model.parameters
    assert "core.nu" in params
    assert "stepper.dt" in params
    assert float(params["core.nu"]) == pytest.approx(1e-3)
    assert float(params["stepper.dt"]) == pytest.approx(DT)
    assert set(params) >= {"core.nu", "background.n2",
                           "stepper.dt"}
    assert len(params) >= 3
    with pytest.raises(MissingParameterError, match="provided"):
        params["nope.value"]


def test_parameters_ramp_returned_raw_and_at_time():
    ramp = Ramp(0.0, 1.0, period=10.0)
    model = make_model(modules=(Core(), Provider(ramp)))
    raw = model.parameters["toy.value"]
    assert isinstance(raw, Ramp)
    resolved = model.parameters.at_time(5.0)
    assert float(resolved["toy.value"]) == pytest.approx(0.5)


def test_parameters_info_returns_the_declaration(model):
    info = model.parameters.info("core.nu")
    assert info.units == "1"
    assert info.attr == "nu"


def test_module_lookup_typed_and_ambiguous(model):
    background = model.module(Background)
    assert isinstance(background, Background)
    with pytest.raises(LookupError, match="candidates"):
        model.module(Module)          # matches both modules
    with pytest.raises(LookupError, match="no live module"):
        model.module(Provider)


def test_module_lookup_returns_the_live_module(model):
    model.update_parameters({"background.n2": 4e-5})
    live = model.module(Background)
    assert float(live.n2) == pytest.approx(4e-5)


def test_diagnostics_bound_and_evaluated(model):
    model.set_fields(b=np.full(N, 2.0))
    assert float(model.diagnostics.b_sum()) == pytest.approx(
        2.0 * N)
    # explicit state argument
    assert float(model.diagnostics.b_sum(model.state)) == (
        pytest.approx(2.0 * N))


def test_diagnostics_missing_parameter_is_lazy(model):
    bound = model.diagnostics.needs_ghost   # binding never raises
    with pytest.raises(MissingParameterError):
        bound()                             # the CALL raises


def test_diagnostics_unknown_name_lists_available(model):
    with pytest.raises(AttributeError, match="b_sum"):
        _ = model.diagnostics.does_not_exist


def test_dir_lists_diagnostics(model):
    assert "b_sum" in dir(model.diagnostics)


# ================================================================
#  set_fields / set_state (PROGNOSTIC writes; clear panic)
# ================================================================
def test_set_fields_accepts_array_callable_field(model):
    model.set_fields(b=ic())
    assert np.allclose(np.asarray(model.state["b"].data), ic())
    model.set_fields(b=lambda x: 0.0 * x + 3.0)
    assert (np.asarray(model.state["b"].data) == 3.0).all()
    donor = model.state["b"].with_data(jnp.full(N, 7.0))
    model.set_fields(b=donor)
    assert (np.asarray(model.state["b"].data) == 7.0).all()


def test_set_fields_rejects_unknown_and_non_prognostic(model):
    with pytest.raises(ValueError, match="unknown field"):
        model.set_fields(w=np.zeros(N))
    with pytest.raises(ValueError, match="set_aux"):
        model.set_fields(bg=np.zeros(N))


def test_set_fields_rejects_wrong_shape(model):
    with pytest.raises(ValueError, match="true shape"):
        model.set_fields(b=np.zeros(N + 1))


def test_set_fields_clears_panic(model):
    panic_the_model(model)
    # the FLAG clears (a resume path); note the multistep ring may
    # still hold non-finite tendencies — the full NaN-resume
    # recipes are reset() and load_snapshot()
    model.set_fields(b=ic(), u=np.zeros(N))
    assert not model.panicked
    assert not bool(model.carry.panic.flag)


def test_set_state_partial_overwrite_ignores_extras(model):
    other = make_model(grid=model.grid)
    other.set_fields(b=ic(), u=np.full(N, 0.25))
    source = other.state                  # carries all lifecycles
    model.set_fields(u=np.full(N, -1.0))
    model.set_state(source)
    assert np.allclose(np.asarray(model.state["b"].data), ic())
    assert (np.asarray(model.state["u"].data) == 0.25).all()
    # AUX/DIAG components in the input were ignored: bg keeps the
    # incumbent default, not the donor's
    assert np.allclose(np.asarray(model.state["bg"].data), 2e-5)


def test_set_state_missing_components_left_untouched(model):
    model.set_fields(u=np.full(N, 0.5))
    partial_state = type(model.state)(
        {"b": model.state["b"].with_data(jnp.full(N, 9.0))})
    model.set_state(partial_state)
    assert (np.asarray(model.state["b"].data) == 9.0).all()
    assert (np.asarray(model.state["u"].data) == 0.5).all()


def test_set_state_clears_panic(model):
    panic_the_model(model)
    model.set_state(make_model(grid=model.grid).state)
    assert not model.panicked


# ================================================================
#  set_aux (the consented host write; panic KEPT)
# ================================================================
def test_set_aux_writes_consented_aux_and_diag(model):
    model.set_aux(q=np.full(N, -2.0))
    assert (np.asarray(model.state["q"].data) == -2.0).all()
    model.set_aux(acc=np.full(N, 0.5))    # consented DIAGNOSTIC
    assert (np.asarray(model.state["acc"].data) == 0.5).all()


def test_set_aux_rejects_unconsented_components(model):
    with pytest.raises(ValueError, match="host_writable"):
        model.set_aux(bg=np.zeros(N))     # AUX without consent
    with pytest.raises(ValueError, match="host_writable"):
        model.set_aux(u=np.zeros(N))      # PROGNOSTIC
    with pytest.raises(ValueError, match="unknown field"):
        model.set_aux(nope=np.zeros(N))


def test_set_aux_keeps_the_panic_flag(model):
    panic_the_model(model)
    model.set_aux(q=np.full(N, 1.0))      # legal while panicked
    assert model.panicked                 # NOT a resume path
    with pytest.raises(PanicError):
        model.advance(1)


def test_set_aux_rewarm_reinits_the_stepper(model):
    model.advance(3)
    assert int(model.carry.stepper_state.warmup) == 1
    model.set_aux(q=np.full(N, 1.0), rewarm=True)
    assert int(model.carry.stepper_state.warmup) == 0


# ================================================================
#  update_parameters (functional writes; panic KEPT)
# ================================================================
def test_update_parameters_writes_module_leaves(model):
    model.update_parameters({"core.nu": 2e-3})
    assert float(model.parameters["core.nu"]) == pytest.approx(
        2e-3)


def test_update_parameters_time_step_provider(model):
    model.update_parameters({"stepper.dt": 2 * DT})
    assert float(model.parameters["stepper.dt"]) == (
        pytest.approx(2 * DT))
    model.advance(1)
    assert float(model.clock.elapsed) == pytest.approx(2 * DT)


def test_update_parameters_rematerializes_owner_aux(model):
    model.update_parameters({"background.n2": 3e-5})
    assert np.allclose(np.asarray(model.state["bg"].data), 6e-5)


def test_update_parameters_skips_host_writable_aux(model):
    model.set_aux(q=np.full(N, 42.0))     # the host write wins
    model.update_parameters({"background.n2": 3e-5})
    assert (np.asarray(model.state["q"].data) == 42.0).all()


def test_update_parameters_rewarm_default_and_knob(model):
    model.advance(3)
    model.update_parameters({"core.nu": 5e-3})
    assert int(model.carry.stepper_state.warmup) == 0
    model.advance(3)
    model.update_parameters({"core.nu": 6e-3}, rewarm=False)
    assert int(model.carry.stepper_state.warmup) == 1


def test_update_parameters_spec_change_requires_reassembly(model):
    ramp = Ramp(0.0, 1.0, period=10.0)
    with pytest.raises(AssemblyError, match="re-assemble"):
        model.update_parameters({"core.nu": ramp})


def test_update_parameters_unknown_name_is_hinted(model):
    with pytest.raises(MissingParameterError, match="provided"):
        model.update_parameters({"nope.value": 1.0})


def test_update_parameters_keeps_the_panic_flag(model):
    panic_the_model(model)
    model.update_parameters({"core.nu": 9e-3})
    assert model.panicked                 # NOT a resume path
    with pytest.raises(PanicError):
        model.advance(1)


# ================================================================
#  reset (the three-operation matrix; OptimalBalance semantics)
# ================================================================
def test_reset_zeroes_prog_and_diag_keeps_aux(model):
    model.set_fields(b=ic(), u=np.full(N, 0.5))
    model.set_aux(q=np.full(N, 42.0), acc=np.full(N, 3.0))
    model.advance(2)
    model.reset()
    assert (np.asarray(model.state["u"].data) == 0.0).all()
    assert (np.asarray(model.state["b"].data) == 0.0).all()
    assert (np.asarray(model.state["acc"].data) == 0.0).all()
    # AUXILIARY is NEVER touched, consented or not
    assert (np.asarray(model.state["q"].data) == 42.0).all()
    assert np.allclose(np.asarray(model.state["bg"].data), 2e-5)


def test_reset_rewarms_and_restarts_the_clock(model):
    model.advance(4)
    model.reset()
    assert int(model.clock.it) == 0
    assert float(model.clock.elapsed) == 0.0
    assert int(model.carry.stepper_state.warmup) == 0


def test_reset_clears_panic(model):
    panic_the_model(model)
    model.reset()
    assert not model.panicked
    model.advance(1)


def test_reset_set_state_equals_fresh_assembly_bitwise():
    grid = make_grid()
    used = make_model(grid=grid)
    used.set_fields(b=ic(), u=np.full(N, 0.3))
    used.advance(5)
    used.reset()
    used.set_fields(b=ic(), u=np.full(N, 0.3))
    used.advance(5)
    fresh = make_model(grid=grid)
    fresh.set_fields(b=ic(), u=np.full(N, 0.3))
    fresh.advance(5)
    for name in ("u", "b"):
        assert np.array_equal(
            np.asarray(used.state[name].data),
            np.asarray(fresh.state[name].data))


# ================================================================
#  advance — argument contracts and typed returns
# ================================================================
def test_advance_returns_the_typed_result(model):
    result = model.advance(3)
    assert isinstance(result, AdvanceResult)
    assert result.steps_done == 3
    assert result.panicked is False
    assert result.panic_it is None
    assert result.wall_seconds >= 0.0


def test_advance_zero_steps_is_a_noop(model):
    result = model.advance(0)
    assert result.steps_done == 0
    assert int(model.clock.it) == 0


def test_advance_validates_steps(model):
    with pytest.raises(ValueError, match="non-negative"):
        model.advance(-1)


def test_advance_sync_false_is_reserved(model):
    with pytest.raises(NotImplementedError, match="CS-5"):
        model.advance(1, sync=False)


def test_advance_entry_raises_on_panicked_carry(model):
    panic_the_model(model)
    with pytest.raises(PanicError, match="clear the panic flag"):
        model.advance(1)


def test_tendency_and_variant_landed_in_wave_7(model):
    # wave 7 (2.8): the read-only composed tendency and derived models
    model.set_fields(u=ic(), b=ic())
    tendency = model.tendency(model.state)
    assert tendency.component_names == ("u", "b")
    variant = model.variant()
    assert (jax.tree_util.tree_structure(model.state)
            == jax.tree_util.tree_structure(variant.state))


# ================================================================
#  snapshot / load_snapshot (the io-store binding)
# ================================================================
def test_snapshot_manifest_header(model, tmp_path):
    model.set_fields(b=ic())
    model.advance(3)
    model.snapshot(tmp_path / "snap")
    manifest = read_manifest(tmp_path / "snap")
    assert manifest.iteration == 3
    assert manifest.time == pytest.approx(3 * DT)
    assert manifest.dt == pytest.approx(DT)
    assert manifest.fingerprint == model.fingerprint.digest
    assert len(manifest.leaves) > 0


def test_snapshot_load_restores_leaves_bitwise(model, tmp_path):
    model.set_fields(b=ic(), u=np.full(N, 0.25))
    model.set_aux(q=np.full(N, 5.0))
    model.advance(3)
    model.snapshot(tmp_path / "snap")
    reference = model.carry
    other = make_model(grid=model.grid)
    other.load_snapshot(tmp_path / "snap")
    restored = other.carry
    ref_leaves = jax.tree_util.tree_leaves(reference)
    new_leaves = jax.tree_util.tree_leaves(restored)
    assert len(ref_leaves) == len(new_leaves)
    for mine, theirs in zip(ref_leaves, new_leaves, strict=True):
        assert np.array_equal(np.asarray(mine),
                              np.asarray(theirs))
    assert int(other.clock.it) == 3


def test_load_snapshot_fingerprint_mismatch_diffs(model, tmp_path):
    model.snapshot(tmp_path / "snap")
    other = make_model(stepper=AdamBashforth(DT, order=3))
    with pytest.raises(SnapshotMismatchError,
                       match="structurally different"):
        other.load_snapshot(tmp_path / "snap")


def test_load_snapshot_dt_sign_errors(model, tmp_path):
    model.snapshot(tmp_path / "snap")
    backward = make_model(grid=model.grid,
                          stepper=AdamBashforth(-DT, order=2))
    with pytest.raises(SnapshotMismatchError, match="sign"):
        backward.load_snapshot(tmp_path / "snap")


def test_load_snapshot_dt_magnitude_warns(model, tmp_path):
    model.snapshot(tmp_path / "snap")
    slower = make_model(grid=model.grid,
                        stepper=AdamBashforth(2 * DT, order=2))
    with pytest.warns(UserWarning, match="magnitude"):
        slower.load_snapshot(tmp_path / "snap")


def test_load_snapshot_clears_panic(model, tmp_path):
    model.snapshot(tmp_path / "snap")
    other = make_model(grid=model.grid)
    panic_the_model(other)
    other.load_snapshot(tmp_path / "snap")
    assert not other.panicked
    other.advance(1)


# ================================================================
#  ModelState — the carry pytree contract
# ================================================================
def test_carry_is_a_copy(model):
    snap = model.carry
    model.advance(1)
    # the CS-12 twin survives the donation of the live carry
    assert int(snap.clock.it) == 0


def test_carry_treedef_is_stable_across_lifecycle_ops(model):
    def structure():
        return jax.tree_util.tree_structure(model._carry)

    before = structure()
    model.set_fields(b=ic())
    assert structure() == before
    model.update_parameters({"core.nu": 2e-3})
    assert structure() == before
    model.advance(2)
    assert structure() == before
    model.reset()
    assert structure() == before


def test_modelstate_is_frozen(model):
    carry = model.carry
    with pytest.raises(AttributeError, match="frozen"):
        carry.state = None
    with pytest.raises(AttributeError, match="frozen"):
        del carry.clock


def test_modelstate_replace_swaps_named_slots(model):
    carry = model.carry
    swapped = carry.replace(clock=carry.clock.reset())
    assert swapped.state is carry.state
    assert int(swapped.clock.it) == 0
    with pytest.raises(TypeError, match="unknown carry slots"):
        carry.replace(nope=1)
