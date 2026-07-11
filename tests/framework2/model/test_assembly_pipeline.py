"""Tests for the nine-step assembly pipeline (model/assembly.py).

Covers the wave-4 surface: the ``assemble()`` happy path (artifact
attributes, traced-halo negotiation, freeze, fingerprint stability),
step-order enforcement (dispatch merge before bind), the step-1
error paths (missing references, field collisions), the bind-time
time-dependent parameter gate, ``AssemblyRecord`` equality/hash and
the unhashable-static lint, and the frozen-grid verify path.
"""
from functools import partial
from typing import ClassVar

import jax.numpy as jnp
import pytest

from fridom.framework.utils import jaxify
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.errors import GridFrozenError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.model.assembly import (
    AssemblyArtifacts,
    Fingerprint,
    assemble,
)
from fridom.model.declarations import (
    FieldDeclaration,
    FieldReference,
    Lifecycle,
)
from fridom.model.errors import (
    AssemblyError,
    DispatchCollisionError,
    FieldCollisionError,
    MissingFieldError,
    TimeDependentParameterError,
)
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration
from fridom.model.report import AssemblyReport
from fridom.spatial.space_patterns import (
    Collocated,
    Profile,
    Staggered,
)
from fridom.model.stages import StageKind
from fridom.model.terms import Treatment, term
from fridom.model.time_dependent import Ramp


# ================================================================
#  Toy modules (API-sketch style, on the real fr.Module)
# ================================================================
class Stepper:

    """Duck-typed explicit stepper: dt leaf + one static (order)."""

    supported_treatments = frozenset({Treatment.EXPLICIT})

    def __init__(self, dt=1.0, order=2):
        self.dt = dt
        self.order = order


@partial(jaxify, dynamic=("nu",))
class Core(Module):

    """Two-field PROGNOSTIC core; one term uses the diff operator."""

    def __init__(self, nu=1e-3):
        self.nu = jnp.asarray(nu)

    field_declarations = (
        FieldDeclaration("u", space=Staggered("x"),
                         long_name="Velocity"),
        FieldDeclaration("b", space=Collocated(),
                         long_name="Buoyancy"),
    )
    parameter_declarations = (
        ParameterDeclaration("core.nu", attr="nu", units="1"),)

    @term(advances=("u",))
    def pressure_force(self, state, _ctx):
        return {"u": state["b"].diff("x")}

    @term(advances=("b",))
    def restoring(self, state, _ctx):
        return {"b": state["u"].to(state["b"].function_space)
                * (-1.0)}


@partial(jaxify, dynamic=("n2",))
class Background(Module):

    """AUX-owning module with a bare self_update stage."""

    def __init__(self, n2=1e-5):
        self.n2 = jnp.asarray(n2)

    field_declarations = (
        FieldDeclaration("bg", space=Profile("x"),
                         lifecycle=Lifecycle.AUXILIARY,
                         default=2.0, units="1"),
    )
    parameter_declarations = (
        ParameterDeclaration("background.n2", attr="n2",
                             units="1/s^2"),)

    def self_update(self, state, _ctx):
        return {"bg": state["bg"] * 0.5}


@partial(jaxify, dynamic=("value",))
class Provider(Module):

    """Pure parameter provider (scalar or Ramp leaf)."""

    def __init__(self, value=1.0):
        self.value = (value if isinstance(value, Ramp)
                      else jnp.asarray(value))

    parameter_declarations = (
        ParameterDeclaration("toy.value", attr="value", units="1"),)


class BareReader(Module):

    """bind() does a BARE read of toy.value (the gated spelling)."""

    def bind(self, table):
        self.seen = table.parameters["toy.value"]


class AtTimeReader(Module):

    """bind() reads toy.value via the sanctioned at_time(0.0)."""

    def bind(self, table):
        self.seen = table.parameters.at_time(0.0)["toy.value"]


class NeedsW(Module):

    """Unsatisfied reference (assembly step 1)."""

    field_references = (
        FieldReference("w", hint="declare a vertical velocity"),)


class AlsoB(Module):

    """Colliding declaration of the core's field 'b'."""

    field_declarations = (
        FieldDeclaration("b", space=Collocated()),)


# ================================================================
#  Fixtures
# ================================================================
def make_grid():
    return Grid((IntervalMesh(8, (0.0, 1.0), periodic=True,
                              name="x"),))


@pytest.fixture
def grid():
    return make_grid()


def make_artifacts(grid, modules=None, stepper=None, **kwargs):
    if modules is None:
        modules = (Core(), Background())
    if stepper is None:
        stepper = Stepper()
    return assemble(grid=grid, modules=modules,
                    time_stepper=stepper, **kwargs)


# ================================================================
#  The happy path (steps 1-7 + 9)
# ================================================================
def test_artifact_attributes_present_and_consistent(grid):
    arts = make_artifacts(grid, name="happy")
    assert isinstance(arts, AssemblyArtifacts)
    # the seam contract: exactly these attribute names (wave 4.2)
    for attr in ("field_table", "binding_table", "remat_table",
                 "composer", "schedule", "record", "fingerprint",
                 "report", "resharding"):
        assert getattr(arts, attr) is not None
    assert arts.field_table.names == ("u", "b", "bg")
    assert arts.field_table.prognostic == ("u", "b")
    assert arts.field_table.auxiliary == ("bg",)
    assert "core.nu" in arts.binding_table
    assert "background.n2" in arts.binding_table
    assert "stepper.dt" in arts.binding_table
    assert arts.schedule is arts.composer.schedule
    assert arts.record.name == "happy"
    assert isinstance(arts.report, AssemblyReport)
    assert isinstance(arts.fingerprint, Fingerprint)


def test_remat_table_retains_aux_defaults(grid):
    arts = make_artifacts(grid)
    entries = arts.remat_table.entries
    assert len(entries) == 1
    assert entries[0].field == "bg"
    assert entries[0].owner == 1
    assert entries[0].default == 2.0
    assert entries[0].space is arts.field_table["bg"].space


def test_negotiate_demanded_the_traced_halo(grid):
    make_artifacts(grid)
    # the diff term's FiniteDifference(order=2) demands halo 1
    assert grid.fingerprint is not None          # freeze happened
    assert grid.fingerprint.halo["x"] >= 1


def test_schedule_collects_terms_and_stages(grid):
    arts = make_artifacts(grid)
    terms = arts.schedule.kind_entries(None)
    assert [entry.key for entry in terms] == [
        "Core/pressure_force", "Core/restoring"]
    updates = arts.schedule.kind_entries(StageKind.SELF_UPDATE)
    assert [entry.key for entry in updates] == [
        "Background/self_update"]


def test_step_fn_is_memoized_and_shared(grid):
    a1 = make_artifacts(grid)
    a2 = make_artifacts(grid)
    assert callable(a1.record.step_fn())
    assert a1.record.step_fn() is a2.record.step_fn()


def test_empty_prognostic_composition_is_legal(grid):
    arts = make_artifacts(grid, modules=(Background(),))
    assert arts.field_table.prognostic == ()
    assert grid.fingerprint is not None


def test_field_free_composition_is_legal(grid):
    arts = make_artifacts(grid, modules=(Provider(),))
    assert len(arts.field_table) == 0
    assert grid.fingerprint is not None


# ================================================================
#  Fingerprint stability and diff
# ================================================================
def test_fingerprint_stable_across_identical_assemblies(grid):
    a1 = make_artifacts(grid)
    a2 = make_artifacts(grid)
    assert a1.fingerprint.digest == a2.fingerprint.digest
    assert a1.fingerprint.diff(a2.fingerprint) == (
        "fingerprints match")


def test_fingerprint_changes_on_stepper_static():
    a1 = make_artifacts(make_grid(), stepper=Stepper(order=2))
    a2 = make_artifacts(make_grid(), stepper=Stepper(order=3))
    assert a1.fingerprint.digest != a2.fingerprint.digest
    diff = a1.fingerprint.diff(a2.fingerprint)
    assert "stepper statics differ:" in diff
    assert "order" in diff


def test_fingerprint_sees_scalar_to_ramp_spec_change():
    ramp = Ramp(0.0, 1.0, period=10.0)
    a1 = make_artifacts(make_grid(),
                        modules=(Core(), Provider(1.0)))
    a2 = make_artifacts(make_grid(),
                        modules=(Core(), Provider(ramp)))
    assert a1.fingerprint.digest != a2.fingerprint.digest
    assert "parameter toy.value differ" in a1.fingerprint.diff(
        a2.fingerprint)


# ================================================================
#  AssemblyRecord: structural equality, hash, the static lint
# ================================================================
def test_record_equality_and_hash_across_identical_assemblies(grid):
    a1 = make_artifacts(grid)
    a2 = make_artifacts(grid)
    assert a1.record == a2.record
    assert hash(a1.record) == hash(a2.record)
    assert a1.record is not a2.record


def test_record_name_is_excluded_from_equality(grid):
    a1 = make_artifacts(grid, name="one")
    a2 = make_artifacts(grid, name="two")
    assert a1.record == a2.record
    assert hash(a1.record) == hash(a2.record)


def test_records_differ_across_stepper_statics():
    a1 = make_artifacts(make_grid(), stepper=Stepper(order=2))
    a2 = make_artifacts(make_grid(), stepper=Stepper(order=3))
    assert a1.record != a2.record


def test_unhashable_static_lint_names_the_offender(grid):
    stepper = Stepper()
    stepper.weights = [0.5, 0.5]         # an unhashable static
    with pytest.raises(AssemblyError, match="'weights'"):
        make_artifacts(grid, stepper=stepper)


# ================================================================
#  Step order: merge (3) runs before bind (4)
# ================================================================
class WideDiff(Module):

    """Dispatch override + a bind-time read of the merged registry."""

    override = FiniteDifference(order=4)
    dispatch: ClassVar = {("diff", Collocated()): override}

    def bind(self, table):
        self.seen_diff = table.grid.dispatch.resolve(
            "diff", table["b"].space)


def test_bind_sees_the_merged_registry(grid):
    wide = WideDiff()
    make_artifacts(grid, modules=(Core(), wide))
    assert wide.seen_diff is WideDiff.override


def test_pattern_key_override_widens_the_traced_halo(grid):
    make_artifacts(grid, modules=(Core(), WideDiff()))
    # FiniteDifference(order=4) demands halo 2 through the trace
    assert grid.fingerprint.halo["x"] >= 2


class AlsoWideDiff(Module):

    """Second module overriding the same resolved dispatch key."""

    dispatch: ClassVar = {
        ("diff", Collocated()): FiniteDifference(order=6)}


def test_same_resolved_key_from_two_modules_collides(grid):
    with pytest.raises(DispatchCollisionError) as err:
        make_artifacts(grid,
                       modules=(Core(), WideDiff(), AlsoWideDiff()))
    message = str(err.value)
    assert "WideDiff" in message
    assert "AlsoWideDiff" in message


# ================================================================
#  Step-1 error paths
# ================================================================
def test_reference_miss_raises_missing_field_error(grid):
    with pytest.raises(MissingFieldError) as err:
        make_artifacts(grid, modules=(Core(), NeedsW()))
    message = str(err.value)
    assert "'w'" in message
    assert "declare a vertical velocity" in message   # the hint
    assert "NeedsW" in message                        # attribution


def test_field_collision_names_both_modules(grid):
    with pytest.raises(FieldCollisionError) as err:
        make_artifacts(grid, modules=(Core(), AlsoB()))
    message = str(err.value)
    assert "Core" in message
    assert "AlsoB" in message


# ================================================================
#  The bind-time time-dependent parameter gate (step 4)
# ================================================================
def test_bare_ramp_read_in_bind_raises(grid):
    ramp = Ramp(0.0, 1.0, period=10.0)
    with pytest.raises(TimeDependentParameterError,
                       match="at_time"):
        make_artifacts(
            grid, modules=(Core(), Provider(ramp), BareReader()))


def test_at_time_read_in_bind_passes(grid):
    ramp = Ramp(0.0, 1.0, period=10.0)
    reader = AtTimeReader()
    make_artifacts(
        grid, modules=(Core(), Provider(ramp), reader))
    assert float(reader.seen) == pytest.approx(0.0)


def test_bare_scalar_read_in_bind_passes(grid):
    reader = BareReader()
    make_artifacts(
        grid, modules=(Core(), Provider(3.0), reader))
    assert float(reader.seen) == pytest.approx(3.0)


# ================================================================
#  The frozen-grid verify path (step 7)
# ================================================================
def test_second_model_with_equal_demands_passes_verify(grid):
    make_artifacts(grid)
    frozen = grid.fingerprint
    arts = make_artifacts(grid)          # identical demands
    assert grid.fingerprint.halo == frozen.halo
    assert arts.resharding.changed is False


def test_smaller_demands_pass_the_verify_path(grid):
    make_artifacts(grid)                 # freezes with halo >= 1
    arts = make_artifacts(grid, modules=(Background(),))
    assert arts.resharding.changed is False


class WideHalo(Module):

    """Halo-trace-exempt module with an oversized declared halo."""

    extra_halo = HaloSpec({"x": 64})


def test_larger_demands_raise_grid_frozen_error(grid):
    make_artifacts(grid)
    with pytest.raises(GridFrozenError,
                       match="most demanding model first"):
        make_artifacts(grid, modules=(Core(), WideHalo()))


def test_extra_halo_widens_the_negotiation(grid):
    make_artifacts(grid,
                   modules=(Core(), Background(), WideHalo()))
    assert grid.fingerprint.halo["x"] >= 64


# ================================================================
#  AssemblyRecord smoke: the composed body actually runs
# ================================================================
def test_step_fn_runs_over_a_zero_state(grid):
    arts = make_artifacts(grid)
    import jax.numpy as jnp  # noqa: PLC0415

    from fridom.spatial.fields.vector_field import (  # noqa: PLC0415
        VectorField,
    )
    from fridom.model.context import (  # noqa: PLC0415
        StepContext,
    )
    state = VectorField({
        record.name: grid.create_field(record.space,
                                       name=record.name)
        for record in arts.field_table})
    ctx = StepContext(params={}, clock=jnp.asarray(0.0),
                      dt=jnp.asarray(1.0),
                      stage_dt=jnp.asarray(1.0))
    body = arts.record.step_fn()
    new_state, sums = body(state, (Core(), Background()), ctx)
    assert set(new_state.component_names) == {"u", "b", "bg"}
    assert set(sums.explicit.component_names) == {"u", "b"}
