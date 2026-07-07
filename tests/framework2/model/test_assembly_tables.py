"""Tests for the static assembly tables (model/assembly.py)."""
from functools import partial

import jax
import jax.numpy as jnp
import pytest

from fridom.framework.utils import jaxify
from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.spaces.nodal import NodeSet
from fridom.framework2.model import params
from fridom.framework2.model.assembly import (
    ParameterBinding,
    ParameterBindingTable,
    Params,
    RematerializationEntry,
    RematerializationTable,
)
from fridom.framework2.model.declarations import (
    FieldDeclaration,
    Lifecycle,
)
from fridom.framework2.model.errors import (
    AssemblyError,
    MissingParameterError,
    ParameterCollisionError,
)
from fridom.framework2.model.parameters import (
    USE_PROVIDED,
    Param,
    ParameterDeclaration,
    ParameterReference,
)
from fridom.framework2.model.space_patterns import (
    Collocated,
    Dof,
    Profile,
)
from fridom.framework2.model.time_dependent import Ramp


# ================================================================
#  Duck-typed cross-cluster fakes (no model/module.py import)
# ================================================================
class Stepper:
    def __init__(self, dt=60.0):
        self.dt = dt


class Core:
    parameter_declarations = (
        ParameterDeclaration("nonhydro.dsqr", attr="dsqr",
                             units="1"),)

    def __init__(self, dsqr=1.0):
        self.dsqr = dsqr


class Coriolis:
    parameter_declarations = (
        ParameterDeclaration(params.CORIOLIS_F0, attr="f0",
                             units="1/s"),)

    def __init__(self, f0=1e-4):
        self.f0 = f0


class Advection:

    """Consumer with a Param-defaulted constructor slot."""

    def __init__(self,
                 scaling=Param("scaling.rossby",  # noqa: B008
                               default=1.0)):
        self.scaling = scaling


class NeedsN2:
    parameter_references = (
        ParameterReference(params.STRATIFICATION_N2),)


# ================================================================
#  ParameterBindingTable build and query surface
# ================================================================
def test_build_binds_modules_and_stepper():
    table = ParameterBindingTable.build(
        (Core(), Coriolis()), Stepper(60.0))
    assert table.names == ("nonhydro.dsqr", "coriolis.f0",
                           "stepper.dt")
    assert table["nonhydro.dsqr"].slot == 0
    assert table["coriolis.f0"].slot == 1
    assert len(table) == 3


def test_the_stepper_is_just_another_provider_row():
    table = ParameterBindingTable.build((), Stepper(60.0))
    entry = table[params.TIME_STEP]
    assert entry.slot == "stepper"
    assert entry.attr == "dt"
    assert "stepper.dt" in table
    assert params.TIME_STEP in table


def test_iteration_yields_binding_rows():
    table = ParameterBindingTable.build((Core(),), Stepper())
    rows = list(table)
    assert all(isinstance(row, ParameterBinding) for row in rows)
    assert rows[0].name == "nonhydro.dsqr"


def test_getitem_missing_carries_registry_hint_and_provided():
    table = ParameterBindingTable.build((Core(),), Stepper())
    with pytest.raises(MissingParameterError) as err:
        table[params.STRATIFICATION_N2]
    message = str(err.value)
    assert "stratification module" in message      # registry hint
    assert "nonhydro.dsqr" in message              # provided list
    assert "stepper.dt" in message


def test_table_is_frozen_and_hashable():
    a = ParameterBindingTable.build((Core(),), Stepper())
    b = ParameterBindingTable.build((Core(),), Stepper())
    with pytest.raises(AttributeError, match="frozen"):
        a._entries = ()
    assert a == b
    assert hash(a) == hash(b)
    assert a.fingerprint_token() == b.fingerprint_token()


# ================================================================
#  One provider per name; the aliasing lint
# ================================================================
def test_two_module_providers_collide_naming_both():
    with pytest.raises(ParameterCollisionError) as err:
        ParameterBindingTable.build((Core(), Core(2.0)), Stepper())
    message = str(err.value)
    assert "nonhydro.dsqr" in message
    assert "modules[0] (Core)" in message
    assert "modules[1] (Core)" in message


def test_module_colliding_with_the_stepper_row():
    class RogueDt:
        parameter_declarations = (
            ParameterDeclaration("stepper.dt", attr="dt"),)

        def __init__(self):
            self.dt = 1.0

    with pytest.raises(ParameterCollisionError) as err:
        ParameterBindingTable.build((RogueDt(),), Stepper())
    assert "time stepper" in str(err.value)


def test_aliased_module_objects_are_rejected():
    core = Core()
    with pytest.raises(AssemblyError, match="same object"):
        ParameterBindingTable.build((core, core), Stepper())


# ================================================================
#  References: REQUIRED, identity defaults, no_default
# ================================================================
def test_required_miss_carries_hint_and_provided_list():
    with pytest.raises(MissingParameterError) as err:
        ParameterBindingTable.build((Core(), NeedsN2()), Stepper())
    message = str(err.value)
    assert "modules[1] (NeedsN2)" in message
    assert "stratification.n2" in message
    assert "stratification module" in message      # registry hint
    assert "nonhydro.dsqr" in message              # provided list


def test_reference_hint_wins_over_the_registry_hint():
    class Needy:
        parameter_references = (
            ParameterReference("bgc.mu", hint="add a BGC module"),)

    with pytest.raises(MissingParameterError,
                       match="add a BGC module"):
        ParameterBindingTable.build((Needy(),), Stepper())


def test_identity_default_binds_a_constant_entry():
    table = ParameterBindingTable.build((Advection(),), Stepper())
    entry = table["scaling.rossby"]
    assert entry.slot is None
    assert entry.value == 1.0
    values = table.eval_params((Advection(),), Stepper(), 0.0)
    assert values["scaling.rossby"] == 1.0


def test_matching_identity_defaults_share_one_entry():
    table = ParameterBindingTable.build(
        (Advection(), Advection()), Stepper())
    assert table.names.count("scaling.rossby") == 1


def test_conflicting_identity_defaults_raise():
    class OtherScaling:
        parameter_references = (
            ParameterReference("scaling.rossby", default=2.0),)

    with pytest.raises(AssemblyError, match="conflicting"):
        ParameterBindingTable.build(
            (Advection(), OtherScaling()), Stepper())


def test_default_on_a_no_default_registry_name_raises():
    class Sneaky:
        parameter_references = (
            ParameterReference(params.STRATIFICATION_N2,
                               default=0.0),)

    with pytest.raises(AssemblyError, match="no_default"):
        ParameterBindingTable.build((Sneaky(),), Stepper())


# ================================================================
#  Explicit-wins and USE_PROVIDED
# ================================================================
def test_untouched_param_slot_declares_the_reference():
    table = ParameterBindingTable.build((Advection(),), Stepper())
    assert "scaling.rossby" in table


def test_explicit_value_is_an_owned_value_no_linkage():
    table = ParameterBindingTable.build(
        (Advection(scaling=0.3),), Stepper())
    assert "scaling.rossby" not in table


def test_use_provided_forces_resolution_through_the_table():
    class DsqrProvider:
        parameter_declarations = (
            ParameterDeclaration("nonhydro.dsqr", attr="value"),)

        def __init__(self, value=0.04):
            self.value = value

    table = ParameterBindingTable.build(
        (Core(dsqr=USE_PROVIDED), DsqrProvider()), Stepper())
    entry = table["nonhydro.dsqr"]
    assert entry.slot == 1
    assert entry.attr == "value"


def test_use_provided_without_a_provider_is_required():
    with pytest.raises(MissingParameterError,
                       match=r"nonhydro\.dsqr"):
        ParameterBindingTable.build(
            (Core(dsqr=USE_PROVIDED),), Stepper())


def test_use_provided_on_an_undeclared_slot_raises():
    class Free:
        def __init__(self):
            self.knob = USE_PROVIDED

    with pytest.raises(AssemblyError, match="USE_PROVIDED"):
        ParameterBindingTable.build((Free(),), Stepper())


# ================================================================
#  The provided-must-be-dynamic lint (jaxified providers)
# ================================================================
@partial(jaxify, dynamic=("good",))
class JaxProvider:
    parameter_declarations = (
        ParameterDeclaration("demo.good", attr="good"),)

    def __init__(self):
        self.good = 1.0
        self.bad = 2.0


def test_dynamic_provided_leaf_passes_the_lint():
    table = ParameterBindingTable.build((JaxProvider(),), Stepper())
    assert "demo.good" in table


def test_static_provided_leaf_fails_the_lint():
    @partial(jaxify, dynamic=("good",))
    class StaticProvider:
        parameter_declarations = (
            ParameterDeclaration("demo.bad", attr="bad"),)

        def __init__(self):
            self.good = 1.0
            self.bad = 2.0

    with pytest.raises(AssemblyError, match="dynamic"):
        ParameterBindingTable.build((StaticProvider(),), Stepper())


# ================================================================
#  eval_params (stage-time reads) and Params
# ================================================================
def test_eval_params_resolves_a_ramp_at_stage_time():
    modules = (Core(dsqr=Ramp(0.0, 1.0, period=10.0)),)
    stepper = Stepper(60.0)
    table = ParameterBindingTable.build(modules, stepper)
    at_start = table.eval_params(modules, stepper, 0.0)
    at_end = table.eval_params(modules, stepper, 10.0)
    assert float(at_start["nonhydro.dsqr"]) == 0.0
    assert float(at_end["nonhydro.dsqr"]) == 1.0
    assert float(at_end["stepper.dt"]) == 60.0
    assert float(at_end[params.TIME_STEP]) == 60.0


def test_eval_params_reads_current_leaves():
    modules = (Core(dsqr=1.0),)
    stepper = Stepper()
    table = ParameterBindingTable.build(modules, stepper)
    swept = (Core(dsqr=7.0),)
    assert table.eval_params(swept, stepper, 0.0)[
        "nonhydro.dsqr"] == 7.0


def test_params_is_a_frozen_mapping():
    values = Params({"a.x": 1.0, "b.y": 2.0})
    assert len(values) == 2
    assert list(values) == ["a.x", "b.y"]
    assert "a.x" in values
    assert "c.z" not in values
    assert dict(values.items()) == {"a.x": 1.0, "b.y": 2.0}


def test_params_unknown_name_lists_the_bound_names():
    values = Params({"a.x": 1.0})
    with pytest.raises(MissingParameterError, match=r"a\.x"):
        values["typo.name"]


def test_params_is_a_pytree_with_dynamic_leaves():
    values = Params({"a.x": 1.0, "b.y": 2.0})
    leaves, treedef = jax.tree_util.tree_flatten(values)
    assert leaves == [1.0, 2.0]
    _, treedef2 = jax.tree_util.tree_flatten(
        Params({"a.x": 5.0, "b.y": 6.0}))
    assert treedef == treedef2   # sweeps never retrace


# ================================================================
#  host_view (the table-level read-only mapping)
# ================================================================
def test_host_view_returns_ramp_objects_raw():
    ramp = Ramp(0.0, 1.0, period=10.0)
    modules = (Core(dsqr=ramp),)
    stepper = Stepper()
    table = ParameterBindingTable.build(modules, stepper)
    view = table.host_view(modules, stepper)
    assert view["nonhydro.dsqr"] is ramp
    assert view["stepper.dt"] == 60.0
    assert len(view) == 2
    assert set(view) == {"nonhydro.dsqr", "stepper.dt"}


def test_host_view_missing_name_is_hinted():
    table = ParameterBindingTable.build((), Stepper())
    view = table.host_view((), Stepper())
    with pytest.raises(MissingParameterError,
                       match="stratification module"):
        view[params.STRATIFICATION_N2]


# ================================================================
#  RematerializationTable fixtures
# ================================================================
def _seed_resolver(grid, mesh):
    """Register a ('declared_space', mesh) row (grid-level seam)."""
    def resolver(tag, bc):
        node_set = {Dof.COLLOCATED: NodeSet.CENTER,
                    Dof.STAGGERED: NodeSet.RIGHT}[tag]
        if bc is None:
            bc = BC.NONE
        return mesh.nodal(node_set, bc=bc)
    grid.dispatch[("declared_space", mesh)] = resolver


@pytest.fixture(scope="module")
def grid():
    mesh = IntervalMesh(8, (0.0, 1.0), periodic=True, name="x")
    grid = Grid((mesh,))
    _seed_resolver(grid, mesh)
    return grid


class Strat:

    """Owner with an unbound-owner-method AUX default."""

    def __init__(self, n2=1e-5):
        self.n2 = n2

    def make_n2(self, grid, space):
        return grid.create_field(
            space, data=jnp.full(space.shape, self.n2), name="n2")


@pytest.fixture
def strat_table(grid):
    entry = RematerializationEntry(
        field="n2", owner=0, default=Strat.make_n2,
        space=Profile().resolve(grid))
    return RematerializationTable((entry,))


# ================================================================
#  RematerializationTable: the one shared path
# ================================================================
def test_allocation_pass_materializes_every_entry(strat_table, grid):
    fields = strat_table.materialize((Strat(),), grid)
    assert set(fields) == {"n2"}
    assert jnp.allclose(fields["n2"].data, 1e-5)


def test_one_path_identity(strat_table, grid):
    """Allocation and update_parameters share THE one code path."""
    modules = (Strat(3e-5),)
    allocated = strat_table.materialize(modules, grid)["n2"]
    updated = strat_table.materialize(
        modules, grid, owners={0})["n2"]
    assert allocated.function_space is updated.function_space
    assert jnp.array_equal(allocated.data, updated.data)  # bitwise


def test_defaults_read_the_owners_current_leaves(strat_table, grid):
    owner = Strat(1e-5)
    before = strat_table.materialize((owner,), grid,
                                     owners={0})["n2"]
    owner.n2 = 4e-5
    after = strat_table.materialize((owner,), grid,
                                    owners={0})["n2"]
    assert jnp.allclose(before.data, 1e-5)
    assert jnp.allclose(after.data, 4e-5)


def test_owner_filter_is_per_owner(grid):
    space = Profile().resolve(grid)
    table = RematerializationTable((
        RematerializationEntry("n2", 0, Strat.make_n2, space),
        RematerializationEntry("m2", 1, Strat.make_n2, space),
    ))
    modules = (Strat(1.0), Strat(2.0))
    only_first = table.materialize(modules, grid, owners={0})
    assert set(only_first) == {"n2"}
    both = table.materialize(modules, grid)
    assert set(both) == {"n2", "m2"}
    assert jnp.allclose(both["m2"].data, 2.0)


def test_host_writable_entries_are_update_exempt(grid):
    space = Profile().resolve(grid)
    table = RematerializationTable((
        RematerializationEntry("flux", 0, 0.5, space,
                               host_writable=True),
    ))
    # allocation (owners=None): the default is initialization-only
    allocated = table.materialize((Strat(),), grid)
    assert jnp.allclose(allocated["flux"].data, 0.5)
    # update_parameters: the host write is the source of truth
    assert table.materialize((Strat(),), grid, owners={0}) == {}


def test_default_forms(grid):
    collocated = Collocated().resolve(grid)
    table = RematerializationTable((
        RematerializationEntry("zero", 0, None, collocated),
        RematerializationEntry("const", 0, 2.5, collocated),
        RematerializationEntry("coord", 0, lambda x: 2.0 * x,
                               collocated),
    ))
    fields = table.materialize((Strat(),), grid)
    assert jnp.allclose(fields["zero"].data, 0.0)
    assert jnp.allclose(fields["const"].data, 2.5)
    expected = grid.create_field(collocated,
                                 init=lambda x: 2.0 * x)
    assert jnp.array_equal(fields["coord"].data, expected.data)


def test_bound_method_defaults_are_rejected(grid):
    owner = Strat()
    with pytest.raises(TypeError, match="bound"):
        RematerializationEntry("n2", 0, owner.make_n2,
                               Profile().resolve(grid))


def test_owner_method_must_return_a_field(grid):
    def broken(self, grid, space):  # noqa: ARG001
        return 3.0

    table = RematerializationTable((
        RematerializationEntry("n2", 0, broken,
                               Profile().resolve(grid)),))
    with pytest.raises(TypeError, match="ScalarField"):
        table.materialize((Strat(),), grid)


def test_from_declaration_retains_the_aux_default(grid):
    declaration = FieldDeclaration(
        "n2", space=Profile(), lifecycle=Lifecycle.AUXILIARY,
        default=Strat.make_n2, host_writable=False)
    entry = RematerializationEntry.from_declaration(
        declaration, owner=3, space=Profile().resolve(grid))
    assert entry.field == "n2"
    assert entry.owner == 3
    assert entry.default is Strat.make_n2
    assert entry.host_writable is False


def test_from_declaration_rejects_non_aux(grid):
    declaration = FieldDeclaration.tracer("b")
    with pytest.raises(ValueError, match="AUXILIARY"):
        RematerializationEntry.from_declaration(
            declaration, owner=0, space=Collocated().resolve(grid))


def test_table_rejects_duplicates_and_foreign_rows(grid):
    space = Profile().resolve(grid)
    entry = RematerializationEntry("n2", 0, None, space)
    with pytest.raises(AssemblyError, match="twice"):
        RematerializationTable((entry, entry))
    with pytest.raises(TypeError, match="Rematerialization"):
        RematerializationTable(("n2",))


def test_fingerprint_token_is_structural(grid):
    space = Profile().resolve(grid)
    table = RematerializationTable((
        RematerializationEntry("n2", 0, Strat.make_n2, space),))
    (row,) = table.fingerprint_token()
    assert row[0] == "n2"
    assert row[2] == "owner_method"
