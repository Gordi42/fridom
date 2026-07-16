"""Tests for the resolved field table (model/field_table.py)."""
import dataclasses

import pytest

from fridom.model.declarations import (
    FieldDeclaration,
    FieldReference,
    Lifecycle,
)
from fridom.model.errors import (
    AssemblyError,
    FieldCollisionError,
    MissingFieldError,
)
from fridom.model.field_table import (
    FieldRecord,
    FieldTable,
    VelocitySelector,
)
from fridom.model.roles import TRACER, Role, Velocity
from fridom.spatial.bc import BC
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import (
    Collocated,
    Dof,
    Profile,
    Staggered,
)
from fridom.spatial.spaces.nodal import NodeSet


# ================================================================
#  Fixtures: a tiny (x, z) grid with locally seeded resolver rows
# ================================================================
def _seed_resolver(grid, mesh):
    """Register a ('declared_space', mesh) row (grid-level seam)."""
    def resolver(tag, bc, family="nodal"):  # noqa: ARG001
        node_set = {Dof.COLLOCATED: NodeSet.CENTER,
                    Dof.STAGGERED: NodeSet.RIGHT}[tag]
        if bc is None:
            bc = BC.NONE
        return mesh.nodal(node_set, bc=bc)
    grid.dispatch[("declared_space", mesh)] = resolver
    return resolver


@pytest.fixture(scope="module")
def meshes():
    x = IntervalMesh(8, (0.0, 1.0), periodic=True, name="x")
    z = IntervalMesh(8, (-1.0, 0.0), periodic=False, name="z")
    return x, z


@pytest.fixture(scope="module")
def grid(meshes):
    grid = Grid(meshes)
    for mesh in meshes:
        _seed_resolver(grid, mesh)
    return grid


def _declarations():
    """Build the reference composition (core/strat/coupler)."""
    core = (
        FieldDeclaration.velocity("u", "x", space=Staggered("x")),
        FieldDeclaration.velocity("v", "y", space=Staggered("y")),
        FieldDeclaration("w", space=Staggered("z"),
                         lifecycle=Lifecycle.DIAGNOSTIC,
                         roles=(Velocity("z"),)),
    )
    strat = (
        FieldDeclaration.tracer("b", units="m/s^2"),
        FieldDeclaration("n2", space=Profile("z"),
                         lifecycle=Lifecycle.AUXILIARY,
                         default=1e-5, units="1/s^2"),
    )
    coupler = (
        FieldDeclaration("flux", space=Collocated(),
                         lifecycle=Lifecycle.DIAGNOSTIC,
                         host_writable=True),
    )
    return (("Core", core), ("Stratification", strat),
            ("Coupler", coupler))


def _records(grid):
    records = []
    for owner, (owner_type, declarations) in \
            enumerate(_declarations()):
        records.extend(
            FieldRecord.from_declaration(
                declaration, owner=owner, owner_type=owner_type,
                grid=grid)
            for declaration in declarations)
    return tuple(records)


@pytest.fixture(scope="module")
def table(grid):
    return FieldTable(_records(grid))


@pytest.fixture(scope="module")
def state(grid, table):
    fields = {name: grid.create_field(table[name].space, name=name)
              for name in table.names}
    return VectorField(fields)


# ================================================================
#  Build + query surface
# ================================================================
def test_declaration_order_is_component_order(table):
    assert table.names == ("u", "v", "w", "b", "n2", "flux")
    assert [r.name for r in table] == list(table.names)
    assert len(table) == 6
    assert "u" in table
    assert "eta" not in table


def test_from_declaration_resolves_the_space(table, meshes):
    x, z = meshes
    assert table["u"].space.factors == (x.right, z.center)
    assert table["v"].space.factors == (x.center, z.center)
    assert table["n2"].space.factors == (x.constant, z.center)
    assert table["u"].space.layout is None  # bare, pre-layout


def test_record_carries_the_structural_row(table):
    record = table["b"]
    assert record.owner == 1
    assert record.owner_type == "Stratification"
    assert record.pattern == Collocated()
    assert record.lifecycle is Lifecycle.PROGNOSTIC
    assert TRACER in record.roles
    assert record.metadata.name == "b"
    assert record.metadata.units == "m/s^2"


def test_lifecycle_splits(table):
    assert table.prognostic == ("u", "v", "b")
    assert table.auxiliary == ("n2",)
    assert table.diagnostic == ("w", "flux")


def test_host_writable_listing(table):
    assert table.host_writable == ("flux",)


# ================================================================
#  Freeze + hashability
# ================================================================
def test_table_is_frozen(table):
    with pytest.raises(AttributeError, match="frozen"):
        table.names = ()
    with pytest.raises(AttributeError, match="frozen"):
        table._records = ()


def test_records_are_frozen(table):
    with pytest.raises(dataclasses.FrozenInstanceError):
        table["u"].name = "other"


def test_structural_equality_and_hash(grid):
    a = FieldTable(_records(grid))
    b = FieldTable(_records(grid))
    assert a == b
    assert hash(a) == hash(b)
    smaller = FieldTable(_records(grid)[:3])
    assert a != smaller


def test_fingerprint_token_is_stable(grid):
    a = FieldTable(_records(grid))
    b = FieldTable(_records(grid))
    assert a.fingerprint_token() == b.fingerprint_token()
    rows = a.fingerprint_token()
    assert rows[0][0] == "u"
    assert rows[0][3] == "PROGNOSTIC"


def test_rejects_non_records():
    with pytest.raises(TypeError, match="FieldRecord"):
        FieldTable(("u",))


# ================================================================
#  Collisions and missing fields
# ================================================================
def test_collision_names_both_modules(grid):
    records = _records(grid)
    twin = dataclasses.replace(records[3], owner=2,
                               owner_type="Rogue")
    with pytest.raises(FieldCollisionError) as err:
        FieldTable((*records, twin))
    message = str(err.value)
    assert "'b'" in message
    assert "Stratification" in message
    assert "Rogue" in message


def test_getitem_missing_lists_declared(table):
    with pytest.raises(MissingFieldError, match="eta"):
        table["eta"]
    with pytest.raises(MissingFieldError, match="u, v, w"):
        table["eta"]


def test_require_satisfied_returns_the_record(table):
    record = table.require(FieldReference("w"))
    assert record is table["w"]


def test_require_missing_carries_the_hint(table):
    reference = FieldReference(
        "eta", hint="declared by a free-surface module")
    with pytest.raises(MissingFieldError) as err:
        table.require(reference, module="Barotropic")
    message = str(err.value)
    assert "Barotropic" in message
    assert "free-surface module" in message
    assert "'eta'" in message


# ================================================================
#  Role selection (open sets; class-vs-instance)
# ================================================================
def test_select_by_instance(table):
    assert table.select(TRACER) == ("b",)
    assert table.select(Velocity("x")) == ("u",)


def test_select_by_family_class(table):
    assert table.select(Velocity) == ("u", "v", "w")


def test_select_zero_matches_is_a_noop(table):
    assert table.select(Role("mybgc.nutrient")) == ()


def test_select_rejects_non_roles(table):
    with pytest.raises(TypeError, match="Role"):
        table.select("fridom.tracer")
    with pytest.raises(TypeError, match="Role"):
        table.select(int)


# ================================================================
#  VelocitySelector
# ================================================================
def test_velocity_family_splits(table):
    selector = table.velocity()
    assert isinstance(selector, VelocitySelector)
    assert selector.components == ("u", "v", "w")
    assert selector.labels == (("u", "x"), ("v", "y"), ("w", "z"))
    # y is no coordinate of the (x, z) grid: v is transverse (V-N1)
    assert selector.directional == ("u", "w")
    assert selector.transverse == ("v",)
    # the diagnosed w has no momentum equation (V-H2)
    assert selector.prognostic == ("u", "v")


def test_label_of(table):
    selector = table.velocity()
    assert selector.label_of("v") == "y"
    with pytest.raises(ValueError, match="not in the velocity"):
        selector.label_of("b")


def test_duplicate_label_is_ambiguous(grid):
    records = (
        FieldRecord.from_declaration(
            FieldDeclaration.velocity("u", "x", space=Staggered("x")),
            owner=0, owner_type="CoreA", grid=grid),
        FieldRecord.from_declaration(
            FieldDeclaration.velocity("ub", "x", space=Staggered("x")),
            owner=1, owner_type="CoreB", grid=grid),
    )
    with pytest.raises(AssemblyError) as err:
        FieldTable(records).velocity()
    message = str(err.value)
    assert "CoreA" in message
    assert "CoreB" in message
    assert "'x'" in message


def test_two_labels_on_one_field_is_ambiguous(grid):
    record = FieldRecord.from_declaration(
        FieldDeclaration("q", space=Collocated(),
                         roles=(Velocity("x"), Velocity("y"))),
        owner=0, owner_type="Core", grid=grid)
    with pytest.raises(AssemblyError, match="multiple Velocity"):
        FieldTable((record,)).velocity()


def test_empty_family_is_legal(grid):
    record = FieldRecord.from_declaration(
        FieldDeclaration.tracer("dye"), owner=0,
        owner_type="Tracer", grid=grid)
    selector = FieldTable((record,)).velocity()
    assert selector.components == ()
    assert selector.prognostic == ()


# ================================================================
#  Empty-PROGNOSTIC legality (CS-13)
# ================================================================
def test_empty_prognostic_table_is_legal(grid):
    record = FieldRecord.from_declaration(
        FieldDeclaration("n2", space=Profile("z"),
                         lifecycle=Lifecycle.AUXILIARY),
        owner=0, owner_type="Stratification", grid=grid)
    table = FieldTable((record,))
    assert table.prognostic == ()
    assert table.auxiliary == ("n2",)


def test_empty_table_is_legal():
    table = FieldTable(())
    assert len(table) == 0
    assert table.names == ()
    assert table.prognostic == ()


# ================================================================
#  subset (the model-mediated prognostic read)
# ================================================================
def test_subset_prognostic(table, state):
    subset = table.subset(state, Lifecycle.PROGNOSTIC)
    assert subset.component_names == ("u", "v", "b")
    for name in subset.component_names:
        assert subset[name] is state[name]


def test_subset_auxiliary_and_diagnostic(table, state):
    assert table.subset(
        state, Lifecycle.AUXILIARY).component_names == ("n2",)
    assert table.subset(
        state, Lifecycle.DIAGNOSTIC).component_names == (
            "w", "flux")


def test_subset_preserves_the_state_class(table, state):
    class State(VectorField):
        pass

    vocab = State(dict(state.components))
    subset = table.subset(vocab, Lifecycle.PROGNOSTIC)
    assert type(subset) is State


def test_subset_of_empty_lifecycle_raises(grid, state):
    record = FieldRecord.from_declaration(
        FieldDeclaration("n2", space=Profile("z"),
                         lifecycle=Lifecycle.AUXILIARY),
        owner=0, owner_type="Stratification", grid=grid)
    table = FieldTable((record,))
    with pytest.raises(ValueError, match="CS-13"):
        table.subset(state, Lifecycle.PROGNOSTIC)


def test_subset_of_a_foreign_state_raises(table, grid):
    foreign = VectorField(
        {"u": grid.create_field(table["u"].space, name="u")})
    with pytest.raises(MissingFieldError, match="'v'"):
        table.subset(foreign, Lifecycle.PROGNOSTIC)
