"""Tests for the grid-free space descriptors (model/space_patterns.py)."""
import pytest

from fridom.spatial.bc import BC, BCStructure
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.registry import (
    DispatchError,
    OperatorRegistry,
)
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodeSet
from fridom.spatial.spaces.tensor_product import (
    TensorProductSpace,
)
from fridom.spatial.space_patterns import (
    Collocated,
    Dof,
    Profile,
    SpacePattern,
    SpaceRule,
    Staggered,
)


# ================================================================
#  Fixtures: a tiny (x, z) grid with locally seeded resolver rows
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


@pytest.fixture(scope="module")
def bare_grid():
    """Build a grid WITHOUT resolver rows.

    The default registry seeds per-mesh resolver rows since wave 4,
    so an empty registry is passed explicitly.
    """
    return Grid((IntervalMesh(8, (0.0, 1.0), periodic=True,
                              name="x"),),
                dispatch=OperatorRegistry({}))


# ================================================================
#  Dof
# ================================================================
def test_dof_members():
    assert list(Dof) == [Dof.COLLOCATED, Dof.STAGGERED, Dof.CONSTANT]


# ================================================================
#  SpacePattern value semantics
# ================================================================
def test_value_equality_and_hash():
    assert Collocated() == Collocated()
    assert hash(Collocated()) == hash(Collocated())
    assert Collocated() == SpacePattern()
    assert {Collocated(): "row"}[SpacePattern()] == "row"


def test_tag_order_is_canonicalized():
    assert Staggered("x", "z") == Staggered("z", "x")
    assert hash(Staggered("x", "z")) == hash(Staggered("z", "x"))


def test_distinct_patterns_differ():
    assert Collocated() != Staggered("x")
    assert Collocated() != Collocated(bc={"z": BC.DIRICHLET})
    assert Profile("z") != Profile()
    assert Collocated() != Collocated(scalars=Scalars.COMPLEX)


def test_create_matches_direct_construction():
    direct = SpacePattern(
        tags=(("z", Dof.STAGGERED),), bc=(("z", BC.DIRICHLET),),
        require=("z",))
    created = SpacePattern.create(
        tags={"z": Dof.STAGGERED}, bc={"z": BC.DIRICHLET},
        require=["z"])
    assert direct == created
    assert hash(direct) == hash(created)


def test_bc_tuple_normalizes_to_structure():
    pattern = Collocated(bc={"z": (BC.DIRICHLET, BC.NONE)})
    assert pattern == Collocated(
        bc={"z": BCStructure((BC.DIRICHLET, BC.NONE))})


def test_construction_validation():
    with pytest.raises(TypeError, match="Dof member"):
        SpacePattern(default="collocated")
    with pytest.raises(TypeError, match="tags"):
        SpacePattern(tags=(("z", "staggered"),))
    with pytest.raises(ValueError, match="duplicate"):
        SpacePattern(tags=(("z", Dof.STAGGERED),
                           ("z", Dof.COLLOCATED)))
    with pytest.raises(TypeError, match="bc"):
        SpacePattern(bc=(("z", "dirichlet"),))
    with pytest.raises(TypeError, match="require"):
        SpacePattern(require=(1,))
    with pytest.raises(TypeError, match="Scalars"):
        SpacePattern(scalars="complex")


def test_repr_round_trips():
    namespace = {"SpacePattern": SpacePattern, "Dof": Dof,
                 "BC": BC, "Scalars": Scalars}
    for pattern in (Collocated(), Staggered("z"),
                    Profile("z", bc={"z": BC.NEUMANN}),
                    SpacePattern(wall_bc=(("z", BC.DIRICHLET),)),
                    Collocated(require=("x",),
                               scalars=Scalars.COMPLEX)):
        assert eval(repr(pattern), namespace) == pattern  # noqa: S307


# ================================================================
#  wall_bc: topology-conditional BCs (C8, topology-driven walls)
# ================================================================
def test_wall_bc_value_semantics():
    direct = SpacePattern(wall_bc=(("z", BC.DIRICHLET),))
    created = SpacePattern.create(wall_bc={"z": BC.DIRICHLET})
    assert direct == created
    assert hash(direct) == hash(created)
    assert direct != SpacePattern()
    assert direct != SpacePattern(bc=(("z", BC.DIRICHLET),))


def test_wall_bc_construction_validation():
    with pytest.raises(TypeError, match="wall_bc"):
        SpacePattern(wall_bc=(("z", "dirichlet"),))
    with pytest.raises(ValueError, match="duplicate"):
        SpacePattern(wall_bc=(("z", BC.DIRICHLET),
                              ("z", BC.NEUMANN)))
    with pytest.raises(ValueError, match="both bc and wall_bc"):
        SpacePattern(bc=(("z", BC.NEUMANN),),
                     wall_bc=(("z", BC.DIRICHLET),))


# ================================================================
#  Factories
# ================================================================
def test_collocated_factory():
    pattern = Collocated()
    assert pattern.default is Dof.COLLOCATED
    assert pattern.tags == ()


def test_staggered_factory():
    pattern = Staggered("x", "z")
    assert pattern.default is Dof.COLLOCATED
    assert dict(pattern.tags) == {"x": Dof.STAGGERED,
                                  "z": Dof.STAGGERED}
    with pytest.raises(ValueError, match="at least one"):
        Staggered()


def test_profile_factory():
    pattern = Profile("z")
    assert pattern.default is Dof.CONSTANT
    assert dict(pattern.tags) == {"z": Dof.COLLOCATED}
    assert Profile().default is Dof.CONSTANT
    assert Profile().tags == ()


# ================================================================
#  Resolution against the resolver rows
# ================================================================
def test_collocated_resolves_all_center(grid, meshes):
    x, z = meshes
    space = Collocated().resolve(grid)
    assert space.factors == (x.center, z.center)
    assert space.layout is None  # bare, pre-layout


def test_staggered_resolves_per_name(grid, meshes):
    x, z = meshes
    space = Staggered("z").resolve(grid)
    assert space.factors == (x.center, z.right)


def test_unmatched_name_degrades_gracefully(grid, meshes):
    # Staggered("y") on an (x, z) grid: fully collocated
    x, z = meshes
    assert Staggered("y").resolve(grid).factors == (
        x.center, z.center)
    assert Profile("y").resolve(grid).factors == (
        x.constant, z.constant)


def test_profile_resolves_constant_factors(grid, meshes):
    x, z = meshes
    space = Profile("z").resolve(grid)
    assert space.factors == (x.constant, z.center)
    one_dof = Profile().resolve(grid)
    assert all(isinstance(f, ConstantSpace)
               for f in one_dof.factors)
    assert one_dof.shape == (1, 1)


def test_bc_enters_the_resolved_space(grid, meshes):
    _, z = meshes
    space = Collocated(bc={"z": BC.DIRICHLET}).resolve(grid)
    assert space.factor("z") is z.nodal(NodeSet.CENTER,
                                        bc=BC.DIRICHLET)
    assert space.factor("x") is meshes[0].center


def test_bc_on_constant_axis_raises(grid):
    with pytest.raises(ValueError, match="constant factor"):
        Profile(bc={"z": BC.DIRICHLET}).resolve(grid)


def test_wall_bc_applies_on_the_bounded_factor_only(grid, meshes):
    # x is periodic, z is bounded: only the z entry bites
    x, z = meshes
    pattern = SpacePattern.create(
        wall_bc={"x": BC.DIRICHLET, "z": BC.DIRICHLET})
    space = pattern.resolve(grid)
    assert space.factor("z") is z.nodal(NodeSet.CENTER,
                                        bc=BC.DIRICHLET)
    assert space.factor("x") is x.center  # BC-free, tag ignored


def test_wall_bc_matches_the_unconditional_bc_on_bounded(grid):
    # on the bounded axis, wall_bc resolves exactly like bc
    walled = SpacePattern.create(wall_bc={"z": BC.DIRICHLET})
    pinned = Collocated(bc={"z": BC.DIRICHLET})
    assert walled.resolve(grid) is pinned.resolve(grid)


def test_wall_bc_periodic_regression_identical_interned_space():
    # a fully periodic grid resolves to the IDENTICAL interned
    # spaces with and without wall_bc entries (C8 regression)
    periodic = Grid((
        IntervalMesh(8, (0.0, 1.0), periodic=True, name="x"),
        IntervalMesh(8, (0.0, 1.0), periodic=True, name="z"),
    ))
    walled = SpacePattern.create(
        tags={"z": Dof.STAGGERED},
        wall_bc={"z": BC.DIRICHLET})
    plain = Staggered("z")
    assert walled.resolve(periodic) is plain.resolve(periodic)


def test_wall_bc_on_constant_axis(grid):
    # bounded constant factor: raises like an unconditional bc;
    # periodic constant factor: the conditional entry is inert
    with pytest.raises(ValueError, match="constant factor"):
        Profile(wall_bc={"z": BC.DIRICHLET}).resolve(grid)
    space = Profile(wall_bc={"x": BC.DIRICHLET}).resolve(grid)
    assert space is Profile().resolve(grid)


def test_resolution_is_interned_pure(grid):
    pattern = Staggered("z", bc={"z": BC.NONE})
    assert pattern.resolve(grid) is pattern.resolve(grid)


def test_require_matched_passes(grid, meshes):
    x, z = meshes
    space = Collocated(require=("x", "z")).resolve(grid)
    assert space.factors == (x.center, z.center)


def test_require_unmatched_raises(grid):
    with pytest.raises(ValueError, match="require"):
        Staggered("y", require=("y",)).resolve(grid)


def test_scalars_request(grid, meshes):
    x, z = meshes
    space = Collocated(scalars=Scalars.COMPLEX).resolve(grid)
    assert space.scalars is Scalars.COMPLEX
    assert space.factors == (x.center.as_complex(),
                             z.center.as_complex())
    real = Collocated(scalars=Scalars.REAL).resolve(grid)
    assert real.scalars is Scalars.REAL


def test_missing_resolver_row_is_hinted(bare_grid):
    with pytest.raises(DispatchError, match="declared_space") as err:
        Collocated().resolve(bare_grid)
    message = str(err.value)
    assert "grid-level" in message
    assert "resolver" in message


def test_profile_needs_no_resolver_rows(bare_grid):
    # CONSTANT resolves through the universal mesh.constant
    space = Profile().resolve(bare_grid)
    assert space is bare_grid.factors[0].constant


# ================================================================
#  SpaceRule
# ================================================================
def test_space_rule_resolves(grid, meshes):
    x, z = meshes
    rule = SpaceRule(lambda g: g.factors[0].center
                     * g.factors[1].right)
    assert rule.resolve(grid).factors == (x.center, z.right)


def test_space_rule_identity_semantics():
    def fn(grid):
        return grid.factors[0].center

    # identity, not value: two rules over one fn are distinct keys
    assert SpaceRule(fn) != SpaceRule(fn)
    assert len({SpaceRule(fn), SpaceRule(fn)}) == 2
    rule = same = SpaceRule(fn)
    assert rule == same
    assert "fn" in repr(rule)


def test_space_rule_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        SpaceRule("not a rule")


def test_space_rule_double_resolve_purity_check(grid):
    def impure(g):
        # bypasses interning: a fresh product object per call
        return TensorProductSpace(
            tuple(mesh.center for mesh in g.factors))

    with pytest.raises(ValueError, match="impure"):
        SpaceRule(impure).resolve(grid)
