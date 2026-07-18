"""Tests for BoundaryTrace / BoundaryEmbed (operators/boundary.py)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh
from fridom.spatial.operators.base import OperatorRequirements
from fridom.spatial.operators.boundary import (
    BoundaryEmbed,
    BoundaryTrace,
)
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.nodal import NodeSet
from fridom.spatial.spaces.trace import Side, TraceSpace

N = 6


@pytest.fixture
def mx():
    return IntervalMesh(N, (0.0, 1.0), name="x")  # periodic


@pytest.fixture
def mz():
    return IntervalMesh(4, (0.0, 3.0), periodic=False, name="z")


@pytest.fixture
def grid(mx, mz):
    my = IntervalMesh(N, (0.0, 2.0), name="y")
    return Grid((mx, my, mz))


def _cell_field(grid, init):
    mx, my, mz = grid.factors
    return grid.create_field(mx.center * my.center * mz.center, init=init)


# ================================================================
#  Identity, interning, kind, requirements
# ================================================================
def test_trace_interns_on_side_and_depth():
    assert BoundaryTrace(Side.HIGH) is BoundaryTrace(Side.HIGH)
    assert BoundaryTrace(Side.LOW) is not BoundaryTrace(Side.HIGH)
    assert BoundaryTrace(Side.HIGH).dispatch_kind == "trace"
    assert BoundaryTrace(Side.HIGH).side is Side.HIGH
    assert BoundaryTrace(Side.HIGH).depth == 0


def test_trace_bound_variant_is_interned():
    op = BoundaryTrace(Side.HIGH)
    assert op["z"] is op["z"]
    assert op["z"].unbound is op


def test_trace_rejects_non_side():
    with pytest.raises(TypeError, match="side must be a Side"):
        BoundaryTrace("high")


def test_trace_requirements_are_axis_local(mz):
    req = BoundaryTrace(Side.HIGH).requirements(mz.center)
    assert isinstance(req, OperatorRequirements)
    assert req.halo == 0
    assert req.layout == "local"


def test_embed_is_interned_and_carries_the_embed_kind():
    assert BoundaryEmbed() is BoundaryEmbed()
    assert BoundaryEmbed().dispatch_kind == "embed"


# ================================================================
#  BoundaryTrace codomain + loud rejections
# ================================================================
def test_trace_codomain_nodal(mz):
    op = BoundaryTrace(Side.HIGH)
    t = op.codomain(mz.center)
    assert isinstance(t, TraceSpace)
    assert t.parent_node_set is NodeSet.CENTER
    assert t.side is Side.HIGH
    assert t is mz.trace(NodeSet.CENTER, Side.HIGH)


def test_trace_codomain_cellavg_colocates_center(mz):
    # the FV row co-locates with cell centres
    t = BoundaryTrace(Side.LOW).codomain(mz.cell_avg)
    assert t is mz.trace(NodeSet.CENTER, Side.LOW)


def test_trace_codomain_preserves_complex(mz):
    t = BoundaryTrace(Side.HIGH).codomain(mz.center.as_complex())
    assert t.scalars is Scalars.COMPLEX
    assert t is mz.trace(NodeSet.CENTER, Side.HIGH).as_complex()


def test_trace_rejects_periodic(mx):
    with pytest.raises(SpaceMismatchError, match="periodic"):
        BoundaryTrace(Side.HIGH).codomain(mx.center)


def test_trace_rejects_coefficient(mx):
    coeff = mx.fourier(origin=mx.center)
    with pytest.raises(SpaceMismatchError, match="coefficient"):
        BoundaryTrace(Side.HIGH).codomain(coeff)


def test_trace_rejects_constant_and_trace(mz):
    with pytest.raises(SpaceMismatchError, match="collapsed"):
        BoundaryTrace(Side.HIGH).codomain(mz.constant)
    with pytest.raises(SpaceMismatchError, match="collapsed"):
        BoundaryTrace(Side.HIGH).codomain(
            mz.trace(NodeSet.CENTER, Side.HIGH))


def test_trace_rejects_faceavg(mz):
    # bounded dual cells never reach the wall
    with pytest.raises(SpaceMismatchError, match="boundary-adjacent"):
        BoundaryTrace(Side.HIGH).codomain(mz.face_avg)


def test_trace_rejects_nonzero_depth(mz):
    with pytest.raises(SpaceMismatchError, match="interior fixed-depth"):
        BoundaryTrace(Side.HIGH, depth=1).codomain(mz.center)


def test_trace_side_aware_dirichlet_guard(mz):
    # Outer is a member of BOTH walls: Dirichlet eliminates the
    # requested wall's DOF -> reject at that wall, either side
    outer_d = mz.nodal(NodeSet.OUTER, bc=BC.DIRICHLET)
    with pytest.raises(SpaceMismatchError, match="HIGH wall"):
        BoundaryTrace(Side.HIGH).codomain(outer_d)
    with pytest.raises(SpaceMismatchError, match="LOW wall"):
        BoundaryTrace(Side.LOW).codomain(outer_d)


def test_trace_dirichlet_one_wall_still_traces_the_other(mz):
    # strictly finer than Restriction: Left is a LOW member only, so a
    # LOW-Dirichlet Left is still traceable at HIGH
    left_d = mz.nodal(NodeSet.LEFT, bc=(BC.DIRICHLET, BC.NONE))
    assert BoundaryTrace(Side.HIGH).codomain(left_d) is mz.trace(
        NodeSet.LEFT, Side.HIGH)
    with pytest.raises(SpaceMismatchError, match="LOW wall"):
        BoundaryTrace(Side.LOW).codomain(left_d)


# ================================================================
#  BoundaryTrace exactness (uniform + stretched)
# ================================================================
def test_trace_selects_the_boundary_row_uniform(grid):
    f = _cell_field(grid, lambda x, y, z: x + y + z**2)
    high = f.trace("z", Side.HIGH)
    low = f.trace("z", Side.LOW)
    data = np.asarray(f.data)
    assert high.function_space.bare.factor("z") is grid.factors[2].trace(
        NodeSet.CENTER, Side.HIGH)
    assert np.allclose(np.asarray(high.data), data[:, :, -1:])
    assert np.allclose(np.asarray(low.data), data[:, :, :1])


def test_trace_is_exact_on_a_stretched_mesh(mx):
    mz = MappedIntervalMesh(
        4, (0.0, 1.0), lambda s: s + 0.1 * np.sin(2 * np.pi * s),
        periodic=False, name="z")
    grid = Grid((mx, mz))
    f = grid.create_field(mx.center * mz.center,
                          init=lambda x, z: 2.0 * z + x)
    high = f.trace("z", Side.HIGH)
    # a pure index selection carries no metric: exact on the mapped mesh
    assert np.allclose(np.asarray(high.data), np.asarray(f.data)[:, -1:])


def test_trace_outer_high_is_the_surface_face(grid):
    # w on Outer(z): HIGH picks the surface face w(0) (the n+1-th face)
    mx, my, mz = grid.factors
    w = grid.create_field(mx.center * my.center * mz.outer,
                          init=lambda x, y, z: x + y + z)
    surf = w.trace("z", Side.HIGH)
    assert np.allclose(np.asarray(surf.data),
                       np.asarray(w.data)[:, :, -1:])


# ================================================================
#  BoundaryEmbed codomain, exactness, round-trips
# ================================================================
def test_embed_codomain_is_the_bc_free_parent(mz):
    t = mz.trace(NodeSet.OUTER, Side.HIGH)
    assert BoundaryEmbed().codomain(t) is mz.outer


def test_embed_codomain_preserves_complex(mz):
    t = mz.trace(NodeSet.CENTER, Side.LOW).as_complex()
    assert BoundaryEmbed().codomain(t) is mz.center.as_complex()


def test_embed_rejects_non_trace(mz):
    with pytest.raises(SpaceMismatchError, match="embed"):
        BoundaryEmbed().codomain(mz.center)


def test_embed_materializes_the_boundary_row_sparsely(grid):
    f = _cell_field(grid, lambda x, y, z: x + y + z)
    high = f.trace("z", Side.HIGH)
    sparse = high.embed("z")
    data = np.asarray(sparse.data)
    top = np.asarray(f.data)[:, :, -1:]
    assert data.shape[2] == 4
    assert np.allclose(data[:, :, -1:], top)
    assert np.allclose(data[:, :, :-1], 0.0)


def test_trace_embed_round_trips_exactly(grid):
    f = _cell_field(grid, lambda x, y, z: np.cos(1.0) + x * z + y)
    for side in (Side.LOW, Side.HIGH):
        t = f.trace("z", side)
        assert np.allclose(np.asarray(t.embed("z").trace("z", side).data),
                           np.asarray(t.data))


# ================================================================
#  Verb routing + seeding via grid.dispatch
# ================================================================
def test_trace_verb_lands_on_the_right_space(grid):
    f = _cell_field(grid, lambda x, y, z: x + y + z)
    t = f.trace("z", Side.HIGH)
    assert isinstance(t.function_space.bare.factor("z"), TraceSpace)
    assert t.function_space.bare.factor("z").side is Side.HIGH


def test_seeding_resolves_trace_and_embed(grid):
    mz = grid.factors[2]
    assert isinstance(grid.dispatch.resolve("trace", mz.center),
                      BoundaryTrace)
    assert isinstance(
        grid.dispatch.resolve(
            "embed", mz.trace(NodeSet.CENTER, Side.HIGH)),
        BoundaryEmbed)
    # the FV CellAvg row is seeded too (co-located CENTER trace)
    assert isinstance(grid.dispatch.resolve("trace", mz.cell_avg),
                      BoundaryTrace)


# ================================================================
#  Accessor pattern: trace of measure / metric / immersed (§3 end)
# ================================================================
def test_measure_trace_is_the_top_cell_thickness_uniform(grid):
    mx, my, mz = grid.factors
    space = mx.center * my.center * mz.center
    top = grid.measure(space, "z").trace("z", Side.HIGH)
    full = np.asarray(grid.measure(space, "z").data)
    # uniform mesh: every cell is dz = 3.0 / 4
    assert np.allclose(np.asarray(top.data)[..., 0], full[..., -1])
    assert np.allclose(np.asarray(top.data), 3.0 / 4)


def test_measure_trace_is_the_top_cell_thickness_stretched(mx):
    mz = MappedIntervalMesh(
        5, (0.0, 1.0), lambda s: s**1.5, periodic=False, name="z")
    grid = Grid((mx, mz))
    space = mx.center * mz.center
    top = grid.measure(space, "z").trace("z", Side.HIGH)
    full = np.asarray(grid.measure(space, "z").data)
    assert np.allclose(np.asarray(top.data)[..., 0], full[..., -1])


def test_metric_trace_flows_traced_values_in_jit(mx):
    ms = IntervalMesh(4, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H},
        params={"H": lambda x: 1.0 + 0.2 * jnp.sin(x)})
    grid = Grid((mx, ms), mapping=mapping)
    space = mx.center * ms.center

    def loss(hval):
        h = grid.create_field(mx.center, data=jnp.full((N,), hval))
        metric = grid.metric(space, "dz_dsigma", params={"H": h})
        return jnp.sum(metric.trace("sigma", Side.HIGH).data)

    # a traced metric slice flows values (no caching); grad is finite
    assert np.isfinite(float(jax.grad(loss)(1.3)))


def test_metric_trace_matches_the_full_slice(mx):
    ms = IntervalMesh(4, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H},
        params={"H": lambda x: 1.0 + 0.2 * jnp.sin(x)})
    grid = Grid((mx, ms), mapping=mapping)
    space = mx.center * ms.center
    metric = grid.metric(space, "dz_dsigma")
    top = metric.trace("sigma", Side.HIGH)
    assert np.allclose(np.asarray(top.data)[..., 0],
                       np.asarray(metric.data)[..., -1])


def test_metric_on_a_trace_factor_raises(mx):
    # the twin of the measure guard: a metric has no per-factor
    # representation on a trace; trace the full metric field instead
    ms = IntervalMesh(4, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H},
        params={"H": lambda x: 1.0 + 0.2 * jnp.sin(x)})
    grid = Grid((mx, ms), mapping=mapping)
    query = mx.center * ms.trace(NodeSet.CENTER, Side.HIGH)
    with pytest.raises(ValueError, match="boundary trace along"):
        grid.metric(query, "dz_dsigma")


def test_immersed_fraction_trace_on_a_walled_grid(mx):
    mz = IntervalMesh(5, (0.0, 1.0), periodic=False, name="z")
    grid = Grid((mx, mz),
                immersed=ImmersedDomain(lambda x, z: 1.0 - 0.5 * z - 0.05 * x))
    frac = grid.immersed.fraction(mx.center * mz.center)
    top = frac.trace("z", Side.HIGH)
    assert np.allclose(np.asarray(top.data)[..., 0],
                       np.asarray(frac.data)[..., -1])
