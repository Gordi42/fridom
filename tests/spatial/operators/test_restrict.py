"""Tests for fridom.spatial.operators.restrict."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh
from fridom.spatial.operators.base import OperatorRequirements
from fridom.spatial.operators.restrict import Restriction
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.nodal import NodeSet
from fridom.spatial.spaces.tensor_product import TensorProductSpace

N = 6


@pytest.fixture
def mx():
    return IntervalMesh(N, (0.0, 1.0), name="x")  # periodic


@pytest.fixture
def mz():
    return IntervalMesh(N, (0.0, 3.0), periodic=False, name="z")


@pytest.fixture
def walled(mz):
    return Grid((mz,))


@pytest.fixture
def restrict():
    return Restriction()


# ================================================================
#  Identity, interning, requirements
# ================================================================
def test_is_interned_and_carries_the_restrict_kind():
    assert Restriction() is Restriction()
    assert Restriction().dispatch_kind == "restrict"


def test_requirements_declare_the_one_slot_above_footprint(restrict, mz):
    # Inner[m] == Outer[m + 1] reads one slot ABOVE each output, so the
    # per-shard footprint is the asymmetric reach (0, 1) (halo 1) -- the
    # ghost the exchange must fill for the last interior face of a shard
    # to reach the neighbour's boundary face across the seam. Declaring
    # halo 0 silently elides that sync on a sharded axis (the seam bug).
    req = restrict.requirements(mz.outer)
    assert isinstance(req, OperatorRequirements)
    assert req.reach == (0, 1)
    assert req.halo == 1
    assert req.layout == "any"


# ================================================================
#  Signature: Outer -> Inner only
# ================================================================
def test_codomain_maps_outer_to_inner(restrict, mz):
    assert restrict.codomain(mz.outer) is mz.inner


def test_codomain_preserves_complex(restrict, mz):
    out = restrict.codomain(mz.outer.as_complex())
    assert out is mz.inner.as_complex()
    assert out.scalars is Scalars.COMPLEX


@pytest.mark.parametrize("factory", ["center", "left", "right", "inner"])
def test_codomain_rejects_non_outer_node_sets(restrict, mz, factory):
    with pytest.raises(SpaceMismatchError, match="Outer"):
        restrict.codomain(getattr(mz, factory))


def test_codomain_rejects_a_periodic_factor(restrict, mx):
    # a periodic mesh carries no Outer face set at all
    with pytest.raises(SpaceMismatchError, match="Outer"):
        restrict.codomain(mx.center)


def test_codomain_rejects_an_outer_without_an_inner_sibling(restrict):
    # a ChebyshevMesh carries the Lobatto Outer but no interior-face
    # Inner family: the row un-seeds itself with a taught message
    mesh = ChebyshevMesh(6, (0.0, 1.0), name="z")
    with pytest.raises(SpaceMismatchError, match="no Inner"):
        restrict.codomain(mesh.outer)


def test_codomain_rejects_a_dirichlet_outer(restrict, mz):
    # a Dirichlet condition drops the member boundary DOF of Outer, so
    # the interior-selection alignment no longer holds
    with pytest.raises(SpaceMismatchError, match="boundary"):
        restrict.codomain(mz.nodal(NodeSet.OUTER, bc=BC.DIRICHLET))


# ================================================================
#  Exactness: the restriction drops the two boundary faces
# ================================================================
def test_restriction_selects_the_interior_faces_exactly(walled, mz):
    w = walled.create_field(mz.outer, init=lambda z: z**2 - 0.3 * z)
    r = Restriction()["z"](w)
    assert r.function_space.bare.factor("z") is mz.inner
    outer = np.asarray(w.data)
    inner = np.asarray(r.data)
    # Inner[m] == Outer[m + 1]: the n + 1 faces minus the two walls
    assert np.allclose(inner, outer[1:-1])
    assert inner.shape[0] == outer.shape[0] - 2


def test_restriction_is_exact_on_a_stretched_mesh():
    # a pure node selection carries no metric, so it is exact on a
    # stretched (mapped) mesh as well as a uniform one
    mesh = MappedIntervalMesh(
        N, (0.0, 1.0), lambda s: s + 0.1 * np.sin(2 * np.pi * s),
        periodic=False, name="z")
    grid = Grid((mesh,))
    w = grid.create_field(mesh.outer, init=lambda z: 2.0 * z + 1.0)
    r = Restriction()["z"](w)
    assert np.allclose(np.asarray(r.data),
                       np.asarray(w.data)[1:-1])


# ================================================================
#  Distributed seam (the (0, 1) footprint must be synced)
# ================================================================
def _seam_grid(device_ids):
    # tiny horizontal so the negotiation shards the bounded vertical
    # (the fallback axis): the Outer(z) face set then straddles the
    # shard seams, exercising the restriction's one-slot-above ghost.
    mx = IntervalMesh(4, (0.0, 1.0), periodic=True, name="x")
    my = IntervalMesh(4, (0.0, 1.0), periodic=True, name="y")
    mz = IntervalMesh(16, (0.0, 3.0), periodic=False, name="z")
    grid = Grid((mx, my, mz), device_ids=device_ids)
    space = TensorProductSpace.of(mx.center, my.center, mz.outer)
    return grid, space


def _seam_init(x, y, z):
    # a column with a strong vertical gradient at the shard seams, so a
    # stale (zero) ghost is a large, unmistakable error there
    return jnp.sin(z) + 0.1 * x * y + z * z


@pytest.mark.multi_device
def test_restrict_syncs_the_seam_face_on_a_sharded_vertical(
        forced_devices):
    # Regression: Restriction reads Outer[m + 1], so the last interior
    # face of each shard is the neighbour shard's boundary face. With
    # the (0, 1) requirement _ensure_valid syncs that ghost; the pre-fix
    # halo-0 declaration left it at the reshard's zero fill, giving an
    # O(1) error exactly on the z-shard seams (3/4, 7/8, 11/12 here).
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid_many, space_many = _seam_grid(None)
    assert any(name == "z" for name, _
               in grid_many.decomposition.default_layout.device_axes)
    grid_one, space_one = _seam_grid((0,))

    def restricted(grid, space):
        w = grid.create_field(space, init=_seam_init)
        r = Restriction()["z"](w)
        gathered = grid.decomposition.gather(
            r._data, r.function_space)
        return np.asarray(gathered), np.asarray(w.data)

    r_many, w_many = restricted(grid_many, space_many)
    r_one, _ = restricted(grid_one, space_one)
    # forced-CPU reassociation keeps a pure selection bitwise-equal; a
    # real device-count bug is O(1) at the seam (5.9 pre-fix here)
    assert np.array_equal(r_many, r_one)
    # and it is exactly the interior faces of the (gathered) Outer field
    assert np.array_equal(r_many, w_many[..., 1:-1])


def test_restrict_is_reverse_differentiable():
    # step-path operator (the hydrostatic vertical-advection flux): a
    # jax.grad of a quadratic loss through the restriction matches a
    # central finite difference (the added sync is differentiable; the
    # forward kernel is untouched).
    mz = IntervalMesh(N, (0.0, 3.0), periodic=False, name="z")
    grid = Grid((mz,))

    def loss(scale):
        w = grid.create_field(
            mz.outer, init=lambda z: jnp.sin(z) + 1.0)
        r = Restriction()["z"](w.with_data(w.data * scale))
        return jnp.sum(r.data ** 2)

    g = float(jax.grad(loss)(1.3))
    eps = 1e-4
    fd = (loss(1.3 + eps) - loss(1.3 - eps)) / (2 * eps)
    assert np.isfinite(g)
    assert abs(g - float(fd)) <= 1e-4 * abs(float(fd))
