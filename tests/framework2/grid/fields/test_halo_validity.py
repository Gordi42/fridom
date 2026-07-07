"""
Halo-validity bookkeeping on fields (task 1.8, stage A).

Description
-----------
``ScalarField._halo_valid`` is the per-name count of currently valid
ghost layers: static pytree aux data that participates in the
treedef (jit caches must key on sync placement). Stage A is
behavior-neutral — every exchange still happens — so these tests pin
the plumbing only: the zero default, the ``grid.sync`` stamp, the
preserving functional updates, and the pytree contract. The
consumption-side placement lands in stage C
(``notes/framework2/sync_redo_plan.md``).
"""
import jax
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh


@pytest.fixture
def grid():
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    my = IntervalMesh(16, (0.0, 2.0), name="y")
    return Grid((mx, my))


@pytest.fixture
def f(grid):
    return grid.create_field(init=lambda x, y: x + y, name="f")


# ================================================================
#  Construction defaults
# ================================================================
def test_created_fields_claim_zero_validity(f):
    assert f.halo_valid == HaloSpec.zero(("x", "y"))


def test_validity_covers_exactly_the_space_names(grid):
    mx = grid.factors[0]
    single = grid.create_field(mx.center)
    assert single.halo_valid == HaloSpec.zero(("x",))


def test_arithmetic_results_claim_zero_validity(f):
    assert (f + f).halo_valid == HaloSpec.zero(("x", "y"))
    assert (2.0 * f).halo_valid == HaloSpec.zero(("x", "y"))


# ================================================================
#  The sync stamp
# ================================================================
def test_grid_sync_stamps_the_negotiated_widths(grid, f):
    synced = grid.sync(f)
    assert synced.halo_valid == grid.decomposition.halo.over(
        ("x", "y"))


def test_operator_results_carry_the_kernel_claim(grid, f):
    # consumption-side contract (stage C): the application syncs the
    # operand (widths), the kernel consumes its reach on the applied
    # axis, and the result keeps the remainder
    d = f.diff("x")
    w = grid.decomposition.halo
    assert d.halo_valid["x"] == w["x"] - 1
    assert d.halo_valid["y"] == w["y"]
    # the triggered sync was memoized onto the operand
    assert f.halo_valid == w.over(("x", "y"))


# ================================================================
#  Preserving functional updates
# ================================================================
def test_with_metadata_preserves_validity(grid, f):
    synced = grid.sync(f)
    assert synced.with_metadata(
        name="g").halo_valid == synced.halo_valid


def test_conj_preserves_validity(grid):
    f = grid.create_field(
        init=lambda x, y: x + 0 * y, name="c").as_complex()
    synced = grid.sync(f)
    assert synced.conj().halo_valid == synced.halo_valid


def test_with_data_resets_validity(grid, f):
    synced = grid.sync(f)
    fresh = synced.with_data(jnp.zeros(f.shape))
    assert fresh.halo_valid == HaloSpec.zero(("x", "y"))


# ================================================================
#  Pytree contract
# ================================================================
def test_flatten_unflatten_roundtrips_validity(grid, f):
    synced = grid.sync(f)
    leaves, treedef = jax.tree_util.tree_flatten(synced)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.halo_valid == synced.halo_valid


def test_treedef_keys_on_validity(grid, f):
    # a cached jit trace embeds sync placement, so two fields that
    # differ only in claimed validity must not hit the same cache
    synced = grid.sync(f)
    assert f.halo_valid != synced.halo_valid
    t_fresh = jax.tree_util.tree_structure(f)
    t_synced = jax.tree_util.tree_structure(synced)
    assert t_fresh != t_synced


def test_jit_boundary_restores_validity(grid, f):
    synced = grid.sync(f)

    @jax.jit
    def identity(field):
        return field

    out = identity(synced)
    assert out.halo_valid == synced.halo_valid
    assert jnp.array_equal(out.data, synced.data)
