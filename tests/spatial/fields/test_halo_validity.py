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
(``design/plans/done/sync_redo_plan.md``).
"""
import jax
import jax.numpy as jnp
import pytest

from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh


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


def test_arithmetic_on_fresh_operands_claims_zero_validity(f):
    # storage-frame arithmetic claims the operands' minimum, which
    # for fresh (zero-claim) operands is zero
    assert (f + f).halo_valid == HaloSpec.zero(("x", "y"))
    assert (2.0 * f).halo_valid == HaloSpec.zero(("x", "y"))


# ================================================================
#  Arithmetic claim propagation (storage-frame combine)
# ================================================================
def test_add_keeps_the_operands_minimum_validity(grid, f):
    # both operands fully synced: their valid ghost slots combine
    # into valid ghost slots (linear fills commute with +)
    a = grid.sync(f)
    b = grid.sync(grid.create_field(init=lambda x, y: x * y))
    assert (a + b).halo_valid == grid.decomposition.halo.over(
        ("x", "y"))
    assert (a - b).halo_valid == a.halo_valid.merge_min(b.halo_valid)


def test_mixed_validity_degrades_to_the_minimum(grid, f):
    # one synced, one fresh operand: the sum can only claim the
    # layers every operand had — zero
    synced = grid.sync(f)
    assert (synced + f).halo_valid == HaloSpec.zero(("x", "y"))
    assert (f - synced).halo_valid == HaloSpec.zero(("x", "y"))


def test_stencil_output_sums_keep_the_kernel_claim(grid, f):
    # the tendency-sum pattern: two diff outputs on one staggered
    # space keep their per-axis claims through the sum
    g = grid.create_field(init=lambda x, y: x * y)
    d1, d2 = f.diff("x"), g.diff("x")
    s = d1 + d2
    assert s.halo_valid == d1.halo_valid.merge_min(d2.halo_valid)
    # Center -> Right consumes the high side; the sum keeps the shared
    # two-sided claim (low side full, high side down one)
    w = grid.decomposition.halo["x"]
    assert s.halo_valid.interval("x") == (w, max(w - 1, 0))


def test_scaling_and_negation_keep_validity(grid, f):
    synced = grid.sync(f)
    assert (2.0 * synced).halo_valid == synced.halo_valid
    assert (synced / 2.0).halo_valid == synced.halo_valid
    assert (-synced).halo_valid == synced.halo_valid


def test_scalar_shift_keeps_validity_on_periodic_axes(grid, f):
    # the periodic wrap fill reproduces constants
    synced = grid.sync(f)
    assert (synced + 1.0).halo_valid == synced.halo_valid
    assert (1.0 - synced).halo_valid == synced.halo_valid


def test_scalar_shift_drops_the_claim_on_bounded_axes():
    # the bounded fills (Dirichlet odd/vacant) do not reproduce
    # constants: the shifted field's bounded-axis claim resets
    mx = IntervalMesh(16, (0.0, 1.0), periodic=False, name="x")
    my = IntervalMesh(16, (0.0, 2.0), name="y")
    grid = Grid((mx, my))
    synced = grid.sync(grid.create_field(init=lambda x, y: x + y))
    shifted = synced + 1.0
    assert shifted.halo_valid["x"] == 0
    assert shifted.halo_valid["y"] == synced.halo_valid["y"]


def test_lifted_combines_take_the_true_shape_route(grid, f):
    # a real + complex combine lifts one operand (promotion): the
    # storage fast path does not apply and the result re-stores
    # with zero claims
    synced = grid.sync(f)
    complexified = grid.sync(f.as_complex())
    assert (synced + complexified).halo_valid == HaloSpec.zero(
        ("x", "y"))


def test_storage_frame_combines_match_the_true_shape_values(grid, f):
    # value parity: the fast path changes the frame, not the math
    a = grid.sync(f)
    b = grid.sync(grid.create_field(init=lambda x, y: x * y))
    assert jnp.array_equal((a + b).data, a.data + b.data)
    assert jnp.array_equal((a - b).data, a.data - b.data)
    assert jnp.array_equal((3.0 * a).data, 3.0 * a.data)
    assert jnp.array_equal((a + 2.5).data, a.data + 2.5)
    assert jnp.array_equal((-a).data, -(a.data))


def test_valid_ghost_slots_of_a_sum_equal_the_fill(grid, f):
    # the soundness contract: the sum's claimed ghost slots hold
    # exactly what a fresh exchange would write there
    a = grid.sync(f)
    b = grid.sync(grid.create_field(init=lambda x, y: x * y))
    s = a + b
    refilled = grid.sync(
        s.with_data(s.data))  # zero-claim twin, freshly exchanged
    assert jnp.array_equal(s._data, refilled._data)


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
    # Center -> Right consumes only the high side (two-sided claim)
    assert d.halo_valid.interval("x") == (w["x"], max(w["x"] - 1, 0))
    assert d.halo_valid["y"] == w["y"]
    # the triggered sync is memoized in the external identity cache,
    # never onto the treedef-participating operand (direction a)
    assert f.halo_valid == HaloSpec.zero(("x", "y"))


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
