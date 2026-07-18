"""Tests for GridTransfer (spatial/operators/transfer.py)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh
from fridom.spatial.operators.transfer import GridTransfer
from fridom.spatial.spaces.tensor_product import TensorProductSpace


# ================================================================
#  Self-contained builders (each shard duplicates its helpers)
# ================================================================
def cell_space(grid, kind="center"):
    """Build the all-cell product space (Center or CellAvg)."""
    factors = tuple(getattr(mesh, kind) for mesh in grid.factors)
    if len(factors) == 1:
        return factors[0]
    return TensorProductSpace.of(*factors)


def random_field(grid, seed, kind="center"):
    """Build a random real cell field spanning every coordinate."""
    space = cell_space(grid, kind)
    shape = tuple(mesh.n_cells for mesh in grid.factors)
    data = jax.random.normal(jax.random.PRNGKey(seed), shape)
    return grid.create_field(space, data=data)


def weighted_dot(a, b):
    """Return the measure-weighted L2 product (krylov `_dot`)."""
    return float(jnp.sum((a * b).integrate().data))


def adjoint_rel(transfer, seed=0):
    """Relative violation of <R f, g>_H == <f, P g>_h."""
    fine = random_field(transfer.fine, seed)
    coarse = random_field(transfer.coarse, seed + 1)
    lhs = weighted_dot(transfer.restrict(fine), coarse)
    rhs = weighted_dot(fine, transfer.prolong(coarse))
    return abs(lhs - rhs) / (abs(rhs) + 1e-300)


# ================================================================
#  Adjointness -- the GA-1 gate
# ================================================================
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize(
    ("n", "factor"),
    [pytest.param(8, 2, id="even-x2"),
     pytest.param(12, 2, id="even-x2-alt"),
     pytest.param(9, 3, id="odd-x3"),
     pytest.param(8, 4, id="even-x4")])
def test_adjoint_periodic(order, n, factor):
    fine = Grid((IntervalMesh(n, (0.0, 1.0), name="x"),))
    transfer = GridTransfer(fine, fine.coarsened(factor), order=order)
    assert adjoint_rel(transfer) < 1e-14


@pytest.mark.parametrize("order", [1, 2])
def test_adjoint_bounded(order):
    fine = Grid((IntervalMesh(8, (0.0, 1.0), periodic=False,
                              name="x"),))
    transfer = GridTransfer(fine, fine.coarsened(2), order=order)
    assert adjoint_rel(transfer) < 1e-14


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("kind", ["center", "cell_avg"])
def test_adjoint_2d_both_families(order, kind):
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),
                 IntervalMesh(6, (0.0, 2.0), name="y")))
    transfer = GridTransfer(fine, fine.coarsened(2), order=order)
    coarse_field = random_field(transfer.coarse, 5, kind)
    fine_field = random_field(transfer.fine, 4, kind)
    lhs = weighted_dot(transfer.restrict(fine_field), coarse_field)
    rhs = weighted_dot(fine_field, transfer.prolong(coarse_field))
    assert abs(lhs - rhs) / abs(rhs) < 1e-14


@pytest.mark.parametrize("order", [1, 2])
def test_adjoint_semicoarsened(order):
    # x coarsened, y kept at full resolution (MG-D4)
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),
                 IntervalMesh(6, (0.0, 2.0), name="y")))
    transfer = GridTransfer(fine, fine.coarsened({"x": 2}), order=order)
    assert transfer.ratios == {"x": 2, "y": 1}
    assert adjoint_rel(transfer) < 1e-14


@pytest.mark.parametrize("order", [1, 2])
def test_adjoint_mapped_stretch(order):
    mesh = MappedIntervalMesh(8, (0.0, 1.0), lambda s: s ** 1.5,
                              periodic=False, name="z")
    fine = Grid((mesh,))
    transfer = GridTransfer(fine, fine.coarsened(2), order=order)
    assert adjoint_rel(transfer) < 1e-14


@pytest.mark.parametrize("order", [1, 2])
def test_adjoint_on_immersed_grid(order):
    immersed = ImmersedDomain(
        lambda x, y: ((x < 0.7) & (y >= 0.0)).astype(float))
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),
                 IntervalMesh(8, (0.0, 2.0), name="y")),
                immersed=immersed)
    transfer = GridTransfer(fine, fine.coarsened(2), order=order)
    # adjoint in the plain (geometry) measure-weighted product; the
    # transfer is geometric, not mask-aware, in iteration 1
    assert adjoint_rel(transfer) < 1e-14
    assert transfer.coarse.immersed is not None


# ================================================================
#  Conservation and constant preservation (GA-1)
# ================================================================
@pytest.mark.parametrize("order", [1, 2])
def test_restrict_conserves_the_integral(order):
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),
                 IntervalMesh(8, (0.0, 2.0), name="y")))
    transfer = GridTransfer(fine, fine.coarsened(2), order=order)
    field = random_field(fine, 11)
    restricted = transfer.restrict(field)
    assert abs(field.integrate().item()
               - restricted.integrate().item()) < 1e-13


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("periodic", [True, False])
def test_prolong_preserves_constants(order, periodic):
    fine = Grid((IntervalMesh(8, (0.0, 1.0), periodic=periodic,
                              name="x"),))
    coarse = fine.coarsened(2)
    transfer = GridTransfer(fine, coarse, order=order)
    ones = coarse.create_field(cell_space(coarse),
                               data=jnp.ones((4,)))
    prolonged = transfer.prolong(ones)
    assert float(jnp.max(jnp.abs(prolonged.data - 1.0))) < 1e-14


# ================================================================
#  Round-trip (order 1 is a left inverse)
# ================================================================
@pytest.mark.parametrize("periodic", [True, False])
def test_order1_restrict_is_left_inverse_of_prolong(periodic):
    fine = Grid((IntervalMesh(8, (0.0, 1.0), periodic=periodic,
                              name="x"),))
    coarse = fine.coarsened(2)
    transfer = GridTransfer(fine, coarse, order=1)
    x = random_field(coarse, 21)
    back = transfer.restrict(transfer.prolong(x))
    assert float(jnp.max(jnp.abs(back.data - x.data))) < 1e-13


def test_order2_round_trip_is_not_identity():
    # only order 1 is a left inverse; order 2 is not (documented)
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    coarse = fine.coarsened(2)
    transfer = GridTransfer(fine, coarse, order=2)
    x = random_field(coarse, 22)
    back = transfer.restrict(transfer.prolong(x))
    assert float(jnp.max(jnp.abs(back.data - x.data))) > 1e-6


# ================================================================
#  Prolong / restrict on a hand-built independent grid pair
# ================================================================
def test_pair_from_independent_grids():
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    coarse = Grid((IntervalMesh(4, (0.0, 1.0), name="x"),))
    transfer = GridTransfer(fine, coarse, order=1)
    assert transfer.ratios == {"x": 2}
    assert adjoint_rel(transfer) < 1e-14


# ================================================================
#  Autodiff regression (differentiability policy)
# ================================================================
@pytest.mark.parametrize("order", [1, 2])
def test_grad_through_restrict_matches_fd(order):
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    coarse = fine.coarsened(2)
    transfer = GridTransfer(fine, coarse, order=order)
    x0 = jax.random.normal(jax.random.PRNGKey(31), (8,))

    def loss(data):
        field = fine.create_field(cell_space(fine), data=data)
        return jnp.sum(transfer.restrict(field).data ** 2)

    grad = np.asarray(jax.grad(loss)(x0))
    assert np.all(np.isfinite(grad))
    eps = 1e-5
    fd = np.zeros(8)
    for i in range(8):
        plus = loss(x0.at[i].add(eps))
        minus = loss(x0.at[i].add(-eps))
        fd[i] = (plus - minus) / (2 * eps)
    assert np.max(np.abs(grad - fd)) / np.max(np.abs(fd)) < 1e-4


@pytest.mark.parametrize("order", [1, 2])
def test_grad_through_prolong_matches_fd(order):
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    coarse = fine.coarsened(2)
    transfer = GridTransfer(fine, coarse, order=order)
    x0 = jax.random.normal(jax.random.PRNGKey(32), (4,))

    def loss(data):
        field = coarse.create_field(cell_space(coarse), data=data)
        return jnp.sum(transfer.prolong(field).data ** 2)

    grad = np.asarray(jax.grad(loss)(x0))
    assert np.all(np.isfinite(grad))
    eps = 1e-5
    fd = np.zeros(4)
    for i in range(4):
        plus = loss(x0.at[i].add(eps))
        minus = loss(x0.at[i].add(-eps))
        fd[i] = (plus - minus) / (2 * eps)
    assert np.max(np.abs(grad - fd)) / np.max(np.abs(fd)) < 1e-4


# ================================================================
#  Construction and input validation (taught errors)
# ================================================================
def test_rejects_bad_order():
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    with pytest.raises(ValueError, match="order"):
        GridTransfer(fine, fine.coarsened(2), order=3)


def test_rejects_mismatched_names():
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    other = Grid((IntervalMesh(4, (0.0, 1.0), name="y"),))
    with pytest.raises(ValueError, match="same coordinate names"):
        GridTransfer(fine, other)


def test_rejects_non_integer_ratio():
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    coarse = Grid((IntervalMesh(3, (0.0, 1.0), name="x"),))
    with pytest.raises(ValueError, match="positive integer"):
        GridTransfer(fine, coarse)


def test_restrict_rejects_a_field_off_the_fine_grid():
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    coarse = fine.coarsened(2)
    transfer = GridTransfer(fine, coarse)
    with pytest.raises(ValueError, match="fine grid"):
        transfer.restrict(random_field(coarse, 1))


def test_prolong_rejects_a_field_off_the_coarse_grid():
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    coarse = fine.coarsened(2)
    transfer = GridTransfer(fine, coarse)
    with pytest.raises(ValueError, match="coarse grid"):
        transfer.prolong(random_field(fine, 1))


def test_rejects_a_complex_field():
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    coarse = fine.coarsened(2)
    transfer = GridTransfer(fine, coarse)
    field = random_field(fine, 1).as_complex()
    with pytest.raises(ValueError, match="real scalar"):
        transfer.restrict(field)


def test_rejects_a_staggered_face_field():
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    coarse = fine.coarsened(2)
    transfer = GridTransfer(fine, coarse)
    face = fine.create_field(fine.factors[0].right)
    with pytest.raises(ValueError, match="collocated cell"):
        transfer.restrict(face)


def test_metadata_is_carried_across_the_transfer():
    fine = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    coarse = fine.coarsened(2)
    transfer = GridTransfer(fine, coarse)
    field = random_field(fine, 1).with_metadata(name="theta")
    assert transfer.restrict(field).name == "theta"
    field_c = random_field(coarse, 1).with_metadata(name="theta")
    assert transfer.prolong(field_c).name == "theta"


# ================================================================
#  Multi-device parity and collective inspection (GA-2)
# ================================================================
def _single_vs_forced(order, n, factor, kind="center"):
    """Run restrict/prolong on 1 and all devices; return both."""
    def run(device_ids):
        fine = Grid((IntervalMesh(n, (0.0, 1.0), name="x"),
                     IntervalMesh(n, (0.0, 2.0), name="y")),
                    device_ids=device_ids)
        transfer = GridTransfer(fine, fine.coarsened(factor),
                                order=order)
        fine_field = random_field(fine, 41, kind)
        coarse_field = random_field(transfer.coarse, 42, kind)
        restricted = np.asarray(transfer.restrict(fine_field).data)
        prolonged = np.asarray(transfer.prolong(coarse_field).data)
        return restricted, prolonged

    ids = tuple(range(jax.device_count()))
    return run((0,)), run(ids)


@pytest.mark.multi_device
@pytest.mark.parametrize("order", [1, 2])
def test_forced4_parity_aligned(order):
    (r1, p1), (r4, p4) = _single_vs_forced(order, 16, 2)
    assert np.allclose(r1, r4, atol=1e-14)
    assert np.allclose(p1, p4, atol=1e-14)


@pytest.mark.multi_device
@pytest.mark.parametrize("order", [1, 2])
def test_forced4_parity_non_nesting_fallback(order):
    # 12 cells over 4 devices -> per-shard extent 3, not divisible by
    # the ratio 2: the non-nesting blocking falls back to the global
    # reblock path. Correctness must still match single-device.
    (r1, p1), (r4, p4) = _single_vs_forced(order, 12, 2)
    assert np.allclose(r1, r4, atol=1e-13)
    assert np.allclose(p1, p4, atol=1e-13)


@pytest.mark.multi_device
def test_forced4_order1_restrict_has_no_all_gather():
    ids = tuple(range(jax.device_count()))
    fine = Grid((IntervalMesh(16, (0.0, 1.0), name="x"),
                 IntervalMesh(16, (0.0, 2.0), name="y")),
                device_ids=ids)
    transfer = GridTransfer(fine, fine.coarsened(2), order=1)
    field = random_field(fine, 43)

    def restrict(data):
        return transfer.restrict(
            fine.create_field(cell_space(fine), data=data)).data

    hlo = jax.jit(restrict).lower(field.data).compile().as_text()
    assert "all-gather" not in hlo


@pytest.mark.multi_device
def test_forced4_rejects_mismatched_device_sets():
    if jax.device_count() < 2:
        pytest.skip("needs several devices")
    # fine sharded over two devices, coarse on one: distinct device
    # sets, which the transfer refuses (a hierarchy shares one mesh).
    fine = Grid((IntervalMesh(16, (0.0, 1.0), name="x"),),
                device_ids=(0, 1))
    coarse = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),),
                  device_ids=(0,))
    with pytest.raises(ValueError, match="shared device set"):
        GridTransfer(fine, coarse)


# ================================================================
#  Profile fields: ConstantSpace pass-through (GM-D3)
# ================================================================
def sigma_profile_pair(nx, ny, nz, factor=2):
    """Fine/coarse 3-D grid pair with a mapped (sigma) vertical."""
    sigma = MappedIntervalMesh(nz, (0.0, 1.0), lambda s: s ** 1.5,
                               periodic=False, name="z")
    fine = Grid((IntervalMesh(nx, (0.0, 1.0), name="x"),
                 IntervalMesh(ny, (0.0, 2.0), name="y"),
                 sigma))
    coarse = fine.coarsened({"x": factor, "y": factor})
    return fine, coarse


def profile_space(grid):
    """Cell x, cell y, constant z: a barotropic ``Profile`` space."""
    mx, my, mz = grid.factors
    return TensorProductSpace.of(mx.center, my.center, mz.constant)


def profile_field(grid, seed):
    """Random Profile field (shape ``(nx, ny, 1)``)."""
    nx, ny, _ = (mesh.n_cells for mesh in grid.factors)
    data = jax.random.normal(jax.random.PRNGKey(seed), (nx, ny, 1))
    return grid.create_field(profile_space(grid), data=data)


@pytest.mark.parametrize("order", [1, 2])
def test_profile_transfer_shapes_and_spaces(order):
    # a Profile field (cell x/y, constant z) transfers on its horizontal
    # factors alone; the constant z factor passes through untransferred
    fine, coarse = sigma_profile_pair(8, 8, 6)
    transfer = GridTransfer(fine, coarse, order=order)
    field = profile_field(fine, 3)
    restricted = transfer.restrict(field)
    assert restricted.grid is coarse
    assert tuple(restricted.data.shape) == (4, 4, 1)
    assert restricted.function_space.factors[2].is_constant
    back = transfer.prolong(restricted)
    assert back.grid is fine
    assert tuple(back.data.shape) == (8, 8, 1)
    assert back.function_space.factors[2].is_constant


@pytest.mark.parametrize("order", [1, 2])
def test_profile_adjointness_with_constant_z(order):
    # <R f, g>_H == <f, P g>_h with a ConstantSpace z factor present, on
    # a mapped (sigma) vertical so the measure weighting is exercised
    fine, coarse = sigma_profile_pair(8, 6, 6)
    transfer = GridTransfer(fine, coarse, order=order)
    fine_field = profile_field(fine, 7)
    coarse_field = profile_field(coarse, 8)
    lhs = weighted_dot(transfer.restrict(fine_field), coarse_field)
    rhs = weighted_dot(fine_field, transfer.prolong(coarse_field))
    assert abs(lhs - rhs) / (abs(rhs) + 1e-300) < 1e-12


def test_profile_field_is_accepted_by_the_input_gate():
    # the ConstantSpace z factor no longer trips _check_input (GM-D3)
    fine, coarse = sigma_profile_pair(8, 8, 6)
    transfer = GridTransfer(fine, coarse)
    # neither direction raises on the Profile field
    transfer.restrict(profile_field(fine, 1))
    transfer.prolong(profile_field(coarse, 2))
