"""Tests for fridom.spatial.operators.fallback (stage F1)."""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.fallback import (
    Fallback,
    UpwindOne,
    graded_ladder,
    graded_reconstruction,
)
from fridom.spatial.operators.weno import WenoReconstruction


# ================================================================
#  Fixtures and helpers
# ================================================================
@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")  # periodic


@pytest.fixture
def my():
    return IntervalMesh(16, (0.0, 1.0), periodic=False, name="y")


def sin_cell_averages(a, b, n):
    """Exact cell averages of sin(2 pi y) over n cells of [a, b]."""
    edges = np.linspace(a, b, n + 1)
    anti = -np.cos(2.0 * np.pi * edges) / (2.0 * np.pi)
    dx = (b - a) / n
    return (anti[1:] - anti[:-1]) / dx


# ================================================================
#  Ladder builder and construction
# ================================================================
def test_graded_ladder_is_widest_first():
    ladder = graded_ladder(5, "left")
    assert isinstance(ladder[0], WenoReconstruction)
    assert [rung.order for rung in ladder] == [5, 3, 1]
    assert isinstance(ladder[-1], UpwindOne)
    assert all(rung.bias == "left" for rung in ladder)


def test_graded_ladder_order_three():
    ladder = graded_ladder(3, "right")
    assert [rung.order for rung in ladder] == [3, 1]
    assert all(rung.bias == "right" for rung in ladder)


def test_fallback_reduced_rows_and_structure():
    op = graded_reconstruction(5, "left")
    assert isinstance(op, Fallback)
    assert op.reduced_rows == 2  # K = halo(3) - slack(1)
    assert op.interior.order == 5
    assert [r.order for r in op.boundary] == [3, 1]


def test_fallback_is_interned_on_structure():
    ladder = graded_ladder(5, "left")
    a = Fallback(ladder[0], ladder[1:])
    b = Fallback(ladder[0], ladder[1:])
    assert a is b  # interned on (interior, tuple(boundary))


def test_upwind_one_self_interns_on_bias():
    # D6: structurally-equal rungs are the same object (the identity
    # the graded Fallback keys its interning on)
    assert UpwindOne("left") is UpwindOne("left")
    assert UpwindOne("right") is UpwindOne("right")
    assert UpwindOne("left") is not UpwindOne("right")


def test_upwind_one_rejects_bad_bias():
    with pytest.raises(ValueError, match="bias"):
        UpwindOne("up")


def test_upwind_one_binding_does_not_mutate_singleton():
    base = UpwindOne("right")
    bound = base["x"]
    assert base.bound_axis is None  # unbound singleton untouched
    assert bound.bound_axis == "x"
    assert bound is UpwindOne("right")["x"]


def test_graded_reconstruction_self_interns():
    # now the leaf rungs self-intern, graded_reconstruction's Fallback
    # coalesces on its own structure (no memo, no papering over)
    assert graded_reconstruction(5) is graded_reconstruction(5)
    right = graded_reconstruction(3, "right")
    assert right is graded_reconstruction(3, "right")
    assert graded_reconstruction(5) is not graded_reconstruction(3)


def test_fallback_rejects_wrong_rung_count():
    with pytest.raises(ValueError, match="boundary rungs"):
        Fallback(WenoReconstruction(5), (WenoReconstruction(3),))


def test_fallback_rejects_wrong_rung_order():
    with pytest.raises(ValueError, match="order"):
        Fallback(WenoReconstruction(5),
                 (WenoReconstruction(3), WenoReconstruction(3)))


def test_fallback_rejects_bias_mismatch():
    with pytest.raises(ValueError, match="bias"):
        Fallback(WenoReconstruction(5, "left"),
                 (WenoReconstruction(3, "right"), UpwindOne("right")))


def test_dispatch_kind():
    assert graded_reconstruction(5).dispatch_kind == "reconstruct"
    assert UpwindOne().dispatch_kind == "reconstruct"


# ================================================================
#  Codomain: bounded is legal (unlike plain WENO)
# ================================================================
def test_codomain_periodic_is_right(mx):
    op = graded_reconstruction(5, "left")
    assert op.codomain(mx.cell_avg) is mx.right


def test_codomain_bounded_is_inner(my):
    op = graded_reconstruction(5, "left")
    # the decisive legality: bounded CellAvg -> Inner (no raise)
    assert op.codomain(my.cell_avg) is my.inner


def test_plain_weno_still_raises_on_bounded(my):
    # unchanged: plain WENO-5 is a space error on a bounded axis
    with pytest.raises(SpaceMismatchError, match="designed-for"):
        WenoReconstruction(5).codomain(my.cell_avg)


def test_codomain_rejects_non_cell_avg(my):
    op = graded_reconstruction(5, "left")
    with pytest.raises(SpaceMismatchError, match="CellAvg"):
        op.codomain(my.inner)


def test_codomain_rejects_complex(my):
    op = graded_reconstruction(5, "left")
    with pytest.raises(SpaceMismatchError, match="complex"):
        op.codomain(my.cell_avg.as_complex())


def test_requirements(my, mx):
    op = graded_reconstruction(5, "left")
    bounded = op.requirements(my.cell_avg)
    assert bounded.halo == 3  # interior WENO-5 halo dominates
    # F6: the bounded axis may be sharded (the wall rows are patched
    # behind the decomposition seam), so layout is "any", not "local"
    assert bounded.layout == "any"
    periodic = op.requirements(mx.cell_avg)
    assert periodic.halo == 3
    assert periodic.layout == "any"


# ================================================================
#  The decisive gate: no exterior read (NaN poison)
# ================================================================
@pytest.mark.parametrize("bias", ["left", "right"])
def test_nan_poison_gate_proves_interior_only(bias):
    n = 16
    mesh = IntervalMesh(n, (0.0, 1.0), periodic=False, name="y")
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"y": 3}))
    f = grid.create_field(
        mesh.cell_avg,
        data=jnp.asarray(sin_cell_averages(0.0, 1.0, n)))
    width = grid.decomposition.halo["y"]
    # poison every exterior halo cell with NaN and CLAIM the ghosts
    # valid, so the consumption-side sync does not overwrite them:
    # any exterior read now propagates a NaN into the true output
    storage = f._data
    poisoned = storage.at[:width].set(jnp.nan)
    poisoned = poisoned.at[storage.shape[0] - width:].set(jnp.nan)
    f._data = poisoned
    f._halo_valid = HaloSpec({"y": width})

    op = graded_reconstruction(5, bias)["y"]
    result = op(f)
    assert result.function_space.bare is mesh.inner
    assert bool(jnp.all(jnp.isfinite(result.data)))


# ================================================================
#  Interior convergence (design order at interior faces)
# ================================================================
@pytest.mark.parametrize("bias", ["left", "right"])
def test_interior_convergence_is_fifth_order(bias):
    op = graded_reconstruction(5, bias)
    errors = []
    sizes = (32, 64, 128)
    for n in sizes:
        mesh = IntervalMesh(n, (0.0, 1.0), periodic=False, name="y")
        grid = Grid((mesh,))
        grid.negotiate(halo=HaloSpec({"y": 3}))
        f = grid.create_field(
            mesh.cell_avg,
            data=jnp.asarray(sin_cell_averages(0.0, 1.0, n)))
        g = op["y"](f)
        assert g.function_space.bare is mesh.inner
        y = np.asarray(grid.evaluation_nodes(mesh.inner).data)
        err = np.abs(np.asarray(g.data) - np.sin(2.0 * np.pi * y))
        # interior faces only: drop the K reduced rows at each wall
        # and the critical points of sin (WENO-JS degrades there)
        interior = np.zeros_like(y, dtype=bool)
        interior[op.reduced_rows: y.size - op.reduced_rows] = True
        mask = interior & (np.abs(np.cos(2.0 * np.pi * y)) > 0.3)
        errors.append(err[mask].max())
    slopes = [np.log2(errors[k] / errors[k + 1]) for k in range(2)]
    assert min(slopes) > 4.4  # design order 5 at interior faces


def test_boundary_rows_are_finite_and_reduced(my):
    # the wall rows are computed (finite) and differ from what the
    # exterior-reading wide kernel would give (they are reduced)
    grid = Grid((my,))
    grid.negotiate(halo=HaloSpec({"y": 3}))
    n = my.n_cells
    f = grid.create_field(
        my.cell_avg, data=jnp.asarray(sin_cell_averages(0.0, 1.0, n)))
    g = graded_reconstruction(5, "left")["y"](f)
    assert bool(jnp.all(jnp.isfinite(g.data)))
    assert g.data.shape == (n - 1,)


# ================================================================
#  Periodic path matches plain WENO (no walls to reduce)
# ================================================================
def test_periodic_matches_plain_weno(mx):
    grid = Grid((mx,))
    grid.negotiate(halo=HaloSpec({"x": 3}))
    rng = np.random.default_rng(3)
    f = grid.create_field(mx.cell_avg,
                          data=jnp.asarray(rng.normal(size=8)))
    graded = graded_reconstruction(5, "left")["x"](f)
    plain = WenoReconstruction(5, "left")["x"](f)
    assert graded.function_space.bare is mx.right
    assert np.allclose(np.asarray(graded.data), np.asarray(plain.data))


def test_metadata_is_kept(my):
    grid = Grid((my,))
    grid.negotiate(halo=HaloSpec({"y": 3}))
    f = grid.create_field(my.cell_avg, name="q", units="kg")
    assert graded_reconstruction(5)["y"](f).name == "q"
