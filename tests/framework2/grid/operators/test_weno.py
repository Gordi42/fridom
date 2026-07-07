"""Tests for fridom.framework2.grid.operators.weno."""
import copy
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators import weno as wk
from fridom.framework2.grid.operators.base import EigenbasisError
from fridom.framework2.grid.operators.fallback import (
    Fallback,
    graded_reconstruction,
)
from fridom.framework2.grid.operators.reconstruct import (
    LinearReconstruction,
)
from fridom.framework2.grid.operators.select import Where
from fridom.framework2.grid.operators.weno import (
    WenoReconstruction,
    weno_reconstruct,
    weno_tables,
    weno_weights,
)


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")


def sin_averages(n):
    """Exact cell averages of sin(2 pi x) over n unit-interval cells."""
    edges = np.linspace(0.0, 1.0, n + 1)
    antiderivative = -np.cos(2.0 * np.pi * edges) / (2.0 * np.pi)
    return (antiderivative[1:] - antiderivative[:-1]) * n


def wrap(arr, width):
    """Periodic halo fill of a 1D cell array."""
    return np.concatenate([arr[-width:], arr, arr[:width]])


def true_faces(full, n, order, bias, width):
    """Slice the kernel output onto the n true faces (after cell i)."""
    m0 = order // 2 if bias == "left" else order // 2 - 1
    return full[width - m0: width - m0 + n]


# ================================================================
#  Construction and static surface
# ================================================================
def test_dispatch_kind():
    assert WenoReconstruction().dispatch_kind == "reconstruct"


def test_order_and_bias_knobs():
    op = WenoReconstruction()
    assert op.order == 5
    assert op.bias == "left"
    op3 = WenoReconstruction(3, bias="right")
    assert op3.order == 3
    assert op3.bias == "right"


def test_even_order_is_rejected():
    with pytest.raises(ValueError, match="odd"):
        WenoReconstruction(4)


def test_unsupported_odd_orders_are_rejected():
    with pytest.raises(ValueError, match="iteration 1"):
        WenoReconstruction(7)
    with pytest.raises(ValueError, match="iteration 1"):
        WenoReconstruction(1)


def test_non_integer_order_is_rejected():
    with pytest.raises(TypeError, match="integer"):
        WenoReconstruction(5.0)


def test_unknown_bias_is_rejected():
    with pytest.raises(ValueError, match="bias"):
        WenoReconstruction(5, bias="up")


@pytest.mark.parametrize(("order", "halo"), [(3, 2), (5, 3)])
def test_requirements(order, halo, mx):
    req = WenoReconstruction(order).requirements(mx.cell_avg)
    assert req.halo == halo
    assert req.layout == "any"


def test_eigenvalues_raises(mx):
    # nonlinear: no symbol, the raising base is correct
    with pytest.raises(EigenbasisError):
        WenoReconstruction().eigenvalues(None, mx.cell_avg)


# ================================================================
#  Per-factor signature (codomain)
# ================================================================
def test_codomain_periodic_cell_avg_to_right(mx):
    for bias in ("left", "right"):
        op = WenoReconstruction(5, bias=bias)
        assert op.codomain(mx.cell_avg) is mx.right


def test_codomain_rejects_non_cell_avg(mx):
    op = WenoReconstruction()
    for space in (mx.center, mx.right, mx.face_avg):
        with pytest.raises(SpaceMismatchError, match="CellAvg"):
            op.codomain(space)


def test_codomain_rejects_complex(mx):
    with pytest.raises(SpaceMismatchError, match="complex"):
        WenoReconstruction().codomain(mx.cell_avg.as_complex())


def test_codomain_rejects_bounded_axes(my):
    # bounded-axis boundary biasing is designed-for: a space error,
    # never a silent fallback
    with pytest.raises(SpaceMismatchError, match="designed-for"):
        WenoReconstruction().codomain(my.cell_avg)


# ================================================================
#  The boundary="graded" constructor knob (F2, R2 parity)
# ================================================================
def test_graded_boundary_returns_bounded_legal_fallback(my, mx):
    # boundary="graded" mints the graded Fallback (a different class),
    # bounded-legal: CellAvg -> Inner resolves and does NOT raise
    op = WenoReconstruction(5, boundary="graded")
    assert isinstance(op, Fallback)
    assert op.codomain(my.cell_avg) is my.inner
    assert op.codomain(mx.cell_avg) is mx.right  # periodic still Right
    assert op.interior.order == 5
    assert op.interior.bias == "left"
    assert [rung.order for rung in op.boundary] == [3, 1]


def test_graded_boundary_is_interned():
    # the constructor spelling is a stable interned handle
    a = WenoReconstruction(5, boundary="graded")
    b = WenoReconstruction(5, boundary="graded")
    assert a is b
    # ... and (now the leaf rungs self-intern, so graded_reconstruction's
    # Fallback coalesces on its own) the knob IS the factory object: the
    # memo the knob used to carry is gone, identity holds by construction
    assert a is graded_reconstruction(5)


@pytest.mark.parametrize("bias", ["left", "right"])
def test_graded_boundary_routes_both_biases(bias):
    op = WenoReconstruction(5, bias=bias, boundary="graded")
    assert isinstance(op, Fallback)
    assert op.interior.bias == bias
    assert all(rung.bias == bias for rung in op.boundary)
    # distinct interned handle per bias
    assert op is WenoReconstruction(5, bias=bias, boundary="graded")


def test_graded_boundary_validates_order_and_bias():
    with pytest.raises(ValueError, match="iteration 1"):
        WenoReconstruction(7, boundary="graded")
    with pytest.raises(ValueError, match="bias"):
        WenoReconstruction(5, bias="up", boundary="graded")


def test_boundary_none_is_the_plain_kernel_unchanged(my, mx):
    # default and explicit "none" are the plain periodic-only kernel:
    # same class, still a space error on a bounded axis
    for op in (WenoReconstruction(5), WenoReconstruction(5, boundary="none")):
        assert isinstance(op, WenoReconstruction)
        assert not isinstance(op, Fallback)
        assert op.order == 5
        with pytest.raises(SpaceMismatchError, match="designed-for"):
            op.codomain(my.cell_avg)
        assert op.codomain(mx.cell_avg) is mx.right


def test_unknown_boundary_is_rejected():
    with pytest.raises(ValueError, match="boundary must be one of"):
        WenoReconstruction(5, boundary="wat")


# ================================================================
#  Plain-path self-interning (D6) + binding safety
# ================================================================
def test_plain_kernel_self_interns_on_structure():
    # structurally-equal plain kernels are the same object (D6)
    assert WenoReconstruction(5) is WenoReconstruction(5)
    assert WenoReconstruction(3, "right") is WenoReconstruction(3, "right")
    # distinct structure => distinct objects
    assert WenoReconstruction(5) is not WenoReconstruction(3)
    assert WenoReconstruction(5, "left") is not WenoReconstruction(5, "right")


def test_binding_does_not_mutate_the_unbound_singleton():
    base = WenoReconstruction(5)
    bound = base["x"]
    # the unbound singleton is untouched by binding (the _rebind
    # copy.copy hazard: __copy__ hands _rebind a fresh clone)
    assert base.bound_axis is None
    assert bound.bound_axis == "x"
    # binding is itself interned on (base, axis)
    assert bound is WenoReconstruction(5)["x"]
    assert bound.unbound is base


def test_copy_bypasses_interning():
    base = WenoReconstruction(5)
    clone = copy.copy(base)
    assert clone is not base
    assert clone.order == 5
    assert clone.bias == "left"


# ================================================================
#  Coefficient tables (exact rational identities)
# ================================================================
@pytest.mark.parametrize("r", [2, 3])
def test_optimal_weights_reproduce_the_full_stencil(r):
    # sum_m d_m * candidate_m == the (2r-1)-cell reconstruction at
    # the shared face — the linear-weight consistency the formal
    # order rests on, checked exactly over the rationals
    full = wk._shu_row(2 * r - 1, r)
    combo = [Fraction(0)] * (2 * r - 1)
    for m in range(r):
        row = wk._shu_row(r, r - m)
        for offset, coeff in enumerate(row):
            combo[m + offset] += wk._OPTIMAL_WEIGHTS[r][m] * coeff
    assert combo == list(full)


def test_shu_rows_match_the_classic_weno5_tables():
    # Jiang & Shu (1996) candidate reconstructions at x_{i+1/2}
    assert wk._shu_row(3, 3) == (
        Fraction(1, 3), Fraction(-7, 6), Fraction(11, 6))
    assert wk._shu_row(3, 2) == (
        Fraction(-1, 6), Fraction(5, 6), Fraction(1, 3))
    assert wk._shu_row(3, 1) == (
        Fraction(1, 3), Fraction(5, 6), Fraction(-1, 6))


def test_right_tables_mirror_the_left_tables():
    for order in (3, 5):
        left = weno_tables(order, "left")
        right = weno_tables(order, "right")
        r = len(left.optimal)
        assert right.optimal == left.optimal
        for m in range(r):
            assert right.offsets[m] == r - 1 - left.offsets[m]
            assert right.coeffs[m] == tuple(reversed(left.coeffs[m]))


# ================================================================
#  Kernel exactness and symmetry
# ================================================================
@pytest.mark.parametrize("bias", ["left", "right"])
@pytest.mark.parametrize("order", [3, 5])
def test_polynomial_exactness(order, bias):
    # every candidate reconstructs degree <= r - 1 exactly and the
    # weights sum to one, so WENO-3 is exact on linears and WENO-5
    # on quadratics (the doc's average-to-point exactness promise)
    length = 20
    j = np.arange(length)
    if order == 3:
        averages = 2.0 * (j + 0.5) + 1.0
        exact = lambda x: 2.0 * x + 1.0  # noqa: E731
    else:
        averages = ((j + 1.0) ** 3 - j**3) / 3.0
        exact = lambda x: x**2  # noqa: E731
    out = np.asarray(
        weno_reconstruct(jnp.asarray(averages), 0, order, bias))
    t = np.arange(length - order + 1)
    faces = t + order // 2 + (1 if bias == "left" else 0)
    np.testing.assert_allclose(out, exact(faces.astype(float)),
                               rtol=1e-12, atol=1e-11)


@pytest.mark.parametrize("order", [3, 5])
def test_right_bias_is_the_mirror_of_left_bias(order):
    # exact in exact arithmetic; floating point only reorders the
    # candidate summation, so the comparison is tight-tolerance
    rng = np.random.default_rng(3)
    arr = jnp.asarray(rng.normal(size=24))
    right = weno_reconstruct(arr, 0, order, "right")
    mirrored = weno_reconstruct(arr[::-1], 0, order, "left")[::-1]
    np.testing.assert_allclose(np.asarray(right),
                               np.asarray(mirrored),
                               rtol=1e-13, atol=1e-14)


def test_kernel_rejects_short_axes():
    with pytest.raises(ValueError, match="shorter"):
        weno_reconstruct(jnp.ones(4), 0, 5, "left")


# ================================================================
#  Smooth-case behavior (measured convergence, 3 resolutions)
# ================================================================
@pytest.mark.parametrize("bias", ["left", "right"])
@pytest.mark.parametrize(
    ("order", "sizes", "mask_min"),
    [(3, (32, 64, 128), 0.5), (5, (16, 32, 64), 0.3)])
def test_smooth_convergence_at_design_order(order, sizes, mask_min,
                                            bias):
    # max error over faces away from the critical points of sin
    # (where WENO-JS is known to degrade); the measured slope must
    # reach the formal order p
    errors = []
    for n in sizes:
        mesh = IntervalMesh(n, (0.0, 1.0), name="x")
        grid = Grid((mesh,))
        grid.negotiate(halo=HaloSpec({"x": order // 2 + 1}))
        f = grid.create_field(mesh.cell_avg,
                              data=jnp.asarray(sin_averages(n)))
        g = WenoReconstruction(order, bias=bias)["x"](f)
        assert g.function_space.bare is mesh.right
        x = np.asarray(grid.evaluation_nodes(mesh.right).data)
        err = np.abs(np.asarray(g.data) - np.sin(2.0 * np.pi * x))
        mask = np.abs(np.cos(2.0 * np.pi * x)) > mask_min
        errors.append(err[mask].max())
    slopes = [np.log2(errors[k] / errors[k + 1]) for k in range(2)]
    assert min(slopes) > order - 0.6


@pytest.mark.parametrize(
    ("order", "sizes", "min_ratio"),
    [(3, (64, 128, 256), 1.6), (5, (16, 32, 64), 3.0)])
def test_weights_approach_the_linear_optimum(order, sizes, min_ratio):
    # on smooth data the nonlinear weights converge to the optimal
    # (linear) weights: O(dx) for r = 2, O(dx^2) for r = 3, measured
    # away from the critical points
    width = order
    optimal = weno_tables(order, "left").optimal
    deviations = []
    for n in sizes:
        padded = jnp.asarray(wrap(sin_averages(n), width))
        weights = weno_weights(padded, 0, order, "left")
        x = (np.arange(n) + 1.0) / n
        mask = np.abs(np.cos(2.0 * np.pi * x)) > 0.8
        deviations.append(max(
            np.abs(true_faces(np.asarray(w), n, order, "left",
                              width)[mask] - d).max()
            for w, d in zip(weights, optimal, strict=True)))
    ratios = [deviations[k] / deviations[k + 1] for k in range(2)]
    assert min(ratios) > min_ratio


def test_weights_are_normalized():
    rng = np.random.default_rng(5)
    arr = jnp.asarray(rng.normal(size=20))
    for order in (3, 5):
        weights = weno_weights(arr, 0, order, "left")
        total = np.asarray(sum(weights))
        np.testing.assert_allclose(total, 1.0, rtol=1e-12)


# ================================================================
#  ENO property at a step (no new extrema, TV vs linear weights)
# ================================================================
@pytest.mark.parametrize("bias", ["left", "right"])
@pytest.mark.parametrize("order", [3, 5])
def test_eno_no_new_extrema_at_a_step(order, bias):
    n = 64
    width = order
    x_mid = (np.arange(n) + 0.5) / n
    step = np.where((x_mid > 0.25) & (x_mid < 0.75), 1.0, 0.0)
    padded = jnp.asarray(wrap(step, width))
    faces = true_faces(
        np.asarray(weno_reconstruct(padded, 0, order, bias)),
        n, order, bias, width)
    tol = 1e-6
    assert faces.min() >= step.min() - tol
    assert faces.max() <= step.max() + tol


@pytest.mark.parametrize("order", [3, 5])
def test_total_variation_beats_the_linear_reconstruction(order):
    # the same-order linear (optimal-weight) reconstruction rings at
    # the step (Gibbs); WENO must not add total variation
    n = 64
    width = order
    r = (order + 1) // 2
    x_mid = (np.arange(n) + 0.5) / n
    step = np.where((x_mid > 0.25) & (x_mid < 0.75), 1.0, 0.0)
    padded = wrap(step, width)

    weno_faces = true_faces(
        np.asarray(weno_reconstruct(jnp.asarray(padded), 0, order,
                                    "left")),
        n, order, "left", width)
    row = [float(c) for c in wk._shu_row(order, r)]
    windows = [padded[j:j + n + 2 * width - order + 1]
               for j in range(order)]
    linear_full = sum(c * w for c, w in zip(row, windows,
                                            strict=True))
    linear_faces = true_faces(linear_full, n, order, "left", width)

    def total_variation(values):
        closed = np.concatenate([values, values[:1]])
        return np.abs(np.diff(closed)).sum()

    tv_cells = total_variation(step)
    assert total_variation(weno_faces) <= tv_cells + 1e-6
    assert total_variation(linear_faces) > tv_cells + 1e-3
    assert (linear_faces.max() > step.max() + 1e-3
            or linear_faces.min() < step.min() - 1e-3)


# ================================================================
#  Operator application (alignment against the raw kernel)
# ================================================================
@pytest.mark.parametrize("bias", ["left", "right"])
@pytest.mark.parametrize("order", [3, 5])
def test_operator_matches_the_wrapped_kernel(order, bias):
    n = 16
    rng = np.random.default_rng(7)
    values = rng.normal(size=n)
    mesh = IntervalMesh(n, (0.0, 1.0), name="x")
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"x": order // 2 + 1}))
    f = grid.create_field(mesh.cell_avg, data=jnp.asarray(values))
    g = WenoReconstruction(order, bias=bias)["x"](f)
    assert g.function_space.bare is mesh.right
    padded = jnp.asarray(wrap(values, order))
    expected = true_faces(
        np.asarray(weno_reconstruct(padded, 0, order, bias)),
        n, order, bias, order)
    np.testing.assert_allclose(np.asarray(g.data), expected,
                               rtol=1e-13, atol=1e-14)


def test_metadata_is_kept():
    mesh = IntervalMesh(16, (0.0, 1.0), name="x")
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"x": 3}))
    f = grid.create_field(mesh.cell_avg, name="q", units="kg")
    assert WenoReconstruction()["x"](f).name == "q"  # same quantity


def test_weno_needs_the_negotiated_halo(mx):
    # the default-negotiated halo (2) is too small for the
    # right-biased order-5 window; the reach check raises
    grid = Grid((mx,))
    f = grid.create_field(mx.cell_avg)
    with pytest.raises(ValueError, match="halo width"):
        WenoReconstruction(5, bias="right")["x"](f)


def test_left_bias_order_3_runs_on_the_default_halo(mx):
    # left-biased order 3 reaches one cell each side beyond the
    # face's cell: the default halo (2) covers it without
    # renegotiation
    grid = Grid((mx,))
    f = grid.create_field(mx.cell_avg,
                          data=jnp.asarray(sin_averages(8)))
    g = WenoReconstruction(3, bias="left")["x"](f)
    assert g.function_space.bare is mx.right


def test_2d_application_binds_per_axis():
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    my = IntervalMesh(16, (0.0, 1.0), name="y")
    grid = Grid((mx, my))
    grid.negotiate(halo=HaloSpec({"x": 3, "y": 3}))
    rng = np.random.default_rng(9)
    values = rng.normal(size=(16, 16))
    f = grid.create_field(mx.cell_avg * my.cell_avg,
                          data=jnp.asarray(values))
    g = WenoReconstruction(5)["y"](f)
    assert g.function_space.bare is mx.cell_avg * my.right
    # column-wise equality with the 1D operator
    grid_y = Grid((IntervalMesh(16, (0.0, 1.0), name="y"),))
    grid_y.negotiate(halo=HaloSpec({"y": 3}))
    for column in (0, 7):
        f_y = grid_y.create_field(
            grid_y.factors[0].cell_avg,
            data=jnp.asarray(values[column]))
        g_y = WenoReconstruction(5)["y"](f_y)
        assert np.array_equal(np.asarray(g.data[column]),
                              np.asarray(g_y.data))


# ================================================================
#  The upwind pair (biased instances + ("select", ...) dispatch)
# ================================================================
def test_upwind_pair_selects_the_correct_side():
    n = 16
    mesh = IntervalMesh(n, (0.0, 1.0), name="x")
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"x": 3}))
    rng = np.random.default_rng(11)
    f = grid.create_field(mesh.cell_avg,
                          data=jnp.asarray(rng.normal(size=n)))
    left = WenoReconstruction(5, bias="left")["x"](f)
    right = WenoReconstruction(5, bias="right")["x"](f)
    # the biased pair genuinely differs on rough data
    assert not np.array_equal(np.asarray(left.data),
                              np.asarray(right.data))
    u = grid.create_field(mesh.right,
                          init=lambda x: jnp.sin(2.0 * jnp.pi * x))
    cond = grid.create_field(
        mesh.right, data=(u.data > 0).astype(f.dtype))
    select = grid.dispatch.resolve("select", mesh.right)
    assert isinstance(select, Where)
    upwind = select(cond, left, right)
    assert upwind.function_space.bare is mesh.right
    expected = jnp.where(u.data > 0, left.data, right.data)
    assert np.array_equal(np.asarray(upwind.data),
                          np.asarray(expected))
    # +flux everywhere -> the left-biased (upwind) side exactly
    ones = grid.create_field(mesh.right, data=jnp.ones(n))
    assert np.array_equal(np.asarray(select(ones, left, right).data),
                          np.asarray(left.data))
    zeros = grid.create_field(mesh.right, data=jnp.zeros(n))
    assert np.array_equal(
        np.asarray(select(zeros, left, right).data),
        np.asarray(right.data))


# ================================================================
#  Registry reading: WENO is opt-in, never a default row
# ================================================================
def test_weno_is_opt_in_not_the_default(mx):
    grid = Grid((mx,))
    default = grid.dispatch.resolve("reconstruct", mx.cell_avg)
    assert isinstance(default, LinearReconstruction)
    weno_op = WenoReconstruction(5)
    merged = grid.dispatch.merge(
        {("reconstruct", mx.cell_avg): weno_op})
    assert merged.resolve("reconstruct", mx.cell_avg) is weno_op
    # the default table is layered, not mutated
    assert isinstance(grid.dispatch.resolve("reconstruct",
                                            mx.cell_avg),
                      LinearReconstruction)


# ================================================================
#  Device-count invariance (bitwise, sharded periodic mesh)
# ================================================================
def test_weno_reconstruct_is_device_count_invariant():
    def compute(device_ids):
        mesh = IntervalMesh(16, (0.0, 1.0), name="x")
        grid = Grid((mesh,), device_ids=device_ids)
        grid.negotiate(halo=HaloSpec({"x": 3}))
        f = grid.create_field(
            mesh.cell_avg,
            init=lambda x: jnp.sin(2.0 * jnp.pi * x)
            + 0.3 * jnp.cos(4.0 * jnp.pi * x))
        left = WenoReconstruction(5, bias="left")["x"](f)
        right = WenoReconstruction(5, bias="right")["x"](f)
        cond = grid.create_field(
            mesh.right,
            init=lambda x: (jnp.sin(2.0 * jnp.pi * x) > 0)
            .astype(jnp.float64))
        upwind = Where()(cond, left, right)
        return (left.data, right.data, upwind.data)

    for many, one in zip(compute(None), compute((0,)), strict=True):
        assert np.array_equal(np.asarray(many), np.asarray(one))


# ================================================================
#  jit-cleanliness (one trace across repeated same-shape calls)
# ================================================================
def test_weno_kernel_traces_once(compile_counter):
    kernel = jax.jit(weno_reconstruct,
                     static_argnames=("axis", "order", "bias"))
    rng = np.random.default_rng(15)
    arrs = [jnp.asarray(rng.normal(size=(24, 8))) for _ in range(3)]
    compile_counter.reset()
    results = [kernel(arr, axis=0, order=5, bias="left")
               for arr in arrs]
    jax.block_until_ready(results)
    assert compile_counter.count == 1
