"""Non-uniform (stretched-factor) WENO tables and kernels.

Prefix-mirrored shard of ``tests/spatial/operators/test_weno.py`` for
route (ii) of ``design/plans/active/high_order_mapped_plan.md``: the
width-derived Shu-1998 tables, the width co-operand ``cell_widths``,
and the biased reconstructions on ``MappedIntervalMesh`` factors
(bare kernel, ``WenoReconstruction``, and the graded ``Fallback``).

The reference tables are recomputed here from an **independent**
numpy-polynomial evaluation of the same definitions (the primitive-
function Lagrange basis and Shu's smoothness integral, both in
``numpy.polynomial``), so the production generator -- a hand-rolled
``jnp`` elementwise expression -- is checked against numbers it does
not share a line of code with. Self-contained by the suite's
convention (``--import-mode=importlib``, no cross-test imports).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import Polynomial

from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import (
    MappedIntervalMesh,
)
from fridom.spatial.operators.fallback import graded_reconstruction
from fridom.spatial.operators.weno import (
    _OPTIMAL_WEIGHTS,
    _SMOOTHNESS_ROWS,
    _SMOOTHNESS_SCALE,
    WENO_EPS,
    WenoReconstruction,
    _shu_row,
    _static_centered_row,
    _static_linear_row,
    _window_views,
    cell_widths,
    centered_row_windows,
    linear_row_windows,
    nonuniform_tables,
    weno_combine,
    weno_reconstruct,
    weno_tables,
)


# ================================================================
#  Independent reference (numpy.polynomial, the survey's spike)
# ================================================================
def lagrange_basis(nodes):
    """Lagrange basis polynomials on ``nodes``."""
    basis = []
    for j, xj in enumerate(nodes):
        poly = Polynomial([1.0])
        for k, xk in enumerate(nodes):
            if k != j:
                poly = poly * Polynomial([-xk, 1.0]) / (xj - xk)
        basis.append(poly)
    return basis


def recon_basis(faces):
    """phi_l(x): the weight polynomial of cell average l in p(x)."""
    basis = lagrange_basis(faces)
    derivs = [b.deriv() for b in basis]
    widths = np.diff(faces)
    return [widths[cell] * sum(derivs[j]
                               for j in range(cell + 1, len(faces)))
            for cell in range(len(widths))]


def smooth_form(phis, cell):
    """Shu's smoothness quadratic form over ``cell`` (scaled by dx)."""
    a, b = cell
    dx = b - a
    r = len(phis)
    form = np.zeros((r, r))
    for k in range(1, r):
        dphis = [p.deriv(k) for p in phis]
        for i in range(r):
            for m in range(r):
                prod = (dphis[i] * dphis[m]).integ()
                form[i, m] += dx ** (2 * k - 1) * (prod(b) - prod(a))
    return form


def reference_tables(widths, r, *, lstsq=False):
    """Candidate rows, ideal weights, beta forms and the full row.

    Evaluated in local, cell-scaled coordinates (the target face at 0,
    lengths in units of the upwind cell) -- in absolute coordinates the
    monomial basis of ``numpy.polynomial`` loses four digits on a
    stretched column, which is the conditioning finding of
    ``design/research/nonuniform_weno_survey.md`` section 4.
    """
    widths = np.asarray(widths, dtype=float)
    faces = np.concatenate([[0.0], np.cumsum(widths)])
    faces = (faces - faces[r]) / widths[r - 1]
    cand, betas = [], []
    for m in range(r):
        phis = recon_basis(faces[m:m + r + 1])
        cand.append([p(0.0) for p in phis])
        betas.append(smooth_form(phis, (faces[r - 1], faces[r])))
    cand = np.array(cand)
    full = np.array([p(0.0) for p in recon_basis(faces)])
    if lstsq:
        # a genuinely independent derivation of the ideal weights:
        # least-squares solve of the embedding, no closed form
        embed = np.zeros((2 * r - 1, r))
        for m in range(r):
            embed[m:m + r, m] = cand[m]
        ideal = np.linalg.lstsq(embed, full, rcond=None)[0]
    else:
        first = full[0] / cand[0][0]
        last = full[-1] / cand[r - 1][r - 1]
        ideal = (np.array([first, 1.0 - first]) if r == 2
                 else np.array([first, 1.0 - first - last, last]))
    return cand, ideal, np.array(betas), full


def reference_weno(values, widths, r):
    """Combine the reference tables into a WENO-JS face value."""
    cand, ideal, betas, _ = reference_tables(widths, r)
    q = np.array([cand[m] @ values[m:m + r] for m in range(r)])
    beta = np.array([values[m:m + r] @ betas[m] @ values[m:m + r]
                     for m in range(r)])
    alpha = ideal / (beta + WENO_EPS) ** 2
    return float((alpha @ q) / alpha.sum())


def generator_tables(widths, order):
    """Run the production generator on plain 1-element windows."""
    windows = tuple(jnp.asarray([w]) for w in widths)
    tables = nonuniform_tables(windows, order)
    r = (order + 1) // 2
    coeffs = np.array([[float(np.asarray(c)[0])
                        for c in tables.coeffs[m]] for m in range(r)])
    optimal = np.array([float(np.asarray(d)[0])
                        for d in tables.optimal])
    rows = np.array([[[float(np.asarray(c)[0]) for c in row]
                      for row in tables.beta_rows[m]]
                     for m in range(r)])
    return tables, coeffs, optimal, rows


def form_of(rows):
    """Rebuild the quadratic form from its square-root rows."""
    return sum(np.outer(row, row) for row in rows)


def random_widths(rng, order, count):
    """Positive cell widths with neighbour ratios up to ~20."""
    for _ in range(count):
        widths = np.exp(rng.uniform(-1.5, 1.5, order))
        yield widths / widths.min() * rng.uniform(0.05, 5.0)


# ================================================================
#  Meshes
# ================================================================
def wavy_map(s):
    """Smooth wavy stretching of the unit computational interval."""
    return s + 0.1 * jnp.sin(2.0 * jnp.pi * s) / (2.0 * jnp.pi)


TANH_STRETCH = 1.5


def tanh_map(s):
    """Map the unit interval as the coastal-upwelling column."""
    return (1.0 + jnp.tanh(TANH_STRETCH * (2.0 * s - 1.0))
            / jnp.tanh(TANH_STRETCH)) / 2.0


def periodic_mesh(n=16):
    """Stretched periodic z factor (the shard's wavy fixture)."""
    return MappedIntervalMesh(n, (0.0, 1.0), wavy_map, periodic=True,
                              name="z")


def bounded_mesh(n=16):
    """Stretched bounded z factor (the tanh column)."""
    return MappedIntervalMesh(n, (0.0, 1.0), tanh_map, name="z")


def faces_of(mapping, n):
    """Evaluate the n + 1 physical face positions."""
    return np.asarray(mapping(jnp.linspace(0.0, 1.0, n + 1)))


def centers_of(mapping, n):
    """Evaluate the n physical cell-center positions."""
    return np.asarray(mapping((jnp.arange(n) + 0.5) / n))


def cell_averages(mapping, n):
    """Exact cell averages of sin(2 pi x) on the mapped cells."""
    faces = faces_of(mapping, n)
    anti = -np.cos(2.0 * np.pi * faces) / (2.0 * np.pi)
    return np.diff(anti) / np.diff(faces)


def wrap(arr, width):
    """Periodic halo fill of a 1D cell array."""
    return np.concatenate([arr[-width:], arr, arr[:width]])


# ================================================================
#  1. The uniform lattice: the generator reproduces the static tables
# ================================================================
@pytest.mark.parametrize("order", [3, 5])
def test_generator_reproduces_the_static_tables(order):
    r = (order + 1) // 2
    static = weno_tables(order, "left")
    _, coeffs, optimal, rows = generator_tables((1.0,) * order, order)

    assert coeffs.shape == (r, r)
    np.testing.assert_allclose(
        coeffs, np.array([[float(c) for c in cand]
                          for cand in static.coeffs]),
        rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        optimal, np.array([float(d) for d in _OPTIMAL_WEIGHTS[r]]),
        rtol=0.0, atol=1e-15)

    # the smoothness QUADRATIC FORMS agree (the rows themselves are a
    # different -- exactly rank-(r-1) -- factorization of the same form)
    for m in range(r):
        static_form = sum(
            float(scale) * np.outer(row, row)
            for scale, row in zip(_SMOOTHNESS_SCALE[r],
                                  np.array(_SMOOTHNESS_ROWS[r][m],
                                           dtype=float), strict=True))
        np.testing.assert_allclose(form_of(rows[m]), static_form,
                                   rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("order", [3, 5])
def test_uniform_lattice_kernel_matches_the_static_kernel(order):
    # feeding a constant width window must reproduce the static
    # kernel's values (not merely its tables)
    rng = np.random.default_rng(11)
    values = rng.normal(size=order + 6)
    arr = jnp.asarray(values)
    widths = jnp.full_like(arr, 0.375)  # any uniform spacing
    for bias in ("left", "right"):
        static = np.asarray(weno_reconstruct(arr, 0, order, bias))
        derived = np.asarray(
            weno_reconstruct(arr, 0, order, bias, widths=widths))
        np.testing.assert_allclose(derived, static, rtol=1e-13,
                                   atol=1e-14)


# ================================================================
#  2. Random stretched windows against the numpy reference
# ================================================================
@pytest.mark.parametrize("order", [3, 5])
def test_generator_matches_the_reference_tables(order):
    r = (order + 1) // 2
    rng = np.random.default_rng(3)
    for widths in random_widths(rng, order, 12):
        _, coeffs, optimal, rows = generator_tables(widths, order)
        ref_c, ref_d, ref_b, ref_full = reference_tables(widths, r)
        np.testing.assert_allclose(coeffs, ref_c, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(optimal, ref_d, rtol=0.0, atol=1e-12)
        for m in range(r):
            np.testing.assert_allclose(form_of(rows[m]), ref_b[m],
                                       rtol=0.0, atol=1e-11)
        # the full row is exercised through the linear kernel, one
        # basis vector at a time
        width_windows = tuple(jnp.asarray([w]) for w in widths)
        for i in range(order):
            unit = tuple(jnp.asarray([1.0 if j == i else 0.0])
                         for j in range(order))
            row_i = float(np.asarray(linear_row_windows(
                unit, order, "left", width_windows))[0])
            assert abs(row_i - ref_full[i]) < 1e-12


@pytest.mark.parametrize("order", [3, 5])
def test_ideal_weights_are_positive_sum_one_and_embed(order):
    r = (order + 1) // 2
    rng = np.random.default_rng(5)
    for widths in random_widths(rng, order, 12):
        _, coeffs, optimal, _ = generator_tables(widths, order)
        assert optimal.min() > 0.0          # never clipped, never needs to be
        assert abs(optimal.sum() - 1.0) < 1e-15
        # sum_m d_m c^(m), embedded, equals the full row F
        _, _, _, ref_full = reference_tables(widths, r)
        combo = np.zeros(order)
        for m in range(r):
            combo[m:m + r] += optimal[m] * coeffs[m]
        np.testing.assert_allclose(combo, ref_full, rtol=0.0,
                                   atol=1e-12)


@pytest.mark.parametrize("order", [3, 5])
def test_ideal_weights_match_an_independent_least_squares_solve(order):
    # the closed form d_0 = F_0 / c^(0)_0 etc. against a least-squares
    # solve of the same embedding (an ill-conditioned but wholly
    # independent derivation -- hence the looser tolerance)
    r = (order + 1) // 2
    rng = np.random.default_rng(9)
    for widths in random_widths(rng, order, 10):
        _, _, optimal, _ = generator_tables(widths, order)
        _, ref_d, _, _ = reference_tables(widths, r, lstsq=True)
        np.testing.assert_allclose(optimal, ref_d, rtol=0.0, atol=1e-9)


# ================================================================
#  3. Scale invariance and polynomial exactness
# ================================================================
@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("factor", [1e-4, 1e3])
def test_tables_are_scale_invariant(order, factor):
    rng = np.random.default_rng(13)
    for widths in random_widths(rng, order, 10):
        _, c_one, d_one, b_one = generator_tables(widths, order)
        _, c_two, d_two, b_two = generator_tables(
            factor * widths, order)
        np.testing.assert_allclose(c_two, c_one, rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(d_two, d_one, rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(b_two, b_one, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("order", [3, 5])
def test_polynomial_exactness_of_the_rows(order):
    r = (order + 1) // 2
    rng = np.random.default_rng(17)
    for widths in random_widths(rng, order, 10):
        faces = np.concatenate([[0.0], np.cumsum(widths)])
        _, coeffs, _, _ = generator_tables(widths, order)
        width_windows = tuple(jnp.asarray([w]) for w in widths)
        for degree in range(order):
            poly = Polynomial([0.0] * degree + [1.0])
            anti = poly.integ()
            averages = np.diff(anti(faces)) / np.diff(faces)
            exact = poly(faces[r])
            scale = max(abs(exact), 1.0)
            if degree < r:  # each candidate reproduces degree <= r-1
                for m in range(r):
                    got = float(coeffs[m] @ averages[m:m + r])
                    assert abs(got - exact) < 1e-11 * scale
            # the full row reproduces degree <= 2r-2 = order - 1
            windows = tuple(jnp.asarray([v]) for v in averages)
            got = float(np.asarray(linear_row_windows(
                windows, order, "left", width_windows))[0])
            assert abs(got - exact) < 1e-11 * scale


def test_centered_row_of_size_two_is_the_width_weighted_mean():
    # the spec's closed form: (w_1 v_0 + w_0 v_1) / (w_0 + w_1)
    w0, w1 = 0.3, 1.7
    v0, v1 = -2.0, 5.0
    got = float(np.asarray(centered_row_windows(
        (jnp.asarray([v0]), jnp.asarray([v1])), 2,
        (jnp.asarray([w0]), jnp.asarray([w1]))))[0])
    assert abs(got - (w1 * v0 + w0 * v1) / (w0 + w1)) < 1e-15


@pytest.mark.parametrize("size", [2, 4])
def test_centered_row_is_exact_on_polynomials(size):
    rng = np.random.default_rng(19)
    for widths in random_widths(rng, size, 10):
        faces = np.concatenate([[0.0], np.cumsum(widths)])
        width_windows = tuple(jnp.asarray([w]) for w in widths)
        for degree in range(size):
            poly = Polynomial([0.0] * degree + [1.0])
            averages = np.diff(poly.integ()(faces)) / np.diff(faces)
            got = float(np.asarray(centered_row_windows(
                tuple(jnp.asarray([v]) for v in averages), size,
                width_windows))[0])
            exact = poly(faces[size // 2])
            assert abs(got - exact) < 1e-11 * max(abs(exact), 1.0)


# ================================================================
#  4. The bare kernel and the operator on a stretched periodic axis
# ================================================================
@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("bias", ["left", "right"])
def test_kernel_matches_the_reference_scheme(order, bias):
    n = 16
    r = (order + 1) // 2
    mesh = periodic_mesh(n)
    faces = faces_of(wavy_map, n)
    widths = np.diff(faces)
    values = cell_averages(wavy_map, n)
    padded = jnp.asarray(wrap(values, order))
    padded_w = jnp.asarray(wrap(widths, order))
    got = np.asarray(weno_reconstruct(padded, 0, order, bias,
                                      widths=padded_w))
    # window t reconstructs the face of window cell m0; compare a few
    # interior windows against the reference scheme (right bias = the
    # mirror, so its window is read reversed)
    for t in range(len(got)):
        window = np.asarray(padded)[t:t + order]
        window_w = np.asarray(padded_w)[t:t + order]
        if bias == "right":
            window, window_w = window[::-1], window_w[::-1]
        assert abs(got[t] - reference_weno(window, window_w, r)) < 1e-12
    assert mesh.n_cells == n


@pytest.mark.parametrize("order", [3, 5])
def test_right_bias_is_the_mirror_of_the_left_bias(order):
    n = 16
    faces = faces_of(wavy_map, n)
    values = cell_averages(wavy_map, n)
    padded = jnp.asarray(wrap(values, order))
    padded_w = jnp.asarray(wrap(np.diff(faces), order))
    right = np.asarray(
        weno_reconstruct(padded, 0, order, "right", widths=padded_w))
    mirrored = np.asarray(weno_reconstruct(
        padded[::-1], 0, order, "left", widths=padded_w[::-1]))[::-1]
    np.testing.assert_allclose(right, mirrored, rtol=0.0, atol=1e-15)


def design_order_errors(order, sizes, *, derived):
    """L-inf reconstruction error over three stretched resolutions."""
    errors = []
    for n in sizes:
        mesh = periodic_mesh(n)
        grid = Grid((mesh,))
        grid.negotiate(halo=HaloSpec({"z": order // 2 + 1}))
        values = cell_averages(wavy_map, n)
        f = grid.create_field(mesh.cell_avg, data=jnp.asarray(values))
        widths = cell_widths(f, "z") if derived else None
        padded = np.asarray(grid.sync(f)._data)
        width = grid.decomposition.halo["z"]
        full = np.asarray(weno_reconstruct(
            jnp.asarray(padded), 0, order, "left",
            widths=None if widths is None else jnp.asarray(widths)))
        m0 = order // 2
        got = full[width - m0: width - m0 + n]
        faces = faces_of(wavy_map, n)[1:]
        exact = np.sin(2.0 * np.pi * faces)
        # mask the critical points of sin (the WENO-JS order loss
        # there is a scheme property, not a mesh effect)
        keep = np.abs(np.cos(2.0 * np.pi * faces)) > 0.3
        errors.append(np.abs(got - exact)[keep].max())
    return np.array(errors)


def test_weno5_keeps_design_order_on_a_stretched_axis():
    sizes = (16, 32, 64)
    derived = design_order_errors(5, sizes, derived=True)
    rates = np.log2(derived[:-1] / derived[1:])
    assert rates.min() > 4.6, rates
    # the negative control: the STATIC tables on the same data drop to
    # 2nd order (survey variant A -- the guard's original reason)
    static = design_order_errors(5, sizes, derived=False)
    static_rates = np.log2(static[:-1] / static[1:])
    assert static_rates[-1] < 2.5, static_rates
    assert derived[-1] < 0.1 * static[-1]


def test_weno3_keeps_design_order_on_a_stretched_axis():
    sizes = (32, 64, 128)
    derived = design_order_errors(3, sizes, derived=True)
    rates = np.log2(derived[:-1] / derived[1:])
    # WENO-JS with r = 2 approaches its optimal weights only at O(h),
    # so ~2.7-3.0 is the design rate on a uniform mesh too.
    # No negative control here: the r = 2 rows differ from the uniform
    # ones only at O(h), so the static tables carry an O(h^2) row error
    # that hides under the O(h^3) WENO term until n ~ 256 (measured:
    # rates 3.10, 3.04, 2.73, 2.63 over n = 64 .. 1024 on a 4:1
    # stretch) -- too coarse a separation for a cheap test. The order-3
    # tables are pinned against the reference scheme instead
    # (test_kernel_matches_the_reference_scheme).
    assert rates.min() > 2.5, rates


@pytest.mark.parametrize("bias", ["left", "right"])
def test_operator_matches_the_wrapped_kernel_on_a_mapped_axis(bias):
    n = 16
    order = 5
    mesh = periodic_mesh(n)
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"z": 3}))
    values = cell_averages(wavy_map, n)
    f = grid.create_field(mesh.cell_avg, data=jnp.asarray(values))
    g = WenoReconstruction(order, bias=bias)["z"](f)
    assert g.function_space.bare is mesh.right

    widths = np.diff(faces_of(wavy_map, n))
    padded = jnp.asarray(wrap(values, order))
    padded_w = jnp.asarray(wrap(widths, order))
    full = np.asarray(weno_reconstruct(padded, 0, order, bias,
                                       widths=padded_w))
    m0 = order // 2 if bias == "left" else order // 2 - 1
    expected = full[order - m0: order - m0 + n]
    np.testing.assert_allclose(np.asarray(g.data), expected,
                               rtol=1e-13, atol=1e-14)


def test_uniform_factor_takes_the_static_path_bitwise():
    n = 16
    order = 5
    mesh = IntervalMesh(n, (0.0, 1.0), name="x")
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"x": 3}))
    rng = np.random.default_rng(23)
    f = grid.create_field(mesh.cell_avg,
                          data=jnp.asarray(rng.normal(size=n)))
    assert cell_widths(f, "x") is None      # no co-operand at all
    g = WenoReconstruction(order)["x"](f)
    padded = jnp.asarray(wrap(np.asarray(f.data), order))
    expected = np.asarray(
        weno_reconstruct(padded, 0, order, "left"))[order - 2:
                                                    order - 2 + n]
    assert np.array_equal(np.asarray(g.data), expected)


# ================================================================
#  5. The graded Fallback on a bounded tanh-stretched column
# ================================================================
@pytest.mark.parametrize("bias", ["left", "right"])
def test_graded_interior_faces_match_the_periodic_kernel(bias):
    n = 16
    order = 5
    mesh = bounded_mesh(n)
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"z": 3}))
    values = cell_averages(tanh_map, n)
    f = grid.create_field(mesh.cell_avg, data=jnp.asarray(values))
    g = graded_reconstruction(order, bias)["z"](f)
    got = np.asarray(g.data)                      # faces 1 .. n-1

    widths = np.diff(faces_of(tanh_map, n))
    r = (order + 1) // 2
    # K = 2 reduced faces per side; the interior faces are 3 .. n-3
    for face in range(3, n - 2):
        start = face - 1 - (order // 2 if bias == "left"
                            else order // 2 - 1)
        window = values[start:start + order]
        window_w = widths[start:start + order]
        if bias == "right":
            window, window_w = window[::-1], window_w[::-1]
        assert abs(got[face - 1]
                   - reference_weno(window, window_w, r)) < 1e-12


@pytest.mark.parametrize("bias", ["left", "right"])
def test_graded_wall_rungs_use_the_exact_wall_geometry(bias):
    n = 16
    order = 5
    mesh = bounded_mesh(n)
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"z": 3}))
    values = cell_averages(tanh_map, n)
    widths = np.diff(faces_of(tanh_map, n))
    f = grid.create_field(mesh.cell_avg, data=jnp.asarray(values))
    got = np.asarray(graded_reconstruction(order, bias)["z"](f).data)

    # left wall: face 1 is the 1st-order upwind cell, face 2 the
    # width-derived WENO-3 rung
    upwind = 0 if bias == "left" else 1
    assert got[0] == pytest.approx(values[upwind], abs=1e-14)
    # face 2 (output slot 1) is the WENO-3 rung: cells 0..2 (left
    # bias, offset 1) / 1..3 (right bias, offset 0)
    offset3 = 1 if bias == "left" else 0
    start = 2 - 1 - offset3
    window, window_w = values[start:start + 3], widths[start:start + 3]
    if bias == "right":
        window, window_w = window[::-1], window_w[::-1]
    assert abs(got[1] - reference_weno(window, window_w, 2)) < 1e-12

    # right wall (face n-1 is wall-adjacent, face n-2 the WENO-3 rung)
    upwind = n - 2 if bias == "left" else n - 1
    assert got[n - 2] == pytest.approx(values[upwind], abs=1e-14)
    start = (n - 2) - 1 - offset3
    window, window_w = values[start:start + 3], widths[start:start + 3]
    if bias == "right":
        window, window_w = window[::-1], window_w[::-1]
    assert abs(got[n - 3] - reference_weno(window, window_w, 2)) < 1e-12


@pytest.mark.parametrize("bias", ["left", "right"])
def test_graded_reads_no_exterior_value_on_a_stretched_column(bias):
    n = 16
    mesh = bounded_mesh(n)
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"z": 3}))
    f = grid.create_field(
        mesh.cell_avg, data=jnp.asarray(cell_averages(tanh_map, n)))
    width = grid.decomposition.halo["z"]
    storage = f._data
    poisoned = storage.at[:width].set(jnp.nan)
    poisoned = poisoned.at[storage.shape[0] - width:].set(jnp.nan)
    f._data = poisoned
    f._halo_valid = HaloSpec({"z": width})

    result = graded_reconstruction(5, bias)["z"](f)
    assert result.function_space.bare is mesh.inner
    assert bool(jnp.all(jnp.isfinite(result.data)))


def test_graded_interior_convergence_on_a_stretched_column():
    # design order at the faces the interior kernel keeps
    order = 5
    errors = []
    sizes = (32, 64, 128)
    for n in sizes:
        mesh = bounded_mesh(n)
        grid = Grid((mesh,))
        grid.negotiate(halo=HaloSpec({"z": 3}))
        f = grid.create_field(
            mesh.cell_avg,
            data=jnp.asarray(cell_averages(tanh_map, n)))
        got = np.asarray(graded_reconstruction(order, "left")["z"](f)
                         .data)
        faces = faces_of(tanh_map, n)[1:n]
        exact = np.sin(2.0 * np.pi * faces)
        interior = slice(2, n - 3)
        errors.append(np.abs(got - exact)[interior].max())
    errors = np.array(errors)
    rates = np.log2(errors[:-1] / errors[1:])
    assert rates.min() > 4.5, rates


# ================================================================
#  6. The width co-operand and its frames
# ================================================================
def test_cell_widths_primal_frame_is_the_cell_width_vector():
    n = 16
    mesh = periodic_mesh(n)
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"z": 3}))
    width = grid.decomposition.halo["z"]
    f = grid.create_field(mesh.cell_avg)
    widths = np.asarray(cell_widths(f, "z")).ravel()
    expected = np.diff(faces_of(wavy_map, n))
    assert widths.shape[0] == n + 2 * width
    np.testing.assert_allclose(widths[width:width + n], expected,
                               rtol=0.0, atol=1e-15)
    # the periodic ghost slots hold the wrap fill (the measure's exact
    # periodic extension), so a wrap-around window sees real widths
    np.testing.assert_allclose(widths[:width], expected[-width:],
                               rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(widths[width + n:], expected[:width],
                               rtol=0.0, atol=1e-15)
    # the Center (nodal) family shares the primal frame
    g = grid.create_field(mesh.center)
    np.testing.assert_allclose(
        np.asarray(cell_widths(g, "z")).ravel()[width:width + n],
        expected, rtol=0.0, atol=1e-15)


def test_cell_widths_dual_frame_periodic_right():
    n = 16
    mesh = periodic_mesh(n)
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"z": 3}))
    width = grid.decomposition.halo["z"]
    f = grid.create_field(mesh.right)
    widths = np.asarray(cell_widths(f, "z")).ravel()
    centers = centers_of(wavy_map, n)
    # Right slot j is face j + 1; its dual cell runs center j -> j + 1,
    # the last one wrapping across the periodic seam
    expected = np.append(np.diff(centers), centers[0] + 1.0
                         - centers[-1])
    np.testing.assert_allclose(widths[width:width + n], expected,
                               rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(widths[:width], expected[-width:],
                               rtol=0.0, atol=1e-15)


def test_cell_widths_dual_frame_bounded_inner_carries_half_cells():
    n = 16
    mesh = bounded_mesh(n)
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"z": 3}))
    width = grid.decomposition.halo["z"]
    f = grid.create_field(mesh.inner)
    widths = np.asarray(cell_widths(f, "z")).ravel()
    centers = centers_of(tanh_map, n)
    # true slots: faces 1 .. n-1, the interior dual cells
    np.testing.assert_allclose(widths[width:width + n - 1],
                               np.diff(centers), rtol=0.0, atol=1e-15)
    # the two WALL faces 0 and n live one slot outside the true DOFs
    # and own the clipped half cells
    assert widths[width - 1] == pytest.approx(centers[0] - 0.0,
                                              abs=1e-15)
    assert widths[width + n - 1] == pytest.approx(1.0 - centers[-1],
                                                  abs=1e-15)


def test_cell_widths_refuses_an_unsupported_node_set():
    mesh = bounded_mesh(8)
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"z": 3}))
    f = grid.create_field(mesh.outer)
    with pytest.raises(SpaceMismatchError, match="primal cells"):
        cell_widths(f, "z")


def test_cell_widths_needs_a_ghost_slot_for_the_wall_cells():
    mesh = bounded_mesh(8)
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"z": 0}))
    f = grid.create_field(mesh.inner)
    with pytest.raises(ValueError, match="at least one slot"):
        cell_widths(f, "z")


# ================================================================
#  Argument validation of the public seams
# ================================================================
def test_nonuniform_tables_checks_the_window_count():
    with pytest.raises(ValueError, match="5 windows"):
        nonuniform_tables((jnp.asarray([1.0]),) * 4, 5)


def test_centered_row_windows_checks_the_window_count():
    with pytest.raises(ValueError, match="4 windows"):
        centered_row_windows((jnp.asarray([1.0]),) * 4, 4,
                             (jnp.asarray([1.0]),) * 3)


@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("bias", ["left", "right"])
def test_static_rows_are_the_optimal_weight_combination(order, bias):
    tables = weno_tables(order, bias)
    row = [0.0] * order
    for m, offset in enumerate(tables.offsets):
        for i, coeff in enumerate(tables.coeffs[m]):
            row[offset + i] += tables.optimal[m] * coeff
    assert _static_linear_row(order, bias) == tuple(row)


@pytest.mark.parametrize("size", [2, 4])
def test_static_centered_row_is_the_shu_row(size):
    row = tuple(float(c) for c in _shu_row(size, size // 2))
    assert _static_centered_row(size) == row
    # and the uniform kernel applies it in window order, bitwise
    rng = np.random.default_rng(43)
    values = jnp.asarray(rng.normal(size=size + 3))
    windows = _window_views(values, 0, size)
    expected = None
    for weight, view in zip(row, windows, strict=True):
        term = weight * view
        expected = term if expected is None else expected + term
    assert np.array_equal(
        np.asarray(centered_row_windows(windows, size)),
        np.asarray(expected))


@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("bias", ["left", "right"])
def test_static_kernels_apply_their_rows_in_window_order(order, bias):
    rng = np.random.default_rng(29)
    values = jnp.asarray(rng.normal(size=order + 4))
    windows = _window_views(values, 0, order)
    row = _static_linear_row(order, bias)
    expected = None
    for weight, view in zip(row, windows, strict=True):
        term = weight * view
        expected = term if expected is None else expected + term
    assert np.array_equal(
        np.asarray(linear_row_windows(windows, order, bias)),
        np.asarray(expected))


def test_weno_combine_static_path_matches_weno_reconstruct():
    order = 5
    rng = np.random.default_rng(31)
    values = jnp.asarray(rng.normal(size=order + 4))
    for bias in ("left", "right"):
        windows = _window_views(values, 0, order)
        assert np.array_equal(
            np.asarray(weno_combine(windows, order, bias)),
            np.asarray(weno_reconstruct(values, 0, order, bias)))


def test_linear_row_windows_right_bias_mirrors_the_left_one():
    order = 5
    rng = np.random.default_rng(37)
    values = jnp.asarray(rng.normal(size=order + 4))
    widths = jnp.asarray(np.exp(rng.uniform(-1.0, 1.0, order + 4)))
    right = np.asarray(linear_row_windows(
        _window_views(values, 0, order), order, "right",
        _window_views(widths, 0, order)))
    mirrored = np.asarray(linear_row_windows(
        _window_views(values[::-1], 0, order), order, "left",
        _window_views(widths[::-1], 0, order)))[::-1]
    np.testing.assert_allclose(right, mirrored, rtol=0.0, atol=1e-15)


# ================================================================
#  7. Differentiability (AGENTS.md policy)
# ================================================================
@pytest.mark.parametrize("boundary", ["none", "graded"])
def test_grad_through_a_stretched_reconstruction(boundary):
    n = 8
    mesh = (periodic_mesh(n) if boundary == "none"
            else bounded_mesh(n))
    grid = Grid((mesh,))
    grid.negotiate(halo=HaloSpec({"z": 3}))
    op = WenoReconstruction(5, boundary=boundary)["z"]
    rng = np.random.default_rng(41)
    values = jnp.asarray(rng.normal(size=n))

    def loss(data):
        f = grid.create_field(mesh.cell_avg, data=data)
        return jnp.sum(op(f).data ** 2)

    grad = np.asarray(jax.grad(loss)(values))
    assert np.all(np.isfinite(grad))

    eps = 1e-6
    for k in (0, n // 2, n - 1):
        bump = np.zeros(n)
        bump[k] = eps
        plus = float(loss(values + jnp.asarray(bump)))
        minus = float(loss(values - jnp.asarray(bump)))
        fd = (plus - minus) / (2.0 * eps)
        assert abs(grad[k] - fd) <= 1e-4 * max(abs(fd), 1.0)


# ================================================================
#  8. Multi-device: a sharded stretched periodic axis
# ================================================================
def _reconstruction_on_a_sharded_stretched_axis(device_ids):
    # 16 cells over 4 forced devices => 4 cells/shard; WENO-5 halo 3
    # fits (cells >= width + 1 = 4), so the mapped axis is genuinely
    # distributed and the width co-operand is sharded with the data
    mesh = periodic_mesh(16)
    grid = Grid((mesh,), device_ids=device_ids)
    grid.negotiate(halo=HaloSpec({"z": 3}))
    f = grid.create_field(
        mesh.cell_avg,
        data=jnp.asarray(cell_averages(wavy_map, 16)))
    return grid, WenoReconstruction(5, bias="left")["z"](f)


@pytest.mark.multi_device
def test_stretched_reconstruction_is_device_count_invariant(
        forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid_many, g_many = _reconstruction_on_a_sharded_stretched_axis(
        None)
    _, g_one = _reconstruction_on_a_sharded_stretched_axis((0,))
    if forced_devices and forced_devices > 1:
        assert not grid_many.decomposition.default_layout.is_local("z")
    assert bool(jnp.all(jnp.isfinite(g_many.data)))
    gathered = np.asarray(grid_many.decomposition.gather(
        g_many._data, g_many.function_space))
    np.testing.assert_allclose(gathered, np.asarray(g_one.data),
                               rtol=0.0, atol=1e-12)


def _graded_on_a_sharded_stretched_column(device_ids):
    # 16 cells over 4 forced devices => 4 cells/shard, WENO-5 halo 3
    # fits; the two wall shards patch their K reduced faces from the
    # width CO-ARRAY handed through ``patch_physical_ends``
    mesh = bounded_mesh(16)
    grid = Grid((mesh,), device_ids=device_ids)
    grid.negotiate(halo=HaloSpec({"z": 3}))
    f = grid.create_field(
        mesh.cell_avg,
        data=jnp.asarray(cell_averages(tanh_map, 16)))
    return grid, graded_reconstruction(5, "left")["z"](f)


@pytest.mark.multi_device
def test_stretched_graded_column_is_device_count_invariant(
        forced_devices):
    # the decisive gate for the co-array seam: with the widths merely
    # CLOSED OVER, the wall rungs of every shard but the first would
    # read the global width array with block-local indices
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    grid_many, g_many = _graded_on_a_sharded_stretched_column(None)
    _, g_one = _graded_on_a_sharded_stretched_column((0,))
    if forced_devices and forced_devices > 1:
        assert not grid_many.decomposition.default_layout.is_local("z")
    assert bool(jnp.all(jnp.isfinite(g_many.data)))
    gathered = np.asarray(grid_many.decomposition.gather(
        g_many._data, g_many.function_space))
    np.testing.assert_allclose(gathered, np.asarray(g_one.data),
                               rtol=0.0, atol=1e-12)


@pytest.mark.multi_device
def test_bounded_dual_wall_cells_survive_sharding(forced_devices):
    # ``cell_widths`` writes the two wall half cells through the same
    # physical-end seam, so they must land on the boundary shards
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    values = []
    for device_ids in (None, (0,)):
        mesh = bounded_mesh(16)
        grid = Grid((mesh,), device_ids=device_ids)
        grid.negotiate(halo=HaloSpec({"z": 3}))
        f = grid.create_field(mesh.inner)
        values.append(np.asarray(cell_widths(f, "z")).ravel())
    width = 3
    centers = centers_of(tanh_map, 16)
    for widths in values:
        # the wall half cells sit one slot outside the true DOFs of
        # the FIRST and LAST block
        assert widths[width - 1] == pytest.approx(centers[0], abs=1e-15)
        assert widths[-width] == pytest.approx(1.0 - centers[-1],
                                               abs=1e-15)
