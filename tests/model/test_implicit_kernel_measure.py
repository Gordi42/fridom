"""Measure-aware column oracles for the VerticalDiffusion kernel.

The prefix-mirrored shard of ``test_implicit_kernel`` (AGENTS.md
oversized-module rule) covering the measure-aware band builder
``_diffusion_bands``: the conservative face-averaged flux form on a
stretched column (constant-flux exactness, width-weighted conservation,
second-order convergence, the Dirichlet wall corner), the terrain-
following column (along-sigma physical widths ``J * dsigma``, the
flat-chart equivalence to an unmapped stretched column, a combined
stretched-sigma + terrain grid), and the variable (field-valued) kappa
(dense reference, the kappa-summed merge, the wrong-space rejection).
Self-contained: the small builders are duplicated, not imported.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.model.implicit import VerticalDiffusion, _diffusion_bands
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh

N = 12
KAPPA = 0.05


# ================================================================
#  Builders
# ================================================================
def kappa_const(value):
    def kappa(_module, _state, _ctx, _name):
        return value
    return kappa


def stretched_grid(n=N, stretch=lambda s: s ** 2):
    """Return a single-factor bounded z column, mapped (stretched)."""
    mz = MappedIntervalMesh(n, (0.0, 1.0), stretch,
                            periodic=False, name="z")
    return Grid((mz,)), mz


def terrain_grid(nx=4, nz=N, sigma_stretch=None):
    """Return a terrain-following (x, sigma) grid: zp = sigma * H(x)."""
    mx = IntervalMesh(nx, (0.0, 2 * np.pi), periodic=True, name="x")
    if sigma_stretch is None:
        ms = IntervalMesh(nz, (0.0, 1.0), periodic=False, name="sigma")
    else:
        ms = MappedIntervalMesh(nz, (0.0, 1.0), sigma_stretch,
                                periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, height: sigma * height},
        params={"height": lambda x: 1.0 + 0.2 * np.sin(x)})
    return Grid((mx, ms), mapping=mapping), mx, ms


def apply_op(grid, space, data, kappa=KAPPA, bc=("neumann", "neumann"),
             axis="z"):
    op = VerticalDiffusion(axis, ("b",), kappa_const(kappa), bc=bc)
    state = VectorField(
        {"b": grid.create_field(space, data=jnp.asarray(data))})
    return np.asarray(op.apply(None, state, None)["b"].data)


def solve_op(grid, space, data, dt_gamma, kappa=KAPPA,
             bc=("neumann", "neumann"), axis="z"):
    op = VerticalDiffusion(axis, ("b",), kappa_const(kappa), bc=bc)
    rhs = {"b": grid.create_field(space, data=jnp.asarray(data))}
    return np.asarray(op.solve(None, rhs, dt_gamma, None)["b"].data)


# ================================================================
#  1. Exactness: constant flux (linear in physical z) has zero
#     interior divergence on a stretched column
# ================================================================
def test_stretched_constant_flux_has_zero_interior_divergence():
    grid, mz = stretched_grid()
    zc = np.asarray(grid.evaluation_nodes(mz.center, "z").data).reshape(-1)
    slope = 3.0
    q = slope * zc + 0.7  # constant physical flux kappa * slope
    result = apply_op(grid, mz.center, q)
    # interior rows vanish (constant flux => zero divergence)
    assert np.max(np.abs(result[1:-1])) < 1e-12
    # the Neumann wall rows carry the flux mismatch kappa*slope/dz_wall
    mcell = np.asarray(grid.measure(mz.center, "z").data).reshape(-1)
    assert result[0] == pytest.approx(KAPPA * slope / mcell[0], abs=1e-12)
    assert result[-1] == pytest.approx(
        -KAPPA * slope / mcell[-1], abs=1e-12)


# ================================================================
#  2. Conservation: the physical-width-weighted column sum is
#     invariant under apply and under a solve step (Neumann)
# ================================================================
def _physical_widths(grid, space, axis):
    widths = np.asarray(grid.measure(space, axis).data)
    corrections = grid.mapping.column_corrections if getattr(
        grid, "mapping", None) is not None else {}
    if axis in corrections:
        mapped, base = corrections[axis]
        widths = widths * np.asarray(
            grid.metric(space, f"d{mapped}_d{base}", params=None).data)
    return widths


def test_stretched_neumann_conserves_the_width_weighted_sum():
    grid, mz = stretched_grid()
    rng = np.random.default_rng(0)
    q = rng.standard_normal(N)
    result = apply_op(grid, mz.center, q)
    mcell = np.asarray(grid.measure(mz.center, "z").data).reshape(-1)
    assert abs(np.sum(mcell * result)) < 1e-13
    # the solve preserves the same weighted sum through a dt_gamma step
    solved = solve_op(grid, mz.center, q, 0.3)
    assert np.sum(mcell * solved) == pytest.approx(
        np.sum(mcell * q), abs=1e-12)


def test_terrain_neumann_conserves_the_jacobian_weighted_sum():
    grid, mx, ms = terrain_grid()
    rng = np.random.default_rng(1)
    data = rng.standard_normal((4, N))
    space = mx.center * ms.center
    result = apply_op(grid, space, data, axis="sigma")
    phys = _physical_widths(grid, space, "sigma")
    assert np.max(np.abs(np.sum(phys * result, axis=1))) < 1e-13
    solved = solve_op(grid, space, data, 0.2, axis="sigma")
    assert np.allclose(np.sum(phys * solved, axis=1),
                       np.sum(phys * data, axis=1), atol=1e-12)


# ================================================================
#  3. Convergence: second order in the interior on a smooth stretch
# ================================================================
def test_second_order_convergence_on_a_smooth_stretch():
    # manufactured q = cos(pi z) (q_z = 0 at both walls => Neumann-
    # compatible); L_h q -> (kappa q_z)_z = -kappa pi^2 cos(pi z). The
    # face-averaged conservative band is 2nd order in the interior on a
    # smoothly stretched column (the wall rows are 1st order, excluded)
    def smooth(s):
        return s + 0.2 * s * (1.0 - s)

    errors = []
    for n in (32, 64, 128):
        mz = MappedIntervalMesh(n, (0.0, 1.0), smooth,
                                periodic=False, name="z")
        grid = Grid((mz,))
        zc = np.asarray(
            grid.evaluation_nodes(mz.center, "z").data).reshape(-1)
        q = np.cos(np.pi * zc)
        result = apply_op(grid, mz.center, q)
        exact = -KAPPA * np.pi ** 2 * np.cos(np.pi * zc)
        errors.append(np.max(np.abs(result[1:-1] - exact[1:-1])))
    # each doubling drops the error by >= 3.5x (order ~2)
    assert errors[0] / errors[1] > 3.5
    assert errors[1] / errors[2] > 3.5


# ================================================================
#  4. Dirichlet wall corner: hand-built reference + decaying solve
# ================================================================
def test_dirichlet_corner_matches_a_hand_built_reference():
    grid, mz = stretched_grid()
    field = grid.create_field(mz.center, data=jnp.zeros(N))
    _lower, diag, _upper, _axis = _diffusion_bands(
        field, "z", KAPPA, ("dirichlet", "dirichlet"))
    diag = np.asarray(diag).reshape(-1)
    mcell = np.asarray(grid.measure(mz.center, "z").data).reshape(-1)
    mface = np.asarray(
        grid.measure(mz.outer, "z").data).reshape(-1)
    # bottom Dirichlet corner: -(up_coupling + wall_coupling), the wall
    # distance is the clipped bottom half-cell mface[0]
    up0 = KAPPA / (mface[1] * mcell[0])
    wall0 = KAPPA / (mface[0] * mcell[0])
    assert diag[0] == pytest.approx(-(up0 + wall0), rel=1e-12)
    # top Dirichlet corner uses the clipped top half-cell mface[-1]
    lown = KAPPA / (mface[-2] * mcell[-1])
    walln = KAPPA / (mface[-1] * mcell[-1])
    assert diag[-1] == pytest.approx(-(lown + walln), rel=1e-12)


def test_dirichlet_solve_decays_a_constant_profile():
    grid, mz = stretched_grid()
    const = np.full(N, 2.5)
    solved = solve_op(grid, mz.center, const, 5.0,
                      bc=("dirichlet", "dirichlet"))
    # the Dirichlet band has no constant nullspace: the walls pull the
    # profile below its uniform value
    assert np.all(np.isfinite(solved))
    assert np.max(np.abs(solved)) < 2.5


# ================================================================
#  5. Flat-chart equivalence: terrain (H const) == unmapped uniform
# ================================================================
def test_flat_chart_terrain_matches_the_unmapped_column():
    depth = 2.0
    mx = IntervalMesh(3, (0.0, 2 * np.pi), periodic=True, name="x")
    ms = IntervalMesh(N, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, height: sigma * height},
        params={"height": lambda x: depth + 0.0 * x})
    grid_t = Grid((mx, ms), mapping=mapping)
    space_t = mx.center * ms.center
    field_t = grid_t.create_field(space_t, data=jnp.zeros((3, N)))
    lo_t, dg_t, up_t, _ = _diffusion_bands(
        field_t, "sigma", KAPPA, ("dirichlet", "neumann"))

    mzu = IntervalMesh(N, (0.0, depth), periodic=False, name="z")
    grid_u = Grid((mzu,))
    field_u = grid_u.create_field(mzu.center, data=jnp.zeros(N))
    lo_u, dg_u, up_u, _ = _diffusion_bands(
        field_u, "z", KAPPA, ("dirichlet", "neumann"))

    for band_t, band_u in ((lo_t, lo_u), (dg_t, dg_u), (up_t, up_u)):
        # every x-column of the flat terrain band equals the plain column
        assert np.allclose(
            np.asarray(band_t), np.asarray(band_u).reshape(-1),
            atol=1e-13)
    # one solve agrees too
    rng = np.random.default_rng(2)
    col = rng.standard_normal(N)
    solved_u = solve_op(grid_u, mzu.center, col, 0.3,
                        bc=("dirichlet", "neumann"))
    solved_t = solve_op(grid_t, space_t,
                        np.broadcast_to(col, (3, N)), 0.3,
                        bc=("dirichlet", "neumann"), axis="sigma")
    assert np.allclose(solved_t[0], solved_u, atol=1e-10)


# ================================================================
#  6. Terrain: per-column bands match a physical-width reference
# ================================================================
def test_terrain_bands_match_a_per_column_physical_reference():
    grid, mx, ms = terrain_grid()
    space = mx.center * ms.center
    field = grid.create_field(space, data=jnp.zeros((4, N)))
    lower, diag, upper, axis_index = _diffusion_bands(
        field, "sigma", KAPPA, ("neumann", "neumann"))
    assert axis_index == 1
    mcell_s = np.asarray(grid.measure(space, "sigma").data).reshape(-1)
    mface_s = np.asarray(
        grid.measure(space.replace(sigma=ms.outer), "sigma").data
    ).reshape(-1)
    height = np.asarray(
        grid.metric(space, "dzp_dsigma", params=None).data)[:, 0]
    for j in range(4):
        m_cell = height[j] * mcell_s
        m_face = height[j] * mface_s
        up_coupling = KAPPA / (m_face[1:] * m_cell)   # face c+1/2
        low_coupling = KAPPA / (m_face[:-1] * m_cell)  # face c-1/2
        # Neumann drops the wall couplings from the diagonal; the
        # sub/super ends are unused (zeroed)
        ref_diag = -(low_coupling + up_coupling)
        ref_diag[0] += low_coupling[0]
        ref_diag[-1] += up_coupling[-1]
        ref_upper = up_coupling.copy()
        ref_upper[-1] = 0.0
        ref_lower = low_coupling.copy()
        ref_lower[0] = 0.0
        assert np.allclose(np.asarray(upper)[j], ref_upper, atol=1e-13)
        assert np.allclose(np.asarray(lower)[j], ref_lower, atol=1e-13)
        assert np.allclose(np.asarray(diag)[j], ref_diag, atol=1e-13)


def test_combined_stretched_sigma_and_terrain_is_finite():
    grid, mx, ms = terrain_grid(sigma_stretch=lambda s: s ** 1.3)
    space = mx.center * ms.center
    rng = np.random.default_rng(3)
    data = rng.standard_normal((4, N))
    applied = apply_op(grid, space, data, axis="sigma")
    solved = solve_op(grid, space, data, 0.1, axis="sigma")
    assert np.all(np.isfinite(applied))
    assert np.all(np.isfinite(solved))
    # conservation still holds on the combined grid
    phys = _physical_widths(grid, space, "sigma")
    assert np.max(np.abs(np.sum(phys * applied, axis=1))) < 1e-12


# ================================================================
#  7. Variable (field-valued) kappa
# ================================================================
def _dense_variable_kappa(kcell, dz):
    """Dense Neumann band of the face-averaged variable-kappa column."""
    n = kcell.shape[0]
    kface = 0.5 * (kcell[:-1] + kcell[1:])  # interior faces
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:-1] = kface / dz ** 2
    lower[1:] = kface / dz ** 2
    diag = -(upper + lower)  # Neumann: wall couplings dropped
    return (np.diag(diag) + np.diag(upper[:-1], 1)
            + np.diag(lower[1:], -1))


def test_field_valued_kappa_matches_a_dense_reference():
    mz = IntervalMesh(N, (0.0, 1.0), periodic=False, name="z")
    grid = Grid((mz,))
    rng = np.random.default_rng(4)
    kcell = 0.02 + 0.1 * rng.uniform(size=N)  # smoothly varying kappa
    kappa_field = grid.create_field(mz.center, data=jnp.asarray(kcell))

    def kappa(_module, _state, _ctx, _name):
        return kappa_field

    op = VerticalDiffusion("z", ("b",), kappa)
    q = np.cos((np.arange(N) + 0.5) * 3 * np.pi / N)
    state = VectorField({"b": grid.create_field(mz.center,
                                                data=jnp.asarray(q))})
    result = np.asarray(op.apply(None, state, None)["b"].data)
    reference = _dense_variable_kappa(kcell, 1.0 / N) @ q
    assert np.allclose(result, reference, atol=1e-12)


def test_kappa_summed_merge_with_a_field_and_a_scalar():
    mz = IntervalMesh(N, (0.0, 1.0), periodic=False, name="z")
    grid = Grid((mz,))
    rng = np.random.default_rng(5)
    kcell = 0.02 + 0.1 * rng.uniform(size=N)
    kappa_field = grid.create_field(mz.center, data=jnp.asarray(kcell))

    op_field = VerticalDiffusion(
        "z", ("b",), lambda _m, _s, _c, _n: kappa_field)
    op_scalar = VerticalDiffusion("z", ("b",), kappa_const(0.03))
    merged = op_field.merged_with(op_scalar)  # kappa = field + 0.03
    reference = VerticalDiffusion(
        "z", ("b",),
        lambda _m, _s, _c, _n: grid.create_field(
            mz.center, data=jnp.asarray(kcell + 0.03)))

    rhs0 = np.linspace(-1.0, 1.0, N)
    rhs = {"b": grid.create_field(mz.center, data=jnp.asarray(rhs0))}
    merged_solved = np.asarray(merged.solve(None, rhs, 0.1, None)["b"].data)
    ref = {"b": grid.create_field(mz.center, data=jnp.asarray(rhs0))}
    ref_solved = np.asarray(reference.solve(None, ref, 0.1, None)["b"].data)
    assert np.allclose(merged_solved, ref_solved, atol=1e-12)


def test_field_valued_kappa_on_the_wrong_space_is_rejected():
    mz = IntervalMesh(N, (0.0, 1.0), periodic=False, name="z")
    grid = Grid((mz,))
    # kappa on the wall-including Outer space (N + 1 nodes), not the
    # solved field's own Center space
    wrong = grid.create_field(mz.outer, data=jnp.ones(N + 1))

    op = VerticalDiffusion("z", ("b",), lambda _m, _s, _c, _n: wrong)
    state = VectorField({"b": grid.create_field(mz.center,
                                                data=jnp.zeros(N))})
    with pytest.raises(ValueError, match="own function space"):
        op.apply(None, state, None)
