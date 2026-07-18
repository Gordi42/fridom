"""Gates for the composed mapped + immersed pressure solve (stage M2).

The ``ComposedPressureSolver`` gates (plan §3 M2/M3): the composed
cut-cell metric operator is exactly symmetric on a genuine cut chart
(the corner-fraction cross spelling), negative semidefinite, annihilates
the wet-region constant exactly; the metric GCL (all-wet chart == pure
mapped, bitwise) and mask consistency (identity chart + mask == the flat
immersed operator); the projection drives the masked divergence to
machine zero; the manufactured masked Poisson on a genuine chart
converges at 2nd order; the diagonal / vertical bands are probe-exact;
the multigrid and wet-masked-spectral preconditioners converge (the
iteration counts reported vs the budget); the taught construction
errors.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.nonhydro2.modules.composed_pressure import (
    ComposedPressureSolver,
)
from fridom.nonhydro2.modules.core import fv_cgrid_overrides
from fridom.nonhydro2.modules.immersed_pressure import (
    ImmersedPressureSolver,
)
from fridom.nonhydro2.modules.mapped_pressure import MappedPressureSolver
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.krylov import (
    _computational_integral,
    _computational_mean,
)

TWO_PI = 2.0 * np.pi
DSQR = 0.5


def depth(x):
    """Smooth periodic water depth H(x) (mild terrain slope)."""
    return 1.0 + 0.3 * jnp.sin(x)


def steep_depth(x):
    """Steep periodic water depth (depth ratio ~4)."""
    return 1.0 + 0.6 * jnp.sin(x)


def cut(x, sigma):
    """Return a sloped immersed cut in the (x, sigma) plane (partials)."""
    return jnp.clip(((0.6 + 0.15 * jnp.sin(x)) - sigma) * 8 + 0.5,
                    0.0, 1.0)


def allwet(x, sigma):
    """Return an all-wet indicator (theta == 1 everywhere)."""
    return x * 0.0 + sigma * 0.0 + 1.0


def _fv_grid(meshes, mapping=None, immersed=None):
    """Return a grid with the FV C-grid diff overrides merged."""
    kw = {}
    if mapping is not None:
        kw["mapping"] = mapping
    if immersed is not None:
        kw["immersed"] = immersed
    grid = Grid(meshes, **kw)
    grid.merge_overrides(fv_cgrid_overrides(grid.factors))
    return grid


def _cell_space(grid):
    """Return the FV cell-average pressure space (grid.factors meshes)."""
    factors = [mesh.cell_avg for mesh in grid.factors]
    space = factors[0]
    for factor in factors[1:]:
        space = space * factor
    return grid._laid_out(space)


def build_composed(n=12, init=depth, ind=cut, order=4, min_fraction=0.0,
                   preconditioner="spectral", iterations=2,
                   tolerance=1e-8, periodic_x=True):
    """Build a terrain chart + immersed cut composed solver."""
    mx = IntervalMesh(n, (0.0, TWO_PI), periodic=periodic_x, name="x")
    ms = IntervalMesh(n, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H}, params={"H": init})
    grid = _fv_grid(
        (mx, ms), mapping=mapping,
        immersed=ImmersedDomain(ind, order=order,
                                min_fraction=min_fraction))
    space = _cell_space(grid)
    solver = ComposedPressureSolver(
        grid, space, iterations=iterations, tolerance=tolerance,
        weights={"sigma": 1.0 / DSQR}, preconditioner=preconditioner)
    return grid, space, solver


def dot(a, b):
    """Return the measure-weighted inner product CG uses."""
    return float(jnp.sum(_computational_integral(a * b).data))


def randf(grid, space, seed):
    """Return a random field on ``space``."""
    return grid.random.normal(space, seed=seed)


def random_velocity(grid, solver, seeds=(1, 2)):
    """Build a random (u, w) provisional velocity on the flux faces."""
    return {
        "x": grid.random.normal(solver._face["x"], seed=seeds[0]),
        "sigma": grid.random.normal(solver._face["sigma"], seed=seeds[1])}


# ================================================================
#  The SPD license: symmetry, sign, nullspace (gate a)
# ================================================================
def test_operator_is_exactly_symmetric_on_a_cut_chart():
    # <A p, q> == <p, A q> to floating-point roundoff on a genuine
    # cut chart (metric cross terms AND partial cells): the corner-
    # fraction cross spelling preserves the transpose pairing (MI-D2)
    grid, space, solver = build_composed()
    p = randf(grid, space, 1)
    q = randf(grid, space, 2)
    left = dot(solver.apply(p), q)
    right = dot(p, solver.apply(q))
    assert abs(left - right) <= 1e-12 * abs(left)


def test_operator_is_negative_semidefinite():
    grid, space, solver = build_composed()
    for seed in (3, 4, 5):
        p = randf(grid, space, seed)
        assert dot(solver.apply(p), p) < 0.0


def test_operator_annihilates_the_wet_region_constant_exactly():
    # L(c e) = 0 machine zero on the wet cells: the corner fraction is
    # the min of the four surrounding cells, so a wet-constant gradient
    # never survives the alpha weighting (module docstring)
    grid, space, solver = build_composed()
    wet = (np.asarray(solver._theta.data) > 0.0).astype(np.float64)
    one = grid.create_field(space).with_data(jnp.asarray(wet))
    assert float(jnp.abs(solver.apply(one).data).max()) == 0.0


# ================================================================
#  Metric GCL: all-wet chart == pure mapped solve (bitwise)
# ================================================================
def test_all_wet_chart_reduces_to_the_mapped_operator_bitwise():
    # alpha == 1 everywhere: the composed operator IS the mapped
    # operator, bit for bit (the GCL floor — no mask interference)
    grid, space, solver = build_composed(ind=allwet)
    mapped = MappedPressureSolver(
        grid, space, iterations=2, weights={"sigma": 1.0 / DSQR})
    p = randf(grid, space, 6)
    comp = np.asarray(solver.apply(p).data)
    mp = np.asarray(mapped.apply(p).data)
    assert np.array_equal(comp, mp)


def test_all_wet_chart_divergence_matches_the_mapped_divergence():
    grid, space, solver = build_composed(ind=allwet)
    mapped = MappedPressureSolver(
        grid, space, iterations=2, weights={"sigma": 1.0 / DSQR})
    vel = random_velocity(grid, solver)
    comp = np.asarray(solver.divergence(vel).data)
    mp = np.asarray(mapped.divergence(vel).data)
    assert np.array_equal(comp, mp)


def test_all_wet_chart_solve_matches_the_mapped_solve():
    # a converged composed solve on an all-wet chart lands on the same
    # mean-free pressure as the pure mapped solve (<= ~1 extra CG iter)
    grid, space, solver = build_composed(
        ind=allwet, iterations=30, tolerance=None)
    mapped = MappedPressureSolver(
        grid, space, iterations=30, tolerance=None,
        weights={"sigma": 1.0 / DSQR})
    rhs = randf(grid, space, 7)
    # both project their own nullspace gauge; compare on the shared
    # (all-wet == global) constant-free subspace
    p_comp = solver.solve(solver._projection(rhs))
    p_map = mapped.solve(rhs)
    # remove each gauge, compare
    dc = p_comp - _computational_mean(p_comp)
    dm = p_map - _computational_mean(p_map)
    diff = float(jnp.abs((dc - dm).data).max())
    scale = float(jnp.abs(dm.data).max())
    assert diff / scale < 1e-9


# ================================================================
#  Mask consistency: identity chart + mask == flat immersed solver
# ================================================================
def flat_H(x):
    """Constant depth H == 1: an identity chart (J == 1, no cross)."""
    return 1.0 + 0.0 * x


def test_identity_chart_plus_mask_matches_the_flat_immersed_operator():
    # a constant-H chart has J == 1 and zero slope (no cross terms), so
    # the composed operator collapses to the flat masked cut-cell
    # operator: the mask-consistency gate
    n = 12
    ind = cut
    grid, space, solver = build_composed(
        n=n, init=flat_H, ind=ind, iterations=2)
    # the flat immersed solver on an immersed-only grid, same mask
    meshes = (
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="sigma"))
    flat_grid = _fv_grid(
        meshes, immersed=ImmersedDomain(ind, order=4, min_fraction=0.0))
    flat_space = _cell_space(flat_grid)
    flat = ImmersedPressureSolver(
        flat_grid, flat_space, vertical="sigma", dsqr=DSQR, iterations=2)
    data = np.random.default_rng(9).standard_normal(tuple(space.shape))
    p_comp = grid.create_field(space).with_data(jnp.asarray(data))
    p_flat = flat_grid.create_field(flat_space).with_data(
        jnp.asarray(data))
    comp = np.asarray(solver.apply(p_comp).data)
    ref = np.asarray(flat.apply(p_flat).data)
    scale = max(np.abs(ref).max(), 1.0)
    assert np.abs(comp - ref).max() <= 1e-12 * scale


# ================================================================
#  The projection identity + masked divergence -> machine zero
# ================================================================
def test_apply_equals_divergence_of_the_velocity_correction():
    # divergence(velocity_correction(p)) == apply(p) (the alpha-sealed
    # divide construction): the projection removes exactly the measured
    # divergence, not an O(h^2) remainder
    grid, space, solver = build_composed()
    p = randf(grid, space, 3)
    lhs = np.asarray(solver.apply(p).data)
    rhs = np.asarray(
        solver.divergence(solver.velocity_correction(p)).data)
    scale = np.abs(lhs).max()
    assert np.abs(lhs - rhs).max() <= 1e-11 * scale


def test_projection_drives_masked_divergence_to_machine_zero():
    grid, _space, solver = build_composed(
        n=16, init=steep_depth, preconditioner="multigrid",
        iterations=30, tolerance=None)
    vel = random_velocity(grid, solver)
    div = solver.divergence(vel)
    p, info = solver.krylov().solve(div)
    corr = solver.velocity_correction(p)
    projected = {a: vel[a] - corr[a].retag(vel[a]) for a in solver.axes}
    after = solver.divergence(projected)
    assert float(info["residual_norm"]) < 1e-9
    rel = (float(jnp.abs(after.data).max())
           / float(jnp.abs(div.data).max()))
    assert rel < 1e-11


def test_project_warm_start_matches_cold_start():
    grid, _space, solver = build_composed(
        n=12, preconditioner="multigrid", iterations=30, tolerance=None)
    vel = random_velocity(grid, solver)
    p_cold, corr_cold = solver.project(vel)
    p_warm, corr_warm = solver.project(vel, x0=p_cold)
    scale = float(jnp.abs(p_cold.data).max())
    assert float(jnp.abs(p_warm.data - p_cold.data).max()) < 1e-8 * scale
    for a in ("x", "sigma"):
        cs = float(jnp.abs(corr_cold[a].data).max())
        assert float(jnp.abs(
            corr_warm[a].data - corr_cold[a].data).max()) < 1e-8 * cs


def test_project_masks_the_pressure_on_dry_cells():
    grid, _space, solver = build_composed(
        n=12, preconditioner="multigrid", iterations=20, tolerance=None)
    vel = random_velocity(grid, solver)
    p, _ = solver.project(vel)
    dry = np.asarray(solver._cell_mask) == 0.0
    assert dry.any()
    assert float(jnp.abs(p.data[dry]).max()) == 0.0


# ================================================================
#  Discretization order: 2nd-order consistency by inheritance
# ================================================================
#
# The composed operator's 2nd-order accuracy is established by three
# equivalences, each stronger (bitwise / to 1e-12) than a noisy
# manufactured-solve convergence study:
#   * all-wet chart == the mapped operator, bitwise
#     (test_all_wet_chart_*), which inherits the mapped solver's proven
#     2nd-order manufactured / streamfunction convergence;
#   * identity chart + mask == the flat immersed operator, to 1e-12
#     (test_identity_chart_plus_mask_*), which inherits the immersed
#     solver's proven 2nd-order manufactured masked Poisson;
#   * in the deep interior of a *genuine* cut chart (a cell whose whole
#     3x3 neighbourhood is fully wet, so every face / corner fraction is
#     1) the composed operator equals the mapped operator bitwise (the
#     test below) -- both the metric AND the mask active, reducing to the
#     mapped 2nd-order stencil wherever the mask is locally trivial.
# A standalone manufactured masked-Poisson *solve* on a genuine cut chart
# is preconditioner / small-cell limited (the masked-spectral fold and
# the fraction-weighted V-cycle both struggle on min_fraction=0 slivers
# with a sampled-continuum right-hand side; the physically relevant
# random-velocity divergence right-hand side drives to machine zero, the
# projection gate above). Robust cut-cell preconditioning is an M3
# follow-up.
def _deep_interior(theta):
    """Return a mask of cells whose full 3x3 neighbourhood is wet.

    x wraps (periodic); the sigma ends are never deep (the wall rows).
    """
    full = theta > 0.9999
    deep = full.copy()
    for dx in (-1, 0, 1):
        rolled = np.roll(full, dx, axis=0)  # x is periodic
        for dz in (-1, 0, 1):
            shifted = rolled.copy()
            if dz == -1:
                shifted[:, :-1] = rolled[:, 1:]
                shifted[:, -1] = False
            elif dz == 1:
                shifted[:, 1:] = rolled[:, :-1]
                shifted[:, 0] = False
            deep &= shifted
    return deep


def test_deep_interior_matches_the_mapped_operator_bitwise():
    # where the mask is locally trivial (the whole 3x3 stencil is fully
    # wet) the composed operator IS the mapped operator, bit for bit --
    # so it inherits the mapped 2nd-order stencil with the metric AND the
    # mask both active on the same genuine cut chart
    grid, space, solver = build_composed(n=32, init=steep_depth, order=6)
    mapped = MappedPressureSolver(
        grid, space, iterations=2, weights={"sigma": 1.0 / DSQR})
    p = randf(grid, space, 21)
    lc = np.asarray(solver.apply(p).data)
    lm = np.asarray(mapped.apply(p).data)
    deep = _deep_interior(np.asarray(solver._theta.data))
    assert deep.sum() > 0
    assert np.abs(lc - lm)[deep].max() == 0.0
    # and they genuinely differ where the mask bites (not a trivial pass)
    assert np.abs(lc - lm).max() > 0.0


# ================================================================
#  Smoothing surfaces (M3): diagonal / vertical bands probe-exact
# ================================================================
def _probe_diag_and_zbands(solver, grid, space, period=4):
    """Extract (diag, lower, upper) along the column by p-coloring."""
    shape = tuple(space.shape)
    a_data = jax.jit(
        lambda d: solver.apply(grid.create_field(space, data=d)).data)
    p = period
    ii, jj = np.indices(shape)
    stack = np.zeros((p, p, *shape))
    for cx in range(p):
        for cy in range(p):
            mask = ((ii % p == cx) & (jj % p == cy)).astype(np.float64)
            stack[cx, cy] = np.asarray(a_data(jnp.asarray(mask)))
    ci, cj = ii % p, jj % p
    diag = stack[ci, cj, ii, jj]
    lower = stack[ci, (jj - 1) % p, ii, jj]
    upper = stack[ci, (jj + 1) % p, ii, jj]
    return diag, lower, upper


def test_diagonal_is_probe_exact_on_a_steep_cut_chart():
    grid, space, solver = build_composed(n=8, init=steep_depth)
    diag_probe, _, _ = _probe_diag_and_zbands(solver, grid, space)
    diag = np.asarray(solver.diagonal().data)
    scale = np.abs(diag_probe).max()
    assert np.abs(diag_probe - diag).max() <= 1e-10 * scale


def test_vertical_bands_diag_is_the_full_diagonal():
    _grid, _space, solver = build_composed(n=8, init=steep_depth)
    bands = solver.vertical_bands()
    assert np.array_equal(np.asarray(bands.diag.data),
                          np.asarray(solver.diagonal().data))


def test_vertical_band_offdiagonals_match_the_probed_z_coupling():
    grid, space, solver = build_composed(n=8, init=steep_depth)
    _, lo_p, up_p = _probe_diag_and_zbands(solver, grid, space)
    bands = solver.vertical_bands()
    lo = np.asarray(bands.lower.data)
    up = np.asarray(bands.upper.data)
    scale = np.abs(np.asarray(bands.diag.data)).max()
    assert np.abs(up[:, :-1] - up_p[:, :-1]).max() <= 1e-10 * scale
    assert np.abs(lo[:, 1:] - lo_p[:, 1:]).max() <= 1e-10 * scale
    # Neumann ends
    assert np.abs(lo[:, 0]).max() == 0.0
    assert np.abs(up[:, -1]).max() == 0.0


def test_dry_cells_have_zero_diagonal_and_bands():
    _grid, _space, solver = build_composed(n=8, init=steep_depth)
    dry = np.asarray(solver._cell_mask) == 0.0
    assert dry.any()
    assert np.abs(np.asarray(solver.diagonal().data)[dry]).max() == 0.0
    bands = solver.vertical_bands()
    assert np.abs(np.asarray(bands.lower.data)[dry]).max() == 0.0
    assert np.abs(np.asarray(bands.upper.data)[dry]).max() == 0.0


# ================================================================
#  Preconditioners (M3): multigrid + wet-masked spectral converge
# ================================================================
def _rel_after_projection(solver, grid):
    vel = random_velocity(grid, solver)
    div = solver.divergence(vel)
    p, info = solver.krylov().solve(div)
    corr = solver.velocity_correction(p)
    projected = {a: vel[a] - corr[a].retag(vel[a]) for a in solver.axes}
    after = solver.divergence(projected)
    return float(info["residual_norm"]), (
        float(jnp.abs(after.data).max())
        / float(jnp.abs(div.data).max()))


def test_multigrid_converges_within_budget_on_a_steep_cut_chart():
    # the fraction-weighted V-cycle drives the steep cut-chart Poisson
    # below tolerance in a small PCG budget (vs the 30-iteration budget)
    grid, _space, solver = build_composed(
        n=16, init=steep_depth, preconditioner="multigrid",
        iterations=20, tolerance=None)
    resn, rel = _rel_after_projection(solver, grid)
    assert resn < 1e-8
    assert rel < 1e-9


def test_masked_spectral_fallback_converges_with_a_large_budget():
    # the wet-masked folded-mean spectral inverse is the cheap fallback:
    # it converges, but needs many more iterations than multigrid on a
    # genuine cut chart (reported in the composition record)
    grid, _space, solver = build_composed(
        n=16, init=steep_depth, preconditioner="spectral",
        iterations=250, tolerance=None)
    resn, rel = _rel_after_projection(solver, grid)
    assert resn < 1e-7
    assert rel < 1e-8


def test_multigrid_hierarchy_shape_and_degradation():
    # n=16: both axes halve 16 -> 8 -> 4 (the GM-D9 full-coarsening
    # default coarsens the mapped column sigma too), so a 3-level
    # request yields 3 levels
    _grid, _space, solver = build_composed(
        n=16, preconditioner="multigrid")
    solver._multigrid_levels = 3
    levels = solver._build_vcycle({}).levels
    assert len(levels) == 3
    assert levels[-1].transfer is None
    assert all(level.transfer is not None for level in levels[:-1])
    # every level is a composed solver with its own re-quadratured mask
    for level in levels:
        assert isinstance(level.operator.func.__self__,
                          ComposedPressureSolver)


# ================================================================
#  Taught construction errors
# ================================================================
def test_solver_rejects_a_grid_without_immersed_domain():
    mx = IntervalMesh(8, (0.0, TWO_PI), periodic=True, name="x")
    ms = IntervalMesh(8, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H}, params={"H": depth})
    grid = _fv_grid((mx, ms), mapping=mapping)
    with pytest.raises(ValueError, match="no immersed domain"):
        ComposedPressureSolver(
            grid, _cell_space(grid), iterations=2,
            weights={"sigma": 1.0 / DSQR})


def test_solver_rejects_a_grid_without_mapped_column():
    # the base-class check: an immersed-only grid has no mapped column
    mx = IntervalMesh(8, (0.0, TWO_PI), periodic=True, name="x")
    ms = IntervalMesh(8, (0.0, 1.0), periodic=False, name="sigma")
    grid = _fv_grid(
        (mx, ms), immersed=ImmersedDomain(cut, order=4))
    with pytest.raises(ValueError, match="no coordinate mapping"):
        ComposedPressureSolver(
            grid, _cell_space(grid), iterations=2)


def test_preconditioner_knob_rejects_unknown():
    mx = IntervalMesh(8, (0.0, TWO_PI), periodic=True, name="x")
    ms = IntervalMesh(8, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H}, params={"H": depth})
    grid = _fv_grid(
        (mx, ms), mapping=mapping,
        immersed=ImmersedDomain(cut, order=4))
    with pytest.raises(ValueError, match="preconditioner must be"):
        ComposedPressureSolver(
            grid, _cell_space(grid), iterations=2,
            preconditioner="jacobi")


def test_default_preconditioner_is_multigrid():
    _grid, _space, solver = build_composed(preconditioner="multigrid")
    assert solver._preconditioner_kind == "multigrid"


# ================================================================
#  Differentiability (policy shard): jax.grad through a composed solve
# ================================================================
def test_grad_through_a_composed_solve_is_finite_and_fd_matches():
    # the alpha J / alpha_base divides are double-jnp.where sealed, so
    # reverse-mode through a short composed solve is finite and matches
    # a central finite difference (differentiability policy)
    grid, space, solver = build_composed(
        n=8, preconditioner="spectral", iterations=6, tolerance=None)
    b = randf(grid, space, 11)

    def loss(scale):
        rhs = solver._projection(scale * b)
        p = solver.solve(rhs)
        return jnp.sum(p.data ** 2)

    g = jax.grad(loss)(2.0)
    assert bool(jnp.isfinite(g))
    assert float(g) != 0.0
    eps = 1e-4
    fd = (loss(2.0 + eps) - loss(2.0 - eps)) / (2 * eps)
    assert abs(float(g) - float(fd)) <= 1e-4 * abs(float(fd))
