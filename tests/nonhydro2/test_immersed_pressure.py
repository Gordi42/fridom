"""Gates for the masked cut-cell pressure solve (stage I2, IP-D6).

The ``ImmersedPressureSolver`` gates: the cut-cell operator is exactly
symmetric (gate a), the projection drives the masked divergence to
machine zero (gate b), the manufactured masked Poisson converges at
2nd order under refinement (gate e), the all-wet preconditioner is the
exact inverse (~1 iteration), and the taught errors are pinned.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.nonhydro2.modules.core import fv_cgrid_overrides
from fridom.nonhydro2.modules.immersed_pressure import (
    ImmersedPressureSolver,
)
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.banded import _resolve_tridiagonal_method
from fridom.spatial.operators.multigrid import VerticalLineJacobi

TWO_PI = 2.0 * np.pi

#: whether the default jax backend is a GPU (gates the cusparse leg)
_ON_GPU = jax.default_backend() == "gpu"


def _fv_grid(meshes, immersed):
    """Return a grid with the FV diff overrides merged (as the model)."""
    grid = Grid(meshes, immersed=immersed)
    grid.merge_overrides(fv_cgrid_overrides(grid.factors))
    return grid


def _cell_space(grid):
    """Return the cell-average pressure space of ``grid``."""
    factors = [mesh.cell_avg for mesh in grid.factors]
    space = factors[0]
    for factor in factors[1:]:
        space = space * factor
    return grid._laid_out(space)


def _box_solver(n=12, iterations=30, dsqr=0.7, tolerance=1e-8):
    """Build a face-aligned {0, 1} immersed box in a periodic grid."""
    meshes = tuple(
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name=nm)
        for nm in ("x", "y", "z"))
    box = lambda x, y, z: (  # noqa: E731
        (x > 1.0) & (x < 5.0) & (y > 1.0) & (y < 5.0)
        & (z > 0.2) & (z < 0.8)).astype(float)
    grid = _fv_grid(meshes, ImmersedDomain(box))
    space = _cell_space(grid)
    solver = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=dsqr, iterations=iterations,
        tolerance=tolerance)
    return grid, space, solver


def _random_velocity(solver, seed=0):
    """Return a random provisional velocity on the solver's flux faces."""
    rng = np.random.default_rng(seed)
    return {
        a: solver._alpha[a].with_data(
            jnp.asarray(rng.standard_normal(
                solver._alpha[a].data.shape)))
        for a in solver.axes}


# ================================================================
#  Gate a: the cut-cell operator is exactly symmetric
# ================================================================
def test_operator_is_symmetric_on_a_masked_grid():
    # <q, L p>_V = <L q, p>_V to ~1e-13 on a genuine partial-cell grid
    n = 12
    meshes = (
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="z"))
    slope = lambda x, y, z: jnp.clip(  # noqa: E731
        ((0.6 + 0.15 * jnp.sin(y) + 0.1 * z) - x / TWO_PI) * 8 + 0.5,
        0.0, 1.0)
    grid = _fv_grid(meshes, ImmersedDomain(slope, order=4))
    space = _cell_space(grid)
    solver = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.5, iterations=2)
    rng = np.random.default_rng(1)
    shape = solver._theta.data.shape
    q = grid.create_field(
        space, init=lambda x, y, z: jnp.asarray(  # noqa: ARG005
            rng.standard_normal(shape)))
    p = grid.create_field(
        space, init=lambda x, y, z: jnp.asarray(  # noqa: ARG005
            rng.standard_normal(shape)))
    qlp = float(jnp.sum((q * solver.apply(p)).integrate().data))
    lqp = float(jnp.sum((solver.apply(q) * p).integrate().data))
    assert abs(qlp - lqp) <= 1e-13 * max(abs(qlp), 1.0)


# ================================================================
#  Gate b: post-projection masked divergence is machine zero
# ================================================================
def test_projection_drives_masked_divergence_to_machine_zero():
    # fixed-iteration mode: pinned for determinism
    _grid, _space, solver = _box_solver(n=12, iterations=30,
                                        tolerance=None)
    vel = _random_velocity(solver, seed=2)
    _, info = solver.solve_info(solver.divergence(vel))
    _p, corr = solver.project(vel)
    corrected = {
        a: vel[a] - corr[a].retag(vel[a]) for a in solver.axes}
    div = solver.divergence(corrected)
    assert float(info["residual_norm"]) < 1e-9
    assert float(jnp.abs(div.data).max()) < 1e-9


def test_project_warm_start_matches_cold_start():
    # Phase E (GE-1): warm-starting the masked projection from the
    # previous solved pressure lands on the same wet-mean-free pressure
    # and drives the masked divergence to the same machine-zero.
    _grid, _space, solver = _box_solver(n=12, iterations=30,
                                        tolerance=None)
    vel = _random_velocity(solver, seed=2)
    p_cold, _ = solver.project(vel)
    p_warm, corr = solver.project(vel, x0=p_cold)
    scale = float(jnp.abs(p_cold.data).max())
    assert float(jnp.abs(p_warm.data - p_cold.data).max()) < 1e-8 * scale
    corrected = {
        a: vel[a] - corr[a].retag(vel[a]) for a in solver.axes}
    div = solver.divergence(corrected)
    assert float(jnp.abs(div.data).max()) < 1e-9


# ================================================================
#  All-wet: the preconditioner is the exact inverse (~1 iteration)
# ================================================================
def test_all_wet_solver_converges_in_one_iteration():
    n = 10
    meshes = (
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="z"))
    allwet = lambda x, y, z: x * 0.0 + y * 0.0 + z * 0.0 + 1.0  # noqa: E731
    grid = _fv_grid(meshes, ImmersedDomain(allwet))
    space = _cell_space(grid)
    solver = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.7, iterations=2)
    vel = _random_velocity(solver, seed=3)
    _, info = solver.solve_info(solver.divergence(vel))
    # the masked preconditioner equals the operator on an all-wet
    # domain (D = I, the exact spectral inverse) — PCG converges in one
    # iteration to machine zero
    assert float(info["residual_norm"]) < 1e-12


# ================================================================
#  Gate e: manufactured masked Poisson, 2nd-order convergence
# ================================================================
def _b(y):
    return 0.28 + 0.12 * jnp.sin(y)


def _p_exact(y, z):
    return jnp.cos(np.pi * (z - _b(y)) / (1.0 - _b(y)))


def _lap(y, z):
    pyy = jax.grad(lambda yy: jax.grad(_p_exact, 0)(yy, z))(y)
    pzz = jax.grad(lambda zz: jax.grad(_p_exact, 1)(y, zz))(z)
    return pyy + pzz


def test_manufactured_masked_poisson_converges_at_second_order():
    # a sloping bottom B(y) with genuine x/z partial cells (smooth
    # order-8 quadrature of the indicator, no floor so the wet region
    # stays connected — a floored sliver would be an isolated cell, an
    # extra nullspace mode). p_exact = cos(pi sigma) has zero conormal
    # flux at the cut boundary; the RHS is the quadratured wet-part
    # Laplacian (1/V) int_wet lap dV. Measured overall L2 order 2.38,
    # every residual at the 1e-14 floor.
    order_q = 8
    ind = lambda x, y, z: (z > _b(y)).astype(float)  # noqa: E731,ARG005
    errors = []
    ns = (16, 24, 32, 48)
    for n in ns:
        meshes = (
            IntervalMesh(4, (0.0, TWO_PI), periodic=True, name="x"),
            IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y"),
            IntervalMesh(n, (0.0, 1.0), periodic=False, name="z"))
        grid = _fv_grid(
            meshes, ImmersedDomain(ind, order=order_q,
                                   min_fraction=0.0))
        space = _cell_space(grid)
        # fixed-iteration mode: pinned for determinism
        solver = ImmersedPressureSolver(
            grid, space, vertical="z", dsqr=1.0, iterations=500,
            tolerance=None)
        rhs_data = grid._discretize(
            space,
            lambda x, y, z: ind(x, y, z) * jnp.vectorize(_lap)(y, z),
            order_q)
        rhs = grid.create_field(
            space, init=lambda x, y, z: 0.0 * x)  # noqa: ARG005
        rhs = rhs.with_data(rhs_data)
        exact = grid.create_field(
            space,
            init=lambda x, y, z: jnp.vectorize(
                _p_exact)(y, z) + 0.0 * x)
        p, info = solver.solve_info(rhs)
        assert float(info["residual_norm"]) < 1e-9
        wet = solver._wet
        measure = float(jnp.sum(wet.integrate().data))
        p_mean = float(jnp.sum((wet * p).integrate().data)) / measure
        e_mean = float(
            jnp.sum((wet * exact).integrate().data)) / measure
        diff = (p.data - p_mean) - (exact.data - e_mean)
        df = exact.with_data(diff)
        errors.append(float(jnp.sqrt(
            jnp.sum((wet * df * df).integrate().data) / measure)))
    overall = (np.log2(errors[0] / errors[-1])
               / np.log2(ns[-1] / ns[0]))
    assert overall > 1.8


# ================================================================
#  Taught errors
# ================================================================
def test_iterations_property_and_divergence_axis_check():
    _grid, _space, solver = _box_solver(n=8, iterations=17)
    assert solver.iterations == 17
    vel = _random_velocity(solver, seed=4)
    del vel["z"]  # a component short of the axis set
    with pytest.raises(ValueError, match="one component per axis"):
        solver.divergence(vel)


def test_solver_rejects_a_grid_without_immersed_domain():
    meshes = tuple(
        IntervalMesh(8, (0.0, TWO_PI), periodic=True, name=nm)
        for nm in ("x", "y", "z"))
    grid = Grid(meshes)
    grid.merge_overrides(fv_cgrid_overrides(grid.factors))
    with pytest.raises(ValueError, match="no immersed domain"):
        ImmersedPressureSolver(
            grid, _cell_space(grid), vertical="z", dsqr=1.0,
            iterations=2)


def test_solver_rejects_mapped_plus_immersed():
    mapping = CoordinateMapping(
        maps={"zp": lambda z, h: z * h},
        params={"h": lambda x: 1.0 + 0.2 * jnp.sin(x)})
    meshes = (
        IntervalMesh(8, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(8, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(8, (0.0, 1.0), periodic=False, name="z"))
    grid = Grid(
        meshes, mapping=mapping,
        immersed=ImmersedDomain(
            lambda x, y, z: x * 0.0 + 1.0))  # noqa: ARG005
    grid.merge_overrides(fv_cgrid_overrides(grid.factors))
    with pytest.raises(NotImplementedError, match="mapped"):
        ImmersedPressureSolver(
            grid, _cell_space(grid), vertical="z", dsqr=1.0,
            iterations=2)


# ================================================================
#  Optional convergence tolerance (plumbing to ConjugateGradient)
# ================================================================
def test_solver_carries_the_tolerance_to_the_krylov_solver():
    grid, space, _ = _box_solver(n=8, iterations=20)
    solver = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.7, iterations=20,
        tolerance=1e-8)
    assert solver.tolerance == 1e-8
    assert solver.krylov().tolerance == 1e-8


def test_default_tolerance_is_1e_8():
    # build the solver directly (no _box_solver tolerance forwarding) so
    # this observes the ImmersedPressureSolver constructor default
    grid, space, _ = _box_solver(n=8, iterations=20)
    solver = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.7, iterations=20)
    assert solver.tolerance == 1e-8
    assert solver.krylov().tolerance == 1e-8


# ================================================================
#  Smoothing surfaces (B1): diagonal() and vertical_bands()
# ================================================================
def _bounded_masked_solver(n=8, dsqr=0.5, iterations=2):
    """Build sloped topography (bounded z) with partials + dry cells."""
    meshes = (
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="z"))
    slope = lambda x, y, z: jnp.clip(  # noqa: E731
        ((0.6 + 0.15 * jnp.sin(y) + 0.1 * z) - x / TWO_PI) * 8 + 0.5,
        0.0, 1.0)
    grid = _fv_grid(meshes, ImmersedDomain(slope, order=4))
    space = _cell_space(grid)
    solver = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=dsqr, iterations=iterations)
    return grid, space, solver


def _probe_diag_and_zbands(grid, space, solver, z_axis, period=4):
    """Extract (diag, lower, upper) along ``z_axis`` by 4^3 p-coloring."""
    shape = tuple(space.shape)
    a_data = jax.jit(
        lambda d: solver.apply(grid.create_field(space, data=d)).data)
    p = period
    ii, jj, kk = np.indices(shape)
    stack = np.zeros((p, p, p, *shape))
    for cx in range(p):
        for cy in range(p):
            for cz in range(p):
                mask = ((ii % p == cx) & (jj % p == cy)
                        & (kk % p == cz)).astype(np.float64)
                stack[cx, cy, cz] = np.asarray(
                    a_data(jnp.asarray(mask)))
    ci, cj, ck = ii % p, jj % p, kk % p
    assert z_axis == 2  # (x, y, z): the column axis is the third
    diag = stack[ci, cj, ck, ii, jj, kk]
    lower = stack[ci, cj, (kk - 1) % p, ii, jj, kk]
    upper = stack[ci, cj, (kk + 1) % p, ii, jj, kk]
    return diag, lower, upper


def test_immersed_diagonal_is_probe_exact():
    # no cross terms: diag = -sum_a (alpha_f- + alpha_f+)/h^2 exactly,
    # matched to the 4^3-coloured probe on every cell (partials and the
    # cells adjacent to the immersed boundary included)
    grid, space, solver = _bounded_masked_solver()
    diag_probe, _, _ = _probe_diag_and_zbands(grid, space, solver, 2)
    diag = np.asarray(solver.diagonal().data)
    scale = max(np.abs(diag_probe).max(), 1.0)
    assert np.abs(diag_probe - diag).max() <= 1e-11 * scale


def test_immersed_vertical_band_is_probe_exact_and_symmetric():
    grid, space, solver = _bounded_masked_solver()
    _, lo_p, up_p = _probe_diag_and_zbands(grid, space, solver, 2)
    bands = solver.vertical_bands()
    assert bands.axis == 2
    lo = np.asarray(bands.lower.data)
    up = np.asarray(bands.upper.data)
    scale = max(np.abs(np.asarray(bands.diag.data)).max(), 1.0)
    # the vertical leg IS the exact z-off-diagonal (no cross terms)
    assert np.abs(up[:, :, :-1] - up_p[:, :, :-1]).max() <= 1e-11 * scale
    assert np.abs(lo[:, :, 1:] - lo_p[:, :, 1:]).max() <= 1e-11 * scale
    # symmetric per column, Neumann ends
    assert np.abs(lo[:, :, 1:] - up[:, :, :-1]).max() < 1e-12
    assert np.abs(lo[:, :, 0]).max() == 0.0
    assert np.abs(up[:, :, -1]).max() == 0.0


def test_immersed_dry_cells_have_zero_diagonal_and_bands():
    _grid, _space, solver = _bounded_masked_solver()
    dry = np.asarray(solver._cell_mask) == 0.0
    assert dry.any()  # the topography really carves out dry cells
    diag = np.asarray(solver.diagonal().data)
    assert np.abs(diag[dry]).max() == 0.0
    bands = solver.vertical_bands()
    assert np.abs(np.asarray(bands.lower.data)[dry]).max() == 0.0
    assert np.abs(np.asarray(bands.upper.data)[dry]).max() == 0.0


def test_immersed_line_sweep_is_finite_and_zero_on_dry_cells():
    # the smoother's double-jnp.where guard: a dry (zero-diagonal)
    # column takes a finite zero update; wet columns move
    grid, space, solver = _bounded_masked_solver()
    smoother = VerticalLineJacobi(solver.vertical_bands(), omega=0.8)
    b = grid.random.normal(space, seed=1)
    out = smoother.sweep(grid.create_field(space), b, solver.apply)
    data = np.asarray(out.data)
    assert np.all(np.isfinite(data))
    dry = np.asarray(solver._cell_mask) == 0.0
    assert np.abs(data[dry]).max() == 0.0
    assert np.abs(data[~dry]).max() > 0.0


def test_immersed_line_sweep_grad_is_finite():
    grid, space, solver = _bounded_masked_solver()
    smoother = VerticalLineJacobi(solver.vertical_bands(), omega=0.8)
    b = grid.random.normal(space, seed=2)

    def loss(scale):
        out = smoother.sweep(
            grid.create_field(space), scale * b, solver.apply)
        return jnp.sum(out.data ** 2)

    grad = jax.grad(loss)(2.0)
    assert bool(jnp.isfinite(grad))
    assert float(grad) != 0.0


def test_vertical_bands_reject_a_periodic_vertical():
    # the box solver's z is periodic: line smoothing needs a bounded
    # (non-cyclic) column, so vertical_bands refuses it loudly
    _grid, _space, solver = _box_solver(n=8, iterations=2)
    with pytest.raises(NotImplementedError, match="bounded"):
        solver.vertical_bands()


# ================================================================
#  Multigrid preconditioner (B3)
# ================================================================
def _box_bounded(n=16, nz=8):
    """Build a bounded-z {0,1}-box immersed FV grid (line-smoothable)."""
    meshes = (
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(nz, (0.0, 1.0), periodic=False, name="z"))
    box = lambda x, y, z: (  # noqa: E731
        (x > 1.0) & (x < 5.0) & (y > 1.0) & (y < 5.0)
        & (z > 0.2) & (z < 0.8)).astype(float)
    grid = _fv_grid(meshes, ImmersedDomain(box))
    return grid, _cell_space(grid)


def test_preconditioner_knob_rejects_unknown():
    grid, space = _box_bounded(n=8)
    with pytest.raises(ValueError, match="preconditioner must be"):
        ImmersedPressureSolver(grid, space, vertical="z", dsqr=0.5,
                               iterations=5, preconditioner="jacobi")


def test_multigrid_hierarchy_shape_and_degradation():
    # n=16: x, y semicoarsen 16 -> 8 -> 4 (z stays), so a 3-level
    # request yields 3 levels; only the coarsest has transfer=None
    grid, space = _box_bounded(n=16)
    mg = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.5, iterations=5,
        preconditioner="multigrid", multigrid_levels=3)
    levels = mg._build_vcycle().levels
    assert len(levels) == 3
    assert levels[-1].transfer is None
    assert all(level.transfer is not None for level in levels[:-1])
    # a 4-cell grid cannot coarsen (2 < 4): degrades to one smoothing-
    # only level (must work, not raise)
    tiny_grid, tiny_space = _box_bounded(n=4)
    tiny = ImmersedPressureSolver(
        tiny_grid, tiny_space, vertical="z", dsqr=0.5, iterations=5,
        preconditioner="multigrid", multigrid_levels=4)
    assert len(tiny._build_vcycle().levels) == 1


def test_multigrid_levels_defaults_to_floor_limited_depth():
    # the None default (omitted) stores None and forwards to a
    # floor-limited hierarchy: n=16 semicoarsens x, y 16 -> 8 -> 4, a
    # three-level V-cycle, while an int still caps the depth (cap 2)
    grid, space = _box_bounded(n=16)
    default = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.5, iterations=5,
        preconditioner="multigrid")
    assert default._multigrid_levels is None
    assert len(default._build_vcycle().levels) == 3
    capped = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.5, iterations=5,
        preconditioner="multigrid", multigrid_levels=2)
    assert len(capped._build_vcycle().levels) == 2


def test_multigrid_tridiagonal_method_is_stored_and_forwarded():
    # the knob is stored and reaches every level's line smoother
    grid, space = _box_bounded(n=16)
    mg = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.5, iterations=5,
        preconditioner="multigrid", multigrid_levels=3,
        multigrid_tridiagonal_method="pcr")
    assert mg._multigrid_tridiagonal_method == "pcr"
    vcycle = mg._build_vcycle()
    assert vcycle.levels
    assert all(level.smoother.method == "pcr"
               for level in vcycle.levels)
    # the default resolves the kernel against the backend at solve time
    default = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.5, iterations=5)
    assert default._multigrid_tridiagonal_method == "auto"


def test_immersed_solver_rejects_unknown_tridiagonal_method():
    grid, space = _box_bounded(n=8)
    with pytest.raises(ValueError,
                       match="tridiagonal method must be one of"):
        ImmersedPressureSolver(
            grid, space, vertical="z", dsqr=0.5, iterations=5,
            preconditioner="multigrid",
            multigrid_tridiagonal_method="thomas")


def test_multigrid_matches_the_spectral_solve():
    # the multigrid-preconditioned PCG converges and agrees with the
    # spectral-preconditioned PCG on the WET region (dry cells are an
    # unconstrained nullspace: L has zero rows/cols there, so the
    # transfers leave them uncoupled — project() masks them to zero)
    grid, space = _box_bounded(n=16)
    spec = ImmersedPressureSolver(grid, space, vertical="z", dsqr=0.7,
                                  iterations=40)
    mg = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.7, iterations=30,
        preconditioner="multigrid", multigrid_levels=3)
    vel = _random_velocity(spec, seed=3)
    rhs = spec.divergence(vel)
    p_spec = jax.jit(spec.solve)(rhs)
    p_mg = jax.jit(mg.solve)(rhs)
    # the multigrid solve converged
    residual = mg.apply(p_mg) - rhs
    rel = (float(jnp.abs(residual.data).max())
           / float(jnp.abs(rhs.data).max()))
    assert rel < 1e-8
    # ... and matches spectral on the wet cells
    wet = np.asarray(spec._cell_mask)
    diff = np.abs((np.asarray(p_mg.data) - np.asarray(p_spec.data)) * wet)
    scale = np.abs(np.asarray(p_spec.data) * wet).max()
    assert diff.max() / scale < 1e-5


def test_projection_property_removes_the_wet_mean():
    # the exposed projection is the V-orthogonal wet-mean removal the
    # V-cycle installs per level (idempotent; drives the wet integral
    # to zero, dry cells untouched)
    grid, space, solver = _bounded_masked_solver(n=8)
    project = solver.projection
    f = grid.random.normal(space, seed=5)
    pf = project(f)
    wet_mean = float(jnp.sum((solver._wet * pf).integrate().data))
    assert abs(wet_mean) < 1e-12
    # idempotent
    again = project(pf)
    assert float(jnp.abs((again - pf).data).max()) < 1e-12


# ================================================================
#  Perf-guard gap A: the vertical-line smoother on EVERY level
# ================================================================
def _sloped_partial_solver(n=16, nz=8, iterations=18, levels=3):
    """Sloping-bottom immersed FV multigrid solver with partial cells.

    Bounded z (line-smoothable) and genuine fractional cell fractions
    (order-4 quadrature of the indicator), so the anisotropy smoother
    is exercised on the cut-cell geometry multigrid exists for.
    """
    meshes = (
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(nz, (0.0, 1.0), periodic=False, name="z"))
    slope = lambda x, y, z: jnp.clip(  # noqa: E731
        ((0.55 + 0.12 * jnp.sin(y) + 0.1 * x / TWO_PI) - z) * 8 + 0.5,
        0.0, 1.0)
    grid = _fv_grid(meshes, ImmersedDomain(slope, order=4))
    space = _cell_space(grid)
    solver = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.7, iterations=iterations,
        preconditioner="multigrid", multigrid_levels=levels,
        tolerance=None)
    return grid, space, solver


def test_every_level_smoother_is_a_vertical_line_jacobi():
    # a silent swap to a point (isotropic) smoother looks faster per
    # V-cycle but stalls on partial-cell columns (the budget guard
    # below). Pin the anisotropy smoother on every level built the
    # production way — host-side, no compile.
    grid, space = _box_bounded(n=16)
    mg = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.5, iterations=5,
        preconditioner="multigrid", multigrid_levels=3)
    levels = mg._build_vcycle().levels
    assert levels
    assert all(isinstance(level.smoother, VerticalLineJacobi)
               for level in levels)


def test_sloped_immersed_multigrid_converges_within_budget():
    # perf-guard gap A convergence budget: the vertical-line V-cycle
    # drives the sloped partial-cell Poisson below tolerance in a small
    # PCG budget. A point-smoother swap stalls here (rel ~1e-4,
    # measured), so this fails loudly if the anisotropy smoother is lost.
    _grid, _space, solver = _sloped_partial_solver(iterations=18)
    vel = _random_velocity(solver, seed=6)
    rhs = solver.divergence(vel)
    p = jax.jit(solver.solve)(rhs)
    residual = solver.apply(p) - rhs
    rel = (float(jnp.abs(residual.data).max())
           / float(jnp.abs(rhs.data).max()))
    assert rel < 1e-8


# ================================================================
#  Perf-guard gap B: the "auto" knob resolves end to end
# ================================================================
def _resolved_level_methods(solver):
    """Resolve every level smoother's kernel against the backend."""
    vcycle = solver._build_vcycle()
    assert vcycle.levels
    assert all(level.smoother.method == "auto"
               for level in vcycle.levels)
    return {_resolve_tridiagonal_method(level.smoother.method)
            for level in vcycle.levels}


@pytest.mark.skipif(_ON_GPU, reason="pcr is the off-GPU resolution")
def test_auto_method_wires_to_pcr_off_gpu():
    # end-to-end wiring: the model default "auto" reaches every level's
    # smoother and resolves (host-side) to the portable pcr kernel off a
    # GPU — a silent loss costs 9-18x on the CG solve
    grid, space = _box_bounded(n=16)
    mg = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.5, iterations=5,
        preconditioner="multigrid", multigrid_levels=3)
    assert mg._multigrid_tridiagonal_method == "auto"
    assert _resolved_level_methods(mg) == {"pcr"}


@pytest.mark.skipif(not _ON_GPU, reason="cusparse needs a CUDA GPU")
def test_auto_method_wires_to_cusparse_on_gpu():
    # the GPU leg of the same wiring: "auto" resolves to the batched
    # cuSPARSE kernel on a CUDA backend (runs on the A100 suite)
    grid, space = _box_bounded(n=16)
    mg = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.5, iterations=5,
        preconditioner="multigrid", multigrid_levels=3)
    assert mg._multigrid_tridiagonal_method == "auto"
    assert _resolved_level_methods(mg) == {"cusparse"}


# ================================================================
#  Stretched vertical column (plan §6, the stretch-aware bands fix)
# ================================================================
def _stretch(z):
    """Monotone vertical clustering (dS/dz in [0.85, 1.15] > 0)."""
    return z + 0.15 * jnp.sin(2 * np.pi * z) / (2 * np.pi)


def _stretched_immersed_solver(n=8, prec="multigrid", iterations=20):
    """Return a stretched-z immersed FV solver (MappedIntervalMesh z)."""
    from fridom.spatial.meshes.mapped_interval import (  # noqa: PLC0415
        MappedIntervalMesh,
    )
    slope = lambda x, y, z: jnp.clip(  # noqa: E731, ARG005
        (z - 0.2 - 0.1 * jnp.sin(x)) * 6 + 0.5, 0.0, 1.0)
    meshes = (
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y"),
        MappedIntervalMesh(n, (0.0, 1.0), _stretch, periodic=False,
                           name="z"))
    grid = _fv_grid(meshes, ImmersedDomain(slope, order=4))
    space = _cell_space(grid)
    solver = ImmersedPressureSolver(
        grid, space, vertical="z", dsqr=0.5, iterations=iterations,
        tolerance=None, preconditioner=prec)
    return grid, space, solver


def test_stretched_vertical_is_detected():
    _grid, _space, solver = _stretched_immersed_solver()
    assert solver._stretched_vertical is True


def test_stretched_immersed_diagonal_and_bands_are_probe_exact():
    # the stretch-aware bands (plan §6): the analytic diagonal / vertical
    # off-diagonals read the physical grid.measure widths, so they still
    # match the p-coloured probe on a MappedIntervalMesh column
    grid, space, solver = _stretched_immersed_solver()
    diag_probe, lo_p, up_p = _probe_diag_and_zbands(grid, space, solver, 2)
    diag = np.asarray(solver.diagonal().data)
    scale = max(np.abs(diag_probe).max(), 1.0)
    assert np.abs(diag_probe - diag).max() <= 1e-11 * scale
    bands = solver.vertical_bands()
    lo = np.asarray(bands.lower.data)
    up = np.asarray(bands.upper.data)
    assert np.abs(up[:, :, :-1] - up_p[:, :, :-1]).max() <= 1e-11 * scale
    assert np.abs(lo[:, :, 1:] - lo_p[:, :, 1:]).max() <= 1e-11 * scale


def test_stretched_immersed_multigrid_solve_converges():
    # a stretched-z immersed model has no spectral transform, so the
    # multigrid V-cycle is the only preconditioner; the stretch-aware
    # bands make it assemble and converge (plan §6)
    grid, _space, solver = _stretched_immersed_solver(iterations=25)
    vel = {a: grid.random.normal(solver._face[a], seed=i)
           for i, a in enumerate(solver.axes)}
    rhs = solver.divergence(vel)
    p = jax.jit(solver.solve)(rhs)
    residual = solver.apply(p) - rhs
    rel = (float(jnp.abs(residual.data).max())
           / float(jnp.abs(rhs.data).max()))
    assert rel < 1e-8
