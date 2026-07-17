"""Tests for the MultigridVCycle engine (spatial/operators/multigrid.py).

The gates of plan §B2/§B5 on cheap flat isotropic Poisson hierarchies
(a hand 2nd-difference Laplacian on ``Grid.coarsened`` levels, transfers
from ``GridTransfer``, damped point-Jacobi smoothers): the V(1,1) cycle
contracts and its rate is grid-independent (GB-1 contraction), the
preconditioner is symmetric in the measure-weighted product (GB-1
symmetry), a zero residual gives a zero correction, the cycle beats no
preconditioning inside CG, the jitted cycle compiles once and its
CG-preconditioned HLO is flat in the CG iteration count, ``pre_sweeps
!= post_sweeps`` is refused, and ``jax.grad`` flows through a
cycle-preconditioned solve.

Note on the contraction rate: damped **point** Jacobi has a smoothing
floor (the true 2-grid factor is ~0.36 in 2D / ~0.54 in 3D at the
optimal omega=0.8, matching Trottenberg's LFA tables), so a V(1,1)
rate below ~0.4 is not reachable with this smoother — the plan's
aspirational rho<=0.2 needs a stronger smoother (Gauss-Seidel /
Chebyshev / V(2,2), the recorded upgrades) or the line smoother of the
real anisotropic workload. These tests therefore gate the honestly
achievable rate and, more strongly, grid-independence.
"""
import jax
import jax.numpy as jnp
import pytest

from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh
from fridom.spatial.operators.krylov import ConjugateGradient
from fridom.spatial.operators.multigrid import (
    DampedJacobi,
    MultigridLevel,
    MultigridVCycle,
    VerticalBands,
    VerticalLineJacobi,
)
from fridom.spatial.operators.staggering import uniform_spacing
from fridom.spatial.operators.transfer import GridTransfer


# ================================================================
#  Self-contained flat-isotropic Poisson hierarchy builders
# ================================================================
def cell_space(grid):
    """Return the all-Center product space of a grid."""
    factors = tuple(mesh.center for mesh in grid.factors)
    space = factors[0]
    for factor in factors[1:]:
        space = space * factor
    return grid._laid_out(space)


def laplacian(names):
    """Return the 2nd-order Laplacian ``sum_a d2/da2`` (neg-def)."""
    def apply(field):
        out = None
        for axis in names:
            term = field.diff(axis).diff(axis)
            out = term if out is None else out + term
        return out
    return apply


def diagonal_field(grid, space):
    """Return the constant diagonal ``-2 sum_a 1/da^2`` of A."""
    value = sum(-2.0 / uniform_spacing(space.factor(a)) ** 2
                for a in space.active_axis_names)
    template = grid.create_field(space)
    return template.with_data(
        jnp.full(template.data.shape, value, dtype=template.data.dtype))


def build_hierarchy(n, dim, num_levels, *, omega=0.8, coarse_sweeps=8,
                    pre=1):
    """Return (grid, space, apply_A, vcycle) for a flat Poisson tower."""
    names = tuple("xyz"[:dim])
    grid = Grid(tuple(
        IntervalMesh(n, (0.0, 1.0), name=nm) for nm in names))
    grids = [grid]
    spaces = [cell_space(grid)]
    for _ in range(num_levels - 1):
        coarse = grids[-1].coarsened(2)
        grids.append(coarse)
        spaces.append(cell_space(coarse))
    levels = []
    for level in range(num_levels):
        names_l = spaces[level].active_axis_names
        smoother = DampedJacobi(
            diagonal_field(grids[level], spaces[level]), omega=omega)
        transfer = (GridTransfer(grids[level], grids[level + 1], order=2)
                    if level < num_levels - 1 else None)
        levels.append(MultigridLevel(
            laplacian(names_l), smoother,
            lambda f: f - f.mean(), transfer))
    vcycle = MultigridVCycle(
        tuple(levels), pre_sweeps=pre, post_sweeps=pre,
        coarse_sweeps=coarse_sweeps)
    return grids[0], spaces[0], laplacian(spaces[0].active_axis_names), \
        vcycle


def weighted_dot(a, b):
    """Return the measure-weighted inner product (krylov `_dot`)."""
    return float(jnp.sum((a * b).integrate().data))


def contraction(grid, space, apply_a, vcycle, *, seed=0, niter=15):
    """Asymptotic error contraction of the stationary V-cycle iteration."""
    err = grid.random.normal(space, seed=seed)
    err = err - err.mean()

    @jax.jit
    def step(data):
        field = err.with_data(data)
        nxt = field - vcycle(apply_a(field))
        return (nxt - nxt.mean()).data

    data = err.data
    norms = []
    for _ in range(niter):
        data = step(data)
        norms.append(float(jnp.sqrt(jnp.sum(data ** 2))))
    return norms[-1] / norms[-2]


# ================================================================
#  (a) GB-1 contraction: the cycle contracts, grid-independently
# ================================================================
def test_vcycle_contracts_below_the_point_jacobi_floor():
    # a full-depth 2D flat-Poisson tower: damped point-Jacobi V(1,1)
    # contracts at ~0.42 (its smoothing floor). The gate is the honest
    # achievable rate; the plan's rho<=0.2 needs a stronger smoother.
    grid, space, apply_a, vcycle = build_hierarchy(32, 2, 5)
    rho = contraction(grid, space, apply_a, vcycle)
    assert rho < 0.5


def test_contraction_rate_is_grid_independent():
    # the multigrid signature: the rate does not grow with resolution
    rho16 = contraction(*build_hierarchy(16, 2, 4))
    rho32 = contraction(*build_hierarchy(32, 2, 5))
    assert rho32 < 0.5
    assert abs(rho32 - rho16) < 0.1


# ================================================================
#  (b) GB-1 symmetry: M^-1 is symmetric in the weighted product
# ================================================================
@pytest.mark.parametrize("num_levels", [2, 3])
def test_preconditioner_is_symmetric(num_levels):
    grid, space, _, vcycle = build_hierarchy(8, 3, num_levels)
    u = grid.random.normal(space, seed=1)
    u = u - u.mean()
    v = grid.random.normal(space, seed=2)
    v = v - v.mean()
    left = weighted_dot(vcycle(u), v)
    right = weighted_dot(u, vcycle(v))
    assert abs(left - right) <= 1e-10 * abs(left)


# ================================================================
#  (c) fixed point: a zero residual gives a zero correction
# ================================================================
def test_zero_residual_gives_zero_correction():
    grid, space, _, vcycle = build_hierarchy(8, 3, 3)
    zero = grid.create_field(space)
    correction = vcycle(zero)
    assert float(jnp.abs(correction.data).max()) == 0.0


# ================================================================
#  (d) the cycle-preconditioned CG beats the unpreconditioned solve
# ================================================================
def test_vcycle_preconditioning_beats_unpreconditioned_cg():
    grid, space, apply_a, vcycle = build_hierarchy(16, 2, 4)
    rhs = grid.random.normal(space, seed=3)
    rhs = rhs - rhs.mean()
    pre = ConjugateGradient(
        apply_a, preconditioner=vcycle, iterations=3, project_mean=True)
    raw = ConjugateGradient(
        apply_a, iterations=3, project_mean=True)
    _, pre_info = pre.solve(rhs)
    _, raw_info = raw.solve(rhs)
    assert float(pre_info["residual_norm"]) < float(
        raw_info["residual_norm"])


# ================================================================
#  (e) trace/compile stability
# ================================================================
def test_cycle_compiles_once_across_calls(compile_counter):
    grid, space, _, vcycle = build_hierarchy(8, 3, 3)
    rhs1 = grid.random.normal(space, seed=4)
    rhs1 = rhs1 - rhs1.mean()
    rhs2 = rhs1 * 3.0

    @jax.jit
    def run(data):
        return vcycle(rhs1.with_data(data)).data

    warm = run(rhs1.data)
    compile_counter.reset()
    a = run(rhs1.data)
    assert compile_counter.count == 0  # cache hit, no recompile
    b = run(rhs2.data)
    assert compile_counter.count == 0  # swept values, one trace
    assert jnp.allclose(a, warm)
    assert jnp.allclose(b, 3.0 * warm)


def test_preconditioned_cg_hlo_is_flat_in_the_iteration_count():
    # the cycle is a fixed block inside the CG scan body, so growing the
    # CG iteration count does not grow the HLO (the krylov O(1) property)
    grid, space, apply_a, vcycle = build_hierarchy(8, 3, 3)
    rhs = grid.random.normal(space, seed=5)
    rhs = rhs - rhs.mean()

    def hlo_lines(iterations):
        cg = ConjugateGradient(
            apply_a, preconditioner=vcycle, iterations=iterations,
            project_mean=True)
        lowered = jax.jit(
            lambda d: cg(rhs.with_data(d)).data).lower(rhs.data)
        return lowered.as_text().count("\n")

    assert hlo_lines(20) == hlo_lines(5)


# ================================================================
#  (f) pre_sweeps != post_sweeps is refused (symmetry)
# ================================================================
def test_asymmetric_sweeps_raise():
    _, _, _, _ = build_hierarchy(8, 3, 2)  # warm the builders
    grid = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    space = cell_space(grid)
    level = MultigridLevel(
        laplacian(("x",)),
        DampedJacobi(diagonal_field(grid, space)),
        lambda f: f - f.mean(), None)
    with pytest.raises(ValueError, match="pre_sweeps must equal"):
        MultigridVCycle((level,), pre_sweeps=1, post_sweeps=2)


def test_interior_level_needs_a_transfer():
    grid = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    space = cell_space(grid)
    level = MultigridLevel(
        laplacian(("x",)),
        DampedJacobi(diagonal_field(grid, space)),
        lambda f: f - f.mean(), None)
    with pytest.raises(ValueError, match="only the coarsest"):
        MultigridVCycle((level, level))


def test_empty_levels_raise():
    with pytest.raises(ValueError, match="at least one level"):
        MultigridVCycle(())


def test_coarsest_level_must_have_no_transfer():
    grid = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    space = cell_space(grid)
    transfer = GridTransfer(grid, grid.coarsened(2), order=2)
    level = MultigridLevel(
        laplacian(("x",)),
        DampedJacobi(diagonal_field(grid, space)),
        lambda f: f - f.mean(), transfer)
    with pytest.raises(ValueError, match="must have transfer=None"):
        MultigridVCycle((level,))


def test_non_positive_sweeps_raise():
    grid = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    space = cell_space(grid)
    level = MultigridLevel(
        laplacian(("x",)),
        DampedJacobi(diagonal_field(grid, space)),
        lambda f: f - f.mean(), None)
    with pytest.raises(ValueError, match=">= 1"):
        MultigridVCycle((level,), pre_sweeps=0, post_sweeps=0)


def test_properties_expose_the_configuration():
    grid, space, _, vcycle = build_hierarchy(8, 3, 2, omega=0.7,
                                              coarse_sweeps=5)
    assert len(vcycle.levels) == 2
    assert vcycle.pre_sweeps == 1
    assert vcycle.post_sweeps == 1
    assert vcycle.coarse_sweeps == 5
    smoother = vcycle.levels[0].smoother
    assert smoother.omega == 0.7
    assert smoother.diagonal is vcycle.levels[0].smoother.diagonal
    d = diagonal_field(grid, space)
    bands = VerticalBands(d, d, d, 0)
    line = VerticalLineJacobi(bands, omega=0.9)
    assert line.omega == 0.9
    assert line.bands is bands


# ================================================================
#  (h) N3: measure-weighted line smoothing on a stretched column
# ================================================================
def _mapped_stretch(sigma):
    """Return a monotone periodic clustering (non-uniform widths)."""
    return sigma + 0.18 * jnp.sin(2.0 * jnp.pi * sigma) / (2.0 * jnp.pi)


def _measure_line_bands(grid, space, z_axis, dx):
    r"""Return the measure-self-adjoint tridiagonal of the stretched line.

    The unit-coefficient vertical Laplacian's off-diagonals in physical
    measure form: ``upper[c] = 1 / (m_inner[c] m_cell[c])`` and (periodic
    column) ``lower[c] = 1 / (m_inner[c-1] m_cell[c])``. These are
    Euclidean-asymmetric (``lower[c] != upper[c-1]``) but self-adjoint
    under the physical cell measure — ``m_cell[c-1] upper[c-1] ==
    m_cell[c] lower[c]``, i.e. ``diag(m_cell) T`` is symmetric — exactly
    the N3 construction ``MappedPressureSolver.vertical_bands`` builds on
    a bounded stretched column, exercised here at the engine level.
    """
    face = grid.create_field(space).diff("sigma").function_space
    m_inner = grid.measure(face, "sigma").data
    m_cell = grid.measure(space, "sigma").data
    band = jnp.ones_like(m_inner) / m_inner
    lower = jnp.roll(band, 1, axis=z_axis) / m_cell
    upper = band / m_cell
    diag = (-(band + jnp.roll(band, 1, axis=z_axis)) / m_cell
            - 2.0 / dx ** 2)
    template = grid.create_field(space)
    shape = template.data.shape
    return VerticalBands(
        template.with_data(jnp.broadcast_to(lower, shape)),
        template.with_data(jnp.broadcast_to(diag, shape)),
        template.with_data(jnp.broadcast_to(upper, shape)), z_axis)


def mapped_line_hierarchy(nx, nz, num_levels):
    """Build a periodic-x, stretched-sigma vertical-line Poisson tower."""
    def build(nx_):
        mx = IntervalMesh(nx_, (0.0, 1.0), periodic=True, name="x")
        ms = MappedIntervalMesh(nz, (0.0, 1.0), _mapped_stretch,
                                periodic=True, name="sigma")
        return Grid((mx, ms))
    grids = [build(nx)]
    for _ in range(num_levels - 1):
        grids.append(grids[-1].coarsened({"x": 2}))
    spaces = [cell_space(g) for g in grids]
    z_axis = spaces[0].names.index("sigma")
    dx = uniform_spacing(spaces[0].factor("x"))
    levels = []
    for lvl in range(num_levels):
        g, sp = grids[lvl], spaces[lvl]
        smoother = VerticalLineJacobi(
            _measure_line_bands(g, sp, z_axis, dx), omega=0.8)
        transfer = (GridTransfer(grids[lvl], grids[lvl + 1], order=2)
                    if lvl < num_levels - 1 else None)
        levels.append(MultigridLevel(
            laplacian(sp.active_axis_names), smoother,
            lambda f: f - f.mean(), transfer))
    return grids[0], spaces[0], \
        laplacian(spaces[0].active_axis_names), \
        MultigridVCycle(tuple(levels))


def test_stretched_line_vcycle_is_measure_symmetric():
    # N3 engine gate: a VerticalLineJacobi smoother whose tridiagonal is
    # Euclidean-ASYMMETRIC but self-adjoint under the physical cell
    # measure (diag(m_cell) T symmetric), plus the measure-adjoint
    # GridTransfer pair, composes into a V(1,1) cycle symmetric in CG's
    # measure-weighted product on a stretched column (probed 2e-16)
    grid, space, _, vcycle = mapped_line_hierarchy(8, 8, 2)
    u = grid.random.normal(space, seed=1)
    u = u - u.mean()
    v = grid.random.normal(space, seed=2)
    v = v - v.mean()
    left = weighted_dot(vcycle(u), v)
    right = weighted_dot(u, vcycle(v))
    assert abs(left - right) <= 1e-12 * abs(left)


def test_stretched_line_vcycle_beats_unpreconditioned_cg():
    # the measure-weighted V-cycle is an effective preconditioner on a
    # stretched column: at a fixed budget the preconditioned CG residual
    # is far below the unpreconditioned one (the operator-level echo of
    # the solver-level 7-vs-300 iteration count on the real bounded
    # terrain+stretch grid, test_mapped_pressure_stretched)
    grid, space, apply_a, vcycle = mapped_line_hierarchy(16, 8, 3)
    rhs = grid.random.normal(space, seed=3)
    rhs = rhs - rhs.mean()
    pre = ConjugateGradient(
        apply_a, preconditioner=vcycle, iterations=3, project_mean=True)
    raw = ConjugateGradient(apply_a, iterations=3, project_mean=True)
    _, pre_info = pre.solve(rhs)
    _, raw_info = raw.solve(rhs)
    assert (float(pre_info["residual_norm"])
            < 0.5 * float(raw_info["residual_norm"]))


# ================================================================
#  (g) autodiff: grad through a cycle-preconditioned solve (policy)
# ================================================================
def test_grad_through_preconditioned_solve_matches_fd():
    grid, space, apply_a, vcycle = build_hierarchy(8, 3, 3)
    base = grid.random.normal(space, seed=6)
    base = base - base.mean()

    def loss(scale):
        cg = ConjugateGradient(
            apply_a, preconditioner=vcycle, iterations=3,
            project_mean=True)
        return jnp.sum(cg(scale * base).data ** 2)

    grad = jax.grad(loss)(2.0)
    eps = 1e-5
    fd = (loss(2.0 + eps) - loss(2.0 - eps)) / (2 * eps)
    assert bool(jnp.isfinite(grad))
    assert abs(float(grad) - float(fd)) <= 1e-4 * abs(float(fd))
