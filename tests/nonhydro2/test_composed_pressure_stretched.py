r"""Stretched-base shard of the composed mapped + immersed PCG solve.

The GM-D9 full-coarsening default lifted onto a **stretched** composed
base (a ``MappedIntervalMesh`` vertical carrying both a terrain
``CoordinateMapping`` and an ``ImmersedDomain``): the composed solver
inherits the mapped ``_coarsen_vertical`` decision unchanged, so a
stretched composed column now coarsens its vertical alongside the
horizontals. The inherited ``_prewarm_hierarchy`` warms the coarse-grid
memo so the host-validated coarse ``MappedIntervalMesh`` ctor never
re-runs under the solve trace, and each coarse composed level
re-quadratures the immersed fractions on its own coarse spaces (MI-D3).

Covers: the hierarchy actually full-coarsens the stretched column
(level shapes, not just convergence); the V-cycle converges within the
CG budget on a stretched cut chart; full-vs-semi velocity-correction
parity (the coarsening axis is a preconditioner choice, never physics —
the raw pressures differ only by the wet-region nullspace gauge
constant, so the gate is on the corrections, not the pressure); the
semicoarsening opt-out still keeps the vertical full through the chain.
Self-contained: the small builders mirror
``test_composed_pressure.py`` and ``test_mapped_pressure_stretched.py``.

Record: ``design/roadmap/done.md`` (composed stretched full-coarsening,
owner ruled build-now 2026-07-19).
"""
import jax
import jax.numpy as jnp
import numpy as np

from fridom.nonhydro2.modules.composed_pressure import (
    ComposedPressureSolver,
)
from fridom.nonhydro2.modules.core import fv_cgrid_overrides
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh
from fridom.spatial.operators.krylov import _computational_mean

TWO_PI = 2.0 * np.pi
DSQR = 0.5


def depth(x):
    """Smooth periodic water depth H(x) (mild terrain slope)."""
    return 1.0 + 0.3 * jnp.sin(x)


def steep_depth(x):
    """Steep periodic water depth (depth ratio ~4)."""
    return 1.0 + 0.6 * jnp.sin(x)


def stretch(sigma):
    """Monotone sigma clustering (dS/dsigma in [0.85, 1.15] > 0)."""
    return sigma + 0.15 * jnp.sin(2 * np.pi * sigma) / (2 * np.pi)


def cut(x, sigma):
    """Return a sloped immersed cut in the (x, sigma) plane (partials)."""
    return jnp.clip(((0.6 + 0.15 * jnp.sin(x)) - sigma) * 8 + 0.5,
                    0.0, 1.0)


def _fv_grid(meshes, mapping, immersed):
    """Return a stretched terrain + immersed grid with the FV overrides."""
    grid = Grid(meshes, mapping=mapping, immersed=immersed)
    grid.merge_overrides(fv_cgrid_overrides(grid.factors))
    return grid


def _cell_space(grid):
    """Return the FV cell-average pressure space (grid.factors meshes)."""
    factors = [mesh.cell_avg for mesh in grid.factors]
    space = factors[0]
    for factor in factors[1:]:
        space = space * factor
    return grid._laid_out(space)


def build_grid(nx, nz, init=depth, ind=cut, order=4, min_fraction=0.0):
    """Build a stretched sigma x terrain FV grid with an immersed cut."""
    mx = IntervalMesh(nx, (0.0, TWO_PI), periodic=True, name="x")
    ms = MappedIntervalMesh(nz, (0.0, 1.0), stretch, periodic=False,
                            name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H}, params={"H": init})
    grid = _fv_grid(
        (mx, ms), mapping,
        ImmersedDomain(ind, order=order, min_fraction=min_fraction))
    return grid, _cell_space(grid)


def build_solver(nx=16, nz=16, init=depth, ind=cut,
                 preconditioner="multigrid", iterations=30,
                 tolerance=None, **kwargs):
    """Build a composed solver on the stretched terrain + immersed grid."""
    grid, space = build_grid(nx, nz, init, ind)
    solver = ComposedPressureSolver(
        grid, space, iterations=iterations, tolerance=tolerance,
        weights={"sigma": 1.0 / DSQR}, preconditioner=preconditioner,
        **kwargs)
    return grid, space, solver


def random_velocity(grid, solver, seeds=(1, 2)):
    """Build a random (u, w) provisional velocity on the flux faces."""
    return {
        "x": grid.random.normal(solver._face["x"], seed=seeds[0]),
        "sigma": grid.random.normal(solver._face["sigma"], seed=seeds[1])}


def level_shapes(solver):
    """Return the per-level pressure-space shapes of the V-cycle chain."""
    return [tuple(level.operator.func.__self__._space.shape)
            for level in solver._build_vcycle({}).levels]


# ================================================================
#  The stretched composed base is detected
# ================================================================
def test_stretched_composed_base_is_detected():
    _grid, _space, solver = build_solver()
    assert solver._stretched_base
    assert solver.base == "sigma"
    assert solver.mapped == "zp"
    assert solver.coupled == ("x",)
    # the immersed cut is genuine (some cells are dry)
    assert bool((np.asarray(solver._cell_mask) == 0.0).any())


# ================================================================
#  (a) the hierarchy actually full-coarsens the stretched column
# ================================================================
def test_stretched_composed_hierarchy_full_coarsens():
    # the override is lifted: the stretched composed column inherits the
    # GM-D9 full-coarsening default, so BOTH axes halve to the four-cell
    # floor (16 -> 8 -> 4). The inherited _prewarm_hierarchy warms the
    # coarse MappedIntervalMesh memo so the build never re-derives it
    # under the trace; each coarse level re-quadratures the immersed
    # fractions on its own coarse spaces (MI-D3).
    _grid, _space, solver = build_solver(nx=16, nz=16)
    assert solver._multigrid_coarsen_vertical is True   # default on
    assert solver._coarsen_vertical is True             # inherited
    assert level_shapes(solver) == [(16, 16), (8, 8), (4, 4)]
    # every level is a composed solver with its own re-quadratured mask
    for level in solver._build_vcycle({}).levels:
        assert isinstance(level.operator.func.__self__,
                          ComposedPressureSolver)


# ================================================================
#  (b) the V-cycle converges within the CG budget
# ================================================================
def test_stretched_composed_multigrid_converges_within_budget():
    # the fraction-weighted V-cycle on the stretched cut chart drives the
    # masked divergence below tolerance inside a small PCG budget (well
    # under the 30-iteration cap; the probe measured 9-10 iterations)
    grid, _space, solver = build_solver(
        nx=16, nz=16, init=steep_depth, iterations=30, tolerance=None)
    vel = random_velocity(grid, solver)
    div = solver.divergence(vel)
    p, info = solver.krylov().solve(div)
    corr = solver.velocity_correction(p)
    projected = {a: vel[a] - corr[a].retag(vel[a]) for a in solver.axes}
    after = solver.divergence(projected)
    assert float(info["residual_norm"]) < 1e-8
    rel = (float(jnp.abs(after.data).max())
           / float(jnp.abs(div.data).max()))
    assert rel < 1e-9


# ================================================================
#  (c) full-vs-semi velocity-correction parity (NOT raw pressure)
# ================================================================
def test_full_and_semi_coarsening_agree_on_the_correction():
    # the coarsening axis is a preconditioner choice, never physics: on a
    # stretched composed column the full-coarsened and semicoarsened
    # solves produce the SAME velocity corrections (gauge-invariant — the
    # raw pressures differ only by the wet-region nullspace constant, so
    # the gate is on the corrections, not the pressure). Both ride the
    # SAME grid so one divergence is valid for either.
    grid, space, full = build_solver(
        nx=16, nz=16, init=steep_depth, iterations=40, tolerance=None,
        multigrid_coarsen_vertical=True)
    semi = ComposedPressureSolver(
        grid, space, iterations=40, tolerance=None,
        weights={"sigma": 1.0 / DSQR}, preconditioner="multigrid",
        multigrid_coarsen_vertical=False)
    vel = random_velocity(grid, full)
    div = full.divergence(vel)
    p_full = jax.jit(full.solve)(div)
    p_semi = jax.jit(semi.solve)(div)
    corr_full = full.velocity_correction(p_full)
    corr_semi = semi.velocity_correction(p_semi)
    for a in ("x", "sigma"):
        scale = float(jnp.abs(corr_semi[a].data).max())
        diff = float(jnp.abs(
            (corr_full[a] - corr_semi[a]).data).max())
        assert diff / scale < 1e-10


# ================================================================
#  (d) the semicoarsening opt-out still keeps the vertical full
# ================================================================
def test_stretched_semicoarsening_knob_keeps_the_vertical_full():
    # the explicit opt-out survives the lift: multigrid_coarsen_vertical
    # =False keeps the stretched sigma column full at every level (x
    # halves 16 -> 8 -> 4; the sigma column stays 16 throughout)
    _grid, _space, solver = build_solver(
        nx=16, nz=16, multigrid_coarsen_vertical=False)
    assert solver._stretched_base
    assert solver._coarsen_vertical is False
    assert level_shapes(solver) == [(16, 16), (8, 16), (4, 16)]


# ================================================================
#  Differentiability (AGENTS.md): grad through the stretched solve
# ================================================================
# reverse-mode through the FULL-coarsening multigrid solve now builds on
# >1 device: the fine->coarse transfer's ``restrict`` spells ``P^T`` with
# forward primitives, so the transpose of ``jnp.roll`` no longer lands in
# the forward graph where the XLA SPMD partitioner miscompiled it at one
# cell per device (transfer.py; the fix that dropped the mapped and
# composed single_device marks). Runs on any device count.
def test_stretched_composed_solve_grad_matches_fd():
    # the end-to-end differentiability invariant (AGENTS.md) through the
    # stretched composed multigrid-preconditioned solve: jax.grad of a
    # quadratic loss w.r.t. the column weight, finite and matching a
    # central FD (tiny grid, few iterations)
    grid, space = build_grid(8, 8)
    rhs = grid.random.normal(space, seed=5)
    rhs = rhs - _computational_mean(rhs)

    def loss(w):
        solver = ComposedPressureSolver(
            grid, space, iterations=8, weights={"sigma": w},
            preconditioner="multigrid", tolerance=None)
        return jnp.sum(solver.solve(solver._projection(rhs)).data ** 2)

    w0 = 1.0 / DSQR
    grad = float(jax.grad(loss)(w0))
    h = 1e-4
    fd = float((loss(w0 + h) - loss(w0 - h)) / (2 * h))
    assert bool(jnp.isfinite(grad))
    assert grad != 0.0
    assert abs(grad - fd) <= 1e-4 * abs(fd)
