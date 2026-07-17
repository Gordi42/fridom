r"""Stretched-column (N1 + N2) shard of the mapped PCG pressure solve.

The measure-adjoint base-axis down-hop (N2) that restores exact SPD
when the mapped column's base mesh is a **stretched**
``MappedIntervalMesh`` (its physical column measure diverges from the
computational one, so the plain ``0.5/0.5`` corner down-hop is no
longer the up-hop's adjoint under CG's physical measure-weighted inner
product), the N1 taught error rejecting the spectral preconditioner on
such a column at construction, the ``preconditioner="none"`` plain-CG
correctness stopgap, and the differentiability of the new weighted-hop
path. Self-contained: the small builders mirror
``test_mapped_pressure_fv.py`` (the combined stretch + terrain grid
auto-resolves to the FV C-grid).

Record: ``design/research/stretched_terrain_combined.md`` §3, §6
(N1/N2), §7 addendum.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.nonhydro2.modules.core import fv_cgrid_overrides
from fridom.nonhydro2.modules.mapped_pressure import (
    MappedPressureSolver,
)
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh

N = 8
DSQR = 0.25


def depth(x):
    """Smooth periodic terrain depth H(x) (20% slope)."""
    return 1.0 + 0.2 * jnp.sin(x)


def stretch(sigma):
    """Monotone sigma clustering (dS/dsigma in [0.85, 1.15] > 0)."""
    return sigma + 0.15 * jnp.sin(2 * np.pi * sigma) / (2 * np.pi)


def build_grid(n=N, init=depth):
    """Build a combined stretch + terrain FV grid (stretched sigma x H)."""
    mx = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="x")
    ms = MappedIntervalMesh(n, (0.0, 1.0), stretch, periodic=False,
                            name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H},
        params={"H": init})
    grid = Grid((mx, ms), mapping=mapping)
    grid.merge_overrides(fv_cgrid_overrides(grid.factors))
    return grid, mx, ms


def cell_space(mx, ms):
    """Return the FV (CellAvg x CellAvg) pressure/divergence space."""
    return mx.cell_avg * ms.cell_avg


def build_solver(n=N, init=depth, **kwargs):
    """Build a plain-CG (stopgap) solver on the combined grid."""
    grid, mx, ms = build_grid(n, init)
    kwargs.setdefault("iterations", 80)
    kwargs.setdefault("weights", {"sigma": 1.0 / DSQR})
    kwargs.setdefault("preconditioner", "none")
    solver = MappedPressureSolver(grid, cell_space(mx, ms), **kwargs)
    return solver, grid, mx, ms


def dot(a, b):
    """Return the measure-weighted inner product CG uses (physical L2)."""
    return float(jnp.sum((a * b).integrate().data))


def stretch_strong(sigma):
    """Strong tanh clustering near sigma=1 (physical width ratio ~20:1)."""
    a = 2.2
    return 1.0 + jnp.tanh(a * (sigma - 1.0)) / jnp.tanh(a)


def build_mg_grid(nx, nz, stretch_fn=stretch, init=depth):
    """Return a combined stretch + terrain FV grid (independent nx, nz)."""
    mx = IntervalMesh(nx, (0.0, 2 * np.pi), periodic=True, name="x")
    ms = MappedIntervalMesh(nz, (0.0, 1.0), stretch_fn, periodic=False,
                            name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H},
        params={"H": init})
    grid = Grid((mx, ms), mapping=mapping)
    grid.merge_overrides(fv_cgrid_overrides(grid.factors))
    return grid, mx, ms


def iters_to_tol(precond, stretch_fn=stretch, *, budget, nx=16, nz=16,
                 tol=1e-6):
    """Return the PCG step count to ``tol`` (info['iterations'])."""
    grid, mx, ms = build_mg_grid(nx, nz, stretch_fn)
    space = cell_space(mx, ms)
    solver = MappedPressureSolver(
        grid, space, iterations=budget, tolerance=tol,
        weights={"sigma": 1.0 / DSQR}, preconditioner=precond,
        multigrid_levels=5)
    rhs = grid.create_field(
        space,
        init=lambda x, sigma: jnp.exp(
            -((x - 3.0) ** 2 + (sigma - 0.5) ** 2) * 3.0))
    rhs = rhs - rhs.mean()

    @jax.jit
    def run(d):
        _x, info = solver.krylov().solve(rhs.with_data(d))
        return info["iterations"]

    return int(run(rhs.data))


# ================================================================
#  The stretched base column is detected
# ================================================================
def test_stretched_base_is_detected():
    solver, *_ = build_solver()
    assert solver._stretched_base
    assert solver.base == "sigma"
    assert solver.mapped == "zp"
    assert solver.coupled == ("x",)


# ================================================================
#  N2: exact SPD under the physical measure-weighted inner product
# ================================================================
def test_operator_is_exactly_symmetric_on_a_stretched_column():
    # the measure-adjoint base-axis down-hop restores the transpose
    # pairing: <A p, q> == <p, A q> under CG's physical measure (record
    # §3, probed 3.7e-16; the plain unweighted hop is rel 2.6e-5)
    solver, grid, mx, ms = build_solver()
    space = cell_space(mx, ms)
    p = grid.random.normal(space, seed=1)
    q = grid.random.normal(space, seed=2)
    left = dot(solver.apply(p), q)
    right = dot(p, solver.apply(q))
    assert abs(left - right) <= 1e-12 * abs(left)


def test_operator_annihilates_constants_on_a_stretched_column():
    # compatibility survives the weighting: A . 1 stays machine zero
    solver, grid, mx, ms = build_solver()
    one = grid.create_field(
        cell_space(mx, ms),
        init=lambda x, sigma: 1.0 + 0.0 * x * sigma)
    assert float(jnp.abs(solver.apply(one).data).max()) < 1e-12


def test_weighted_hop_only_engages_on_a_stretched_column():
    # the guard is exact: forcing the plain (unweighted) hop breaks the
    # symmetry the measure-adjoint hop restores, so the two operators
    # differ and the plain one is measurably asymmetric (the N2 fix is
    # doing real work, not a no-op)
    solver, grid, mx, ms = build_solver()
    space = cell_space(mx, ms)
    p = grid.random.normal(space, seed=3)
    q = grid.random.normal(space, seed=4)
    weighted = solver.apply(p)
    solver._stretched_base = False  # force the plain down-hop
    plain = solver.apply(p)
    assert not np.allclose(np.asarray(weighted.data),
                           np.asarray(plain.data))
    asym = abs(dot(plain, q) - dot(p, solver.apply(q)))
    assert asym > 1e-8 * abs(dot(plain, q))


# ================================================================
#  Plain-CG stopgap: preconditioner="none" converges
# ================================================================
def test_none_preconditioner_cg_reduces_the_residual():
    # unpreconditioned CG on the negative-definite A (same iterates as
    # standard CG on the SPD -A): the correctness stopgap converges on
    # the combined grid
    solver, grid, mx, ms = build_solver(iterations=80, tolerance=None)
    space = cell_space(mx, ms)
    rhs = grid.random.normal(space, seed=9)
    rhs = rhs - rhs.mean()
    p = solver.solve(rhs)
    r0 = float(jnp.abs(rhs.data).max())
    r_end = float(jnp.abs((solver.apply(p) - rhs).data).max())
    assert r_end / r0 < 1e-8
    # the solve pins the mean-free gauge
    assert float(jnp.abs(p.mean().data.ravel()[0])) < 1e-12


def test_none_preconditioner_is_an_accepted_choice():
    # "none" builds without the ValueError the unknown-choice guard
    # raises, and carries no preconditioner into the CG object
    solver, *_ = build_solver(preconditioner="none")
    assert solver.krylov().preconditioner is None


# ================================================================
#  N1: taught errors, and the deferred multigrid route
# ================================================================
def test_spectral_on_a_stretched_column_is_rejected_at_construction():
    # the default preconditioner is spectral, so a bare construction on
    # a stretched column must raise the taught error (not the cryptic
    # DispatchError at the first solve)
    grid, mx, ms = build_grid()
    space = cell_space(mx, ms)
    with pytest.raises(NotImplementedError,
                       match="spectral preconditioner needs"):
        MappedPressureSolver(grid, space, iterations=5,
                             weights={"sigma": 1.0 / DSQR})
    # explicit spectral names the plain-CG stopgap in the message
    with pytest.raises(NotImplementedError, match="preconditioner='none'"):
        MappedPressureSolver(grid, space, iterations=5,
                             weights={"sigma": 1.0 / DSQR},
                             preconditioner="spectral")


def test_multigrid_on_a_stretched_column_builds_and_solves():
    # N3: the V-cycle now LEARNS the grid.measure widths (measure-aware
    # vertical bands + diagonal), so it builds on a stretched column
    # instead of raising "no uniform cell width" — and the
    # preconditioned solve converges (record §6 N3, §7 addendum ruling 3)
    grid, mx, ms = build_mg_grid(16, 8)
    space = cell_space(mx, ms)
    solver = MappedPressureSolver(
        grid, space, iterations=20, tolerance=None,
        weights={"sigma": 1.0 / DSQR}, preconditioner="multigrid")
    solver.krylov()  # builds the V-cycle (previously raised)
    rhs = grid.random.normal(space, seed=7)
    rhs = rhs - rhs.mean()
    p = solver.solve(rhs)
    r0 = float(jnp.abs(rhs.data).max())
    r_end = float(jnp.abs((solver.apply(p) - rhs).data).max())
    assert r_end / r0 < 1e-8
    # the solve pins the mean-free gauge
    assert float(jnp.abs(p.mean().data.ravel()[0])) < 1e-12


def test_multigrid_vcycle_is_symmetric_on_a_stretched_column():
    # the SPD gate (record §6 N3): a V(1,1) cycle whose vertical-line
    # smoother carries the measure-self-adjoint tridiagonal
    # (diag(m_cell) T symmetric) and whose semicoarsening transfers are
    # measure-adjoint (R = M_H^-1 P^T M_h, the uncoarsened stretched
    # column cancelling) is symmetric under CG's physical measure-
    # weighted product: <M u, v> == <u, M v> to roundoff (probed 1.9e-16)
    grid, mx, ms = build_mg_grid(16, 8)
    space = cell_space(mx, ms)
    solver = MappedPressureSolver(
        grid, space, iterations=5, weights={"sigma": 1.0 / DSQR},
        preconditioner="multigrid", multigrid_levels=5)
    vcycle = solver._build_vcycle({})
    u = grid.random.normal(space, seed=1)
    u = u - u.mean()
    v = grid.random.normal(space, seed=2)
    v = v - v.mean()
    left = dot(vcycle(u), v)
    right = dot(u, vcycle(v))
    assert abs(left - right) <= 1e-12 * abs(left)


@pytest.mark.parametrize("stretch_fn", [stretch, stretch_strong],
                         ids=["mild", "strong"])
def test_multigrid_beats_none_on_a_stretched_column(stretch_fn):
    # the effectiveness gate (record §6 N3, §7 ruling 3): PCG with the
    # measure-aware V-cycle reaches the tolerance in substantially fewer
    # iterations than unpreconditioned CG on the same combined
    # stretch+terrain grid — at least halved, conservatively (probed 7
    # vs 221-300, a >30x reduction that stays flat to nz=64)
    none = iters_to_tol("none", stretch_fn, budget=300)
    mg = iters_to_tol("multigrid", stretch_fn, budget=40)
    assert mg < 40           # the V-cycle converged inside its budget
    assert 2 * mg <= none    # at least halved (in fact far more)


def test_multigrid_bands_stay_differentiable():
    # the measure-weighted vertical bands K^bb_f / (m_inner m_cell) and
    # the measure-form diagonal are built from LOGICAL-frame, strictly
    # positive grid.measure widths (never the storage frame's zero-
    # padded ghost slots), so jax.grad through the band assembly is
    # finite and matches a central FD — the new N3 arithmetic adds no
    # masked singularity (contrast the end-to-end SOLVE grad below,
    # which still NaNs on the PRE-EXISTING core diff singularity)
    grid, mx, ms = build_mg_grid(16, 16)
    space = cell_space(mx, ms)

    def loss(w):
        solver = MappedPressureSolver(
            grid, space, iterations=5, weights={"sigma": w},
            preconditioner="multigrid")
        bands = solver.vertical_bands()
        return jnp.sum(bands.lower.data ** 2 + bands.diag.data ** 2
                       + bands.upper.data ** 2)

    w0 = 1.0 / DSQR
    grad = float(jax.grad(loss)(w0))
    assert bool(jnp.isfinite(grad))
    assert grad != 0.0
    h = 1e-4
    fd = float((loss(w0 + h) - loss(w0 - h)) / (2 * h))
    assert abs(grad - fd) <= 1e-4 * abs(fd)


@pytest.mark.xfail(
    reason="the stretched+terrain SOLVE is not yet reverse-mode "
           "differentiable through EITHER preconditioner: the same "
           "PRE-EXISTING core-layer masked singularity as "
           "test_stretched_terrain_solve_grad_matches_fd (the "
           "stretched-mesh 'diff' measure division, staggering.py:582, "
           "over the zero-padded bounded-axis ghost slots) NaNs the "
           "reverse gradient in the operator applications the V-cycle "
           "shares with the CG residual. It is independent of N3 (the "
           "V-cycle's own bands are differentiable — see "
           "test_multigrid_bands_stay_differentiable — and the plain-CG "
           "solve NaNs identically). Flips to XPASS when the core "
           "diff-guard branch lands.",
    strict=False, raises=AssertionError)
def test_multigrid_solve_grad_matches_fd():
    # the end-to-end differentiability invariant (AGENTS.md) through the
    # multigrid-preconditioned solve: jax.grad of a quadratic loss w.r.t.
    # the column weight, finite and matching a central FD
    grid, mx, ms = build_mg_grid(16, 8)
    space = cell_space(mx, ms)
    rhs = grid.random.normal(space, seed=5)
    rhs = rhs - rhs.mean()

    def loss(w):
        solver = MappedPressureSolver(
            grid, space, iterations=8, weights={"sigma": w},
            preconditioner="multigrid", tolerance=None)
        return jnp.sum(solver.solve(rhs).data ** 2)

    w0 = 1.0 / DSQR
    grad = float(jax.grad(loss)(w0))
    h = 1e-4
    fd = float((loss(w0 + h) - loss(w0 - h)) / (2 * h))
    assert bool(jnp.isfinite(grad))
    assert abs(grad - fd) <= 1e-4 * abs(fd)


# ================================================================
#  Differentiability (AGENTS.md): the new weighted-hop path
# ================================================================
def test_weighted_hop_stays_differentiable():
    # the measure-adjoint down-hop is step-path: jax.grad through the
    # corner cross chain that carries it (_cross_to_face -> _down_b_hop)
    # is finite and matches a central FD. This regression-guards the
    # reciprocal-multiply form of the weighting: a naive storage-frame
    # ``reduced / m_cell`` divide is a masked singularity (the base
    # measure's bounded-axis ghost slots are zero), which NaNs the
    # reverse gradient here
    solver, grid, *_ = build_solver()
    g_b = grid.random.normal(solver._face["sigma"], seed=3)

    def loss(c):
        return jnp.sum(solver._cross_to_face("x", c * g_b).data ** 2)

    c0 = 1.3
    grad = float(jax.grad(loss)(c0))
    assert bool(jnp.isfinite(grad))
    assert grad != 0.0
    h = 1e-4
    fd = float((loss(c0 + h) - loss(c0 - h)) / (2 * h))
    assert abs(grad - fd) <= 1e-4 * abs(fd)


@pytest.mark.xfail(
    reason="the stretched+terrain SOLVE is not yet reverse-mode "
           "differentiable: a PRE-EXISTING core-layer masked "
           "singularity (the stretched-mesh 'diff' measure division, "
           "staggering.py:582 'result._data / measure._data' over the "
           "zero-padded bounded-axis ghost slots, plus sibling "
           "storage-frame measure divides) NaNs the reverse gradient. "
           "It is independent of N1/N2 (the plain pre-N2 operator NaNs "
           "identically) and the new weighted hop is itself "
           "differentiable (see test_weighted_hop_stays_differentiable). "
           "Guarding those core divides (double-jnp.where) is a separate "
           "spatial-layer branch; this test flips to XPASS when it lands.",
    strict=False, raises=AssertionError)
def test_stretched_terrain_solve_grad_matches_fd():
    # the end-to-end differentiability invariant the policy asks for:
    # jax.grad of a quadratic loss through a short plain-CG solve w.r.t.
    # the column weight, finite and matching a central FD (tiny grid,
    # preconditioner="none", few iterations)
    grid, mx, ms = build_grid()
    space = cell_space(mx, ms)
    rhs = grid.random.normal(space, seed=5)
    rhs = rhs - rhs.mean()

    def loss(w):
        solver = MappedPressureSolver(
            grid, space, iterations=6, weights={"sigma": w},
            preconditioner="none", tolerance=None)
        return jnp.sum(solver.solve(rhs).data ** 2)

    w0 = 1.0 / DSQR
    grad = float(jax.grad(loss)(w0))
    h = 1e-4
    fd = float((loss(w0 + h) - loss(w0 - h)) / (2 * h))
    assert bool(jnp.isfinite(grad))
    assert abs(grad - fd) <= 1e-4 * abs(fd)
