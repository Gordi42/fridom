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

TWO_PI = 2.0 * np.pi


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
