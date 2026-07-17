"""
Terrain-following / boundary-fitted pressure solve (stage C3).

The CS-D2 gates on ``z = sigma * H(x)`` (target b) and
``yp = y * Y_N(x)`` (target c): the mapped-flat identity (a
constant-H mapping reproduces the unmapped run to rounding), a
manufactured solution converging at 2nd order, PCG iteration counts
resolution-independent (the preconditioner promise), the projection
driving the mapped divergence to solver tolerance, the
boundary-fitted channel against an ``ImmersedDomain`` staircase
reference, compile-once across steps, and the forced-4 multi-device
projection.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.nonhydro2.modules.mapped_pressure import (
    MappedPressureSolver,
)
from fridom.spatial.bc import BC
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.krylov import ConjugateGradient
from fridom.spatial.spaces.nodal import NodeSet

N = 8
DT = 0.02
DSQR = 0.25
H0 = 0.7
TWO_PI = 2.0 * np.pi

#: minimum observed convergence order for the 2nd-order operator
ORDER_FLOOR = 1.8


def depth(x):
    """Smooth periodic water depth H(x) (20% slope)."""
    return 1.0 + 0.2 * jnp.sin(x)


def depth_x(x):
    """Return the analytic derivative of ``depth``."""
    return 0.2 * jnp.cos(x)


def build_column_grid(n, init=depth):
    """2D terrain-following grid, ``zp = sigma * H(x)``."""
    mx = IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x")
    ms = IntervalMesh(n, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H},
        params={"H": init})
    return Grid((mx, ms), mapping=mapping), mx, ms


# ================================================================
#  Gate 1: the mapped-flat identity
# ================================================================
# Rotation is opt-in (nh.Model's coriolis=None means NO rotation), so
# the builders below name the f0 = 1 f-plane the old implicit default
# installed: the mapped/flat comparison must compare the same physics,
# and a linear nonhydro model with no rotation at all would advance
# neither u nor v (the D1.4 coverage lint).
def make_flat_model(**kwargs):
    mx = IntervalMesh(N, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(N, (0.0, TWO_PI), periodic=True, name="y")
    mz = IntervalMesh(N, (0.0, H0), periodic=False, name="z")
    return nh.Model(grid=Grid((mx, my, mz)),
                    coriolis=nh.FPlaneCoriolis(f0=1.0), **kwargs)


def make_mapped_flat_model(**kwargs):
    mx = IntervalMesh(N, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(N, (0.0, TWO_PI), periodic=True, name="y")
    mz = IntervalMesh(N, (0.0, 1.0), periodic=False, name="z")
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H},
        params={"H": lambda x: H0 + 0.0 * x})
    # 4 PCG iterations, not the model default 30: on a CONSTANT-H
    # mapping the folded preconditioner is the exact inverse, so the
    # solve converges on iteration 1 and every further iteration is
    # an exact no-op (the guarded ratio). Measured: identical
    # deviation from the flat run (5.5e-15 relative after 200 steps)
    # at 4, 8 and 30 iterations
    return nh.Model(grid=Grid((mx, my, mz), mapping=mapping),
                    coriolis=nh.FPlaneCoriolis(f0=1.0),
                    pressure_iterations=4, **kwargs)


def test_mapped_flat_identity_reproduces_the_unmapped_run():
    # constant H: the mapped operator's coefficients fold to the
    # exact flat ones (K^xx = H0, K^bb = w/H0 — both sides of the
    # Poisson equation carry the same J = H0 factor) and the folded
    # preconditioner is the exact inverse, so PCG converges on the
    # first iteration. NOT asserted bitwise: the mapped path solves
    # through PCG + metric-field multiplications while the flat path
    # is the direct spectral inverse — two different compiled
    # programs whose last bits legitimately differ. Measured
    # deviation after 200 steps: <= 6e-15 relative; asserted at
    # 1e-12 as the comfortable to-rounding bound.
    flat = make_flat_model(dt=DT, dsqr=DSQR, advection=False)
    mapped = make_mapped_flat_model(dt=DT, dsqr=DSQR,
                                    advection=False)
    hor = (np.arange(N) + 0.5) * (TWO_PI / N)
    ver = (np.arange(N) + 0.5) / N  # z_phys = ver * H0 on both
    x, y, z = np.meshgrid(hor, hor, ver, indexing="ij")
    fields = {
        "u": 0.01 * np.sin(y) * np.cos(np.pi * z),
        "v": 0.01 * np.sin(x),
        "b": 0.01 * np.cos(x) * np.cos(np.pi * z),
    }
    flat.set_fields(**fields)
    mapped.set_fields(**fields)
    flat.advance(200)
    mapped.advance(200)
    for c in ("u", "v", "w", "b", "p"):
        a = np.asarray(flat.state[c].data)
        b = np.asarray(mapped.state[c].data)
        scale = np.abs(a).max()
        assert np.abs(a - b).max() <= 1e-12 * scale, c


# ================================================================
#  Gate 2: manufactured solution, 2nd-order convergence
# ================================================================
def p_exact(x, s):
    """Analytic pressure with zero conormal flux at the walls."""
    return (s * (1.0 - s)) ** 2 * jnp.cos(x)


def flux_x(x, s):
    """Evaluate the analytic x-flux ``K^xx p_x + K^xs p_s``."""
    px = jax.grad(p_exact, argnums=0)(x, s)
    ps = jax.grad(p_exact, argnums=1)(x, s)
    return depth(x) * px - s * depth_x(x) * ps


def flux_s(x, s):
    """Evaluate the analytic column flux ``K^sx p_x + K^ss p_s``."""
    px = jax.grad(p_exact, argnums=0)(x, s)
    ps = jax.grad(p_exact, argnums=1)(x, s)
    return (-s * depth_x(x) * px
            + ((s * depth_x(x)) ** 2 + 1.0) / depth(x) * ps)


def rhs_exact(x, s):
    """Evaluate the manufactured right-hand side ``A p``."""
    dfx = jax.grad(flux_x, argnums=0)(x, s)
    dfs = jax.grad(flux_s, argnums=1)(x, s)
    return dfx + dfs


def test_manufactured_solution_converges_at_second_order():
    # measured errors (max norm): 4.66e-3, 1.17e-3, 2.92e-4 at
    # n = 16, 32, 64 — observed orders 2.00, 2.00
    errors = []
    for n in (16, 32, 64):
        grid, mx, ms = build_column_grid(n)
        space = mx.center * ms.center
        rhs = grid.create_field(
            space,
            init=lambda x, sigma: jnp.vectorize(rhs_exact)(
                x, sigma + 0.0 * x))
        exact = grid.create_field(
            space,
            init=lambda x, sigma: p_exact(x, sigma) + 0.0 * x)
        solver = MappedPressureSolver(grid, space, iterations=12)
        p = solver.solve(rhs)
        diff = (p - p.mean()) - (exact - exact.mean())
        errors.append(float(jnp.abs(diff.data).max()))
    orders = np.log2(np.asarray(errors[:-1])
                     / np.asarray(errors[1:]))
    assert np.all(orders > ORDER_FLOOR)


# ================================================================
#  Gate 3: PCG iterations are resolution-independent (CS-D2)
# ================================================================
def test_pcg_residual_is_resolution_independent():
    # the folded-coefficient spectral preconditioner keeps the
    # reduction per iteration flat across resolutions. Measured
    # relative residual at a fixed 8-iteration budget (20% slope):
    # 3.5e-8 (n=16), 3.7e-8 (n=32), 3.8e-8 (n=64) — about one
    # decade per iteration; at 50% slope: 1.2e-4 .. 1.7e-4.
    residuals = []
    for n in (16, 32, 64):
        grid, mx, ms = build_column_grid(n)
        space = mx.center * ms.center
        rhs = grid.random.normal(space, seed=3)
        rhs = rhs - rhs.mean()
        rhs_norm = float(jnp.sqrt(jnp.sum(
            (rhs * rhs).integrate().data)))
        solver = MappedPressureSolver(
            grid, space, iterations=8,
            weights={"sigma": 1.0 / DSQR})
        _, info = solver.krylov().solve(rhs)
        residuals.append(float(info["residual_norm"]) / rhs_norm)
    residuals = np.asarray(residuals)
    assert np.all(residuals < 1e-6)
    # flat across the dyadic chain: no resolution degrades the
    # budget by more than one decade
    assert residuals.max() / residuals.min() < 10.0


# ================================================================
#  Gate 4: the projection removes the mapped divergence
# ================================================================
def test_projection_drives_mapped_divergence_to_tolerance():
    grid, mx, ms = build_column_grid(24)
    space = mx.center * ms.center
    # 16 iterations: the projection is converged there (measured
    # relative divergence 1.4e-15 at 16 and at 20)
    # fixed-iteration mode: pinned for determinism
    solver = MappedPressureSolver(
        grid, space, iterations=16, weights={"sigma": 1.0 / DSQR},
        tolerance=None)
    u = grid.random.normal(mx.right * ms.center, seed=4)
    w = grid.random.normal(
        mx.center * ms.nodal(NodeSet.INNER, bc=BC.DIRICHLET),
        seed=5)
    vel = {"x": u, "sigma": w}
    div = solver.divergence(vel)
    p = solver.solve(div)
    corr = solver.velocity_correction(p)
    after = solver.divergence({
        "x": u - corr["x"].retag(u),
        "sigma": w - corr["sigma"].retag(w)})
    rel = (float(jnp.abs(after.data).max())
           / float(jnp.abs(div.data).max()))
    # measured 1.4e-15 at the 16-iteration budget
    assert rel < 1e-10


# ================================================================
#  Target c: the boundary-fitted channel (yp = y * Y_N(x))
# ================================================================
def channel_width(x):
    """Return the northern boundary position Y_N(x)."""
    return 1.0 - 0.2 * jnp.cos(x)


def make_channel_model(n=N, **kwargs):
    mx = IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(n, (0.0, 1.0), periodic=False, name="y")
    mz = IntervalMesh(n, (0.0, 1.0), periodic=False, name="z")
    mapping = CoordinateMapping(
        maps={"yp": lambda y, YN: y * YN},
        params={"YN": channel_width})
    # 16 PCG iterations, not the model default 30: the solve has
    # converged by 16 — measured post-projection divergence 2.14e-15
    # and transport spread 1.1e-16 at BOTH 16 and 30 (the assertions
    # keep their tolerances); 8 would NOT do (divergence 1.8e-9)
    return nh.Model(grid=Grid((mx, my, mz), mapping=mapping),
                    coriolis=nh.FPlaneCoriolis(f0=1.0),
                    pressure_iterations=16, **kwargs)


def test_boundary_fitted_channel_assembles_and_projects():
    # the same machinery on a horizontal coordinate: a uniform
    # inflow through the narrowing channel projects divergence-free
    # (in the mapped sense), the along-channel transport becomes
    # exactly constant (the flux form telescopes), and the flow
    # accelerates through the narrows as 1 / Y_N
    # fixed-iteration mode: pinned for determinism
    model = make_channel_model(dt=DT, dsqr=0.5, advection=False,
                               pressure_tolerance=None)
    hor = (np.arange(N) + 0.5) * (TWO_PI / N)
    ver = (np.arange(N) + 0.5) / N
    _, _, z = np.meshgrid(hor, ver, ver, indexing="ij")
    model.set_fields(u=1.0 + 0.0 * z, b=0.01 * np.cos(np.pi * z))
    model.advance(5)
    assert not model.panicked
    solver = MappedPressureSolver(
        model.state["u"].grid, model.state["p"].function_space,
        iterations=1, weights={"z": 1.0 / 0.5})
    div = solver.divergence({
        "x": model.state["u"], "y": model.state["v"],
        "z": model.state["w"]})
    assert float(jnp.abs(div.data).max()) < 1e-10
    jac = solver._jacobian(model.state["u"])
    transport = np.asarray(
        (model.state["u"] * jac).data).sum(axis=(1, 2))
    assert np.ptp(transport) < 1e-10 * np.abs(transport).max()
    u = np.asarray(model.state["u"].data)
    # width 0.8 .. 1.2: mass conservation forces u ~ 1 / Y_N
    assert u.max() > 1.15
    assert u.min() < 0.87


def masked_projection(n, iterations=240):
    """
    Project a uniform inflow on an immersed staircase channel.

    Description
    -----------
    The masked flux-form Poisson operator is SPD on the wet
    subspace, so plain CG applies; it runs UNpreconditioned — the
    unmasked spectral inverse is only semidefinite on the wet
    subspace (its global k = 0 gauge mixes with the wet-constant
    nullspace) and destabilizes the iteration. The budget is set by
    the residual gate of the caller (< 1e-6): measured residual at
    n = 32 is 1.6e-10 at 240 iterations, 1.3e-7 at 200 and 4.6e-4 at
    150 — 240 is the smallest round budget that clears the gate with
    orders of margin (the qualitative corrcoef/rms comparison
    plateaus much earlier, by ~100).
    """
    mx = IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(n, (0.0, 1.0), periodic=False, name="y")
    dom = ImmersedDomain(lambda x, y: y < channel_width(x))
    grid = Grid((mx, my), immersed=dom)
    m_u = grid.immersed.fraction(mx.right * my.center)
    m_v = grid.immersed.fraction(mx.center * my.inner)
    registry = grid.dispatch
    gx = registry.resolve("diff", mx.center)["x"]
    gy = registry.resolve("diff", my.center)["y"]
    dx = registry.resolve("diff", mx.right)["x"]
    dirichlet = my.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    dy = registry.resolve("diff", dirichlet)["y"]

    def apply_a(p):
        fx = m_u * gx(p)
        fy = (m_v * gy(p)).retag(dirichlet)
        return dx(fx) + dy(fy)

    u = m_u  # the masked uniform inflow u = 1 on wet faces
    rhs = dx(u)
    cg = ConjugateGradient(apply_a, iterations=iterations)
    p = cg(rhs)
    u_new = u - m_u * gx(p)
    residual = dx(u_new) + dy(
        (-m_v * gy(p)).retag(dirichlet))
    return grid, np.asarray(u_new.data), np.asarray(residual.data)


def test_boundary_fitted_matches_the_immersed_staircase():
    # CS-D3 cross-check: the boundary-fitted projection against the
    # ImmersedDomain staircase reference — qualitative (same
    # large-scale pattern, both divergence-free), not a convergence
    # comparison. The staircase u is compared at the physical
    # positions of its wet cells, away from the masked boundary.
    n = 32
    mx = IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(n, (0.0, 1.0), periodic=False, name="y")
    mapping = CoordinateMapping(
        maps={"yp": lambda y, YN: y * YN},
        params={"YN": channel_width})
    grid = Grid((mx, my), mapping=mapping)
    space = mx.center * my.center
    solver = MappedPressureSolver(grid, space, iterations=20)
    u0 = grid.create_field(
        mx.right * my.center,
        init=lambda x, y: 1.0 + 0.0 * x * y)
    v0 = grid.create_field(
        mx.center * my.nodal(NodeSet.INNER, bc=BC.DIRICHLET),
        init=lambda x, y: 0.0 * x * y)
    div = solver.divergence({"x": u0, "y": v0})
    p = solver.solve(div)
    corr = solver.velocity_correction(p)
    u_fit = np.asarray((u0 - corr["x"].retag(u0)).data)

    _, u_stair, residual = masked_projection(n)
    # both flows are divergence-free to their solver budgets
    assert np.abs(residual).max() < 1e-6

    x_faces = (np.arange(n) + 1.0) * (TWO_PI / n)
    centres = (np.arange(n) + 0.5) / n
    widths = np.asarray(channel_width(jnp.asarray(x_faces)))
    fitted, staircase = [], []
    for i in range(n):
        y_phys = centres * widths[i]  # the mapped u's positions
        # staircase wet centers, clear of the masked boundary
        wet = centres < 0.8 * widths[i]
        fitted.append(np.interp(centres[wet], y_phys, u_fit[i]))
        staircase.append(u_stair[i][wet])
    fitted = np.concatenate(fitted)
    staircase = np.concatenate(staircase)
    corrcoef = np.corrcoef(fitted, staircase)[0, 1]
    rms = (np.sqrt(np.mean((fitted - staircase) ** 2))
           / np.sqrt(np.mean(staircase ** 2)))
    assert corrcoef > 0.95
    assert rms < 0.1


# ================================================================
#  jit gates: compile-once across steps
# ================================================================
def test_mapped_second_advance_compiles_nothing(compile_counter):
    model = make_mapped_flat_model(dt=DT, dsqr=DSQR,
                                   advection=False)
    ver = (np.arange(N) + 0.5) / N
    _, _, z = np.meshgrid(ver, ver, ver, indexing="ij")
    model.set_fields(b=0.01 * np.cos(np.pi * z))
    model.advance(4)
    compile_counter.reset()
    model.advance(4)
    assert compile_counter.count == 0


# ================================================================
#  Multi-device: the mapped projection under decomposition
# ================================================================
@pytest.mark.multi_device
def test_mapped_projection_is_device_count_invariant():
    # to tight rounding, not bitwise (the spherical-shallow-water
    # precedent): the sharded metric-scaled stencil chains compile
    # to per-shard fusion patterns whose last-ulp rounding differs
    # between the 1- and 4-device programs, and the CG dot products
    # reduce across shards in a different order.
    #
    # The solve is jitted: eagerly it costs ~85x more per call on
    # forced-4 (16.4 s vs 194 ms). Jitting it was impossible before
    # the lax.scan conversion (ROADMAP 3.6) -- the forced-4 program
    # did not finish compiling.
    def run(device_ids):
        n = 16
        mx = IntervalMesh(n, (0.0, TWO_PI), periodic=True,
                          name="x")
        ms = IntervalMesh(n, (0.0, 1.0), periodic=False,
                          name="sigma")
        mapping = CoordinateMapping(
            maps={"zp": lambda sigma, H: sigma * H},
            params={"H": depth})
        grid = Grid((mx, ms), mapping=mapping,
                    device_ids=device_ids)
        space = mx.center * ms.center
        solver = MappedPressureSolver(
            grid, space, iterations=12,
            weights={"sigma": 1.0 / DSQR})
        u = grid.random.normal(mx.right * ms.center, seed=4)
        w = grid.random.normal(
            mx.center * ms.nodal(NodeSet.INNER, bc=BC.DIRICHLET),
            seed=5)
        div = solver.divergence({"x": u, "sigma": w})
        p = jax.jit(solver.solve)(div)
        corr = solver.velocity_correction(p)
        after = solver.divergence({
            "x": u - corr["x"].retag(u),
            "sigma": w - corr["sigma"].retag(w)})
        return (np.asarray(p.data), np.asarray(after.data),
                float(jnp.abs(div.data).max()))

    p4, after4, scale4 = run(None)
    p1, _after1, _scale1 = run((0,))
    np.testing.assert_allclose(p4, p1, rtol=0.0, atol=1e-11)
    assert np.abs(after4).max() < 1e-9 * scale4
