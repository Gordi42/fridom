"""Numerical guard for the S2 pressure-solve reframe.

The ``SpectralPressureSolver`` inverts the ``dsqr``-weighted discrete
Laplacian by delegating to ``fr``'s ``SpectralSolve`` (the realized-map
composition ``backward @ inverse @ forward``, S2). The bitwise-identity
of that composition to the retired imperative body is now pinned in the
framework's own ``test_spectral_solve``; here we pin the *physics*: the
solve drives the discrete divergence to machine zero and lands in the
mean-free gauge.

The walled sections (C7) pin the bounded-axis path: the solve runs on
the Neumann-tagged sibling of the divergence space (the DCT-II /
Cosine-II pressure parity at rigid lids), reproduces manufactured
discrete eigenfunctions to machine precision, projects a hand-made
staggered velocity divergence-free, and keeps the mean gauge — while
the periodic path stays retag-free.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.nonhydro2.modules.pressure import SpectralPressureSolver
from fridom.spatial.bc import BC
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.composed import (
    Divergence,
    Gradient,
    Laplacian,
)
from fridom.spatial.spaces.nodal import NodeSet

N = 8
NY = 6
LZ = 1.0


def make_grid(n=N, length=2 * np.pi):
    return Grid(tuple(
        IntervalMesh(n, (0.0, length), periodic=True, name=name)
        for name in ("x", "y", "z")))


def make_walled_grid(n=N, ny=NY, lz=LZ):
    # x, y periodic; z bounded (rigid lids)
    mx = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="x")
    my = IntervalMesh(ny, (0.0, 2 * np.pi), periodic=True, name="y")
    mz = IntervalMesh(n, (0.0, lz), periodic=False, name="z")
    return Grid((mx, my, mz)), (mx, my, mz)


def _khat(k, dx):
    """Return the staggered-difference magnitude 2 sin(k dx/2)/dx."""
    return 2.0 * np.sin(k * dx / 2.0) / dx


# ================================================================
#  Periodic grid (regression: the walled seam must not change it)
# ================================================================
def test_pressure_solve_drives_divergence_to_zero():
    # solving lap(p) = div and reapplying the weighted Laplacian must
    # recover div to machine zero -- i.e. div(u* - grad p) == 0, the
    # incompressibility constraint the projection enforces
    grid = make_grid()
    div = grid.create_field(
        init=lambda x, y, z: jnp.sin(x) * jnp.cos(2 * y) * jnp.cos(z))
    dsqr = jnp.asarray(0.25)
    solver = SpectralPressureSolver(
        grid, div.function_space, vertical="z")
    p = solver.solve(div, dsqr=dsqr)
    weighted_lap = Laplacian(metric={"z": 1.0 / dsqr}).expand(
        div.function_space.bare, grid).scalar()
    residual = weighted_lap(p) - div
    maxdiff = float(jnp.abs(residual.data).max())
    assert maxdiff < 1e-12


def test_single_precision_solve_matches_full_within_tolerance():
    # the single-precision pressure solve returns float64 pressure
    # close to the full solve; the transform pipeline runs in c64/f32
    grid = make_grid()
    div = grid.create_field(
        init=lambda x, y, z: jnp.sin(x) * jnp.cos(2 * y) * jnp.cos(z))
    dsqr = jnp.asarray(0.25)
    full = SpectralPressureSolver(grid, div.function_space, vertical="z")
    low = SpectralPressureSolver(
        grid, div.function_space, vertical="z", single_precision=True)
    assert low._single_precision is True
    p_full = full.solve(div, dsqr=dsqr)
    p_low = low.solve(div, dsqr=dsqr)
    assert p_low.dtype == p_full.dtype  # state precision preserved
    rel = float(jnp.linalg.norm(p_low.data - p_full.data)
                / jnp.linalg.norm(p_full.data))
    assert rel < 1e-5


def test_single_precision_default_off():
    grid = make_grid()
    div = grid.create_field(
        init=lambda x, y, z: jnp.sin(x) * jnp.cos(y) * jnp.cos(z))
    solver = SpectralPressureSolver(grid, div.function_space, vertical="z")
    assert solver._single_precision is False


def test_pressure_solve_is_mean_free():
    # the k = 0 nullspace is regularized to the mean-free gauge
    grid = make_grid()
    div = grid.create_field(
        init=lambda x, y, z: jnp.sin(x) * jnp.cos(2 * y) * jnp.cos(z))
    solver = SpectralPressureSolver(
        grid, div.function_space, vertical="z")
    p = solver.solve(div, dsqr=jnp.asarray(1.0))
    assert float(jnp.abs(p.mean().data.ravel()[0])) < 1e-13


def test_periodic_solver_never_retags(monkeypatch):
    # on a fully periodic grid the Neumann sibling IS the space itself
    # (interned identity): the solve path must not touch retag at all
    grid = make_grid()
    div = grid.create_field(
        init=lambda x, y, z: jnp.sin(x) * jnp.cos(2 * y) * jnp.cos(z))
    solver = SpectralPressureSolver(
        grid, div.function_space, vertical="z")
    assert solver._solve_space is div.function_space.bare

    def forbidden(self, target):  # noqa: ARG001
        raise AssertionError("retag must not run on a periodic grid")

    monkeypatch.setattr(ScalarField, "retag", forbidden)
    p = solver.solve(div, dsqr=jnp.asarray(0.25))
    assert p.function_space.bare is div.function_space.bare


# ================================================================
#  Walled grid: manufactured discrete Poisson solutions
# ================================================================
@pytest.mark.parametrize(
    ("kx", "ky", "m"),
    [
        pytest.param(1, 0, 3, id="kx1-m3"),
        pytest.param(2, 1, 1, id="kx2-ky1-m1"),
        pytest.param(0, 0, 5, id="pure-vertical"),
        pytest.param(1, 2, 0, id="barotropic-m0"),
        pytest.param(3, 2, N - 1, id="top-vertical-mode"),
    ],
)
def test_walled_manufactured_poisson_solution(kx, ky, m):
    # p_exact = cos(kx x + ky y) cos(m pi z / Lz) is a discrete
    # eigenfunction of the walled Div @ Diag @ Grad chain; feeding
    # rhs = lambda * p_exact through the solver must reproduce
    # p_exact to machine precision, with the exact trig eigenvalue
    # k_hat_z = 2 sin(pi m dz / (2 Lz)) / dz on the vertical axis
    grid, (mx, my, mz) = make_walled_grid()
    dsqr = jnp.asarray(0.25)
    dx, dy, dz = 2 * np.pi / N, 2 * np.pi / NY, LZ / N
    lam = -(_khat(kx, dx) ** 2 + _khat(ky, dy) ** 2
            + _khat(m * np.pi / LZ, dz) ** 2 / float(dsqr))
    space = mx.center * my.center * mz.center  # BC-free, as div comes

    def make_init(kx, ky, m):
        def init(x, y, z):
            return (jnp.cos(kx * x + ky * y)
                    * jnp.cos(m * np.pi * z / LZ))
        return init

    p_exact = grid.create_field(space, init=make_init(kx, ky, m))
    rhs = p_exact * lam
    solver = SpectralPressureSolver(
        grid, rhs.function_space, vertical="z")
    p = solver.solve(rhs, dsqr=dsqr)
    # the solution comes back on the caller's BC-free space
    assert p.function_space.bare is space
    assert float(jnp.abs(p.data - p_exact.data).max()) < 1e-12


def test_walled_solver_runs_on_the_neumann_sibling():
    grid, (mx, my, mz) = make_walled_grid()
    space = mx.center * my.center * mz.center
    solver = SpectralPressureSolver(grid, space, vertical="z")
    tagged = space.replace(
        z=mz.nodal(NodeSet.CENTER, bc=BC.NEUMANN))
    assert solver._solve_space is tagged
    assert solver._solve_space is not space


# ================================================================
#  Walled grid: the projection acceptance gate (no model)
# ================================================================
def test_walled_projection_drives_divergence_to_zero():
    # mirror core.py's _project on a hand-made staggered state:
    # u, v on the periodic faces, w on Inner(DIRICHLET) (the rigid-lid
    # wall-normal velocity), random interior data; after subtracting
    # the weighted pressure gradient the discrete divergence must be
    # machine zero relative to the original divergence
    grid, (mx, my, mz) = make_walled_grid()
    dsqr = jnp.asarray(0.25)
    u = grid.random.normal(mx.right * my.center * mz.center, seed=1)
    v = grid.random.normal(mx.center * my.right * mz.center, seed=2)
    w = grid.random.normal(
        mx.center * my.center
        * mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET), seed=3)
    vel = VectorField({"u": u, "v": v, "w": w})
    div = Divergence()(vel)
    solver = SpectralPressureSolver(
        div.grid, div.function_space, vertical="z")
    p = solver.solve(div, dsqr=dsqr)
    grad = Gradient()(p)
    # the vertical gradient lands on the BC-free Inner sibling;
    # adopt w's Dirichlet tag before the subtraction
    dp_dz = grad["z"].retag(w.function_space.bare)
    vel_new = VectorField({
        "u": u - grad["x"],
        "v": v - grad["y"],
        "w": w - dp_dz / dsqr,
    })
    residual = Divergence()(vel_new)
    rel = (float(jnp.abs(residual.data).max())
           / float(jnp.abs(div.data).max()))
    assert rel < 1e-12


# ================================================================
#  Walled grid: the mean gauge (the (0, 0, 0) structural zero)
# ================================================================
def test_walled_solve_lands_in_the_mean_free_gauge():
    grid, (mx, my, mz) = make_walled_grid()
    space = mx.center * my.center * mz.center
    rhs = grid.random.normal(space, seed=4)
    solver = SpectralPressureSolver(grid, space, vertical="z")
    p = solver.solve(rhs, dsqr=jnp.asarray(1.0))
    assert float(jnp.abs(p.mean().data.ravel()[0])) < 1e-13


def test_walled_constant_rhs_is_annihilated():
    # the constant mode is the operator's nullspace: the inverse
    # symbol is regularized to zero there, so a constant rhs returns
    # (numerically) zero pressure instead of an amplified blow-up
    grid, (mx, my, mz) = make_walled_grid()
    space = mx.center * my.center * mz.center
    rhs = grid.create_field(
        space, init=lambda x, y, z: jnp.ones_like(x))  # noqa: ARG005
    solver = SpectralPressureSolver(grid, space, vertical="z")
    p = solver.solve(rhs, dsqr=jnp.asarray(1.0))
    assert float(jnp.abs(p.data).max()) < 1e-13


# ================================================================
#  Project-state == project-tendency (parity audit, Sketch A)
# ================================================================
def _rel_l2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a) + np.linalg.norm(b)
    return 0.0 if denom == 0.0 else 2.0 * np.linalg.norm(a - b) / denom


@pytest.mark.parametrize("sign", [1.0, -1.0],
                         ids=["forward", "backward"])
def test_project_state_equals_project_tendency(sign):
    # the Sketch-A exact-equivalence regression (parity audit rows
    # 1 and 2): for an explicit one-stage scheme (AB1, u* = u + dt F)
    # started from a divergence-free state, project-the-state
    # (production, CONSTRAINT stage) coincides with the old stack's
    # project-the-tendency: div u = 0 => div u* = dt div F =>
    # phi = dt psi => u* - grad phi == u + dt (F - grad psi). The
    # stored diagnostic must be the NORMALIZED pressure p = phi/dt
    # = psi (dt-independent); the backward leg (dt < 0) pins the
    # sign convention: phi flips with stage_dt, psi does not, so the
    # stored p keeps its physical sign and value.
    dt = sign * 0.02
    dsqr = 0.25  # non-unit so the 1/dsqr vertical weighting bites
    grid = make_grid()
    model = nh.Model(grid=grid, dt=dt, dsqr=dsqr, advection=False,
                     time_stepper=AdamBashforth(dt, order=1))
    ax = (np.arange(N) + 0.5) * (2 * np.pi / N)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    # exactly divergence-free IC: u varies only along y, v only
    # along x (both centered on those axes), w = 0; b varies in x
    # AND z so the projected w-tendency is genuinely nonzero (a
    # horizontally uniform b is hydrostatic: w would stay at
    # roundoff and the relative comparison would be noise-vs-noise)
    model.set_fields(u=0.01 * np.sin(y), v=0.01 * np.sin(x),
                     b=0.01 * np.cos(x) * np.cos(z))
    state0 = model.state
    # ---- path 2 (reference): project the tendency by hand --------
    tend = model.tendency(state0, constraints=False)
    div = Divergence()(VectorField(
        {c: tend[c] for c in ("u", "v", "w")}))
    solver = SpectralPressureSolver(
        div.grid, div.function_space, vertical="z")
    psi = solver.solve(div, dsqr=jnp.asarray(dsqr))
    grad = Gradient()(psi)
    ref = {
        "u": state0["u"] + (tend["u"]
                            - grad["x"].retag(state0["u"])) * dt,
        "v": state0["v"] + (tend["v"]
                            - grad["y"].retag(state0["v"])) * dt,
        "w": state0["w"] + (tend["w"]
                            - grad["z"].retag(state0["w"]) / dsqr) * dt,
    }
    # ---- path 1 (production): one composed step ------------------
    model.advance(1)
    # tolerance rationale: the two paths are distinct compiled
    # programs (two spectral solves in a different order), so per the
    # bitwise-equality umbrella the comparison is tolerance-based;
    # 1e-12 relative is comfortable for a single step
    for c in ("u", "v", "w"):
        assert _rel_l2(model.state[c].data, ref[c].data) < 1e-12
    # the direct pin of the p = phi/stage_dt normalization
    assert _rel_l2(model.state["p"].data, psi.data) < 1e-12
