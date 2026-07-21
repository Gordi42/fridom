r"""Solver-level tests for the terrain barotropic pressure solve.

Prefix-mirrored shard of ``hy.modules.barotropic_pressure`` (the
``BarotropicPressureSolver`` of the volume-exact implicit free surface,
GM-D1 option 1 / GM-D2). These exercise the SPD flux-form operator, its
flat mean-depth spectral preconditioner and the walled-axis closure
(GM-D6) directly, without the full model — the walled-channel hydrostatic
**model** does not assemble (a pre-existing package-wide horizontal-wall
gap in the core and velocity staggering), so the wall closure is verified
at the solver level. The model-level gates (cancellation, volume,
autodiff) live in ``test_free_surface_terrain_implicit.py``.
Self-contained per the AGENTS oversized-module rule.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.hydrostatic.modules.barotropic_pressure import (
    BarotropicPressureSolver,
    _dirichlet_mid,
    _neumann_sibling,
)
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.operators.base import resolve_codomain
from fridom.spatial.operators.integrate import Integral

IM = fr.spatial.meshes.IntervalMesh
JN = "dzp_dz"
CSQR, DT = jnp.asarray(3.0), jnp.asarray(0.05)


# ================================================================
#  Builders (self-contained per the AGENTS oversized-module rule)
# ================================================================
def terrain_grid(n, nz=8, a=0.4, y_periodic=True):
    """Return a doubly-periodic (or y-walled) sigma grid, depth H(x)."""
    def depth(x, y):  # noqa: ARG001 — H varies along x only
        return 1.0 + a * jnp.sin(2 * jnp.pi * x)
    return fr.spatial.Grid((
        IM(n, (0.0, 1.0), periodic=True, name="x"),
        IM(n, (0.0, 1.0), periodic=y_periodic, name="y"),
        IM(nz, (-1.0, 0.0), periodic=False, name="z")),
        mapping=CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                  params={"H": depth}))


def ps_space_of(grid):
    """Return the 2D barotropic ``Profile(x, y)`` pressure space."""
    return fr.spatial.Profile("x", "y").resolve(grid).bare


def make_solver(grid, *, eps=0.0, iterations=40, tolerance=None,
                preconditioner="spectral", multigrid_levels=None):
    """Return a solver on the grid's ps space (tight-budget gates)."""
    return BarotropicPressureSolver(
        grid, ps_space_of(grid), ("zp", "z"), "z",
        epsilon=eps, iterations=iterations,
        tolerance=tolerance, preconditioner=preconditioner,
        multigrid_levels=multigrid_levels)


def mean_free(field):
    """Return the plain-mean-removed field (the constants gauge)."""
    return field - field.mean()


def checkerboard(grid, black):
    """Return the {0, 1} red/black indicator on the ps cell (parity)."""
    template = grid.create_field(ps_space_of(grid))
    shape = template.data.shape
    ix, iy = np.indices(shape[:2])
    parity = ((ix + iy) % 2 == 0)
    if not black:
        parity = ~parity
    mask = parity.reshape(shape[:2] + (1,) * (len(shape) - 2))
    return template.with_data(jnp.asarray(
        np.broadcast_to(mask.astype(float), shape)))


def rand_ps(grid, seed):
    """Return a random field on the ps cell."""
    rng = np.random.default_rng(seed)
    template = grid.create_field(ps_space_of(grid))
    return template.with_data(jnp.asarray(
        rng.standard_normal(template.data.shape)))


def inner(a, b):
    """Return the measure-weighted inner product ``int a b dV``."""
    return float(jnp.sum((a * b).integrate().data))


def wall_transport(grid, seed=0):
    r"""Return ``(u, v, raw_T)`` for a wall-closed transport divergence.

    The hydrostatic model cannot build wall-tagged horizontal velocities,
    so this reproduces the terrain transport divergence
    ``T = int[D_x(Ju) + D_y(Jv)] dz`` with the Dirichlet wall closure
    (the same tagged divergence leg the operator uses) on random
    velocities laid on the collocated-difference faces. The Dirichlet tag
    is the identity on a periodic axis, so this also serves a
    doubly-periodic grid.
    """
    coll = fr.spatial.Collocated().resolve(grid)
    reg = grid.dispatch
    rng = np.random.default_rng(seed)
    faces, div, vel = {}, {}, {}
    for ax in ("x", "y"):
        g = reg.resolve("diff", coll.factor(ax))[ax]
        face = resolve_codomain(g, coll)
        faces[ax] = face
        tagged = _dirichlet_mid(face.bare, ax)
        div[ax] = (tagged, reg.resolve("diff", tagged.factor(ax))[ax])
        template = grid.create_field(face)
        vel[ax] = template.with_data(jnp.asarray(
            rng.standard_normal(template.data.shape)))

    def raw_t(uu, vv):
        ju = uu * grid.metric(uu.function_space.bare, JN)
        jv = vv * grid.metric(vv.function_space.bare, JN)
        tx, dx = div["x"]
        ty, dy = div["y"]
        return Integral()["z"](dx(ju.retag(tx)) + dy(jv.retag(ty)))

    return vel["x"], vel["y"], raw_t


# ================================================================
#  Properties
# ================================================================
def test_properties():
    solver = make_solver(terrain_grid(8), eps=1.0, iterations=17,
                         tolerance=1e-7)
    assert solver.axes == ("x", "y")
    assert solver.epsilon == 1.0
    assert solver.iterations == 17
    assert solver.tolerance == 1e-7


# ================================================================
#  GB-3 (self-adjointness): <A p, q> == <p, A q>
# ================================================================
@pytest.mark.parametrize("eps", [0.0, 1.0])
@pytest.mark.parametrize("a", [0.0, 0.4, 0.8])
def test_operator_is_self_adjoint(a, eps):
    grid = terrain_grid(16, a=a)
    solver = make_solver(grid, eps=eps)
    op = solver.operator(gravity=CSQR, dt=DT)
    p, q = rand_ps(grid, 1), rand_ps(grid, 2)
    apq, paq = inner(op(p), q), inner(p, op(q))
    assert abs(apq - paq) <= 1e-12 * abs(apq)


@pytest.mark.parametrize("eps", [0.0, 1.0])
def test_walled_operator_is_self_adjoint(eps):
    grid = terrain_grid(16, a=0.4, y_periodic=False)
    solver = make_solver(grid, eps=eps)
    op = solver.operator(gravity=CSQR, dt=DT)
    p, q = rand_ps(grid, 3), rand_ps(grid, 4)
    apq, paq = inner(op(p), q), inner(p, op(q))
    assert abs(apq - paq) <= 1e-12 * abs(apq)


# ================================================================
#  Convergence: the solve inverts its own operator
# ================================================================
def test_solve_inverts_operator():
    grid = terrain_grid(16, a=0.4)
    solver = make_solver(grid, eps=1.0, iterations=40, tolerance=None)
    rhs = rand_ps(grid, 5)
    ps = solver.solve(rhs, gravity=CSQR, dt=DT)
    residual = rhs - solver.operator(gravity=CSQR, dt=DT)(ps)
    rel = np.sqrt(inner(residual, residual)) / np.sqrt(inner(rhs, rhs))
    assert rel < 1e-11


def test_flat_chart_preconditioner_is_exact():
    # on an a=0 chart H_a == 1 and Hbar == 1, so the flat mean-depth
    # preconditioner is the exact inverse and PCG converges in ~1 step
    grid = terrain_grid(16, a=0.0)
    solver = make_solver(grid, eps=1.0, iterations=30, tolerance=1e-8)
    cg = solver.krylov(gravity=CSQR, dt=DT)
    _ps, info = cg.solve(rand_ps(grid, 6))
    assert int(info["iterations"]) <= 2


def test_flat_chart_depths_are_unit():
    grid = terrain_grid(16, a=0.0)
    solver = make_solver(grid, eps=1.0)
    hbar = float(solver._mean_depth())
    h_x = np.asarray(solver._face_depth("x").data)
    assert hbar == pytest.approx(1.0, abs=1e-13)
    assert np.allclose(h_x, 1.0, atol=1e-13)


# ================================================================
#  GB-4 (walls, GM-D6): exact cancellation on a walled sigma channel
# ================================================================
@pytest.mark.parametrize(
    "y_periodic", [pytest.param(True, id="periodic"),
                   pytest.param(False, id="walled")])
def test_transport_divergence_cancels(y_periodic):
    grid = terrain_grid(16, a=0.4, y_periodic=y_periodic)
    solver = make_solver(grid, eps=0.0, iterations=40, tolerance=None)
    u, v, raw_t = wall_transport(grid, seed=7)
    t0 = raw_t(u, v)
    pre = float(jnp.abs(t0.data).max())
    rhs = -DT * CSQR * t0
    ps = solver.solve(rhs, gravity=CSQR, dt=DT)
    # the z-uniform correction (the same C-grid gradient the operator
    # legs use), broadcast onto the collocated-difference velocity faces
    u_new = u - DT * ps.diff("x").to(u)
    v_new = v - DT * ps.diff("y").to(v)
    post = float(jnp.abs(raw_t(u_new, v_new).data).max())
    assert post <= 1e-13 * pre


# ================================================================
#  Rigid lid (eps=0): the solution is mean-free (the plain gauge, GM-D7)
# ================================================================
def test_rigid_lid_solution_is_mean_free():
    grid = terrain_grid(16, a=0.4)
    solver = make_solver(grid, eps=0.0, iterations=40, tolerance=None)
    u, v, raw_t = wall_transport(grid, seed=9)
    rhs = -DT * CSQR * raw_t(u, v)
    ps = solver.solve(rhs, gravity=CSQR, dt=DT)
    mean = float(ps.mean().data.ravel()[0])
    scale = float(jnp.abs(ps.data).max())
    assert abs(mean) <= 1e-12 * scale


# ================================================================
#  Warm start (the x0 seam, Phase E semantics)
# ================================================================
@pytest.mark.parametrize("eps", [0.0, 1.0])
def test_warm_start_matches_cold_start(eps):
    grid = terrain_grid(16, a=0.4)
    solver = make_solver(grid, eps=eps, iterations=40, tolerance=None)
    u, v, raw_t = wall_transport(grid, seed=8)
    rhs = -DT * CSQR * raw_t(u, v)
    cold = solver.solve(rhs, gravity=CSQR, dt=DT)
    warm = solver.solve(rhs, rand_ps(grid, 12), gravity=CSQR, dt=DT)
    diff = float(jnp.abs((warm - cold).data).max())
    scale = float(jnp.abs(cold.data).max())
    assert diff <= 1e-11 * scale


def test_warm_start_rigid_lid_stays_mean_free():
    # a non-mean-free guess must not shift the eps=0 gauge (CG projects
    # the guess at the three sites)
    grid = terrain_grid(16, a=0.4)
    solver = make_solver(grid, eps=0.0, iterations=40, tolerance=None)
    u, v, raw_t = wall_transport(grid, seed=8)
    rhs = -DT * CSQR * raw_t(u, v)
    x0 = rand_ps(grid, 13)
    x0 = x0.with_data(x0.data + 5.0)       # a large nonzero mean
    ps = solver.solve(rhs, x0, gravity=CSQR, dt=DT)
    mean = float(ps.mean().data.ravel()[0])
    assert abs(mean) <= 1e-12 * float(jnp.abs(ps.data).max())


# ================================================================
#  Wall-closure siblings (the reproduced helpers)
# ================================================================
def test_neumann_sibling_is_identity_on_periodic():
    space = ps_space_of(terrain_grid(8))
    assert _neumann_sibling(space) is space


def test_neumann_sibling_tags_a_walled_axis():
    space = ps_space_of(terrain_grid(8, y_periodic=False))
    sib = _neumann_sibling(space)
    assert sib is not space
    assert not sib.factor("y").bc.is_free


# ================================================================
#  Phase C: the point-Jacobi multigrid preconditioner (GM-D3/D4/D7)
# ================================================================
def test_rejects_unknown_preconditioner():
    with pytest.raises(ValueError, match="preconditioner must be one of"):
        make_solver(terrain_grid(8), preconditioner="jacobi")


def test_multigrid_levels_default_is_floor_limited():
    # None coarsens x, y to the four-cell floor: nx=16 -> 8 -> 4, a
    # three-level V-cycle (the vertical stays full, MG-D3)
    solver = make_solver(terrain_grid(16, a=0.8), preconditioner="multigrid")
    assert solver._multigrid_levels is None
    vcycle = solver._build_vcycle(gravity=CSQR, dt=DT)
    assert len(vcycle.levels) == 3


def test_multigrid_levels_int_caps_the_depth():
    solver = make_solver(terrain_grid(16, a=0.8), preconditioner="multigrid",
                         multigrid_levels=2)
    vcycle = solver._build_vcycle(gravity=CSQR, dt=DT)
    assert len(vcycle.levels) == 2


def test_vcycle_smoother_is_point_jacobi_and_coarse_sweeps_eight():
    # GM-D4: the isotropic 2-D solve uses a damped point-Jacobi smoother
    # at omega=2/3, symmetric V(1, 1), coarse_sweeps=8 (the engine
    # defaults kept)
    solver = make_solver(terrain_grid(16, a=0.8), preconditioner="multigrid")
    vcycle = solver._build_vcycle(gravity=CSQR, dt=DT)
    assert vcycle.pre_sweeps == vcycle.post_sweeps == 1
    assert vcycle.coarse_sweeps == 8
    assert all(level.smoother.omega == pytest.approx(2.0 / 3.0)
               for level in vcycle.levels)
    assert all(type(level.smoother).__name__ == "DampedJacobi"
               for level in vcycle.levels)


# ---- diagonal exactness vs the 2-colour (red/black) probe --------
@pytest.mark.parametrize("eps", [0.0, 1.0])
def test_diagonal_matches_two_colour_probe(eps):
    # apply A to a red/black checkerboard: a black cell's neighbours are
    # all red (the five-point stencil), so A(black)[black] == diag[black];
    # the two colours together recover the whole diagonal exactly
    grid = terrain_grid(16, a=0.8)
    solver = make_solver(grid, eps=eps)
    op = solver.operator(gravity=CSQR, dt=DT)
    diag = np.asarray(solver.diagonal(gravity=CSQR, dt=DT).data)
    black = checkerboard(grid, black=True)
    a_black = np.asarray(op(black).data)
    a_red = np.asarray(op(checkerboard(grid, black=False)).data)
    recon = np.where(np.asarray(black.data) > 0.5, a_black, a_red)
    assert np.abs(recon - diag).max() <= 1e-12 * np.abs(diag).max()


# ---- V-cycle SPD symmetry: <M r, s> == <r, M s> (on the range) ----
@pytest.mark.parametrize("eps", [0.0, 1.0])
def test_vcycle_is_self_adjoint(eps):
    # the fixed symmetric V(1, 1) with adjoint transfers and a symmetric
    # smoother is a symmetric operator on A's range (the mean-free
    # residuals CG feeds it — for eps=0 the constants are the nullspace)
    grid = terrain_grid(16, a=0.8)
    solver = make_solver(grid, eps=eps, preconditioner="multigrid")
    vcycle = solver._build_vcycle(gravity=CSQR, dt=DT)
    r, s = mean_free(rand_ps(grid, 30)), mean_free(rand_ps(grid, 31))
    mrs, rms = inner(vcycle(r), s), inner(r, vcycle(s))
    assert abs(mrs - rms) <= 1e-12 * max(abs(mrs), 1e-30)


# ---- convergence: mg hits tol within budget; matches spectral -----
def test_multigrid_converges_and_matches_spectral():
    grid = terrain_grid(64, a=0.8)
    rhs = mean_free(rand_ps(grid, 40))
    sp = make_solver(grid, eps=1.0, iterations=60, tolerance=1e-8)
    mg = make_solver(grid, eps=1.0, iterations=60, tolerance=1e-8,
                     preconditioner="multigrid")
    ps_sp, info_sp = sp.krylov(gravity=CSQR, dt=DT).solve(rhs)
    ps_mg, info_mg = mg.krylov(gravity=CSQR, dt=DT).solve(rhs)
    # the mg-preconditioned CG hits 1e-8 well inside the 60-iteration
    # budget on a steep chart (spike: 11-13 flat; spectral degrades)
    assert int(info_mg["iterations"]) < int(info_sp["iterations"])
    assert int(info_mg["iterations"]) <= 15
    rel = (float(jnp.abs((ps_mg - ps_sp).data).max())
           / float(jnp.abs(ps_sp.data).max()))
    assert rel <= 1e-6


# ---- eps=0 gauge: mean-free ps under a non-mean-free warm start ----
def test_multigrid_rigid_lid_is_mean_free_from_biased_start():
    grid = terrain_grid(16, a=0.8)
    solver = make_solver(grid, eps=0.0, iterations=40, tolerance=None,
                         preconditioner="multigrid")
    u, v, raw_t = wall_transport(grid, seed=41)
    rhs = -DT * CSQR * raw_t(u, v)
    x0 = rand_ps(grid, 42)
    x0 = x0.with_data(x0.data + 6.0)           # a large nonzero mean
    ps = solver.solve(rhs, x0, gravity=CSQR, dt=DT)
    mean = float(ps.mean().data.ravel()[0])
    assert abs(mean) <= 1e-12 * float(jnp.abs(ps.data).max())
