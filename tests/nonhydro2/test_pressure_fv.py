"""Walled finite-volume pressure solve (plan stage F4, FV-D4).

The FV twin of ``test_pressure.py``'s walled section: the spectral
pressure solve on the ``CellAvg`` cell-average family with rigid lids.
The solve runs on the **Neumann-tagged average sibling** of the
divergence space (the DCT-II / Cosine-II pressure parity at the walls,
the ``CellAvg`` origin now carrying the walled transform), reproduces
manufactured discrete eigenfunctions to machine precision, projects a
hand-made staggered FV velocity divergence-free (walled-z AND a
walled-x channel), and keeps the periodic path retag-free — exactly as
the nodal solve does, since the 2nd-order FV and nodal stencils are the
same numbers (scoping study §1).
"""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.nonhydro2.modules.core import fv_cgrid_overrides
from fridom.nonhydro2.modules.pressure import SpectralPressureSolver
from fridom.spatial.bc import BC
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.composed import Divergence, Gradient
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.nodal import NodeSet

N = 8
NY = 6
LZ = 1.0


def _fv_grid(meshes):
    """Grid carrying the FV C-grid diff overrides (what a model merges)."""
    grid = Grid(meshes)
    grid.merge_overrides(fv_cgrid_overrides(grid.factors))
    return grid


def make_walled_grid(n=N, ny=NY, lz=LZ):
    # x, y periodic; z bounded (rigid lids)
    mx = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="x")
    my = IntervalMesh(ny, (0.0, 2 * np.pi), periodic=True, name="y")
    mz = IntervalMesh(n, (0.0, lz), periodic=False, name="z")
    return _fv_grid((mx, my, mz)), (mx, my, mz)


def make_periodic_grid(n=N, length=2 * np.pi):
    meshes = tuple(
        IntervalMesh(n, (0.0, length), periodic=True, name=name)
        for name in ("x", "y", "z"))
    return _fv_grid(meshes), meshes


def _khat(k, dx):
    """Return the staggered-difference magnitude 2 sin(k dx/2)/dx."""
    return 2.0 * np.sin(k * dx / 2.0) / dx


# ================================================================
#  Walled grid: manufactured discrete Poisson solutions (gate 1)
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
def test_walled_fv_manufactured_poisson_solution(kx, ky, m):
    # p_exact = cos(kx x + ky y) cos(m pi z / Lz) is a discrete
    # eigenfunction of the walled FV Div @ Diag @ Grad chain (the FV
    # stencils are bitwise the nodal ones); feeding rhs = lambda *
    # p_exact reproduces p_exact to machine precision, with the exact
    # trig eigenvalue k_hat_z = 2 sin(pi m dz / (2 Lz)) / dz on z
    grid, (mx, my, mz) = make_walled_grid()
    dsqr = jnp.asarray(0.25)
    dx, dy, dz = 2 * np.pi / N, 2 * np.pi / NY, LZ / N
    lam = -(_khat(kx, dx) ** 2 + _khat(ky, dy) ** 2
            + _khat(m * np.pi / LZ, dz) ** 2 / float(dsqr))
    space = mx.cell_avg * my.cell_avg * mz.cell_avg  # BC-free, as div

    def init(x, y, z):
        return jnp.cos(kx * x + ky * y) * jnp.cos(m * np.pi * z / LZ)

    p_exact = grid.create_field(space, init=init)
    rhs = p_exact * lam
    solver = SpectralPressureSolver(
        grid, rhs.function_space, vertical="z")
    p = solver.solve(rhs, dsqr=dsqr)
    # the solution comes back on the caller's BC-free CellAvg space
    assert p.function_space.bare is space
    assert float(jnp.abs(p.data - p_exact.data).max()) < 1e-12


# ================================================================
#  Walled grid: the FV solver runs on the Neumann AVERAGE sibling
#  (gate 3)
# ================================================================
def test_walled_fv_solver_runs_on_the_neumann_average_sibling():
    grid, (mx, my, mz) = make_walled_grid()
    space = mx.cell_avg * my.cell_avg * mz.cell_avg
    solver = SpectralPressureSolver(grid, space, vertical="z")
    tagged = space.replace(z=mz.average(CellAvg, bc=BC.NEUMANN))
    assert solver._solve_space is tagged
    assert solver._solve_space is not space


def test_periodic_fv_solver_never_retags(monkeypatch):
    # on a fully periodic FV grid the average sibling IS the space
    # itself (interned identity): the solve must not touch retag
    grid, meshes = make_periodic_grid()
    mx, my, mz = meshes
    space = mx.cell_avg * my.cell_avg * mz.cell_avg
    solver = SpectralPressureSolver(grid, space, vertical="z")
    assert solver._solve_space is space

    def forbidden(self, target):  # noqa: ARG001
        raise AssertionError("retag must not run on a periodic FV grid")

    monkeypatch.setattr(ScalarField, "retag", forbidden)
    div = grid.create_field(
        space, init=lambda x, y, z: jnp.sin(x) * jnp.cos(2 * y) * jnp.cos(z))
    p = solver.solve(div, dsqr=jnp.asarray(0.25))
    assert p.function_space.bare is space


# ================================================================
#  Walled FV projection drives the discrete divergence to zero
#  (gate 2): a walled-z lid AND a walled-x channel
# ================================================================
def _project_divergence_free(vertical, u, v, w):
    """Mirror _project on a hand-made staggered FV state; return rel err."""
    dsqr = jnp.asarray(0.25)
    vel = VectorField({"u": u, "v": v, "w": w})
    div = Divergence()(vel)
    solver = SpectralPressureSolver(
        div.grid, div.function_space, vertical=vertical)
    p = solver.solve(div, dsqr=dsqr)
    grad = Gradient()(p)
    comps = {"u": u, "v": v, "w": w}
    new = {}
    for name, axis in (("u", "x"), ("v", "y"), ("w", "z")):
        g = grad[axis].retag(comps[name].function_space.bare)
        new[name] = comps[name] - (g / dsqr if axis == vertical else g)
    residual = Divergence()(VectorField(new))
    return (float(jnp.abs(residual.data).max())
            / float(jnp.abs(div.data).max()))


def test_walled_z_fv_projection_drives_divergence_to_zero():
    # rigid lid on z: u, v on the periodic faces (CellAvg transverse),
    # w on Inner(z, DIRICHLET) -- the FV wall-normal velocity
    grid, (mx, my, mz) = make_walled_grid()
    u = grid.random.normal(mx.right * my.cell_avg * mz.cell_avg, seed=1)
    v = grid.random.normal(mx.cell_avg * my.right * mz.cell_avg, seed=2)
    w = grid.random.normal(
        mx.cell_avg * my.cell_avg
        * mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET), seed=3)
    assert _project_divergence_free("z", u, v, w) < 1e-12


def test_walled_x_fv_channel_projection_drives_divergence_to_zero():
    # a lateral wall on x instead: u derives the Dirichlet tag on the
    # inner x faces; the axis-generic gradient retag closes the channel
    mx = IntervalMesh(N, (0.0, LZ), periodic=False, name="x")
    my = IntervalMesh(NY, (0.0, 2 * np.pi), periodic=True, name="y")
    mz = IntervalMesh(N, (0.0, 2 * np.pi), periodic=True, name="z")
    grid = _fv_grid((mx, my, mz))
    u = grid.random.normal(
        mx.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
        * my.cell_avg * mz.cell_avg, seed=4)
    v = grid.random.normal(mx.cell_avg * my.right * mz.cell_avg, seed=5)
    w = grid.random.normal(mx.cell_avg * my.cell_avg * mz.right, seed=6)
    # the vertical weight rides z here; use x as the "vertical" only for
    # the wall geometry -- keep the isotropic (dsqr=1 on all) form by
    # weighting none: reuse the helper with vertical="z" (periodic)
    assert _project_divergence_free("z", u, v, w) < 1e-12


# ================================================================
#  Walled FV mean gauge (the (0, 0, 0) structural zero)
# ================================================================
def test_walled_fv_solve_lands_in_the_mean_free_gauge():
    grid, (mx, my, mz) = make_walled_grid()
    space = mx.cell_avg * my.cell_avg * mz.cell_avg
    rhs = grid.random.normal(space, seed=7)
    solver = SpectralPressureSolver(grid, space, vertical="z")
    p = solver.solve(rhs, dsqr=jnp.asarray(1.0))
    assert float(jnp.abs(p.mean().data.ravel()[0])) < 1e-13
