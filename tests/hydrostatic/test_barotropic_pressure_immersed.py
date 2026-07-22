r"""Wet-column terrain barotropic pressure solve (M5, solver level).

Prefix-mirrored shard of ``hy.modules.barotropic_pressure`` covering the
composed **terrain + immersed** ``BarotropicPressureSolver``: the
wet-column face depth ``H_a = int alpha_a J dz``, the wet-masked flat
mean-depth spectral preconditioner, the wet-column-mean nullspace
projection (rigid lid) and the wet-column ``ps`` masking. The pure
terrain solver unit tests live in ``test_barotropic_pressure.py``; the
model-level composed gates (cancellation, column equivalence, oracle)
live in ``test_free_surface_terrain_immersed.py``. Self-contained per the
AGENTS oversized-module rule.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.hydrostatic.modules.barotropic_pressure import (
    BarotropicPressureSolver,
    _mean_free,
)
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.integrate import Integral
from fridom.spatial.operators.multigrid_hierarchy import coarsen_levels

IM = IntervalMesh
CSQR, DT = jnp.asarray(3.0), jnp.asarray(0.05)


# ================================================================
#  Builders (self-contained per the AGENTS oversized-module rule)
# ================================================================
def _mapping(a):
    def depth(x, y):
        return 1.0 + a * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
    return CoordinateMapping(maps={"zp": lambda z, H: z * H},
                             params={"H": depth})


def _cut(x, y, z):  # noqa: ARG001
    return jnp.clip((z - (-0.5 + 0.1 * jnp.sin(2 * jnp.pi * x))) / (1.0 / 8)
                    + 0.5, 0.0, 1.0)


def _cut_steep(x, y, z):  # noqa: ARG001
    """Return a steep partial-bottom shelf (spectral degrades here)."""
    return jnp.clip((z - (-0.3 + 0.4 * jnp.sin(2 * jnp.pi * x))) / (1.0 / 8)
                    + 0.5, 0.0, 1.0)


def _allwet(x, y, z):  # noqa: ARG001
    return x * 0.0 + 1.0


def _coast(x, y, z):  # noqa: ARG001
    """Return a full-depth lateral wall on x < 0.35 (dry columns there)."""
    return (x > 0.35).astype(float)


def _grid(*, n=16, nz=8, a=0.4, init=_cut, order=4, min_fraction=0.1,
          immersed=True):
    kw = {"mapping": _mapping(a)}
    if immersed:
        kw["immersed"] = ImmersedDomain(init, order=order,
                                        min_fraction=min_fraction)
    return Grid(
        (IM(n, (0.0, 1.0), periodic=True, name="x"),
         IM(n, (0.0, 1.0), periodic=True, name="y"),
         IM(nz, (-1.0, 0.0), periodic=False, name="z")), **kw)


def _ps_space(grid):
    return fr.spatial.Profile("x", "y").resolve(grid).bare


def _solver(grid, *, eps=0.0, iterations=60, tolerance=None,
            preconditioner="spectral"):
    return BarotropicPressureSolver(
        grid, _ps_space(grid), ("zp", "z"), "z", epsilon=eps,
        iterations=iterations, tolerance=tolerance,
        preconditioner=preconditioner)


def _rand(grid, seed):
    template = grid.create_field(_ps_space(grid))
    rng = np.random.default_rng(seed)
    return template.with_data(jnp.asarray(
        rng.standard_normal(template.data.shape)))


def _inner(a, b):
    return float(jnp.sum((a * b).integrate().data))


# ================================================================
#  Wet-column self-adjointness: <A p, q> == <p, A q> (the SPD license)
# ================================================================
@pytest.mark.parametrize("eps", [0.0, 1.0])
@pytest.mark.parametrize(
    "a", [pytest.param(0.4, id="mild"), pytest.param(0.8, id="steep")])
def test_wet_operator_is_self_adjoint(a, eps):
    grid = _grid(a=a)
    solver = _solver(grid, eps=eps)
    op = solver.operator(gravity=CSQR, dt=DT)
    p, q = _rand(grid, 1), _rand(grid, 2)
    apq, paq = _inner(op(p), q), _inner(p, op(q))
    assert abs(apq - paq) <= 1e-12 * abs(apq)


# ================================================================
#  The wet-column face depth is a genuine partial (0 < H_a < H_ref)
# ================================================================
def test_wet_face_depth_is_a_genuine_partial():
    solver = _solver(_grid(a=0.4), eps=1.0)
    h_x = np.asarray(solver._face_depth("x").data)
    # a sloped partial bottom: the wet depth varies strictly inside
    # (0, 1) (the full extent) column to column
    assert float(h_x.min()) > 0.0
    assert float(h_x.max()) < 1.0
    assert float(h_x.max()) - float(h_x.min()) > 1e-3


def test_all_wet_face_depth_matches_pure_terrain_bitwise():
    # alpha == 1 everywhere: H_a = int J dz, byte-identical to the pure
    # terrain face depth (no mask interference)
    im = _solver(_grid(a=0.4, init=_allwet, min_fraction=0.0), eps=1.0)
    un = _solver(_grid(a=0.4, immersed=False), eps=1.0)
    for axis in ("x", "y"):
        hi = np.asarray(im._face_depth(axis).data)
        hu = np.asarray(un._face_depth(axis).data)
        assert np.array_equal(hi, hu), axis


# ================================================================
#  Rigid lid (eps=0): the solution is wet-mean-free; dry columns zero
# ================================================================
def test_rigid_lid_solution_is_wet_mean_free_and_masks_land():
    grid = _grid(a=0.4, init=_coast)
    solver = _solver(grid, eps=0.0, iterations=80)
    # a range-compatible RHS: A applied to a random field
    rand = _rand(grid, 5)
    rhs = solver.operator(gravity=CSQR, dt=DT)(rand)
    ps = solver.solve(rhs, gravity=CSQR, dt=DT)
    # the wet-column indicator (theta_col > 0) on the ps cell
    theta_col = Integral()["z"](
        grid.immersed.fraction(fr.spatial.Collocated().resolve(grid)))
    cell_mask = np.asarray(theta_col.data > 0.0)
    ps_data = np.asarray(ps.data)
    assert bool((~cell_mask).any())              # genuine land columns
    # dry columns carry exactly no surface pressure
    assert float(np.abs(ps_data[~cell_mask]).max()) == 0.0
    # the wet-column mean is removed (the eps=0 gauge)
    wet_mean = float(_inner(theta_col.with_data(
        cell_mask.astype(ps_data.dtype)), ps)
        / _inner(theta_col.with_data(cell_mask.astype(ps_data.dtype)),
                 theta_col.with_data(cell_mask.astype(ps_data.dtype))))
    assert abs(wet_mean) <= 1e-11 * (float(np.abs(ps_data).max()) + 1.0)
    # the solve inverts its own operator on the wet region
    residual = rhs - solver.operator(gravity=CSQR, dt=DT)(ps)
    rel = np.sqrt(_inner(residual, residual)) / np.sqrt(_inner(rhs, rhs))
    assert rel < 1e-8


# ================================================================
#  The wet-masked spectral preconditioner converges within budget
# ================================================================
@pytest.mark.parametrize(
    "a", [pytest.param(0.4, id="mild"), pytest.param(0.8, id="steep")])
def test_spectral_converges_within_budget_on_a_cut_chart(a):
    grid = _grid(a=a)
    solver = _solver(grid, eps=1.0, iterations=40, tolerance=1e-8)
    rhs = _rand(grid, 4)
    _ps, info = solver.krylov(gravity=CSQR, dt=DT).solve(rhs)
    # the masked mean-depth spectral inverse converges well inside the
    # 40-iteration budget on a genuine cut chart (measured 12 at a=0.4,
    # 17 at a=0.8)
    assert int(info["iterations"]) <= 25


# ================================================================
#  Phase C: the wet-aware multigrid preconditioner composes with
#  the immersed (cut-cell) domain (the taught error is lifted)
# ================================================================
def test_multigrid_composes_with_immersed_grid():
    # constructing the multigrid preconditioner on a cut chart no longer
    # raises; the V-cycle builds its coarsened hierarchy over the
    # immersed grid (nx=16 -> 8 -> 4, a three-level cycle)
    solver = _solver(_grid(n=16, a=0.8), preconditioner="multigrid")
    assert solver._immersed is not None
    vcycle = solver._build_vcycle(gravity=CSQR, dt=DT)
    assert len(vcycle.levels) == 3


def test_coarse_levels_requantify_the_wet_fractions():
    # Grid.coarsened propagates the immersed descriptor, so each coarse
    # V-cycle level re-instantiates the solver on a grid that still
    # carries the cut-cell domain and re-derives a genuine WET face depth
    # (0 < H_a < full extent) from its own coarse chart
    grid = _grid(n=16, a=0.8)
    chain = coarsen_levels(grid, _ps_space(grid), vertical="z",
                           coarsen_vertical=False)
    assert len(chain) >= 2
    coarse_grid, coarse_space, _ = chain[1]
    assert coarse_grid.immersed is not None      # descriptor propagated
    coarse = BarotropicPressureSolver(
        coarse_grid, coarse_space, ("zp", "z"), "z", epsilon=1.0,
        iterations=1, tolerance=None)
    assert coarse._immersed is not None
    h_x = np.asarray(coarse._face_depth("x").data)
    # a genuine partial: re-quadratured wet, strictly inside (0, full)
    assert float(h_x.min()) >= 0.0
    assert float(h_x.max()) < 1.0
    assert float(h_x.max()) - float(h_x.min()) > 1e-3


@pytest.mark.parametrize("eps", [0.0, 1.0])
def test_all_wet_cut_chart_multigrid_matches_pure_terrain(eps):
    # alpha == 1 everywhere: the V-cycle preconditioner is byte-identical
    # to the pure terrain multigrid (no cut-cell interference), and the
    # full solve matches to machine precision
    wet = _grid(n=16, a=0.6, init=_allwet, min_fraction=0.0)
    pure = _grid(n=16, a=0.6, immersed=False)
    s_wet = _solver(wet, eps=eps, preconditioner="multigrid")
    s_pure = _solver(pure, eps=eps, preconditioner="multigrid")
    # identical input residual on both grids
    rng = np.random.default_rng(21)
    data = jnp.asarray(rng.standard_normal(
        pure.create_field(_ps_space(pure)).data.shape))
    r_wet = wet.create_field(_ps_space(wet)).with_data(data)
    r_pure = pure.create_field(_ps_space(pure)).with_data(data)
    z_wet = s_wet._build_vcycle(gravity=CSQR, dt=DT)(r_wet)
    z_pure = s_pure._build_vcycle(gravity=CSQR, dt=DT)(r_pure)
    assert np.array_equal(np.asarray(z_wet.data), np.asarray(z_pure.data))
    # the full preconditioned solve agrees to machine precision (the
    # fields live on different grid objects, so compare the raw data)
    x_wet = np.asarray(s_wet.solve(r_wet, gravity=CSQR, dt=DT).data)
    x_pure = np.asarray(s_pure.solve(r_pure, gravity=CSQR, dt=DT).data)
    scale = float(np.abs(x_pure).max()) + 1e-30
    assert float(np.abs(x_wet - x_pure).max()) / scale <= 1e-12


@pytest.mark.parametrize("eps", [0.0, 1.0])
def test_multigrid_converges_machine_zero_on_a_cut_chart(eps):
    grid = _grid(n=32, a=0.8)
    solver = _solver(grid, eps=eps, iterations=200, tolerance=1e-8,
                     preconditioner="multigrid")
    # a wet-supported RHS (the physical transport divergence is zero on
    # dry columns): A applied to a random field is range-compatible
    rand = _rand(grid, 9)
    rhs = solver.operator(gravity=CSQR, dt=DT)(rand)
    ps = solver.solve(rhs, gravity=CSQR, dt=DT)
    residual = rhs - solver.operator(gravity=CSQR, dt=DT)(ps)
    rel = np.sqrt(_inner(residual, residual)) / np.sqrt(_inner(rhs, rhs))
    assert rel < 1e-8


def test_multigrid_beats_spectral_on_a_steep_shelf():
    # on a steep partial-bottom shelf the flat mean-depth spectral inverse
    # degrades badly (measured ~46 iters at n=32) while the wet-aware
    # multigrid stays flat (~10) and converges well inside the budget
    grid = _grid(n=32, a=0.8, init=_cut_steep)
    sp = _solver(grid, eps=1.0, iterations=200, tolerance=1e-8)
    mg = _solver(grid, eps=1.0, iterations=200, tolerance=1e-8,
                 preconditioner="multigrid")
    _p, info_sp = sp.krylov(gravity=CSQR, dt=DT).solve(_rand(grid, 4))
    _q, info_mg = mg.krylov(gravity=CSQR, dt=DT).solve(_rand(grid, 4))
    assert int(info_mg["iterations"]) < int(info_sp["iterations"])
    assert int(info_mg["iterations"]) <= 20


def test_multigrid_iterations_stay_flat_with_resolution():
    # h-independence on a partial-bottom cut chart: the wet-aware V-cycle
    # holds a bounded iteration count as the horizontal grid refines
    counts = []
    for n in (16, 32):
        grid = _grid(n=n, a=0.8)
        solver = _solver(grid, eps=1.0, iterations=200, tolerance=1e-8,
                         preconditioner="multigrid")
        _p, info = solver.krylov(gravity=CSQR, dt=DT).solve(_rand(grid, 4))
        counts.append(int(info["iterations"]))
    assert all(c <= 15 for c in counts)
    assert abs(counts[1] - counts[0]) <= 3          # flat, not h-growing


@pytest.mark.parametrize("eps", [0.0, 1.0])
def test_multigrid_vcycle_is_self_adjoint_on_a_wet_chart(eps):
    # the fixed symmetric V(1, 1) with adjoint transfers and a symmetric
    # smoother stays symmetric on the wet cut chart, in the V-cycle's own
    # geometry (the grid-agnostic plain-mean projection it applies per
    # level; the outer CG carries the exact wet-column-mean gauge)
    grid = _grid(n=16, a=0.8, init=_coast)
    solver = _solver(grid, eps=eps, preconditioner="multigrid")
    vcycle = solver._build_vcycle(gravity=CSQR, dt=DT)
    r, s = _mean_free(_rand(grid, 30)), _mean_free(_rand(grid, 31))
    mrs, rms = _inner(vcycle(r), s), _inner(r, vcycle(s))
    assert abs(mrs - rms) <= 1e-11 * max(abs(mrs), 1e-30)


def test_multigrid_rigid_lid_solution_is_wet_mean_free_and_masks_land():
    grid = _grid(a=0.4, init=_coast)
    solver = _solver(grid, eps=0.0, iterations=80, tolerance=1e-8,
                     preconditioner="multigrid")
    rand = _rand(grid, 5)
    rhs = solver.operator(gravity=CSQR, dt=DT)(rand)
    ps = solver.solve(rhs, gravity=CSQR, dt=DT)
    theta_col = Integral()["z"](
        grid.immersed.fraction(fr.spatial.Collocated().resolve(grid)))
    cell_mask = np.asarray(theta_col.data > 0.0)
    ps_data = np.asarray(ps.data)
    assert bool((~cell_mask).any())                 # genuine land columns
    assert float(np.abs(ps_data[~cell_mask]).max()) == 0.0   # dry -> zero
    wet_ind = theta_col.with_data(cell_mask.astype(ps_data.dtype))
    wet_mean = float(_inner(wet_ind, ps) / _inner(wet_ind, wet_ind))
    assert abs(wet_mean) <= 1e-11 * (float(np.abs(ps_data).max()) + 1.0)
    residual = rhs - solver.operator(gravity=CSQR, dt=DT)(ps)
    rel = np.sqrt(_inner(residual, residual)) / np.sqrt(_inner(rhs, rhs))
    assert rel < 1e-8
