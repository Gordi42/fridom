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
)
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.integrate import Integral

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
        grid, _ps_space(grid), ("zp", "z"), "z", epsilon=eps, inv_depth=1.0,
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
    op = solver.operator(csqr=CSQR, dt=DT)
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
    rhs = solver.operator(csqr=CSQR, dt=DT)(rand)
    ps = solver.solve(rhs, csqr=CSQR, dt=DT)
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
    residual = rhs - solver.operator(csqr=CSQR, dt=DT)(ps)
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
    _ps, info = solver.krylov(csqr=CSQR, dt=DT).solve(rhs)
    # the masked mean-depth spectral inverse converges well inside the
    # 40-iteration budget on a genuine cut chart (measured 12 at a=0.4,
    # 17 at a=0.8)
    assert int(info["iterations"]) <= 25


# ================================================================
#  Taught error: the multigrid preconditioner is not yet wet-aware
# ================================================================
def test_multigrid_plus_immersed_is_a_taught_error():
    with pytest.raises(NotImplementedError,
                       match="multigrid preconditioner does not yet"):
        _solver(_grid(a=0.4), preconditioner="multigrid")
