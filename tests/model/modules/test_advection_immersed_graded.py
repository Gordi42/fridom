r"""Graded-mask biased advection on immersed (cut-cell) grids (GA-D1..D6).

The mask-keyed generalization of the wall graded closure
(``spatial/operators/graded.py``): on an immersed grid
``UpwindAdvection`` / ``WENOAdvection`` (orders 3/5) choose, per output
face, the widest rung of the existing ladder whose union window is
entirely wet (`graded.apply_graded_mask`), over a pre-masked operand.
The anchor gate is **staircase equivalence**: a face-aligned immersed
box reproduces the walled graded biased model bit for bit (the
advection tendency is machine-zero; a full masked-pressure run agrees
to the CG residual). Both reconstruction families are exercised — the
FV ``CellAvg`` route (nonhydro2, family="fv") and the nodal
``Center``/``Inner`` route (the hydrostatic model) — and both
cell-frame shifts (the primal tracer / transverse-momentum ``shift = 0``
and the dual momentum self-advection ``shift = 1``).

This is a prefix-mirrored shard of ``advection.py`` (AGENTS oversized-
module rule); it is self-contained (small builders duplicated).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.hydrostatic as hy
import fridom.nonhydro2 as nh
from fridom.model import params as fr_params
from fridom.model.context import StepContext
from fridom.model.model import _chunk_body
from fridom.model.modules.advection import (
    UpwindAdvection,
    WENOAdvection,
)
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

TWO_PI = 2.0 * np.pi
IM = IntervalMesh

# the biased schemes covered (orders 3 and 5, upwind and WENO)
SCHEMES = [
    pytest.param(lambda: UpwindAdvection(3), id="upwind3"),
    pytest.param(lambda: UpwindAdvection(5), id="upwind5"),
    pytest.param(lambda: WENOAdvection(3), id="weno3"),
    pytest.param(lambda: WENOAdvection(5), id="weno5"),
]


def _ctx():
    """Return a minimal step context with the default Rossby scaling."""
    return StepContext(
        params={fr_params.SCALING_ROSSBY: jnp.asarray(1.0)},
        clock=None, dt=0.02, stage_dt=0.02, tendency_sums=None)


def _advection_module(model):
    """Return the single flux-form biased advection module of ``model``."""
    (module,) = [m for m in model.modules
                 if isinstance(m, UpwindAdvection)]
    return module


# ================================================================
#  Builders (self-contained per the AGENTS oversized-module rule)
# ================================================================
def _box_model(fac, *, npg=12, lo=3, pressure_iterations=60):
    """Build a face-aligned {0, 1} box in a periodic ``npg`` cube (FV)."""
    box = lambda x, y, z: (  # noqa: E731
        (x > lo) & (x < 9) & (y > lo) & (y < 9)
        & (z > lo) & (z < 9)).astype(float)
    return nh.Model(
        grid=Grid(tuple(
            IM(npg, (0.0, 12.0), periodic=True, name=nm)
            for nm in ("x", "y", "z")), immersed=ImmersedDomain(box)),
        dt=0.02, advection=fac(), coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=0.0),
        pressure_iterations=pressure_iterations)


def _walled_model(fac, *, pressure_iterations=60):
    """Build the 6^3 walled FV twin of the box's wet region."""
    return nh.Model(
        grid=Grid(tuple(
            IM(6, (3.0, 9.0), periodic=False, name=nm)
            for nm in ("x", "y", "z"))),
        dt=0.02, advection=fac(), coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=0.0),
        pressure_iterations=pressure_iterations, family="fv")


def _seed_box_and_wall(imm, wal, lo=3, seed=11):
    """Set the same random IC on the walled model and the box's wet region."""
    rng = np.random.default_rng(seed)
    shapes = {k: wal.state[k].data.shape for k in ("u", "v", "w")}
    ic = {k: 0.1 * rng.standard_normal(shapes[k]) for k in ("u", "v", "w")}
    bic = 0.1 * rng.standard_normal((6, 6, 6))
    wal.set_fields(b=bic, **ic)
    full = {}
    for k in ("u", "v", "w"):
        arr = np.zeros((12, 12, 12))
        s = shapes[k]
        arr[lo:lo + s[0], lo:lo + s[1], lo:lo + s[2]] = ic[k]
        full[k] = arr
    barr = np.zeros((12, 12, 12))
    barr[lo:9, lo:9, lo:9] = bic
    imm.set_fields(b=barr, **full)


def _wet_subblock(imm, wal, name, lo=3):
    """Return the box wet sub-block of ``imm`` aligned to ``wal``."""
    s = wal.state[name].data.shape
    return np.asarray(imm.state[name].data)[
        lo:lo + s[0], lo:lo + s[1], lo:lo + s[2]]


# ================================================================
#  Gate: staircase equivalence — the advection tendency is machine-zero
#  against the walled graded model (the FV path, both shifts)
# ================================================================
@pytest.mark.parametrize("fac", SCHEMES)
def test_staircase_advection_tendency_matches_walled_to_machine_zero(fac):
    imm = _box_model(fac, pressure_iterations=5)
    wal = _walled_model(fac, pressure_iterations=5)
    _seed_box_and_wall(imm, wal)
    ctx = _ctx()
    t_imm = _advection_module(imm)._advect(imm.state, ctx)
    t_wal = _advection_module(wal)._advect(wal.state, ctx)
    for name in ("u", "v", "w", "b"):
        s = wal.state[name].data.shape
        sub = np.asarray(t_imm[name].data)[3:3 + s[0], 3:3 + s[1],
                                           3:3 + s[2]]
        diff = np.abs(sub - np.asarray(t_wal[name].data)).max()
        # bitwise for upwind (linear rows), reversed-summation ulps for
        # WENO's nonlinear weights (GA-D1/D2: the pre-mask reproduces the
        # wall path's synthesized zeros, the union selector its rung)
        assert diff < 1e-13, (name, diff)


# ================================================================
#  Gate: the wall="centered2" bottom rung also matches the walled model
#  (the mask ladder carries the symmetric bottom rung, GA-D5)
# ================================================================
@pytest.mark.parametrize("wall", ["upwind1", "centered2"])
def test_wall_rung_option_matches_walled_tendency(wall):
    def fac():
        return UpwindAdvection(5, wall=wall)
    imm = _box_model(fac, pressure_iterations=5)
    wal = _walled_model(fac, pressure_iterations=5)
    _seed_box_and_wall(imm, wal)
    ctx = _ctx()
    t_imm = _advection_module(imm)._advect(imm.state, ctx)
    t_wal = _advection_module(wal)._advect(wal.state, ctx)
    for name in ("u", "v", "w", "b"):
        s = wal.state[name].data.shape
        sub = np.asarray(t_imm[name].data)[3:3 + s[0], 3:3 + s[1],
                                           3:3 + s[2]]
        assert np.abs(sub - np.asarray(t_wal[name].data)).max() < 1e-13


# ================================================================
#  Gate: staircase equivalence — a full masked-pressure run tracks the
#  walled model to the CG residual (the I2 precedent tolerance)
# ================================================================
@pytest.mark.parametrize("fac", SCHEMES)
def test_staircase_run_tracks_walled_model(fac):
    imm = _box_model(fac)
    wal = _walled_model(fac, pressure_iterations=1)
    _seed_box_and_wall(imm, wal)
    imm.advance(12)
    wal.advance(12)
    assert not imm.panicked
    for name in ("u", "v", "w", "b"):
        sub = _wet_subblock(imm, wal, name)
        diff = np.abs(sub - np.asarray(wal.state[name].data)).max()
        # the advection is bitwise (above); the residual here is the
        # masked CG vs the walled solve — pressure-limited, the level the
        # centered gate reaches (tests/nonhydro2/test_immersed_model.py)
        assert diff < 1e-10, (name, diff)


# ================================================================
#  Gate: all-wet immersed reproduces the unimmersed run (both families)
# ================================================================
def _nh_meshes():
    return tuple(IM(10, (0.0, TWO_PI), periodic=(nm != "z"), name=nm)
                 for nm in ("x", "y", "z"))


def _hy_meshes():
    return (IM(8, (0.0, 1.0), periodic=True, name="x"),
            IM(8, (0.0, 1.0), periodic=True, name="y"),
            IM(6, (0.0, 1.0), periodic=False, name="z"))


@pytest.mark.parametrize("fac", SCHEMES)
def test_all_wet_immersed_matches_unimmersed_fv(fac):
    allwet = ImmersedDomain(lambda x, y, z: x * 0.0 + 1.0)  # noqa: ARG005
    im = nh.Model(grid=Grid(_nh_meshes(), immersed=allwet), dt=0.02,
                  advection=fac(), coriolis=nh.FPlaneCoriolis(f0=1.0),
                  pressure_iterations=3)
    un = nh.Model(grid=Grid(_nh_meshes()), dt=0.02, advection=fac(),
                  coriolis=nh.FPlaneCoriolis(f0=1.0))
    rng = np.random.default_rng(7)
    ic = {k: 0.3 * rng.standard_normal(im.state[k].data.shape)
          for k in ("u", "v", "w", "b")}
    im.set_fields(**ic)
    un.set_fields(**ic)
    im.advance(8)
    un.advance(8)
    for k in ("u", "v", "w", "b", "p"):
        diff = np.abs(np.asarray(im.state[k].data)
                      - np.asarray(un.state[k].data)).max()
        assert diff < 1e-11, (k, diff)


@pytest.mark.parametrize(
    "fac", [pytest.param(lambda: UpwindAdvection(3), id="upwind3"),
            pytest.param(lambda: WENOAdvection(5), id="weno5")])
def test_all_wet_immersed_matches_unimmersed_nodal(fac):
    # the hydrostatic model is the pure-nodal route (Center tracers,
    # Right/Center momentum): the mask path must reduce to the plain
    # reconstruction bit for bit when every DOF is wet
    allwet = ImmersedDomain(lambda x, y, z: x * 0.0 + 1.0)  # noqa: ARG005
    im = hy.Model(grid=Grid(_hy_meshes(), immersed=allwet),
                  dt=0.02, advection=fac())
    un = hy.Model(grid=Grid(_hy_meshes()), dt=0.02, advection=fac())
    rng = np.random.default_rng(3)
    ic = {k: 0.3 * rng.standard_normal(im.state[k].data.shape)
          for k in ("u", "v", "b")}
    im.set_fields(**ic)
    un.set_fields(**ic)
    im.advance(8)
    un.advance(8)
    for k in ("u", "v", "b"):
        diff = np.abs(np.asarray(im.state[k].data)
                      - np.asarray(un.state[k].data)).max()
        assert diff < 1e-11, (k, diff)


# ================================================================
#  Gate: a uniform tracer is reconstructed exactly on every rung
# ================================================================
@pytest.mark.parametrize("fac", SCHEMES)
def test_uniform_tracer_reconstructed_exactly_on_every_rung(fac):
    # free-stream preservation: face(const) == const on every wet face,
    # whatever rung the mask selects (the reduced rungs and the interior
    # kernel are all exact on constants)
    imm = _box_model(fac, pressure_iterations=1)
    const = 1.75
    imm.set_fields(b=np.full(imm.state["b"].data.shape, const))
    module = _advection_module(imm)
    b = imm.state["b"]
    for axis in ("x", "y", "z"):
        flux_space = module._flux_space(b, imm.state["u"], axis)
        left, right = module._biased_pair(b, axis)
        alpha = imm.grid.immersed.fraction(flux_space)
        wet = np.asarray(alpha.data) > 0.0
        for recon in (left, right):
            face = recon(b).to(flux_space)
            vals = np.asarray(face.data)
            assert np.abs(vals[wet] - const).max() < 1e-12, axis


# ================================================================
#  Gate: theta-weighted tracer content conserved to machine zero on
#  genuine partial cells with biased advection active
# ================================================================
@pytest.mark.parametrize(
    "fac", [pytest.param(lambda: UpwindAdvection(5), id="upwind5"),
            pytest.param(lambda: WENOAdvection(5), id="weno5")])
def test_theta_weighted_tracer_conserved_on_partials(fac):
    n = 12
    # a smooth slanted wall carves genuine partial cells (order=2
    # quadrature, min_fraction floor); n2=0 so advection alone governs b
    slope = lambda x, y, z: jnp.clip(  # noqa: E731, ARG005
        1.3 - 0.3 * x, 0.0, 1.0)
    grid = Grid(tuple(
        IM(n, (0.0, TWO_PI), periodic=True, name=nm)
        for nm in ("x", "y", "z")),
        immersed=ImmersedDomain(slope, order=2, min_fraction=0.1))
    model = nh.Model(
        grid=grid, dt=0.01, advection=fac(),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=0.0),
        pressure_iterations=25)
    rng = np.random.default_rng(3)
    model.set_fields(
        b=rng.standard_normal(model.state["b"].data.shape),
        **{k: 0.2 * rng.standard_normal(model.state[k].data.shape)
           for k in ("u", "v", "w")})
    theta = grid.immersed.fraction(model.state["b"].function_space)

    def total():
        return float(jnp.sum((theta * model.state["b"]).integrate().data))

    before = total()
    model.advance(20)
    assert not model.panicked
    after = total()
    # partial cells exist (a true test of the fraction weighting)
    th = np.asarray(theta.data)
    assert ((th > 1e-6) & (th < 1.0 - 1e-6)).sum() > 0
    assert abs(after - before) <= 1e-12 * max(abs(before), 1.0)


# ================================================================
#  Gate: a narrow wet pocket (1-2 cells wide) is stable and conserving
# ================================================================
@pytest.mark.parametrize(
    "fac", [pytest.param(lambda: UpwindAdvection(5), id="upwind5"),
            pytest.param(lambda: WENOAdvection(5), id="weno5")])
def test_narrow_wet_pocket_is_stable_and_conserving(fac):
    # a 2-cell-wide wet slab in z (cells 5,6 of 12): the widest order-5
    # window cannot fit, so the ladder self-serves the bottom rungs and
    # the run stays finite; theta-weighted b is conserved to roundoff
    n = 12
    slab = lambda x, y, z: (  # noqa: E731, ARG005
        (z > 5.0) & (z < 7.0)).astype(float)
    grid = Grid(tuple(
        IM(n, (0.0, 12.0), periodic=True, name=nm)
        for nm in ("x", "y", "z")), immersed=ImmersedDomain(slab))
    model = nh.Model(
        grid=grid, dt=0.01, advection=fac(),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=0.0),
        pressure_iterations=20)
    rng = np.random.default_rng(4)
    model.set_fields(
        b=rng.standard_normal(model.state["b"].data.shape),
        **{k: 0.2 * rng.standard_normal(model.state[k].data.shape)
           for k in ("u", "v", "w")})
    theta = grid.immersed.fraction(model.state["b"].function_space)

    def total():
        return float(jnp.sum((theta * model.state["b"]).integrate().data))

    before = total()
    model.advance(12)
    assert not model.panicked
    assert bool(jnp.all(jnp.isfinite(model.state["b"].data)))
    assert abs(total() - before) <= 1e-12 * max(abs(before), 1.0)


# ================================================================
#  Gate: reverse-mode autodiff through the immersed biased mask path
# ================================================================
def test_grad_through_immersed_biased_run_matches_fd():
    # jax.grad of a quadratic loss w.r.t. an initial field through a short
    # masked biased run is finite and FD-matched (AGENTS differentiability
    # policy): the data path crosses the pre-mask double-``where`` and the
    # static-selector reconstruction of the mask closure
    slope = lambda x, y, z: jnp.clip(  # noqa: E731, ARG005
        1.3 - 0.3 * x, 0.0, 1.0)
    grid = Grid(
        (IM(8, (0.0, 6.0), periodic=False, name="x"),
         IM(8, (0.0, TWO_PI), periodic=True, name="y"),
         IM(6, (0.0, 1.0), periodic=False, name="z")),
        immersed=ImmersedDomain(slope, order=2, min_fraction=0.1))
    model = nh.Model(
        grid=grid, dt=0.01, advection=UpwindAdvection(3),
        coriolis=nh.FPlaneCoriolis(f0=1.0), pressure_iterations=10)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        k: 0.2 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "w", "b")})

    record = model._artifacts.record
    leaves, treedef = jax.tree_util.tree_flatten(model._carry)
    leaf = model._carry.state["b"].storage
    (idx,) = [i for i, ref in enumerate(leaves) if ref is leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        carry = jax.tree_util.tree_unflatten(treedef, new)
        final = _chunk_body(record, 6, carry, model._stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    grad = np.asarray(jax.grad(loss)(leaf))
    assert bool(np.all(np.isfinite(grad)))
    direction = jnp.asarray(
        np.random.default_rng(1).standard_normal(leaf.shape),
        dtype=leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(leaf + eps * direction))
          - float(loss(leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)
