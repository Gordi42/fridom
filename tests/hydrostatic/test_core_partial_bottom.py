"""Partial-bottom-cell hydrostatic pressure-gradient correction (PB-D2).

Prefix-mirrored shard of ``hy.modules.core`` covering the well-balanced
partial-bottom pressure gradient (Pacanowski & Gnanadesikan): a resting
stratified column sampled at the wet-centroid heights (PB-D4) over
sloping immersed bathymetry stays at rest to machine precision, the
correction is a byte no-op off a partial bottom cell (all-wet /
staircase), it is >= 2nd-order convergent for a smooth profile, and it
is reverse-mode differentiable and device-count invariant. Self-contained
per the AGENTS oversized-module / self-contained-shard rule.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.hydrostatic as hy
from fridom.model.context import StepContext
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh

IM = IntervalMesh
CTX = StepContext(params={}, clock=jnp.asarray(0.0), dt=jnp.asarray(0.01),
                  stage_dt=jnp.asarray(0.01))


# ================================================================
#  Self-contained builders
# ================================================================
def _col_indicator(nx):
    """Per-column-constant bottom (the classic partial-cell staircase).

    The bottom depth is flat within each x-cell and steps across
    columns, so each column carries a single partial bottom cell (no
    sub-cell corner cuts) -- the Pacanowski & Gnanadesikan well-balanced
    setup.
    """
    def indic(x, y, z):  # noqa: ARG001
        col = jnp.floor(x * nx)
        return (z > 0.18 + 0.42 * (col + 0.5) / nx).astype(float)
    return indic


def _grid(nx, nz, order=8, *, zmesh=None, indic=None):
    """Build an (x, y periodic; z bounded) immersed grid + cell space."""
    zm = zmesh or IM(nz, (0.0, 1.0), periodic=False, name="z")
    return Grid(
        (IM(nx, (0.0, 1.0), periodic=True, name="x"),
         IM(1, (0.0, 1.0), periodic=True, name="y"), zm),
        immersed=ImmersedDomain(indic or _col_indicator(nx), order=order,
                                min_fraction=0.0))


def _rest_pgf(model, strat):
    """Return (corrected, uncorrected) masked horizontal PGF at rest.

    ``b`` is sampled at the physical wet-centroid heights
    ``zeta = z_c + centroid_offset`` (PB-D4), zero on dry cells.
    """
    grid = model.grid
    imm = grid.immersed
    core = model.module(hy.Core)
    bs = model.state["b"].function_space
    z_c = grid.evaluation_nodes(bs, "z")
    delta = imm.centroid_offset(bs, "z")
    mask = imm.mask(bs)

    @jax.jit
    def run():
        zeta = z_c.data + delta.data
        st = model.state.replace(
            b=model.state["b"].with_data(jnp.where(mask.data, strat(zeta),
                                                   0.0)))
        p_hyd = core._diagnose_p_hyd(st, CTX)["p_hyd"]
        st = st.replace(p_hyd=p_hyd)
        out = core.pressure_gradient(st, CTX)
        mu = imm.mask(out["u"].function_space)
        plain = (-p_hyd.diff("x")).retag(st["u"])
        return (jnp.abs(out["u"].data * mu.data).max(),
                jnp.abs(plain.data * mu.data).max())

    cor, unc = (float(v) for v in run())
    return cor, unc


# ================================================================
#  Genuine partial cells exist (the setup is a real partial cut)
# ================================================================
def test_setup_has_genuine_partial_bottom_cells():
    model = hy.Model(
        grid=_grid(6, 8),
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(0.01, order=3),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)
    imm = model.grid.immersed
    bs = model.state["b"].function_space
    theta = np.asarray(imm.fraction(bs).data)
    delta = np.asarray(imm.centroid_offset(bs, "z").data)
    assert ((theta > 1e-6) & (theta < 1.0 - 1e-6)).sum() > 0
    assert (delta > 1e-6).sum() > 0             # genuine offsets exist


# ================================================================
#  G3: well-balancedness -- rest state is machine zero (keystone)
# ================================================================
@pytest.mark.parametrize(("nx", "nz", "order"),
                         [(6, 8, 8), (8, 16, 6), (5, 12, 4)])
def test_g3_rest_state_is_machine_zero_flat(nx, nz, order):
    """Check b = N^2 z at rest over a cut z-grid gives zero PGF."""
    model = hy.Model(
        grid=_grid(nx, nz, order),
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(0.01, order=3),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)
    cor, unc = _rest_pgf(model, lambda z: 2.5 * z)
    assert unc > 1e-3               # the uncorrected error is real
    assert cor < 1e-12              # the correction cancels it exactly


def test_g3_rest_state_is_machine_zero_stretched():
    """Stretched-z (separable, not a chart) is also machine zero."""
    zm = MappedIntervalMesh(
        8, (0.0, 1.0),
        lambda t: t - 0.12 * jnp.sin(2 * jnp.pi * t) / (2 * jnp.pi),
        periodic=False, name="z")
    model = hy.Model(
        grid=_grid(6, 8, 8, zmesh=zm),
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(0.01, order=3),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)
    cor, unc = _rest_pgf(model, lambda z: 2.5 * z)
    assert unc > 1e-3
    assert cor < 1e-12


# ================================================================
#  G4: smooth stratification -- >= 2nd order, below uncorrected
# ================================================================
def test_g4_smooth_profile_is_second_order_and_below_uncorrected():
    def strat(z):
        return jnp.tanh((z - 0.5) / 0.15)
    dz = []
    cors = []
    uncs = []
    for nz in (8, 16, 32, 64):
        model = hy.Model(
            grid=_grid(6, nz, 8),
            core=hy.Core(gravity=1.0),
            time_stepper=AdamBashforth(0.01, order=3),
            stratification=hy.ConstantStratification(n2=1.0),
            free_surface=hy.ExplicitFreeSurface(),
            advection=False)
        cor, unc = _rest_pgf(model, strat)
        dz.append(1.0 / nz)
        cors.append(cor)
        uncs.append(unc)
        assert cor < unc               # below uncorrected at every n
    # >= 2nd-order convergence (fit the last three, robust to the coarse
    # rung); the corrected slope must beat 1.8.
    p = np.polyfit(np.log(dz[1:]), np.log(cors[1:]), 1)[0]
    assert p >= 1.8, (dz, cors, uncs, p)


# ================================================================
#  G1: all-wet -- the correction is a byte no-op vs the plain diff
# ================================================================
def test_g1_all_wet_correction_is_a_byte_noop():
    # an all-wet immersed grid has delta identically 0 (no bottom cut),
    # so the correction never fires and the pressure gradient is
    # byte-identical to the plain diff (the flat, uncorrected path).
    model = hy.Model(
        grid=_grid(6, 8, 8, indic=lambda x, y, z: x * 0.0 + 1.0),  # noqa: ARG005
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(0.02, order=3),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)
    core = model.module(hy.Core)
    assert core._pb_active is False        # no bottom cut -> skipped
    rng = np.random.default_rng(7)
    bdat = 0.3 * rng.standard_normal(model.state["b"].data.shape)

    @jax.jit
    def run():
        st = model.state.replace(
            b=model.state["b"].with_data(jnp.asarray(bdat)))
        p = core._diagnose_p_hyd(st, CTX)["p_hyd"]
        st = st.replace(p_hyd=p)
        out = core.pressure_gradient(st, CTX)
        plain_u = (-p.diff("x")).retag(st["u"])
        plain_v = (-p.diff("y")).retag(st["v"])
        return (jnp.abs(out["u"].data - plain_u.data).max(),
                jnp.abs(out["v"].data - plain_v.data).max())

    du, dv = (float(v) for v in run())
    assert du == 0.0
    assert dv == 0.0


# ================================================================
#  G2: staircase (order=None) is a byte no-op vs the plain diff
# ================================================================
def test_g2_staircase_is_a_byte_noop():
    grid = _grid(6, 8)
    grid = Grid(
        (IM(6, (0.0, 1.0), periodic=True, name="x"),
         IM(1, (0.0, 1.0), periodic=True, name="y"),
         IM(8, (0.0, 1.0), periodic=False, name="z")),
        immersed=ImmersedDomain(
            lambda x, y, z: (z > 0.4).astype(float), order=None))  # noqa: ARG005
    model = hy.Model(
        grid=grid,
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(0.01, order=3),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)
    core = model.module(hy.Core)
    rng = np.random.default_rng(1)
    bdat = 0.3 * rng.standard_normal(model.state["b"].data.shape)

    @jax.jit
    def run():
        st = model.state.replace(
            b=model.state["b"].with_data(jnp.asarray(bdat)))
        p = core._diagnose_p_hyd(st, CTX)["p_hyd"]
        st = st.replace(p_hyd=p)
        out = core.pressure_gradient(st, CTX)
        plain_u = (-p.diff("x")).retag(st["u"])
        plain_v = (-p.diff("y")).retag(st["v"])
        return (jnp.abs(out["u"].data - plain_u.data).max(),
                jnp.abs(out["v"].data - plain_v.data).max())

    du, dv = (float(v) for v in run())
    assert du == 0.0
    assert dv == 0.0


# ================================================================
#  G5: reverse-mode autodiff shard (Model.propagator, FD rtol 1e-4)
# ================================================================
def test_g5_grad_wrt_initial_buoyancy_matches_fd():
    """Grad through a short cut-cell run w.r.t. the initial b: FD-matched.

    The correction reads ``b`` (linear, static weight) in the pressure
    gradient, so this certifies the added step-path arithmetic is clean.
    """
    model = hy.Model(
        grid=_grid(6, 8, 4),
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(2e-3, order=3),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)
    rng = np.random.default_rng(11)
    model.set_fields(**{
        k: 0.1 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "b")})
    run = model.propagator(wrt=("b",), steps=6)
    b0 = model._carry.state["b"].storage

    def loss(field):
        return sum(jnp.sum(f.data ** 2) for f in run((field,)).state)

    grad = np.asarray(jax.grad(loss)(b0))
    assert bool(np.all(np.isfinite(grad)))
    direction = jnp.asarray(rng.standard_normal(b0.shape), dtype=b0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(b0 + eps * direction))
          - float(loss(b0 - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


# ================================================================
#  G6: forced-4 device-count invariance of the cut-cell step
# ================================================================
def _advance_state(nx, ny, nz, device_ids):
    grid = Grid(
        (IM(nx, (0.0, 1.0), periodic=True, name="x"),
         IM(ny, (0.0, 1.0), periodic=True, name="y"),
         IM(nz, (0.0, 1.0), periodic=False, name="z")),
        immersed=ImmersedDomain(_col_indicator(nx), order=4,
                                min_fraction=0.1),
        device_ids=device_ids)
    model = hy.Model(
        grid=grid,
        core=hy.Core(gravity=10.0),
        time_stepper=AdamBashforth(2e-3, order=3),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=True)
    rng = np.random.default_rng(4)
    model.set_fields(**{
        k: 0.1 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "b")})
    model.advance(6)
    sharded = [n for n, _ in grid.decomposition.default_layout.device_axes]
    return {k: np.asarray(model.state[k].data)
            for k in ("u", "v", "b")}, sharded


@pytest.mark.multi_device
def test_g6_forced4_device_count_invariant(forced_devices):
    """A z-sharded cut-cell run matches the single-device run."""
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    # 4x4x16: the tiny horizontal forces the negotiation onto z, so the
    # partial-bottom correction crosses the reshard-to-axis-local path.
    many, sharded = _advance_state(4, 4, 16, None)
    one, _ = _advance_state(4, 4, 16, (0,))
    assert "z" in sharded
    for name in ("u", "v", "b"):
        # forced-host cpu reassociates the multi-device reductions, so
        # parity holds to a tight atol (the test_multi_device convention)
        assert np.allclose(many[name], one[name], rtol=0.0, atol=1e-11), \
            name
