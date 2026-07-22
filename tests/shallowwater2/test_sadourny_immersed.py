r"""Fraction-weighted immersed Sadourny momentum (stage B, SA-D1..D6).

The prefix-mirrored shard (AGENTS oversized-module rule) covering the
cut-cell momentum fix of ``shallowwater2/modules/sadourny.py``: the
corner mass fluxes and the kinetic energy carry the same open-area
fraction the thickness transport carries, so the semi-discrete
**wet-weighted energy** production is machine zero on genuine partial
cells (B-G1, the keystone), not only in the fully-wet interior.

Before the fix the corner mass fluxes ``fu`` / ``fv`` and the kinetic
energy were unweighted: the wet-weighted energy production rate was
``~2e-3`` relative (measured), an O(1) break of the skew antisymmetry
at cut cells. After the fix it is ``~1e-16``.

Gates: B-G1 (wet-weighted energy machine zero), B-G2 (mass unregressed),
B-G3 (staircase == walled), B-G4 (all-wet == unimmersed), B-G5 (autodiff
shard — the SA-D4 seals), B-G6 (forced-4 device invariance). Each is
self-contained (the shard duplicates the small builders).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model.model import _chunk_body
from fridom.shallowwater2.modules.immersed_weighting import weight_flux
from fridom.shallowwater2.modules.sadourny import (
    _wet_corner_thickness,
    _wet_kinetic_energy,
)
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

IM = IntervalMesh
TWO_PI = 2.0 * np.pi


# ================================================================
#  Builders (self-contained per the AGENTS oversized-module rule)
# ================================================================
def _slope(x, y):  # noqa: ARG001
    """Smooth analytic side wall -> genuine x-partial cut cells."""
    return jnp.clip(1.4 - 0.35 * x, 0.0, 1.0)


def _partial_grid(*, periodic_x=True, device_ids=None):
    """Immersed grid with genuine (order-2) partial cells; y periodic."""
    return Grid(
        (IM(12, (0.0, 6.0), periodic=periodic_x, name="x"),
         IM(12, (0.0, 6.0), periodic=True, name="y")),
        immersed=ImmersedDomain(_slope, order=2, min_fraction=0.0),
        device_ids=device_ids)


def _model(grid, *, csqr=0.8, ro=0.3, f0=0.0, dt=0.01, advection=True):
    """Build an immersed shallow-water model (f0=0 for the energy gate).

    ``f0 = 0`` has no nondimensional Rossby spelling (Ro -> inf), so
    it maps to no rotation at all — the same physics (f = 0).
    """
    coriolis = (None if f0 == 0.0
                else sw.modules.FPlaneCoriolis(rossby_number=ro / f0))
    return sw.Model(
        grid=grid,
        core=sw.Core(froude_number=ro, depth=csqr),
        scaling=fr.scaling.GravityWave(),
        coriolis=coriolis, advection=advection,
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))


def _fill_random(model, seed):
    """Fill the prognostics; p zeroed on dry cells (structural)."""
    rng = np.random.default_rng(seed)
    mask = np.asarray(
        model.grid.immersed.mask(model.state["p"].function_space).data)
    model.set_fields(
        p=0.3 * rng.standard_normal(model.state["p"].data.shape) * mask,
        u=0.2 * rng.standard_normal(model.state["u"].data.shape),
        v=0.2 * rng.standard_normal(model.state["v"].data.shape))


def _wet_energy_terms(model):
    r"""Per-term rate of the discrete wet-weighted energy.

    The functional consistent with the fraction-weighted scheme is

    .. math::
        E = \sum \tfrac12 \alpha_u\,\bar h^x u^2
            + \tfrac12 \alpha_v\,\bar h^y v^2
            + \tfrac12 \theta\, p^2 ,
        \qquad h = c^2 + \mathrm{Ro}\,p ,

    with the open-area face fractions ``alpha_u = fraction(u space)``,
    ``alpha_v = fraction(v space)`` and the wet plan-area
    ``theta = fraction(p space)``. Its time derivative gives five
    contributions (the geometry weights are constant): the h-weighted
    kinetic-energy rates and the ``0.5 u^2 dh`` corrections from
    ``dh = Ro dp``, plus ``p dp`` — exactly the walled ``h_energy_terms``
    pattern with the fraction weights inserted. Returns the five floats.
    """
    z = model.state
    dz = model.tendency(z)
    ro = float(model.parameters[fr.model.params.SCALING_NONLINEARITY])
    imm = model.grid.immersed
    u, v, p = z["u"], z["v"], z["p"]
    du, dv, dp = dz["u"], dz["v"], dz["p"]
    h = z["csqr"].to(p) + ro * p
    au = imm.fraction(u.function_space)
    av = imm.fraction(v.function_space)
    tp = imm.fraction(p.function_space)
    terms = (
        au * u * du * h.to(u),
        0.5 * ro * au * (u * u) * dp.to(u),
        av * v * dv * h.to(v),
        0.5 * ro * av * (v * v) * dp.to(v),
        tp * p * dp,
    )
    return [float(t.integrate().data.ravel()[0]) for t in terms]


# ================================================================
#  B-G1 (keystone): wet-weighted energy production is machine zero
# ================================================================
@pytest.mark.parametrize("periodic_x", [
    pytest.param(True, id="periodic-x"),
    pytest.param(False, id="walled-x"),
])
@pytest.mark.parametrize("seed", [3, 11])
def test_wet_weighted_energy_rate_is_machine_zero(periodic_x, seed):
    # SA-D1 + SA-D5: on genuine partial cells the alpha-weighted corner
    # mass fluxes and the fraction-weighted kinetic energy make the
    # gravity + Sadourny pair conserve the wet-weighted energy to
    # machine zero (semi-discrete). The un-fixed scheme produced a
    # ~2e-3 relative rate here (the antisymmetry break being fixed).
    model = _model(_partial_grid(periodic_x=periodic_x), f0=0.0)
    _fill_random(model, seed=seed)
    theta = np.asarray(model.grid.immersed.fraction(
        model.state["p"].function_space).data)
    assert ((theta > 1e-6) & (theta < 1.0 - 1e-6)).sum() > 0  # partials
    terms = _wet_energy_terms(model)
    scale = sum(abs(t) for t in terms)
    assert abs(sum(terms)) / scale < 1e-12, (sum(terms), scale)


# ================================================================
#  B-G2: wet mass production unregressed (machine zero, raw tendency)
# ================================================================
@pytest.mark.parametrize("periodic_x", [True, False])
def test_wet_mass_rate_is_machine_zero(periodic_x):
    # the fraction-weighted thickness transport is untouched by the
    # momentum fix; theta-weighted mass stays exactly conserved
    model = _model(_partial_grid(periodic_x=periodic_x), f0=1.0)
    _fill_random(model, seed=5)
    dp = model.tendency(model.state)["p"]
    tp = model.grid.immersed.fraction(model.state["p"].function_space)
    rate = float((tp * dp).integrate().data.ravel()[0])
    scale = float((tp * abs(dp)).integrate().data.ravel()[0])
    assert abs(rate) / scale < 1e-13


# ================================================================
#  B-G3: staircase immersed tendency == walled model (unregressed)
# ================================================================
def test_staircase_tendency_matches_walled_model():
    # a {0,1} staircase band (no partials) must reproduce the genuine
    # walled channel's Sadourny tendency: the fraction weighting
    # degenerates to boolean masking + free-slip corner closure
    npg, lo, hi = 12, 3, 9
    box = lambda x, y: ((y > lo) & (y < hi)).astype(float)  # noqa: E731, ARG005
    imm = _model(
        Grid((IM(npg, (0.0, 12.0), periodic=True, name="x"),
              IM(npg, (0.0, 12.0), periodic=True, name="y")),
             immersed=ImmersedDomain(box)), f0=0.7, dt=0.02)
    wal = _model(
        Grid((IM(npg, (0.0, 12.0), periodic=True, name="x"),
              IM(hi - lo, (float(lo), float(hi)), periodic=False,
                 name="y"))), f0=0.7, dt=0.02)
    rng = np.random.default_rng(11)
    shp = {k: wal.state[k].data.shape for k in ("u", "v", "p")}
    ic = {k: 0.1 * rng.standard_normal(shp[k]) for k in ("u", "v", "p")}
    wal.set_fields(**ic)
    full = {}
    for k in ("u", "v", "p"):
        arr = np.zeros(imm.state[k].data.shape)
        arr[:, lo:lo + shp[k][1]] = ic[k]
        full[k] = arr
    imm.set_fields(**full)
    di = imm.tendency(imm.state)
    dw = wal.tendency(wal.state)
    for k in ("u", "v", "p"):
        s = wal.state[k].data.shape
        sub = np.asarray(di[k].data)[:, lo:lo + s[1]]
        diff = np.abs(sub - np.asarray(dw[k].data)).max()
        assert diff < 1e-12, (k, diff)


# ================================================================
#  B-G4: all-wet immersed tendency == unimmersed (machine precision)
# ================================================================
@pytest.mark.parametrize("periodic_y", [
    pytest.param(True, id="periodic-y"),
    pytest.param(False, id="walled-y"),
])
def test_all_wet_tendency_matches_unimmersed(periodic_y):
    # At alpha = 1 / theta = 1 the fraction weights are a MATHEMATICAL
    # no-op: an eager reconstruction of every Sadourny intermediate
    # (fu/fv, the kinetic energy, the corner thickness, q, the assembled
    # du) is exactly bitwise across the two grids (verified 2026-07-19).
    # The residual is JIT-only and <= 1 ulp of the field scale (measured
    # 1.0 ulp worst case over seeds x periodic/walled): the immersed
    # branch's HLO carries the extra (identity-at-alpha=1) momentum
    # fraction ops -- the weight_flux on fu/fv and the /theta divide in
    # _wet_kinetic_energy -- and XLA contracts their stencil FMAs
    # differently from the flat branch (the documented FMA-contraction
    # class, core.py gravity docstring). The p tendency stays bitwise;
    # only the momentum shifts. The site is IRREDUCIBLE (it is the
    # mandated SA-D1/SA-D5 weighting); on walled it coincides in
    # magnitude with the PRE-EXISTING (unchanged) _gravity_immersed
    # artifact. Pinned to a few tens of ulp, NOT 1e-13.
    tol_ulps = 32
    def meshes():
        return (IM(10, (0.0, TWO_PI), periodic=True, name="x"),
                IM(10, (0.0, TWO_PI), periodic=periodic_y, name="y"))

    allwet = ImmersedDomain(lambda x, y: x * 0.0 + 1.0)  # noqa: ARG005
    im = _model(Grid(meshes(), immersed=allwet), f0=0.7)
    un = _model(Grid(meshes()), f0=0.7)
    rng = np.random.default_rng(7)
    ic = {k: 0.2 * rng.standard_normal(im.state[k].data.shape)
          for k in ("u", "v", "p")}
    im.set_fields(**ic)
    un.set_fields(**ic)
    di = im.tendency(im.state)
    du = un.tendency(un.state)
    for k in ("u", "v", "p"):
        a = np.asarray(di[k].data)
        b = np.asarray(du[k].data)
        scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1.0)
        diff = float(np.abs(a - b).max())
        assert diff <= tol_ulps * np.spacing(scale), (k, diff, scale)


# ================================================================
#  The fix, white-box: SA-D1 alpha-weighted corner mass fluxes
# ================================================================
def test_corner_mass_fluxes_are_alpha_weighted():
    # SA-D1: fu / fv are the SAME alpha-weighted mass fluxes the
    # thickness transport carries, interpolated to the corner — the
    # weight that restores the vorticity-flux antisymmetry. On a cut
    # face (alpha < 1) the weighted flux differs from the raw one.
    model = _model(_partial_grid(periodic_x=True), f0=0.0)
    _fill_random(model, seed=2)
    z = model.state
    imm = model.grid.immersed
    u, p, c = z["u"], z["p"], z["csqr"]
    ro = float(model.parameters[fr.model.params.SCALING_NONLINEARITY])
    p_full = c.to(p) + ro * p
    au = np.asarray(imm.fraction(u.function_space).data)
    weighted = np.asarray(weight_flux(imm, u * p_full.to(u)).data)
    raw = np.asarray((u * p_full.to(u)).data)
    np.testing.assert_allclose(weighted, au * raw, rtol=0, atol=1e-14)
    assert (au < 1.0 - 1e-9).any()          # a genuinely cut face
    assert np.abs(weighted - raw).max() > 1e-6   # weighting bites


# ================================================================
#  SA-D2: the wet-count corner thickness (never diluted by dry c^2)
# ================================================================
def test_wet_corner_thickness_drops_dry_cells():
    # a box mask: at a corner touching one dry cell the wet-count
    # average uses ONLY the wet neighbours, never the dry cell's
    # background c^2 (a plain 4-average would dilute it)
    box = lambda x, y: ((x > 2) & (x < 10)).astype(float)  # noqa: E731, ARG005
    grid = Grid((IM(12, (0.0, 12.0), periodic=True, name="x"),
                 IM(12, (0.0, 12.0), periodic=True, name="y")),
                immersed=ImmersedDomain(box))
    model = _model(grid, csqr=0.8, f0=0.0)
    _fill_random(model, seed=1)
    z = model.state
    imm = grid.immersed
    u, v, p, c = z["u"], z["v"], z["p"], z["csqr"]
    ro = float(model.parameters[fr.model.params.SCALING_NONLINEARITY])
    p_full = c.to(p) + ro * p
    corner = u.function_space.bare.replace(
        y=v.function_space.bare.factor("y"))
    zeta = v.diff("x").retag(corner) - u.diff("y").retag(corner)
    h_wet = np.asarray(_wet_corner_thickness(imm, p_full, zeta).data)
    h_plain = np.asarray(p_full.to(zeta).data)
    # the fraction of wet neighbours at each corner, in [0, 1]
    mask_c = imm.mask(p.function_space)
    m_float = mask_c.with_data(mask_c.data.astype(p_full.data.dtype))
    m_frac = np.asarray(m_float.to(zeta).data)
    # fully-wet corners: identical to the plain interpolation (bitwise);
    # partly-wet corners: the dry cell drops out, so h differs
    full = m_frac > 1.0 - 1e-9
    assert np.array_equal(h_wet[full], h_plain[full])
    cut = (m_frac > 1e-9) & (m_frac < 1.0 - 1e-9)
    assert cut.any()
    assert np.abs(h_wet[cut] - h_plain[cut]).max() > 1e-3


# ================================================================
#  SA-D5: the fraction-weighted kinetic energy (flat limit at all-wet)
# ================================================================
def test_wet_kinetic_energy_reduces_to_flat_when_all_wet():
    grid = Grid((IM(8, (0.0, 8.0), periodic=True, name="x"),
                 IM(8, (0.0, 8.0), periodic=True, name="y")),
                immersed=ImmersedDomain(lambda x, y: x * 0.0 + 1.0))  # noqa: ARG005
    model = _model(grid, f0=0.0)
    rng = np.random.default_rng(4)
    model.set_fields(**{
        k: 0.2 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "p")})
    z = model.state
    u, v, p = z["u"], z["v"], z["p"]
    ekin = np.asarray(_wet_kinetic_energy(grid.immersed, u, v, p).data)
    flat = np.asarray((0.5 * ((u * u).to(p) + (v * v).to(p))).data)
    np.testing.assert_array_equal(ekin, flat)


# ================================================================
#  B-G5: autodiff shard — grad through the new sealed divides
# ================================================================
def _leaf_loss(model, leaf, n_steps):
    """Quadratic loss splicing ``leaf`` into the carry via _chunk_body."""
    record = model._artifacts.record
    stepper = model._stepper
    leaves, treedef = jax.tree_util.tree_flatten(model._carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        carry = jax.tree_util.tree_unflatten(treedef, new)
        state = _chunk_body(record, n_steps, carry, stepper).state
        return (jnp.sum(state["p"].data ** 2)
                + jnp.sum(state["u"].data ** 2)
                + jnp.sum(state["v"].data ** 2))

    return loss


def test_immersed_grad_wrt_ic_is_finite_and_matches_fd():
    # the SA-D4 seals: the wet-count corner thickness and the
    # fraction-weighted kinetic energy each divide by a fraction that
    # is an exact zero on dry cells (numerator zero too — a masked
    # 0/0). Without the double-where seal the reverse mode NaNs every
    # gradient with a data path. Genuine partials (min_fraction=0) so
    # the divides are true partials, not a {0,1} staircase.
    model = _model(_partial_grid(periodic_x=True), f0=1.0)
    _fill_random(model, seed=0)
    p_leaf = model._carry.state["p"].storage
    loss = _leaf_loss(model, p_leaf, n_steps=8)

    grad = np.asarray(jax.grad(loss)(p_leaf))
    assert bool(np.all(np.isfinite(grad)))

    rng = np.random.default_rng(3)
    direction = jnp.asarray(rng.standard_normal(p_leaf.shape),
                            dtype=p_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(p_leaf + eps * direction))
          - float(loss(p_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


# ================================================================
#  Taught error: chart + immersed is unsupported (silent wrong physics)
# ================================================================
def test_chart_plus_immersed_is_a_taught_error():
    # a grid carrying BOTH a chart and an immersed domain must be
    # refused at bind: the metric chart advection path (_advect_chart)
    # is unmasked, so it would silently ignore the immersed mask and
    # advect across the wet-region boundary. sw2 mapped+immersed is a
    # recorded follow-up of the mapped+immersed composition plan.
    grid = fr.spatial.spherical.Grid(
        (16, 8), radius=1.0, lat_extent=(-1.0, 1.0), device_ids=(0,))
    grid = grid.with_immersed(
        ImmersedDomain(lambda lon, lat: lon * 0.0 + 1.0))  # noqa: ARG005
    with pytest.raises(NotImplementedError,
                       match="BOTH an embedding chart"):
        sw.Model(
            grid=grid,
            core=sw.Core(froude_number=0.3, depth=0.7,
                         coords=("lon", "lat")),
            scaling=fr.scaling.GravityWave(),
            coriolis=None, advection=True,
            time_stepper=fr.model.time_steppers.AdamBashforth(2e-3))


# ================================================================
#  B-G6: forced-4 device-count invariance of the immersed fix
# ================================================================
@pytest.mark.multi_device
def test_immersed_energy_fix_is_device_count_invariant(forced_devices):
    # the fraction-weighted momentum is halo-trace exempt (the fraction
    # fields are materialized, not traced); the sharded run must match
    # the single-device run. y is periodic (sharded); the genuine
    # partials vary in the walled x.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        grid = _partial_grid(periodic_x=False, device_ids=device_ids)
        model = _model(grid, f0=0.7)
        rng = np.random.default_rng(4)
        mask = np.asarray(
            grid.immersed.mask(model.state["p"].function_space).data)
        model.set_fields(
            u=rng.standard_normal(model.state["u"].data.shape),
            v=rng.standard_normal(model.state["v"].data.shape),
            p=rng.standard_normal(model.state["p"].data.shape) * mask)
        model.advance(6)
        results[tag] = {c: np.asarray(model.state[c].data)
                        for c in ("u", "v", "p")}
        if tag == "many" and forced_devices is not None:
            assert "devices" in model.state["u"]._data.sharding.spec
    assert max(
        float(np.abs(results["many"][c] - results["one"][c]).max())
        for c in ("u", "v", "p")) < 1e-11
