r"""The masked sphere, linear half: ``sw.Core`` on chart + immersed grids.

The prefix-mirrored shard (AGENTS oversized-module rule) of
``shallowwater2/modules/core.py`` covering the chart + immersed
composition of the wave term — an orthogonal chart (the lat-lon
sphere) carrying a full-cell staircase land mask — and the
two-dimensional static depth ``D(zonal, meridional)`` that real
bathymetry needs. Gates:

- all-wet mask == unmasked sphere (tendency; pinned in ulp);
- identity chart + mask == flat immersed model;
- a latitude-band staircase == the genuinely walled sphere band;
- theta * sqrt_g mass and the masked linear energy: machine-zero rates,
  and mass to rounding over a run in a closed basin;
- a resting ocean over bathymetry with land stays exactly at rest;
- no flow through land faces, dry cells stay dead;
- reverse-mode autodiff through a masked-sphere run (finite, FD);
- the taught errors for what stays unsupported.

Each test is self-contained (the shard duplicates the small builders).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model.model import _chunk_body
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

IM = IntervalMesh
LAT_MAX = 1.2
NLON, NLAT = 16, 12
OMEGA = 1.5


# ================================================================
#  Builders (self-contained per the AGENTS oversized-module rule)
# ================================================================
def _basin(lon, lat):
    """Return a closed basin with an island (full-cell staircase)."""
    ocean = (lon > 0.9) & (lon < 5.2) & (jnp.abs(lat) < 0.85)
    island = (jnp.abs(lon - 3.0) < 0.45) & (jnp.abs(lat - 0.1) < 0.25)
    return (ocean & ~island).astype(float)


def _all_wet(lon, lat):
    return 1.0 + 0.0 * lon + 0.0 * lat


def _bathymetry(lon, lat):
    """Return a smooth positive 2-D depth D(lon, lat)."""
    return 0.7 + 0.2 * jnp.sin(lon) * jnp.cos(lat) + 0.1 * jnp.cos(2 * lat)


def _sphere(immersed=None, *, nlon=NLON, nlat=NLAT,
            lat_extent=(-LAT_MAX, LAT_MAX)):
    return fr.spatial.spherical.Grid(
        (nlon, nlat), radius=1.0, lat_extent=lat_extent,
        immersed=immersed)


def _model(grid, *, depth=_bathymetry, rotation=True,
           coords=("lon", "lat"), dt=2e-3):
    """Build the linear dimensional model (gravity 1)."""
    coriolis = None
    if rotation:
        coriolis = sw.modules.RotationCoriolis(
            omega=(0.0, 0.0, OMEGA), coords=coords,
            metric_weight="csqr")
    return sw.Model(
        grid=grid,
        core=sw.Core(gravity=1.0, depth=depth, coords=coords),
        coriolis=coriolis, advection=None,
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))


def _fill_random(model, seed=3, amp=0.1):
    """Random prognostics, zeroed on the dry DOFs (the MaskState law)."""
    rng = np.random.default_rng(seed)
    imm = model.grid.immersed
    fields = {}
    for name in ("u", "v", "p"):
        data = amp * rng.standard_normal(model.state[name].data.shape)
        if imm is not None:
            data = data * np.asarray(
                imm.mask(model.state[name].function_space).data)
        fields[name] = data
    model.set_fields(**fields)
    return fields


def _total(field):
    return float(field.integrate().data.ravel()[0])


# ================================================================
#  The two-dimensional static depth D(zonal, meridional)
# ================================================================
def test_two_parameter_depth_is_the_2d_csqr_field():
    core = sw.Core(gravity=2.0, depth=_bathymetry, coords=("lon", "lat"))
    assert core.variable_depth
    model = sw.Model(
        grid=_sphere(), core=core,
        time_stepper=fr.model.time_steppers.AdamBashforth(1e-3))
    csqr, p = model.state["csqr"], model.state["p"]
    assert csqr.data.shape == p.data.shape
    lon = model.grid.evaluation_nodes(p.function_space, "lon").data
    lat = model.grid.evaluation_nodes(p.function_space, "lat").data
    np.testing.assert_allclose(
        np.asarray(csqr.data), 2.0 * np.asarray(_bathymetry(lon, lat)),
        rtol=1e-14)


def test_one_parameter_depth_stays_the_meridional_profile():
    model = _model(_sphere(), depth=lambda y: 0.7 + 0.1 * jnp.cos(y),
                   rotation=False)
    assert model.state["csqr"].data.shape == (1, NLAT)


def test_2d_depth_under_a_gravity_ramp_is_rewritten_per_stage():
    ramp = fr.model.Ramp(1.0, 2.0, period=1.0)
    model = sw.Model(
        grid=_sphere(),
        core=sw.Core(gravity=ramp, depth=_bathymetry,
                     coords=("lon", "lat")),
        time_stepper=fr.model.time_steppers.AdamBashforth(1e-2))
    before = np.asarray(model.state["csqr"].data).copy()
    model.advance(3)
    after = np.asarray(model.state["csqr"].data)
    assert after.shape == before.shape
    assert np.all(after > before)


def test_depth_signature_taught_errors():
    with pytest.raises(TypeError, match="two-dimensional depth"):
        sw.Core(gravity=1.0, depth=lambda x, y, z: x + y + z)
    with pytest.raises(TypeError, match="drop meridional="):
        sw.Core(gravity=1.0, depth=_bathymetry, meridional="lat",
                coords=("lon", "lat"))


# ================================================================
#  All-wet mask == unmasked sphere
# ================================================================
def test_all_wet_tendency_matches_unmasked_sphere():
    plain = _model(_sphere())
    masked = _model(_sphere(ImmersedDomain(_all_wet)))
    fields = _fill_random(plain)
    masked.set_fields(**fields)
    dp, dm = plain.tendency(plain.state), masked.tendency(masked.state)
    for name in ("u", "v", "p"):
        ref = np.asarray(dp[name].data)
        err = np.max(np.abs(np.asarray(dm[name].data) - ref))
        ulp = np.finfo(ref.dtype).eps * np.max(np.abs(ref))
        assert err <= 4 * ulp, (name, err / ulp)


def test_all_wet_run_matches_unmasked_sphere():
    plain = _model(_sphere())
    masked = _model(_sphere(ImmersedDomain(_all_wet)))
    fields = _fill_random(plain)
    masked.set_fields(**fields)
    plain.advance(20)
    masked.advance(20)
    for name in ("u", "v", "p"):
        np.testing.assert_allclose(
            np.asarray(masked.state[name].data),
            np.asarray(plain.state[name].data), rtol=0, atol=1e-14)


# ================================================================
#  Identity chart + mask == flat immersed
# ================================================================
def _flat_box(x, y):
    return ((x > 1.0) & (x < 5.0) & (y > 1.5) & (y < 4.5)).astype(float)


def test_identity_chart_mask_matches_flat_immersed():
    def meshes():
        return (IM(12, (0.0, 6.0), periodic=True, name="x"),
                IM(12, (0.0, 6.0), periodic=False, name="y"))

    depth = lambda x, y: 0.8 + 0.1 * jnp.sin(x) + 0.05 * y  # noqa: E731
    flat = _model(
        Grid(meshes(), immersed=ImmersedDomain(_flat_box)),
        depth=depth, rotation=False, coords=("x", "y"))
    chart = _model(
        Grid(meshes(), immersed=ImmersedDomain(_flat_box),
             mapping=fr.spatial.CoordinateMapping(
                 chart={"X": lambda x, y: (x, y, 0.0 * x)},
                 orthogonal=True)),
        depth=depth, rotation=False, coords=("x", "y"))
    fields = _fill_random(flat)
    chart.set_fields(**fields)
    df, dc = flat.tendency(flat.state), chart.tendency(chart.state)
    for name in ("u", "v", "p"):
        np.testing.assert_allclose(
            np.asarray(dc[name].data), np.asarray(df[name].data),
            rtol=0, atol=1e-14)
    flat.advance(20)
    chart.advance(20)
    for name in ("u", "v", "p"):
        np.testing.assert_allclose(
            np.asarray(chart.state[name].data),
            np.asarray(flat.state[name].data), rtol=0, atol=1e-14)


# ================================================================
#  A latitude-band staircase == the genuinely walled sphere band
# ================================================================
def test_lat_band_staircase_matches_walled_sphere():
    lo, hi = 3, 9
    dlat = 2 * LAT_MAX / NLAT
    lat_lo, lat_hi = -LAT_MAX + lo * dlat, -LAT_MAX + hi * dlat
    band = lambda lon, lat: (  # noqa: E731
        (lat > lat_lo) & (lat < lat_hi)).astype(float) + 0.0 * lon
    imm = _model(_sphere(ImmersedDomain(band)))
    wal = _model(_sphere(nlat=hi - lo, lat_extent=(lat_lo, lat_hi)))
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
    # the bathymetry is sampled at the same physical nodes in the band
    np.testing.assert_allclose(
        np.asarray(imm.state["csqr"].data)[:, lo:hi],
        np.asarray(wal.state["csqr"].data), rtol=1e-13)
    imm.advance(15)
    wal.advance(15)
    for k in ("u", "v", "p"):
        sub = np.asarray(imm.state[k].data)[:, lo:lo + shp[k][1]]
        np.testing.assert_allclose(
            sub, np.asarray(wal.state[k].data), rtol=0, atol=1e-13)


# ================================================================
#  Conservation: theta * sqrt_g mass and the masked linear energy
# ================================================================
def test_mass_and_energy_rates_are_machine_zero():
    model = _model(_sphere(ImmersedDomain(_basin)))
    _fill_random(model, seed=5)
    imm = model.grid.immersed
    z = model.state
    dz = model.tendency(z)
    u, v, p = z["u"], z["v"], z["p"]
    theta = imm.fraction(p.function_space)
    rate = _total(theta * dz["p"])
    scale = _total(theta * abs(dz["p"]))
    assert scale > 0
    assert abs(rate) / scale < 1e-13
    # E = sum sqrt_g (alpha c^2 U^2 / 2 + theta p^2 / 2): gravity is
    # skew under it and the c^2-weighted rotation does no work
    csqr = z["csqr"]
    terms = (
        imm.fraction(u.function_space) * csqr.to(u) * u * dz["u"],
        imm.fraction(v.function_space) * csqr.to(v) * v * dz["v"],
        theta * p * dz["p"],
    )
    rates = [_total(t) for t in terms]
    assert abs(sum(rates)) / sum(abs(r) for r in rates) < 1e-12


def test_closed_basin_mass_is_conserved_to_rounding():
    model = _model(_sphere(ImmersedDomain(_basin)))
    theta = model.grid.immersed.fraction(
        model.state["p"].function_space)
    mask = np.asarray(theta.data)
    model.set_fields(
        p=lambda lon, lat: 0.5 + 0.05 * jnp.sin(lon) * jnp.cos(lat),
        u=lambda lon, lat: 0.05 * jnp.cos(lat) + 0.0 * lon)
    # the MaskState constraint owns the dry DOFs from the first step
    model.set_fields(p=np.asarray(model.state["p"].data) * mask)
    mass0 = _total(theta * model.state["p"])
    model.advance(300)
    mass1 = _total(theta * model.state["p"])
    assert abs(mass1 - mass0) / abs(mass0) < 1e-13
    assert np.ptp(np.asarray(model.state["p"].data)[mask > 0]) > 1e-3


# ================================================================
#  Rest stays rest; no flow through land
# ================================================================
def test_resting_ocean_over_bathymetry_with_land_stays_at_rest():
    model = _model(_sphere(ImmersedDomain(_basin)))
    model.advance(25)
    for name in ("u", "v", "p"):
        assert not np.any(np.asarray(model.state[name].data))


def test_no_flow_through_land_faces():
    model = _model(_sphere(ImmersedDomain(_basin)))
    _fill_random(model, seed=7)
    imm = model.grid.immersed
    dz = model.tendency(model.state)
    model.advance(40)
    for name in ("u", "v", "p"):
        dry = np.asarray(
            imm.mask(model.state[name].function_space).data) == 0
        assert dry.any()
        assert not np.any(np.asarray(model.state[name].data)[dry])
        wet_values = np.asarray(model.state[name].data)[~dry]
        assert np.all(np.isfinite(wet_values))
        assert np.any(wet_values)
    # the raw continuity tendency of a dry cell is an exact zero
    dry_p = np.asarray(imm.mask(dz["p"].function_space).data) == 0
    assert not np.any(np.asarray(dz["p"].data)[dry_p])


# ================================================================
#  Autodiff: grad through a masked-sphere run (finite, FD-matched)
# ================================================================
def test_masked_sphere_ic_grad_is_finite_and_matches_fd():
    model = _model(_sphere(ImmersedDomain(_basin), nlon=12, nlat=8))
    _fill_random(model)
    record, stepper = model._artifacts.record, model._stepper
    p_leaf = model._carry.state["p"].storage
    leaves, treedef = jax.tree_util.tree_flatten(model._carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is p_leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        carry = jax.tree_util.tree_unflatten(treedef, new)
        state = _chunk_body(record, 6, carry, stepper).state
        return (jnp.sum(state["p"].data ** 2)
                + jnp.sum(state["u"].data ** 2)
                + jnp.sum(state["v"].data ** 2))

    grad = np.asarray(jax.grad(loss)(p_leaf))
    assert bool(np.all(np.isfinite(grad)))
    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(p_leaf.shape),
                            dtype=p_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(p_leaf + eps * direction))
          - float(loss(p_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


# ================================================================
#  Taught errors: what stays unsupported
# ================================================================
def test_partial_cells_on_the_sphere_are_a_taught_error():
    grid = _sphere(ImmersedDomain(_basin, order=2))
    with pytest.raises(NotImplementedError, match="full-cell staircase"):
        _model(grid)


def test_immersed_on_a_non_orthogonal_chart_is_a_taught_error():
    grid = Grid(
        (IM(8, (0.0, 1.0), periodic=True, name="x"),
         IM(8, (0.0, 1.0), periodic=True, name="y")),
        mapping=fr.spatial.CoordinateMapping(
            chart={"X": lambda x, y: (x + 0.4 * y, y, 0.0 * x)}),
        immersed=ImmersedDomain(lambda x, y: x * 0.0 + 1.0))  # noqa: ARG005
    with pytest.raises(NotImplementedError, match="declared orthogonal"):
        _model(grid, depth=0.7, rotation=False, coords=("x", "y"))
