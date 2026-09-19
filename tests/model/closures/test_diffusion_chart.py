r"""Harmonic / biharmonic closures on an embedding chart.

Prefix-mirrored shard of ``fr.model.closures.diffusion`` covering the
orthogonal thin-shell Laplace-Beltrami legs (spherical-models plan):
the identity-chart bitwise reduction, the analytic sphere Laplacian
(the :math:`l = 1` harmonic), sqrt(g)-weighted conservation across the
polar-cap walls, the ``slip='no'`` refusal, and the house autodiff
regression. The hydrostatic core is the chart-capable 3-D host.
Self-contained builders (AGENTS oversized-module rule).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.closures.diffusion import (
    BiharmonicFriction,
    HarmonicDiffusion,
    HarmonicFriction,
)
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

IM = fr.spatial.meshes.IntervalMesh
RADIUS = 2.0
HOR = ("lon", "lat")


def _sphere(nlon=16, nlat=32, nz=4):
    return fr.spatial.spherical.Grid(
        (nlon, nlat), radius=RADIUS, lat_extent=(-1.2, 1.2),
        vertical=IM(nz, (-1.0, 0.0), periodic=False, name="z"))


def _model(grid, closures, hor=HOR, dt=1e-4):
    return hy.Model(
        grid=grid, core=hy.Core(gravity=1.0, horizontal=hor),
        time_stepper=AdamBashforth(dt, order=3),
        buoyancy=hy.BuoyancyTracer(),
        free_surface=hy.ExplicitFreeSurface(horizontal=hor),
        modules_extra=closures)


def _area_sum(field):
    weight = np.asarray(field.grid.metric(
        field.function_space.bare, "sqrt_g").data)
    return float((weight * np.asarray(field.data)).sum())


def test_identity_chart_closures_are_bitwise_flat():
    def meshes():
        return (IM(8, (0.0, 1.0), name="x"),
                IM(8, (0.0, 1.0), periodic=False, name="y"),
                IM(4, (-1.0, 0.0), periodic=False, name="z"))

    def closures():
        return [HarmonicDiffusion(kappa=1e-2, kappa_v=1e-3),
                HarmonicFriction(nu=2e-2, nu_v=1e-3),
                BiharmonicFriction(nu=1e-4)]

    chart = _model(fr.spatial.Grid(
        meshes(), mapping=fr.spatial.CoordinateMapping(
            chart={"X": lambda x, y: (x, y, 0.0 * x)},
            orthogonal=True)), closures(), hor=("x", "y"))
    flat = _model(fr.spatial.Grid(meshes()), closures(), hor=("x", "y"))
    rng = np.random.default_rng(2)
    fields = {k: rng.standard_normal(flat.state[k].shape)
              for k in ("u", "v", "b")}
    chart.set_fields(**fields)
    flat.set_fields(**fields)
    with jax.disable_jit():
        dzc = chart.tendency(chart.state)
        dzf = flat.tendency(flat.state)
    for name in ("u", "v", "b"):
        assert np.array_equal(np.asarray(dzc[name].data),
                              np.asarray(dzf[name].data))


def test_sphere_laplacian_of_the_first_harmonic():
    # q = sin(lat) is the l = 1 spherical harmonic:
    # laplace q = -l (l + 1) q / a^2 = -2 sin(lat) / a^2
    kappa = 0.7
    model = _model(_sphere(), [HarmonicDiffusion(kappa=kappa)])
    model.set_fields(b=lambda lon, lat, z: jnp.sin(lat) + 0.0 * (lon + z))
    tend = model.tendency(model.state)["b"]
    lat = np.asarray(model.grid.evaluation_nodes(
        tend.function_space, "lat").data)
    exact = -2.0 * kappa * np.sin(lat) / RADIUS ** 2 * np.ones(tend.shape)
    got = np.asarray(tend.data)
    # interior rows (the cap rows close with the no-flux wall)
    err = np.abs(got[:, 1:-1] - exact[:, 1:-1]).max()
    assert err < 5e-3 * np.abs(exact).max()


@pytest.mark.parametrize("closure", [
    pytest.param(lambda: HarmonicDiffusion(kappa=0.3, kappa_v=0.1),
                 id="harmonic"),
])
def test_chart_diffusion_conserves_the_area_weighted_content(closure):
    model = _model(_sphere(16, 8), [closure()])
    rng = np.random.default_rng(6)
    model.set_fields(b=rng.standard_normal(model.state["b"].shape))
    tend = model.tendency(model.state)["b"]
    assert abs(_area_sum(tend)) < 1e-12 * _area_sum(abs(tend))


def test_no_slip_on_a_chart_is_a_taught_error():
    with pytest.raises(NotImplementedError, match="slip='no'"):
        _model(_sphere(16, 8), [HarmonicFriction(nu=1e-2, slip="no")])


def test_grad_wrt_viscosity_through_a_spherical_run_matches_fd():
    model = _model(
        _sphere(16, 8),
        [HarmonicFriction(nu=5e-2), HarmonicDiffusion(kappa=5e-2),
         BiharmonicFriction(nu=1e-3)], dt=1e-3)
    rng = np.random.default_rng(8)
    model.set_fields(**{
        k: 0.1 * rng.standard_normal(model.state[k].shape)
        for k in ("u", "v", "b")})
    run = model.propagator(wrt=("friction.nu",), steps=6)

    def loss(nu):
        state = run((nu,)).state
        return sum(jnp.sum(state[name].data ** 2)
                   for name in ("u", "v", "b"))

    nu0 = jnp.asarray(5e-2)
    grad = float(jax.grad(loss)(nu0))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    eps = 1e-5
    fd = (float(loss(nu0 + eps)) - float(loss(nu0 - eps))) / (2 * eps)
    assert grad == pytest.approx(fd, rel=1e-4)
