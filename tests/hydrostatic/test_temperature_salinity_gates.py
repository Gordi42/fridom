r"""``hy.TemperatureSalinity``: terrain, autodiff and device gates.

Prefix-mirrored shard of ``test_temperature_salinity.py`` (AGENTS
oversized-module rule; self-contained builders):

- **sigma pressure-gradient error** — a horizontally uniform
  ``T(z_p)``, ``S(z_p)`` at rest over a seamount leaves a truncation-
  order residual that converges at second order, for the depth-
  dependent TEOS-10 polynomial as for a plain buoyancy profile;
- **autodiff** — ``jax.grad`` through a short run w.r.t. ``eos.alpha``,
  an initial ``T`` (TEOS-10) and an initial ``S`` on an immersed grid
  (the dry cells' ``S = 0`` under the TEOS-10 root) matches a central
  finite difference (AGENTS differentiability policy);
- **device-count invariance** on forced host devices.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.hydrostatic.params import EOS_ALPHA
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.immersed_domain import ImmersedDomain

IM = fr.spatial.meshes.IntervalMesh
G = 9.81
LENGTH = 1.0e5
DEPTH = 1000.0
DT = 50.0
ORDER_FLOOR = 1.7
CenteredAdvection = fr.model.modules.CenteredAdvection


# ================================================================
#  Builders
# ================================================================
def make_grid(n=8, nz=6, **kwargs):
    """Return a 100 km doubly-periodic, 1 km deep bounded-z grid."""
    return fr.spatial.Grid((
        IM(n, (0.0, LENGTH), periodic=True, name="x"),
        IM(n, (0.0, LENGTH), periodic=True, name="y"),
        IM(nz, (-DEPTH, 0.0), periodic=False, name="z")), **kwargs)


def bottom(x, y):
    """Return the depth of a +-20 % seamount / trough pattern [m]."""
    return DEPTH * (1.0 + 0.2 * jnp.sin(2 * jnp.pi * x / LENGTH)
                    * jnp.cos(2 * jnp.pi * y / LENGTH))


def terrain_grid(n):
    """Return the sigma grid ``zp = z H(x, y)``, ``z`` in (-1, 0)."""
    return fr.spatial.Grid(
        (IM(n, (0.0, LENGTH), periodic=True, name="x"),
         IM(n, (0.0, LENGTH), periodic=True, name="y"),
         IM(n, (-1.0, 0.0), periodic=False, name="z")),
        mapping=CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                  params={"H": bottom}))


def make_model(buoyancy, *, grid=None, advection=CenteredAdvection,
               **kwargs):
    """Assemble a dimensional hydrostatic model on the buoyancy module."""
    return hy.Model(
        grid=make_grid() if grid is None else grid,
        core=hy.Core(gravity=G),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=hy.FPlaneCoriolis(f0=1.0e-4),
        buoyancy=buoyancy,
        free_surface=hy.ExplicitFreeSurface(),
        advection=(advection() if isinstance(advection, type)
                   else advection), **kwargs)


def temperature(x, y, z):
    """Return a thermocline with a horizontal warm anomaly."""
    return (4.0 + 14.0 * np.exp(z / 300.0)
            + 0.5 * np.sin(2 * np.pi * x / LENGTH)
            * np.cos(2 * np.pi * y / LENGTH))


def salinity(x, y, z):
    """Return a halocline with a horizontal fresh anomaly."""
    return (35.0 - 0.8 * np.exp(z / 200.0)
            + 0.1 * np.cos(2 * np.pi * x / LENGTH) + 0.0 * y)


def physical_height(model):
    """Return ``zp`` at the tracer cells of a terrain model."""
    space = model.state["b"].function_space.bare
    grid = model.grid

    def at(name):
        return np.asarray(grid.evaluation_nodes(space, name).data)

    return at("z") * np.asarray(bottom(at("x"), at("y")))


def velocity_loss(final):
    """Return the quadratic loss of the final horizontal velocities."""
    return (jnp.sum(final.state["u"].data ** 2)
            + jnp.sum(final.state["v"].data ** 2))


# ================================================================
#  Gate: the sigma pressure-gradient error of a T/S rest state
# ================================================================
def _rest_tendency(n, buoyancy):
    model = make_model(buoyancy, grid=terrain_grid(n), advection=None,
                       modules_extra=(_Hold(),))
    zp = physical_height(model)
    model.set_fields(T=4.0 + 14.0 * np.exp(zp / 300.0),
                     S=35.0 - 0.8 * np.exp(zp / 200.0))
    tendency = model.tendency(model.state)
    return max(float(jnp.abs(tendency["u"].data).max()),
               float(jnp.abs(tendency["v"].data).max()))


class _Hold(fr.model.Module):

    """Advance ``T`` / ``S`` by a zero term (a linear assembly's lint)."""

    field_references = (fr.model.FieldReference("T"),
                        fr.model.FieldReference("S"))

    @fr.model.term(advances=("T", "S"), linear=True)
    def hold(self, state, _ctx):
        """Return a zero tendency for both tracers."""
        return {"T": 0.0 * state["T"], "S": 0.0 * state["S"]}


@pytest.mark.parametrize(
    "make_eos", [pytest.param(hy.LinearEOS, id="linear"),
                 pytest.param(hy.TEOS10EOS, id="teos10")])
def test_terrain_rest_state_pressure_gradient_error_converges(make_eos):
    # horizontally uniform T(zp), S(zp) over a seamount: the residual
    # acceleration is the sigma-coordinate truncation error of the
    # slope-corrected pressure gradient, and converges at second order.
    # Measured spurious acceleration at n = 16, 32, 64 [m/s^2]:
    #   linear  4.30e-6, 1.13e-6, 2.91e-7   (orders 1.93, 1.96)
    #   teos10  4.49e-6, 1.34e-6, 3.80e-7   (orders 1.75, 1.82)
    # i.e. a spurious geostrophic current PGE / f of 4 cm/s at 16
    # sigma levels over a +-20 % seamount with a 300 m thermocline,
    # 0.3-0.4 cm/s at 64: the usual sigma-coordinate figure, inherited
    # from the core's gradient (a plain b profile gives the linear row).
    errs = np.array([_rest_tendency(n, hy.TemperatureSalinity(make_eos()))
                     for n in (16, 32, 64)])
    orders = np.log2(errs[:-1] / errs[1:])
    assert np.all(orders > ORDER_FLOOR)
    assert errs[0] < 6e-6
    assert errs[-1] < 5e-7


def test_same_depth_reference_keeps_compressibility_out_of_the_error():
    # the dynamic anomaly is measured against a reference parcel AT THE
    # SAME DEPTH; the bulk compressibility (4.5 kg/m^3 per km, ten
    # times the stratification) therefore never reaches the sigma
    # pressure gradient: the TEOS-10 residual stays within a small
    # factor of the linear EOS residual for the same T(zp), S(zp)
    # (measured 1.04x at n = 16)
    linear = _rest_tendency(16, hy.TemperatureSalinity(hy.LinearEOS()))
    full = _rest_tendency(16, hy.TemperatureSalinity(hy.TEOS10EOS()))
    assert full < 1.5 * linear


# ================================================================
#  Gate: autodiff through a short run (FD-matched, rtol 1e-4)
# ================================================================
def central_fd(loss, x0, eps=1e-4):
    """Central finite difference of ``loss`` at the scalar ``x0``."""
    h = eps * abs(float(x0))
    return (float(loss(x0 + h)) - float(loss(x0 - h))) / (2.0 * h)


def test_grad_wrt_eos_alpha_matches_fd():
    model = make_model(hy.TemperatureSalinity(hy.LinearEOS()))
    model.set_fields(T=temperature, S=salinity)
    run = model.propagator(wrt=(EOS_ALPHA,), steps=8)
    alpha0 = jnp.asarray(hy.LinearEOS().tunable["alpha"])

    def loss(alpha):
        return velocity_loss(run((alpha,)))

    grad = float(jax.grad(loss)(alpha0))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    assert grad == pytest.approx(central_fd(loss, alpha0), rel=1e-4)


def _directional_check(model, name, steps=8):
    run = model.propagator(wrt=(name,), steps=steps)
    leaf = model._carry.state[name].storage

    def loss(data):
        return velocity_loss(run((data,)))

    grad = np.asarray(jax.grad(loss)(leaf))
    assert bool(np.all(np.isfinite(grad)))
    rng = np.random.default_rng(3)
    direction = jnp.asarray(rng.standard_normal(leaf.shape),
                            dtype=leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    assert abs(directional) > 0.0
    # the quadratic loss of a polynomial EOS is quartic in the tracer:
    # the central FD converges to the gradient as eps -> 0 (measured
    # rel. 2e-4 at eps = 1e-3, 2e-6 at 1e-4)
    eps = 1e-4
    fd = (float(loss(leaf + eps * direction))
          - float(loss(leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


def test_grad_wrt_initial_temperature_matches_fd_teos10():
    model = make_model(hy.TemperatureSalinity(hy.TEOS10EOS()))
    model.set_fields(T=temperature, S=salinity)
    _directional_check(model, "T")


def test_grad_wrt_initial_salinity_matches_fd_on_an_immersed_grid():
    # dry cells hold S = 0 under the TEOS-10 root sqrt((S + 32) / S_u):
    # forward-finite and reverse-finite (no masked singularity), and the
    # masked b keeps the dry cells out of the gradient
    def wet(x, y, z):  # noqa: ARG001
        return (z > -0.5 * DEPTH).astype(float)

    grid = make_grid(nz=8, immersed=ImmersedDomain(wet))
    model = make_model(hy.TemperatureSalinity(hy.TEOS10EOS()), grid=grid)
    model.set_fields(T=temperature, S=salinity)
    _directional_check(model, "S")


# ================================================================
#  Gate: device-count invariance (forced host devices)
# ================================================================
def _run(device_ids, make_eos):
    grid = fr.spatial.Grid(
        (IM(8, (0.0, LENGTH), periodic=True, name="x"),
         IM(8, (0.0, LENGTH), periodic=True, name="y"),
         IM(16, (-DEPTH, 0.0), periodic=False, name="z")),
        device_ids=device_ids)
    model = make_model(hy.TemperatureSalinity(make_eos()), grid=grid,
                       chunk_size=8)
    model.set_fields(T=temperature, S=salinity)
    model.advance(8)
    sharded = [name for name, _
               in grid.decomposition.default_layout.device_axes]
    return ({k: np.asarray(model.state[k].data)
             for k in ("u", "v", "T", "S", "b", "ps")}, sharded)


@pytest.mark.multi_device
@pytest.mark.parametrize(
    "make_eos", [pytest.param(hy.LinearEOS, id="linear"),
                 pytest.param(hy.TEOS10EOS, id="teos10")])
def test_device_count_invariance(forced_devices, make_eos):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many, sharded = _run(None, make_eos)
    one, _ = _run((0,), make_eos)
    assert sharded
    for name in ("u", "v", "T", "S", "b", "ps"):
        a, b = many[name], one[name]
        if jax.default_backend() == "cpu":
            # forced-host reassociation floor, relative to the field
            scale = max(np.abs(b).max(), 1e-300)
            assert np.abs(a - b).max() <= 1e-11 * scale, name
        else:
            assert np.array_equal(a, b), name
