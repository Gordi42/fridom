"""Parameterful hydrostatic diagnostics: ekin, epot, eta and b_total."""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.coordinate_mapping import CoordinateMapping

IM = fr.spatial.meshes.IntervalMesh


def make_grid(nx=8, nz=4):
    """Return a doubly-periodic horizontal, bounded-vertical grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, 1.0), periodic=False, name="z")))


def make_model(n2=2.0, gravity=1.0):
    """Return a minimal linear hydrostatic model (advection=None)."""
    return hy.Model(
        grid=make_grid(),
        core=hy.Core(gravity=gravity),
        time_stepper=AdamBashforth(1e-3, order=3),
        coriolis=hy.FPlaneCoriolis(f0=1.0),
        buoyancy=hy.ConstantStratification(n2=n2),
        free_surface=hy.ExplicitFreeSurface(),
        advection=None)


def test_ekin_is_half_the_horizontal_speed_squared():
    model = make_model()
    rng = np.random.default_rng(0)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape))
    ekin = model.diagnostics.ekin()
    state = model.state
    centre = state["p_hyd"].function_space
    u_c = np.asarray(state["u"].to(centre).data)
    v_c = np.asarray(state["v"].to(centre).data)
    expected = 0.5 * (u_c**2 + v_c**2)
    assert np.allclose(np.asarray(ekin.data), expected)
    assert ekin.name == "ekin"


def test_ekin_is_zero_at_rest():
    model = make_model()
    ekin = model.diagnostics.ekin()
    assert float(np.abs(ekin.data).max()) == 0.0


def test_epot_is_half_b_squared_over_n2():
    n2 = 3.0
    model = make_model(n2=n2)
    rng = np.random.default_rng(1)
    model.set_fields(b=rng.standard_normal(model.state["b"].shape))
    epot = model.diagnostics.epot()
    b = np.asarray(model.state["b"].data)
    assert np.allclose(np.asarray(epot.data), 0.5 * b**2 / n2)
    assert epot.name == "epot"


def test_epot_can_evaluate_on_a_passed_state():
    n2 = 2.0
    model = make_model(n2=n2)
    rng = np.random.default_rng(2)
    b = model.grid.create_field(
        model.state["b"].function_space,
        data=rng.standard_normal(model.state["b"].shape))
    state = model.state.replace(b=b)
    epot = model.diagnostics.epot(state)
    assert np.allclose(np.asarray(epot.data),
                       0.5 * np.asarray(b.data)**2 / n2)


def test_eta_is_surface_pressure_over_gravity():
    gravity = 2.5
    model = make_model(gravity=gravity)
    rng = np.random.default_rng(4)
    model.set_fields(ps=rng.standard_normal(model.state["ps"].shape))
    eta = model.diagnostics.eta()
    ps = np.asarray(model.state["ps"].data)
    assert np.allclose(np.asarray(eta.data), ps / gravity)
    assert eta.name == "eta"
    assert eta.xr.attrs["units"] == "m"


def test_eta_can_evaluate_on_a_passed_state():
    gravity = 3.0
    model = make_model(gravity=gravity)
    rng = np.random.default_rng(5)
    ps = model.grid.create_field(
        model.state["ps"].function_space,
        data=rng.standard_normal(model.state["ps"].shape))
    state = model.state.replace(ps=ps)
    eta = model.diagnostics.eta(state)
    assert np.allclose(np.asarray(eta.data),
                       np.asarray(ps.data) / gravity)


# ================================================================
#  b_total: the anomaly plus the ConstantStratification background
# ================================================================
def test_b_total_adds_the_background_stratification():
    n2 = 3.0
    model = make_model(n2=n2)
    rng = np.random.default_rng(6)
    model.set_fields(b=rng.standard_normal(model.state["b"].shape))
    total = model.diagnostics.b_total()
    b = model.state["b"]
    z = np.asarray(b.evaluation_nodes("z").data)
    assert np.allclose(np.asarray(total.data),
                       np.asarray(b.data) + n2 * z)
    assert total.function_space is b.function_space
    assert total.name == "b_total"
    assert total.xr.attrs["units"] == "m/s^2"


def test_b_total_can_evaluate_on_a_passed_state():
    n2 = 2.0
    model = make_model(n2=n2)
    rng = np.random.default_rng(7)
    b = model.grid.create_field(
        model.state["b"].function_space,
        data=rng.standard_normal(model.state["b"].shape))
    state = model.state.replace(b=b)
    total = model.diagnostics.b_total(state)
    z = np.asarray(b.evaluation_nodes("z").data)
    assert np.allclose(np.asarray(total.data),
                       np.asarray(b.data) + n2 * z)


def test_b_total_is_contributed_by_the_stratification_module():
    # without a ConstantStratification there is no background to add
    model = hy.Model(
        grid=make_grid(),
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(1e-3, order=3),
        coriolis=hy.FPlaneCoriolis(f0=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=None)
    with pytest.raises(AttributeError,
                       match="no diagnostic named 'b_total'"):
        _ = model.diagnostics.b_total


def test_b_total_nondimensional_background_is_n2_z_in_physical_units():
    # Rotational frame: at rest b_total = (eps/Fr^2) z (the advection
    # carries eps, so the background gradient is N^2_eff / eps), and
    # the b_total unit row U^2/(eps H) turns it into N^2 z with the
    # Froude definition N = U/(Fr H) and z = H z
    length, speed, rossby, froude = 2.0e3, 0.5, 0.25, 0.125
    height = 1.0  # the vertical extent of make_grid
    model = hy.Model(
        grid=make_grid(),
        core=hy.Core(),
        scaling=fr.scaling.Rotational(L=length, U=speed),
        coriolis=hy.FPlaneCoriolis(rossby_number=rossby),
        buoyancy=hy.ConstantStratification(froude_number=froude),
        free_surface=hy.ExplicitFreeSurface(froude_number=0.25),
        advection=None,
        time_stepper=AdamBashforth(1e-3, order=3))
    total = model.diagnostics.b_total()
    z = np.asarray(model.state["b"].evaluation_nodes("z").data)
    assert np.allclose(np.asarray(total.data), rossby / froude**2 * z)
    n_freq = speed / (froude * height)
    physical = model.units.factor("b_total") * np.asarray(total.data)
    assert np.allclose(physical, n_freq**2 * height * z)


# ================================================================
#  b_total on a mapped column: the physical height, not the base z
# ================================================================
def sloped(x, y):
    """Sloped bottom, 20% of the mean depth."""
    return 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def make_column_grid(mapping, nx=4, nz=3):
    """Doubly-periodic horizontal, base column z in [-1, 0]."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (-1.0, 0.0), periodic=False, name="z")), mapping=mapping)


def make_column_model(grid, n2=2.0, extra=()):
    """Return a linear hydrostatic model with a stratification on grid."""
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(1e-3, order=3),
        buoyancy=hy.ConstantStratification(n2=n2),
        free_surface=hy.ExplicitFreeSurface(),
        advection=None,
        modules_extra=extra)


def column_nodes(b):
    """Broadcast-ready x, y, z node coordinates of the buoyancy."""
    return tuple(np.asarray(b.evaluation_nodes(name).data)
                 for name in ("x", "y", "z"))


def test_b_total_uses_the_physical_height_on_a_terrain_column():
    n2 = 2.0
    grid = make_column_grid(CoordinateMapping(
        maps={"zp": lambda z, H: z * H}, params={"H": sloped}))
    model = make_column_model(grid, n2=n2)
    total = model.diagnostics.b_total()      # at rest: the background
    b = model.state["b"]
    x, y, z = column_nodes(b)
    zp = np.asarray(sloped(jnp.asarray(x), jnp.asarray(y))) * z
    assert np.allclose(np.asarray(total.data), n2 * zp)
    # the sloped column differs from the base coordinate it maps
    assert not np.allclose(zp, np.broadcast_to(z, zp.shape))


def test_b_total_follows_the_free_surface_on_a_zstar_column():
    n2 = 2.0
    grid = make_column_grid(hy.zstar_mapping(sloped))
    model = make_column_model(grid, n2=n2, extra=(hy.ZStarGeometry(),))
    lifted = model.state.replace(eta=model.state["eta"] + 0.1)
    total = model.diagnostics.b_total(lifted)
    b = lifted["b"]
    x, y, z = column_nodes(b)
    depth = np.asarray(sloped(jnp.asarray(x), jnp.asarray(y)))
    # zp = eta + (H + eta) z at the buoyancy nodes, eta read off the state
    assert np.allclose(np.asarray(total.data), n2 * (0.1 + (depth + 0.1) * z))
    # the model's own state is the reference column, eta = 0
    assert np.allclose(np.asarray(model.diagnostics.b_total().data),
                       n2 * depth * z)
