"""Nonlinear advection on the hydrostatic model (stage H2b).

The diagnosed vertical velocity ``w`` lives on the both-boundary
``Outer`` faces; the shared flux-form advection resolves it onto the
interior flux faces through the seeded ``Outer -> Inner`` restriction
(``fr.operators.Restriction``). These tests cover the factory wiring,
the vertical/horizontal transport, the zero-boundary-flux closure (mass
to roundoff), and the semi-discrete energy behaviour.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.energy import EnergyMetric
from fridom.model.model import _chunk_body
from fridom.model.modules.advection import (
    CenteredAdvection,
    UpwindAdvection,
    WENOAdvection,
)
from fridom.model.term_predicates import owned_by
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

IM = fr.spatial.meshes.IntervalMesh


def make_grid(nx=8, nz=8, depth=1.0):
    """Doubly-periodic horizontal, bounded-vertical grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, depth), periodic=False, name="z")))


def make_model(grid=None, *, advection=True, dt=1e-3):
    """Build a hydrostatic model with the given advection option."""
    if grid is None:
        grid = make_grid()
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(dt, order=3),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=advection)


def _set(model, grid, **inits):
    """Materialize the named component profiles and set the state."""
    data = {}
    for name, init in inits.items():
        space = model.state[name].function_space
        data[name] = grid.create_field(space, init=init).data
    model.set_fields(**data)


def _xcoord(grid, model):
    """Return the cell-centre x coordinate of the buoyancy space (1D)."""
    return np.asarray(grid.evaluation_nodes(
        model.state["b"].function_space, "x").data).ravel()


# ================================================================
#  Factory wiring
# ================================================================
def test_default_installs_centered_advection():
    model = make_model()
    modules = [type(m).__name__ for m in model.modules]
    assert "CenteredAdvection" in modules


def test_advection_false_is_the_linear_model():
    model = make_model(advection=False)
    modules = [type(m).__name__ for m in model.modules]
    assert not any("Advection" in name for name in modules)


@pytest.mark.parametrize("scheme", [
    UpwindAdvection(order=3), WENOAdvection(order=5),
    CenteredAdvection()])
def test_biased_and_centered_instances_are_accepted(scheme):
    model = make_model(advection=scheme)
    assert scheme in tuple(model.modules)


# ================================================================
#  Assemble-and-run smoke (all three schemes)
# ================================================================
@pytest.mark.parametrize("scheme", [
    True, UpwindAdvection(order=3), WENOAdvection(order=5)])
def test_assemble_and_run(scheme):
    grid = make_grid()
    model = make_model(grid, advection=scheme)
    _set(model, grid,
         u=lambda x, y, z: 0.1 * np.sin(2 * np.pi * x) + 0.0 * (y + z),
         b=lambda x, y, z: np.sin(2 * np.pi * x) * z + 0.0 * y)
    model.run(steps=3)
    assert np.all(np.isfinite(np.asarray(model.state["b"].data)))


# ================================================================
#  The zero-boundary-flux closure: tracer mass to roundoff
# ================================================================
# Pinned to surface_flux=False (the legacy fixed-domain closure): the
# constancy-preserving default (surface_flux=None -> auto on for the
# hydrostatic Outer-w grid) advects THROUGH the top face and so exchanges
# tracer content with the moving surface, breaking exact mass
# conservation by design (see test_surface_flux_auto_default_*).
@pytest.mark.parametrize("scheme", [
    CenteredAdvection(surface_flux=False),
    UpwindAdvection(order=3, surface_flux=False),
    WENOAdvection(order=5, surface_flux=False)])
def test_tracer_mass_conserved_to_roundoff(scheme):
    # the restriction drops w's boundary faces, so the advective flux
    # through the top/bottom faces is a structural zero: the flux
    # divergence telescopes and the total buoyancy is conserved to
    # machine precision (a uniform mesh -> the cell measure is constant,
    # so the raw sum is the mass up to that constant).
    grid = make_grid()
    model = make_model(grid, advection=scheme)
    _set(model, grid,
         u=lambda x, y, z: 0.2 * np.sin(2 * np.pi * x) + 0.0 * (y + z),
         v=lambda x, y, z: 0.2 * np.sin(2 * np.pi * y) * (1 + 0.5 * z)
         + 0.0 * x,
         b=lambda x, y, z: np.cos(2 * np.pi * x) * np.sin(np.pi * z)
         + 0.0 * y)
    mass0 = float(np.sum(np.asarray(model.state["b"].data)))
    model.run(steps=25)
    mass1 = float(np.sum(np.asarray(model.state["b"].data)))
    scale = float(np.sum(np.abs(np.asarray(model.state["b"].data))))
    assert abs(mass1 - mass0) <= 1e-11 * max(scale, 1.0)


def test_the_closure_is_localized_to_the_surface_cell():
    # LEGACY closure (surface_flux=False): A(b = const) probes the
    # advection's effective discrete divergence. It is machine-zero in
    # every interior cell (the 3D flow is discretely divergence-free
    # there), and non-zero only in the top (surface) cell, where the
    # restriction has dropped w(0) -> the zero-boundary-flux closure. The
    # bottom cell stays exact (w = 0 is seeded there). The default now
    # cancels this surface term (test_surface_flux_auto_default_*).
    grid = make_grid()
    model = make_model(grid,
                       advection=CenteredAdvection(surface_flux=False))
    _set(model, grid,
         u=lambda x, y, z: 0.3 * np.sin(2 * np.pi * x) + 0.0 * (y + z),
         v=lambda x, y, z: 0.3 * np.sin(2 * np.pi * y) * (1 + 0.5 * z)
         + 0.0 * x,
         b=lambda x, y, z: 1.0 + 0.0 * (x + y + z))
    tau = model.tendency(model.state, filter=owned_by(CenteredAdvection))
    tb = np.asarray(tau["b"].data)  # (x, y, z)
    interior = np.max(np.abs(tb[..., :-1]))
    surface = np.max(np.abs(tb[..., -1]))
    assert interior <= 1e-12
    assert surface > 1e-2


# ================================================================
#  Vertical transport: the diagnosed w advects b
# ================================================================
def test_diagnosed_w_transports_buoyancy_vertically():
    # a horizontally divergent flow diagnoses w = -int (d_x u + d_y v);
    # with a linear stratification b = z (d_z b = 1) and a
    # divergence-free interior, the buoyancy advection reduces to the
    # pure vertical transport A(b) = -w. u = sin(2 pi x)/(2 pi) gives
    # d_x u = cos(2 pi x), so w = -z cos(2 pi x) and A(b) = z cos(2 pi x)
    # in the interior: same sign as cos(2 pi x), growing with depth.
    grid = make_grid(nz=8)
    model = make_model(grid)
    _set(model, grid,
         u=lambda x, y, z: np.sin(2 * np.pi * x) / (2 * np.pi)
         + 0.0 * (y + z),
         b=lambda x, y, z: z + 0.0 * (x + y))
    tau = model.tendency(model.state, filter=owned_by(CenteredAdvection))
    tb = np.asarray(tau["b"].data)  # (x, y, z)
    cos_x = np.cos(2 * np.pi * _xcoord(grid, model))
    # at every interior z-level the tendency tracks cos(2 pi x)
    for k in range(1, tb.shape[-1] - 1):
        col = tb[:, 0, k]
        assert np.corrcoef(col, cos_x)[0, 1] > 0.99
        assert np.max(np.abs(col)) > 1e-3
    # magnitude grows with depth-integrated w (deeper level -> larger)
    amp = [np.max(np.abs(tb[:, 0, k])) for k in range(1, tb.shape[-1] - 1)]
    assert amp[-1] > amp[0]


# ================================================================
#  Horizontal tracer transport sanity
# ================================================================
def test_horizontal_transport_by_a_uniform_flow():
    # a uniform zonal flow U advects a passive x-mode: A(b) = -U d_x b,
    # a quarter-wavelength out of phase with b (cos vs -sin), and the
    # x-variance moves without changing the horizontal mean.
    grid = make_grid(nx=16, nz=4)
    model = make_model(grid)
    _set(model, grid,
         u=lambda x, y, z: 1.0 + 0.0 * (x + y + z),
         b=lambda x, y, z: np.sin(2 * np.pi * x) + 0.0 * (y + z))
    tau = model.tendency(model.state, filter=owned_by(CenteredAdvection))
    tb = np.asarray(tau["b"].data)
    cos_x = np.cos(2 * np.pi * _xcoord(grid, model))
    # -U d_x sin(2 pi x) = -2 pi cos(2 pi x): negative correlation
    assert np.corrcoef(tb[:, 0, 0], cos_x)[0, 1] < -0.99
    # the horizontal mean tendency vanishes (no spurious source)
    assert abs(float(np.mean(tb))) <= 1e-12


# ================================================================
#  Semi-discrete energy conservation
# ================================================================
def _full_skew(metric, *, advection):
    """Full-RHS energy rate <X, M f(X)> / <X, M X> of a divergent state."""
    grid = make_grid()
    model = make_model(grid, advection=advection)
    _set(model, grid,
         u=lambda x, y, z: 0.2 * np.sin(2 * np.pi * x) + 0.0 * (y + z),
         v=lambda x, y, z: 0.2 * np.sin(2 * np.pi * y) * (1 + 0.5 * z)
         + 0.0 * x,
         b=lambda x, y, z: np.cos(2 * np.pi * x) * np.sin(np.pi * z)
         + 0.0 * y,
         ps=lambda x, y: 0.1 * np.cos(2 * np.pi * x) + 0.0 * y)
    state = model.state
    tau = model.tendency(state)  # full RHS (all terms + diagnostics)
    scale = float(metric.inner(state, state).real)
    return float(metric.inner(state, tau).real) / scale


def test_advection_preserves_the_semidiscrete_energy_skew():
    # the H2 gate is the linear energy skew <X, M dX/dt> = O(1e-16).
    # Adding CenteredAdvection (the default, constancy-preserving surface
    # closure) must not degrade it: the closure makes advection the
    # discrete advective form -v.grad(q) with the divergence-free
    # continuity velocity (w through the surface face), whose quadratic
    # energy rate <q, M A(q)> = int (div v) q^2/2 is machine-zero, so the
    # full-RHS skew stays at the machine-precision level with the
    # vertical leg active. This is the sense in which the H2 quadratic
    # energy is conserved to time-discretization accuracy.
    metric = EnergyMetric(hy.energy.hydrostatic_energy_weights(1.0, 1.0))
    linear = _full_skew(metric, advection=False)
    advected = _full_skew(metric, advection=True)
    assert abs(linear) <= 1e-13
    # advection leaves the (machine-zero) skew unchanged: w != 0 here,
    # so the vertical leg is genuinely exercised
    assert abs(advected) <= 1e-13


def test_advection_tendency_is_energy_orthogonal():
    # the isolated advection tendency's contribution to the energy rate
    # <q, M A(q)> is machine-zero (the semi-discrete conservation
    # property, the nonhydro relative_energy_rates pattern) -- for the
    # default constancy-preserving closure because the advective form
    # with the divergence-free continuity velocity conserves q^2.
    metric = EnergyMetric(hy.energy.hydrostatic_energy_weights(1.0, 1.0))
    grid = make_grid()
    model = make_model(grid)
    _set(model, grid,
         u=lambda x, y, z: 0.3 * np.sin(2 * np.pi * x) + 0.0 * (y + z),
         v=lambda x, y, z: 0.3 * np.sin(2 * np.pi * y) * (1 + 0.5 * z)
         + 0.0 * x,
         b=lambda x, y, z: np.cos(2 * np.pi * x) * np.sin(np.pi * z)
         + 0.0 * y)
    state = model.state
    tau = model.tendency(state, filter=owned_by(CenteredAdvection))
    scale = float(metric.inner(state, state).real)
    assert abs(float(metric.inner(state, tau).real)) <= 1e-13 * scale


# ================================================================
#  The opt-in constancy-preserving surface closure (surface_flux)
# ================================================================
_DIVERGENT = {
    "u": lambda x, y, z: 0.3 * np.sin(2 * np.pi * x) + 0.0 * (y + z),
    "v": lambda x, y, z: 0.3 * np.sin(2 * np.pi * y) * (1 + 0.5 * z)
    + 0.0 * x,
}


def test_surface_flux_auto_default_is_constant_preserving():
    # the DEFAULT closure (surface_flux=None auto-resolves ON for the
    # hydrostatic Outer-w grid) advects through the top face with the
    # one-sided face value, so A(b = const) is machine-zero in EVERY cell
    # (surface cell included). The velocity is horizontally divergent, so
    # w(0) != 0 and the surface source is genuinely present to cancel.
    grid = make_grid(nx=16, nz=8)
    const_b = {"b": lambda x, y, z: 2.5 + 0.0 * (x + y + z)}
    default = make_model(grid)  # advection=True -> auto surface flux ON
    resolved = next(m for m in default.modules
                    if isinstance(m, CenteredAdvection))
    assert resolved._surface_flux_on is True
    _set(default, grid, **_DIVERGENT, **const_b)
    tb = np.asarray(default.tendency(
        default.state, filter=owned_by(CenteredAdvection))["b"].data)
    assert np.max(np.abs(tb)) <= 1e-13  # every cell, surface included
    # the legacy closure (surface_flux=False) still carries the source
    off = make_model(grid, advection=CenteredAdvection(surface_flux=False))
    _set(off, grid, **_DIVERGENT, **const_b)
    tb_off = np.asarray(off.tendency(
        off.state, filter=owned_by(CenteredAdvection))["b"].data)
    assert np.max(np.abs(tb_off[..., :-1])) <= 1e-12  # interior quiet
    assert np.max(np.abs(tb_off[..., -1])) > 1e-2  # surface loud


@pytest.mark.parametrize("scheme", [
    CenteredAdvection(), WENOAdvection(order=5)])
def test_surface_flux_auto_on_for_any_construction_path(scheme):
    # a user-passed bare module auto-resolves the closure ON on the
    # hydrostatic grid too, so passing a scheme does not silently bypass
    # the constancy-preserving default (any scheme, any path).
    model = make_model(advection=scheme)  # fresh 8x8x8 grid
    assert scheme._surface_flux_on is True
    _set(model, model.grid,
         **_DIVERGENT, b=lambda x, y, z: 2.5 + 0.0 * (x + y + z))
    tb = np.asarray(model.tendency(
        model.state, filter=owned_by(type(scheme)))["b"].data)
    assert np.max(np.abs(tb)) <= 1e-13


def test_surface_flux_grad_through_a_short_run_matches_fd():
    # differentiability policy (AGENTS.md): jax.grad of a quadratic loss
    # through a short surface-flux run (via the pure _chunk_body kernel)
    # w.r.t. an initial field is finite and matches a central FD. The
    # data path crosses the -q*A(1) surface correction (its lean A(1) is
    # a flux divergence of the advecting velocity).
    grid = make_grid(nx=8, nz=4)
    model = hy.Model(
        grid=grid,
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(2e-3, order=3),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=CenteredAdvection(surface_flux=True))
    rng = np.random.default_rng(3)
    model.set_fields(**{
        k: 0.1 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "b")})
    record = model._artifacts.record
    carry, stepper = model._carry, model._stepper
    b_leaf = carry.state["b"].storage
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is b_leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        spliced = jax.tree_util.tree_unflatten(treedef, new)
        final = _chunk_body(record, 6, spliced, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    grad = np.asarray(jax.grad(loss)(b_leaf))
    assert bool(np.all(np.isfinite(grad)))
    rng = np.random.default_rng(7)
    direction = jnp.asarray(rng.standard_normal(b_leaf.shape),
                            dtype=b_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(b_leaf + eps * direction))
          - float(loss(b_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)
