"""SmagorinskyLilly at walls: the free-slip strain retag.

Prefix-mirrored shard of ``test_smagorinsky_lilly.py`` (oversized-module
rule): the walled-grid behaviour of the closure — the decisive
constant-coefficient reduction to the background friction, free-slip
mirror symmetry, energy dissipation, corner composition, autodiff, and
decomposition of the walled axis. The builders are duplicated
(self-contained shard) rather than imported across files.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.model.closures.diffusion import HarmonicFriction
from fridom.model.model import Model, _chunk_body
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.nonhydro2.modules.smagorinsky_lilly import SmagorinskyLilly
from fridom.nonhydro2.modules.stratification import ConstantStratification
from fridom.nonhydro2.params import (
    SMAG_BUOYANCY_MULTIPLIER,
    SMAG_CS,
    STRATIFICATION_N2,
)
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 8
L = 1.0
DT = 1e-3


def make_grid(periodic, length=L, nz=N, device_ids=None):
    """Build the x/y/z grid; ``periodic`` is a per-axis mapping."""
    specs = (("x", N, L, periodic["x"]), ("y", N, L, periodic["y"]),
             ("z", nz, length, periodic["z"]))
    return Grid(tuple(
        IntervalMesh(n, (0.0, ln), periodic=p, name=nm)
        for nm, n, ln, p in specs), device_ids=device_ids)


def smag_model(grid, n2=0.0, **kwargs):
    return Model(
        grid=grid,
        modules=(DynamicalCore(), ConstantStratification(n2=n2),
                 SmagorinskyLilly(**kwargs)),
        time_stepper=AdamBashforth(DT, order=3))


def friction_model(grid, nu, slip="free", n2=0.0):
    return Model(
        grid=grid,
        modules=(DynamicalCore(), ConstantStratification(n2=n2),
                 HarmonicFriction(nu, slip=slip)),
        time_stepper=AdamBashforth(DT, order=3))


def data(field):
    return np.asarray(field.data)


def cos_z(m):
    """Return a wall-parallel cosine (free-slip) z-profile of order m."""
    return lambda x, y, z: np.cos(np.pi * m * z / L) + 0.0 * (x + y)


def sin_z(m):
    """Return a wall-normal sine (Dirichlet) z-profile of order m."""
    return lambda x, y, z: np.sin(np.pi * m * z / L) + 0.0 * (x + y)


# ================================================================
#  Decisive: Cs=0 walled stress reduces to the background friction
# ================================================================
def test_cs_zero_walled_tangential_stress_is_harmonic_friction():
    # a wall-parallel velocity varying only in the wall-normal z: the
    # Cs=0 stress is the pure strain-tensor divergence of the shear,
    # which (tau = nu Sigma, no factor 2) is HALF the harmonic rate ->
    # bit-for-bit HarmonicFriction(nu_bg/2, slip="free").
    nu_bg = 3e-3
    grid = make_grid({"x": True, "y": True, "z": False})
    smag = smag_model(grid, smagorinsky_constant=0.0,
                      background_viscosity=nu_bg)
    fric = friction_model(grid, nu=0.5 * nu_bg, slip="free")
    for m in (1, 2, 3):
        smag.set_fields(u=cos_z(m))
        fric.set_fields(u=cos_z(m))
        ts = data(smag.tendency(smag.state)["u"])
        tf = data(fric.tendency(fric.state)["u"])
        np.testing.assert_array_equal(ts, tf)
        assert np.abs(ts).max() > 0.0


def test_cs_zero_wall_normal_stress_is_harmonic_friction():
    # the wall-normal velocity (Inner[Dirichlet] along z) closes on its
    # own tag: the diagonal Sigma_zz carries no 1/2, so the Cs=0 stress
    # is the FULL harmonic rate -> bit-for-bit HarmonicFriction(nu_bg).
    nu_bg = 3e-3
    grid = make_grid({"x": True, "y": True, "z": False})
    smag = smag_model(grid, smagorinsky_constant=0.0,
                      background_viscosity=nu_bg)
    fric = friction_model(grid, nu=nu_bg, slip="free")
    for m in (1, 2, 3):
        smag.set_fields(w=sin_z(m))
        fric.set_fields(w=sin_z(m))
        ts = data(smag.tendency(smag.state)["w"])
        tf = data(fric.tendency(fric.state)["w"])
        np.testing.assert_array_equal(ts, tf)
        assert np.abs(ts).max() > 0.0


# ================================================================
#  Free-slip: uniform wall-parallel flow feels no wall drag
# ================================================================
def test_free_slip_uniform_wall_parallel_flow_has_no_drag():
    # a uniform tangential flow has zero strain everywhere; the
    # free-slip wall adds no drag, so the stress is exactly zero
    grid = make_grid({"x": True, "y": True, "z": False})
    model = smag_model(grid, smagorinsky_constant=0.16,
                       background_viscosity=1e-2)
    model.set_fields(u=lambda x, y, z: np.ones_like(x + y + z))
    td = model.tendency(model.state)
    assert np.abs(data(td["u"])).max() == 0.0


# ================================================================
#  Free-slip == even restriction of the doubled periodic domain
# ================================================================
def test_free_slip_matches_doubled_periodic_mirror():
    # the free-slip wall is a mirror plane: an arbitrary z-profile on the
    # walled [0, L] reflects to a periodic [0, 2L] field (even about z=L,
    # which forces zero gradient at both walls). The nonlinear stress on
    # the [0, L] half must be bit-identical.
    kwargs = {"smagorinsky_constant": 0.2, "background_viscosity": 1e-3}
    walled = smag_model(make_grid({"x": True, "y": True, "z": False}),
                        **kwargs)
    periodic = smag_model(
        make_grid({"x": True, "y": True, "z": True}, length=2 * L, nz=2 * N),
        **kwargs)
    rng = np.random.default_rng(0)
    r = rng.standard_normal(N)
    rp = np.concatenate([r, r[::-1]])  # even reflection about z = L
    walled.set_fields(u=np.broadcast_to(r, (N, N, N)).copy())
    periodic.set_fields(u=np.broadcast_to(rp, (N, N, 2 * N)).copy())
    tw = data(walled.tendency(walled.state)["u"])
    tp = data(periodic.tendency(periodic.state)["u"])
    np.testing.assert_array_equal(tw, tp[:, :, :N])


# ================================================================
#  Free-slip walled run dissipates kinetic energy monotonically
# ================================================================
def test_free_slip_walled_run_dissipates_kinetic_energy():
    grid = make_grid({"x": True, "y": True, "z": False})
    model = smag_model(grid, n2=1.0, smagorinsky_constant=0.16,
                       background_viscosity=1e-2,
                       background_diffusivity=1e-2)
    model.set_fields(u=cos_z(1), v=cos_z(2))
    energies = []
    for _ in range(5):
        model.advance(2)
        energies.append(float(np.sum(data(model.diagnostics.ekin()))))
    energies = np.asarray(energies)
    assert np.isfinite(energies).all()
    assert (np.diff(energies) < 0.0).all()


# ================================================================
#  Two walled axes compose per-axis (corner cells included)
# ================================================================
def test_two_walled_axes_compose_and_step_finite():
    grid = make_grid({"x": False, "y": True, "z": False})
    model = smag_model(grid, smagorinsky_constant=0.16,
                       background_viscosity=1e-2,
                       background_diffusivity=1e-2)
    model.set_fields(
        u=lambda x, y, z: np.sin(np.pi * x) * np.cos(np.pi * z) + 0.0 * y,
        v=lambda x, y, z: np.cos(np.pi * y) * np.sin(np.pi * z) + 0.0 * x,
        b=lambda x, y, z: np.sin(np.pi * z) + 0.0 * (x + y))
    final = _chunk_body(model._artifacts.record, 5,
                        model._carry, model._stepper)
    assert all(bool(np.all(np.isfinite(np.asarray(f.data))))
               for f in final.state)


# ================================================================
#  Taught rejection: walled finite-volume (CellAvg) grid is future work
# ================================================================
def test_walled_finite_volume_grid_is_a_taught_rejection():
    grid = make_grid({"x": True, "y": True, "z": False})
    with pytest.raises(NotImplementedError,
                       match="walled finite-volume"):
        Model(grid=grid,
              modules=(DynamicalCore(family="fv"),
                       ConstantStratification(n2=0.0),
                       SmagorinskyLilly()),
              time_stepper=AdamBashforth(DT, order=3))


# ================================================================
#  Reverse-mode AD through a walled free-slip run matches central FD
# ================================================================
def _walled_grad_loss(n_steps=8, cs=0.16, n2=15.0):
    """Build a grad-ready loss over a walled free-slip Smagorinsky run.

    ``n2`` is tuned so a fraction of the cells clip (exercising the
    guarded sqrt) while ``Cs`` still drives the unclipped cells.
    """
    grid = make_grid({"x": True, "y": True, "z": False})
    model = smag_model(grid, n2=n2, smagorinsky_constant=cs)
    rng = np.random.default_rng(1)
    sh = (N, N, N)
    model.set_fields(u=0.2 * rng.standard_normal(sh),
                     v=0.2 * rng.standard_normal(sh),
                     w=0.2 * rng.standard_normal((N, N, N - 1)),
                     b=0.01 * rng.standard_normal(sh))
    closure = next(m for m in model._carry.modules
                   if isinstance(m, SmagorinskyLilly))
    ctx = SimpleNamespace(params={
        SMAG_CS: cs, SMAG_BUOYANCY_MULTIPLIER: 1.0, STRATIFICATION_N2: n2})
    n_clipped = int(np.sum(data(closure._eddy_viscosity(
        model.state, ctx)) == 0.0))
    record = model._artifacts.record
    carry = model._carry
    stepper = model._stepper
    leaf = closure.smagorinsky_constant
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    idx = next(i for i, lf in enumerate(leaves) if lf is leaf)

    def loss(theta):
        packed = list(leaves)
        packed[idx] = theta
        c = jax.tree_util.tree_unflatten(treedef, packed)
        final = _chunk_body(record, n_steps, c, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    return loss, jnp.asarray(leaf, dtype=jnp.float64), n_clipped


def test_free_slip_walled_grad_is_finite_and_matches_fd():
    loss, x0, n_clipped = _walled_grad_loss()
    assert 0 < n_clipped < N ** 3  # the clip fires (some cells, not all)
    g = float(jax.grad(loss)(x0))
    assert np.isfinite(g)
    eps = 1e-4
    fd = float((loss(x0 * (1 + eps)) - loss(x0 * (1 - eps)))
               / (2 * x0 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-4)


# ================================================================
#  Sharded walled axis == single device (the ghost-fill survives
#  decomposition of the walled axis)
# ================================================================
@pytest.mark.multi_device
def test_sharded_walled_stress_matches_single_device(forced_devices):
    # an all-walled grid: the default layout shards the walled x axis
    # (the periodic-preferring layout picks a walled axis only when all
    # are walled), so the staggered strain diff and the free-slip retag
    # cross a shard boundary of a bounded axis. The result must equal
    # the single-device run (and corners compose along three walls).
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    nu_bg = 3e-3
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        grid = make_grid({"x": False, "y": False, "z": False},
                         device_ids=device_ids)
        model = smag_model(grid, smagorinsky_constant=0.16,
                           background_viscosity=nu_bg,
                           background_diffusivity=nu_bg)
        model.set_fields(
            u=lambda x, y, z: 0.3 * np.sin(np.pi * x) * np.cos(np.pi * y)
            + 0.0 * z,
            v=lambda x, y, z: 0.2 * np.cos(np.pi * x) * np.sin(np.pi * y)
            + 0.0 * z,
            b=lambda x, y, z: 0.1 * np.sin(np.pi * x) * np.sin(np.pi * z)
            + 0.0 * y)
        results[tag] = {nm: data(model.tendency(model.state)[nm])
                        for nm in ("u", "v", "w", "b")}
        if tag == "many" and jax.device_count() > 1:
            assert model.state["u"]._data.sharding.spec[0] == "devices"
    for nm in ("u", "v", "w", "b"):
        np.testing.assert_allclose(results["many"][nm], results["one"][nm],
                                   rtol=1e-12, atol=1e-14)
