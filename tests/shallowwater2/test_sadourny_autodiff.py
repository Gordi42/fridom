"""Sadourny advection: reverse-mode autodiff through a model run.

The vector-invariant scheme divides the relative vorticity by the full
thickness, ``q = zeta / p_full``. In the unsealed ghost/padding cells
``p_full`` is an exact zero (and ``zeta`` is zero at rest), so the bare
quotient is a masked ``0/0``: harmless for the forward run (those cells
are stripped before any output) but a NaN poison for reverse-mode
autodiff, whose VJP ignores the downstream ghost-sealing and turns every
gradient with a data path into NaN. ``_potential_vorticity`` guards the
division; these tests pin that the guard keeps the forward result bitwise
identical on valid cells while making ``jax.grad`` finite and correct.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model.model import _chunk_body
from fridom.shallowwater2.modules.sadourny import (
    _potential_vorticity,
    _sealed_metric_divide,
)
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

from .conftest import gaussian_bump, make_grid


# ================================================================
#  Helpers
# ================================================================
def advecting_model(nu=0.02, *, csqr=1.0, rossby=0.2):
    """Return a tiny nonlinear (Sadourny) SW model with friction."""
    model = sw.Model(
        grid=make_grid(),
        core=sw.Core(froude_number=rossby, depth=csqr),
        scaling=fr.scaling.GravityWave(),
        coriolis=sw.modules.FPlaneCoriolis(rossby_number=rossby),
        advection=True,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3),
        modules_extra=(fr.model.closures.HarmonicFriction(nu=nu),))
    model.set_fields(p=gaussian_bump(amp=0.05))
    return model


def leaf_loss(model, leaf, n_steps):
    """Return a quadratic loss in one carry leaf via the pure kernel.

    Substitutes ``leaf`` (found by identity in the flattened carry)
    with the differentiation variable, advances ``n_steps`` through
    ``_chunk_body`` (the pure, differentiable step kernel — the public
    ``advance`` path is not differentiable), and returns
    ``sum(u**2 + v**2 + p**2)`` of the interior DOFs.
    """
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


def friction_nu(model):
    """Return the harmonic-friction ``nu`` leaf on the carry."""
    return next(m.nu for m in model._carry.modules
                if type(m).__name__ == "HarmonicFriction")


# ================================================================
#  The guard, unit-tested directly (padding NaN vs bitwise interior)
# ================================================================
def test_potential_vorticity_guard_removes_padding_nan():
    """The guard eliminates the ghost 0/0 without touching valid cells."""
    model = advecting_model()
    state = model._carry.state
    u, v, p, c = state["u"], state["v"], state["p"], state["csqr"]
    rossby = 0.2
    p_full = c.to(p) + rossby * p
    corner = u.function_space.bare.replace(
        y=v.function_space.bare.factor("y"))
    zeta = (v.diff("x").retag(corner) - u.diff("y").retag(corner))
    den = p_full.to(zeta)

    bare = zeta / den                      # the un-guarded quotient
    guarded = _potential_vorticity(zeta, den)

    # the bug: exact zeros in the padded denominator -> NaN in storage
    assert int((np.asarray(den.storage) == 0.0).sum()) > 0
    assert bool(np.isnan(np.asarray(bare.storage)).any())
    # the fix: no NaN anywhere in the guarded storage
    assert not bool(np.isnan(np.asarray(guarded.storage)).any())
    # and the valid interior is bitwise identical (only padding changed)
    assert np.array_equal(np.asarray(bare.data),
                          np.asarray(guarded.data))


# ================================================================
#  Reverse-mode gradients are finite and match finite differences
# ================================================================
def test_reverse_grad_wrt_nu_is_finite_and_matches_fd():
    """Check grad w.r.t. friction nu is finite and matches FD."""
    model = advecting_model()
    nu0 = friction_nu(model)
    loss = leaf_loss(model, nu0, n_steps=12)

    grad = float(jax.grad(loss)(nu0))
    assert np.isfinite(grad)

    h = 1e-4 * float(nu0)
    fd = (float(loss(nu0 + h)) - float(loss(nu0 - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=1e-4)


def test_reverse_grad_wrt_initial_pressure_is_finite_and_matches_fd():
    """Check grad w.r.t. the initial pressure is finite (directional)."""
    model = advecting_model()
    p_leaf = model._carry.state["p"].storage
    loss = leaf_loss(model, p_leaf, n_steps=12)

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


def test_advected_run_stays_finite():
    """Forward sanity: the nonlinear run produces no NaN interior."""
    model = advecting_model()
    state = _chunk_body(model._artifacts.record, 12,
                        model._carry, model._stepper).state
    for name in ("u", "v", "p"):
        assert bool(np.all(np.isfinite(np.asarray(state[name].data))))


# ================================================================
#  Immersed (cut-cell) path: the same masked PV division, guarded
# ================================================================
# ``_advect_immersed`` divides ``zeta / p_full`` before applying the
# boolean wet mask. Masking *after* the bare quotient leaves not only
# the never-valid padding but every immersed **interior dry cell**
# (``p_full == 0`` there) as a live ``0/0`` whose reverse-mode VJP
# (``-zeta/h^2``, ``h = 0``) is NaN — even worse than the flat path,
# where only padding is at risk. Routing through ``_potential_vorticity``
# guards the denominator before the mask. Closures (HarmonicFriction)
# are a taught error on immersed grids, so the differentiation variable
# is the initial condition (a genuine data path through the guard).
def immersed_advecting_model():
    """Return a tiny immersed (cut-cell) nonlinear SW model.

    A dry box carves an interior wet region on a periodic grid, so the
    wet-region boundary carries the immersed masks the guard must
    survive.
    """
    box = lambda x, y: (  # noqa: E731
        (x > 2) & (x < 10) & (y > 2) & (y < 10)).astype(float)
    grid = Grid(
        (IntervalMesh(12, (0.0, 12.0), periodic=True, name="x"),
         IntervalMesh(12, (0.0, 12.0), periodic=True, name="y")),
        immersed=ImmersedDomain(box))
    model = sw.Model(
        grid=grid,
        core=sw.Core(froude_number=0.3, depth=0.8),
        scaling=fr.scaling.GravityWave(),
        coriolis=sw.modules.FPlaneCoriolis(rossby_number=0.3),
        advection=True,
        time_stepper=fr.model.time_steppers.AdamBashforth(0.01, order=3))
    rng = np.random.default_rng(0)
    mask = np.asarray(
        grid.immersed.mask(model.state["p"].function_space).data)
    model.set_fields(
        p=0.1 * rng.standard_normal(model.state["p"].data.shape) * mask,
        u=0.1 * rng.standard_normal(model.state["u"].data.shape),
        v=0.1 * rng.standard_normal(model.state["v"].data.shape))
    return model


def test_immersed_reverse_grad_wrt_ic_is_finite_and_matches_fd():
    """Grad through the immersed run w.r.t. the IC: finite, FD-matched."""
    model = immersed_advecting_model()
    p_leaf = model._carry.state["p"].storage
    loss = leaf_loss(model, p_leaf, n_steps=10)

    grad = np.asarray(jax.grad(loss)(p_leaf))
    # the pre-fix bug NaNed every entry with a data path; the guard
    # keeps them all finite
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
#  The chart (metric) path: the sqg_p kinetic-energy divide, guarded
# ================================================================
# ``_advect_chart`` divides the chart kinetic energy by the centre
# metric ``sqrt_g`` (``sqg_p``). On a walled chart (the lat-lon sphere's
# polar caps) that weight is an exact zero in the never-valid
# storage/halo padding, where the numerator vanishes too, so the bare
# quotient is a masked ``0/0`` whose reverse-mode VJP is a NaN poison —
# the same class as the PV divide above. ``_sealed_metric_divide`` guards
# the denominator; this proves the guard removes the padding NaN while
# leaving the valid interior bitwise identical.
LAT_MAX = float(np.deg2rad(80.0))


def sphere_advecting_model(*, csqr=0.7, rossby=0.4):
    """Return a tiny nonlinear Sadourny model on the lat-lon sphere."""
    grid = fr.spatial.spherical.Grid(
        (16, 8), radius=1.0, lat_extent=(-LAT_MAX, LAT_MAX),
        device_ids=(0,))
    model = sw.Model(
        grid=grid,
        core=sw.Core(froude_number=rossby, depth=csqr,
                     coords=("lon", "lat")),
        scaling=fr.scaling.GravityWave(),
        coriolis=None, advection=True,
        time_stepper=fr.model.time_steppers.AdamBashforth(2e-3, order=3))
    rng = np.random.default_rng(3)
    model.set_fields(
        u=0.1 * rng.standard_normal(model.state["u"].shape),
        v=0.1 * rng.standard_normal(model.state["v"].shape),
        p=0.03 * rng.standard_normal(model.state["p"].shape))
    return model


def test_sealed_metric_divide_removes_padding_nan():
    """The sqg_p guard eliminates the ghost 0/0 without touching cells."""
    model = sphere_advecting_model()
    grid = model.grid
    state = model._carry.state
    u, v, p = state["u"], state["v"], state["p"]

    # reconstruct the chart kinetic-energy divide exactly as
    # ``_advect_chart`` builds it (numerator and denominator both an
    # exact zero in the walled polar/halo padding)
    sqg_u = grid.metric(u.function_space.bare, "sqrt_g")
    sqg_v = grid.metric(v.function_space.bare, "sqrt_g")
    sqg_p = grid.metric(p.function_space.bare, "sqrt_g")
    g_uu = grid.metric(u.function_space.bare, "g_lonlon")
    g_vv = grid.metric(v.function_space.bare, "g_latlat")
    ekin_num = 0.5 * (((sqg_u * g_uu) * (u * u)).to(p)
                      + ((sqg_v * g_vv) * (v * v)).to(p))

    bare = ekin_num / sqg_p                    # the un-guarded quotient
    guarded = _sealed_metric_divide(ekin_num, sqg_p)

    # the bug: exact zeros in the padded denominator -> NaN in storage
    assert int((np.asarray(sqg_p.storage) == 0.0).sum()) > 0
    assert bool(np.isnan(np.asarray(bare.storage)).any())
    # the fix: no NaN anywhere in the guarded storage
    assert not bool(np.isnan(np.asarray(guarded.storage)).any())
    # and the valid interior is bitwise identical (only padding changed)
    assert np.array_equal(np.asarray(bare.data),
                          np.asarray(guarded.data))
