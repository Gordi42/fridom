"""Reverse-mode autodiff through a spherical (chart) shallow-water run.

The metric-aware core divergence divides by ``sqrt_g`` — the
``MetricScaled`` reciprocal of ``spatial/operators/mapped.py``. On the
walled lat-lon sphere ``sqrt_g`` is an exact zero in the never-valid
polar/halo padding, so the bare quotient's reverse VJP is a masked
``0 * inf = NaN`` that poisons every gradient with a data path through
the chart divergence — even with ``coriolis=None``, which isolates the
``MetricScaled`` seal from the (separately sealed) coriolis metric
weights. ``_sealed_metric_divide`` keeps the forward run bitwise
identical on valid cells while making ``jax.grad`` finite and matched
to a central finite difference (rtol 1e-4). The nonlinear Sadourny
scheme carries its own guarded metric divides, so this isolation runs
``advection=False``. Recipe: ``tests/model/test_model_autodiff.py``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model.model import _chunk_body

LAT_MAX = float(np.deg2rad(80.0))
N = 8


# ================================================================
#  Builders (self-contained per the AGENTS oversized-module rule)
# ================================================================
def sphere_grid(nlon=2 * N, nlat=N, radius=1.0):
    """Lat-lon sphere chart grid, polar caps excluded (walled lat)."""
    return fr.spatial.spherical.Grid(
        (nlon, nlat), radius=radius, lat_extent=(-LAT_MAX, LAT_MAX))


def sphere_model(*, coriolis, csqr=0.7, ro=0.4, advection=False):
    """Assemble a tiny spherical shallow-water model.

    ``advection=False`` (the default) isolates the linear core's metric
    divergence (the ``MetricScaled`` reciprocal). ``advection=True``
    additionally exercises the nonlinear Sadourny scheme, whose chart
    path divides the kinetic energy by the centre metric ``sqrt_g``
    (``sqg_p``, an exact zero in the walled polar/halo padding); that
    divide carries its own ``_sealed_metric_divide`` guard.
    """
    return sw.Model(
        grid=sphere_grid(), coords=("lon", "lat"), csqr=csqr,
        rossby_number=ro, coriolis=coriolis, advection=advection,
        time_stepper=fr.model.time_steppers.AdamBashforth(1e-3, order=3))


def set_random(model, seed=3, amp=0.1):
    """Random prognostics (walls structural: v has no cap DOFs)."""
    rng = np.random.default_rng(seed)
    model.set_fields(
        u=amp * rng.standard_normal(model.state["u"].shape),
        v=amp * rng.standard_normal(model.state["v"].shape),
        p=0.3 * amp * rng.standard_normal(model.state["p"].shape))


def ic_loss(model, n_steps):
    """Quadratic loss in the initial pressure storage via _chunk_body.

    Splices the initial-pressure storage leaf (found by identity in the
    flattened carry) with the differentiation variable, advances
    ``n_steps`` through the pure kernel ``_chunk_body`` (the public
    ``advance`` path is not differentiable), and reduces the final
    state to ``sum(u**2 + v**2 + p**2)`` of the interior DOFs.
    """
    record = model._artifacts.record
    stepper = model._stepper
    p_leaf = model._carry.state["p"].storage
    leaves, treedef = jax.tree_util.tree_flatten(model._carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is p_leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        carry = jax.tree_util.tree_unflatten(treedef, new)
        state = _chunk_body(record, n_steps, carry, stepper).state
        return (jnp.sum(state["p"].data ** 2)
                + jnp.sum(state["u"].data ** 2)
                + jnp.sum(state["v"].data ** 2))

    return loss, p_leaf


# ================================================================
#  The reverse gradient is finite and matches finite differences
# ================================================================
@pytest.mark.parametrize("with_rotation", [
    pytest.param(False, id="no-rotation"),
    pytest.param(True, id="rotation"),
])
def test_sphere_ic_grad_is_finite_and_matches_fd(with_rotation):
    """Grad through the sphere chart run w.r.t. the IC: finite, FD."""
    coriolis = None
    if with_rotation:
        coriolis = sw.modules.RotationCoriolis(
            omega=(0.0, 0.0, 1.5), coords=("lon", "lat"),
            metric_weight="csqr")
    model = sphere_model(coriolis=coriolis)
    set_random(model)
    loss, p_leaf = ic_loss(model, n_steps=6)

    grad = np.asarray(jax.grad(loss)(p_leaf))
    # the pre-seal bug NaNed every entry with a data path through the
    # chart divergence; the MetricScaled seal keeps them finite
    assert bool(np.all(np.isfinite(grad)))

    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(p_leaf.shape),
                            dtype=p_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(p_leaf + eps * direction))
          - float(loss(p_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


def test_sphere_run_stays_finite():
    """Forward sanity: the linear chart run produces no NaN interior."""
    model = sphere_model(coriolis=None)
    set_random(model)
    state = _chunk_body(model._artifacts.record, 6,
                        model._carry, model._stepper).state
    for name in ("u", "v", "p"):
        assert bool(np.all(np.isfinite(np.asarray(state[name].data))))


# ================================================================
#  The nonlinear (Sadourny) chart run: the sqg_p kinetic-energy seal
# ================================================================
def test_sphere_advection_ic_grad_is_finite_and_matches_fd():
    """Grad through the Sadourny chart run w.r.t. the IC: finite, FD.

    ``advection=True`` routes through ``SadournyAdvection._advect_chart``,
    which divides the chart kinetic energy by the centre metric
    ``sqg_p``. That weight is an exact zero in the walled sphere's
    never-valid polar/halo padding, so the bare quotient's reverse VJP
    is the masked ``0 * inf = NaN`` that poisoned every gradient with a
    data path through the kinetic energy; ``_sealed_metric_divide`` keeps
    the forward run bitwise identical on valid cells while making
    ``jax.grad`` finite and matched to a central finite difference.
    """
    model = sphere_model(coriolis=None, advection=True)
    set_random(model)
    loss, p_leaf = ic_loss(model, n_steps=6)

    grad = np.asarray(jax.grad(loss)(p_leaf))
    # the pre-seal bug NaNed every entry with a data path through the
    # Sadourny chart kinetic energy; the sqg_p seal keeps them finite
    assert bool(np.all(np.isfinite(grad)))

    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(p_leaf.shape),
                            dtype=p_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(p_leaf + eps * direction))
          - float(loss(p_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)
