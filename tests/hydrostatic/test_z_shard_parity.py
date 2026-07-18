"""Forced-4 z-shard seam parity of the hydrostatic step.

Description
-----------
Regression for the ``Restriction`` (``Outer -> Inner``) seam bug: the
vertical advective flux restricts the diagnosed ``w`` (on the
both-boundary ``Outer`` faces) onto the interior ``Inner`` flux faces,
and ``Restriction`` reads ``Outer[m + 1]`` -- one slot above each
output. When the negotiation shards the bounded vertical (a small
horizontal extent forces this), the last interior face of every shard
is the neighbour shard's boundary face, reached across the seam through
one ghost slot. The pre-fix ``halo 0`` declaration left that ghost at
the reshard's zero fill, so the buoyancy tendency diverged on the exact
z-shard seams; the ``(0, 1)`` footprint makes ``_ensure_valid`` sync it.

These tests compare an auto-negotiated (z-sharded under the forced-4
suite) run against an explicit single-device run and are ``multi_device``
marked (a no-op skip on one device). Self-contained builders per the
AGENTS oversized-module / self-contained-shard rule.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.modules.advection import (
    CenteredAdvection,
    WENOAdvection,
)
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

TWO_PI = 2.0 * np.pi


def _make_grid(nx, ny, nz, device_ids):
    """Doubly-periodic horizontal, bounded vertical (shards on z)."""
    return fr.spatial.Grid(
        (
            fr.spatial.meshes.IntervalMesh(
                nx, (0.0, TWO_PI), periodic=True, name="x"),
            fr.spatial.meshes.IntervalMesh(
                ny, (0.0, TWO_PI), periodic=True, name="y"),
            fr.spatial.meshes.IntervalMesh(
                nz, (0.0, 1.0), periodic=False, name="z"),
        ),
        device_ids=device_ids)


def _make_model(grid, advection):
    """Return a small nonlinear hydrostatic model on the advection."""
    return hy.Model(
        grid=grid, dt=1e-3, csqr=1000.0,
        free_surface=hy.SplitExplicitFreeSurface(substeps=16),
        coriolis=hy.FPlaneCoriolis(f0=1.0),
        stratification=hy.ConstantStratification(n2=1.0),
        advection=advection,
        time_stepper=AdamBashforth(1e-3, order=2, eps=0.1),
        chunk_size=8)


def _run(nx, ny, nz, advection, device_ids):
    """Advance a fresh model and return the gathered prognostic state."""
    grid = _make_grid(nx, ny, nz, device_ids)
    model = _make_model(grid, advection)
    model.set_fields(
        u=lambda x, y, z: jnp.sin(x) * jnp.cos(y) * jnp.cos(jnp.pi * z),
        v=lambda x, y, z: 0.3 * jnp.cos(x) * jnp.cos(jnp.pi * z),  # noqa: ARG005
        b=lambda x, y, z: 0.01 * jnp.cos(jnp.pi * z))  # noqa: ARG005
    model.advance(8)
    sharded = [name for name, _
               in grid.decomposition.default_layout.device_axes]
    state = {k: np.asarray(model.state[k].data)
             for k in ("u", "v", "b", "ps")}
    return state, sharded


def _close(a, b, atol):
    """Backend-aware parity: bitwise off cpu, tight atol on forced cpu.

    Forced-host-device cpu emulation reassociates the multi-device FP
    reductions relative to the single-device program, so pointwise
    parity holds only to a tight absolute tolerance there (the
    ``test_multi_device.invariant`` convention); a device-count bug is
    O(1) at the seam, far above it.
    """
    a, b = np.asarray(a), np.asarray(b)
    if jax.default_backend() == "cpu":
        return np.allclose(a, b, rtol=0.0, atol=atol)
    return np.array_equal(a, b)


@pytest.mark.multi_device
def test_centered_z_shard_parity(forced_devices):
    # 4x4x16: the tiny horizontal forces the negotiation onto the
    # vertical, so z shards and the Outer(z) w straddles the seams.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many, sharded = _run(4, 4, 16, CenteredAdvection(), None)
    one, _ = _run(4, 4, 16, CenteredAdvection(), (0,))
    assert "z" in sharded
    # centered has no other vertical seam, so every prognostic field is
    # device-count invariant to the reassociation floor (~1e-16 here)
    for name in ("u", "v", "b", "ps"):
        assert _close(many[name], one[name], atol=1e-11), name


@pytest.mark.multi_device
def test_weno_z_shard_parity_buoyancy(forced_devices):
    # 8x8x16: weno5's wider horizontal halo makes the vertical the
    # shardable fallback, so z shards here too.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many, sharded = _run(8, 8, 16, WENOAdvection(order=5), None)
    one, _ = _run(8, 8, 16, WENOAdvection(order=5), (0,))
    assert "z" in sharded
    # the buoyancy seam is the direct Restriction defect (pre-fix rel
    # ~5e-2 / abs ~2.4e-4 on the seams; the fix drops it to ~1e-7). u/v
    # carry a SEPARATE, pre-existing weno vertical-reconstruction seam
    # (the WenoReconstruction footprint exceeds the negotiated z-halo of
    # 2; cured only by a wider z-halo) that this fix does not address,
    # so only b is asserted tight here and the momenta are checked
    # finite.
    assert _close(many["b"], one["b"], atol=1e-5)
    for name in ("u", "v", "ps"):
        assert np.isfinite(many[name]).all(), name
        assert np.isfinite(one[name]).all(), name
