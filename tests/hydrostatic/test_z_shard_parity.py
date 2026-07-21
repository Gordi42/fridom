"""Forced-4 z-shard seam parity of the hydrostatic step.

Description
-----------
Two distinct z-shard seam regressions on the bounded vertical, both
device-count-invariance checks (auto-negotiated z-sharded run vs an
explicit single-device run):

* The ``Restriction`` (``Outer -> Inner``) seam (merge ``ef1a4d08``).
  The vertical advective flux restricts the diagnosed ``w`` (on the
  both-boundary ``Outer`` faces) onto the interior ``Inner`` flux faces,
  and ``Restriction`` reads ``Outer[m + 1]`` -- one slot above each
  output. When the negotiation shards the bounded vertical (a small
  horizontal extent forces this), the last interior face of every shard
  is the neighbour shard's boundary face, reached across the seam
  through one ghost slot. The pre-fix ``halo 0`` declaration left that
  ghost at the reshard's zero fill, so the buoyancy tendency diverged
  on the exact z-shard seams; the ``(0, 1)`` footprint makes
  ``_ensure_valid`` sync it. Exercised by the *centered* test, whose
  vertical flux has no other seam.

* The WENO union-window seam (merge for ``weno_momentum_z_seam.md``).
  ``WENOAdvection``'s one-pass ``_SelectedFaceReconstruction`` reads an
  ``order + 1`` cell union window but, under the halo trace, recorded
  only the left biased half's footprint (``(2, 2)`` for the primal
  order-5 ``Center -> face`` direction) where the union needs the
  biased *pair's* ``(2, 3)``. A cell-centered quantity's vertical
  reconstruction (``u``/``v``/``b`` are z-centered in the hydrostatic
  model, where ``w`` is diagnosed rather than prognostic) therefore
  negotiated z-halo 2 and computed a silent ``~1e-5`` seam in ``u``/
  ``v`` (amplitude-masked in ``b``) at every shard boundary. The fix
  presents the union footprint to the negotiation, lifting z-halo to 3
  and dropping every seam to the reassociation floor. Exercised by the
  *weno* test on all four prognostic fields.

These tests are ``multi_device`` marked (a no-op skip on one device).
Self-contained builders per the AGENTS oversized-module /
self-contained-shard rule.
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
        grid=grid,
        core=hy.Core(gravity=1000.0),
        time_stepper=AdamBashforth(1e-3, order=2, eps=0.1),
        coriolis=hy.FPlaneCoriolis(f0=1.0),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.SplitExplicitFreeSurface(substeps=16),
        advection=advection,
        chunk_size=8)


def _run(nx, ny, nz, advection, device_ids, b_amp=0.01):
    """Advance a fresh model and return the gathered prognostic state.

    Returns ``(state, sharded, halo)``: the host-array state, the
    negotiated sharded axis names, and the negotiated per-name ghost
    widths (``grid.decomposition.halo``).
    """
    grid = _make_grid(nx, ny, nz, device_ids)
    model = _make_model(grid, advection)
    model.set_fields(
        u=lambda x, y, z: jnp.sin(x) * jnp.cos(y) * jnp.cos(jnp.pi * z),
        v=lambda x, y, z: 0.3 * jnp.cos(x) * jnp.cos(jnp.pi * z),  # noqa: ARG005
        b=lambda x, y, z: b_amp * jnp.cos(jnp.pi * z))  # noqa: ARG005
    model.advance(8)
    sharded = [name for name, _
               in grid.decomposition.default_layout.device_axes]
    state = {k: np.asarray(model.state[k].data)
             for k in ("u", "v", "b", "ps")}
    return state, sharded, dict(grid.decomposition.halo.widths)


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
    many, sharded, _ = _run(4, 4, 16, CenteredAdvection(), None)
    one, _, _ = _run(4, 4, 16, CenteredAdvection(), (0,))
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
    # b at O(1) amplitude (cos(pi z), stable under n2 = 1): the pre-fix
    # b seam scaled with the IC amplitude, so a 0.01 IC masked it below
    # tolerance -- the residual was the SAME WENO union-window under-
    # provisioning as u/v (the Restriction path is correctly provisioned
    # since ef1a4d08), not residual Restriction error. With the union
    # footprint negotiated (z-halo 3) it is at the reassociation floor.
    many, sharded, halo = _run(
        8, 8, 16, WENOAdvection(order=5), None, b_amp=1.0)
    one, _, _ = _run(8, 8, 16, WENOAdvection(order=5), (0,), b_amp=1.0)
    assert "z" in sharded
    # the union window's pair reach (footprint_reach(order + 1, m0) =
    # (2, 3) primal) lifts the negotiated vertical halo to 3; a demand
    # regression back to the left half (halo 2) fails here loudly
    # instead of only through the parity drift below.
    assert halo["z"] >= 3
    # with the union footprint provisioned, u/v/b/ps are all device-
    # count invariant to the reassociation floor (pre-fix: u ~9e-6,
    # v ~6e-6 seams at the shard boundaries; b masked ~2e-5 at O(1))
    for name in ("u", "v", "b", "ps"):
        assert _close(many[name], one[name], atol=1e-11), name
