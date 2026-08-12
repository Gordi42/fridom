r"""Wide reconstruction stencils on a thin periodic axis.

A periodic axis with fewer cells than the negotiated halo (``nz = 1``
is the flat "2-D" direction) is filled by tiling the true region, so a
wide stencil sees a well-defined window and needs no scheme downgrade.
The load-bearing claims, each with a test below:

- ``WENOAdvection`` at orders 3 and 5 (halo 2 and 3) assembles and
  steps on ``nz = 1`` and ``nz = 2``, both at or beyond the axis
  length.
- A flat run reproduces a z-replicated deep run **bitwise** on the
  horizontal state: the fill makes the column constant, so the
  reconstruction along z is the identity and the flux difference
  cancels. The vertical velocity is zero to roundoff (the CG
  projection is not bit-exact, so it is not identically zero).
- ``jax.grad`` through ``Model.propagator`` on the flat grid is finite
  and matches a central finite difference (AGENTS.md,
  "Differentiability policy"). WENO's smoothness indicators all vanish
  on a constant column, so this is the case where a masked singularity
  in the weight normalization would surface. The gradient on the ghost
  slots is exactly zero -- the tiled fill is a pure function of the
  true DOFs and leaks nothing back.

Root cause and analysis:
``design/research/thin_axis_halo_investigation.md``.

Self-contained per the oversized-module shard convention: the small
builders are duplicated rather than imported across test files.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 8
DT = 5e-3
STEPS = 4
TWO_PI = 2.0 * np.pi
# WENO's nonlinear weights make the loss only piecewise smooth, so the
# central difference needs a small step to reach its asymptotic regime
# (measured: 9e-2 relative error at 1e-4, 2e-5 at 1e-7).
FD_EPS = 1e-7


def make_model(nz, order=5):
    """Return a tiny periodic ``N x N x nz`` WENO model."""
    grid = Grid((
        IntervalMesh(N, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(N, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(nz, (0.0, TWO_PI), periodic=True, name="z"),
    ))
    return nh.Model(
        grid=grid,
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        buoyancy=nh.ConstantStratification(n2=1.0),
        advection=nh.modules.WENOAdvection(order=order))


def seeded(nz, order=5):
    """Return a seeded model with a z-independent sheared IC."""
    model = make_model(nz, order=order)
    ax = (np.arange(N) + 0.5) * (TWO_PI / N)
    x, y = np.meshgrid(ax, ax, indexing="ij")
    u2d, v2d = 0.2 * np.sin(y), 0.2 * np.cos(x)
    model.set_fields(u=np.repeat(u2d[:, :, None], nz, axis=2),
                     v=np.repeat(v2d[:, :, None], nz, axis=2))
    return model


@pytest.mark.parametrize("nz", [1, 2])
@pytest.mark.parametrize("order", [3, 5])
def test_wide_weno_runs_on_a_thin_axis(nz, order):
    model = seeded(nz, order=order)
    # the biased reconstruction negotiates order // 2 + 1 ghosts, so
    # every pair here reaches at least as far as the axis is long
    assert dict(model.grid.decomposition.halo.widths)["z"] == order // 2 + 1
    model.advance(2)
    for name in ("u", "v", "w", "p"):
        arr = np.asarray(model.state[name].data)
        assert bool(np.all(np.isfinite(arr))), name


def test_flat_run_reproduces_a_z_replicated_deep_run():
    # the tiled fill makes the flat column constant, so the flat run
    # is the deep run's every level
    flat, deep = seeded(1), seeded(4)
    flat.advance(STEPS)
    deep.advance(STEPS)
    for name in ("u", "v", "p"):
        got = np.asarray(flat.state[name].data)
        ref = np.asarray(deep.state[name].data)
        # the deep run stays exactly z-uniform ...
        assert np.max(np.abs(ref - ref[:, :, :1])) == 0.0, name
        # ... and the flat run is that column, bitwise
        assert np.max(np.abs(got[:, :, 0] - ref[:, :, 0])) == 0.0, name


def test_vertical_velocity_is_zero_to_roundoff_on_a_flat_axis():
    # every window along z is identical, so the reconstructed face
    # values are too and the flux difference cancels; what remains is
    # CG-projection roundoff, ~1e-20 against an O(0.2) horizontal flow
    model = seeded(1)
    model.advance(STEPS)
    assert np.max(np.abs(np.asarray(model.state["w"].data))) < 1e-15


def test_grad_through_a_flat_run_matches_fd_directionally():
    """WENO on the constant column keeps reverse mode exact."""
    model = seeded(1)
    run = model.propagator(wrt=("u",), steps=STEPS)
    u0 = model._carry.state["u"].storage

    def loss(field):
        return sum(jnp.sum(f.data ** 2) for f in run((field,)).state)

    grad = np.asarray(jax.grad(loss)(u0))
    assert bool(np.all(np.isfinite(grad)))

    # the fill re-derives every ghost from the true DOFs, so no
    # sensitivity survives on the ghost slots of the flat axis
    width = dict(model.grid.decomposition.halo.widths)["z"]
    assert np.max(np.abs(np.delete(grad, width, axis=2))) == 0.0

    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(u0.shape), dtype=u0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    fd = (float(loss(u0 + FD_EPS * direction))
          - float(loss(u0 - FD_EPS * direction))) / (2.0 * FD_EPS)
    assert directional == pytest.approx(fd, rel=1e-4)
