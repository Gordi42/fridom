r"""Thin axes: a wide stencil on a periodic one, walls on a thin one.

Two regimes, in that order below.

**Periodic.** A periodic axis with fewer cells than the negotiated halo
(``nz = 1`` is the flat "2-D" direction) is filled by tiling the true
region, so a wide stencil sees a well-defined window and needs no
scheme downgrade. The load-bearing claims, each with a test below:

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

**Walled.** A thin *bounded* axis is a different story, and the two
claims that matter are opposite in sign:

- ``nz = 2`` walled leaves the wall-normal ``w`` exactly ONE z DOF, and
  its vertical divergence is genuinely nonzero (the two wall values are
  zero, so ``dw/dz`` is ``+-1/dz``). Any "one DOF along the axis means
  no variation" shortcut would destroy a shipped configuration.
- ``nz = 1`` walled leaves ``w`` **zero** DOFs, which is the honest
  answer: no normal flow at both walls leaves no interior face. It must
  *run*, stepping an empty wall-normal velocity, and it does on a walled
  ``z`` and a walled ``y`` alike. It used to assemble and then raise
  from inside the halo fill at the first step, because the
  ``materialize=True`` fill spelling read a DOF for the Dirichlet vacant
  slot where its documented twin returned exact zeros.

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
from fridom.spatial.operators.finite_difference import FiniteDifference
from fridom.spatial.operators.flux_diff import FluxDifference

N = 8
DT = 5e-3
STEPS = 4
TWO_PI = 2.0 * np.pi
# WENO's nonlinear weights make the loss only piecewise smooth, so the
# central difference needs a small step to reach its asymptotic regime
# (measured: 9e-2 relative error at 1e-4, 2e-5 at 1e-7).
FD_EPS = 1e-7
# the walled runs below carry no advection (a wide walled stencil needs
# 4 cells), so the loss is a quadratic polynomial in the initial field
# and the central difference is exact at a comfortable step
WALL_FD_EPS = 1e-4


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


# ================================================================
#  The walled thin axis
# ================================================================
def walled_model(nz, *, wall="z"):
    """Return a tiny model with one bounded axis of ``nz`` cells."""
    grid = Grid(tuple(
        IntervalMesh(nz if name == wall else N,
                     (0.0, 1.0) if name == wall else (0.0, TWO_PI),
                     periodic=(name != wall), name=name)
        for name in ("x", "y", "z")), device_ids=(0,))
    return nh.Model(
        grid=grid, time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        buoyancy=nh.ConstantStratification(n2=1.0), advection=False)


def seeded_walled(nz, *, wall="z"):
    """Return a walled model with a z-independent sheared IC."""
    model = walled_model(nz, wall=wall)
    ax = (np.arange(N) + 0.5) * (TWO_PI / N)
    x, y = np.meshgrid(ax, ax, indexing="ij")
    u2d, v2d = 0.2 * np.sin(y), 0.2 * np.cos(x)
    model.set_fields(u=np.repeat(u2d[:, :, None], nz, axis=2),
                     v=np.repeat(v2d[:, :, None], nz, axis=2))
    return model


def test_walled_nz2_puts_w_on_a_single_z_dof():
    """The shipped configuration a ``shape[0] == 1`` rule would break."""
    model = walled_model(2)
    w = model.state["w"]
    assert w.function_space.shape == (N, N, 1)
    assert model.grid.factors[2].n_cells == 2
    assert not model.grid.factors[2].periodic


@pytest.mark.parametrize("op", [FiniteDifference(), FluxDifference()],
                         ids=["fd", "flux_diff"])
def test_a_single_z_dof_w_still_has_a_nonzero_vertical_divergence(op):
    r"""``dw/dz`` of a constant ``w`` column is :math:`\pm 1/dz`.

    The two wall values are zero, so the vertical divergence is real
    even though the column holds one DOF. Not vacuous: the same query
    on the *periodic* flat axis is exactly zero.
    """
    model = walled_model(2)
    dz = model.grid.factors[2].dx
    model.set_fields(w=np.ones(model.state["w"].function_space.shape))
    got = np.asarray(op["z"](model.state["w"]).data)
    assert got.shape[2] == 2
    assert np.allclose(got[:, :, 0], 1.0 / dz)
    assert np.allclose(got[:, :, 1], -1.0 / dz)


def test_walled_nz2_steps():
    model = seeded_walled(2)
    model.advance(2)
    for name in ("u", "v", "w", "p"):
        arr = np.asarray(model.state[name].data)
        assert bool(np.all(np.isfinite(arr))), name


@pytest.mark.parametrize("wall", ["z", "y"])
def test_a_walled_one_cell_axis_steps_with_an_empty_normal_velocity(wall):
    """A walled 1-cell axis runs, with no wall-normal velocity at all.

    Owner's semantics call (2026-08-12): make it run rather than refuse
    at assembly. One cell between two no-normal-flow walls leaves no
    interior face, so the wall-normal velocity is a genuinely empty
    array -- and stepping it must not reach into the halo fill for a DOF
    that does not exist.
    """
    normal = {"z": "w", "y": "v"}[wall]
    axis = ("x", "y", "z").index(wall)
    model = walled_model(1, wall=wall)
    assert model.state[normal].function_space.shape[axis] == 0
    assert model.state[normal].data.size == 0
    model.advance(2)
    for name in ("u", "v", "w", "p"):
        arr = np.asarray(model.state[name].data)
        assert bool(np.all(np.isfinite(arr))), name
    assert np.asarray(model.state[normal].data).size == 0


@pytest.mark.parametrize("nz", [1, 2])
def test_grad_through_a_walled_thin_run_matches_fd_directionally(nz):
    """Reverse mode stays exact through the walled thin-axis fill.

    ``nz = 1`` is the case the fill used to refuse; ``nz = 2`` pins that
    the zero-slot rewrite did not perturb the working neighbour. The
    empty ``w`` factor is where a zero-size reduction would poison the
    VJP if the fill leaked a division or a reshape into it.
    """
    model = seeded_walled(nz)
    run = model.propagator(wrt=("u",), steps=STEPS)
    u0 = model._carry.state["u"].storage

    def loss(field):
        return sum(jnp.sum(f.data ** 2) for f in run((field,)).state)

    grad = np.asarray(jax.grad(loss)(u0))
    assert bool(np.all(np.isfinite(grad)))
    assert np.max(np.abs(grad)) > 0.0

    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(u0.shape), dtype=u0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    fd = (float(loss(u0 + WALL_FD_EPS * direction))
          - float(loss(u0 - WALL_FD_EPS * direction))) / (2.0 * WALL_FD_EPS)
    assert directional == pytest.approx(fd, rel=1e-4)
