r"""ExplicitFreeSurface on walled *horizontal* grids (the tag-only arm).

Prefix-mirrored shard of ``hy.modules.free_surface`` covering the
explicit free surface on grids that are bounded in a horizontal axis.
Before the BC-sibling tag-only ``.to`` arm (``ScalarField.to`` /
``HaloTracer.to``) these grids failed to assemble: the pressure-gradient
term reads ``ps.diff("x")`` onto a bare bounded ``Inner(x)`` face and
then ``.to`` a wall-tagged sibling, which the old resolver mistook for a
conversion (``("interpolate", Inner(x))`` — a row that deliberately does
not exist). The arm recognises the tag-only relabel and adopts the tag.

Only the **explicit** free surface is covered here: the implicit and
split-explicit variants have separate follow-on wall fixes on other
branches. Self-contained builders (AGENTS oversized-module rule).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.model import _chunk_body
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.immersed_domain import ImmersedDomain

IM = fr.spatial.meshes.IntervalMesh

# (periodic-x, periodic-y) for each walled-horizontal configuration
WALLS = {
    "x": (False, True),
    "y": (True, False),
    "xy": (False, False),
}


def _grid(periodic, nx=8, ny=8, nz=4):
    """Return a horizontally (partly) walled, bounded-z grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=periodic[0], name="x"),
        IM(ny, (0.0, 1.0), periodic=periodic[1], name="y"),
        IM(nz, (0.0, 1.0), periodic=False, name="z")))


def _model(grid, *, advection=False, f0=0.5, csqr=1.0, n2=0.0, dt=1e-3):
    """Return a linear explicit-free-surface hydrostatic model."""
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=csqr),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=hy.FPlaneCoriolis(f0=f0),
        stratification=hy.ConstantStratification(n2=n2),
        free_surface=hy.ExplicitFreeSurface(),
        advection=advection)


def _random_ic(model, scale=0.1, seed=0):
    """Seed every prognostic field with small random data."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        k: scale * rng.standard_normal(model.state[k].shape)
        for k in ("u", "v", "b", "ps")})


# ================================================================
#  Assembles and runs finite on walled horizontal axes
# ================================================================
@pytest.mark.parametrize("wall", list(WALLS), ids=list(WALLS))
@pytest.mark.parametrize(
    "advection",
    [pytest.param(False, id="linear"), pytest.param(True, id="advected")],
)
def test_explicit_assembles_and_runs_finite_on_walls(wall, advection):
    model = _model(_grid(WALLS[wall]), advection=advection)
    _random_ic(model)
    model.advance(10)
    assert not model.panicked
    for k in ("u", "v", "b", "ps"):
        data = np.asarray(model.state[k].data)
        assert bool(np.isfinite(data).all()), (wall, advection, k)


# ================================================================
#  Volume conservation: the ps integral is a linear invariant
# ================================================================
@pytest.mark.parametrize(
    "advection",
    [pytest.param(False, id="linear"), pytest.param(True, id="advected")],
)
def test_ps_volume_conserved_to_machine_precision(advection):
    # d/dt int(ps) = -csqr int(div U) = 0 on no-flux walls, so the
    # measure-weighted ps integral drifts only at round-off. Measured
    # drift: 0.0 (linear) / ~1.7e-18 (advected); pinned well above.
    model = _model(_grid(WALLS["xy"]), advection=advection, csqr=2.0)
    _random_ic(model)

    def volume():
        return float(jnp.sum(model.state["ps"].integrate().data))

    before = volume()
    model.advance(20)
    after = volume()
    assert not model.panicked
    assert abs(after - before) < 1e-14 * max(abs(before), 1.0)


# ================================================================
#  Mirror-symmetry physics gate (the channel image trick)
# ================================================================
def _mirror_grids(nx, ny=3, nz=4):
    """Return (walled-x channel, doubled periodic-x) grid pair.

    The walled channel stores ``nx`` cells over ``[0, 1]``; the doubled
    domain stores ``2*nx`` cells over ``[0, 2]`` at the identical
    ``dx = 1/nx`` so the two C-grids collocate on ``[0, 1]``.
    """
    walled = fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=False, name="x"),
        IM(ny, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, 1.0), periodic=False, name="z")))
    doubled = fr.spatial.Grid((
        IM(2 * nx, (0.0, 2.0), periodic=True, name="x"),
        IM(ny, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, 1.0), periodic=False, name="z")))
    return walled, doubled


def test_channel_matches_the_mirror_image_run():
    r"""A walled-x channel equals its doubled periodic mirror image.

    Solid walls at ``x = 0, 1`` are the reflection symmetry of a
    doubly-long periodic domain with a mirror-symmetric state: cell
    scalars (``ps``) even-extended, the wall-normal velocity (``u``)
    odd-extended (the two wall faces forced to zero). The centered
    linear scheme (``f = 0``, ``N^2 = 0``, no advection) preserves the
    symmetry exactly, so the walled run reproduces the periodic run
    restricted to ``[0, 1]`` to round-off.

    C-grid correspondence (``dx = 1/nx``): ``ps`` cell ``i`` sits at
    ``(i + 1/2)/nx`` — walled cell ``i`` collocates with periodic cell
    ``i`` for ``i < nx``. ``u`` (the right cell face) sits at
    ``(k + 1)/nx``: walled interior face ``k`` (``k = 0 .. nx-2``,
    ``x = 1/nx .. (nx-1)/nx``) collocates with periodic face ``k``. The
    periodic wall faces ``k = nx-1`` (``x = 1``) and ``k = 2*nx-1``
    (``x = 0``) carry the odd-extension zeros.
    """
    nx, ny, nz, steps, dt = 6, 3, 4, 12, 2e-3
    walled, doubled = _mirror_grids(nx, ny, nz)
    mw = _model(walled, f0=0.0, n2=0.0, csqr=1.0, dt=dt)
    mp = _model(doubled, f0=0.0, n2=0.0, csqr=1.0, dt=dt)

    # walled initial condition (uniform in y, z), a genuine wave state
    ps_cells = np.cos(np.pi * (np.arange(nx) + 0.5) / nx) + 0.3
    u_faces = 0.2 * np.sin(np.pi * np.arange(1, nx) / nx)  # nx-1 faces
    wps = np.zeros((nx, ny, 1))
    wps[:, :, 0] = ps_cells[:, None]
    wu = np.zeros((nx - 1, ny, nz))
    wu[:] = u_faces[:, None, None]
    mw.set_fields(ps=wps, u=wu)

    # even extension of ps, odd extension of u (wall faces zeroed)
    ps_ext = np.concatenate([ps_cells, ps_cells[::-1]])
    u_ext = np.concatenate([u_faces, [0.0], -u_faces[::-1], [0.0]])
    pps = np.zeros((2 * nx, ny, 1))
    pps[:, :, 0] = ps_ext[:, None]
    pu = np.zeros((2 * nx, ny, nz))
    pu[:] = u_ext[:, None, None]
    mp.set_fields(ps=pps, u=pu)

    mw.advance(steps)
    mp.advance(steps)
    assert not mw.panicked
    assert not mp.panicked

    wps_f = np.asarray(mw.state["ps"].data)
    pps_f = np.asarray(mp.state["ps"].data)
    wu_f = np.asarray(mw.state["u"].data)
    pu_f = np.asarray(mp.state["u"].data)
    # ps cells and u interior faces restricted to the channel [0, 1]
    ps_drift = np.abs(wps_f - pps_f[:nx]).max()
    u_drift = np.abs(wu_f - pu_f[:nx - 1]).max()
    # measured: ps 0.0, u ~5.6e-17; pinned honestly above
    assert ps_drift < 1e-12
    assert u_drift < 1e-12


# ================================================================
#  Immersed (cut-cell) mask on a walled horizontal grid
# ================================================================
def test_immersed_mask_on_walls_assembles_and_runs_finite():
    # explicit free surface + a face-aligned immersed bottom on a fully
    # walled-horizontal grid: the arm carries the mapped/immersed
    # pressure-gradient retag through assembly and the run stays finite
    grid = fr.spatial.Grid((
        IM(8, (0.0, 1.0), periodic=False, name="x"),
        IM(8, (0.0, 1.0), periodic=False, name="y"),
        IM(4, (0.0, 1.0), periodic=False, name="z")),
        immersed=ImmersedDomain(lambda x, y, z: (z > 0.25).astype(float)))  # noqa: ARG005
    model = hy.Model(
        grid=grid,
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(1e-3, order=3),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)
    _random_ic(model)
    model.advance(10)
    assert not model.panicked
    assert bool(np.isfinite(np.asarray(model.state["ps"].data)).all())


# ================================================================
#  Autodiff regression (differentiability policy)
# ================================================================
def test_grad_through_walled_run_matches_finite_difference():
    r"""``jax.grad`` w.r.t. the initial ``ps`` on a walled-x run.

    Differentiate the pure kernel ``_chunk_body`` (the public
    ``advance`` path is not differentiable) w.r.t. the initial surface
    pressure and check a random directional projection against a central
    finite difference.
    """
    model = _model(_grid(WALLS["x"], nx=8, ny=8, nz=4),
                   f0=0.5, dt=2e-3)
    rng = np.random.default_rng(1)
    model.set_fields(ps=0.1 * rng.standard_normal(model.state["ps"].shape))
    record = model._artifacts.record
    carry = model._carry
    stepper = model._stepper

    leaf = carry.state["ps"].storage
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        spliced = jax.tree_util.tree_unflatten(treedef, new)
        final = _chunk_body(record, 10, spliced, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    grad = np.asarray(jax.grad(loss)(leaf))
    assert bool(np.all(np.isfinite(grad)))

    direction = jnp.asarray(
        np.random.default_rng(2).standard_normal(leaf.shape),
        dtype=leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(leaf + eps * direction))
          - float(loss(leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)
