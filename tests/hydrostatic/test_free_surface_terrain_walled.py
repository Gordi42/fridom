r"""Free surface on a terrain (sigma) grid with a walled *horizontal* axis.

The terrain + walled-horizontal layer of the walled-horizontal gap (the
flat / immersed layers closed 2026-07-18,
``test_free_surface_walled.py``). On a sigma-chart terrain grid the
slope metric ``d<mapped>_d<axis>`` chains the discrete ``H_x`` onto the
walled-axis interior faces (``Inner``) and must reach the cell centres
(``Core._diagnose_w`` via ``slope_velocity_on_w``, and the
baroclinic ``_slope_gradient``). That ``Inner -> Center`` move has no
BC-free ``interpolate`` row, so before the odd-tangent Dirichlet retag
in ``CoordinateMapping._at_space`` these grids failed to assemble. The
tangent is odd at the wall (``H_x = 0`` for a wall-mirror-even depth),
so the Dirichlet sibling resolves it exactly.

All three free surfaces are covered on a walled chart: the **explicit**
and **implicit** variants, and the **split-explicit** subcycle (H3 now
retired — the volume-exact terrain transport form, GM-D1 option 1). Self-
contained builders (AGENTS oversized-module rule).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.model import _chunk_body
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.coordinate_mapping import CoordinateMapping

IM = fr.spatial.meshes.IntervalMesh
N2, CSQR = 2.0, 1.0

# (periodic-x, periodic-y) for each walled-horizontal configuration
WALLS = {
    "x": (False, True),
    "y": (True, False),
    "xy": (False, False),
}


def _depth(x, y):
    """Return a wall-mirror-even terrain depth (even about x=0,1, y=0,1)."""
    return 1.0 + 0.2 * jnp.cos(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def _mapping():
    # a fresh (grid-bound) descriptor per grid
    return CoordinateMapping(maps={"zp": lambda z, H: z * H},
                             params={"H": _depth})


def _grid(periodic, nx=8, ny=8, nz=4):
    """Return a (partly) walled-horizontal sigma-chart terrain grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=periodic[0], name="x"),
        IM(ny, (0.0, 1.0), periodic=periodic[1], name="y"),
        IM(nz, (-1.0, 0.0), periodic=False, name="z")),
        mapping=_mapping())


def _model(grid, *, free_surface=None, advection=False, f0=0.5,
           dt=2e-3, n2=N2, coriolis=True):
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=CSQR),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=hy.FPlaneCoriolis(f0=f0) if coriolis else None,
        stratification=hy.ConstantStratification(n2=n2),
        free_surface=free_surface or hy.ExplicitFreeSurface(),
        advection=advection)


def _random_ic(model, scale=0.1, seed=0):
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        k: scale * rng.standard_normal(model.state[k].shape)
        for k in ("u", "v", "b", "ps")})


def _nodes(grid, space, name):
    return grid.evaluation_nodes(space.bare, name).data


# ================================================================
#  Assembles and runs finite on walled horizontal axes
# ================================================================
@pytest.mark.parametrize("wall", list(WALLS), ids=list(WALLS))
@pytest.mark.parametrize(
    "advection",
    [pytest.param(False, id="linear"), pytest.param(True, id="advected")],
)
@pytest.mark.parametrize(
    "free_surface",
    [pytest.param(hy.ExplicitFreeSurface, id="explicit"),
     pytest.param(hy.ImplicitFreeSurface, id="implicit")],
)
def test_free_surface_assembles_and_runs_finite_on_walls(
        wall, advection, free_surface):
    model = _model(_grid(WALLS[wall]), free_surface=free_surface(),
                   advection=advection)
    _random_ic(model)
    model.advance(10)
    assert not model.panicked
    for k in ("u", "v", "b", "ps"):
        data = np.asarray(model.state[k].data)
        assert bool(np.isfinite(data).all()), (wall, advection, k)


@pytest.mark.parametrize("wall", list(WALLS), ids=list(WALLS))
def test_split_explicit_runs_finite_on_terrain_walls(wall):
    # the split-explicit barotropic subcycle (H3, retired) now engages on
    # a walled terrain grid: the physical transport depth H_a = int J dz is
    # retagged onto the Dirichlet-tagged transport face, so div(H_a ubar)
    # keys the walled diff row (zero normal transport through the wall).
    model = hy.Model(
        grid=_grid(WALLS[wall]),
        core=hy.Core(gravity=CSQR),
        time_stepper=AdamBashforth(2e-3, order=3),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        stratification=hy.ConstantStratification(n2=N2),
        free_surface=hy.SplitExplicitFreeSurface(substeps=16),
        advection=False)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        k: 0.1 * rng.standard_normal(model.state[k].shape)
        for k in ("u", "v", "b", "ps")})
    model.advance(10)
    assert not model.panicked
    for k in ("u", "v", "b", "ps", "U", "V"):
        assert bool(np.isfinite(np.asarray(model.state[k].data)).all()), (
            wall, k)


def _split_model(grid, dt):
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=CSQR),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=None,
        stratification=hy.ConstantStratification(n2=0.0),
        free_surface=hy.SplitExplicitFreeSurface(substeps=16),
        advection=False)


def test_split_terrain_channel_matches_the_mirror_image_run():
    r"""A walled-x terrain split channel equals its doubled periodic mirror.

    The volume-exact terrain subcycle preserves the reflection symmetry of
    a doubly-long periodic domain exactly: ``ps`` cell scalars
    even-extended, the wall-normal transport (``u``, ``U``) odd-extended
    (the two wall faces forced to zero), the wall-mirror-even depth giving
    an odd slope at the wall. Measured drift: ps 2e-17, u / U exactly 0.0.
    """
    nx, ny, nz, steps, dt = 6, 3, 4, 12, 2e-3
    walled = fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=False, name="x"),
        IM(ny, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (-1.0, 0.0), periodic=False, name="z")),
        mapping=_mirror_mapping())
    doubled = fr.spatial.Grid((
        IM(2 * nx, (0.0, 2.0), periodic=True, name="x"),
        IM(ny, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (-1.0, 0.0), periodic=False, name="z")),
        mapping=_mirror_mapping())
    mw = _split_model(walled, dt)
    mp = _split_model(doubled, dt)

    ps_cells = np.cos(np.pi * (np.arange(nx) + 0.5) / nx) + 0.3
    u_faces = 0.2 * np.sin(np.pi * np.arange(1, nx) / nx)  # nx-1 faces
    wps = np.zeros((nx, ny, 1))
    wps[:, :, 0] = ps_cells[:, None]
    wu = np.zeros((nx - 1, ny, nz))
    wu[:] = u_faces[:, None, None]
    wU = np.zeros((nx - 1, ny, 1))
    wU[:, :, 0] = u_faces[:, None]
    mw.set_fields(ps=wps, u=wu, U=wU)

    ps_ext = np.concatenate([ps_cells, ps_cells[::-1]])
    u_ext = np.concatenate([u_faces, [0.0], -u_faces[::-1], [0.0]])
    pps = np.zeros((2 * nx, ny, 1))
    pps[:, :, 0] = ps_ext[:, None]
    pu = np.zeros((2 * nx, ny, nz))
    pu[:] = u_ext[:, None, None]
    pU = np.zeros((2 * nx, ny, 1))
    pU[:, :, 0] = u_ext[:, None]
    mp.set_fields(ps=pps, u=pu, U=pU)

    mw.advance(steps)
    mp.advance(steps)
    assert not mw.panicked
    assert not mp.panicked
    wps_f = np.asarray(mw.state["ps"].data)
    pps_f = np.asarray(mp.state["ps"].data)
    wu_f = np.asarray(mw.state["u"].data)
    pu_f = np.asarray(mp.state["u"].data)
    wU_f = np.asarray(mw.state["U"].data)
    pU_f = np.asarray(mp.state["U"].data)
    # the run is non-trivial: ps and u move well away from the IC
    assert np.abs(wps_f - wps).max() > 1e-3
    assert np.abs(wu_f - wu).max() > 1e-3
    assert np.abs(wps_f - pps_f[:nx]).max() < 1e-12
    assert np.abs(wu_f - pu_f[:nx - 1]).max() < 1e-12
    assert np.abs(wU_f - pU_f[:nx - 1]).max() < 1e-12


@pytest.mark.parametrize("wall", ["x", "y", "xy"])
def test_split_terrain_ps_volume_conserved_on_walls(wall):
    # the volume-exact terrain subcycle conserves the physical (J-weighted)
    # ps volume to round-off on a no-flux wall (div(H_a ubar) telescopes to
    # zero against the walls). Measured drift <= 1e-16; pinned above.
    model = hy.Model(
        grid=_grid(WALLS[wall]),
        core=hy.Core(gravity=CSQR),
        time_stepper=AdamBashforth(2e-3, order=3),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        stratification=hy.ConstantStratification(n2=N2),
        free_surface=hy.SplitExplicitFreeSurface(substeps=16),
        advection=False)
    _random_ic(model, seed=3)

    def volume():
        return float(jnp.sum(model.state["ps"].integrate().data))

    before = volume()
    model.advance(20)
    after = volume()
    assert not model.panicked
    assert abs(after - before) < 1e-13 * max(abs(before), 1.0)


# ================================================================
#  Volume conservation: the implicit ps integral is exact
# ================================================================
@pytest.mark.parametrize("wall", ["x", "y", "xy"])
def test_implicit_ps_volume_conserved_to_machine_precision(wall):
    # the implicit free surface projects the barotropic transport onto
    # the no-flux-wall nullspace, so the J-weighted (physical) surface-
    # pressure volume int(ps) is a machine-precision invariant on a
    # terrain + walled grid. Measured drift <= 3.5e-18; pinned above.
    model = _model(_grid(WALLS[wall]),
                   free_surface=hy.ImplicitFreeSurface())
    _random_ic(model, seed=3)

    def volume():
        return float(jnp.sum(model.state["ps"].integrate().data))

    before = volume()
    model.advance(20)
    after = volume()
    assert not model.panicked
    assert abs(after - before) < 1e-14 * max(abs(before), 1.0)


# ================================================================
#  Rest state over topography (the sigma PG-error gate, walled)
# ================================================================
def _rest_tendency(grid, model):
    coll = model.state["b"].function_space
    zp = (_nodes(grid, coll, "z")
          * _depth(_nodes(grid, coll, "x"), _nodes(grid, coll, "y")))
    model.set_fields(
        u=np.zeros(model.state["u"].shape),
        v=np.zeros(model.state["v"].shape),
        b=np.asarray(-N2 * zp),
        ps=np.zeros(model.state["ps"].shape))
    dX = model.tendency(model.state)
    return max(float(jnp.abs(dX["u"].data).max()),
               float(jnp.abs(dX["v"].data).max()))


def test_rest_state_pressure_gradient_error_converges_on_a_wall():
    # a stratified fluid at rest over topography on a walled-x channel:
    # the slope-corrected gradient leaves only a truncation-order
    # residual current that vanishes at ~2nd order (the wall-adjacent
    # slope metric is exact for a wall-mirror-even depth). Measured
    # orders (12->24->48): 1.81, 1.95.
    errs = []
    for n in (12, 24, 48):
        grid = _grid(WALLS["x"], nx=n, ny=n, nz=n)
        errs.append(_rest_tendency(grid, _model(grid, coriolis=False)))
    orders = np.log2(np.asarray(errs[:-1]) / np.asarray(errs[1:]))
    assert bool(np.all(orders > 1.6)), (errs, orders)


def test_rest_state_stays_near_rest_over_a_short_run():
    grid = _grid(WALLS["xy"], nx=16, ny=16, nz=8)
    model = _model(grid, coriolis=False)
    coll = model.state["b"].function_space
    zp = (_nodes(grid, coll, "z")
          * _depth(_nodes(grid, coll, "x"), _nodes(grid, coll, "y")))
    model.set_fields(
        u=np.zeros(model.state["u"].shape),
        v=np.zeros(model.state["v"].shape),
        b=np.asarray(-N2 * zp),
        ps=np.zeros(model.state["ps"].shape))
    model.run(10, progress=False)
    assert not model.panicked
    assert float(jnp.abs(model.state["u"].data).max()) < 1e-2
    assert float(jnp.abs(model.state["v"].data).max()) < 1e-2


# ================================================================
#  Mirror-symmetry physics gate (the channel image trick, terrain)
# ================================================================
def _mirror_depth(x, y):
    # even about x=0 and x=1 (in x); the doubled sampling on [0, 2] is
    # then exactly the x-even extension of the [0, 1] channel depth
    return 1.0 + 0.2 * jnp.cos(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def _mirror_mapping():
    return CoordinateMapping(maps={"zp": lambda z, H: z * H},
                             params={"H": _mirror_depth})


def _bfun(x, y, z):
    # a genuinely x-structured (but x-even) buoyancy so the diagnosed
    # w's slope terms (u Z_x + v Z_y) drive db/dt in the mirror run
    return 0.3 * jnp.cos(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y) * (z + 0.5)


def _mirror_model(grid, dt):
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=CSQR),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=None,
        stratification=hy.ConstantStratification(n2=N2),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)


def test_channel_matches_the_mirror_image_run():
    r"""A walled-x terrain channel equals its doubled periodic mirror.

    Solid walls at ``x = 0, 1`` are the reflection symmetry of a
    doubly-long periodic domain with a mirror-symmetric state: cell
    scalars (``ps``, ``b``) even-extended, the wall-normal ``u``
    odd-extended (the wall faces forced to zero). The terrain depth is
    even about the walls, so its slope ``Z_x`` is odd there (exactly the
    Dirichlet parity the retag claims); the centered linear scheme
    preserves the symmetry exactly, so the walled run reproduces the
    periodic run restricted to ``[0, 1]`` to round-off. Measured drift:
    ps 6.9e-18, u 5.6e-17, b 1.4e-17.
    """
    nx, ny, nz, steps, dt = 6, 3, 4, 12, 2e-3
    walled = fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=False, name="x"),
        IM(ny, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (-1.0, 0.0), periodic=False, name="z")),
        mapping=_mirror_mapping())
    doubled = fr.spatial.Grid((
        IM(2 * nx, (0.0, 2.0), periodic=True, name="x"),
        IM(ny, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (-1.0, 0.0), periodic=False, name="z")),
        mapping=_mirror_mapping())
    mw = _mirror_model(walled, dt)
    mp = _mirror_model(doubled, dt)

    ps_cells = np.cos(np.pi * (np.arange(nx) + 0.5) / nx) + 0.3
    u_faces = 0.2 * np.sin(np.pi * np.arange(1, nx) / nx)  # nx-1 faces
    wps = np.zeros((nx, ny, 1))
    wps[:, :, 0] = ps_cells[:, None]
    wu = np.zeros((nx - 1, ny, nz))
    wu[:] = u_faces[:, None, None]
    bw = mw.state["b"].function_space
    wb = np.broadcast_to(np.asarray(_bfun(
        _nodes(walled, bw, "x"), _nodes(walled, bw, "y"),
        _nodes(walled, bw, "z"))), mw.state["b"].shape)
    mw.set_fields(ps=wps, u=wu, b=wb)

    ps_ext = np.concatenate([ps_cells, ps_cells[::-1]])
    u_ext = np.concatenate([u_faces, [0.0], -u_faces[::-1], [0.0]])
    pps = np.zeros((2 * nx, ny, 1))
    pps[:, :, 0] = ps_ext[:, None]
    pu = np.zeros((2 * nx, ny, nz))
    pu[:] = u_ext[:, None, None]
    bp = mp.state["b"].function_space
    pb = np.broadcast_to(np.asarray(_bfun(
        _nodes(doubled, bp, "x"), _nodes(doubled, bp, "y"),
        _nodes(doubled, bp, "z"))), mp.state["b"].shape)
    mp.set_fields(ps=pps, u=pu, b=pb)

    mw.advance(steps)
    mp.advance(steps)
    assert not mw.panicked
    assert not mp.panicked

    wps_f = np.asarray(mw.state["ps"].data)
    pps_f = np.asarray(mp.state["ps"].data)
    wu_f = np.asarray(mw.state["u"].data)
    pu_f = np.asarray(mp.state["u"].data)
    wb_f = np.asarray(mw.state["b"].data)
    pb_f = np.asarray(mp.state["b"].data)
    assert np.abs(wps_f - pps_f[:nx]).max() < 1e-12
    assert np.abs(wu_f - pu_f[:nx - 1]).max() < 1e-12
    assert np.abs(wb_f - pb_f[:nx]).max() < 1e-12


# ================================================================
#  Autodiff regression (differentiability policy)
# ================================================================
def test_grad_through_walled_terrain_run_matches_finite_difference():
    r"""``jax.grad`` w.r.t. the initial ``b`` on a walled-x terrain run.

    Differentiate the pure kernel ``_chunk_body`` w.r.t. the initial
    buoyancy (which flows through the slope-corrected pressure gradient,
    hence the Dirichlet-retagged ``Z_x`` metric) and check a random
    directional projection against a central finite difference.
    """
    model = _model(_grid(WALLS["x"], nx=8, ny=8, nz=4), dt=2e-3)
    rng = np.random.default_rng(1)
    model.set_fields(b=0.1 * rng.standard_normal(model.state["b"].shape))
    record = model._artifacts.record
    carry = model._carry
    stepper = model._stepper

    leaf = carry.state["b"].storage
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        spliced = jax.tree_util.tree_unflatten(treedef, new)
        final = _chunk_body(record, 8, spliced, stepper)
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
