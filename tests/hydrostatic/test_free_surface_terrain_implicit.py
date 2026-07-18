r"""The implicit free surface on a terrain-following (sigma) grid (H3).

Prefix-mirrored shard of ``hy.modules.free_surface`` covering the
volume-exact terrain barotropic solve of ``hy.ImplicitFreeSurface``
(GM-D1 option 1 / GM-D2, the deferred H3 item). The model-level gates:
GB-1 exact cancellation, GB-2 flat-chart limit against the shipped flat
spectral solve, GB-3 barotropic-volume conservation, GB-5 autodiff, GB-6
the solve engages. The operator / walls / self-adjointness unit tests
live in ``test_barotropic_pressure.py``. Self-contained per the AGENTS
oversized-module rule.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.hydrostatic.params import CSQR
from fridom.model.context import StepContext
from fridom.model.model import _chunk_body
from fridom.spatial.coordinate_mapping import CoordinateMapping

IM = fr.spatial.meshes.IntervalMesh


# ================================================================
#  Builders (self-contained per the AGENTS oversized-module rule)
# ================================================================
def terrain_grid(n, nz=8, a=0.4):
    """Return a doubly-periodic horizontal sigma grid, depth H(x, y)."""
    def depth(x, y):
        return 1.0 + a * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
    return fr.spatial.Grid((
        IM(n, (0.0, 1.0), periodic=True, name="x"),
        IM(n, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (-1.0, 0.0), periodic=False, name="z")),
        mapping=CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                  params={"H": depth}))


def plain_grid(n, nz=8):
    """Return the flat (no-mapping) twin of :func:`terrain_grid`."""
    return fr.spatial.Grid((
        IM(n, (0.0, 1.0), periodic=True, name="x"),
        IM(n, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (-1.0, 0.0), periodic=False, name="z")))


def model(grid, *, eps=1.0, csqr=3.0, dt=0.05, iterations=30,
          tolerance=1e-8, f0=0.5, preconditioner="spectral",
          multigrid_levels=None):
    """Return a linear hydrostatic model on the implicit free surface."""
    return hy.Model(
        grid=grid, dt=dt, csqr=csqr,
        stratification=hy.ConstantStratification(n2=0.0),
        coriolis=hy.FPlaneCoriolis(f0=f0), advection=False,
        free_surface=hy.ImplicitFreeSurface(
            epsilon=eps, pressure_iterations=iterations,
            pressure_tolerance=tolerance,
            pressure_preconditioner=preconditioner,
            multigrid_levels=multigrid_levels),
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=2))


def ctx_of(csqr, dt):
    """Return a minimal StepContext for a direct CONSTRAINT-stage call."""
    return StepContext(params={CSQR: jnp.asarray(csqr)},
                       clock=jnp.asarray(0.0), dt=jnp.asarray(dt),
                       stage_dt=jnp.asarray(dt))


# ================================================================
#  GB-6: the taught error is gone — the terrain solve engages
# ================================================================
@pytest.mark.parametrize("eps", [0.0, 1.0])
def test_terrain_implicit_solve_engages(eps):
    grid = terrain_grid(16, a=0.4)
    m = model(grid, eps=eps)
    fs = m.module(hy.ImplicitFreeSurface)
    assert fs._column == ("zp", "z")
    assert "ps" in m.state.component_names
    rng = np.random.default_rng(0)
    fields = {k: 0.2 * rng.standard_normal(m.state[k].shape)
              for k in ("u", "v")}
    if eps > 0:
        fields["ps"] = 0.2 * rng.standard_normal(m.state["ps"].shape)
    m.set_fields(**fields)
    m.advance(6)
    assert not m.panicked
    assert bool(jnp.isfinite(m.state["ps"].data).all())


def test_terrain_extra_halo_is_two_cells():
    terrain = model(terrain_grid(8), eps=1.0).module(hy.ImplicitFreeSurface)
    flat = model(plain_grid(8), eps=1.0).module(hy.ImplicitFreeSurface)
    assert terrain._column is not None
    assert dict(terrain.extra_halo.widths) == {"x": 2, "y": 2}
    assert flat._column is None
    assert dict(flat.extra_halo.widths) == {"x": 1, "y": 1}


# ================================================================
#  GB-1: exact cancellation of the RAW transport divergence (eps=0)
# ================================================================
@pytest.mark.parametrize("a", [0.4, 0.8])
def test_rigid_lid_transport_divergence_cancels(a):
    grid = terrain_grid(16, a=a)
    m = model(grid, eps=0.0, csqr=3.0, iterations=50, tolerance=None)
    rng = np.random.default_rng(1)
    m.set_fields(u=rng.standard_normal(m.state["u"].shape),
                 v=rng.standard_normal(m.state["v"].shape))
    fs = m.module(hy.ImplicitFreeSurface)
    pre = float(jnp.abs(fs._terrain_transport_div(m.state)[0].data).max())
    out = fs._barotropic_solve(m.state, ctx_of(3.0, 0.05))
    post_state = m.state.replace(u=out["u"], v=out["v"])
    post = float(
        jnp.abs(fs._terrain_transport_div(post_state)[0].data).max())
    assert pre > 1.0                       # a genuine divergence
    assert post <= 1e-13 * pre             # cancelled to machine zero


# ================================================================
#  GB-2: flat-chart (a=0) matches the shipped flat spectral solve
# ================================================================
@pytest.mark.parametrize("eps", [0.0, 1.0])
def test_flat_chart_matches_the_flat_spectral_solve(eps):
    # a=0 sigma chart: J == 1 everywhere, so discover_column still fires
    # (the terrain path runs) but the operator equals the flat one
    chart = model(terrain_grid(16, a=0.0), eps=eps, csqr=3.0)
    plain = model(plain_grid(16), eps=eps, csqr=3.0)
    fc = chart.module(hy.ImplicitFreeSurface)
    fp = plain.module(hy.ImplicitFreeSurface)
    assert fc._column is not None
    assert fp._column is None
    rng = np.random.default_rng(2)
    ic = {k: rng.standard_normal(chart.state[k].shape)
          for k in ("u", "v")}
    if eps > 0:
        ic["ps"] = rng.standard_normal(chart.state["ps"].shape)
    chart.set_fields(**ic)
    plain.set_fields(**ic)
    ctx = ctx_of(3.0, 0.05)
    psc = np.asarray(fc._barotropic_solve(chart.state, ctx)["ps"].data)
    psp = np.asarray(fp._barotropic_solve(plain.state, ctx)["ps"].data)
    psc = psc - psc.mean()                 # both mean-free (the gauge)
    psp = psp - psp.mean()
    rel = np.abs(psc - psp).max() / max(np.abs(psp).max(), 1e-30)
    assert rel < 1e-11


# ================================================================
#  GB-3: barotropic volume conservation (eps=1, multi-step run)
# ================================================================
def test_barotropic_volume_is_conserved():
    grid = terrain_grid(16, a=0.4)
    m = model(grid, eps=1.0, csqr=3.0, iterations=40, tolerance=None)
    rng = np.random.default_rng(7)
    m.set_fields(u=0.2 * rng.standard_normal(m.state["u"].shape),
                 v=0.2 * rng.standard_normal(m.state["v"].shape),
                 ps=0.2 * rng.standard_normal(m.state["ps"].shape))

    def volume():
        return float(jnp.sum(m.state["ps"].integrate().data))

    v0 = volume()
    scale = float(jnp.abs(m.state["ps"].integrate().data).max()) + 1.0
    m.advance(12)
    assert not m.panicked
    # backward Euler dissipates the amplitude but conserves int ps to
    # round-off (option 1, the constant mode is exactly resolved)
    assert abs(volume() - v0) <= 1e-12 * scale


# ================================================================
#  GB-5: reverse-mode autodiff through a short terrain-implicit run
# ================================================================
def test_grad_through_terrain_implicit_run_matches_fd():
    m = hy.Model(
        grid=terrain_grid(8, nz=4, a=0.4), dt=0.01, csqr=1.0,
        stratification=hy.ConstantStratification(n2=0.0),
        coriolis=hy.FPlaneCoriolis(f0=0.5), advection=False,
        free_surface=hy.ImplicitFreeSurface(
            epsilon=1.0, pressure_iterations=20),
        time_stepper=fr.model.time_steppers.AdamBashforth(0.01, order=2))
    rng = np.random.default_rng(11)
    m.set_fields(**{k: 0.1 * rng.standard_normal(m.state[k].data.shape)
                    for k in ("u", "v", "ps")})
    record, carry, stepper = m._artifacts.record, m._carry, m._stepper
    ps_leaf = carry.state["ps"].storage
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is ps_leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        spliced = jax.tree_util.tree_unflatten(treedef, new)
        final = _chunk_body(record, 5, spliced, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    grad = np.asarray(jax.grad(loss)(ps_leaf))
    # a masked 1/H 0/0 would NaN every entry with a data path; option 1
    # carries no such division, so the terrain gradient is clean
    assert bool(np.all(np.isfinite(grad)))
    rng2 = np.random.default_rng(5)
    direction = jnp.asarray(rng2.standard_normal(ps_leaf.shape),
                            dtype=ps_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(ps_leaf + eps * direction))
          - float(loss(ps_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


# ================================================================
#  Phase C: the multigrid-preconditioned terrain solve (GC-1..GC-3)
# ================================================================
@pytest.mark.parametrize("eps", [0.0, 1.0])
def test_terrain_multigrid_model_assembles_and_steps_finite(eps):
    grid = terrain_grid(16, a=0.8)
    m = model(grid, eps=eps, preconditioner="multigrid")
    fs = m.module(hy.ImplicitFreeSurface)
    assert fs.pressure_preconditioner == "multigrid"
    assert fs._column == ("zp", "z")
    rng = np.random.default_rng(50)
    fields = {k: 0.2 * rng.standard_normal(m.state[k].shape)
              for k in ("u", "v")}
    if eps > 0:
        fields["ps"] = 0.2 * rng.standard_normal(m.state["ps"].shape)
    m.set_fields(**fields)
    m.advance(6)
    assert not m.panicked
    assert bool(jnp.isfinite(m.state["ps"].data).all())


def test_terrain_multigrid_matches_spectral_solution():
    # the two preconditioners are two routes to the same solve, so a
    # single CONSTRAINT-stage call agrees to the CG tolerance
    grid = terrain_grid(32, a=0.8)
    ms = model(grid, eps=1.0, iterations=60, tolerance=1e-8)
    mm = model(grid, eps=1.0, iterations=60, tolerance=1e-8,
               preconditioner="multigrid")
    rng = np.random.default_rng(51)
    ic = {k: rng.standard_normal(ms.state[k].shape)
          for k in ("u", "v", "ps")}
    ms.set_fields(**ic)
    mm.set_fields(**ic)
    ctx = ctx_of(3.0, 0.05)
    ps_s = np.asarray(
        ms.module(hy.ImplicitFreeSurface)._barotropic_solve(
            ms.state, ctx)["ps"].data)
    ps_m = np.asarray(
        mm.module(hy.ImplicitFreeSurface)._barotropic_solve(
            mm.state, ctx)["ps"].data)
    ps_s = ps_s - ps_s.mean()
    ps_m = ps_m - ps_m.mean()
    rel = np.abs(ps_s - ps_m).max() / max(np.abs(ps_s).max(), 1e-30)
    assert rel <= 1e-6


# ---- GC-3: forced-4-device parity of the multigrid terrain solve --
def _mg_depth(x, y):
    """Steep separable terrain depth H(x, y) for the forced-4 gate."""
    return 1.0 + 0.8 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def _terrain_mg_solve(nx, device_ids):
    """Run the mg terrain barotropic solve on the given devices."""
    grid = fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(8, (-1.0, 0.0), periodic=False, name="z")),
        mapping=CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                  params={"H": _mg_depth}),
        device_ids=device_ids)
    m = model(grid, eps=1.0, iterations=12, tolerance=1e-8,
              preconditioner="multigrid")
    rng = np.random.default_rng(52)
    m.set_fields(**{k: rng.standard_normal(m.state[k].shape)
                    for k in ("u", "v", "ps")})
    fs = m.module(hy.ImplicitFreeSurface)
    out = fs._barotropic_solve(m.state, ctx_of(3.0, 0.05))
    return np.asarray(out["ps"].data)


@pytest.mark.multi_device
@pytest.mark.parametrize(
    "nx", [pytest.param(16, id="aligned-x16"),
           pytest.param(12, id="replicated-x12-coarse6")])
def test_forced4_terrain_multigrid_matches_single_device(nx):
    # MG-D5: the tiny 2-D coarse levels (x=12 coarsens 12 -> 6, and 6
    # does not divide four devices, so the coarse level lives replicated
    # on the same mesh) must produce the single-device result bitwise-
    # close through every level's smoother, transfer and inner product
    ids = tuple(range(jax.device_count()))
    one = _terrain_mg_solve(nx, (0,))
    many = _terrain_mg_solve(nx, ids)
    scale = np.max(np.abs(one))
    assert np.max(np.abs(one - many)) / scale < 1e-8
