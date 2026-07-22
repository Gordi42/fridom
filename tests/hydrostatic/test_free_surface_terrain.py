r"""The free surface on a terrain-following (sigma) grid.

Depth fix (research record ``stretched_terrain_combined.md`` §6): the
physical column depth ``H(x, y) = \int J\,dz`` (not the computational
extent) sets the depth-mean divisor and the barotropic energy weight,
and the depth-mean divergence is the flux-form transport divergence
``\int[\partial_x(Ju) + \partial_y(Jv)]\,dz / H`` — the exact adjoint
(under the physical-volume inner product) of the ``-\nabla_h ps``
momentum force, so the barotropic gravity pair conserves energy to
roundoff. The variable-coefficient **implicit** free surface (H3, the
volume-exact solve) and the transport-depth-consistent **split**
subcycle both now engage on a chart (their gates live in
``test_free_surface_terrain_implicit.py`` /
``test_free_surface_terrain_split.py``). Self-contained builders (AGENTS
oversized-module rule).
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
from fridom.spatial.operators.integrate import Integral

IM = fr.spatial.meshes.IntervalMesh
N2, CSQR = 2.0, 1.0


def _depth(x, y):
    return 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def _terrain_grid(n, nz=None):
    nz = nz or n
    return fr.spatial.Grid((
        IM(n, (0.0, 1.0), periodic=True, name="x"),
        IM(n, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (-1.0, 0.0), periodic=False, name="z")),
        mapping=CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                  params={"H": _depth}))


def _a0_chart(n, nz=None):
    """Return a sigma chart with CONSTANT depth H == 1 (J == 1)."""
    nz = nz or n
    return fr.spatial.Grid((
        IM(n, (0.0, 1.0), periodic=True, name="x"),
        IM(n, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (-1.0, 0.0), periodic=False, name="z")),
        mapping=CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                  params={"H": lambda x, y: 1.0 + 0.0 * x
                                          + 0.0 * y}))


def _flat_grid(n, nz=None):
    """Return the plain (no-mapping) flat twin of :func:`_terrain_grid`."""
    nz = nz or n
    return fr.spatial.Grid((
        IM(n, (0.0, 1.0), periodic=True, name="x"),
        IM(n, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (-1.0, 0.0), periodic=False, name="z")))


def _model(grid, *, free_surface=None, coriolis=None, dt=2e-3,
           stepper=None):
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=CSQR),
        time_stepper=stepper or AdamBashforth(dt, order=3),
        coriolis=coriolis,
        stratification=hy.ConstantStratification(n2=N2),
        free_surface=free_surface or hy.ExplicitFreeSurface(),
        advection=False)


def _bound_fs(model):
    """Return the assembled model's (already bound) free surface."""
    return model.module(hy.ExplicitFreeSurface)


# ================================================================
#  Physical column depth H(x, y) = int J dz (not the extent)
# ================================================================
def test_physical_depth_matches_the_analytic_depth():
    grid = _terrain_grid(16)
    fs = _bound_fs(_model(grid))
    coll = fr.spatial.Collocated().resolve(grid)
    one = grid.create_field(coll, data=jnp.ones(coll.shape))
    depth_field = fs._physical_depth(one)
    xs = grid.evaluation_nodes(depth_field.function_space, "x").data
    ys = grid.evaluation_nodes(depth_field.function_space, "y").data
    # the column extent is 1, so H(x, y) = int_{-1}^0 H dz = H(x, y)
    assert float(jnp.abs(depth_field.data - _depth(xs, ys)).max()) < 1e-13
    # ... and it is genuinely different from the computational extent 1
    assert float(jnp.abs(depth_field.data - 1.0).max()) > 0.1


def test_physical_depth_equals_the_jacobian_integral_seam():
    # the free surface's in-trace depth agrees with the wired
    # Integral(jacobian=) seam (the canonical physical column extent).
    # Build the model first: it re-negotiates the grid to the
    # hydrostatic core's extra_halo, so ``one`` must be created on the
    # frozen (final-width) grid, not the provisionally-narrower base.
    grid = _terrain_grid(12)
    fs = _bound_fs(_model(grid))
    coll = fr.spatial.Collocated().resolve(grid)
    one = grid.create_field(coll, data=jnp.ones(coll.shape))
    seam = Integral(jacobian=("zp",))["z"](one)
    mine = fs._physical_depth(one)
    assert np.allclose(np.asarray(mine.data), np.asarray(seam.data),
                       atol=1e-13)


# ================================================================
#  H4 (barotropic leg): the surface-pressure <-> depth-mean pair
#  conserves energy to roundoff under the constant-weight metric
# ================================================================
def test_barotropic_energy_is_conserved_to_roundoff():
    # perturb (u, v, ps), no buoyancy, no rotation: the isolated
    # barotropic gravity pair. The volume-exact terrain form (GM-D1
    # option 1) scales the raw transport divergence T* by the CONSTANT
    # gravity g = c^2/H_ref, so the conserved barotropic energy is the
    # physical (J-weighted) kinetic energy plus the CONSTANT-weight
    # surface energy (H_ref/2c^2) int ps^2 dA on the plain 2D area --
    # NOT the J-weighted H(x, y)/c^2 weight of the retired energy-form
    # gravity. Under that metric the skew <X, M dX/dt> vanishes to
    # machine precision: the flux-form transport divergence is the exact
    # plain-measure adjoint of the -grad ps momentum force.
    grid = _terrain_grid(16)
    model = _model(grid)
    fs = _bound_fs(model)
    h_ref = 1.0 / fs._inv_depth
    rng = np.random.default_rng(3)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        b=np.zeros(model.state["b"].shape),
        ps=rng.standard_normal(model.state["ps"].shape))
    st = model.state
    dX = model.tendency(st)

    def jint(f):
        # KE uses the physical (Jacobian-weighted) volume integral --
        # the plain seeded verb on a maps= terrain grid (the
        # physical-integral-default flip): f.integrate() carries the
        # column Jacobian.
        return float(f.integrate().data.ravel()[0])
    # the surface-energy weight is the CONSTANT 1/g = H_ref/c^2 on the
    # plain 2D area: ps carries no z factor, so its .integrate() is the
    # unmapped horizontal measure (no column Jacobian).
    ps_term = h_ref * float(
        ((st["ps"] / CSQR) * dX["ps"]).integrate().data.ravel()[0])
    terms = [jint(st["u"] * dX["u"]), jint(st["v"] * dX["v"]), ps_term]
    skew = sum(terms)
    scale = sum(abs(t) for t in terms)
    assert abs(skew) < 1e-12 * scale


# ================================================================
#  The explicit free surface runs finite and stays bounded
# ================================================================
def test_explicit_free_surface_runs_finite_and_bounded():
    grid = _terrain_grid(16)
    model = _model(grid, coriolis=hy.FPlaneCoriolis(f0=1.0))
    rng = np.random.default_rng(1)
    model.set_fields(
        u=0.1 * rng.standard_normal(model.state["u"].shape),
        v=0.1 * rng.standard_normal(model.state["v"].shape),
        b=0.1 * rng.standard_normal(model.state["b"].shape),
        ps=0.1 * rng.standard_normal(model.state["ps"].shape))
    ps0 = float(jnp.abs(model.state["ps"].data).max())
    model.run(40, progress=False)
    assert bool(jnp.isfinite(model.state["ps"].data).all())
    # bounded (a barotropic gravity wave oscillates, does not blow up)
    assert float(jnp.abs(model.state["ps"].data).max()) < 10.0 * ps0 + 1.0


def test_flat_depth_mean_is_byte_identical():
    # off a terrain grid the depth mean keeps the flat scalar 1/H path
    grid = fr.spatial.Grid((
        IM(8, (0.0, 1.0), periodic=True, name="x"),
        IM(8, (0.0, 1.0), periodic=True, name="y"),
        IM(6, (-1.0, 0.0), periodic=False, name="z")))
    fs = _bound_fs(_model(grid))
    assert fs._column is None
    assert fs._inv_depth == pytest.approx(1.0)


# ================================================================
#  The implicit free surface now engages on a terrain grid (H3; the
#  volume-exact solve, GM-D1/D2) — the taught error is gone. The
#  gates live in test_free_surface_terrain_implicit.py.
# ================================================================
def test_implicit_free_surface_engages_on_terrain():
    grid = _terrain_grid(8)
    model = _model(grid, free_surface=hy.ImplicitFreeSurface())
    assert "ps" in model.state.component_names
    fs = model.module(hy.ImplicitFreeSurface)
    assert fs._column == ("zp", "z")
    rng = np.random.default_rng(0)
    model.set_fields(
        u=0.1 * rng.standard_normal(model.state["u"].shape),
        v=0.1 * rng.standard_normal(model.state["v"].shape),
        ps=0.1 * rng.standard_normal(model.state["ps"].shape))
    model.advance(4)
    assert not model.panicked
    assert bool(jnp.isfinite(model.state["ps"].data).all())
    # (the volume-exact explicit gates follow the split engage test)


def test_split_free_surface_engages_on_terrain():
    # the split-explicit taught error is retired (H3): the volume-exact
    # terrain subcycle (GM-D1 option 1) now engages on a chart grid. The
    # gates live in test_free_surface_terrain_split.py.
    grid = _terrain_grid(8)
    model = _model(grid,
                   free_surface=hy.SplitExplicitFreeSurface(substeps=4),
                   stepper=AdamBashforth(2e-3, order=2))
    assert {"ps", "U", "V"} <= set(model.state.component_names)
    fs = model.module(hy.SplitExplicitFreeSurface)
    assert fs._column == ("zp", "z")
    rng = np.random.default_rng(0)
    model.set_fields(
        u=0.1 * rng.standard_normal(model.state["u"].shape),
        v=0.1 * rng.standard_normal(model.state["v"].shape),
        ps=0.1 * rng.standard_normal(model.state["ps"].shape))
    model.advance(4)
    assert not model.panicked
    assert bool(jnp.isfinite(model.state["ps"].data).all())


# ================================================================
#  Volume-exact explicit terrain gravity (GM-D1 option 1, owner
#  ruling 2026-07-19): the explicit variant adopts the volume-exact
#  form so all three free-surface variants share one discrete
#  barotropic physics (g = c^2/H_ref, no 1/H(x, y) division).
# ================================================================
def test_flat_sigma_chart_matches_the_flat_grid_gravity():
    # gate (i): on a flat (a = 0, H == H_ref) sigma chart the
    # volume-exact terrain gravity coincides with the plain flat-grid
    # depth-mean divergence (the forms agree at constant depth), so the
    # barotropic ps tendency is identical to round-off.
    chart = _model(_a0_chart(16))
    plain = _model(_flat_grid(16))
    fc, fp = _bound_fs(chart), _bound_fs(plain)
    assert fc._column is not None                 # the chart path runs
    assert fp._column is None                     # the flat path runs
    rng = np.random.default_rng(4)
    ic = {k: rng.standard_normal(chart.state[k].shape)
          for k in ("u", "v", "ps")}
    chart.set_fields(**ic)
    plain.set_fields(**ic)
    dc = np.asarray(chart.tendency(chart.state)["ps"].data)
    dp = np.asarray(plain.tendency(plain.state)["ps"].data)
    assert np.abs(dc).max() > 1.0                 # a genuine divergence
    assert np.abs(dc - dp).max() <= 1e-13 * (np.abs(dp).max() + 1.0)


def test_terrain_rest_state_is_preserved():
    # gate (ii): a barotropic rest state (u = v = 0, ps = 0, b = 0) has
    # an exactly-zero barotropic tendency and stays at rest to machine
    # precision through a multi-step run -- the volume-exact gravity
    # generates no spurious motion on a chart.
    grid = _terrain_grid(16)
    model = _model(grid, coriolis=hy.FPlaneCoriolis(f0=1.0))
    model.set_fields(
        u=np.zeros(model.state["u"].shape),
        v=np.zeros(model.state["v"].shape),
        b=np.zeros(model.state["b"].shape),
        ps=np.zeros(model.state["ps"].shape))
    dX = model.tendency(model.state)
    for name in ("ps", "u", "v"):
        assert float(jnp.abs(dX[name].data).max()) < 1e-14
    model.run(20, progress=False)
    assert not model.panicked
    for name in ("ps", "u", "v"):
        assert float(jnp.abs(model.state[name].data).max()) < 1e-13


def test_explicit_barotropic_volume_is_conserved():
    # gate (iii): plain int(ps) (the barotropic volume) is conserved to
    # round-off on a terrain run -- the point of the volume-exact form.
    # The retired energy-form gravity (c^2 T*/H(x, y)) drifts O(slope).
    grid = _terrain_grid(16)
    model = _model(grid, coriolis=hy.FPlaneCoriolis(f0=1.0))
    rng = np.random.default_rng(7)
    model.set_fields(
        u=0.2 * rng.standard_normal(model.state["u"].shape),
        v=0.2 * rng.standard_normal(model.state["v"].shape),
        b=0.2 * rng.standard_normal(model.state["b"].shape),
        ps=0.2 * rng.standard_normal(model.state["ps"].shape))

    def volume():
        return float(jnp.sum(model.state["ps"].integrate().data))

    v0 = volume()
    scale = float(jnp.abs(model.state["ps"].integrate().data).max()) + 1.0
    model.run(30, progress=False)
    assert not model.panicked
    assert abs(volume() - v0) <= 1e-11 * scale


def test_explicit_and_implicit_terrain_track_at_small_dt():
    # gate (iv): explicit and implicit now share the identical discrete
    # barotropic physics (only the integrator differs). At small dt a
    # few steps from a matched IC stay close.
    dt = 5e-4
    exp = _model(_terrain_grid(16), dt=dt)
    imp = _model(_terrain_grid(16), dt=dt,
                 free_surface=hy.ImplicitFreeSurface(
                     epsilon=1.0, pressure_iterations=40,
                     pressure_tolerance=None),
                 stepper=AdamBashforth(dt, order=2))
    rng = np.random.default_rng(2)
    ic = {k: 0.1 * rng.standard_normal(exp.state[k].shape)
          for k in ("u", "v", "ps")}
    exp.set_fields(**ic)
    imp.set_fields(**ic)
    exp.run(8, progress=False)
    imp.run(8, progress=False)
    pe = np.asarray(exp.state["ps"].data)
    pi = np.asarray(imp.state["ps"].data)
    rel = np.abs(pe - pi).max() / (np.abs(pe).max() + 1e-30)
    assert rel < 0.05


def test_grad_through_terrain_explicit_run_matches_fd():
    # gate (v): reverse-mode autodiff through a short explicit terrain
    # run is finite (the volume-exact form carries no 1/H division, so no
    # guarded-division NaN hazard) and matches a central finite
    # difference to rtol 1e-4 (the differentiability policy regression).
    m = hy.Model(
        grid=_terrain_grid(8, nz=4),
        core=hy.Core(gravity=CSQR),
        time_stepper=AdamBashforth(0.01, order=3),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        stratification=hy.ConstantStratification(n2=0.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)
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
    assert bool(np.all(np.isfinite(grad)))
    rng2 = np.random.default_rng(5)
    direction = jnp.asarray(rng2.standard_normal(ps_leaf.shape),
                            dtype=ps_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(ps_leaf + eps * direction))
          - float(loss(ps_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)
