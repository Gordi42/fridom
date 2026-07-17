"""Immersed (cut-cell) free-surface solves (IP-D9, gates c/d).

Prefix-mirrored shard of ``hy.modules.free_surface`` covering the
immersed barotropic paths: the masked depth-mean divergence, the
variable-coefficient implicit PCG (``_solve_immersed``), the rigid-lid
wet-column projection, and the split-explicit per-column-depth subcycle
and mass conservation. The unimmersed variants live in
``test_free_surface{,_implicit,_split}.py``. Self-contained per the
AGENTS oversized-module rule.
"""
import inspect

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.hydrostatic as hy
from fridom.hydrostatic.modules import free_surface
from fridom.hydrostatic.params import CSQR
from fridom.model.context import StepContext
from fridom.model.modules.advection import CenteredAdvection
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.spaces.constant import ConstantSpace

IM = IntervalMesh
FLAT_BOTTOM = 0.5  # a face-aligned immersed bottom (wet = top half)


def _meshes(n, nz, depth):
    return (IM(n, (0.0, 1.0), periodic=True, name="x"),
            IM(n, (0.0, 1.0), periodic=True, name="y"),
            IM(nz, (0.0, depth), periodic=False, name="z"))


def immersed_grid(n=8, nz=8, init=None, order=None, min_fraction=0.0):
    """Return a doubly-periodic horizontal, bounded-z immersed grid."""
    if init is None:
        init = lambda x, y, z: (z > FLAT_BOTTOM).astype(float)  # noqa: E731,ARG005
    return Grid(_meshes(n, nz, 1.0),
                immersed=ImmersedDomain(init, order=order,
                                        min_fraction=min_fraction))


def plain_grid(n=8, nz=8, depth=1.0):
    """Return the unimmersed twin."""
    return Grid(_meshes(n, nz, depth))


def ps_mode(model, kx):
    """Return a depth-uniform cos(kx x) at the ps nodes."""
    fs = model.state["ps"].function_space
    var = tuple(nm for f in fs.bare.factors
                if not isinstance(f, ConstantSpace) for nm in f.names)

    def init(**c):
        return np.cos(2 * np.pi * kx * c["x"]) + 0.0 * sum(c.values())
    init.__signature__ = inspect.Signature(
        [inspect.Parameter(nm, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for nm in var])
    return model.grid.create_field(fs, init=init)


def build(grid, free_surface, *, csqr=4.0, n2=0.0, f0=0.0, dt=0.01):
    return hy.Model(
        grid=grid, dt=dt, csqr=csqr, free_surface=free_surface,
        stratification=hy.ConstantStratification(n2=n2),
        coriolis=hy.FPlaneCoriolis(f0=f0), advection=False)


# ================================================================
#  pressure_iterations knob (mirrors nh.Model)
# ================================================================
def test_pressure_iterations_default_and_property():
    assert hy.ImplicitFreeSurface().pressure_iterations == 30
    assert hy.ImplicitFreeSurface(
        pressure_iterations=12).pressure_iterations == 12


@pytest.mark.parametrize(
    "value",
    [pytest.param(0, id="zero"), pytest.param(-3, id="negative"),
     pytest.param(True, id="bool"), pytest.param(2.0, id="float")],
)
def test_pressure_iterations_validation_rejects(value):
    with pytest.raises(ValueError, match="pressure_iterations"):
        hy.ImplicitFreeSurface(pressure_iterations=value)


# ================================================================
#  pressure_tolerance knob (mirrors nh.Model; opt-in CG break)
# ================================================================
def test_pressure_tolerance_default_and_property():
    assert hy.ImplicitFreeSurface().pressure_tolerance == 1e-8
    assert hy.ImplicitFreeSurface(
        pressure_tolerance=None).pressure_tolerance is None
    assert hy.ImplicitFreeSurface(
        pressure_tolerance=1e-6).pressure_tolerance == 1e-6


def test_pressure_tolerance_reaches_the_conjugate_gradient(monkeypatch):
    # the immersed barotropic solve builds its CG inside
    # ``_solve_immersed``; capture the tolerance kwarg it forwards
    captured = {}
    real = free_surface.ConjugateGradient

    def spy(*args, **kwargs):
        captured["tolerance"] = kwargs.get("tolerance")
        return real(*args, **kwargs)

    monkeypatch.setattr(free_surface, "ConjugateGradient", spy)
    model = build(immersed_grid(),
                  hy.ImplicitFreeSurface(pressure_tolerance=1e-7))
    rng = np.random.default_rng(3)
    model.set_fields(**{
        k: 0.1 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "b", "ps")})
    model.advance(1)
    assert captured["tolerance"] == 1e-7


# ================================================================
#  Depth handling: wet transport depth and wet-depth mean
# ================================================================
def test_transport_depth_is_the_wet_column_depth():
    model = build(immersed_grid(), hy.ImplicitFreeSurface())
    fs = model.module(hy.ImplicitFreeSurface)
    depth = fs._transport_depth(model.state["u"])
    # wet = top 4 of 8 cells over a unit domain -> H = 0.5 everywhere
    assert np.allclose(np.asarray(depth.data), 0.5, atol=1e-14)


def test_guarded_inverse_zeros_a_land_column():
    model = build(immersed_grid(), hy.ImplicitFreeSurface())
    fs = model.module(hy.ImplicitFreeSurface)
    depth = fs._transport_depth(model.state["u"])
    land = depth.with_data(depth.data * 0.0)
    inv = fs._guarded_inverse(land)
    assert float(np.abs(np.asarray(inv.data)).max()) == 0.0


# ================================================================
#  Gate c: implicit all-wet immersed == unimmersed (byte-comparable)
# ================================================================
@pytest.mark.parametrize("iterations", [1, 2])
def test_implicit_all_wet_matches_unimmersed(iterations):
    im = build(
        Grid(_meshes(8, 4, 1.0),
             immersed=ImmersedDomain(lambda x, y, z: x * 0.0 + 1.0)),  # noqa: ARG005
        hy.ImplicitFreeSurface(pressure_iterations=iterations),
        csqr=4.0, f0=0.5, dt=0.02)
    un = build(plain_grid(8, 4), hy.ImplicitFreeSurface(),
               csqr=4.0, f0=0.5, dt=0.02)
    rng = np.random.default_rng(7)
    ic = {k: 0.3 * rng.standard_normal(im.state[k].data.shape)
          for k in ("u", "v", "b", "ps")}
    im.set_fields(**ic)
    un.set_fields(**ic)
    im.advance(8)
    un.advance(8)
    for k in ("u", "v", "b", "ps"):
        diff = np.abs(np.asarray(im.state[k].data)
                      - np.asarray(un.state[k].data)).max()
        assert diff < 1e-12, (k, iterations, diff)


# ================================================================
#  Gate c: small-dt convergence of the masked BE solve to the oracle
# ================================================================
def test_immersed_small_dt_converges_to_the_explicit_oracle():
    total = 0.5
    errs = []
    for n_steps in (20, 40, 80):
        dt = total / n_steps
        exp = build(immersed_grid(16, 8), hy.ExplicitFreeSurface(),
                    csqr=1.0, dt=dt)
        imp = build(immersed_grid(16, 8),
                    hy.ImplicitFreeSurface(pressure_iterations=30),
                    csqr=1.0, dt=dt)
        for m in (exp, imp):
            m.set_fields(ps=ps_mode(m, 1).data)
            m.run(steps=n_steps)
        d = (np.asarray(exp.state["ps"].data)
             - np.asarray(imp.state["ps"].data))
        errs.append(float(np.sqrt((d * d).sum())))
    errs = np.asarray(errs)
    slopes = np.log2(errs[:-1] / errs[1:])
    assert errs[-1] < errs[0]
    assert slopes[-1] > 0.9   # backward Euler is first order


# ================================================================
#  Gate c: rigid lid (eps=0) projects the masked depth mean free
# ================================================================
def test_immersed_rigid_lid_projects_masked_depth_mean_divergence_free():
    grid = immersed_grid(16, 8)
    model = build(grid, hy.ImplicitFreeSurface(
        epsilon=0.0, pressure_iterations=40), csqr=3.0, f0=0.5, dt=0.05)
    rng = np.random.default_rng(1)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape))
    fs = model.module(hy.ImplicitFreeSurface)
    ctx = StepContext(params={CSQR: jnp.asarray(3.0)},
                      clock=jnp.asarray(0.0), dt=jnp.asarray(0.05),
                      stage_dt=jnp.asarray(0.05))
    pre = float(np.max(np.abs(np.asarray(
        fs._depth_mean_div(model.state).data))))
    out = fs._barotropic_solve(model.state, ctx)
    post_state = model.state.replace(u=out["u"], v=out["v"])
    post = float(np.max(np.abs(np.asarray(
        fs._depth_mean_div(post_state).data))))
    uscale = float(np.max(np.abs(np.asarray(out["u"].data))))
    assert pre > 1.0                       # a genuine divergence
    assert post / uscale < 1e-11           # projected to machine zero


def test_immersed_rigid_lid_ps_masked_to_wet_columns():
    # a land column (full-depth wall on x<0.25) -> ps == 0 there
    def coast(x, y, z):  # noqa: ARG001
        return (x > 0.25).astype(float)
    grid = Grid(_meshes(8, 4, 1.0), immersed=ImmersedDomain(coast))
    model = build(grid, hy.ImplicitFreeSurface(
        epsilon=0.0, pressure_iterations=30), csqr=2.0, f0=0.5)
    rng = np.random.default_rng(2)
    model.set_fields(u=rng.standard_normal(model.state["u"].shape),
                     v=rng.standard_normal(model.state["v"].shape))
    fs = model.module(hy.ImplicitFreeSurface)
    ctx = StepContext(params={CSQR: jnp.asarray(2.0)},
                      clock=jnp.asarray(0.0), dt=jnp.asarray(0.01),
                      stage_dt=jnp.asarray(0.01))
    ps = np.asarray(fs._barotropic_solve(
        model.state, ctx)["ps"].data)
    # theta-column indicator on the ps cell: dry columns are x<0.25
    theta = np.asarray(grid.immersed.fraction(
        model.state["b"].function_space).data)
    dry_col = theta.sum(axis=2) == 0.0
    assert float(np.abs(ps.reshape(theta.shape[:2])[dry_col]).max()) == 0.0


# ================================================================
#  Gate c: eps=1 stays bounded far beyond the explicit CFL
# ================================================================
def test_immersed_implicit_stable_beyond_explicit_cfl():
    csqr = 100.0
    dx = 1.0 / 16
    dt = 40.0 * dx / np.sqrt(csqr)
    model = build(immersed_grid(16, 6),
                  hy.ImplicitFreeSurface(pressure_iterations=30),
                  csqr=csqr, f0=0.5, dt=dt)
    rng = np.random.default_rng(3)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        ps=rng.standard_normal(model.state["ps"].shape))
    model.advance(60)
    assert not model.panicked
    umax = float(np.max(np.abs(np.asarray(model.state["u"].data))))
    assert np.isfinite(umax)


# ================================================================
#  _depth_mean_div: masked all-wet == unimmersed (byte-identical)
# ================================================================
def test_masked_depth_mean_div_all_wet_matches_unimmersed():
    im = build(Grid(_meshes(8, 4, 1.0), immersed=ImmersedDomain(
        lambda x, y, z: x * 0.0 + 1.0)),  # noqa: ARG005
        hy.ExplicitFreeSurface())
    un = build(plain_grid(8, 4), hy.ExplicitFreeSurface())
    rng = np.random.default_rng(5)
    ic = {k: rng.standard_normal(im.state[k].data.shape)
          for k in ("u", "v")}
    im.set_fields(**ic)
    un.set_fields(**ic)
    fim = im.module(hy.ExplicitFreeSurface)
    fun = un.module(hy.ExplicitFreeSurface)
    dim = np.asarray(fim._depth_mean_div(im.state).data)
    dun = np.asarray(fun._depth_mean_div(un.state).data)
    assert np.abs(dim - dun).max() == 0.0


# ================================================================
#  Gate d: split-explicit mass conservation on masked columns
# ================================================================
def test_split_theta_mass_conserved_to_machine_zero():
    grid = immersed_grid(8, 8)
    # surface_flux=False: the default constancy-preserving closure
    # advects through the surface face and exchanges tracer content with
    # the moving free surface, so exact theta-mass conservation is the
    # legacy fixed-domain closure's property.
    model = hy.Model(
        grid=grid, dt=0.005, csqr=4.0,
        free_surface=hy.SplitExplicitFreeSurface(substeps=8),
        stratification=hy.ConstantStratification(n2=0.0),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        advection=CenteredAdvection(surface_flux=False))
    rng = np.random.default_rng(0)
    model.set_fields(**{k: 0.2 * rng.standard_normal(
        model.state[k].data.shape) for k in ("u", "v", "b")})
    theta = grid.immersed.fraction(model.state["b"].function_space)

    def mass():
        return float(jnp.sum((theta * model.state["b"]).integrate().data))

    before = mass()
    model.advance(12)
    assert not model.panicked
    after = mass()
    assert abs(after - before) <= 1e-13 * max(abs(before), 1.0)


def test_split_all_wet_matches_unimmersed():
    def mk(box):
        kw = {} if box is None else {"immersed": ImmersedDomain(box)}
        return Grid(_meshes(8, 4, 1.0), **kw)
    im = hy.Model(
        grid=mk(lambda x, y, z: x * 0.0 + 1.0),  # noqa: ARG005
        dt=0.005, csqr=4.0,
        free_surface=hy.SplitExplicitFreeSurface(substeps=8),
        coriolis=hy.FPlaneCoriolis(f0=0.5), advection=False)
    un = hy.Model(
        grid=mk(None), dt=0.005, csqr=4.0,
        free_surface=hy.SplitExplicitFreeSurface(substeps=8),
        coriolis=hy.FPlaneCoriolis(f0=0.5), advection=False)
    rng = np.random.default_rng(4)
    ic = {k: 0.3 * rng.standard_normal(im.state[k].data.shape)
          for k in ("u", "v", "b", "ps", "U", "V")}
    im.set_fields(**ic)
    un.set_fields(**ic)
    im.advance(10)
    un.advance(10)
    for k in ("u", "v", "b", "ps", "U", "V"):
        diff = np.abs(np.asarray(im.state[k].data)
                      - np.asarray(un.state[k].data)).max()
        assert diff < 1e-13, (k, diff)


def test_split_transport_depth_consistency_no_coast_leak():
    # a full-depth land column (x<0.25): the barotropic transport U on a
    # closed face is exactly 0 (transport-depth consistent), so no mass
    # leaks across the coast and the run stays finite. surface_flux=False
    # isolates the coast (lateral) conservation from the moving-surface
    # exchange the default constancy-preserving closure introduces.
    def coast(x, y, z):  # noqa: ARG001
        return (x > 0.25).astype(float)
    grid = Grid(_meshes(8, 4, 1.0), immersed=ImmersedDomain(coast))
    model = hy.Model(
        grid=grid, dt=0.002, csqr=1.0,
        free_surface=hy.SplitExplicitFreeSurface(substeps=8),
        stratification=hy.ConstantStratification(n2=0.0),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        advection=CenteredAdvection(surface_flux=False))
    rng = np.random.default_rng(6)
    model.set_fields(**{k: 0.1 * rng.standard_normal(
        model.state[k].data.shape) for k in ("u", "v", "b")})
    theta = grid.immersed.fraction(model.state["b"].function_space)
    before = float(jnp.sum((theta * model.state["b"]).integrate().data))
    model.advance(20)
    assert not model.panicked
    after = float(jnp.sum((theta * model.state["b"]).integrate().data))
    assert abs(after - before) <= 1e-12 * max(abs(before), 1.0)
    # the barotropic transport U vanishes on the land column
    fs = model.module(hy.SplitExplicitFreeSurface)
    depth_u = np.asarray(fs._transport_depth(model.state["u"]).data)
    land_face = depth_u == 0.0
    u_land = np.asarray(model.state["U"].data)[land_face]
    assert float(np.abs(u_land).max()) == 0.0
