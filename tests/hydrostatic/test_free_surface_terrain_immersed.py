r"""The free surface on a terrain + immersed (cut-cell) grid (M5).

Prefix-mirrored shard of ``hy.modules.free_surface`` covering the
composed **terrain + immersed** wet-column barotropic solve of
``hy.ImplicitFreeSurface`` and ``hy.ExplicitFreeSurface``: the
wet transport divergence RHS, the wet-column face-depth operator
(``BarotropicPressureSolver``), the GB-1 exact cancellation on the wet
region, column equivalence against a shallower chart, the implicit vs
explicit oracle, all-wet / identity-chart reductions, theta-mass
conservation and reverse-mode autodiff. The single-descriptor terrain
solve lives in ``test_free_surface_terrain_implicit.py``, the flat
immersed solve in ``test_free_surface_immersed.py``. Self-contained per
the AGENTS oversized-module rule.
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.hydrostatic as hy
from fridom.hydrostatic.params import GRAVITY
from fridom.model.context import StepContext
from fridom.model.model import _chunk_body
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.spaces.constant import ConstantSpace

IM = IntervalMesh


# ================================================================
#  Builders (self-contained per the AGENTS oversized-module rule)
# ================================================================
def _mapping(a):
    def depth(x, y):
        return 1.0 + a * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
    return CoordinateMapping(maps={"zp": lambda z, H: z * H},
                             params={"H": depth})


def _cut(x, y, z):  # noqa: ARG001
    return jnp.clip((z - (-0.5 + 0.1 * jnp.sin(2 * jnp.pi * x))) / (1.0 / 8)
                    + 0.5, 0.0, 1.0)


def _flat_bottom(x, y, z):  # noqa: ARG001
    """Return a face-aligned flat bottom: wet top 4 of 8 cells on (-1, 0)."""
    return (z > -0.5).astype(float)


def _staircase(x, y, z):
    """Return a face-aligned staircase bottom (theta in {0, 1}).

    The cut lands exactly on z-cell faces (k*dz on the standard interval
    mesh), so every wet cell is fully wet: the fraction is {0, 1} and the
    wet-centroid offset delta is 0 (to floating point). The bottom steps
    across three face levels in x and y, so the mask composition stays
    genuinely 3-D. Used by the identity-chart equivalence gate, where the
    partial-bottom p_hyd correction must be a no-op on both the flat and
    the chart path (see the test's comment).
    """
    bottom = (-0.5 + 0.125 * (x >= 0.5).astype(float)
              + 0.125 * (y >= 0.5).astype(float))
    return (z > bottom).astype(float)


def _allwet(x, y, z):  # noqa: ARG001
    return x * 0.0 + 1.0


def _grid(*, n=8, nz=8, a=0.4, init=_cut, order=4, min_fraction=0.1,
          device_ids=None):
    kw = {"mapping": _mapping(a),
          "immersed": ImmersedDomain(init, order=order,
                                     min_fraction=min_fraction)}
    if device_ids is not None:
        kw["device_ids"] = device_ids
    return Grid(
        (IM(n, (0.0, 1.0), periodic=True, name="x"),
         IM(n, (0.0, 1.0), periodic=True, name="y"),
         IM(nz, (-1.0, 0.0), periodic=False, name="z")), **kw)


def _pure_terrain(*, n=8, nz=8, a=0.4):
    return Grid(
        (IM(n, (0.0, 1.0), periodic=True, name="x"),
         IM(n, (0.0, 1.0), periodic=True, name="y"),
         IM(nz, (-1.0, 0.0), periodic=False, name="z")),
        mapping=_mapping(a))


def _model(grid, fs, *, csqr=3.0, n2=1.0, f0=0.5, dt=0.02):
    return hy.Model(
        grid=grid,
        core=hy.Core(gravity=csqr),
        time_stepper=AdamBashforth(dt, order=2),
        coriolis=hy.FPlaneCoriolis(f0=f0),
        stratification=hy.ConstantStratification(n2=n2),
        free_surface=fs,
        advection=False)


def _ctx(csqr, dt):
    return StepContext(params={GRAVITY: jnp.asarray(csqr)},
                       clock=jnp.asarray(0.0), dt=jnp.asarray(dt),
                       stage_dt=jnp.asarray(dt))


# ================================================================
#  The taught error is gone: the wet-column terrain solve engages
# ================================================================
@pytest.mark.parametrize("eps", [0.0, 1.0])
def test_terrain_immersed_solve_engages(eps):
    m = _model(_grid(a=0.4), hy.ImplicitFreeSurface(
        epsilon=eps, pressure_iterations=40))
    fs = m.module(hy.ImplicitFreeSurface)
    assert fs._column == ("zp", "z")
    assert fs._immersed is not None
    rng = np.random.default_rng(0)
    fields = {k: 0.2 * rng.standard_normal(m.state[k].shape)
              for k in ("u", "v")}
    if eps > 0:
        fields["ps"] = 0.2 * rng.standard_normal(m.state["ps"].shape)
    m.set_fields(**fields)
    m.advance(6)
    assert not m.panicked
    assert bool(jnp.isfinite(m.state["ps"].data).all())


def test_terrain_immersed_extra_halo_is_two_cells():
    fs = _model(_grid(a=0.4), hy.ImplicitFreeSurface()).module(
        hy.ImplicitFreeSurface)
    assert dict(fs.extra_halo.widths) == {"x": 2, "y": 2}


def test_collocation_mask_on_a_chart_is_a_taught_error():
    with pytest.raises(NotImplementedError,
                       match="genuine per-cell quadrature"):
        _model(_grid(a=0.4, order=None, min_fraction=0.0),
               hy.ImplicitFreeSurface())


# ================================================================
#  GB-1: the wet transport divergence cancels (eps=0) on a cut chart
# ================================================================
@pytest.mark.parametrize(
    "a", [pytest.param(0.4, id="mild"), pytest.param(0.8, id="steep")])
def test_wet_transport_divergence_cancels(a):
    m = _model(_grid(a=a), hy.ImplicitFreeSurface(
        epsilon=0.0, pressure_iterations=60, pressure_tolerance=None),
        dt=0.05)
    rng = np.random.default_rng(1)
    m.set_fields(u=rng.standard_normal(m.state["u"].shape),
                 v=rng.standard_normal(m.state["v"].shape))
    fs = m.module(hy.ImplicitFreeSurface)
    pre = float(jnp.abs(fs._terrain_transport_div(m.state)[0].data).max())
    out = fs._barotropic_solve(m.state, _ctx(3.0, 0.05))
    post_state = m.state.replace(u=out["u"], v=out["v"])
    post = float(
        jnp.abs(fs._terrain_transport_div(post_state)[0].data).max())
    assert pre > 1.0                       # a genuine divergence
    assert post <= 1e-12 * pre             # cancelled to machine zero


# ================================================================
#  Column equivalence: a flat immersed bottom on a (J==1) chart
#  reproduces the shallower unimmersed chart (I3 pattern)
# ================================================================
@pytest.mark.parametrize(
    "make_fs",
    [pytest.param(hy.ExplicitFreeSurface, id="explicit"),
     pytest.param(lambda: hy.ImplicitFreeSurface(
         pressure_iterations=25, pressure_tolerance=None), id="implicit")])
def test_column_equivalence_flat_bottom_on_a_chart(make_fs):
    # a flat immersed bottom (wet top 4 of 8) on a J==1 sigma chart
    # reproduces the shallower unimmersed chart (nz=4); gravity-first,
    # BOTH models carry the SAME physical g (no reference-depth fold).
    g = 4.0
    mi = _model(_grid(a=0.0, init=_flat_bottom, min_fraction=0.0),
                make_fs(), csqr=g, n2=2.0, f0=0.8, dt=0.01)
    short = Grid(
        (IM(8, (0.0, 1.0), periodic=True, name="x"),
         IM(8, (0.0, 1.0), periodic=True, name="y"),
         IM(4, (-0.5, 0.0), periodic=False, name="z")),
        mapping=_mapping(0.0))
    mu = _model(short, make_fs(), csqr=g, n2=2.0, f0=0.8, dt=0.01)
    rng = np.random.default_rng(0)
    icu = {k: 0.2 * rng.standard_normal(mu.state[k].data.shape)
           for k in ("u", "v", "b")}
    if "ps" in set(mu._artifacts.field_table.prognostic):
        icu["ps"] = 0.2 * rng.standard_normal(mu.state["ps"].data.shape)
    mu.set_fields(**icu)
    ici = {}
    for k in ("u", "v", "b"):
        arr = np.zeros(mi.state[k].data.shape)
        arr[:, :, 4:8] = icu[k][:, :, 0:4]
        ici[k] = arr
    if "ps" in icu:
        ici["ps"] = icu["ps"]
    mi.set_fields(**ici)
    mi.advance(30)
    mu.advance(30)
    assert not mi.panicked
    assert not mu.panicked
    diffs = {}
    for k in ("u", "v", "b"):
        di = np.asarray(mi.state[k].data)[:, :, 4:8]
        du = np.asarray(mu.state[k].data)[:, :, 0:4]
        diffs[k] = float(np.abs(di - du).max())
    dw = np.asarray(mi.state["w"].data)[:, :, 4:9]
    wu = np.asarray(mu.state["w"].data)[:, :, 0:5]
    diffs["w"] = float(np.abs(dw - wu).max())
    if "ps" in icu:
        diffs["ps"] = float(np.abs(np.asarray(mi.state["ps"].data)
                                   - np.asarray(mu.state["ps"].data)).max())
    assert max(diffs.values()) < 1e-13, diffs


# ================================================================
#  Implicit == explicit oracle on a (J==1) masked chart (small dt)
# ================================================================
def _ps_mode(model):
    fs = model.state["ps"].function_space
    var = tuple(nm for f in fs.bare.factors
                if not isinstance(f, ConstantSpace) for nm in f.names)

    def init(**c):
        return np.cos(2 * np.pi * c["x"]) + 0.0 * sum(c.values())
    init.__signature__ = inspect.Signature(
        [inspect.Parameter(nm, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for nm in var])
    return model.grid.create_field(fs, init=init)


def test_implicit_converges_to_the_explicit_oracle_on_a_masked_chart():
    # the explicit and implicit terrain barotropic solves coincide in
    # the continuum on a J==1 chart (H(x,y)==H_ref), so backward Euler
    # converges to the explicit oracle at first order over a masked chart
    errs = []
    for n_steps in (20, 40, 80):
        dt = 0.5 / n_steps
        exp = _model(_grid(n=16, nz=8, a=0.0, init=_cut),
                     hy.ExplicitFreeSurface(), csqr=1.0, n2=0.0, dt=dt)
        imp = _model(_grid(n=16, nz=8, a=0.0, init=_cut),
                     hy.ImplicitFreeSurface(pressure_iterations=40),
                     csqr=1.0, n2=0.0, dt=dt)
        for m in (exp, imp):
            m.set_fields(ps=_ps_mode(m).data)
            m.advance(n_steps)
        d = (np.asarray(exp.state["ps"].data)
             - np.asarray(imp.state["ps"].data))
        errs.append(float(np.sqrt((d * d).sum())))
    errs = np.asarray(errs)
    slopes = np.log2(errs[:-1] / errs[1:])
    assert errs[-1] < errs[0]
    assert slopes[-1] > 0.9            # backward Euler is first order


# ================================================================
#  All-wet chart == pure terrain (eps=1, bitwise-ish)
# ================================================================
def test_all_wet_chart_matches_pure_terrain():
    im = _model(_grid(a=0.4, init=_allwet, min_fraction=0.0),
                hy.ImplicitFreeSurface(pressure_iterations=40))
    un = _model(_pure_terrain(a=0.4),
                hy.ImplicitFreeSurface(pressure_iterations=40))
    rng = np.random.default_rng(2)
    ic = {k: 0.3 * rng.standard_normal(un.state[k].data.shape)
          for k in ("u", "v", "b", "ps")}
    im.set_fields(**ic)
    un.set_fields(**ic)
    im.advance(4)
    un.advance(4)
    for k in ("u", "v", "b", "ps", "w"):
        diff = np.abs(np.asarray(im.state[k].data)
                      - np.asarray(un.state[k].data)).max()
        assert diff < 1e-13, (k, diff)


# ================================================================
#  Identity chart (a=0, J==1) + mask == flat immersed (tight)
# ================================================================
def test_identity_chart_mask_matches_flat_immersed():
    # The gate runs on a STAIRCASE bottom (cuts on cell faces, every wet
    # cell fully wet: theta in {0, 1}, wet-centroid offset delta == 0). A
    # genuine (non-face-aligned) bottom cut is now INTENTIONALLY off
    # between the two paths: since the partial-bottom p_hyd merge the
    # flat immersed path applies a well-balanced partial-bottom pressure
    # correction (delta > 0), while the terrain-chart path defers it
    # (the recorded PB-D3 deferral,
    # design/plans/active/partial_bottom_phyd_plan.md). On a staircase
    # delta == 0 (to floating point), so the correction's magnitude is at
    # roundoff (~1e-17) and the strong equivalence holds to machine
    # precision. (test_partial_bottom_asymmetry_is_real exercises the
    # intended divergence on a genuine, non-face-aligned cut.)
    def flat_immersed():
        return Grid(
            (IM(8, (0.0, 1.0), periodic=True, name="x"),
             IM(8, (0.0, 1.0), periodic=True, name="y"),
             IM(8, (-1.0, 0.0), periodic=False, name="z")),
            immersed=ImmersedDomain(_staircase, order=4, min_fraction=0.1))
    ch = _model(_grid(a=0.0, init=_staircase),
                hy.ImplicitFreeSurface(pressure_iterations=40))
    fl = _model(flat_immersed(),
                hy.ImplicitFreeSurface(pressure_iterations=40))
    rng = np.random.default_rng(3)
    ic = {k: 0.2 * rng.standard_normal(ch.state[k].data.shape)
          for k in ("u", "v", "b", "ps")}
    ch.set_fields(**ic)
    fl.set_fields(**ic)
    ch.advance(4)
    fl.advance(4)
    for k in ("u", "v", "b", "ps", "w"):
        diff = np.abs(np.asarray(ch.state[k].data)
                      - np.asarray(fl.state[k].data)).max()
        assert diff < 1e-13, (k, diff)


def test_partial_bottom_asymmetry_is_real():
    # The identity-chart / flat-immersed divergence on a genuine
    # (non-face-aligned) bottom cut is intentional, not a bug: the flat
    # immersed path activates the well-balanced partial-bottom p_hyd
    # correction (delta > 0), the terrain-chart path defers it (PB-D3,
    # design/plans/active/partial_bottom_phyd_plan.md). The clean witness
    # is Core._pb_active, resolved once at bind from the
    # wet-centroid offsets.
    def flat_immersed():
        return Grid(
            (IM(8, (0.0, 1.0), periodic=True, name="x"),
             IM(8, (0.0, 1.0), periodic=True, name="y"),
             IM(8, (-1.0, 0.0), periodic=False, name="z")),
            immersed=ImmersedDomain(_cut, order=4, min_fraction=0.1))
    ch = _model(_grid(a=0.0), hy.ImplicitFreeSurface(pressure_iterations=40))
    fl = _model(flat_immersed(),
                hy.ImplicitFreeSurface(pressure_iterations=40))
    assert ch.module(hy.Core)._pb_active is False
    assert fl.module(hy.Core)._pb_active is True


# ================================================================
#  Explicit: the masked terrain gravity == pure terrain when all-wet
# ================================================================
def test_explicit_masked_depth_mean_div_all_wet_matches_pure_terrain():
    im = _model(_grid(a=0.4, init=_allwet, min_fraction=0.0),
                hy.ExplicitFreeSurface())
    un = _model(_pure_terrain(a=0.4), hy.ExplicitFreeSurface())
    rng = np.random.default_rng(5)
    ic = {k: rng.standard_normal(un.state[k].data.shape)
          for k in ("u", "v")}
    im.set_fields(**ic)
    un.set_fields(**ic)
    dim = np.asarray(im.module(hy.ExplicitFreeSurface)._depth_mean_div(
        im.state).data)
    dun = np.asarray(un.module(hy.ExplicitFreeSurface)._depth_mean_div(
        un.state).data)
    assert np.abs(dim - dun).max() == 0.0


# ================================================================
#  theta-mass conservation over steps on a cut chart
# ================================================================
def test_theta_mass_is_conserved_to_machine_zero():
    m = _model(_grid(a=0.4), hy.ImplicitFreeSurface(pressure_iterations=40),
               n2=0.0, dt=0.01)
    rng = np.random.default_rng(0)
    m.set_fields(**{k: 0.2 * rng.standard_normal(m.state[k].data.shape)
                    for k in ("u", "v", "b")})
    theta = m.grid.immersed.fraction(m.state["b"].function_space)

    def mass():
        return float(jnp.sum((theta * m.state["b"]).integrate().data))

    before = mass()
    m.advance(12)
    assert not m.panicked
    after = mass()
    assert abs(after - before) <= 1e-12 * max(abs(before), 1.0)


# ================================================================
#  Dry-DOF hygiene: the slope-advection buoyancy term keeps dead cells
#  dead (the terrain restoring reads only mask-consistent velocities)
# ================================================================
def test_dry_dof_hygiene_over_a_run():
    m = _model(_grid(a=0.4), hy.ImplicitFreeSurface(pressure_iterations=30),
               n2=2.0, dt=0.01)
    rng = np.random.default_rng(4)
    m.set_fields(**{k: 0.2 * rng.standard_normal(m.state[k].data.shape)
                    for k in ("u", "v", "b")})
    theta = np.asarray(
        m.grid.immersed.fraction(m.state["b"].function_space).data)
    dry = theta == 0.0
    assert bool(dry.any())
    m.advance(10)
    assert not m.panicked
    # the buoyancy on dead cells stays exactly zero (MaskState + the
    # slope term reads only min-rule-consistent velocities)
    b = np.asarray(m.state["b"].data)
    assert float(np.abs(b[dry]).max()) == 0.0


# ================================================================
#  Split-explicit now composes on a terrain + immersed grid (the
#  narrowed taught error is lifted, owner ruling 2026-07-19); the
#  deeper gates live in test_free_surface_terrain_split.py.
# ================================================================
def test_split_explicit_composes_on_terrain_immersed():
    m = hy.Model(
        grid=_grid(a=0.4),
        core=hy.Core(gravity=3.0),
        time_stepper=AdamBashforth(0.01, order=2),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.SplitExplicitFreeSurface(substeps=8),
        advection=False)
    fs = m.module(hy.SplitExplicitFreeSurface)
    assert fs._column is not None
    assert fs._immersed is not None
    rng = np.random.default_rng(0)
    m.set_fields(**{k: 0.2 * rng.standard_normal(m.state[k].shape)
                    for k in ("u", "v", "ps")})

    def volume():
        return float(jnp.sum(m.state["ps"].integrate().data))

    v0 = volume()
    m.advance(8)
    assert not m.panicked
    assert bool(jnp.isfinite(m.state["ps"].data).all())
    # the volume-exact (int alpha J dz) signature: plain int(ps) conserved
    assert abs(volume() - v0) < 1e-12 * max(abs(v0), 1.0)


# ================================================================
#  Reverse-mode autodiff through a short terrain + immersed run
# ================================================================
def test_grad_through_terrain_immersed_run_matches_fd():
    m = hy.Model(
        grid=_grid(n=8, nz=4, a=0.4),
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(0.01, order=2),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        stratification=hy.ConstantStratification(n2=0.0),
        free_surface=hy.ImplicitFreeSurface(
            epsilon=1.0,
            pressure_iterations=20),
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
    # the wet-column solve carries alpha J divides sealed double-where,
    # so the gradient is finite and matches a central FD
    assert bool(np.all(np.isfinite(grad)))
    rng2 = np.random.default_rng(5)
    direction = jnp.asarray(rng2.standard_normal(ps_leaf.shape),
                            dtype=ps_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    epsd = 1e-4
    fd = (float(loss(ps_leaf + epsd * direction))
          - float(loss(ps_leaf - epsd * direction))) / (2.0 * epsd)
    assert directional == pytest.approx(fd, rel=1e-4)


# ================================================================
#  Forced-4-device parity of the wet-column terrain barotropic solve
# ================================================================
def _forced4_solve(nx, device_ids):
    grid = _grid(n=nx, nz=8, a=0.4, device_ids=device_ids)
    m = _model(grid, hy.ImplicitFreeSurface(
        pressure_iterations=30, pressure_tolerance=1e-8), dt=0.05)
    rng = np.random.default_rng(52)
    m.set_fields(**{k: rng.standard_normal(m.state[k].shape)
                    for k in ("u", "v", "ps")})
    fs = m.module(hy.ImplicitFreeSurface)
    out = fs._barotropic_solve(m.state, _ctx(3.0, 0.05))
    return np.asarray(out["ps"].data)


@pytest.mark.multi_device
@pytest.mark.parametrize(
    "nx", [pytest.param(16, id="aligned-x16"),
           pytest.param(12, id="replicated-x12")])
def test_forced4_terrain_immersed_matches_single_device(nx):
    ids = tuple(range(jax.device_count()))
    one = _forced4_solve(nx, (0,))
    many = _forced4_solve(nx, ids)
    scale = np.max(np.abs(one))
    assert np.max(np.abs(one - many)) / scale < 1e-8


# ================================================================
#  The wet-aware multigrid preconditioner on the model surface (the
#  taught error is lifted; it composes with the cut-cell domain)
# ================================================================
def test_terrain_immersed_multigrid_preconditioner_engages():
    # pressure_preconditioner="multigrid" drives the wet-column
    # barotropic solve on a terrain + immersed grid without raising, and
    # the surface pressure it writes is finite
    m = _model(_grid(n=16, a=0.8), hy.ImplicitFreeSurface(
        epsilon=1.0, pressure_iterations=30, pressure_tolerance=1e-8,
        pressure_preconditioner="multigrid"), dt=0.05)
    rng = np.random.default_rng(3)
    m.set_fields(**{k: rng.standard_normal(m.state[k].shape)
                    for k in ("u", "v", "ps")})
    fs = m.module(hy.ImplicitFreeSurface)
    out = fs._barotropic_solve(m.state, _ctx(3.0, 0.05))
    assert bool(np.all(np.isfinite(np.asarray(out["ps"].data))))


def _forced4_solve_mg(nx, device_ids):
    grid = _grid(n=nx, nz=8, a=0.4, device_ids=device_ids)
    m = _model(grid, hy.ImplicitFreeSurface(
        pressure_iterations=30, pressure_tolerance=1e-8,
        pressure_preconditioner="multigrid"), dt=0.05)
    rng = np.random.default_rng(52)
    m.set_fields(**{k: rng.standard_normal(m.state[k].shape)
                    for k in ("u", "v", "ps")})
    fs = m.module(hy.ImplicitFreeSurface)
    out = fs._barotropic_solve(m.state, _ctx(3.0, 0.05))
    return np.asarray(out["ps"].data)


@pytest.mark.multi_device
@pytest.mark.parametrize("nx", [pytest.param(16, id="aligned-x16")])
def test_forced4_terrain_immersed_multigrid_matches_single_device(nx):
    ids = tuple(range(jax.device_count()))
    one = _forced4_solve_mg(nx, (0,))
    many = _forced4_solve_mg(nx, ids)
    scale = np.max(np.abs(one))
    assert np.max(np.abs(one - many)) / scale < 1e-8


def test_grad_through_terrain_immersed_multigrid_run_matches_fd():
    # the wet-aware multigrid preconditioner path is reverse-mode
    # differentiable end to end (the preconditioner does not touch the
    # solution, only the convergence; the wet-column alpha J divides are
    # sealed), so jax.grad is finite and matches a central FD
    m = hy.Model(
        grid=_grid(n=8, nz=4, a=0.4),
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(0.01, order=2),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        stratification=hy.ConstantStratification(n2=0.0),
        free_surface=hy.ImplicitFreeSurface(
            epsilon=1.0,
            pressure_iterations=20,
            pressure_preconditioner="multigrid"),
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
    epsd = 1e-4
    fd = (float(loss(ps_leaf + epsd * direction))
          - float(loss(ps_leaf - epsd * direction))) / (2.0 * epsd)
    assert directional == pytest.approx(fd, rel=1e-4)
