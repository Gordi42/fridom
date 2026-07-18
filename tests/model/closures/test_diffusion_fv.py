"""The diffusion closures on finite-volume (CellAvg) walled grids.

Prefix-mirrored shard of ``test_diffusion.py`` (oversized-module rule):
the walled-grid behaviour of the ``_DiffusionClosure`` family on the
**finite-volume** family (``CellAvg`` cells, the nonhydro2 default) —
the structural no-flux tracer wall, the free-slip and no-slip velocity
walls, the wall-normal component, the ``slip=`` API, conservation on
uniform and stretched columns, the terrain along-sigma semantics, the
FV-vs-nodal parity, and the raw-profile / tagged-cell taught
rejections. The FV chain lands the interior flux on the same nodal
``Inner`` face the nodal chain uses (``CellAvg -> Inner``,
``FaceDifference``), so the nodal expected eigenvalues port directly
(the ghost fills are bit-identical to ``Center``).

The builders are duplicated (self-contained shard, import-mode
importlib) rather than imported across test files.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.model.closures.diffusion import (
    BiharmonicDiffusion,
    BiharmonicFriction,
    HarmonicDiffusion,
    HarmonicFriction,
)
from fridom.model.model import Model, _chunk_body
from fridom.model.module import Module
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.core import fv_cgrid_overrides
from fridom.spatial.bc import BC
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh

N = 8
L = 1.0
DX = L / N
DT = 1e-3
EXP_A = 1.5  # exponential-stretch strength (~3.7:1 cell-width ratio)


# ================================================================
#  FV cores (velocities on the C-grid faces, cells on CellAvg) +
#  the grid-aware FV C-grid diff profile that exposes the face flux
# ================================================================
class FVCore(Module):

    """FV toy core: wall-normal u (x), transverse v, an FV tracer b.

    On a 1-D x-walled grid ``u`` is the wall-normal velocity
    (``Inner[Dirichlet]`` along x), ``v`` (staggered on the absent z)
    is the tangential ``CellAvg(x)`` velocity, and ``b`` is the
    ``CellAvg(x)`` tracer — the FV mirror of the nodal Center/Inner
    toy core.
    """

    field_declarations = (
        fr.model.FieldDeclaration.velocity(
            "u", "x", space=fr.spatial.Staggered("x", family="fv")),
        fr.model.FieldDeclaration.velocity(
            "v", "z", space=fr.spatial.Staggered("z", family="fv")),
        fr.model.FieldDeclaration.tracer(
            "b", space=fr.spatial.Collocated(family="fv")),
    )

    def grid_dispatch_overrides(self, grid):
        """Install the face-exposing FV C-grid diff profile."""
        return fv_cgrid_overrides(grid.factors)

    @fr.model.term(advances=("u", "v", "b"), linear=True,
                   transports=("u", "v", "b"))
    def zero(self, state, _ctx):
        return {name: 0.0 * state[name] for name in ("u", "v", "b")}


class FVZCore(Module):

    """FV core for z-walled columns: u tangential along z, w wall-normal.

    ``u`` (staggered on x) is ``CellAvg(z)`` tangential along the walled
    z-axis; ``w`` (staggered on z) is the ``Inner[Dirichlet](z)``
    wall-normal component; ``b`` is the ``CellAvg`` tracer.
    """

    field_declarations = (
        fr.model.FieldDeclaration.velocity(
            "u", "x", space=fr.spatial.Staggered("x", family="fv")),
        fr.model.FieldDeclaration.velocity(
            "w", "z", space=fr.spatial.Staggered("z", family="fv")),
        fr.model.FieldDeclaration.tracer(
            "b", space=fr.spatial.Collocated(family="fv")),
    )

    def grid_dispatch_overrides(self, grid):
        """Install the face-exposing FV C-grid diff profile."""
        return fv_cgrid_overrides(grid.factors)

    @fr.model.term(advances=("u", "w", "b"), linear=True,
                   transports=("u", "w", "b"))
    def zero(self, state, _ctx):
        return {name: 0.0 * state[name] for name in ("u", "w", "b")}


class FVTracerCore(Module):

    """A single FV tracer ``b`` (CellAvg on every axis)."""

    field_declarations = (
        fr.model.FieldDeclaration.tracer(
            "b", space=fr.spatial.Collocated(family="fv")),
    )

    def grid_dispatch_overrides(self, grid):
        """Install the face-exposing FV C-grid diff profile."""
        return fv_cgrid_overrides(grid.factors)

    @fr.model.term(advances=("b",), linear=True, transports=("b",))
    def zero(self, state, _ctx):
        return {"b": 0.0 * state["b"]}


class NodalCore(Module):

    """Nodal mirror of :class:`FVCore` (Center cells, Inner faces)."""

    field_declarations = (
        fr.model.FieldDeclaration.velocity(
            "u", "x", space=fr.spatial.Staggered("x")),
        fr.model.FieldDeclaration.velocity(
            "v", "z", space=fr.spatial.Staggered("z")),
        fr.model.FieldDeclaration.tracer("b"),
    )

    @fr.model.term(advances=("u", "v", "b"), linear=True,
                   transports=("u", "v", "b"))
    def zero(self, state, _ctx):
        return {name: 0.0 * state[name] for name in ("u", "v", "b")}


# ================================================================
#  Grid + model builders (self-contained)
# ================================================================
def make_grid(names=("x",), periodic=False):
    """Build a uniform structured grid; ``periodic`` scalar or tuple."""
    if isinstance(periodic, bool):
        periodic = (periodic,) * len(names)
    return Grid(tuple(
        IntervalMesh(N, (0.0, L), periodic=p, name=name)
        for name, p in zip(names, periodic, strict=True)))


def exp_map(s):
    """Exponential stretch [0, 1] -> [0, 1], strictly increasing."""
    return (jnp.exp(EXP_A * s) - 1.0) / (jnp.exp(EXP_A) - 1.0)


def cell_widths(n):
    """Physical primal cell widths of the exp-stretched z column."""
    s = np.arange(n + 1) / n
    faces = (np.exp(EXP_A * s) - 1.0) / (np.exp(EXP_A) - 1.0)
    return np.diff(faces)


def stretched_1d(n=N):
    """1-D bounded stretched column in z (a MappedIntervalMesh)."""
    return Grid((MappedIntervalMesh(
        n, (0.0, 1.0), exp_map, periodic=False, name="z"),))


def stretched_xz(n=N):
    """Grid: x periodic (uniform), z bounded (stretched)."""
    mx = IntervalMesh(n, (0.0, 1.0), periodic=True, name="x")
    mz = MappedIntervalMesh(n, (0.0, 1.0), exp_map, periodic=False,
                            name="z")
    return Grid((mx, mz))


def terrain_grid(n, hfac):
    """Grid: x periodic, sigma bounded; chart zp = sigma * H(x)."""
    mx = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="x")
    ms = IntervalMesh(n, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H},
        params={"H": lambda x: 1.0 + hfac * jnp.sin(x)})
    return Grid((mx, ms), mapping=mapping)


def make_model(core, closure, grid):
    return Model(grid=grid, modules=(core, closure),
                 time_stepper=AdamBashforth(DT, order=2))


def data(field):
    return np.asarray(field.data)


# ================================================================
#  Discrete-symbol builders (bit-identical to the nodal shard)
# ================================================================
def lam_wall(m):
    """Discrete symbol of the walled cosine/sine mode m (both walls)."""
    return (2.0 * np.sin(np.pi * m / (2 * N)) / DX) ** 2


def cos_mode(m):
    """Build a discrete cosine (Neumann / no-flux / free-slip) mode."""
    return lambda x: np.cos(np.pi * m * x / L)


def sin_mode(m):
    """Build a discrete sine (Dirichlet / no-slip / wall-normal) mode."""
    return lambda x: np.sin(np.pi * m * x / L)


def zero_field(x):
    """Return a zero initial condition (coordinate-signature builder)."""
    return 0.0 * x


def ones_field(x):
    """Return a uniform unit initial condition along ``x``."""
    return np.ones_like(x)


# ================================================================
#  Test 1: walled FV free-slip / no-flux exactness (cosine mode)
# ================================================================
def test_fv_no_flux_tracer_cosine_decays_at_the_discrete_rate():
    # a discrete cosine is the Neumann (no-flux) eigenmode; the FV
    # flux-retag wall closure decays it at exactly -kappa * lam_wall(m)
    kappa = 3e-3
    model = make_model(FVCore(), HarmonicDiffusion(kappa), make_grid())
    for m in (1, 2, 3):
        model.set_fields(b=cos_mode(m))
        td = model.tendency(model.state)
        want = -kappa * lam_wall(m) * data(model.state["b"])
        np.testing.assert_allclose(data(td["b"]), want, atol=1e-13)


def test_fv_free_slip_velocity_cosine_decays_at_the_discrete_rate():
    # free-slip = zero tangential wall stress: the FV tangential
    # velocity cosine mode decays at -nu * lam_wall(m)
    nu = 5e-3
    model = make_model(FVCore(), HarmonicFriction(nu, slip="free"),
                       make_grid())
    for m in (1, 2, 3):
        model.set_fields(v=cos_mode(m), u=zero_field)
        td = model.tendency(model.state)
        want = -nu * lam_wall(m) * data(model.state["v"])
        np.testing.assert_allclose(data(td["v"]), want, atol=1e-13)


def test_fv_biharmonic_no_flux_tracer_cosine_decays_at_lam_squared():
    # the biharmonic -H(H(q)) on a cosine eigenmode decays at
    # -kappa * lam_wall(m)^2 (both passes along the walled axis)
    kappa = 1e-3
    model = make_model(FVCore(), BiharmonicDiffusion(kappa), make_grid())
    for m in (1, 2, 3):
        model.set_fields(b=cos_mode(m))
        td = model.tendency(model.state)
        want = -kappa * lam_wall(m) ** 2 * data(model.state["b"])
        np.testing.assert_allclose(data(td["b"]), want, atol=1e-12)


def test_fv_biharmonic_free_slip_velocity_cosine_decays_at_lam_squared():
    nu = 2e-4
    model = make_model(FVCore(), BiharmonicFriction(nu, slip="free"),
                       make_grid())
    for m in (1, 2, 3):
        model.set_fields(v=cos_mode(m), u=zero_field)
        td = model.tendency(model.state)
        want = -nu * lam_wall(m) ** 2 * data(model.state["v"])
        np.testing.assert_allclose(data(td["v"]), want, atol=1e-12)


# ================================================================
#  Test 2: walled FV no-slip exactness (sine mode, the -3 wall rows)
# ================================================================
def test_fv_no_slip_velocity_sine_decays_at_the_discrete_rate():
    # no-slip = u=0 at the wall (factor-of-2 ghost): the sine mode of
    # the FV tangential velocity decays at exactly -nu * lam_wall(k),
    # the wall cells carrying the -3 (2 + 1) rows (bit-identical ghosts)
    nu = 5e-3
    model = make_model(FVCore(), HarmonicFriction(nu, slip="no"),
                       make_grid())
    for k in (1, 2, 3, N):
        model.set_fields(v=sin_mode(k), u=zero_field)
        td = model.tendency(model.state)
        want = -nu * lam_wall(k) * data(model.state["v"])
        np.testing.assert_allclose(data(td["v"]), want, atol=1e-12)


def test_fv_biharmonic_no_slip_velocity_sine_decays_at_lam_squared():
    nu = 2e-4
    model = make_model(FVCore(), BiharmonicFriction(nu, slip="no"),
                       make_grid())
    for k in (1, 2, 3):
        model.set_fields(v=sin_mode(k), u=zero_field)
        td = model.tendency(model.state)
        want = -nu * lam_wall(k) ** 2 * data(model.state["v"])
        np.testing.assert_allclose(data(td["v"]), want, atol=1e-12)


def test_fv_wall_normal_component_sine_is_slip_independent():
    # the wall-normal velocity (Inner[Dirichlet]) closes on its own tag
    # at -nu * lam_wall(m); slip= does not touch it
    nu = 5e-3
    for slip in ("free", "no"):
        model = make_model(FVCore(), HarmonicFriction(nu, slip=slip),
                           make_grid())
        for m in (1, 2, 3):
            model.set_fields(u=sin_mode(m), v=zero_field)
            td = model.tendency(model.state)
            want = -nu * lam_wall(m) * data(model.state["u"])
            np.testing.assert_allclose(data(td["u"]), want, atol=1e-12)


# ================================================================
#  Test 3: conservation (measure-weighted) on uniform + stretched FV
# ================================================================
def test_fv_no_flux_tracer_conserves_the_measure_weighted_integral():
    model = make_model(FVCore(), HarmonicDiffusion(1.0), make_grid())
    rng = np.random.default_rng(0)
    model.set_fields(b=lambda x: rng.standard_normal(x.shape))
    td = model.tendency(model.state)["b"]
    weighted = float(data(td.integrate()).ravel()[0])
    scale = float(np.abs(data(td)).sum()) + 1.0
    assert abs(weighted) < 1e-12 * scale


def test_fv_stretched_column_conserves_the_measure_weighted_integral():
    # on a stretched FV column the per-cell widths differ, so only the
    # MEASURE-WEIGHTED integral telescopes to machine zero (the plain
    # unweighted cell sum does not)
    model = make_model(FVTracerCore(), HarmonicDiffusion(1.0),
                       stretched_1d())
    rng = np.random.default_rng(1)
    model.set_fields(b=lambda z: rng.standard_normal(z.shape))
    td = model.tendency(model.state)["b"]
    weighted = float(data(td.integrate()).ravel()[0])
    unweighted = float(np.sum(data(td)))
    assert abs(weighted) < 1e-12
    assert abs(unweighted) > 1.0  # the unweighted sum does not telescope


# ================================================================
#  Test 4: FV-vs-nodal parity on a walled uniform grid (both slips)
# ================================================================
@pytest.mark.parametrize("slip", ["free", "no"])
def test_fv_velocity_tendency_matches_the_nodal_tendency(slip):
    # the FV tangential velocity (CellAvg) and the nodal one (Center)
    # share the primal cell midpoints, the same kernels, and the same
    # flux-retag wall closure, so the tendency agrees (eager: bitwise)
    nu = 1e-2
    mf = make_model(FVCore(), HarmonicFriction(nu, slip=slip), make_grid())
    mn = make_model(NodalCore(), HarmonicFriction(nu, slip=slip),
                    make_grid())
    rng = np.random.default_rng(7)
    fields = {c: rng.standard_normal(mf.state[c].data.shape)
              for c in ("u", "v")}
    mf.set_fields(**fields)
    mn.set_fields(**fields)
    tf = data(mf.tendency(mf.state)["v"])
    tn = data(mn.tendency(mn.state)["v"])
    np.testing.assert_allclose(tf, tn, rtol=0, atol=1e-12)


def test_fv_tracer_tendency_matches_the_nodal_tendency():
    kappa = 1e-2
    mf = make_model(FVCore(), HarmonicDiffusion(kappa), make_grid())
    mn = make_model(NodalCore(), HarmonicDiffusion(kappa), make_grid())
    rng = np.random.default_rng(8)
    b0 = rng.standard_normal(mf.state["b"].data.shape)
    mf.set_fields(b=b0)
    mn.set_fields(b=b0)
    tf = data(mf.tendency(mf.state)["b"])
    tn = data(mn.tendency(mn.state)["b"])
    np.testing.assert_allclose(tf, tn, rtol=0, atol=1e-12)


# ================================================================
#  Test 5: periodic FV first coverage (parity with the nodal path)
# ================================================================
def test_fv_periodic_tracer_matches_the_nodal_tracer_bitwise():
    # the fully-periodic FV path (the two-pass FaceDifference /
    # FluxDifference chain) equals the nodal periodic chain bit-for-bit
    kappa = 2e-3
    grid_fv = make_grid(names=("x", "z"), periodic=True)
    grid_nd = make_grid(names=("x", "z"), periodic=True)
    mf = make_model(FVCore(), HarmonicDiffusion(kappa), grid_fv)
    mn = make_model(NodalCore(), HarmonicDiffusion(kappa), grid_nd)

    def init(x, z):
        return np.sin(2 * np.pi * x) * np.cos(4 * np.pi * z)

    mf.set_fields(b=init)
    mn.set_fields(b=init)
    np.testing.assert_array_equal(
        data(mf.tendency(mf.state)["b"]),
        data(mn.tendency(mn.state)["b"]))


def test_fv_periodic_tracer_is_the_direct_two_pass_chain():
    # on a periodic FV grid the closure enters no walled branch: it is
    # exactly the plain (q.diff * k).diff chain, bit-for-bit
    kappa = 2e-3
    grid = make_grid(names=("x", "z"), periodic=True)
    model = make_model(FVCore(), HarmonicDiffusion(kappa), grid)
    model.set_fields(
        b=lambda x, z: np.sin(2 * np.pi * x) * np.cos(4 * np.pi * z))
    td = model.tendency(model.state)
    q = model.state["b"]
    direct = None
    for axis in ("x", "z"):
        contribution = (q.diff(axis) * kappa).diff(axis)
        direct = (contribution if direct is None
                  else direct + contribution)
    np.testing.assert_array_equal(data(td["b"]), data(direct))


# ================================================================
#  Test 6: slip API on FV (per-field mapping; wall-normal ignores slip)
# ================================================================
def test_fv_slip_mapping_selects_the_slip_per_velocity():
    # on a z-walled FV grid u is tangential (CellAvg(z)); u no-slip
    # drags a uniform wall-parallel flow, w (wall-normal) is untouched
    nu = 1e-2
    grid = make_grid(names=("x", "z"), periodic=(True, False))
    model = make_model(
        FVZCore(), HarmonicFriction(nu, slip={"u": "no", "w": "free"}),
        grid)
    model.set_fields(u=lambda x, z: np.ones_like(x) + 0.0 * z,
                     w=lambda x, z: 0.0 * (x + z))
    td = model.tendency(model.state)
    tend_u = data(td["u"])
    assert (tend_u[:, 0] < 0.0).all()       # no-slip: wall drag
    assert (tend_u[:, -1] < 0.0).all()
    np.testing.assert_allclose(tend_u[:, 1:-1], 0.0, atol=1e-13)


def test_fv_wall_normal_velocity_ignores_the_slip_choice():
    # w is the wall-normal (Inner[Dirichlet]) component: its own chain
    # is slip-independent, so its tendency is identical for free vs no,
    # while a tangential u differs between the two
    nu = 1e-2
    grid = make_grid(names=("x", "z"), periodic=(True, False))

    def init_u(x, z):
        return np.sin(2 * np.pi * x) * np.sin(np.pi * z)

    def init_w(x, z):
        return np.cos(2 * np.pi * x) * np.sin(np.pi * z)

    def tend(slip):
        model = make_model(FVZCore(), HarmonicFriction(nu, slip=slip),
                           grid)
        model.set_fields(u=init_u, w=init_w)
        out = model.tendency(model.state)
        return data(out["u"]), data(out["w"])

    u_free, w_free = tend("free")
    u_no, w_no = tend("no")
    np.testing.assert_array_equal(w_free, w_no)   # wall-normal: no slip
    assert np.abs(u_free - u_no).max() > 0.0      # tangential: differs


# ================================================================
#  Test 7: stretched FV no-slip uses the wall cell's own width
# ================================================================
def test_fv_stretched_no_slip_drag_scales_with_the_local_cell_width():
    # a wall-parallel uniform flow feels -2 nu / dn^2 in each wall cell,
    # dn the LOCAL (stretched) FV cell width; the two walls carry
    # different widths so the drags differ -- the correction reads the
    # per-cell measure, not a single dz
    nu = 1e-2
    model = make_model(FVZCore(), HarmonicFriction(nu, slip="no"),
                       stretched_xz())
    model.set_fields(u=lambda x, z: np.ones_like(x) + 0.0 * z,
                     w=lambda x, z: 0.0 * (x + z))
    tend = data(model.tendency(model.state)["u"])
    w = cell_widths(N)
    assert w[0] != w[-1]  # a genuinely stretched column
    np.testing.assert_allclose(tend[:, 0], -2 * nu / w[0] ** 2,
                               rtol=1e-10)
    np.testing.assert_allclose(tend[:, -1], -2 * nu / w[-1] ** 2,
                               rtol=1e-10)
    np.testing.assert_allclose(tend[:, 1:-1], 0.0, atol=1e-12)


def test_fv_stretched_free_slip_uniform_flow_has_no_drag():
    model = make_model(FVZCore(), HarmonicFriction(1e-2, slip="free"),
                       stretched_xz())
    model.set_fields(u=lambda x, z: np.ones_like(x) + 0.0 * z,
                     w=lambda x, z: 0.0 * (x + z))
    assert np.abs(data(model.tendency(model.state)["u"])).max() == 0.0


# ================================================================
#  Test 8: terrain-mapped FV along-sigma (no H(x) coupling)
# ================================================================
def test_fv_terrain_diffusion_carries_no_h_coupling():
    # the chart factor H(x) lives in grid.metric and never enters
    # diff/measure, so the FV terrain diffusion tendency is BITWISE
    # independent of the terrain amplitude (the along-sigma semantic)
    def tendency(hfac):
        closure = HarmonicDiffusion(0.5, kappa_v=0.3, vertical="sigma")
        model = make_model(FVTracerCore(), closure, terrain_grid(8, hfac))
        model.set_fields(
            b=lambda x, sigma: np.sin(x) * np.cos(np.pi * sigma))
        return data(model.tendency(model.state)["b"])

    mild, steep = tendency(0.2), tendency(0.6)
    assert np.abs(mild).max() > 0.0  # a non-trivial operator
    assert np.abs(mild - steep).max() == 0.0


# ================================================================
#  Test 9: reverse-mode AD through a walled FV run
# ================================================================
def _fv_friction_grad_loss(core, grid, fields, slip, nu, n_steps=8):
    """Build a grad-ready loss over a walled FV harmonic-friction run."""
    model = make_model(core, HarmonicFriction(nu, slip=slip), grid)
    model.set_fields(**fields)
    record = model._artifacts.record
    carry = model._carry
    stepper = model._stepper
    leaf = next(m for m in carry.modules
                if isinstance(m, HarmonicFriction)).nu
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    idx = next(i for i, lf in enumerate(leaves) if lf is leaf)

    def loss(theta):
        packed = list(leaves)
        packed[idx] = theta
        c = jax.tree_util.tree_unflatten(treedef, packed)
        final = _chunk_body(record, n_steps, c, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    return loss, jnp.asarray(leaf, dtype=jnp.float64)


@pytest.mark.parametrize("slip", ["free", "no"])
def test_fv_friction_grad_matches_central_fd_on_walls(slip):
    nu = 2e-2
    fields = {"v": sin_mode(2), "u": sin_mode(3)}
    loss, x0 = _fv_friction_grad_loss(
        FVCore(), make_grid(), fields, slip, nu)
    g = float(jax.grad(loss)(x0))
    assert np.isfinite(g)
    eps = 1e-4
    fd = float((loss(x0 * (1 + eps)) - loss(x0 * (1 - eps)))
               / (2 * x0 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-4)


def test_fv_friction_grad_matches_central_fd_on_a_stretched_column():
    # the spatial-layer VJP seal (divide_by_codomain_measure) makes the
    # stretched FV column reverse-mode differentiable end to end
    nu = 2e-2
    fields = {
        "u": lambda x, z: np.sin(2 * np.pi * x) * np.sin(np.pi * z),
        "w": lambda x, z: np.cos(2 * np.pi * x) * np.sin(np.pi * z)}
    loss, x0 = _fv_friction_grad_loss(
        FVZCore(), stretched_xz(), fields, "no", nu)
    g = float(jax.grad(loss)(x0))
    assert np.isfinite(g)
    eps = 1e-4
    fd = float((loss(x0 * (1 + eps)) - loss(x0 * (1 - eps)))
               / (2 * x0 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-4)


# ================================================================
#  Test 10: a walled CellAvg on a raw (non-FV-dispatch) core is taught
# ================================================================
def test_fv_walled_target_without_the_fv_dispatch_is_rejected():
    # a walled CellAvg target on a grid WITHOUT the face-exposing FV
    # C-grid diff profile: the collocated FVDerivative chain cannot
    # close a wall flux, so the bind-time face probe rejects it (rather
    # than silently running the wrong collocated stencil)
    class RawFVCore(Module):
        field_declarations = (
            fr.model.FieldDeclaration.tracer(
                "b", space=fr.spatial.Collocated(family="fv")),
        )

        @fr.model.term(advances=("b",), linear=True, transports=("b",))
        def zero(self, state, _ctx):
            return {"b": 0.0 * state["b"]}

    with pytest.raises(NotImplementedError,
                       match="does not expose a face-located flux"):
        make_model(RawFVCore(), HarmonicDiffusion(1.0), make_grid())


# ================================================================
#  Test 11: a fixed-value (tagged) FV cell wall is out of scope
# ================================================================
def test_fv_tagged_cell_wall_is_a_taught_rejection():
    # a fixed-value (Dirichlet) FV cell wall is the stage-2e
    # boundary-data path, out of scope. On the FV family it cannot even
    # be *declared*: an average cell carries no boundary structure
    # (topology-driven walls, C8), so pinning a BC on a family='fv'
    # collocated coordinate is a taught rejection at the space layer,
    # before the closure. (The closure's own tagged-CellAvg guard is a
    # defensive branch behind this space-layer gate; the nodal
    # fixed-value Center wall is covered in test_diffusion_walls.py.)
    class TaggedFVCore(Module):
        field_declarations = (
            fr.model.FieldDeclaration.tracer(
                "d", space=fr.spatial.Collocated(
                    family="fv", bc={"x": BC.DIRICHLET})),
        )

        @fr.model.term(advances=("d",), linear=True, transports=("d",))
        def zero(self, state, _ctx):
            return {"d": 0.0 * state["d"]}

    with pytest.raises(ValueError, match="no boundary structure"):
        make_model(TaggedFVCore(), HarmonicDiffusion(1e-3), make_grid())


# ================================================================
#  Sharded walled axis: the FV inter-shard halo must be synced -- the
#  fix/walled-shard-halo-validity regression
# ================================================================
@pytest.mark.multi_device
def test_sharded_walled_fv_diffusion_matches_single_device(
        forced_devices):
    # regression: the FV flux difference on a SHARDED walled axis must
    # sync the inter-shard halo. The reconstruction CellAvg -> Inner
    # shrinks the codomain, so its exterior reach cancels at the wall;
    # the pre-fix requirements published that cancelled reach, skipped
    # the sync, and every interior cell adjacent to a shard boundary
    # read stale ghosts. Must match the analytic Neumann rate and the
    # one-device run.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    kappa = 3e-3
    m = 2
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        grid = Grid((IntervalMesh(N, (0.0, L), periodic=False,
                                  name="x"),), device_ids=device_ids)
        model = make_model(FVTracerCore(), HarmonicDiffusion(kappa),
                           grid)
        model.set_fields(b=cos_mode(m))
        td = model.tendency(model.state)
        results[tag] = data(td["b"])
        if tag == "many" and jax.device_count() > 1:
            assert model.state["b"]._data.sharding.spec[0] == "devices"
    want = -kappa * lam_wall(m) * data(model.state["b"])
    np.testing.assert_allclose(results["many"], want, atol=1e-13)
    np.testing.assert_allclose(results["many"], results["one"],
                               rtol=1e-12, atol=1e-14)
