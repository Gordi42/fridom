"""The diffusion closures on mapped grids: stretched + terrain (stage 4).

Prefix-mirrored shard of ``test_diffusion.py`` (oversized-module rule):
the mapped-grid behaviour of the ``_DiffusionClosure`` family — the
**along-coordinate** (along-sigma) semantics on terrain-following charts
(no ``H(x)`` coupling, the ROMS ``MIX_S_UV`` convention) and the
metrically-exact order-2 physical operator on a stretched
``MappedIntervalMesh`` column (conservation via measure-weighted
telescoping). The builders are duplicated (self-contained shard) rather
than imported.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.model.closures.diffusion import (
    BiharmonicDiffusion,
    HarmonicDiffusion,
    HarmonicFriction,
)
from fridom.model.model import Model, _chunk_body
from fridom.model.module import Module
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh

DT = 1e-3
EXP_A = 1.5  # exponential-stretch strength (~3.7:1 cell-width ratio)


# ================================================================
#  Stretched-column geometry (a MappedIntervalMesh, physical z)
# ================================================================
def exp_map(s):
    """Exponential stretch [0, 1] -> [0, 1], strictly increasing."""
    return (jnp.exp(EXP_A * s) - 1.0) / (jnp.exp(EXP_A) - 1.0)


def cell_widths(n):
    """Physical primal cell widths of the exp-stretched z column."""
    s = np.arange(n + 1) / n
    faces = (np.exp(EXP_A * s) - 1.0) / (np.exp(EXP_A) - 1.0)
    return np.diff(faces)


def stretched_1d(n):
    """1D bounded stretched column in z."""
    return Grid((MappedIntervalMesh(
        n, (0.0, 1.0), exp_map, periodic=False, name="z"),))


def stretched_2d(n):
    """Grid: x periodic (uniform), z bounded (stretched)."""
    mx = IntervalMesh(n, (0.0, 1.0), periodic=True, name="x")
    mz = MappedIntervalMesh(n, (0.0, 1.0), exp_map, periodic=False,
                            name="z")
    return Grid((mx, mz))


# ================================================================
#  Terrain-following geometry (uniform meshes + a chart mapping)
# ================================================================
def terrain_grid(n, hfac):
    """Grid: x periodic, sigma bounded; chart zp = sigma * H(x)."""
    mx = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="x")
    ms = IntervalMesh(n, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H},
        params={"H": lambda x: 1.0 + hfac * jnp.sin(x)})
    return Grid((mx, ms), mapping=mapping)


# ================================================================
#  Toy cores
# ================================================================
class TracerCore(Module):

    """A single tracer ``b`` (Collocated on every axis)."""

    field_declarations = (fr.model.FieldDeclaration.tracer("b"),)

    @fr.model.term(advances=("b",), linear=True, transports=("b",))
    def zero(self, state, _ctx):
        return {"b": 0.0 * state["b"]}


class ZVelCore(Module):

    """u tangential along z (staggered x), w wall-normal (staggered z)."""

    field_declarations = (
        fr.model.FieldDeclaration.velocity(
            "u", "x", space=fr.spatial.Staggered("x")),
        fr.model.FieldDeclaration.velocity(
            "w", "z", space=fr.spatial.Staggered("z")),
    )

    @fr.model.term(advances=("u", "w"), linear=True,
                   transports=("u", "w"))
    def zero(self, state, _ctx):
        return {n: 0.0 * state[n] for n in ("u", "w")}


class SigmaVelCore(Module):

    """u tangential along sigma, w wall-normal (staggered sigma)."""

    field_declarations = (
        fr.model.FieldDeclaration.velocity(
            "u", "x", space=fr.spatial.Staggered("x")),
        fr.model.FieldDeclaration.velocity(
            "w", "sigma", space=fr.spatial.Staggered("sigma")),
    )

    @fr.model.term(advances=("u", "w"), linear=True,
                   transports=("u", "w"))
    def zero(self, state, _ctx):
        return {n: 0.0 * state[n] for n in ("u", "w")}


def make_model(core, closure, grid):
    return Model(grid=grid, modules=(core, closure),
                 time_stepper=AdamBashforth(DT, order=2))


def data(field):
    return np.asarray(field.data)


# ================================================================
#  Stretched column: binds, runs, metrically exact at order 2
# ================================================================
def test_stretched_column_binds_and_steps_finite():
    # a bounded MappedIntervalMesh column binds through the walled
    # nodal path with no gate and steps finite (forward is supported)
    model = make_model(TracerCore(), HarmonicDiffusion(0.5),
                       stretched_1d(12))
    model.set_fields(b=lambda z: np.cos(np.pi * z))
    final = _chunk_body(model._artifacts.record, 6,
                        model._carry, model._stepper)
    assert bool(np.all(np.isfinite(data(final.state[-1]))))


def test_stretched_harmonic_converges_to_the_physical_operator():
    # the along-coordinate chain divides by the codomain measure, so on
    # a stretched column it is the PHYSICAL d/dz(k dq/dz); it converges
    # at order ~2 in the interior (q'' = -pi^2 cos(pi z) analytic)
    kappa = 0.7

    def interior_error(n):
        grid = stretched_1d(n)
        model = make_model(TracerCore(), HarmonicDiffusion(kappa), grid)
        model.set_fields(b=lambda z: np.cos(np.pi * z))
        td = data(model.tendency(model.state)["b"])
        z = data(grid.evaluation_nodes(grid.factors[0].center, "z"))
        analytic = -kappa * np.pi**2 * np.cos(np.pi * z)
        return np.abs(td - analytic)[1:-1].max()

    order = np.log2(interior_error(8) / interior_error(16))
    assert 1.8 < order < 2.2


def test_stretched_tracer_conserves_measure_weighted_integral():
    # no-flux walls conserve the tracer, but only in the MEASURE-
    # WEIGHTED sense: on a stretched mesh the per-cell widths differ, so
    # integrate() (weighted) telescopes to machine zero while the plain
    # unweighted cell sum does NOT (the physical-conservation subtlety)
    model = make_model(TracerCore(), HarmonicDiffusion(1.0),
                       stretched_1d(8))
    rng = np.random.default_rng(0)
    model.set_fields(b=lambda z: rng.standard_normal(z.shape))
    td = model.tendency(model.state)["b"]
    weighted = float(data(td.integrate()).ravel()[0])
    unweighted = float(np.sum(data(td)))
    assert abs(weighted) < 1e-12
    assert abs(unweighted) > 1.0  # the unweighted sum does not telescope


# ================================================================
#  Terrain following: the operator is along-coordinate (no H coupling)
# ================================================================
def test_terrain_diffusion_carries_no_h_coupling():
    # the chart factor H(x) lives in grid.metric and never enters
    # diff/measure, so the diffusion tendency is BITWISE independent of
    # the terrain: two different slopes give an identical tendency for a
    # field varying in BOTH x and sigma (pins the along-sigma semantic,
    # so any future accidental metric coupling fails here)
    def tendency(hfac):
        closure = HarmonicDiffusion(0.5, kappa_v=0.3, vertical="sigma")
        model = make_model(TracerCore(), closure, terrain_grid(8, hfac))
        model.set_fields(
            b=lambda x, sigma: np.sin(x) * np.cos(np.pi * sigma))
        return data(model.tendency(model.state)["b"])

    mild, steep = tendency(0.2), tendency(0.6)
    assert np.abs(mild).max() > 0.0  # a non-trivial operator
    assert np.abs(mild - steep).max() == 0.0


def test_terrain_sigma_only_field_matches_the_flat_column():
    # a field of sigma alone: the x-leg is zero and the sigma-leg is in
    # computational sigma (H-free), so the terrain tendency equals the
    # flat (H const) one exactly -- the x-leg carries no H(x) coupling
    def tendency(hfac):
        closure = HarmonicDiffusion(0.5, kappa_v=0.3, vertical="sigma")
        model = make_model(TracerCore(), closure, terrain_grid(8, hfac))
        model.set_fields(b=lambda x, sigma: np.cos(np.pi * sigma)
                         + 0.0 * x)
        return data(model.tendency(model.state)["b"])

    assert np.abs(tendency(0.4) - tendency(0.0)).max() == 0.0


def test_terrain_velocity_friction_binds_and_steps_finite():
    # the friction closure binds on a terrain grid: u is tangential
    # along sigma, w is the wall-normal (Inner[Dirichlet]) component
    model = make_model(
        SigmaVelCore(),
        HarmonicFriction(0.5, nu_v=0.3, vertical="sigma", slip="no"),
        terrain_grid(8, 0.2))
    model.set_fields(
        u=lambda x, sigma: np.sin(x) * np.sin(np.pi * sigma),
        w=lambda x, sigma: np.cos(x) * np.sin(np.pi * sigma))
    final = _chunk_body(model._artifacts.record, 5,
                        model._carry, model._stepper)
    assert all(bool(np.all(np.isfinite(data(f)))) for f in final.state)


def test_biharmonic_terrain_tracer_runs_finite():
    # the iterated biharmonic composes on a terrain grid (both passes
    # along-coordinate); a short run stays finite
    model = make_model(TracerCore(), BiharmonicDiffusion(1e-3),
                       terrain_grid(8, 0.2))
    model.set_fields(b=lambda x, sigma: np.cos(np.pi * sigma) * np.cos(x))
    final = _chunk_body(model._artifacts.record, 4,
                        model._carry, model._stepper)
    assert bool(np.all(np.isfinite(data(final.state[-1]))))


# ================================================================
#  vertical= naming: nu_v acts along the COLUMN coordinate (sigma)
# ================================================================
def test_nu_v_acts_along_the_named_column_coordinate():
    # nu=0, nu_v>0, vertical="sigma": the closure damps a sigma-varying
    # velocity (the sigma leg) but leaves an x-varying one untouched --
    # nu_v acts along the named column coordinate, not the physical z
    model = make_model(
        SigmaVelCore(),
        HarmonicFriction(0.0, nu_v=0.5, vertical="sigma"),
        terrain_grid(8, 0.2))
    model.set_fields(u=lambda x, sigma: np.cos(np.pi * sigma) + 0.0 * x)
    along_sigma = data(model.tendency(model.state)["u"])
    assert np.abs(along_sigma).max() > 0.0

    model.set_fields(u=lambda x, sigma: np.sin(x) + 0.0 * sigma)
    along_x = data(model.tendency(model.state)["u"])
    assert np.abs(along_x).max() == 0.0  # nu=0: no horizontal mixing


# ================================================================
#  Walls compose on a stretched column: local wall-cell width
# ================================================================
def test_no_slip_wall_drag_scales_with_the_local_cell_width():
    # a wall-parallel uniform flow feels -2 nu / dn^2 in each wall cell,
    # where dn is the LOCAL (stretched) cell width; the two walls carry
    # different widths so the drags differ -- the correction reads the
    # per-cell measure, not a single dz
    n, nu = 8, 1e-2
    model = make_model(ZVelCore(), HarmonicFriction(nu, slip="no"),
                       stretched_2d(n))
    model.set_fields(u=lambda x, z: np.ones_like(x) + 0.0 * z)
    tend = data(model.tendency(model.state)["u"])
    w = cell_widths(n)
    assert w[0] != w[-1]  # a genuinely stretched column
    np.testing.assert_allclose(tend[:, 0], -2 * nu / w[0]**2, rtol=1e-10)
    np.testing.assert_allclose(tend[:, -1], -2 * nu / w[-1]**2,
                               rtol=1e-10)
    np.testing.assert_allclose(tend[:, 1:-1], 0.0, atol=1e-12)


def test_free_slip_uniform_flow_has_no_drag_on_a_stretched_column():
    # free-slip = zero tangential wall stress: a uniform wall-parallel
    # flow feels no friction, stretched column included
    model = make_model(ZVelCore(), HarmonicFriction(1e-2, slip="free"),
                       stretched_2d(8))
    model.set_fields(u=lambda x, z: np.ones_like(x) + 0.0 * z)
    assert np.abs(data(model.tendency(model.state)["u"])).max() == 0.0


# ================================================================
#  Reverse-mode AD through a terrain-grid run (differentiability policy)
# ================================================================
def _terrain_friction_grad_loss(slip, nu, n_steps=8):
    """Build a grad-ready loss over a terrain-grid friction run."""
    model = make_model(
        SigmaVelCore(),
        HarmonicFriction(nu, nu_v=0.5 * nu, vertical="sigma", slip=slip),
        terrain_grid(8, 0.2))
    model.set_fields(
        u=lambda x, sigma: np.sin(x) * np.sin(np.pi * sigma),
        w=lambda x, sigma: np.cos(x) * np.sin(np.pi * sigma))
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
def test_friction_grad_matches_central_fd_on_terrain(slip):
    # the along-coordinate friction on a terrain (uniform-mesh + chart)
    # grid is reverse-mode differentiable end to end; grad w.r.t. nu
    # matches a central finite difference
    nu = 2e-2
    loss, x0 = _terrain_friction_grad_loss(slip, nu)
    g = float(jax.grad(loss)(x0))
    assert np.isfinite(g)
    eps = 1e-4
    fd = float((loss(x0 * (1 + eps)) - loss(x0 * (1 - eps)))
               / (2 * x0 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-4)


def _stretched_friction_grad_loss(slip, nu, n_steps=8):
    """Build a grad-ready loss over a stretched-column friction run."""
    model = make_model(
        ZVelCore(),
        HarmonicFriction(nu, nu_v=0.5 * nu, vertical="z", slip=slip),
        stretched_2d(8))
    model.set_fields(
        u=lambda x, z: np.sin(2 * np.pi * x) * np.sin(np.pi * z),
        w=lambda x, z: np.cos(2 * np.pi * x) * np.sin(np.pi * z))
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
def test_friction_grad_matches_central_fd_on_a_stretched_column(slip):
    # the spatial-layer VJP seal (divide_by_codomain_measure) makes the
    # stretched bounded column reverse-mode differentiable end to end
    nu = 2e-2
    loss, x0 = _stretched_friction_grad_loss(slip, nu)
    g = float(jax.grad(loss)(x0))
    assert np.isfinite(g)
    eps = 1e-4
    fd = float((loss(x0 * (1 + eps)) - loss(x0 * (1 - eps)))
               / (2 * x0 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-4)
