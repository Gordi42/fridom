"""SmagorinskyLilly per-cell filter width (W3): the stretched-mesh Delta.

Prefix-mirrored shard of ``test_smagorinsky_lilly.py`` (oversized-module
rule): the per-cell filter width Delta = (prod_i dx_i)^{1/n} read from
``grid.measure``, which lifts the uniform-mesh restriction to stretched
columns. The full nonhydro2 model cannot assemble on a raw stretched
mesh (the spectral pressure projection has no transform on a
``MappedIntervalMesh``), so the closure is exercised directly on a
bound field table with a hand-built state — the same pattern the
targeting/rejection tests in the main mirror use. The builders are
duplicated (self-contained shard).
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import fridom as fr
from fridom.model.declarations import FieldDeclaration
from fridom.model.field_table import FieldRecord, FieldTable
from fridom.nonhydro2.modules.smagorinsky_lilly import SmagorinskyLilly
from fridom.nonhydro2.params import (
    SMAG_BACKGROUND_KAPPA,
    SMAG_BACKGROUND_NU,
    SMAG_BUOYANCY_MULTIPLIER,
    SMAG_CS,
    SMAG_PRANDTL,
    STRATIFICATION_N2,
)
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh

N = 8
L = 1.0
EXP_A = 1.5  # exponential-stretch strength (~3.7:1 cell-width ratio)


def exp_map(s):
    """Exponential stretch [0, 1] -> [0, 1], strictly increasing."""
    return (jnp.exp(EXP_A * s) - 1.0) / (jnp.exp(EXP_A) - 1.0)


def cell_widths(n=N):
    """Physical primal cell widths of the exp-stretched z column."""
    s = np.arange(n + 1) / n
    faces = (np.exp(EXP_A * s) - 1.0) / (np.exp(EXP_A) - 1.0)
    return np.diff(faces)


def stretched_grid(pz=False):
    """x/y periodic uniform, z stretched (walled by default)."""
    return Grid((
        IntervalMesh(N, (0.0, L), periodic=True, name="x"),
        IntervalMesh(N, (0.0, L), periodic=True, name="y"),
        MappedIntervalMesh(N, (0.0, L), exp_map, periodic=pz, name="z")))


def uniform_grid():
    return Grid(tuple(
        IntervalMesh(N, (0.0, L), periodic=p, name=nm)
        for nm, p in (("x", True), ("y", True), ("z", False))))


def bound_closure(grid, **kwargs):
    """Bind a SmagorinskyLilly to a hand-made u/v/w/b table on ``grid``."""
    records = [
        FieldRecord.from_declaration(
            FieldDeclaration.velocity(nm, ax, space=fr.spatial.Staggered(ax)),
            owner=0, owner_type="Core", grid=grid)
        for nm, ax in (("u", "x"), ("v", "y"), ("w", "z"))]
    records.append(FieldRecord.from_declaration(
        FieldDeclaration.tracer("b"), owner=0, owner_type="Core", grid=grid))
    table = FieldTable(tuple(records), grid)
    closure = SmagorinskyLilly(**kwargs)
    closure.bind(table)
    return closure, table


def make_state(grid, table, seed=0):
    rng = np.random.default_rng(seed)
    return {nm: grid.create_field(
        table[nm].space,
        data=0.2 * rng.standard_normal(table[nm].space.shape))
        for nm in ("u", "v", "w", "b")}


def ctx_for(cs=0.16, n2=0.0, bg_nu=0.0, bg_kappa=0.0):
    return SimpleNamespace(params={
        SMAG_CS: cs, SMAG_BUOYANCY_MULTIPLIER: 1.0, STRATIFICATION_N2: n2,
        SMAG_BACKGROUND_NU: bg_nu, SMAG_PRANDTL: 1.0,
        SMAG_BACKGROUND_KAPPA: bg_kappa})


def data(field):
    return np.asarray(field.data)


def center_field(grid):
    """Return a zero cell-centre field, the eddy-viscosity anchor."""
    _closure, table = bound_closure(grid, smagorinsky_constant=0.16)
    space = table["b"].space
    return grid.create_field(space, data=np.zeros(space.shape))


# ================================================================
#  The per-cell filter width from grid.measure
# ================================================================
def test_filter_width_field_is_the_per_cell_geometric_mean():
    grid = stretched_grid()
    delta = data(SmagorinskyLilly()._filter_width_field(center_field(grid)))
    dz = cell_widths()
    dx = L / N
    expected = (dx * dx * dz) ** (1.0 / 3.0)  # per z-cell, const in x, y
    np.testing.assert_allclose(delta[0, 0, :], expected, rtol=1e-12)
    assert delta[0, 0, :].std() > 1e-6  # genuinely varies along z


def test_filter_width_field_reduces_to_the_scalar_on_a_uniform_mesh():
    grid = uniform_grid()
    delta = data(SmagorinskyLilly()._filter_width_field(center_field(grid)))
    scalar = L / N  # cubic uniform: (dx dy dz)^(1/3) = dx = L/N
    np.testing.assert_allclose(delta, scalar, rtol=1e-12)
    assert delta.std() == 0.0  # a constant on the uniform mesh


def test_filter_width_field_halo_fallback_is_a_reach_neutral_scalar():
    # the grid-less halo tracer has no measure; the pointwise width
    # multiply is reach-neutral, so a scalar 1.0 is returned
    mock = SimpleNamespace(grid=object())
    assert SmagorinskyLilly()._filter_width_field(mock) == 1.0


# ================================================================
#  The closure runs on a stretched walled column (W1 + W2 + W3)
# ================================================================
def test_stretched_walled_closure_terms_are_finite():
    grid = stretched_grid()  # walled + stretched z
    closure, table = bound_closure(
        grid, smagorinsky_constant=0.16, background_viscosity=1e-2,
        background_diffusivity=1e-2, slip="no")
    state = make_state(grid, table)
    ctx = ctx_for(cs=0.16, n2=1.0, bg_nu=1e-2, bg_kappa=1e-2)
    stress = closure._stress(state, ctx)
    mixing = closure._mixing(state, ctx)
    for out in (stress, mixing):
        assert all(bool(np.all(np.isfinite(data(f)))) for f in out.values())


class _UnitDeltaSmag(SmagorinskyLilly):

    """SmagorinskyLilly with the filter width pinned to a unit scalar."""

    def _filter_width_field(self, anchor):  # noqa: ARG002
        return 1.0


def test_eddy_viscosity_uses_the_per_cell_filter_width():
    # nu_s = (Cs Delta)^2 |Sigma|; against a unit-Delta twin (same state,
    # same strains) the extracted |Sigma| = nu_s / (Cs Delta)^2 must
    # agree, i.e. the only difference is the per-cell Delta^2 factor.
    grid = stretched_grid()
    real, table = bound_closure(grid, smagorinsky_constant=0.16)
    state = make_state(grid, table)
    unit = _UnitDeltaSmag(smagorinsky_constant=0.16)
    unit.bind(table)
    ctx = ctx_for(cs=0.16, n2=0.0)  # no damping: nu_s = (Cs Delta)^2 |Sigma|
    nu_real = data(real._eddy_viscosity(state, ctx))
    nu_unit = data(unit._eddy_viscosity(state, ctx))
    delta = data(real._filter_width_field(state["b"]))
    # both extract the same strain norm |Sigma| once the Delta factor is
    # removed (nu_unit uses Delta = 1)
    live = nu_unit > 1e-30
    np.testing.assert_allclose(
        nu_real[live] / delta[live] ** 2, nu_unit[live], rtol=1e-11)
    # and the per-cell Delta genuinely varies (the test is non-trivial)
    assert delta.std() > 1e-6


def test_stretched_grad_is_finite_and_matches_fd():
    grid = stretched_grid()
    closure, table = bound_closure(grid, smagorinsky_constant=0.16)
    state = make_state(grid, table)

    def loss(cs):
        ctx = ctx_for(cs=cs, n2=0.4, bg_nu=1e-3)
        out = closure._stress(state, ctx)
        return sum(jnp.sum(f.data ** 2) for f in out.values())

    x0 = jnp.asarray(0.16, dtype=jnp.float64)
    g = float(jax.grad(loss)(x0))
    assert np.isfinite(g)
    eps = 1e-4
    fd = float((loss(x0 * (1 + eps)) - loss(x0 * (1 - eps)))
               / (2 * x0 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-4)
