"""SmagorinskyLilly: physics limits, targeting, term predicates."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.declarations import FieldDeclaration
from fridom.model.errors import (
    AssemblyError,
    MissingParameterError,
)
from fridom.model.field_table import (
    FieldRecord,
    FieldTable,
)
from fridom.model.model import Model, _chunk_body
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.nonhydro2.modules.smagorinsky_lilly import SmagorinskyLilly
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
    MeridionalStratification,
)
from fridom.nonhydro2.params import (
    SMAG_BUOYANCY_MULTIPLIER,
    SMAG_CS,
    STRATIFICATION_N2,
)
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 8
DT = 1e-3
LZ = 2 * np.pi
DZ = LZ / N
LAM1 = (2.0 * np.sin(np.pi / N) / DZ) ** 2  # discrete Laplacian, m=1


def make_grid(periodic=True):
    return Grid(tuple(
        IntervalMesh(N, (0.0, LZ), periodic=periodic, name=name)
        for name in ("x", "y", "z")))


def make_model(n2=0.0, grid=None, stratification=None, **kwargs):
    if stratification is None:
        stratification = ConstantStratification(n2=n2)
    return Model(
        grid=grid or make_grid(),
        modules=(DynamicalCore(), stratification,
                 SmagorinskyLilly(**kwargs)),
        time_stepper=AdamBashforth(DT, order=3))


def coords():
    ax = (np.arange(N) + 0.5) * DZ
    return np.meshgrid(ax, ax, ax, indexing="ij")


def data(field):
    return np.asarray(field.data)


# ================================================================
#  Cs = 0: the closure reduces to the background coefficients
# ================================================================
def test_cs_zero_mixing_is_background_harmonic_at_discrete_rate():
    k_bg = 1e-3
    model = make_model(smagorinsky_constant=0.0,
                       background_viscosity=2e-3,
                       background_diffusivity=k_bg)
    _, _, z = coords()
    model.set_fields(b=np.sin(z))
    td = model.tendency(model.state)
    want = -k_bg * LAM1 * data(model.state["b"])
    np.testing.assert_allclose(data(td["b"]), want, atol=1e-15)


def test_cs_zero_stress_is_background_friction_at_discrete_rate():
    nu_bg = 2e-3
    model = make_model(smagorinsky_constant=0.0,
                       background_viscosity=nu_bg,
                       background_diffusivity=1e-3)
    _, _, z = coords()
    model.set_fields(u=np.sin(z))
    td = model.tendency(model.state)
    # tau_13 = nu * 0.5 * du/dz (the old tau = nu Sigma convention,
    # no factor 2): the shear mode damps at HALF the harmonic rate
    want = -0.5 * nu_bg * LAM1 * data(model.state["u"])
    np.testing.assert_allclose(data(td["u"]), want, atol=1e-15)


# ================================================================
#  The eddy viscosity is quadratic in the state (genuinely nonlinear)
# ================================================================
def test_pure_smagorinsky_stress_scales_quadratically():
    model = make_model(smagorinsky_constant=0.16,
                       background_viscosity=0.0,
                       background_diffusivity=0.0)
    _, _, z = coords()
    model.set_fields(u=0.1 * np.sin(z))
    once = data(model.tendency(model.state)["u"])
    model.set_fields(u=0.2 * np.sin(z))
    twice = data(model.tendency(model.state)["u"])
    np.testing.assert_allclose(twice, 4.0 * once, rtol=1e-12)
    assert np.abs(once).max() > 0.0


def test_smagorinsky_terms_are_nonlinear_and_linearize_drops_them():
    model = make_model(smagorinsky_constant=0.16,
                       background_viscosity=0.0,
                       background_diffusivity=0.0)
    _, _, z = coords()
    model.set_fields(u=0.1 * np.sin(z))
    with pytest.warns(UserWarning, match="coverage lint"):
        linear = fr.model.linearize(model)
    td = linear.tendency(model.state)
    assert np.abs(data(td["u"])).max() == 0.0


# ================================================================
#  Richardson damping: strong stratification kills the eddy part
# ================================================================
def test_richardson_cutoff_reduces_to_the_background():
    kwargs = {"background_viscosity": 2e-3,
              "background_diffusivity": 1e-3}
    smag = make_model(n2=1e8, smagorinsky_constant=0.16, **kwargs)
    background = make_model(n2=1e8, smagorinsky_constant=0.0,
                            **kwargs)
    _, _, z = coords()
    for model in (smag, background):
        model.set_fields(u=0.01 * np.sin(z))
    td_s = smag.tendency(smag.state)
    td_b = background.tendency(background.state)
    for name in ("u", "v", "w", "b"):
        np.testing.assert_allclose(data(td_s[name]),
                                   data(td_b[name]), atol=0.0)


# ================================================================
#  Targeting and the term split
# ================================================================
def test_mixing_targets_tracers_stress_advances_velocities():
    model = make_model()
    closure = model._carry.modules[2]
    assert closure.targets == ("b",)
    terms = {term.name: term for term in closure.tendency_terms()}
    assert terms["stress"].advances == ("u", "v", "w")
    assert terms["mixing"].advances == ("b",)
    assert not terms["stress"].linear
    assert not terms["mixing"].linear


def test_excluding_all_tracers_gives_a_friction_only_closure():
    model = make_model(exclude=("b",))
    closure = model._carry.modules[2]
    assert closure.targets == ()
    assert tuple(t.name for t in closure.tendency_terms()) == (
        "stress",)


def test_advancing_predicate_splits_stress_from_mixing():
    model = make_model(n2=0.0, smagorinsky_constant=0.16,
                       background_viscosity=0.0,
                       background_diffusivity=1e-3)
    _, _, z = coords()
    model.set_fields(u=0.1 * np.sin(z), b=0.1 * np.cos(z))
    no_mixing = model.variant(
        term_filter=~(fr.model.term_predicates.owned_by(SmagorinskyLilly)
                      & fr.model.term_predicates.advancing("b")))
    td = no_mixing.tendency(model.state)
    # the mixing term is gone (only restoring writes b; w = 0)
    assert np.abs(data(td["b"])).max() == 0.0
    # the stress term survives
    assert np.abs(data(td["u"])).max() > 0.0


def test_owned_by_closurebase_drops_the_whole_closure():
    model = make_model()
    _, _, z = coords()
    model.set_fields(u=0.1 * np.sin(z))
    with pytest.warns(UserWarning, match="coverage lint"):
        inviscid = model.variant(
            term_filter=~fr.model.term_predicates.owned_by(fr.model.closures.ClosureBase))
    td = inviscid.tendency(model.state)
    assert np.abs(data(td["u"])).max() == 0.0


# ================================================================
#  Published parameters
# ================================================================
def test_constants_are_published_parameters():
    model = make_model()
    for name in ("smagorinsky.cs", "smagorinsky.prandtl",
                 "smagorinsky.background_nu",
                 "smagorinsky.background_kappa",
                 "smagorinsky.buoyancy_multiplier"):
        assert name in model.parameters


def test_update_parameters_sweeps_the_background_viscosity():
    model = make_model(smagorinsky_constant=0.0,
                       background_viscosity=1e-3,
                       background_diffusivity=0.0)
    _, _, z = coords()
    model.set_fields(u=np.sin(z))
    before = data(model.tendency(model.state)["u"])
    model.update_parameters({"smagorinsky.background_nu": 2e-3})
    after = data(model.tendency(model.state)["u"])
    np.testing.assert_allclose(after, 2.0 * before, atol=1e-15)


def test_buoyancy_multiplier_defaults_to_inverse_prandtl():
    closure = SmagorinskyLilly(turbulent_prandtl_number=4.0)
    assert float(closure.buoyancy_multiplier) == 0.25


def test_explicit_buoyancy_multiplier_wins():
    closure = SmagorinskyLilly(turbulent_prandtl_number=4.0,
                               buoyancy_multiplier=0.5)
    assert float(closure.buoyancy_multiplier) == 0.5


# ================================================================
#  Taught assembly rejections
# ================================================================
# Free-slip / no-slip walled behaviour lives in the prefix shard
# ``test_smagorinsky_lilly_walls.py`` (oversized-module rule); walled
# grids are no longer a blanket rejection.
def test_meridional_stratification_lacks_the_constant_n2():
    grid = make_grid()
    with pytest.raises(MissingParameterError,
                       match=r"stratification\.n2"):
        make_model(
            grid=grid,
            stratification=MeridionalStratification(
                n2=lambda y: 1.0 + 0.0 * y, meridional="y"))


def test_vertical_must_be_a_velocity_axis():
    with pytest.raises(AssemblyError, match="vertical coordinate"):
        make_model(vertical="q")


def _plain_table():
    """Build a table of role-plain tracers u/v/w/b (periodic z)."""
    grid = Grid((IntervalMesh(N, (0.0, LZ), periodic=True,
                              name="z"),))
    records = [
        FieldRecord.from_declaration(
            FieldDeclaration.tracer(name), owner=0,
            owner_type="Core", grid=grid)
        for name in ("u", "v", "w", "b")]
    return FieldTable(records, grid)


def test_no_velocity_roles_is_a_taught_assembly_error():
    closure = SmagorinskyLilly()
    with pytest.raises(AssemblyError,
                       match="no PROGNOSTIC Velocity-role"):
        closure.bind(_plain_table())


def test_non_uniform_mesh_factor_is_rejected():
    grid = Grid((IntervalMesh(N, (0.0, LZ), periodic=True,
                              name="z"),))
    records = [FieldRecord.from_declaration(
        FieldDeclaration.velocity("w", "z", space=fr.spatial.Staggered("z")),
        owner=0, owner_type="Core", grid=grid)]
    records.append(FieldRecord.from_declaration(
        FieldDeclaration.tracer("b"), owner=0, owner_type="Core",
        grid=grid))
    # a duck grid whose mesh factor carries no uniform spacing
    fake_grid = SimpleNamespace(factors=(
        SimpleNamespace(names=("z",), periodic=True),))
    closure = SmagorinskyLilly()
    with pytest.raises(NotImplementedError,
                       match="uniform structured grid"):
        closure.bind(FieldTable(tuple(records), fake_grid))


# ================================================================
#  A short run: finite, energy-dissipating
# ================================================================
def test_run_dissipates_kinetic_energy():
    model = make_model(n2=1.0, smagorinsky_constant=0.16,
                       background_viscosity=1e-2,
                       background_diffusivity=1e-2)
    _, y, _ = coords()
    model.set_fields(u=0.5 * np.sin(y))
    energies = []
    for _ in range(5):
        model.advance(2)
        ekin = model.diagnostics.ekin()
        energies.append(float(np.sum(data(ekin))))
    energies = np.asarray(energies)
    assert np.isfinite(energies).all()
    assert (np.diff(energies) < 0.0).all()


def test_exported_from_the_package_namespaces():
    assert nh.SmagorinskyLilly is SmagorinskyLilly
    assert nh.modules.SmagorinskyLilly is SmagorinskyLilly


# ================================================================
#  Reverse-mode AD: the clipped-strain sqrt is guarded
# ================================================================
# The eddy viscosity is ``(Cs Delta)^2 sqrt(max(|S|^2 - beta N^2, 0))``.
# The ``max(., 0)`` clip feeds exact zeros into the sqrt, whose VJP is
# inf there, so ONE clipped cell anywhere across the field/steps turns
# ``jax.grad`` w.r.t. Cs into NaN. ``_guarded_sqrt`` guards both sqrt
# branches (subgradient 0 in the clipped cells) so the gradient is
# finite and matches finite differences.
def _smag_grad_loss(n2=0.4, cs=0.16, n_steps=10):
    """Build a grad-ready loss over a partially-clipped Smagorinsky run.

    ``n2`` is tuned so a fraction of the cells clip (the hazard fires)
    while ``Cs`` still drives the unclipped cells (a non-trivial
    gradient). Returns ``(loss, cs_leaf, n_clipped_at_t0)``.
    """
    model = make_model(n2=n2, smagorinsky_constant=cs)
    rng = np.random.default_rng(1)
    sh = (N, N, N)
    model.set_fields(u=0.2 * rng.standard_normal(sh),
                     v=0.2 * rng.standard_normal(sh),
                     w=0.2 * rng.standard_normal(sh),
                     b=0.01 * rng.standard_normal(sh))
    closure = next(m for m in model._carry.modules
                   if isinstance(m, SmagorinskyLilly))
    ctx = SimpleNamespace(params={
        SMAG_CS: cs, SMAG_BUOYANCY_MULTIPLIER: 1.0,
        STRATIFICATION_N2: n2})
    nu_s = closure._eddy_viscosity(model.state, ctx)
    n_clipped = int(np.sum(data(nu_s) == 0.0))
    record = model._artifacts.record
    carry = model._carry
    stepper = model._stepper
    leaf = closure.smagorinsky_constant
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    idx = next(i for i, lf in enumerate(leaves) if lf is leaf)

    def loss(theta):
        packed = list(leaves)
        packed[idx] = theta
        c = jax.tree_util.tree_unflatten(treedef, packed)
        final = _chunk_body(record, n_steps, c, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    return loss, jnp.asarray(leaf, dtype=jnp.float64), n_clipped


def test_smagorinsky_grad_is_finite_and_matches_fd_through_the_clip():
    loss, x0, n_clipped = _smag_grad_loss()
    # the setup must actually exercise the clip (some cells, not all)
    assert 0 < n_clipped < N ** 3
    g = float(jax.grad(loss)(x0))
    assert np.isfinite(g)
    eps = 1e-4
    fd = float((loss(x0 * (1 + eps)) - loss(x0 * (1 - eps)))
               / (2 * x0 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-4)


def test_smagorinsky_forward_mode_survives_the_custom_jvp():
    # the custom_jvp (not custom_vjp) keeps forward-mode AD alive: the
    # directional derivative must be finite and agree with reverse mode
    loss, x0, n_clipped = _smag_grad_loss()
    assert 0 < n_clipped < N ** 3
    _, jvp = jax.jvp(loss, (x0,), (jnp.asarray(1.0, dtype=jnp.float64),))
    jvp = float(jvp)
    assert np.isfinite(jvp)
    np.testing.assert_allclose(jvp, float(jax.grad(loss)(x0)), rtol=1e-6)
