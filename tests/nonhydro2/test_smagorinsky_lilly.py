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
    MissingFieldError,
)
from fridom.model.field_table import (
    FieldRecord,
    FieldTable,
)
from fridom.model.model import Model, _chunk_body
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.nonhydro2.modules.buoyancy_tracer import BuoyancyTracer
from fridom.nonhydro2.modules.core import Core
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
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh

N = 8
DT = 1e-3
LZ = 2 * np.pi
DZ = LZ / N
LAM1 = (2.0 * np.sin(np.pi / N) / DZ) ** 2  # discrete Laplacian, m=1


def make_grid(periodic=True):
    return Grid(tuple(
        IntervalMesh(N, (0.0, LZ), periodic=periodic, name=name)
        for name in ("x", "y", "z")))


def make_model(n2=0.0, grid=None, buoyancy=None, **kwargs):
    if buoyancy is None:
        buoyancy = ConstantStratification(n2=n2)
    return Model(
        grid=grid or make_grid(),
        modules=(Core(), buoyancy,
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
    # tau_13 = 2 nu * 0.5 * du/dz = nu du/dz (the standard tau = 2 nu
    # Sigma convention): the shear mode damps at the harmonic rate
    want = -nu_bg * LAM1 * data(model.state["u"])
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
# grids are no longer a blanket rejection. The finite-volume family
# lives in ``test_smagorinsky_lilly_fv.py``.
def test_meridional_stratification_is_a_taught_rejection():
    # the varying N^2(y) background is a *field*, not the constant
    # provide: refused explicitly rather than damped against zero
    grid = make_grid()
    with pytest.raises(NotImplementedError,
                       match="VARYING background stratification"):
        make_model(
            grid=grid,
            buoyancy=MeridionalStratification(
                n2=lambda y: 1.0 + 0.0 * y, meridional="y"))


def test_nondimensional_stratification_is_a_taught_rejection():
    # the internal-wave (eps/Fr)^2 background is not wired into the
    # damping, and the closure's own constants are dimensional
    with pytest.raises(NotImplementedError,
                       match=r"stratification\.froude"):
        Model(
            grid=make_grid(),
            modules=(Core(),
                     nh.FPlaneCoriolis(rossby_number=0.25),
                     ConstantStratification(froude_number=0.5),
                     SmagorinskyLilly()),
            scaling=fr.scaling.InternalWave(),
            time_stepper=AdamBashforth(DT, order=3))


def test_no_buoyancy_module_still_refuses_through_the_field_reference():
    # dropping the stratification.n2 ParameterReference must not lose
    # the refusal of a model with no buoyancy variable at all
    with pytest.raises(MissingFieldError, match=r"'b'"):
        Model(
            grid=make_grid(),
            modules=(Core(), nh.FPlaneCoriolis(f0=1.0),
                     SmagorinskyLilly()),
            time_stepper=AdamBashforth(DT, order=3))


# ================================================================
#  No background stratification at all (nh.BuoyancyTracer)
# ================================================================
def test_buoyancy_tracer_pairs_and_binds_no_background():
    model = make_model(buoyancy=BuoyancyTracer())
    closure = next(m for m in model._carry.modules
                   if isinstance(m, SmagorinskyLilly))
    assert closure._has_background_n2 is False
    # the false provide is NOT introduced (the 1/N^2 consumers still
    # refuse the model through the missing row)
    assert str(STRATIFICATION_N2) not in model.parameters


def test_buoyancy_tracer_matches_a_zero_constant_background():
    # the two legal spellings of "no background" agree bit-for-bit AS
    # ARITHMETIC: the zero background only ever enters as an exact
    # ``+ 0.0``. They are two different traces, though, and bitwise
    # claims hold only between identically compiled paths: on x86 XLA
    # contracts different mul/add pairs into FMAs in the two programs
    # (one ulp; gone under --xla_cpu_max_isa=AVX). Evaluated op by op
    # there is no fusion to differ, so the comparison stays exact.
    kwargs = {"smagorinsky_constant": 0.16,
              "background_viscosity": 1e-3,
              "background_diffusivity": 1e-3}
    tracer = make_model(buoyancy=BuoyancyTracer(), **kwargs)
    zero = make_model(n2=0.0, **kwargs)
    rng = np.random.default_rng(17)
    fields = {name: 0.3 * rng.standard_normal((N, N, N))
              for name in ("u", "v", "w", "b")}
    tracer.set_fields(**fields)
    zero.set_fields(**fields)
    with jax.disable_jit():
        td_t = tracer.tendency(tracer.state)
        td_z = zero.tendency(zero.state)
    for name in ("u", "v", "w", "b"):
        np.testing.assert_array_equal(data(td_t[name]), data(td_z[name]))
    assert np.abs(data(td_t["u"])).max() > 0.0


def test_richardson_damping_is_live_without_a_background():
    # "no background" is NOT "no damping": N^2 = d(b)/dz alone still
    # clips the eddy viscosity wherever the resolved buoyancy is stable
    model = make_model(buoyancy=BuoyancyTracer(),
                       smagorinsky_constant=0.16,
                       background_viscosity=0.0,
                       buoyancy_multiplier=1.0)
    closure = next(m for m in model._carry.modules
                   if isinstance(m, SmagorinskyLilly))
    ctx = SimpleNamespace(params={SMAG_CS: 0.16,
                                  SMAG_BUOYANCY_MULTIPLIER: 1.0})
    _, _, z = coords()
    shear = 0.4 * np.sin(z)
    viscosities = {}
    for strat in (0.0, 5.0):
        model.set_fields(u=shear, b=strat * z)
        viscosities[strat] = data(closure._eddy_viscosity(model.state, ctx))
    unstratified, stratified = viscosities[0.0], viscosities[5.0]
    assert (unstratified > 0.0).all()
    assert (stratified == 0.0).sum() > unstratified.size // 2
    assert (stratified <= unstratified + 1e-15).all()


def test_a_bare_field_table_keeps_the_background_read():
    # the closure unit-test path (a raw FieldTable, no parameter view)
    # cannot see the assembly, so the read is kept and the caller
    # supplies ctx.params[stratification.n2] itself
    closure = SmagorinskyLilly()
    closure.bind(_velocity_table())
    assert closure._has_background_n2 is True


def test_vertical_must_be_a_velocity_axis():
    with pytest.raises(AssemblyError, match="vertical coordinate"):
        make_model(vertical="q")


def _velocity_table():
    """Build a raw u/v/w/b table with the velocity roles (periodic)."""
    grid = make_grid()
    records = [
        FieldRecord.from_declaration(
            FieldDeclaration.velocity(name, axis,
                                      space=fr.spatial.Staggered(axis)),
            owner=0, owner_type="Core", grid=grid)
        for name, axis in (("u", "x"), ("v", "y"), ("w", "z"))]
    records.append(FieldRecord.from_declaration(
        FieldDeclaration.tracer("b"), owner=0, owner_type="Core",
        grid=grid))
    return FieldTable(tuple(records), grid)


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


# Non-uniform (stretched) meshes are supported (per-cell filter width
# from grid.measure); see test_smagorinsky_lilly_delta.py.


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


# ================================================================
#  Grid kinds: mapped (terrain / chart) refused, stretched served
# ================================================================
def _terrain_grid(n=8):
    """``zp = z * H(x)`` with a 20% slope -- a CoordinateMapping."""
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H},
        params={"H": lambda x: 1.0 + 0.2 * jnp.sin(x)})
    return Grid((
        IntervalMesh(n, (0.0, LZ), name="x"),
        IntervalMesh(4, (0.0, LZ), name="y"),
        IntervalMesh(n, (0.0, 1.0), name="z"),
    ), mapping=mapping)


def test_mapped_grid_is_refused_at_bind():
    # the closure derives nu_s from the grid, and on a chart every read
    # (Delta, |Sigma|, N^2) is in computational units -- refused rather
    # than run as plausible-looking wrong physics
    with pytest.raises(NotImplementedError,
                       match="does not support mapped"):
        make_model(n2=1.0, grid=_terrain_grid())


def test_mapped_rejection_names_the_alternative():
    with pytest.raises(NotImplementedError) as excinfo:
        make_model(n2=1.0, grid=_terrain_grid())
    message = str(excinfo.value)
    assert "COMPUTATIONAL units" in message
    assert "HarmonicFriction" in message


def test_stretched_mesh_factor_is_not_caught_by_the_mapped_guard():
    # the contrast that makes the guard precise: a bare
    # MappedIntervalMesh declares NO CoordinateMapping, so grid.measure
    # IS the physical cell spacing and the closure binds. (The per-cell
    # Delta physics is test_smagorinsky_lilly_delta.py; the full nh2
    # model cannot assemble on a raw stretched mesh, so bind on a
    # hand-built table.)
    def wavy(s):
        """Smooth wavy stretching of the unit computational interval."""
        return LZ * (s + 0.1 * jnp.sin(2.0 * jnp.pi * s) / (2.0 * jnp.pi))

    grid = Grid((
        IntervalMesh(N, (0.0, LZ), periodic=True, name="x"),
        IntervalMesh(N, (0.0, LZ), periodic=True, name="y"),
        MappedIntervalMesh(N, (0.0, LZ), wavy, periodic=True, name="z"),
    ))
    assert grid.mapping is None
    records = [
        FieldRecord.from_declaration(
            FieldDeclaration.velocity(nm, ax,
                                      space=fr.spatial.Staggered(ax)),
            owner=0, owner_type="Core", grid=grid)
        for nm, ax in (("u", "x"), ("v", "y"), ("w", "z"))]
    records.append(FieldRecord.from_declaration(
        FieldDeclaration.tracer("b"), owner=0, owner_type="Core",
        grid=grid))
    SmagorinskyLilly().bind(FieldTable(tuple(records), grid))
