r"""The generic Source module: patterns, phase convention, autodiff.

On a zero state with advection off, every other nonhydro term
vanishes, so the (unconstrained) tendency of a forced variable IS the
source term ``A Re[Q e^{-i(2 pi f t + phi)}]`` — an exact check
against the pattern sampled at the variable's own nodes and the
phase convention (``phase = -pi/2`` is a sine).
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.errors import (
    FieldCollisionError,
    MissingFieldError,
    ParameterCollisionError,
)
from fridom.model.model import _chunk_body
from fridom.model.modules.source import Source
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.gaussian_wave_maker import GaussianWaveMaker
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 8
LENGTH = 2 * np.pi
DT = 1e-3
FREQ = 2.0
AMP = 0.3


# ================================================================
#  Fixtures
# ================================================================
def make_grid(walled=()):
    return Grid(tuple(
        IntervalMesh(N, (0.0, LENGTH), periodic=name not in walled,
                     name=name)
        for name in ("x", "y", "z")))


def make_model(*makers, grid=None, walled=()):
    return nh.Model(
        grid=make_grid(walled) if grid is None else grid,
        core=nh.Core(),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        buoyancy=nh.ConstantStratification(n2=1.0),
        advection=False,
        modules_extra=makers)


def harmonic(*, pattern, phase=0.0, amplitude=AMP, frequency=FREQ):
    """Build a Source with a Harmonic law (the blessed spelling)."""
    return Source(
        "wm", pattern,
        law=fr.model.Harmonic(amplitude, frequency, phase=phase))


def nodes(model, variable, axis):
    """Return the variable's own node positions along ``axis``."""
    space = model.state[variable].function_space
    return np.asarray(model.grid.evaluation_nodes(space, axis).data)


def gauss(x, pos, width):
    return np.exp(-((x - pos) ** 2) / width**2)


def tendency(model, variable, t):
    return np.asarray(
        model.tendency(model.state, t=t, constraints=False)[variable].data)


# ================================================================
#  Construction-time validation
# ================================================================
def test_label_must_be_a_non_empty_dot_free_string():
    pat = {"u": fr.model.gaussian(pos={"x": 1.0}, width=1.0)}
    law = fr.model.Harmonic(AMP, FREQ)
    with pytest.raises(TypeError, match="non-empty string"):
        Source("", pat, law)
    with pytest.raises(ValueError, match="dot"):
        Source("a.b", pat, law)


def test_pattern_must_be_a_non_empty_mapping_of_valid_values():
    law = fr.model.Harmonic(AMP, FREQ)
    with pytest.raises(ValueError, match="at least one forced"):
        Source("wm", {}, law)
    with pytest.raises(TypeError, match="callable or a ScalarField"):
        Source("wm", {"u": 3.0}, law)
    with pytest.raises(TypeError, match="prognostic variable names"):
        Source("wm", {3: fr.model.gaussian(pos={"x": 1.0}, width=1.0)},
               law)


def test_callable_naming_no_coordinate_is_rejected_at_bind():
    # an empty-signature callable names no coordinate (classified real);
    # bind refuses it with the taught envelope error
    maker = Source("wm", {"u": lambda: 1.0},
                   law=fr.model.Harmonic(AMP, FREQ))
    with pytest.raises(ValueError, match="names no coordinate"):
        make_model(maker)


def test_unprobeable_callable_is_treated_as_real():
    # a callable that raises when probed at a scalar is classified real
    # (bind re-validates its coordinates); construction still succeeds
    def picky(x):  # noqa: ARG001
        raise RuntimeError("cannot probe at a scalar")

    maker = Source("wm", {"u": picky}, law=fr.model.Harmonic(AMP, FREQ))
    assert "u" not in maker._imag_field


def test_law_must_be_a_time_dependent():
    pat = {"u": fr.model.gaussian(pos={"x": 1.0}, width=1.0)}
    with pytest.raises(TypeError, match="must be a TimeDependent"):
        Source("wm", pat, law=lambda t: t)


# ================================================================
#  Real callable pattern: A cos(2 pi f t + phi) * mask
# ================================================================
def test_real_callable_pattern_matches_the_phase_law():
    phase = 0.4
    maker = harmonic(
        pattern={"u": fr.model.gaussian(
            pos={"x": np.pi, "y": 2.0}, width={"x": 0.5, "y": 1.0})},
        phase=phase)
    model = make_model(maker)
    x = nodes(model, "u", "x")
    y = nodes(model, "u", "y")
    mask = gauss(x, np.pi, 0.5) * gauss(y, 2.0, 1.0)
    for t in (0.03, 0.09, 0.17):
        expected = AMP * np.cos(2 * np.pi * FREQ * t + phase) * mask
        got = tendency(model, "u", t)
        np.testing.assert_allclose(
            got, np.broadcast_to(expected, got.shape), atol=1e-14)


def test_scalar_width_broadcasts_over_named_axes():
    maker = harmonic(
        pattern={"u": fr.model.gaussian(pos={"x": np.pi}, width=0.5)},
        phase=-np.pi / 2)
    model = make_model(maker)
    x = nodes(model, "u", "x")
    t = 1.0 / (4.0 * FREQ)  # sin(2 pi f t) = 1
    got = tendency(model, "u", t)
    np.testing.assert_allclose(
        got, np.broadcast_to(AMP * gauss(x, np.pi, 0.5), got.shape),
        atol=1e-14)


# ================================================================
#  Parity gate: Source(Harmonic(A, f, phase=-pi/2)) == GaussianWaveMaker
# ================================================================
@pytest.mark.parametrize("variable", ["u", "b"])
def test_parity_with_gaussian_wave_maker(variable):
    pos = {"x": np.pi} if variable == "u" else {"z": np.pi}
    width = {"x": 0.5} if variable == "u" else {"z": 0.7}
    src = Source(
        "wm", {variable: fr.model.gaussian(pos=pos, width=width)},
        law=fr.model.Harmonic(AMP, FREQ, phase=-np.pi / 2))
    gwm = GaussianWaveMaker(pos, width, FREQ, AMP, variable=variable)
    model_s = make_model(src)
    model_g = make_model(gwm)
    for t in (0.03, 0.09, 1.0 / (4.0 * FREQ)):
        np.testing.assert_allclose(
            tendency(model_s, variable, t),
            tendency(model_g, variable, t), atol=1e-15, rtol=0.0)


def test_parity_on_a_walled_grid_forces_w_at_its_faces():
    src = Source(
        "wm", {"w": fr.model.gaussian(pos={"z": np.pi}, width=0.7)},
        law=fr.model.Harmonic(AMP, FREQ, phase=-np.pi / 2))
    gwm = GaussianWaveMaker({"z": np.pi}, {"z": 0.7}, FREQ, AMP,
                            variable="w")
    ms = make_model(src, walled=("z",))
    mg = make_model(gwm, walled=("z",))
    t = 1.0 / (4.0 * FREQ)
    np.testing.assert_allclose(
        tendency(ms, "w", t), tendency(mg, "w", t), atol=1e-15)


# ================================================================
#  Complex pattern: the quadrature A[cos Re Q + sin Im Q]
# ================================================================
def _complex_mask(x):
    return jnp.exp(-((x - np.pi) ** 2) / 0.5**2) * (1.0 + 2.0j)


_complex_mask.__signature__ = inspect.Signature(
    [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)])


def test_complex_callable_pattern_expands_into_a_quadrature():
    phase = -np.pi / 2
    maker = harmonic(pattern={"u": _complex_mask}, phase=phase)
    model = make_model(maker)
    x = nodes(model, "u", "x")
    mask = gauss(x, np.pi, 0.5)
    for t in (0.03, 0.07, 0.11):
        theta = 2 * np.pi * FREQ * t + phase
        expected = AMP * (np.cos(theta) * mask
                          + np.sin(theta) * 2.0 * mask)
        got = tendency(model, "u", t)
        np.testing.assert_allclose(
            got, np.broadcast_to(expected, got.shape), atol=1e-14)


def test_complex_scalar_field_pattern_expands_into_a_quadrature():
    grid = make_grid()
    probe = make_model(grid=grid)  # freeze the grid; read b's own space
    bspace = probe.state["b"].function_space.bare
    rng = np.random.default_rng(2)
    re = rng.standard_normal(bspace.shape)
    im = rng.standard_normal(bspace.shape)
    field = grid.create_field(
        bspace.as_complex(), data=jnp.asarray(re + 1j * im))
    phase = 0.3
    model = make_model(
        Source("wm", {"b": field},
               law=fr.model.Harmonic(AMP, FREQ, phase=phase)),
        grid=grid)
    for t in (0.02, 0.08):
        theta = 2 * np.pi * FREQ * t + phase
        expected = AMP * (np.cos(theta) * re + np.sin(theta) * im)
        np.testing.assert_allclose(tendency(model, "b", t), expected,
                                   atol=1e-13)


# ================================================================
#  Field-valued patterns land on the variable's own space
# ================================================================
def test_field_valued_pattern_on_the_variables_own_space():
    grid = make_grid()
    probe = make_model(grid=grid)
    rng = np.random.default_rng(5)
    data = jnp.asarray(rng.standard_normal(
        probe.state["b"].function_space.bare.shape))
    field = probe.state["b"].with_data(data)
    model = make_model(
        Source("wm", {"b": field}, law=fr.model.Harmonic(AMP, FREQ)),
        grid=grid)
    t = 0.05
    np.testing.assert_allclose(
        tendency(model, "b", t),
        AMP * np.cos(2 * np.pi * FREQ * t) * np.asarray(data),
        atol=1e-13)


def test_field_valued_pattern_on_the_wrong_space_is_rejected():
    grid = make_grid()
    probe = make_model(grid=grid)
    # a field on u's staggered space cannot force the collocated b
    wrong = probe.state["u"].with_data(
        jnp.zeros(probe.state["u"].function_space.bare.shape))
    with pytest.raises(ValueError, match="own space"):
        make_model(
            Source("wm", {"b": wrong}, law=fr.model.Harmonic(AMP, FREQ)),
            grid=grid)


# ================================================================
#  Bind-time taught errors
# ================================================================
def test_unknown_variable_is_a_missing_field_error():
    # the pattern field adopts the forced variable's space (LikeField),
    # so a non-existent target surfaces as the taught resolution error
    maker = harmonic(pattern={"nope": fr.model.gaussian(
        pos={"x": 1.0}, width=1.0)})
    with pytest.raises(MissingFieldError, match="no module declares 'nope'"):
        make_model(maker)


def test_non_prognostic_target_is_rejected():
    maker = harmonic(pattern={"p": fr.model.gaussian(
        pos={"x": 1.0}, width=1.0)})
    with pytest.raises(ValueError, match="only PROGNOSTIC"):
        make_model(maker)


def test_unknown_envelope_coordinate_is_rejected():
    maker = harmonic(pattern={"u": fr.model.gaussian(
        pos={"q": 1.0}, width=1.0)})
    with pytest.raises(ValueError, match="grid does not have"):
        make_model(maker)


# ================================================================
#  Distinct labels coexist; duplicate labels collide
# ================================================================
def test_two_labelled_sources_coexist():
    a = Source("phase_a", {"u": fr.model.gaussian(
        pos={"x": 1.0}, width=1.0)},
        law=fr.model.Harmonic(AMP, FREQ, phase=-np.pi / 2))
    b = Source("phase_b", {"v": fr.model.gaussian(
        pos={"y": 1.0}, width=1.0)},
        law=fr.model.Harmonic(AMP, FREQ))
    model = make_model(a, b)
    provided = [n for n in model._artifacts.binding_table.names
                if n.startswith("source.")]
    assert "source.phase_a.amplitude" in provided
    assert "source.phase_b.frequency" in provided


def test_duplicate_labels_collide_in_the_field_table():
    a = harmonic(pattern={"u": fr.model.gaussian(
        pos={"x": 1.0}, width=1.0)})
    b = harmonic(pattern={"u": fr.model.gaussian(
        pos={"x": 1.0}, width=1.0)})
    with pytest.raises(FieldCollisionError):
        make_model(a, b)


def test_duplicate_labels_forcing_distinct_fields_collide_on_parameters():
    a = harmonic(pattern={"u": fr.model.gaussian(
        pos={"x": 1.0}, width=1.0)})
    b = Source("wm", {"v": fr.model.gaussian(pos={"y": 1.0}, width=1.0)},
               law=fr.model.Harmonic(AMP, FREQ))
    with pytest.raises(ParameterCollisionError):
        make_model(a, b)


# ================================================================
#  update_parameters sweeps the Harmonic leaves (no re-assembly)
# ================================================================
def test_update_parameters_sweeps_amplitude_and_frequency():
    maker = harmonic(
        pattern={"u": fr.model.gaussian(pos={"x": np.pi}, width=0.5)},
        phase=-np.pi / 2)
    model = make_model(maker)
    t = 1.0 / (4.0 * FREQ)
    before = tendency(model, "u", t)
    model.update_parameters({"source.wm.amplitude": 2 * AMP})
    np.testing.assert_allclose(tendency(model, "u", t), 2.0 * before,
                               atol=1e-14)
    # halving the frequency turns sin(pi/2) into sin(pi/4)
    model.update_parameters({"source.wm.frequency": FREQ / 2})
    np.testing.assert_allclose(
        tendency(model, "u", t),
        2.0 * before * np.sin(np.pi / 4), atol=1e-14)


# ================================================================
#  Generic (escape-hatch) law: follows the law, publishes nothing
# ================================================================
def test_generic_law_follows_the_law_and_publishes_no_parameters():
    law = fr.model.TimeFunction(lambda t, w: jnp.sin(w * t), params=(3.0,))
    maker = Source(
        "gl", {"u": fr.model.gaussian(pos={"x": np.pi}, width=0.5)}, law)
    model = make_model(maker)
    assert not [n for n in model._artifacts.binding_table.names
                if n.startswith("source.")]
    x = nodes(model, "u", "x")
    mask = gauss(x, np.pi, 0.5)
    for t in (0.03, 0.09):
        got = tendency(model, "u", t)
        np.testing.assert_allclose(
            got, np.broadcast_to(np.sin(3.0 * t) * mask, got.shape),
            atol=1e-14)


def test_complex_pattern_with_a_generic_law_is_rejected():
    law = fr.model.TimeFunction(jnp.sin)
    with pytest.raises(TypeError, match="quadrature expansion"):
        Source("wm", {"u": _complex_mask}, law)


# ================================================================
#  Ramped amplitude: evaluated at the stage time
# ================================================================
def test_ramped_amplitude_evaluates_at_stage_time():
    ramp = fr.model.Ramp(0.0, AMP, period=0.1)
    maker = Source(
        "wm", {"u": fr.model.gaussian(pos={"x": np.pi}, width=0.5)},
        law=fr.model.Harmonic(ramp, FREQ, phase=-np.pi / 2))
    model = make_model(maker)
    x = nodes(model, "u", "x")
    mask = gauss(x, np.pi, 0.5)
    for t in (0.05, 0.15):
        expected = float(ramp.at_time(t)) * np.sin(
            2 * np.pi * FREQ * t) * mask
        got = tendency(model, "u", t)
        np.testing.assert_allclose(
            got, np.broadcast_to(expected, got.shape), atol=1e-14)


# ================================================================
#  Autodiff regression (differentiability policy) — via _chunk_body
# ================================================================
# The public Model.propagator refuses source.<label>.* here: the
# Source module materializes AUXILIARY pattern fields at assembly, and
# the propagator's per-owner "materialized-owner parameter" guard
# (model.py) refuses every scalar leaf of such an owner (the
# GaussianWaveMaker inherits the same limitation). The pattern does not
# depend on amplitude/frequency, so the gradient IS well-defined — the
# differentiability-policy regression differentiates the pure kernel
# _chunk_body directly (AGENTS.md permits either surface), splicing the
# Source's amplitude / frequency leaf by identity.
def _leaf_loss(record, carry, stepper, leaf, steps=10):
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        spliced = jax.tree_util.tree_unflatten(treedef, new)
        final = _chunk_body(record, steps, spliced, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    return loss


def _central_fd(loss, x0, eps=1e-4):
    h = eps * (abs(float(x0)) if float(x0) != 0.0 else 1.0)
    return (float(loss(x0 + h)) - float(loss(x0 - h))) / (2.0 * h)


@pytest.mark.parametrize("attr", ["amplitude", "frequency"])
def test_grad_wrt_harmonic_leaf_matches_fd(attr):
    maker = Source(
        "wm", {"u": fr.model.gaussian(pos={"x": np.pi}, width=0.5)},
        law=fr.model.Harmonic(AMP, FREQ, phase=-np.pi / 2))
    model = make_model(maker)
    rng = np.random.default_rng(0)
    model.set_fields(u=rng.standard_normal((N, N, N)))
    record, carry, stepper = (
        model._artifacts.record, model._carry, model._stepper)
    source = next(m for m in carry.modules if isinstance(m, Source))
    leaf = getattr(source, attr)
    loss = _leaf_loss(record, carry, stepper, leaf, steps=10)

    grad = float(jax.grad(loss)(leaf))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    assert grad == pytest.approx(_central_fd(loss, leaf), rel=1e-4)
