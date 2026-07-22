"""The term-envelope legs of ``AdiabaticRamping`` (§C).

A prefix-mirrored shard of ``test_adiabatic_ramping`` (keeps the
``test_adiabatic_ramping*`` glob mapping to
``src/fridom/model/transforms/adiabatic_ramping.py``).
``envelope=True`` ramps the nonlinear terms as a whole: a
``TendencyEnvelope`` module joins the first leg's internal variant
(via ``extra_modules=``) carrying the ``rho(t)`` Ramp ``0 -> 1``
under ``"ramping.envelope"``; derived legs reuse the ordinary
Ramp-reflection machinery through ``updates=``. Endpoints are exact
by construction: ``rho = 0`` is ``fr.linearize(model)`` and
``rho = 1`` the nominal model, both asserted bitwise on the
tendency. Also covered: the four-leg matrix and involutions on the
envelope Ramp, live (never host-captured) rho delivery, the
ETDRK4-plus-envelope composition (frozen-L stays valid: enveloped
terms are ``linear=False`` by construction), the frozen-grid
extra-modules verify on an immersed composition (the ⊆ lemma), and
the autodiff regression through an enveloped run.

Self-contained per the oversized-module shard convention: the small
builders are duplicated rather than imported across test files.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.framework.utils import dtype_real, jaxify
from fridom.model import params
from fridom.model import term_predicates as terms
from fridom.model.declarations import FieldDeclaration
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.modules.advection import CenteredAdvection
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.model.modules.ramping import TendencyEnvelope
from fridom.model.parameters import ParameterDeclaration
from fridom.model.terms import term
from fridom.model.time_dependent import Ramp
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.model.time_steppers.exponential import ETDRK4
from fridom.model.time_steppers.runge_kutta import (
    ExplicitRungeKutta,
    tableaus,
)
from fridom.model.transforms.adiabatic_ramping import AdiabaticRamping
from fridom.model.transforms.norms import relative_l2
from fridom.nonhydro2.modules.core import Core as NonhydroCore
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
)
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated

N = 8
DT = 2e-3
RAMP = 0.04  # -> round(0.04 / 2e-3) = 20 ramp steps per leg
F0 = 8.0
NONLINEAR = ~terms.linear & terms.explicit


# ================================================================
#  Toy dynamical modules (rotation reads ctx-params f0)
# ================================================================
@jaxify
class RotationScaled(Module):

    """Linear rotation scaled by ``coriolis.f0`` read from ctx."""

    field_declarations = (
        FieldDeclaration("u", space=Collocated()),
        FieldDeclaration("v", space=Collocated()),
    )

    @term(name="cor", advances=("u", "v"), linear=True)
    def cor(self, state, ctx):
        f0 = ctx.params[params.CORIOLIS_F0]
        return {"u": state["v"] * f0, "v": state["u"] * (-f0)}


@partial(jaxify, dynamic=("f0",))
class F0Provider(Module):

    """A pure provider binding ``coriolis.f0``."""

    field_declarations = ()
    parameter_declarations = (
        ParameterDeclaration("coriolis.f0", attr="f0", units="1/s"),)

    def __init__(self, f0=F0):
        self.f0 = jnp.asarray(f0, dtype=dtype_real())


@jaxify
class Nonlinear(Module):

    """A nonlinear (linear=False) cross-advection of u and v."""

    field_declarations = ()

    @term(name="adv", advances=("u", "v"))
    def adv(self, state, _ctx):
        return {"u": state["u"] * state["v"] * 0.3,
                "v": -state["v"] * state["u"] * 0.3}


# ================================================================
#  Builders
# ================================================================
def make_model(modules=None, dt=DT):
    grid = Grid((IntervalMesh(N, (0.0, 1.0), periodic=True,
                              name="x"),))
    if modules is None:
        modules = (RotationScaled(), F0Provider(), Nonlinear())
    return Model(grid=grid, modules=modules,
                 time_stepper=ExplicitRungeKutta(dt,
                                                 tableau=tableaus.RK4),
                 name="toy")


def wave_state(model, u_shift=0.5):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    model.set_fields(u=np.sin(2 * np.pi * x) + u_shift,
                     v=np.cos(2 * np.pi * x) + 0.2)
    return model.state


@pytest.fixture
def model():
    return make_model()


@pytest.fixture
def z0(model):
    return wave_state(model)


@pytest.fixture
def up(model):
    return AdiabaticRamping(model, envelope=True, ramp_period=RAMP,
                            curve="exp")


# ================================================================
#  Envelope resolution and delivery
# ================================================================
def test_envelope_true_resolves_the_default_predicate(up):
    assert up.envelope is not None
    assert up.envelope.fingerprint_token() == "(~linear & explicit)"


def test_envelope_predicate_passes_verbatim(model):
    pred = terms.owned_by(Nonlinear)
    leg = AdiabaticRamping(model, envelope=pred, ramp_period=RAMP)
    assert leg.envelope is pred


def test_envelope_false_has_no_predicate(model):
    leg = AdiabaticRamping(model, ramps={"coriolis.f0": (0.0, F0)},
                           ramp_period=RAMP)
    assert leg.envelope is None
    assert params.RAMPING_ENVELOPE not in leg.ramps


def test_bad_envelope_spec_is_a_taught_type_error(model):
    with pytest.raises(TypeError, match="envelope="):
        AdiabaticRamping(model, envelope="all", ramp_period=RAMP)


def test_envelope_ramp_without_predicate_is_taught(model):
    # ramps={RAMPING_ENVELOPE: ...} on a model without the module
    # needs the enveloped-term selection (envelope=)
    with pytest.raises(TypeError, match=r"ramping\.envelope"):
        AdiabaticRamping(
            model, ramps={params.RAMPING_ENVELOPE: (0.0, 1.0)},
            ramp_period=RAMP)


def test_first_leg_binds_the_envelope_parameter(up, model):
    bound = up.model.parameters[params.RAMPING_ENVELOPE]
    # the leaf IS the Ramp object, live on the internal model — rho
    # is delivered through ctx.params at stage time, never captured
    assert isinstance(bound, Ramp)
    assert float(bound.at_time(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(bound.at_time(RAMP)) == pytest.approx(1.0, abs=1e-12)
    # the passed model never grows the module (§10.3 law 3)
    assert params.RAMPING_ENVELOPE not in model.parameters


def test_explicit_envelope_ramp_wins_over_the_default(model):
    custom = Ramp(0.0, 1.0, period=RAMP / 2, t0=RAMP / 2, curve="exp")
    leg = AdiabaticRamping(
        model, envelope=True,
        ramps={params.RAMPING_ENVELOPE: custom}, ramp_period=RAMP)
    assert leg.ramps[params.RAMPING_ENVELOPE] is custom


# ================================================================
#  Endpoint exactness (bitwise): rho=0 is linearize, rho=1 nominal
# ================================================================
def test_leg_start_is_exactly_the_linearized_model(up, model, z0):
    linear = model.variant(term_filter=terms.linear)
    enveloped = up.model.tendency(z0, t=0.0)
    reference = linear.tendency(z0)
    for name in ("u", "v"):
        assert np.array_equal(np.asarray(enveloped[name].data),
                              np.asarray(reference[name].data))


def test_leg_end_is_exactly_the_nominal_model(up, model, z0):
    enveloped = up.model.tendency(z0, t=RAMP)
    reference = model.tendency(z0)
    for name in ("u", "v"):
        assert np.array_equal(np.asarray(enveloped[name].data),
                              np.asarray(reference[name].data))
    # the endpoints differ from each other (the ramp is not inert)
    linear = model.variant(term_filter=terms.linear)
    assert not np.array_equal(
        np.asarray(reference["u"].data),
        np.asarray(linear.tendency(z0)["u"].data))


# ================================================================
#  The four-leg matrix (dt sign x rho path) and involutions
# ================================================================
def _leg(up, which):
    return {"up": up, "down": up.down,
            "up.backward": up.backward,
            "down.backward": up.down.backward}[which]


@pytest.mark.parametrize(
    ("which", "backward", "p_start", "p_end"),
    [
        pytest.param("up", False, 0.0, 1.0, id="up"),
        pytest.param("down", False, 1.0, 0.0, id="down"),
        pytest.param("up.backward", True, 1.0, 0.0, id="up.backward"),
        pytest.param("down.backward", True, 0.0, 1.0,
                     id="down.backward"),
    ],
)
def test_four_leg_matrix(up, which, backward, p_start, p_end):
    leg = _leg(up, which)
    dt = float(leg.model.parameters[params.TIME_STEP])
    assert leg.is_backward is backward
    assert (dt < 0.0) is backward
    ramp = leg.model.parameters[params.RAMPING_ENVELOPE]
    t_end = -RAMP if backward else RAMP
    assert float(ramp.at_time(0.0)) == pytest.approx(p_start,
                                                     abs=1e-9)
    assert float(ramp.at_time(t_end)) == pytest.approx(p_end,
                                                       abs=1e-9)


def test_down_and_backward_are_involutions(up):
    again = up.down.down.ramps[params.RAMPING_ENVELOPE]
    assert float(again.at_time(0.0)) == pytest.approx(0.0, abs=1e-9)
    assert float(again.at_time(RAMP)) == pytest.approx(1.0, abs=1e-9)
    rev2 = up.backward.backward
    assert not rev2.is_backward
    r = rev2.ramps[params.RAMPING_ENVELOPE]
    assert float(r.at_time(0.0)) == pytest.approx(0.0, abs=1e-9)


def test_backward_is_a_near_inverse_of_up(up, z0):
    residual = float(relative_l2(up.backward(up(z0)), z0))
    assert residual < 1e-7


def test_replace_keeps_the_envelope(up):
    # a linear-only filtered leg keeps the (now inert) envelope and
    # still builds — the empty-match refusal downgrades to a warning
    # under a term_filter (OB's backward_filter=fr.terms.linear leg)
    with pytest.warns(UserWarning, match="matches no collected"):
        replaced = up.replace(term_filter=terms.linear)
    assert replaced.envelope is not None
    assert params.RAMPING_ENVELOPE in replaced.ramps


def test_replace_with_fresh_ramps_readds_the_envelope(up):
    replaced = up.replace(ramps={params.CORIOLIS_F0: (0.0, F0)})
    ramp = replaced.ramps[params.RAMPING_ENVELOPE]
    assert float(ramp.at_time(RAMP)) == pytest.approx(1.0, abs=1e-9)
    assert params.CORIOLIS_F0 in replaced.ramps


# ================================================================
#  ETDRK4 + envelope (frozen L stays valid)
# ================================================================
def test_etdrk4_with_an_enveloped_nonlinearity_runs():
    mx = fr.spatial.meshes.IntervalMesh(
        16, (0.0, 1.0), periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(
        16, (0.0, 1.0), periodic=False, name="y")
    grid = fr.spatial.Grid((mx, my), device_ids=(0,))

    def channel(stepper, **extra):
        # today-parity spelling of the retired (csqr=1.0,
        # rossby_number=0.2, f0=1.0) channel: GravityWave scaling
        # with Fr = 0.2 and the rotation as Ro = Fr / f0
        return sw.Model(
            grid=grid, scaling=fr.scaling.GravityWave(),
            core=sw.Core(froude_number=0.2, depth=1.0),
            coriolis=sw.modules.FPlaneCoriolis(rossby_number=0.2),
            advection=True, time_stepper=stepper, **extra)

    basis = sw.eigenbasis(channel(AdamBashforth(1e-3, order=3)))
    target = channel(ETDRK4(2e-3, basis),
                     term_filter=~terms.linear)
    z0 = sw.random_state(basis, "vortical", seed=3)
    leg = AdiabaticRamping(target, envelope=True, ramp_period=0.02,
                           term_filter=~terms.linear)
    out = leg(z0)
    for name in ("u", "v", "p"):
        assert np.all(np.isfinite(np.asarray(out[name].data)))


# ================================================================
#  Frozen-grid extra-modules verify (the ⊆ lemma) on immersed
# ================================================================
def test_enveloped_variant_verifies_on_a_frozen_immersed_grid():
    two_pi = 2.0 * np.pi
    box = lambda x, y, z: (  # noqa: E731
        (x > 1.0) & (x < 5.0) & (y > 1.0) & (y < 5.0)
        & (z > 0.2) & (z < 0.8)).astype(float)
    grid = Grid(tuple(
        IntervalMesh(8, (0.0, two_pi), periodic=True, name=nm)
        for nm in ("x", "y", "z")), immersed=ImmersedDomain(box))
    parent = Model(
        grid=grid,
        modules=(NonhydroCore(family="fv"),
                 ConstantStratification(n2=0.0, family="fv"),
                 FPlaneCoriolis(f0=1.0), CenteredAdvection()),
        time_stepper=AdamBashforth(1e-2, order=3))
    # the parent froze the grid; the enveloped variant re-assembles
    # on it through the verify path (a scalar multiply is halo-
    # width-neutral, so the ⊆ lemma holds — no GridFrozenError)
    variant = parent.variant(
        extra_modules=(TendencyEnvelope(terms=NONLINEAR),))
    enveloped = [entry.key for entry
                 in variant._artifacts.schedule.kind_entries(None)
                 if entry.enveloped]
    assert enveloped  # the advection terms are wrapped


# ================================================================
#  Autodiff regression (differentiability policy)
# ================================================================
def test_grad_through_an_enveloped_run_matches_fd():
    envelope = TendencyEnvelope(terms=NONLINEAR, envelope=0.7)
    model = make_model(modules=(RotationScaled(), F0Provider(),
                                Nonlinear(), envelope))
    wave_state(model)
    run = model.propagator(wrt=(params.RAMPING_ENVELOPE,), steps=8)
    rho0 = jnp.asarray(0.7, dtype=dtype_real())

    def loss(rho):
        final = run((rho,))
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    grad = float(jax.grad(loss)(rho0))
    assert np.isfinite(grad)
    assert grad != 0.0

    def loss_f(rho):
        return float(loss(jnp.asarray(rho, dtype=dtype_real())))

    h = 1e-4 * 0.7
    fd = (loss_f(0.7 + h) - loss_f(0.7 - h)) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=1e-4)
