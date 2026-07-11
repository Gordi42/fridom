"""Tests for the StateTransform base: the central ``rest`` wiring."""
import numpy as np

from fridom.framework.utils import jaxify
from fridom.model.transforms.base import StateTransform
from fridom.model.transforms.identity import Identity
from fridom.model.transforms.info import TransformInfo
from fridom.model.transforms.shift import Shift
from fridom.model.transforms.signature import StateSignature
from fridom.spatial.fields.vector_field import VectorField

from .conftest import Scale, build_state


# ================================================================
#  A family-mapped toy projector (drops extras, like _project)
# ================================================================
@jaxify
class KeepU(StateTransform):

    """Idempotent projector on (u, v): keeps u, zeroes v.

    DROPS every other input component (the family-built payload
    shape, like the model packages' ``_project``).
    """

    def __init__(self, signature):
        self._sig = signature

    @property
    def domain(self):
        return self._sig

    @property
    def codomain(self):
        return self._sig

    @property
    def idempotent(self):
        return True

    def _evaluate(self, state):
        out = VectorField({"u": state["u"], "v": state["v"] * 0.0})
        return out, TransformInfo.EMPTY


@jaxify
class EmitOnlyU(StateTransform):

    """A payload that drops the mapped 'v' (codomain semantics)."""

    def __init__(self, signature):
        self._sig = signature

    @property
    def domain(self):
        return self._sig

    @property
    def codomain(self):
        return self._sig

    def _evaluate(self, state):
        return VectorField({"u": state["u"]}), TransformInfo.EMPTY


@jaxify
class EmitExtraQ(StateTransform):

    """A payload emitting a component the input does not carry."""

    def __init__(self, signature):
        self._sig = signature

    @property
    def domain(self):
        return self._sig

    @property
    def codomain(self):
        return self._sig

    def _evaluate(self, state):
        out = VectorField({
            "u": state["u"], "v": state["v"],
            "q": state["u"] * 2.0})
        return out, TransformInfo.EMPTY


def _sig(grid, rest="zero"):
    return StateSignature.of_prognostic(
        build_state(grid), rest=rest)


def _tracer_state(grid, names=("u", "v", "w")):
    return build_state(grid, names=names)


# ================================================================
#  Leaf wiring: rest="zero" / rest="pass"
# ================================================================
def test_rest_zero_attaches_zero_field_on_own_space(grid):
    z = _tracer_state(grid)
    out = KeepU(_sig(grid))(z)
    assert out.component_names == ("u", "v", "w")
    assert np.all(np.asarray(out["w"].data) == 0.0)
    assert (out["w"].function_space.bare
            == z["w"].function_space.bare)
    assert out["w"].metadata.name == "w"


def test_rest_pass_passes_the_input_component_through(grid):
    z = _tracer_state(grid)
    out = KeepU(_sig(grid, rest="pass"))(z)
    assert out["w"] is z["w"]


def test_state_minus_projection_carries_the_full_tracer(grid):
    # the motivating law: with rest="zero", completeness holds
    # restricted to the family and the residual carries the tracer
    z = _tracer_state(grid)
    residual = z - KeepU(_sig(grid))(z)
    assert np.allclose(np.asarray(residual["w"].data),
                       np.asarray(z["w"].data))


def test_extras_keep_their_input_position(grid):
    z = _tracer_state(grid, names=("w", "u", "v"))
    out = KeepU(_sig(grid))(z)
    assert out.component_names == ("w", "u", "v")
    assert np.all(np.asarray(out["w"].data) == 0.0)


def test_no_extras_returns_the_payload_output_object(grid):
    z = build_state(grid)
    p = KeepU(_sig(grid))
    canned, _ = p._evaluate(z)
    assert p._apply_rest(z, canned) is canned


def test_polymorphic_domain_skips_the_completion(grid):
    z = _tracer_state(grid)
    assert Identity()(z) is z


def test_payload_emitted_extra_wins_over_rest(grid):
    # Scale's payload acts on the whole state: the emitted extra is
    # never overwritten by the (fill-only) rest completion
    z = _tracer_state(grid)
    out = Scale(_sig(grid), 2.0)(z)
    assert np.allclose(np.asarray(out["w"].data),
                       2.0 * np.asarray(z["w"].data))


def test_mapped_component_dropped_by_payload_stays_dropped(grid):
    z = _tracer_state(grid)
    out = EmitOnlyU(_sig(grid))(z)
    assert out.component_names == ("u", "w")
    assert np.all(np.asarray(out["w"].data) == 0.0)


def test_payload_only_components_append_after_the_input_order(grid):
    z = _tracer_state(grid)
    out = EmitExtraQ(_sig(grid))(z)
    assert out.component_names == ("u", "v", "w", "q")


def test_shift_passes_extras_through_unchanged(grid):
    z = _tracer_state(grid)
    out = Shift(build_state(grid))(z)
    assert np.allclose(np.asarray(out["w"].data),
                       np.asarray(z["w"].data))


# ================================================================
#  The algebra: rest composes through the node structure
# ================================================================
def test_complement_carries_the_full_tracer(grid):
    z = _tracer_state(grid)
    out = KeepU(_sig(grid)).complement(z)
    assert np.allclose(np.asarray(out["w"].data),
                       np.asarray(z["w"].data))
    assert np.all(np.asarray(out["u"].data) == 0.0)
    assert np.allclose(np.asarray(out["v"].data),
                       np.asarray(z["v"].data))


def test_complement_of_pass_rest_zeroes_the_tracer(grid):
    # I - P with rest="pass": both legs carry the tracer, and the
    # pointwise difference cancels it
    z = _tracer_state(grid)
    out = KeepU(_sig(grid, rest="pass")).complement(z)
    assert np.allclose(np.asarray(out["w"].data), 0.0)


def test_sum_of_zero_rest_projections_keeps_the_tracer_zero(grid):
    z = _tracer_state(grid)
    out = (KeepU(_sig(grid)) + KeepU(_sig(grid)))(z)
    assert np.all(np.asarray(out["w"].data) == 0.0)


def test_sum_of_pass_rest_projections_doubles_the_tracer(grid):
    # pointwise-sum semantics: each summand's completed output
    # carries the tracer once, and Sum adds the outputs
    z = _tracer_state(grid)
    sig = _sig(grid, rest="pass")
    out = (KeepU(sig) + KeepU(sig))(z)
    assert np.allclose(np.asarray(out["w"].data),
                       2.0 * np.asarray(z["w"].data))


def test_compose_threads_the_rest_policies_sequentially(grid):
    z = _tracer_state(grid)
    p_zero, p_pass = (KeepU(_sig(grid)),
                      KeepU(_sig(grid, rest="pass")))
    w = np.asarray(z["w"].data)
    assert np.all(np.asarray((p_zero @ p_pass)(z)["w"].data) == 0.0)
    assert np.all(np.asarray((p_pass @ p_zero)(z)["w"].data) == 0.0)
    assert np.allclose(
        np.asarray((p_pass @ p_pass)(z)["w"].data), w)


def test_power_completes_every_iteration(grid):
    z = _tracer_state(grid)
    out = (KeepU(_sig(grid)) ** 2)(z)
    assert out.component_names == ("u", "v", "w")
    assert np.all(np.asarray(out["w"].data) == 0.0)
