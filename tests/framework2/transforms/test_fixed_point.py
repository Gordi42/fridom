"""Tests for the FixedPoint iteration combinator."""
import jax
import jax.numpy as jnp
import pytest

from fridom.framework2.transforms.errors import (
    FixedPointDivergenceError,
    SignatureMismatchError,
    TraceError,
)
from fridom.framework2.transforms.fixed_point import FixedPoint
from fridom.framework2.transforms.identity import Identity
from fridom.framework2.transforms.signature import StateSignature

from .conftest import Scale, build_state, make_cross


def _abs_norm(a, b):
    """Absolute (non-relative) update norm — makes scaling diverge."""
    return float(jnp.sum(jnp.abs((a - b)["u"].data)))


def _equal(a, b):
    return all(jnp.array_equal(a[n].data, b[n].data)
               for n in a.component_names)


# ================================================================
#  Convergence
# ================================================================
def test_tol_convergence_on_fixed_point(state, sig):
    # Identity: the first update is exactly zero -> tol met at iter 1
    fp = FixedPoint(Identity(sig), tol=1e-9, max_it=5)
    out, info = fp.call_with_info(state)
    assert _equal(out, state)
    assert info.iterations == 1
    assert info.stopped_by == "tol"
    assert info.errors == (0.0,)


def test_max_it_stop(state, sig):
    fp = FixedPoint(Scale(sig, 0.5), tol=1e-12, max_it=3)
    _, info = fp.call_with_info(state)
    assert info.iterations == 3
    assert info.stopped_by == "max_it"
    assert len(info.errors) == 3
    assert len(info.children) == 3


def test_max_it_zero_returns_input(state, sig):
    fp = FixedPoint(Scale(sig, 2.0), max_it=0)
    out, info = fp.call_with_info(state)
    assert _equal(out, state)
    assert info.iterations == 0
    assert info.errors == ()


def test_tol_zero_runs_all_iterations(state, sig):
    fp = FixedPoint(Identity(sig), tol=0.0, max_it=4)
    _, info = fp.call_with_info(state)
    assert info.iterations == 4  # never early-stops on tol==0


# ================================================================
#  Divergence policies
# ================================================================
def test_divergence_stop_best(state, sig):
    fp = FixedPoint(Scale(sig, 3.0), norm=_abs_norm, tol=0.0,
                    max_it=4, on_divergence="stop_best")
    out, info = fp.call_with_info(state)
    assert info.stopped_by == "divergence"
    # the best (argmin-error) iterate is the first application (3*s)
    assert _equal(out, Scale(sig, 3.0)(state))
    assert info.returned_iteration == 1


def test_divergence_raise(state, sig):
    fp = FixedPoint(Scale(sig, 3.0), norm=_abs_norm, tol=0.0,
                    max_it=4, on_divergence="raise")
    with pytest.raises(FixedPointDivergenceError, match="diverged"):
        fp(state)


def test_divergence_ignore_runs_to_max_it(state, sig):
    fp = FixedPoint(Scale(sig, 3.0), norm=_abs_norm, tol=0.0,
                    max_it=3, on_divergence="ignore")
    _, info = fp.call_with_info(state)
    assert info.iterations == 3
    assert info.stopped_by == "max_it"


# ================================================================
#  Factory form
# ================================================================
def test_factory_form(state, sig):
    calls = []

    def factory(iterate):
        calls.append(iterate)
        return Scale(sig, 0.5)

    fp = FixedPoint(factory, tol=1e-12, max_it=2)
    _, info = fp.call_with_info(state)
    assert len(calls) == 2  # evaluated per iteration
    assert info.iterations == 2
    assert fp.domain is None  # polymorphic until called
    assert fp.codomain is None


def test_wrapped_form_reports_signature(sig):
    fp = FixedPoint(Scale(sig, 0.5))
    assert fp.domain == sig
    assert fp.codomain == sig


def test_factory_producing_non_endo_raises(state, other_grid):
    sig = StateSignature.of_prognostic(state)
    other = StateSignature.of_prognostic(build_state(other_grid))
    fp = FixedPoint(lambda _s: make_cross(sig, other), max_it=2)
    with pytest.raises(SignatureMismatchError, match="endo"):
        fp(state)


# ================================================================
#  Construction validation and the trace guard
# ================================================================
def test_rejects_negative_max_it(sig):
    with pytest.raises(ValueError, match="max_it"):
        FixedPoint(Scale(sig, 2.0), max_it=-1)


def test_rejects_bad_divergence_policy(sig):
    with pytest.raises(ValueError, match="on_divergence"):
        FixedPoint(Scale(sig, 2.0), on_divergence="explode")


def test_rejects_non_endo_wrapped(state, other_grid):
    sig = StateSignature.of_prognostic(state)
    other = StateSignature.of_prognostic(build_state(other_grid))
    with pytest.raises(SignatureMismatchError, match="endo"):
        FixedPoint(make_cross(sig, other))


def test_trace_guard_under_jit(state, sig):
    fp = FixedPoint(Scale(sig, 0.5), max_it=1)
    assert fp.traceable is False
    with pytest.raises(TraceError, match="Tier-2"):
        jax.jit(fp)(state)


def test_on_iteration_observer(state, sig):
    seen = []
    fp = FixedPoint(Scale(sig, 0.5), tol=1e-12, max_it=3,
                    on_iteration=seen.append)
    fp(state)
    assert len(seen) == 3
    assert seen[0].steps_total == 3


def test_cost(sig):
    fp = FixedPoint(Scale(sig, 0.5), max_it=3)
    cost = fp.cost()
    assert cost.upper_bound is True
    assert cost.model_steps == 0  # Tier-1 inner
    factory_fp = FixedPoint(lambda _s: Scale(sig, 0.5), max_it=3)
    assert factory_fp.cost().upper_bound is True


def test_repr(sig):
    assert "FixedPoint" in repr(FixedPoint(Scale(sig, 0.5)))
    assert "factory" in repr(FixedPoint(lambda _s: Scale(sig, 0.5)))
