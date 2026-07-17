"""Tests for fridom.model.time_dependent (Ramp & co)."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.model.time_dependent import (
    Ramp,
    TimeDependent,
    TimeFunction,
    TimeSeries,
    resolve_at,
)


# ================================================================
#  Reference implementation (host-side, pure python)
# ================================================================
def _clip01(s):
    return min(max(s, 0.0), 1.0)


def _ramp_ref(t, v0, v1, t0, period, shape=lambda s: s):
    return v0 + (v1 - v0) * shape(_clip01((t - t0) / period))


def _cosine_shape(s):
    return 0.5 * (1.0 - math.cos(math.pi * s))


def _cubic(s):
    # asymmetric shape with shape(0)=0, shape(1)=1: exercises the
    # reversed() law beyond what symmetric curves can detect
    return s**3


# ================================================================
#  Ramp evaluation
# ================================================================
def test_linear_ramp_values():
    ramp = Ramp(1.0, 3.0, period=10.0)
    for t in (-5.0, 0.0, 2.5, 5.0, 10.0, 20.0):
        np.testing.assert_allclose(
            ramp(t), _ramp_ref(t, 1.0, 3.0, 0.0, 10.0))


@pytest.mark.parametrize("curve", ["linear", "cosine", "exp"])
def test_named_curves_hit_the_endpoints(curve):
    ramp = Ramp(2.0, 5.0, period=4.0, t0=1.0, curve=curve)
    np.testing.assert_allclose(ramp(1.0), 2.0)
    np.testing.assert_allclose(ramp(5.0), 5.0)
    # before/after the window: clipped to the endpoint values
    np.testing.assert_allclose(ramp(-10.0), 2.0)
    np.testing.assert_allclose(ramp(50.0), 5.0)
    # inside the window: strictly between
    mid = float(ramp(3.0))
    assert 2.0 < mid < 5.0


def test_cosine_ramp_values():
    ramp = Ramp(0.0, 1.0, period=8.0, t0=-2.0, curve="cosine")
    for t in (-3.0, -2.0, 0.0, 2.0, 6.0, 9.0):
        np.testing.assert_allclose(
            ramp(t), _ramp_ref(t, 0.0, 1.0, -2.0, 8.0, _cosine_shape))


def test_custom_callable_curve():
    ramp = Ramp(1.0, 2.0, period=5.0, curve=_cubic)
    for t in (-1.0, 0.0, 2.0, 5.0, 7.0):
        np.testing.assert_allclose(
            ramp(t), _ramp_ref(t, 1.0, 2.0, 0.0, 5.0, _cubic))


def test_unknown_named_curve_raises():
    with pytest.raises(ValueError, match="unknown named curve"):
        Ramp(0.0, 1.0, period=1.0, curve="quadratic")


def test_at_time_matches_call():
    ramp = Ramp(0.0, 1.0, period=3.0, curve="cosine")
    np.testing.assert_allclose(ramp.at_time(1.5), ramp(1.5))


def test_repr_round_trips_the_spec():
    text = repr(Ramp(0.0, 0.1, period=3600.0, curve="cosine"))
    assert text == "Ramp(0.0, 0.1, period=3600.0, t0=0.0, curve='cosine')"
    assert "curve=_cubic" in repr(
        Ramp(0.0, 1.0, period=1.0, curve=_cubic))


# ================================================================
#  Signed times: backward windows and reversed()
# ================================================================
def test_backward_window_signed_t0():
    # a backward Ramp spans [-T, 0] via t0=-T (02_rules V-S2)
    ramp = Ramp(2.0, 5.0, period=4.0, t0=-4.0)
    np.testing.assert_allclose(ramp(-4.0), 2.0)
    np.testing.assert_allclose(ramp(-2.0), 3.5)
    np.testing.assert_allclose(ramp(0.0), 5.0)
    # clipped outside the window
    np.testing.assert_allclose(ramp(-6.0), 2.0)
    np.testing.assert_allclose(ramp(1.0), 5.0)


def test_reversed_reflects_the_window_across_zero():
    ramp = Ramp(0.0, 1.0, period=6.0)          # forward: [0, 6]
    back = ramp.reversed()                     # backward: [-6, 0]
    np.testing.assert_allclose(back(0.0), 1.0)
    np.testing.assert_allclose(back(-6.0), 0.0)
    # the naive value-endpoint swap over [0, T] would be constant
    # for t <= 0; the reflected window actually varies there
    assert 0.0 < float(back(-3.0)) < 1.0


def test_reversed_retrace_law_on_an_asymmetric_curve():
    # r.reversed()(-s) == r(2*t0 + period - s) for EVERY curve
    # (backward progress s sees the forward value at remaining time)
    v0, v1, t0, period = 0.5, 2.5, 3.0, 7.0
    ramp = Ramp(v0, v1, period=period, t0=t0, curve=_cubic)
    back = ramp.reversed()
    for s in (-2.0, 0.0, 1.3, 3.5, 4.9, 7.0, 9.2):
        np.testing.assert_allclose(
            back(-s), ramp(2.0 * t0 + period - s), rtol=1e-12)


def test_reversed_twice_is_the_original_window():
    ramp = Ramp(1.0, 4.0, period=5.0, t0=2.0, curve="exp")
    twice = ramp.reversed().reversed()
    for t in (-1.0, 2.0, 4.5, 7.0, 10.0):
        np.testing.assert_allclose(twice(t), ramp(t), rtol=1e-12)


def test_reversed_keeps_values_and_curve():
    ramp = Ramp(1.0, 4.0, period=5.0, t0=2.0, curve="cosine")
    back = ramp.reversed()
    np.testing.assert_allclose(back.v0, ramp.v0)
    np.testing.assert_allclose(back.v1, ramp.v1)
    np.testing.assert_allclose(back.period, ramp.period)
    np.testing.assert_allclose(back.t0, -(2.0 + 5.0))
    assert "cosine" in repr(back)


# ================================================================
#  resolve_at (identity on plain scalars is load-bearing)
# ================================================================
def test_resolve_at_is_the_identity_on_plain_values():
    for value in (3.0, 7, 2.0 + 1.0j, np.float64(1.5)):
        assert resolve_at(value, 0.3) == value
    marker = object()  # identity, not just equality
    assert resolve_at(marker, 0.0) is marker


def test_resolve_at_evaluates_time_dependent_values():
    ramp = Ramp(0.0, 1.0, period=2.0)
    np.testing.assert_allclose(resolve_at(ramp, 0.5), ramp(0.5))


def test_resolve_at_inside_jit():
    ramp = Ramp(0.0, 1.0, period=2.0)

    @jax.jit
    def consume(value, t):
        # the universal consumer idiom: Ramp-able slot, zero changes
        return resolve_at(value, t) + resolve_at(2.0, t)

    np.testing.assert_allclose(
        consume(ramp, jnp.asarray(0.5)), ramp(0.5) + 2.0)


# ================================================================
#  Scalar composition (derived TimeDependent nodes)
# ================================================================
def test_scalar_composition_forms():
    ramp = Ramp(1.0, 3.0, period=10.0, curve="cosine")
    samples = (-1.0, 0.0, 4.0, 10.0, 12.0)
    cases = [
        (2.0 * ramp + 1.0, lambda v: 2.0 * v + 1.0),
        (ramp * 3.0, lambda v: 3.0 * v),
        (1.0 + ramp, lambda v: 1.0 + v),
        (ramp - 0.5, lambda v: v - 0.5),
        (1.0 - ramp, lambda v: 1.0 - v),
        (-ramp, lambda v: -v),
        (1.0j * ramp, lambda v: 1.0j * v),
    ]
    for derived, expected in cases:
        assert isinstance(derived, TimeDependent)
        for t in samples:
            np.testing.assert_allclose(
                derived(t), expected(ramp(t)), rtol=1e-12)


def test_no_field_or_array_arithmetic():
    # a consumer forgetting resolve_at must fail loudly
    ramp = Ramp(0.0, 1.0, period=1.0)

    class FieldLike:
        pass

    with pytest.raises(TypeError):
        ramp * FieldLike()
    with pytest.raises(TypeError):
        ramp + FieldLike()
    with pytest.raises(TypeError):
        FieldLike() - ramp


# ================================================================
#  The jit contract: static/dynamic split, scan-carry usability
# ================================================================
def test_endpoint_and_timing_sweeps_do_not_recompile(compile_counter):
    @jax.jit
    def evaluate(ramp, t):
        return ramp(t)

    t = jnp.asarray(3.0)
    evaluate(Ramp(0.0, 1.0, period=10.0), t).block_until_ready()

    compile_counter.reset()
    sweeps = [
        (0.5, 2.0, -7.0, 5.0),
        (-1.0, 4.0, 0.0, 3.0),
        (2.5, 2.5, 1.0, -6.0),   # signed period sweeps too
    ]
    for v0, v1, t0, period in sweeps:
        evaluate(Ramp(v0, v1, period=period, t0=t0),
                 t).block_until_ready()
    assert compile_counter.count == 0


def test_curve_change_recompiles_exactly_once(compile_counter):
    @jax.jit
    def evaluate(ramp, t):
        return ramp(t)

    t = jnp.asarray(1.0)
    # warm up the jitted evaluation and (eagerly) the cosine path,
    # so the measured section sees exactly the one jit-cache miss
    Ramp(0.0, 1.0, period=4.0, curve="cosine")(1.0)
    evaluate(Ramp(0.0, 1.0, period=4.0), t).block_until_ready()

    compile_counter.reset()
    evaluate(Ramp(0.0, 1.0, period=4.0, curve="cosine"),
             t).block_until_ready()
    assert compile_counter.count == 1
    # sweeping the new curve's endpoints/timing: cached again
    evaluate(Ramp(0.2, 0.8, period=2.0, t0=1.0, curve="cosine"),
             t).block_until_ready()
    assert compile_counter.count == 1


def test_composed_curve_sweeps_do_not_recompile(compile_counter):
    @jax.jit
    def evaluate(curve, t):
        return curve(t)

    t = jnp.asarray(2.0)
    evaluate(2.0 * Ramp(0.0, 1.0, period=5.0) + 1.0,
             t).block_until_ready()

    compile_counter.reset()
    evaluate(3.0 * Ramp(1.0, 2.0, period=8.0, t0=-4.0) + 0.5,
             t).block_until_ready()
    assert compile_counter.count == 0


def test_ramp_in_scan_carry(compile_counter):
    # a Ramp rides the carry and is evaluated at every scan step
    ts = jnp.linspace(-2.0, 12.0, 29)

    @jax.jit
    def integrate(ramp, ts):
        def body(carry, t):
            return carry, carry(t)
        _, values = jax.lax.scan(body, ramp, ts)
        return values

    ramp = Ramp(1.0, 3.0, period=10.0, curve="cosine")
    values = integrate(ramp, ts)
    expected = [_ramp_ref(float(t), 1.0, 3.0, 0.0, 10.0,
                          _cosine_shape) for t in ts]
    np.testing.assert_allclose(values, expected, rtol=1e-12)

    # endpoint/timing sweeps through the scan: zero recompiles
    compile_counter.reset()
    integrate(Ramp(0.0, 5.0, period=2.0, t0=-3.0, curve="cosine"),
              ts).block_until_ready()
    assert compile_counter.count == 0


# ================================================================
#  TimeFunction (an arbitrary law fn(t, *params))
# ================================================================
def test_time_function_evaluates_the_law():
    curve = TimeFunction(lambda t, w: jnp.sin(w * t), (2.0,))
    assert isinstance(curve, TimeDependent)
    for t in (-1.0, 0.0, 0.7, 3.0):
        np.testing.assert_allclose(curve(t), math.sin(2.0 * t),
                                   atol=1e-14)


def test_time_function_without_params():
    curve = TimeFunction(lambda t: t**2)
    for t in (-2.0, 0.0, 1.5, 4.0):
        np.testing.assert_allclose(curve(t), t**2, atol=1e-14)


def test_time_function_at_time_matches_call():
    curve = TimeFunction(lambda t, a: a * t, (3.0,))
    np.testing.assert_allclose(curve.at_time(2.0), curve(2.0))


def test_time_function_rejects_a_non_callable():
    with pytest.raises(TypeError, match="fn must be callable"):
        TimeFunction(3.0)


def test_time_function_repr_names_the_law():
    text = repr(TimeFunction(_named_law, (2.0, 0.5)))
    assert text == "TimeFunction(_named_law, params=(2.0, 0.5))"


def test_time_function_affine_composition():
    base = TimeFunction(lambda t, w: jnp.sin(w * t), (2.0,))
    derived = 2.0 * base + 1.0
    assert isinstance(derived, TimeDependent)
    for t in (-1.0, 0.4, 2.2):
        np.testing.assert_allclose(
            derived(t), 2.0 * math.sin(2.0 * t) + 1.0, rtol=1e-12)


def test_time_function_param_sweep_does_not_recompile(compile_counter):
    law = lambda t, w: jnp.sin(w * t)  # noqa: E731 — reused identity

    @jax.jit
    def evaluate(curve, t):
        return curve(t)

    t = jnp.asarray(1.0)
    evaluate(TimeFunction(law, (1.0,)), t).block_until_ready()

    compile_counter.reset()
    for w in (2.0, 3.5, 0.5):
        evaluate(TimeFunction(law, (w,)), t).block_until_ready()
    assert compile_counter.count == 0


def test_time_function_law_change_recompiles_once(compile_counter):
    @jax.jit
    def evaluate(curve, t):
        return curve(t)

    t = jnp.asarray(1.0)
    evaluate(TimeFunction(jnp.sin), t).block_until_ready()

    compile_counter.reset()
    evaluate(TimeFunction(jnp.cos), t).block_until_ready()
    assert compile_counter.count == 1


def test_time_function_grad_flows_through_params():
    # d/da [a * t**2] at t = 2 is t**2 = 4 (grad through the pytree)
    curve = TimeFunction(lambda t, a: a * t**2, (3.0,))
    grad = jax.grad(lambda c: c(2.0))(curve)
    np.testing.assert_allclose(grad.params[0], 4.0, rtol=1e-12)


def _named_law(t, a, b):
    """Return a named law so the repr shows a stable ``__name__``."""
    return a * jnp.cos(b * t)


# ================================================================
#  TimeSeries (tabulated data, branch-free jnp.interp)
# ================================================================
def test_time_series_hits_the_sample_points():
    times = [0.0, 1.0, 3.0, 6.0]
    values = [2.0, 5.0, -1.0, 4.0]
    series = TimeSeries(times, values)
    assert isinstance(series, TimeDependent)
    for t, v in zip(times, values, strict=True):
        np.testing.assert_allclose(series(t), v, atol=1e-14)


def test_time_series_interpolates_linearly_between_knots():
    series = TimeSeries([0.0, 2.0], [10.0, 20.0])
    np.testing.assert_allclose(series(0.5), 12.5, atol=1e-14)
    np.testing.assert_allclose(series(1.0), 15.0, atol=1e-14)


def test_time_series_end_clamps_outside_the_range():
    series = TimeSeries([1.0, 2.0, 3.0], [4.0, 6.0, 5.0])
    np.testing.assert_allclose(series(-10.0), 4.0)   # before first
    np.testing.assert_allclose(series(100.0), 5.0)   # after last


def test_time_series_repr_shows_the_knots():
    text = repr(TimeSeries([0.0, 1.0], [2.0, 3.0]))
    assert text == "TimeSeries(times=[0.0, 1.0], values=[2.0, 3.0])"


def test_time_series_repr_survives_tracing():
    # under a trace the knots are tracers: the repr falls back cleanly
    series = TimeSeries([0.0, 1.0], [2.0, 3.0])
    captured = {}

    @jax.jit
    def sample(s, t):
        captured["text"] = repr(s)
        return s(t)

    sample(series, jnp.asarray(0.5))
    assert captured["text"].startswith("TimeSeries(")


@pytest.mark.parametrize(("times", "values", "match"), [
    ([[0.0, 1.0]], [1.0, 2.0], "1-D arrays"),
    ([0.0, 1.0, 2.0], [1.0, 2.0], "equal length"),
    ([0.0], [1.0], "at least two samples"),
    ([0.0, 0.0], [1.0, 2.0], "strictly increasing"),
    ([1.0, 0.0], [1.0, 2.0], "strictly increasing"),
])
def test_time_series_validation_errors(times, values, match):
    with pytest.raises(ValueError, match=match):
        TimeSeries(times, values)


def test_time_series_affine_composition():
    base = TimeSeries([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    derived = 3.0 * base - 1.0
    assert isinstance(derived, TimeDependent)
    for t in (-1.0, 0.5, 1.5, 5.0):
        np.testing.assert_allclose(
            derived(t), 3.0 * float(base(t)) - 1.0, rtol=1e-12)


def test_time_series_retabulation_does_not_recompile(compile_counter):
    @jax.jit
    def evaluate(series, t):
        return series(t)

    t = jnp.asarray(1.3)
    evaluate(TimeSeries([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
             t).block_until_ready()

    # same-length re-tabulation: the knots are dynamic leaves
    compile_counter.reset()
    evaluate(TimeSeries([0.0, 1.5, 4.0], [3.0, -1.0, 2.0]),
             t).block_until_ready()
    evaluate(TimeSeries([-2.0, 0.0, 9.0], [1.0, 1.0, 1.0]),
             t).block_until_ready()
    assert compile_counter.count == 0


def test_time_series_grad_flows_through_values():
    # interp at t = 0.5 is 0.5*values[0] + 0.5*values[1] on [0, 1]
    series = TimeSeries([0.0, 1.0, 2.0], [0.0, 10.0, 20.0])
    grad = jax.grad(lambda c: c(0.5))(series)
    np.testing.assert_allclose(grad.values[1], 0.5, rtol=1e-12)


def test_time_series_in_scan_carry(compile_counter):
    ts = jnp.linspace(-1.0, 5.0, 25)
    series = TimeSeries([0.0, 2.0, 4.0], [1.0, 3.0, -1.0])

    @jax.jit
    def sample(series, ts):
        def body(carry, t):
            return carry, carry(t)
        _, out = jax.lax.scan(body, series, ts)
        return out

    got = sample(series, ts)
    expected = np.interp(np.asarray(ts), [0.0, 2.0, 4.0], [1.0, 3.0, -1.0])
    np.testing.assert_allclose(got, expected, rtol=1e-12)

    compile_counter.reset()
    sample(TimeSeries([1.0, 2.0, 8.0], [0.0, 0.0, 5.0]),
           ts).block_until_ready()
    assert compile_counter.count == 0
