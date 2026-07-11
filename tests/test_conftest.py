"""Self-tests for the suite fixtures (conftest.py)."""
import jax
import jax.numpy as jnp
import pytest
from jax._src.monitoring import get_event_duration_listeners


# ================================================================
#  compile_counter
# ================================================================
def test_compile_counter_counts_compilations(compile_counter):
    # prepare the inputs first: eager operations trace too and would
    # otherwise show up in the count
    x_small = jnp.arange(4.0)
    y_small = jnp.arange(4.0) + 1.0
    x_large = jnp.arange(8.0)

    @jax.jit
    def f(x):
        return 2.0 * x + 1.0

    compile_counter.reset()
    f(x_small).block_until_ready()
    assert compile_counter.count == 1

    # same shape and dtype: served from the jit cache, no retrace
    f(y_small).block_until_ready()
    assert compile_counter.count == 1

    # new shape: exactly one retrace
    f(x_large).block_until_ready()
    assert compile_counter.count == 2


def test_compile_counter_reset(compile_counter):
    x = jnp.arange(4.0)

    @jax.jit
    def g(x):
        return x - 1.0

    compile_counter.reset()
    g(x).block_until_ready()
    assert compile_counter.count == 1

    compile_counter.reset()
    assert compile_counter.count == 0
    # a cached call after reset stays at zero
    g(x).block_until_ready()
    assert compile_counter.count == 0


def test_compile_counter_unregisters(compile_counter):
    # the listeners of previous tests must have been unregistered:
    # exactly one CompileCounter listener (this test's own) is active
    listeners = get_event_duration_listeners()
    assert compile_counter._listener in listeners
    counter_listeners = [
        callback for callback in listeners
        if type(getattr(callback, "__self__", None)).__name__
        == "CompileCounter"
    ]
    assert len(counter_listeners) == 1


# ================================================================
#  forced_devices
# ================================================================
def test_forced_device_count_took_effect(forced_devices):
    """Fail (not skip) when the device forcing did not take effect."""
    if forced_devices is None:
        pytest.skip("only relevant in the multi-device suite")
    assert jax.device_count() == forced_devices
