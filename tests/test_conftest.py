"""Self-tests for the suite fixtures (conftest.py)."""
import types

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
#  periodic cache eviction (pytest_runtest_teardown)
# ================================================================
@pytest.fixture
def conftest_mod(request):
    """Return the suite-level tests/conftest.py module object."""
    for _name, plugin in request.config.pluginmanager.list_name_plugin():
        path = (getattr(plugin, "__file__", "") or "").replace("\\", "/")
        if path.endswith("tests/conftest.py"):
            return plugin
    pytest.fail("tests/conftest.py plugin not found")  # pragma: no cover
    return None  # pragma: no cover


@pytest.fixture
def clear_spy(monkeypatch):
    """Count jax.clear_caches() calls without really clearing."""
    calls = []
    monkeypatch.setattr(jax, "clear_caches", lambda: calls.append(1))
    return calls


def _item(module):
    """Return a stand-in test item carrying just a ``.module`` attribute."""
    return types.SimpleNamespace(module=module)


def test_teardown_clears_on_file_boundary(clear_spy, conftest_mod,
                                          monkeypatch):
    monkeypatch.setattr(conftest_mod, "_clear_cache", True)
    mod_a, mod_b = object(), object()
    conftest_mod.pytest_runtest_teardown(_item(mod_a), _item(mod_b))
    assert clear_spy == [1]


def test_teardown_no_clear_within_file(clear_spy, conftest_mod,
                                       monkeypatch):
    monkeypatch.setattr(conftest_mod, "_clear_cache", True)
    mod = object()
    conftest_mod.pytest_runtest_teardown(_item(mod), _item(mod))
    assert clear_spy == []


def test_teardown_disabled_never_clears(clear_spy, conftest_mod,
                                        monkeypatch):
    monkeypatch.setattr(conftest_mod, "_clear_cache", False)
    mod_a, mod_b = object(), object()
    conftest_mod.pytest_runtest_teardown(_item(mod_a), _item(mod_b))
    assert clear_spy == []


# ================================================================
#  forced_devices
# ================================================================
def test_forced_device_count_took_effect(forced_devices):
    """Fail (not skip) when the device forcing did not take effect."""
    if forced_devices is None:
        pytest.skip("only relevant in the multi-device suite")
    assert jax.device_count() == forced_devices
