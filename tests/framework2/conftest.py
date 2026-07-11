"""Configuration for the framework2 test suite."""
import os

import jax
import pytest

# ================================================================
#  Compile counter
# ================================================================
# jax records a monitoring event every time pjit traces a function,
# i.e. on every jit-cache miss (a compilation or re-compilation).
# Counting these events is the robust way to assert "no retraces":
# unlike the backend-compile event it also fires when the persistent
# compilation cache (tests/conftest.py) already holds the executable.
# The event name is jax-internal but stable; the self-test in
# test_conftest.py fails loudly if it ever changes.
TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"


class CompileCounter:

    """Counts jax compilations (pjit traces) while registered."""

    def __init__(self):
        self.count = 0

    def _listener(self, event, duration, **kwargs):  # noqa: ARG002
        if event == TRACE_EVENT:
            self.count += 1

    def reset(self):
        """Restart the count at zero."""
        self.count = 0


@pytest.fixture
def compile_counter():
    """Count the jax compilations triggered within a test.

    Yields a counter with a ``count`` attribute and a ``reset()``
    method. Because eagerly executed array operations also trace
    (and hence count) on their first occurrence per shape, prepare
    the input arrays first and call ``reset()`` right before the
    section under measurement.
    """
    counter = CompileCounter()
    jax.monitoring.register_event_duration_secs_listener(
        counter._listener)
    yield counter
    jax.monitoring.unregister_event_duration_listener(counter._listener)


# ================================================================
#  Forced host devices
# ================================================================
# The multi-device suite is a separate pytest invocation, run with
# XLA_FLAGS=--xla_force_host_platform_device_count=4 and
# FRIDOM_TEST_FORCED_DEVICES=4 in the environment (XLA_FLAGS must be
# set before jax initializes). FRIDOM_TEST_FORCED_DEVICES carries the
# intended device count into the tests so they can fail (not skip)
# when the forcing did not take effect; the repo-wide
# multi_device/single_device markers are handled in tests/conftest.py.
FORCED_DEVICES_ENV = "FRIDOM_TEST_FORCED_DEVICES"


@pytest.fixture
def forced_devices():
    """Return the forced device count, or None in the default suite."""
    value = os.environ.get(FORCED_DEVICES_ENV)
    return int(value) if value else None
