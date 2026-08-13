"""Tests for fridom.io.series (the in-memory scalar sink).

The oracles: the recorded columns and coordinates read back as arrays,
one row per firing; values come from the same scalar callables
``TimeSeries`` takes (0-d Field or scalar, coerced to float);
``truncate_after`` drops the tail keyed on the iteration coordinate; a
re-fired iteration replaces its row rather than forking the axis;
``close`` keeps the rows. The end-to-end test drives a real
``Model.run(outputs=...)`` and asserts the series equals a
hand-written chunked loop over the same model — the claim that a
``Series`` is the framework spelling of that loop.
"""
from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import dtype_real, jaxify
from fridom.io.series import Series
from fridom.io.streams import OutputStream
from fridom.io.triggers import every
from fridom.model.clock import Clock
from fridom.model.declarations import FieldDeclaration
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration
from fridom.model.terms import term
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated

DT = 0.25
N = 8


# ================================================================
#  Duck-typed model / carry
# ================================================================
class FakeCarry:
    def __init__(self, state, clock):
        self.state = state
        self.clock = clock


class FakeModel:
    def __init__(self, state, clock):
        self._carry = FakeCarry(state, clock)

    @property
    def carry(self):
        return self._carry


def clock_at(it):
    clock = Clock()
    for _ in range(it):
        clock = clock.tick(DT)
    return clock


@pytest.fixture
def grid():
    return Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))


@pytest.fixture
def scalar_field(grid):
    # a 0-d field: integrate the one axis away
    return grid.create_field(init=lambda x: x + 1.0).integrate("x")


@pytest.fixture
def model(scalar_field):
    return FakeModel({"e": scalar_field}, clock_at(0))


# ================================================================
#  Rows and columns
# ================================================================
def test_rows_and_columns(model, scalar_field):
    series = Series(
        {"energy": lambda ms: ms.state["e"], "const": lambda _ms: 3.5},
        trigger=every(steps=1))
    series.bind(model)
    for it in (0, 1, 2):
        series.write(FakeCarry({"e": scalar_field}, clock_at(it)))
    series.close()

    energy = float(np.asarray(scalar_field.data).reshape(()))
    assert len(series) == 3
    assert series.columns == ("energy", "const")
    np.testing.assert_array_equal(series["iteration"], [0, 1, 2])
    np.testing.assert_allclose(series["time"], np.arange(3) * DT)
    np.testing.assert_allclose(series["energy"], [energy] * 3)
    np.testing.assert_allclose(series["const"], [3.5] * 3)


def test_jax_scalar_coerced(model):
    series = Series({"j": lambda _ms: jnp.asarray(2.0)},
                    trigger=every(steps=1))
    series.bind(model)
    series.write(FakeCarry({}, clock_at(0)))
    np.testing.assert_allclose(series["j"], [2.0])


def test_mapping_surface(model):
    series = Series({"a": lambda _ms: 1.0, "b": lambda _ms: 2.0},
                    trigger=every(steps=1))
    series.bind(model)
    series.write(FakeCarry({}, clock_at(0)))
    assert list(series) == ["iteration", "time", "a", "b"]
    assert "a" in series
    assert "time" in series
    assert "missing" not in series
    assert set(series.to_dict()) == {"iteration", "time", "a", "b"}
    np.testing.assert_allclose(series.to_dict()["b"], [2.0])
    np.testing.assert_array_equal(series.iteration, [0])
    assert repr(series) == "Series(columns=['a', 'b'], rows=1)"


def test_unknown_column_raises_with_hint(model):
    series = Series({"a": lambda _ms: 1.0}, trigger=every(steps=1))
    series.bind(model)
    with pytest.raises(KeyError, match="no column 'b'"):
        series["b"]


def test_empty_series_reads_as_empty_arrays(model):
    series = Series({"a": lambda _ms: 1.0}, trigger=every(steps=1))
    series.bind(model)
    assert len(series) == 0
    assert series["a"].shape == (0,)
    assert series["time"].shape == (0,)
    assert series.iteration.shape == (0,)


# ================================================================
#  truncate_after + the fork-free re-fire guard
# ================================================================
def test_truncate_after_drops_the_tail(model):
    series = Series({"a": lambda _ms: 1.0}, trigger=every(steps=1))
    series.bind(model)
    for it in (0, 1, 2, 3):
        series.write(FakeCarry({}, clock_at(it)))
    series.truncate_after(1)
    series.write(FakeCarry({}, clock_at(2)))
    np.testing.assert_array_equal(series["iteration"], [0, 1, 2])
    assert len(series) == 3


def test_refired_iteration_replaces_its_row():
    """A second run() re-fires its start boundary; the axis stays flat."""
    values = iter([1.0, 2.0, 9.0])
    series = Series({"a": lambda _ms: next(values)},
                    trigger=every(steps=1))
    series.bind(FakeModel({}, clock_at(0)))   # bind consumes 1.0
    series.write(FakeCarry({}, clock_at(3)))  # -> 2.0
    series.write(FakeCarry({}, clock_at(3)))  # same it -> replaces
    np.testing.assert_array_equal(series["iteration"], [3])
    np.testing.assert_allclose(series["a"], [9.0])


def test_close_keeps_rows_and_rebinding_continues(model):
    series = Series({"a": lambda _ms: 1.0}, trigger=every(steps=1))
    series.bind(model)
    series.write(FakeCarry({}, clock_at(0)))
    series.close()
    assert len(series) == 1              # the point of the class
    series.bind(model)                   # a second run continues it
    series.write(FakeCarry({}, clock_at(1)))
    np.testing.assert_array_equal(series["iteration"], [0, 1])


# ================================================================
#  Guards
# ================================================================
def test_non_scalar_field_raises(grid):
    field = grid.create_field(init=lambda x: x)
    model = FakeModel({"f": field}, clock_at(0))
    series = Series({"f": lambda ms: ms.state["f"]},
                    trigger=every(steps=1))
    with pytest.raises(ValueError, match="not a scalar"):
        series.bind(model)


def test_non_scalar_array_raises(model):
    series = Series({"a": lambda _ms: jnp.zeros(3)},
                    trigger=every(steps=1))
    with pytest.raises(ValueError, match="not a scalar"):
        series.bind(model)


def test_empty_columns_rejected():
    with pytest.raises(ValueError, match="at least one column"):
        Series({}, trigger=every(steps=1))


def test_non_callable_column_rejected():
    with pytest.raises(TypeError, match="must be callable"):
        Series({"a": 1.0}, trigger=every(steps=1))


def test_reserved_column_names_rejected():
    with pytest.raises(ValueError, match="reserved"):
        Series({"time": lambda _ms: 1.0}, trigger=every(steps=1))


def test_walltime_trigger_rejected(model):
    series = Series({"a": lambda _ms: 1.0},
                    trigger=every(walltime="1h"))
    with pytest.raises(ValueError, match="snapshot/action-only"):
        series.bind(model)


def test_bind_without_carry_raises():
    series = Series({"a": lambda _ms: 1.0}, trigger=every(steps=1))
    with pytest.raises(TypeError, match="carry"):
        series.bind(object())


def test_double_bind_raises(model):
    series = Series({"a": lambda _ms: 1.0}, trigger=every(steps=1))
    series.bind(model)
    with pytest.raises(RuntimeError, match="already bound"):
        series.bind(model)


def test_write_before_bind_raises():
    series = Series({"a": lambda _ms: 1.0}, trigger=every(steps=1))
    with pytest.raises(RuntimeError, match="not bound"):
        series.write(FakeCarry({}, clock_at(0)))


def test_truncate_before_bind_raises():
    series = Series({"a": lambda _ms: 1.0}, trigger=every(steps=1))
    with pytest.raises(RuntimeError, match="not bound"):
        series.truncate_after(0)


def test_implements_the_output_stream_protocol():
    series = Series({"a": lambda _ms: 1.0}, trigger=every(steps=1))
    assert isinstance(series, OutputStream)
    # store-less: no path, so the session dedupes it by identity
    assert not hasattr(series, "path")
    assert series.trigger == every(steps=1)


# ================================================================
#  End to end: Series == the hand-written chunked loop
# ================================================================
@partial(jaxify, dynamic=("gain",))
class GainForcing(Module):

    """One PROGNOSTIC field forced by a provided parameter."""

    def __init__(self, gain=1.0):
        self.gain = jnp.asarray(gain, dtype=dtype_real())

    extra_halo = HaloSpec({"x": 0})

    field_declarations = (
        FieldDeclaration("u", space=Collocated(), long_name="Forced"),)
    parameter_declarations = (
        ParameterDeclaration("toy.gain", attr="gain", units="1"),)

    @term(advances=("u",))
    def force(self, state, ctx):
        gain = ctx.params.get("toy.gain", 0.0)
        u = state["u"]
        return {"u": u.with_data(
            jnp.broadcast_to(gain, u.data.shape).astype(u.data.dtype))}


def make_model():
    grid = Grid((IntervalMesh(N, (0.0, 1.0), periodic=True, name="x"),))
    return Model(grid=grid, modules=(GainForcing(2.0),),
                 time_stepper=AdamBashforth(DT, order=1))


def mean_u(model_state):
    return model_state.state["u"].integrate()


def test_run_with_series_matches_the_chunked_loop():
    series = Series({"u": mean_u}, trigger=every(steps=4))
    model = make_model()
    model.run(steps=12, outputs=(series,), progress=False)

    # the loop the Series replaces
    manual_model = make_model()
    manual = [float(np.asarray(mean_u(manual_model.carry).data).reshape(()))]
    for _ in range(3):
        manual_model.run(steps=4, progress=False)
        manual.append(float(np.asarray(mean_u(manual_model.carry).data).reshape(())))

    np.testing.assert_array_equal(series["iteration"], [0, 4, 8, 12])
    np.testing.assert_allclose(series["time"], np.array([0, 4, 8, 12]) * DT)
    np.testing.assert_allclose(series["u"], manual, rtol=0, atol=0)
    # du/dt = 2 on a unit-length domain, so the integral is 2 t
    np.testing.assert_allclose(series["u"], 2.0 * series["time"],
                               rtol=1e-12)


def test_two_streams_on_one_run_stay_independent():
    fast = Series({"u": mean_u}, trigger=every(steps=2))
    slow = Series({"u": mean_u}, trigger=every(steps=6))
    make_model().run(steps=12, outputs=(fast, slow), progress=False)
    np.testing.assert_array_equal(fast["iteration"],
                                  [0, 2, 4, 6, 8, 10, 12])
    np.testing.assert_array_equal(slow["iteration"], [0, 6, 12])
