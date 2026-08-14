"""The scalar-or-sequence rule on Model's collection keywords.

A list or a tuple is many; anything else is one. This shard covers
the Model surface — ``modules=``, ``io=``, ``allow_unadvanced=``,
``run(outputs=)``, ``variant(extra_modules=)`` and
``propagator(wrt=)`` with its ``theta`` — and in particular the two
cases the old ``tuple(...)`` spelling got silently wrong: a bare
string spliced into its letters, and a bare jax value spliced into
its elements.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import dtype_real, jaxify
from fridom.io import triggers
from fridom.model.declarations import FieldDeclaration
from fridom.model.errors import AssemblyError
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration
from fridom.model.terms import term
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated

DT = 0.5
N = 8


# ================================================================
#  Toy model: du/dt = gain
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


class FrozenField(Module):

    """A PROGNOSTIC field that no term advances (lint bait)."""

    field_declarations = (
        FieldDeclaration("psi", space=Collocated(), long_name="Inert"),)


class Inert(Module):

    """A field-free module — legal as a ``variant`` extra."""


class FakeStream:

    """A minimal OutputStream double."""

    def __init__(self, trigger):
        self.trigger = trigger
        self.path = None
        self.writes = []

    def bind(self, model):
        pass

    def write(self, model_state):
        self.writes.append(int(model_state.clock.it))

    def truncate_after(self, iteration):
        pass

    def close(self):
        pass


def make_grid():
    return Grid((IntervalMesh(N, (0.0, 1.0), periodic=True,
                              name="x"),))


def make_model(modules=None, gain=1.0, **kwargs):
    if modules is None:
        modules = GainForcing(gain)
    return Model(grid=make_grid(), modules=modules,
                 time_stepper=AdamBashforth(DT, order=1), **kwargs)


def stream():
    return FakeStream(triggers.every(steps=1))


# ================================================================
#  modules=
# ================================================================
@pytest.mark.parametrize("wrap", [
    pytest.param(lambda m: m, id="bare"),
    pytest.param(lambda m: [m], id="list"),
    pytest.param(lambda m: (m,), id="tuple"),
])
def test_modules_takes_a_bare_module_a_list_or_a_tuple(wrap):
    model = make_model(modules=wrap(GainForcing(2.0)))
    model.run(steps=2, progress=False)
    assert float(model.state["u"].data[0]) == pytest.approx(2 * DT * 2.0)


def test_modules_list_and_tuple_assemble_the_same_module_count():
    listed = make_model(modules=[GainForcing(), FrozenField()],
                        allow_unadvanced="psi")
    tupled = make_model(modules=(GainForcing(), FrozenField()),
                        allow_unadvanced=("psi",))
    assert len(listed._carry.modules) == len(tupled._carry.modules) == 2


# ================================================================
#  io=
# ================================================================
def test_io_takes_a_bare_stream():
    one = stream()
    assert make_model(io=one)._io == (one,)


def test_io_takes_a_list_of_streams():
    a, b = stream(), stream()
    assert make_model(io=[a, b])._io == (a, b)


# ================================================================
#  allow_unadvanced= (the bare-string case)
# ================================================================
def test_allow_unadvanced_takes_a_bare_name_not_its_letters():
    # "psi" spliced into ("p", "s", "i") is what the old tuple()
    # spelling did, and none of those is a field of this assembly
    model = make_model(modules=[GainForcing(), FrozenField()],
                       allow_unadvanced="psi")
    assert "psi" in model.state


def test_allow_unadvanced_still_refuses_an_unknown_bare_name():
    with pytest.raises(AssemblyError, match="allow_unadvanced"):
        make_model(modules=[GainForcing(), FrozenField()],
                   allow_unadvanced="phi")


def test_an_unwaived_frozen_field_still_trips_the_coverage_lint():
    with pytest.raises(AssemblyError):
        make_model(modules=[GainForcing(), FrozenField()])


# ================================================================
#  run(outputs=)
# ================================================================
def test_run_outputs_takes_a_bare_stream():
    one = stream()
    make_model().run(steps=3, outputs=one, progress=False)
    assert one.writes == [0, 1, 2, 3]


def test_run_outputs_takes_a_list_and_adds_to_the_standing_io():
    standing, extra = stream(), stream()
    model = make_model(io=standing)
    model.run(steps=2, outputs=[extra], progress=False)
    assert standing.writes == extra.writes == [0, 1, 2]


# ================================================================
#  variant(extra_modules=)
# ================================================================
def test_variant_extra_modules_takes_a_bare_module():
    parent = make_model()
    child = parent.variant(extra_modules=Inert())
    assert len(child._carry.modules) == len(parent._carry.modules) + 1


# ================================================================
#  propagator(wrt=) and its theta
# ================================================================
def test_propagator_wrt_takes_a_bare_name_and_theta_a_bare_value():
    # a bare jax value must be spliced WHOLE: the old tuple(theta)
    # would have iterated a 1-d array into several theta entries
    run = make_model().propagator(wrt="toy.gain", steps=4)
    state = run(jnp.asarray(2.0)).state
    assert float(state["u"].data[0]) == pytest.approx(4 * DT * 2.0)


def test_propagator_bare_theta_is_differentiable():
    run = make_model().propagator(wrt="toy.gain", steps=4)

    def loss(gain):
        return jnp.sum(run(gain).state["u"].data ** 2)

    grad = float(jax.grad(loss)(jnp.asarray(2.0)))
    eps = 1e-4
    central = float(
        (loss(jnp.asarray(2.0 + eps)) - loss(jnp.asarray(2.0 - eps)))
        / (2 * eps))
    assert np.isfinite(grad)
    assert grad == pytest.approx(central, rel=1e-4)


def test_propagator_still_takes_a_tuple_wrt_and_theta():
    run = make_model().propagator(wrt=("toy.gain",), steps=4)
    state = run((jnp.asarray(3.0),)).state
    assert float(state["u"].data[0]) == pytest.approx(4 * DT * 3.0)


def test_propagator_theta_length_is_still_checked():
    run = make_model().propagator(wrt="toy.gain", steps=2)
    with pytest.raises(ValueError, match="expected 1 theta"):
        run((jnp.asarray(1.0), jnp.asarray(2.0)))
