"""Shared fixtures for the transforms tests: a grid + state factory."""
from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import dtype_real, jaxify
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.model.declarations import FieldDeclaration
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration
from fridom.model.space_patterns import Collocated
from fridom.model.terms import term
from fridom.model.time_steppers.runge_kutta import (
    ExplicitRungeKutta,
    tableaus,
)
from fridom.model.transforms.base import StateTransform
from fridom.model.transforms.info import TransformInfo
from fridom.model.transforms.signature import StateSignature

N = 8


@pytest.fixture
def mesh():
    return IntervalMesh(N, (0.0, 1.0), periodic=True, name="x")


@pytest.fixture
def grid(mesh):
    return Grid((mesh,))


@pytest.fixture
def other_grid():
    return Grid((IntervalMesh(N, (0.0, 1.0), periodic=True,
                              name="x"),))


def _field(grid, name, shift):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    data = np.sin(2 * np.pi * x) + shift
    return grid.create_field(name=name, data=data)


def build_state(grid, u_shift=0.0, v_shift=0.0, names=("u", "v")):
    """Build a small state on ``grid`` with the given components."""
    shifts = {"u": u_shift, "v": v_shift}
    return VectorField({
        name: _field(grid, name, shifts.get(name, 0.3))
        for name in names})


@pytest.fixture
def make_state(grid):
    def build(u_shift=0.0, v_shift=0.0, names=("u", "v")):
        return build_state(grid, u_shift, v_shift, names)
    return build


@pytest.fixture
def state(make_state):
    return make_state(u_shift=0.5, v_shift=-0.2)


@pytest.fixture
def sig(state):
    return StateSignature.of_prognostic(state)


# ================================================================
#  Concrete Tier-1 test transforms
# ================================================================
@jaxify
class Scale(StateTransform):

    """Endo, scales every component by a constant (not idempotent)."""

    def __init__(self, signature, factor):
        self._sig = signature
        self._factor = factor

    @property
    def domain(self):
        return self._sig

    @property
    def codomain(self):
        return self._sig

    def _evaluate(self, state):
        return state * self._factor, TransformInfo.EMPTY

    def __repr__(self):
        return f"Scale({self._factor})"


def make_cross(domain, codomain):
    """Build a non-endo transform mapping ``domain`` -> ``codomain``."""

    @jaxify
    class Cross(StateTransform):

        """A non-endo passthrough (distinct domain/codomain)."""

        @property
        def domain(self):
            return domain

        @property
        def codomain(self):
            return codomain

        def _evaluate(self, state):
            return state, TransformInfo.EMPTY

    return Cross()


@jaxify
class KeepFirst(StateTransform):

    """Idempotent projector: zeroes every component but the first."""

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
        names = state.component_names
        zeros = {name: state[name] * 0.0 for name in names[1:]}
        return state.replace(**zeros), TransformInfo.EMPTY

    def __repr__(self):
        return "KeepFirst()"


# ================================================================
#  Tier-2 toy dynamical cores (a linear inertial oscillation)
# ================================================================
DT = 2e-3
F0 = 8.0


@jaxify
class Coriolis(Module):

    """Linear f-plane rotation: du/dt = f0 v, dv/dt = -f0 u.

    ``f0`` is a static plain-float class attribute (ScalarField
    multiplies by Python scalars, not device arrays); the bound
    ``coriolis.f0`` parameter is supplied separately by
    :class:`F0Provider`.
    """

    f0 = F0
    field_declarations = (
        FieldDeclaration("u", space=Collocated()),
        FieldDeclaration("v", space=Collocated()),
    )

    @term(name="cor", advances=("u", "v"), linear=True)
    def cor(self, state, _ctx):
        return {"u": state["v"] * self.f0,
                "v": state["u"] * (-self.f0)}


@partial(jaxify, dynamic=("f0",))
class F0Provider(Module):

    """A pure provider binding ``coriolis.f0`` (no consuming term)."""

    field_declarations = ()
    parameter_declarations = (
        ParameterDeclaration("coriolis.f0", attr="f0", units="1/s"),)

    def __init__(self, f0=F0):
        self.f0 = jnp.asarray(f0, dtype=dtype_real())


@jaxify
class NonlinearU(Module):

    """A nonlinear (dropped-by-linearize) self-advection of u."""

    field_declarations = ()

    @term(name="adv", advances=("u",))
    def adv(self, state, _ctx):
        return {"u": state["u"] * state["u"] * 0.1}


@partial(jaxify, dynamic=("rossby",))
class RossbyProvider(Module):

    """A pure provider binding ``scaling.rossby`` (no consuming term)."""

    field_declarations = ()
    parameter_declarations = (
        ParameterDeclaration("scaling.rossby", attr="rossby",
                             units="1"),)

    def __init__(self, rossby=1.0):
        self.rossby = jnp.asarray(rossby, dtype=dtype_real())


def make_model(*, modules=None, dt=DT, name="toy"):
    """Build a small RK4 inertial-oscillation model on a 1-D grid.

    The default composition binds ``coriolis.f0`` (via
    :class:`F0Provider`), so ``TimeAverage``'s ``period=None`` reads
    the inertial period.
    """
    grid = Grid((IntervalMesh(N, (0.0, 1.0), periodic=True,
                              name="x"),))
    if modules is None:
        modules = (Coriolis(), F0Provider())
    return Model(grid=grid, modules=modules,
                 time_stepper=ExplicitRungeKutta(dt, tableau=tableaus.RK4),
                 name=name)


def set_wave_ic(model, u_shift=0.5):
    """Seed the model with a wave-plus-offset IC; return the state."""
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    model.set_fields(u=np.sin(2 * np.pi * x) + u_shift,
                     v=np.cos(2 * np.pi * x))
    return model.state


@pytest.fixture
def toy_model():
    """Return a fresh RK4 inertial-oscillation model (Coriolis)."""
    return make_model()


@pytest.fixture
def toy_state(toy_model):
    """Return a wave-plus-offset IC state on the toy grid."""
    return set_wave_ic(toy_model)
