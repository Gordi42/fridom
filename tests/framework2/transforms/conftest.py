"""Shared fixtures for the transforms tests: a grid + state factory."""
import numpy as np
import pytest

from fridom.framework.utils import jaxify
from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.transforms.base import StateTransform
from fridom.framework2.transforms.info import TransformInfo
from fridom.framework2.transforms.signature import StateSignature

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
