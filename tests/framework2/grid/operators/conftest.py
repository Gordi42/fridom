"""Shared fixtures for the Wave-2 operator-cluster tests.

Provides tiny real Wave-1 meshes/spaces, minimal concrete test
kernels, and a frozen stand-in field/grid pair implementing exactly
the duck-typed fields.md iteration-1 surface the operator layer
consumes (``function_space``, ``grid`` with ``sync``/``dispatch``,
``data``/``_data``, ``metadata``, ``with_data``, arithmetic dunders,
and the plumbing constructor). Real-field integration happens at the
wave merge.
"""
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.base import (
    OperatorRequirements,
    SeparableOperator,
    UnaryOperator,
)
from fridom.framework2.grid.spaces.function_space import FunctionSpace
from fridom.framework2.grid.spaces.nodal import Center, Right


# ================================================================
#  Concrete test operators
# ================================================================
class Stagger(SeparableOperator):

    """Center <-> Right test kernel; adds ``shift`` to the data."""

    dispatch_kind = "diff"

    def __init__(self, halo=1, shift=1.0):
        self.halo = halo
        self.shift = shift

    def codomain(self, domain):
        if isinstance(domain, Center):
            return domain.mesh.right
        if isinstance(domain, Right):
            return domain.mesh.center
        raise SpaceMismatchError(
            f"unsupported domain {domain!r}", left=domain)

    def requirements(self, domain):  # noqa: ARG002
        return OperatorRequirements(halo=self.halo)

    def _apply_factor(self, f, axis):
        space = f.function_space.bare
        if isinstance(space, FunctionSpace):
            new_space = self.codomain(space)
        else:
            new_space = space.replace(
                **{axis: self.codomain(space.factor(axis))})
        return type(f)(f.grid, new_space, f._data + self.shift,
                       f.metadata)


class Keep(SeparableOperator):

    """Signature-preserving test kernel (``domain -> domain``)."""

    dispatch_kind = "keep"

    def __init__(self, halo=0, shift=0.0):
        self.halo = halo
        self.shift = shift

    def codomain(self, domain):
        return domain

    def requirements(self, domain):  # noqa: ARG002
        return OperatorRequirements(halo=self.halo)

    def _apply_factor(self, f, axis):  # noqa: ARG002
        return f.with_data(f.data + self.shift)


class Whole(UnaryOperator):

    """Whole-space (non-separable) identity-signature operator."""

    def codomain(self, domain):
        return domain

    def _apply(self, f):
        return f


# ================================================================
#  Stand-in field and grid (duck-typed fields.md surface)
# ================================================================
class FakeGrid:

    """Stand-in grid: records syncs, carries a dispatch registry."""

    def __init__(self, dispatch=None):
        self.dispatch = dispatch
        self.sync_log = []

    def sync(self, field):
        self.sync_log.append(field)
        return field


class FakeField:

    """Frozen stand-in for the fields.md iteration-1 field surface.

    Halo width is zero, so the storage-shaped ``_data`` and the
    true-shape ``data`` view coincide (and the halo-validity claim
    is the all-zero spec).
    """

    __slots__ = ("_data", "function_space", "grid", "halo_valid",
                 "metadata")

    def __init__(self, grid, function_space, data, metadata=None,
                 halo_valid=None):
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "function_space", function_space)
        object.__setattr__(self, "_data", jnp.asarray(data))
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self, "halo_valid",
            HaloSpec.zero(tuple(function_space.names))
            if halo_valid is None else halo_valid)

    def __setattr__(self, name, value):
        raise AttributeError("FakeField is frozen")

    @property
    def data(self):
        return self._data

    def with_data(self, data):
        return type(self)(self.grid, self.function_space, data,
                          self.metadata)

    def __add__(self, other):
        assert other.function_space is self.function_space
        return self.with_data(self._data + other._data)

    def __mul__(self, scalar):
        return self.with_data(self._data * scalar)

    __rmul__ = __mul__


# ================================================================
#  Fixtures
# ================================================================
# class fixtures: with --import-mode=importlib the helper classes
# cannot be imported from sibling test files, so they are handed out
# through fixtures instead.
@pytest.fixture
def stagger_cls():
    return Stagger


@pytest.fixture
def keep_cls():
    return Keep


@pytest.fixture
def whole_cls():
    return Whole


@pytest.fixture
def field_cls():
    return FakeField


@pytest.fixture
def grid_cls():
    return FakeGrid


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(4, (0.0, 2.0), name="y")


@pytest.fixture
def stagger():
    return Stagger()


@pytest.fixture
def grid():
    return FakeGrid()


@pytest.fixture
def field_1d(grid, mx):
    return FakeField(grid, mx.center, jnp.arange(8.0))


@pytest.fixture
def field_2d(grid, mx, my):
    return FakeField(grid, mx.center * my.center,
                     jnp.arange(32.0).reshape(8, 4))
