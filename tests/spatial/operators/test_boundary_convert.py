"""Tests for the boundary conversions AsProfile / Adopt (boundary.py)."""
import numpy as np
import pytest

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.boundary import Adopt, AsProfile
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodeSet
from fridom.spatial.spaces.trace import Side, TraceSpace

N = 6


@pytest.fixture
def grid():
    mx = IntervalMesh(N, (0.0, 1.0), name="x")
    my = IntervalMesh(N, (0.0, 2.0), name="y")
    mz = IntervalMesh(4, (0.0, 3.0), periodic=False, name="z")
    return Grid((mx, my, mz))


def _trace(grid, side=Side.HIGH):
    mx, my, mz = grid.factors
    f = grid.create_field(mx.center * my.center * mz.center,
                          init=lambda x, y, z: x + y + z)
    return f.trace("z", side)


# ================================================================
#  AsProfile: Trace -> ConstantSpace
# ================================================================
def test_as_profile_interns_and_carries_the_kind():
    assert AsProfile() is AsProfile()
    assert AsProfile().dispatch_kind == "as_profile"


def test_as_profile_codomain_is_the_constant(grid):
    mz = grid.factors[2]
    t = mz.trace(NodeSet.CENTER, Side.HIGH)
    assert AsProfile().codomain(t) is mz.constant


def test_as_profile_codomain_preserves_complex(grid):
    mz = grid.factors[2]
    t = mz.trace(NodeSet.CENTER, Side.LOW).as_complex()
    assert AsProfile().codomain(t) is mz.constant.as_complex()


def test_as_profile_rejects_non_trace(grid):
    with pytest.raises(SpaceMismatchError, match="as_profile"):
        AsProfile().codomain(grid.factors[2].center)


def test_as_profile_is_an_exact_retag(grid):
    t = _trace(grid)
    prof = t.as_profile("z")
    assert isinstance(prof.function_space.bare.factor("z"), ConstantSpace)
    assert prof.function_space.bare.factor("z").is_constant is True
    # data untouched (retag), same size-1 storage
    assert np.allclose(np.asarray(prof.data), np.asarray(t.data))


def test_as_profile_bridges_into_constant_z_arithmetic(grid):
    # the point: once a Profile, it broadcasts into the interior under
    # the strict algebra (which the Trace deliberately refuses)
    mx, my, mz = grid.factors
    prof = _trace(grid).as_profile("z")
    full = grid.create_field(mx.center * my.center * mz.center,
                             init=lambda x, y, z: 1.0 + x + y + z)
    product = full * prof
    assert product.function_space.bare is full.function_space.bare


# ================================================================
#  Adopt: ConstantSpace -> Trace
# ================================================================
def test_adopt_interns_on_axis_side_and_node_set():
    assert Adopt(NodeSet.CENTER, Side.HIGH, "z") is Adopt(
        NodeSet.CENTER, Side.HIGH, "z")
    assert Adopt(NodeSet.CENTER, Side.HIGH, "z") is not Adopt(
        NodeSet.CENTER, Side.LOW, "z")
    assert Adopt(NodeSet.CENTER, Side.HIGH, "z") is not Adopt(
        NodeSet.OUTER, Side.HIGH, "z")
    assert Adopt(NodeSet.CENTER, Side.HIGH, "z").dispatch_kind == "adopt"
    assert Adopt(NodeSet.CENTER, Side.HIGH, "z").axis == "z"


def test_adopt_type_validations():
    with pytest.raises(TypeError, match="node_set must be a NodeSet"):
        Adopt("center", Side.HIGH, "z")
    with pytest.raises(TypeError, match="side must be a Side"):
        Adopt(NodeSet.CENTER, "high", "z")
    with pytest.raises(TypeError, match="axis must be"):
        Adopt(NodeSet.CENTER, Side.HIGH, 0)


def test_adopt_codomain_replaces_the_constant_factor(grid):
    mx, my, mz = grid.factors
    space = mx.center * my.center * mz.constant
    out = Adopt(NodeSet.CENTER, Side.HIGH, "z").codomain(space)
    assert out.factor("z") is mz.trace(NodeSet.CENTER, Side.HIGH)


def test_adopt_codomain_preserves_complex(grid):
    mx, my, mz = grid.factors
    space = (mx.center * my.center * mz.constant).as_complex()
    out = Adopt(NodeSet.OUTER, Side.LOW, "z").codomain(space)
    assert out.factor("z") is mz.trace(NodeSet.OUTER, Side.LOW).as_complex()
    assert out.scalars is Scalars.COMPLEX


def test_adopt_rejects_a_non_constant_factor(grid):
    mx, my, mz = grid.factors
    with pytest.raises(SpaceMismatchError, match="ConstantSpace factor"):
        Adopt(NodeSet.CENTER, Side.HIGH, "z").codomain(
            mx.center * my.center * mz.center)


def test_adopt_rejects_an_absent_axis(grid):
    mx, my, mz = grid.factors
    with pytest.raises(SpaceMismatchError, match="absent"):
        Adopt(NodeSet.CENTER, Side.HIGH, "w").codomain(
            mx.center * my.center * mz.constant)


def test_adopt_is_an_exact_retag(grid):
    mx, my, mz = grid.factors
    prof = grid.create_field(mx.center * my.center * mz.constant,
                             init=lambda x, y: x + y)
    adopted = prof.adopt("z", NodeSet.CENTER, Side.HIGH)
    assert isinstance(adopted.function_space.bare.factor("z"), TraceSpace)
    assert np.allclose(np.asarray(adopted.data), np.asarray(prof.data))


def test_as_profile_and_adopt_round_trip(grid):
    t = _trace(grid, Side.LOW)
    back = t.as_profile("z").adopt("z", NodeSet.CENTER, Side.LOW)
    assert back.function_space.bare is t.function_space.bare
    assert np.allclose(np.asarray(back.data), np.asarray(t.data))
