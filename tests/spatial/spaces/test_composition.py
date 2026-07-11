"""Tests for the shared factor-wise space-tag validators."""
import jax.numpy as jnp
import pytest

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.symbol import Symbol
from fridom.spatial.spaces.composition import (
    compose_spaces,
    union_spaces,
)

N = 16


@pytest.fixture
def mesh_1d():
    return IntervalMesh(N, (0.0, 1.0), name="x")


@pytest.fixture
def mesh_2d():
    mx = IntervalMesh(N, (0.0, 1.0), name="x")
    my = IntervalMesh(N, (0.0, 2.0), name="y")
    return mx, my


# ================================================================
#  union_spaces
# ================================================================
def test_union_same_object_is_the_fast_path(mesh_1d):
    space = mesh_1d.fourier(origin=mesh_1d.center)
    # identical operands short-circuit and return the same object
    assert union_spaces(space, space) is space


def test_union_constant_wildcard_builds_the_interned_product(mesh_2d):
    mx, my = mesh_2d
    x_only = mx.fourier(origin=mx.center) * my.constant
    y_only = mx.constant * my.fourier(origin=my.center)
    result = union_spaces(x_only, y_only)
    expected = (mx.fourier(origin=mx.center)
                * my.fourier(origin=my.center))
    # Constant ⊗ X -> X on each factor; the identical interned object
    assert result is expected


def test_union_keeps_a_shared_non_constant_factor(mesh_2d):
    mx, my = mesh_2d
    both = mx.fourier(origin=mx.center) * my.fourier(origin=my.center)
    x_only = mx.fourier(origin=mx.center) * my.constant
    # position x agrees (fa is fb), position y is Const ⊗ Fourier -> Fourier
    assert union_spaces(both, x_only) is both


def test_union_rejects_a_disagreeing_non_constant_factor(mesh_2d):
    mx, my = mesh_2d
    a = mx.fourier(origin=mx.center) * my.fourier(origin=my.center)
    b = mx.fourier(origin=mx.right) * my.fourier(origin=my.center)
    with pytest.raises(SpaceMismatchError,
                       match="disagree on a non-constant factor"):
        union_spaces(a, b)


def test_union_rejects_a_rank_mismatch(mesh_2d):
    mx, my = mesh_2d
    one_d = mx.fourier(origin=mx.center)
    two_d = mx.fourier(origin=mx.center) * my.constant
    with pytest.raises(SpaceMismatchError,
                       match="incompatible spaces"):
        union_spaces(one_d, two_d)


def test_union_operation_label_is_recorded(mesh_2d):
    mx, my = mesh_2d
    a = mx.fourier(origin=mx.center) * my.fourier(origin=my.center)
    b = mx.fourier(origin=mx.right) * my.fourier(origin=my.center)
    with pytest.raises(SpaceMismatchError) as exc:
        union_spaces(a, b, operation="add")
    assert exc.value.operation == "add"


# ================================================================
#  compose_spaces
# ================================================================
def test_compose_same_axis_chain(mesh_1d):
    center = mesh_1d.fourier(origin=mesh_1d.center)
    right = mesh_1d.fourier(origin=mesh_1d.right)
    # outer @ inner with inner: center -> right, outer: right -> center
    domain, codomain = compose_spaces(center, right, right, center)
    assert domain is center
    assert codomain is center


def test_compose_disjoint_axes_into_the_corner(mesh_2d):
    mx, my = mesh_2d
    cx, rx = mx.fourier(origin=mx.center), mx.fourier(origin=mx.right)
    cy, ry = my.fourier(origin=my.center), my.fourier(origin=my.right)
    # inner acts on x only, outer acts on y only (disjoint axes)
    inner_domain = cx * my.constant
    inner_codomain = rx * my.constant
    outer_domain = mx.constant * cy
    outer_codomain = mx.constant * ry
    domain, codomain = compose_spaces(
        inner_domain, inner_codomain, outer_domain, outer_codomain)
    # domain picks inner on x, falls back to outer on y
    assert domain is cx * cy
    # codomain falls back to inner on x, picks outer on y
    assert codomain is rx * ry


def test_compose_rejects_a_shared_axis_mismatch(mesh_2d):
    mx, my = mesh_2d
    inner_domain = mx.fourier(origin=mx.center) * my.constant
    inner_codomain = mx.fourier(origin=mx.right) * my.constant
    # outer domain on x disagrees with the inner codomain on x
    outer_domain = mx.fourier(origin=mx.center) * my.constant
    outer_codomain = mx.fourier(origin=mx.center) * my.constant
    with pytest.raises(SpaceMismatchError, match="cannot compose"):
        compose_spaces(inner_domain, inner_codomain,
                       outer_domain, outer_codomain)


def test_compose_rejects_a_rank_mismatch(mesh_2d):
    mx, my = mesh_2d
    one_d = mx.fourier(origin=mx.center)
    two_d = mx.fourier(origin=mx.center) * my.constant
    with pytest.raises(SpaceMismatchError,
                       match="incompatible spaces"):
        compose_spaces(one_d, one_d, two_d, two_d)


def test_compose_matches_the_symbol_matmul(mesh_1d):
    # cross-check against the Symbol algebra it backs
    center = mesh_1d.fourier(origin=mesh_1d.center)
    right = mesh_1d.fourier(origin=mesh_1d.right)
    ones = jnp.ones(center.shape[0], dtype=jnp.complex128)
    fwd = Symbol(center, ones, codomain=right)
    bwd = Symbol(right, ones, codomain=center)
    composed = bwd @ fwd
    domain, codomain = compose_spaces(
        fwd.space, fwd.codomain, bwd.space, bwd.codomain)
    assert composed.space is domain
    assert composed.codomain is codomain
