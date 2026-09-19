"""Known-valid halo axes survive repair of another tensor factor."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.decomposition.tensor import TensorDecomposition
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.spaces.nodal import NodeSet


@pytest.mark.multi_device
@pytest.mark.parametrize("periodic_y", [False, True])
@pytest.mark.parametrize("materialize", [False, True])
def test_valid_axis_needs_no_exchange_while_other_axis_is_repaired(
        periodic_y, materialize):
    mx = IntervalMesh(32, (0., 1.), name="x")
    my = IntervalMesh(12, (0., 1.), periodic=periodic_y, name="y")
    fy = my.center if periodic_y else my.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    space = mx.center * fy
    layout = Layout({"x": "devices"})
    decomp = TensorDecomposition(
        (mx, my), ("x", "y"), HaloSpec({"x": 3, "y": 3}), (layout,))
    values = jnp.asarray(np.random.default_rng(322).normal(size=space.shape))
    full = decomp.sync(decomp.pad(values, space, layout), space, layout=layout)
    # Corrupt only y ghosts, consistently along x: x's wrap stays valid.
    bad = full.at[:, :3].set(71.).at[:, -3:].set(-9.)
    valid = HaloSpec({"x": 3, "y": 0})
    fn = jax.jit(lambda arr: decomp.sync(
        arr, space, layout=layout, valid=valid, materialize=materialize))
    compiled = fn.lower(bad).compile()
    np.testing.assert_array_equal(compiled(bad), full)
    hlo = compiled.as_text().lower()
    assert "collective-permute(" not in hlo
    assert "all-gather(" not in hlo


@pytest.mark.multi_device
@pytest.mark.parametrize("valid", [
    None, HaloSpec({}), HaloSpec({"x": (3, 2)}),
])
def test_incomplete_validity_still_exchanges_both_sides(valid):
    mx = IntervalMesh(32, (0., 1.), name="x")
    layout = Layout({"x": "devices"})
    decomp = TensorDecomposition(
        (mx,), ("x",), HaloSpec({"x": 3}), (layout,))
    space = mx.center
    values = jnp.asarray(np.random.default_rng(11).normal(size=space.shape))
    padded = decomp.pad(values, space, layout)
    expected = decomp.sync(padded, space, layout=layout)
    actual = jax.jit(lambda arr: decomp.sync(
        arr, space, layout=layout, valid=valid))(padded)
    np.testing.assert_array_equal(actual, expected)
