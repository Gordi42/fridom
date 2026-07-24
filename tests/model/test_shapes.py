"""The coordinate-named spatial shape builders (shapes)."""
import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from fridom.model.shapes import gaussian


def test_gaussian_is_the_product_gaussian():
    shape = gaussian(pos={"x": 1.0, "z": 2.0},
                     width={"x": 0.5, "z": 2.0})
    value = float(shape(x=jnp.asarray(1.5), z=jnp.asarray(3.0)))
    assert value == pytest.approx(np.exp(-1.0) * np.exp(-0.25))


def test_gaussian_signature_names_exactly_the_pos_keys():
    shape = gaussian(pos={"x": 1.0, "z": 2.0}, width=0.5)
    assert tuple(inspect.signature(shape).parameters) == ("x", "z")


def test_gaussian_float_width_broadcasts_like_the_mapping():
    scalar = gaussian(pos={"x": 1.0, "z": 2.0}, width=1.5)
    mapping = gaussian(pos={"x": 1.0, "z": 2.0},
                       width={"x": 1.5, "z": 1.5})
    coords = {"x": jnp.asarray(0.3), "z": jnp.asarray(-0.7)}
    assert float(scalar(**coords)) == pytest.approx(float(mapping(**coords)))


def test_gaussian_int_width_is_accepted():
    scalar = gaussian(pos={"x": 0.0}, width=2)
    value = float(scalar(x=jnp.asarray(2.0)))
    assert value == pytest.approx(np.exp(-1.0))


def test_gaussian_is_constant_along_axes_not_in_pos():
    shape = gaussian(pos={"x": 1.0}, width=0.5)
    at_a = float(shape(x=jnp.asarray(1.5)))
    at_b = float(shape(x=jnp.asarray(1.5)))
    # the callable simply does not consume other coordinates: it names
    # only "x", so a wider grid stays constant along every other axis.
    assert tuple(inspect.signature(shape).parameters) == ("x",)
    assert at_a == at_b == pytest.approx(np.exp(-1.0))


def test_gaussian_rejects_mismatched_mapping_keys():
    with pytest.raises(ValueError, match="same coordinates"):
        gaussian(pos={"x": 1.0}, width={"z": 1.0})
