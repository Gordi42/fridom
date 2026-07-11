"""The shallow-water energy-weight builders."""
from fridom.shallowwater2.energy import (
    shallowwater_energy_weights,
    shallowwater_varying_energy_weights,
)


def test_shallowwater_energy_weights_builder():
    assert shallowwater_energy_weights(0.5) == {
        "u": 1.0, "v": 1.0, "p": 0.5}


def test_shallowwater_varying_energy_weights_builder():
    # a pure diag(c^2, c^2, 1) dict constructor: the c^2 weight moves
    # onto the velocities (a sentinel stands in for the profile field)
    csqr = object()
    weights = shallowwater_varying_energy_weights(csqr)
    assert weights["u"] is csqr
    assert weights["v"] is csqr
    assert weights["p"] == 1.0
