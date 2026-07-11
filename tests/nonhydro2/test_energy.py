"""The nonhydro energy-weight builder (``diag(1, 1, dsqr, 1/N^2)``)."""
from fridom.nonhydro2.energy import nonhydro_energy_weights


def test_nonhydro_energy_weights_builder():
    assert nonhydro_energy_weights(2.0, 0.25) == {
        "u": 1.0, "v": 1.0, "w": 2.0, "b": 0.25}
