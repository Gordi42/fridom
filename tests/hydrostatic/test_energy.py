"""The hydrostatic energy-weight builder (the model's diag(M))."""
import numpy as np

import fridom as fr
import fridom.hydrostatic as hy

IM = fr.spatial.meshes.IntervalMesh


def test_weights_are_diag_1_1_inv_n2_inv_csqr():
    weights = hy.energy.hydrostatic_energy_weights(0.5, 0.25)
    assert set(weights) == {"u", "v", "b", "ps"}
    assert weights["u"] == 1.0
    assert weights["v"] == 1.0
    assert weights["b"] == 0.5
    assert weights["ps"] == 0.25


def test_weights_accept_a_profile_field_for_inv_n2():
    grid = fr.spatial.Grid((
        IM(4, (0.0, 1.0), periodic=True, name="x"),
        IM(4, (0.0, 1.0), periodic=True, name="y"),
        IM(4, (0.0, 1.0), periodic=False, name="z")))
    inv_n2 = grid.create_field(
        fr.spatial.Collocated().resolve(grid),
        init=lambda x, y, z: 1.0 + 0.0 * (x + y + z))
    weights = hy.energy.hydrostatic_energy_weights(inv_n2, 0.25)
    assert weights["b"] is inv_n2
    assert np.asarray(weights["b"].data).shape == \
        np.asarray(inv_n2.data).shape
    assert weights["ps"] == 0.25
