"""Unit tests for the shallow-water immersed weighting idioms (IP-D10).

The three helpers of ``shallowwater2.modules.immersed_weighting`` that
the linear ``gravity`` core and the ``SadournyAdvection`` transport
share: ``weight_flux`` (open-area flux weight), ``scale_divergence``
(guarded wet-plan-area division), ``mask_field`` (boolean dry-DOF
zeroing).
"""
import numpy as np

import fridom as fr
from fridom.shallowwater2.modules.immersed_weighting import (
    mask_field,
    scale_divergence,
    weight_flux,
)
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh


def _grid(indicator, *, order=None, min_fraction=0.1):
    mx = IntervalMesh(8, (0.0, 8.0), periodic=True, name="x")
    my = IntervalMesh(8, (0.0, 8.0), periodic=True, name="y")
    return Grid((mx, my),
                immersed=ImmersedDomain(indicator, order=order,
                                        min_fraction=min_fraction))


def _cell_field(grid, data):
    space = fr.spatial.Collocated().resolve(grid)
    return grid.create_field(space, data=data)


def _face_field(grid, data):
    space = fr.spatial.Staggered("x").resolve(grid)
    return grid.create_field(space, data=data)


# ================================================================
#  weight_flux: open-area fraction on the flux's own space
# ================================================================
def test_weight_flux_zeros_closed_faces():
    # dry for x < 2 and x > 6: the x-faces straddling the wet-dry
    # boundary carry alpha = 0 (min-rule), so the weighted flux
    # vanishes there while interior faces (alpha = 1) are untouched
    grid = _grid(lambda x, y: ((x > 2) & (x < 6)).astype(float))  # noqa: ARG005
    immersed = grid.immersed
    flux = _face_field(grid, np.ones((8, 8)))
    weighted = weight_flux(immersed, flux)
    alpha = np.asarray(immersed.fraction(flux.function_space).data)
    got = np.asarray(weighted.data)
    assert np.allclose(got, alpha)
    # a closed face somewhere (the boundary faces of the wet band)
    assert (got == 0.0).any()
    # interior wet faces keep the flux unchanged
    assert (got == 1.0).any()


# ================================================================
#  scale_divergence: guarded 1/theta, dry cell stays exactly zero
# ================================================================
def test_scale_divergence_divides_by_wet_fraction():
    # genuine partials via a smooth linear ramp (order-2 exact),
    # min_fraction=0 so the raw fractions come through
    grid = _grid(lambda x, y: np.clip(1.3 - 0.2 * x, 0.0, 1.0),  # noqa: ARG005
                 order=2, min_fraction=0.0)
    immersed = grid.immersed
    res = _cell_field(grid, np.full((8, 8), 3.0))
    theta = np.asarray(immersed.fraction(res.function_space).data)
    scaled = np.asarray(scale_divergence(immersed, res).data)
    wet = theta > 0.0
    # wet cells: res / theta; dry cells: exactly zero (guarded)
    assert np.allclose(scaled[wet], 3.0 / theta[wet])
    assert np.all(scaled[~wet] == 0.0)


def test_scale_divergence_dry_cell_is_exactly_zero():
    grid = _grid(lambda x, y: (x > 4).astype(float))  # noqa: ARG005
    immersed = grid.immersed
    # a non-zero numerator on a dry cell must not produce inf/nan
    res = _cell_field(grid, np.full((8, 8), 5.0))
    scaled = np.asarray(scale_divergence(immersed, res).data)
    theta = np.asarray(immersed.fraction(res.function_space).data)
    assert np.all(np.isfinite(scaled))
    assert np.all(scaled[theta == 0.0] == 0.0)


# ================================================================
#  mask_field: boolean per-space dry-DOF zeroing
# ================================================================
def test_mask_field_zeros_dry_dofs():
    grid = _grid(lambda x, y: ((x > 2) & (x < 6)).astype(float))  # noqa: ARG005
    immersed = grid.immersed
    field = _cell_field(grid, np.arange(64.0).reshape(8, 8))
    masked = np.asarray(mask_field(immersed, field).data)
    mask = np.asarray(immersed.mask(field.function_space).data)
    assert np.all(masked[~mask] == 0.0)
    assert np.allclose(masked[mask],
                       np.arange(64.0).reshape(8, 8)[mask])
