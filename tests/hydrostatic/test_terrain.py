r"""The terrain-column discovery seam (``hydrostatic/modules/terrain``).

Unit tests for :mod:`fridom.hydrostatic.modules.terrain`: the
single-base sigma column is discovered from the grid's
``CoordinateMapping`` and named into its ``grid.metric`` rows; a flat /
stretched-only grid returns ``None`` (the byte-identical flat path); and
a mapping the hydrostatic terrain support does not model (an embedding
chart, a horizontally-based column) is a taught error.
"""
import jax.numpy as jnp
import pytest

import fridom as fr
from fridom.hydrostatic.modules.terrain import (
    discover_column,
    jacobian_name,
)
from fridom.spatial.coordinate_mapping import CoordinateMapping

IM = fr.spatial.meshes.IntervalMesh
MIM = fr.spatial.meshes.MappedIntervalMesh


def _depth(x, y):
    return 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def _terrain_grid(nz_map=None):
    """Sigma grid ``zp = z * H(x, y)`` (base ``z``, physical ``zp``)."""
    mz = nz_map if nz_map is not None else IM(
        6, (-1.0, 0.0), periodic=False, name="z")
    return fr.spatial.Grid(
        (IM(8, (0.0, 1.0), periodic=True, name="x"),
         IM(8, (0.0, 1.0), periodic=True, name="y"), mz),
        mapping=CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                  params={"H": _depth}))


def _flat_grid():
    return fr.spatial.Grid((
        IM(8, (0.0, 1.0), periodic=True, name="x"),
        IM(8, (0.0, 1.0), periodic=True, name="y"),
        IM(6, (-1.0, 0.0), periodic=False, name="z")))


# ================================================================
#  Discovery
# ================================================================
def test_discover_column_finds_the_sigma_column():
    grid = _terrain_grid()
    assert discover_column(grid, "z") == ("zp", "z")


def test_discover_column_none_off_a_mapped_grid():
    assert discover_column(_flat_grid(), "z") is None


def test_discover_column_none_on_a_stretched_only_grid():
    # a MappedIntervalMesh column (stretched sigma) carries NO
    # CoordinateMapping -- its stretching rides grid.measure, the
    # byte-identical flat path.
    grid = fr.spatial.Grid((
        IM(8, (0.0, 1.0), periodic=True, name="x"),
        IM(8, (0.0, 1.0), periodic=True, name="y"),
        MIM(6, (-1.0, 0.0),
            lambda s: -1.0 + s - 0.3 * jnp.sin(jnp.pi * s), name="z")))
    assert discover_column(grid, "z") is None


def test_metric_row_names():
    assert jacobian_name(("zp", "z")) == "dzp_dz"


# ================================================================
#  Taught errors
# ================================================================
def test_discover_column_rejects_a_non_base_vertical():
    # the analytic column's base is 'z'; asking for 'x' (which is only a
    # slope-coupled coordinate, not the base) is a taught error.
    grid = _terrain_grid()
    with pytest.raises(NotImplementedError, match="not the base"):
        discover_column(grid, "x")


def test_discover_column_rejects_an_embedding_chart():
    grid = fr.spatial.Grid(
        (IM(8, (0.1, 0.9), periodic=False, name="u"),
         IM(8, (0.1, 0.9), periodic=False, name="v")),
        mapping=CoordinateMapping(
            chart={"X": lambda u, v: (u, v, 0.0 * u)}, orthogonal=True))
    with pytest.raises(NotImplementedError, match="embedding chart"):
        discover_column(grid, "u")
