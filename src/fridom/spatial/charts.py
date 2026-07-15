"""
Chart presets: ready-made embedding charts for common manifolds.

Description
-----------
Each factory returns a :class:`fridom.spatial.CoordinateMapping` carrying
a standard embedding chart, so a user need not hand-write the embedding
callable or assert its orthogonality. Compose the returned mapping with
any mesh layout via ``fr.spatial.Grid(meshes, mapping=...)``, or reach
for the matching convenience grid (``fr.spatial.spherical.Grid``).
"""
from __future__ import annotations

import jax.numpy as jnp

from fridom.spatial.coordinate_mapping import CoordinateMapping


def lonlat_sphere(radius: float = 1.0) -> CoordinateMapping:
    r"""
    Return the lat-lon sphere embedding chart (orthogonal).

    Description
    -----------
    Returns a :class:`CoordinateMapping` whose embedding chart maps the
    two base coordinates ``lon`` (longitude :math:`\lambda`) and ``lat``
    (latitude :math:`\phi`) to the ambient Cartesian point on a sphere
    of the given radius,

    .. math::

        X(\lambda, \phi) = a\,\bigl(\cos\phi\cos\lambda,\;
        \cos\phi\sin\lambda,\; \sin\phi\bigr).

    The induced metric is diagonal everywhere — meridians and parallels
    are orthogonal, so :math:`g_{\lambda\phi} \equiv 0` — so the mapping
    is declared ``orthogonal=True``. The grid then seeds the diagonal
    index moves that let a vector calculus assemble across the bounded
    latitude walls (and, for a longitude sector, the bounded longitude
    walls too).

    The chart is extent-independent: it composes with any ``lon``/``lat``
    mesh layout. The base coordinates must be named ``lon`` and ``lat``.

    Parameters
    ----------
    radius : float, optional
        The sphere radius :math:`a` (default: 1.0).

    Returns
    -------
    CoordinateMapping
        The orthogonal lat-lon sphere chart, bound on grid attachment.
    """
    return CoordinateMapping(
        chart={"X": lambda lon, lat: (
            radius * jnp.cos(lat) * jnp.cos(lon),
            radius * jnp.cos(lat) * jnp.sin(lon),
            radius * jnp.sin(lat))},
        orthogonal=True)
