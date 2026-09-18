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
        orthogonal=True,
        coordinate_units={"lon": "rad", "lat": "rad"})


def torus(major: float = 2.0, minor: float = 1.0) -> CoordinateMapping:
    r"""
    Return the embedded-torus chart (orthogonal, pole-free, wall-free).

    Description
    -----------
    Returns a :class:`CoordinateMapping` whose embedding chart maps the
    two **periodic** base coordinates ``u`` (toroidal angle) and ``v``
    (poloidal angle) to the ambient Cartesian point on a torus of major
    radius :math:`R` and minor radius :math:`r`,

    .. math::

        X(u, v) = \bigl((R + r\cos v)\cos u,\;
        (R + r\cos v)\sin u,\; r\sin v\bigr).

    The induced metric is diagonal and varies, :math:`g_{uu} =
    (R + r\cos v)^2`, :math:`g_{vv} = r^2`, :math:`\sqrt g =
    r\,(R + r\cos v)`, but — for :math:`R > r` — is **never zero** and
    the manifold is closed in both coordinates, so the chart needs no
    walls. It is the pole-free test chart of the spherical-models plan
    (SP-D3): it exercises the full metric machinery (varying
    :math:`\sqrt g`, non-unit :math:`g_{ii}`, a nonzero scale-factor
    derivative) while isolating metric-term defects from the polar-cap
    wall closures of the lat-lon sphere. The base coordinates must be
    named ``u`` and ``v``.

    Parameters
    ----------
    major : float, optional
        The major radius :math:`R` (default: 2.0).
    minor : float, optional
        The minor radius :math:`r`; must satisfy ``0 < minor < major``
        (default: 1.0).

    Returns
    -------
    CoordinateMapping
        The orthogonal torus chart, bound on grid attachment.

    Raises
    ------
    ValueError
        Unless ``0 < minor < major`` (a ring torus; otherwise the
        metric degenerates on the inner equator).
    """
    if not 0.0 < minor < major:
        raise ValueError(
            f"torus(major={major}, minor={minor}) needs 0 < minor < "
            "major (a ring torus): the metric root "
            "sqrt_g = minor * (major + minor*cos(v)) must stay "
            "positive on the inner equator")
    return CoordinateMapping(
        chart={"X": lambda u, v: (
            (major + minor * jnp.cos(v)) * jnp.cos(u),
            (major + minor * jnp.cos(v)) * jnp.sin(u),
            minor * jnp.sin(v))},
        orthogonal=True,
        coordinate_units={"u": "rad", "v": "rad"})
