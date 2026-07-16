r"""Named analytic initial conditions (``hy.initial_conditions``).

Description
-----------
A small starter set of hydrostatic initial-condition factories: plain
functions taking a model and returning a
:class:`~fridom.hydrostatic.state.State` the caller assigns via
``model.set_state(...)``. The eigenmode-sourced random-phase families
of the shallow-water / nonhydrostatic packages are the H4 deliverable
(``hy.eigenmodes``) and are not built here.

- :func:`single_wave` — one horizontal Fourier mode with a cosine
  vertical structure in the buoyancy (a baroclinic seed);
- :func:`jet` — a barotropic zonal jet with the balancing surface
  pressure (a geostrophic-adjustment seed).
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.hydrostatic.state import State
from fridom.spatial.spaces.constant import ConstantSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.model.model import Model
    from fridom.spatial.fields.scalar_field import ScalarField


def _sample(model: Model, name: str, fn: Callable) -> ScalarField:
    """Materialize ``fn`` on ``name``'s non-constant coordinate nodes."""
    space = model.state[name].function_space
    var = tuple(
        coord for factor in space.bare.factors
        if not isinstance(factor, ConstantSpace)
        for coord in factor.names)

    def init(**coords: object) -> object:
        return fn(**coords)

    init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(coord, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for coord in var])
    return model.grid.create_field(space, init=init, name=name)


def _vertical_extent(model: Model, vertical: str) -> tuple[float, float]:
    """Return the ``(min, max)`` extent of the vertical mesh factor."""
    for mesh in model.grid.factors:
        if vertical in mesh.names:
            lo, hi = mesh.extent
            return float(lo), float(hi)
    msg = f"the grid has no {vertical!r} coordinate"
    raise ValueError(msg)


def single_wave(
    model: Model,
    *,
    wavenumbers: tuple[int, int, int] = (1, 1, 1),
    amplitude: float = 1.0e-3,
    vertical: str = "z",
) -> State:
    r"""Return one horizontal Fourier mode with a cosine vertical structure.

    Description
    -----------
    Seeds the buoyancy with
    ``A sin(2 pi k_x x) sin(2 pi k_y y) cos(m pi (z - z_0) / H)`` and
    leaves ``u, v, ps`` at rest — a baroclinic perturbation the linear
    model disperses into internal waves. Returns a
    :class:`State`; assign with ``model.set_state(...)``.

    Parameters
    ----------
    model : Model
        The assembled hydrostatic model (its grid sizes the field).
    wavenumbers : tuple[int, int, int], optional
        The ``(k_x, k_y, m)`` mode numbers (horizontal cycles over the
        periodic box, vertical half-waves over the depth)
        (default: (1, 1, 1)).
    amplitude : float, optional
        The buoyancy amplitude (default: 1e-3).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``).

    Returns
    -------
    State
        The initial state (only ``b`` perturbed).
    """
    kx, ky, m = wavenumbers
    z0, z1 = _vertical_extent(model, vertical)
    depth = z1 - z0

    def b_fn(x, y, z):  # noqa: ANN001, ANN202
        return (amplitude
                * jnp.sin(2.0 * jnp.pi * kx * x)
                * jnp.sin(2.0 * jnp.pi * ky * y)
                * jnp.cos(m * jnp.pi * (z - z0) / depth))

    b = _sample(model, "b", b_fn)
    return State(model.state.replace(b=b))


def jet(
    model: Model,
    *,
    amplitude: float = 0.1,
    width: float = 0.15,
    meridional: str = "y",
) -> State:
    r"""Return a barotropic zonal jet with a balancing surface pressure.

    Description
    -----------
    Seeds ``u`` with a Gaussian zonal jet centred in the channel and
    ``ps`` with the geostrophically consistent surface-pressure hump
    (:math:`\partial_y p_s = f u` is only exact once the model
    adjusts; here ``ps`` is the depth-independent hump that makes the
    jet close to balance). ``v, b`` start at rest. Returns a
    :class:`State`; assign with ``model.set_state(...)``.

    Parameters
    ----------
    model : Model
        The assembled hydrostatic model.
    amplitude : float, optional
        The peak zonal velocity (default: 0.1).
    width : float, optional
        The Gaussian half-width of the jet (default: 0.15).
    meridional : str, optional
        The meridional coordinate name (default: ``"y"``).

    Returns
    -------
    State
        The initial state (``u`` and ``ps`` set).
    """
    y0, y1 = _vertical_extent(model, meridional)
    yc = 0.5 * (y0 + y1)

    def hump(**coords: object) -> object:
        y = coords[meridional]
        return amplitude * jnp.exp(-((y - yc) ** 2) / (2.0 * width ** 2))

    u = _sample(model, "u", hump)
    ps = _sample(model, "ps", hump)
    return State(model.state.replace(u=u, ps=ps))
