r"""Stratification module: the buoyancy tracer and its restoring.

Description
-----------
``ConstantStratification`` registers the buoyancy tracer ``b`` and
contributes the **single** linear restoring term ``db/dt = -N^2 w``.
Unlike the nonhydrostatic twin it carries **no** ``buoyancy_force``
term (``+b/dsqr`` in a ``dw/dt`` equation): the hydrostatic model has
no vertical momentum equation, and hydrostatic balance
(``d_z p_hyd = b``, the ``HydrostaticCore`` DIAGNOSE) replaces it. The
energy exchange KE <-> PE flows through the ``p_hyd`` gradient on the
momentum and this restoring on the buoyancy, and is exactly
skew-adjoint under the ``diag(1, 1, 1/N^2, 1/c^2)`` energy metric
because the diagnosed ``w`` lives on the both-boundary face set
(the surface DOF ``w(0)``) and ``p_hyd`` is the half-cell center form
(``hy.energy``).

The restoring interpolates the diagnosed ``w`` (on the vertical
``Outer`` faces) onto the ``b`` cell centres via ``w.to(b)`` — the
registered ``Outer -> Center`` interpolation, the adjoint of the
half-cell hydrostatic-pressure pairing.

**Terrain-following column** (a sigma-coordinate grid, :meth:`bind`
captured ``self._column``): on terrain the diagnosed ``w`` is the
**contravariant** vertical volume flux ``J\omega`` (``core._diagnose_w``),
not the physical vertical velocity. Adiabatic buoyancy is advected by
the *physical* vertical velocity

.. math::

    w_{\mathrm{true}} = J\omega + u\,Z_x + v\,Z_y ,

so the terrain restoring couples ``b`` to ``w_true``, adding the
slope-advection half ``-N^2 (u\,Z_x + v\,Z_y)`` (the coordinate-surface
slopes ``Z_i = d<mapped>_d<axis>`` sampled at the ``b`` cell,
interpolating ``u`` / ``v`` onto it). Without it the buoyancy equation
is O(slope)-wrong (the sigma-coordinate internal-wave physics deviates
at first order in the terrain slope) and the KE <-> PE exchange is not
energy-consistent under the physical (Jacobian-weighted) metric — the
half-pair inconsistency of ``design/research/energy_metric_asymmetry.md``.
The term is a plain multiply of finite metric fields (no ``1/J``
division, so no reverse-mode singularity to seal) and vanishes at
rest (``u = v = 0``), so the rest state is preserved. On a flat grid
``self._column`` is ``None`` and the code path is byte-identical to
the plain ``-N^2 w`` form.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.hydrostatic.modules.terrain import (
    discover_column,
    require_chart_immersed_order,
)
from fridom.model.halo_demand import derive_extra_halo

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.decomposition.halo import HaloSpec


@partial(jaxify, dynamic=("n2",))
class ConstantStratification(fr.model.Module):

    r"""Registers ``b``; contributes the linear restoring term.

    Parameters
    ----------
    n2 : float | fr.model.Ramp, optional
        The constant squared buoyancy frequency ``N^2`` (default: 1.0);
        may be an ``fr.model.Ramp`` for a spun-up stratification.
    vertical : str, optional
        The vertical coordinate name — the axis the terrain column is
        discovered on; mirrors ``hy.HydrostaticCore`` (default: ``"z"``).
    horizontal : tuple[str, str], optional
        The (zonal, meridional) coordinate names naming the slope
        metrics ``d<mapped>_d<axis>``; mirrors ``hy.HydrostaticCore``
        (default: ``("x", "y")``).
    """

    def __init__(
        self,
        n2: float | fr.model.Ramp = 1.0,
        *,
        vertical: str = "z",
        horizontal: tuple[str, str] = ("x", "y"),
    ) -> None:
        """Store the stratification leaf and the geometry names."""
        self.n2 = fr.model.leaf(n2)
        self._vertical = vertical
        self._horizontal = tuple(horizontal)
        # captured at bind: the terrain-following column (mapped, base)
        # of a sigma-coordinate grid, or None off a mapped grid (the
        # byte-identical flat / stretched-only path). On a terrain grid
        # the restoring couples b to the physical vertical velocity
        # w_true = Jomega + u*Zx + v*Zy, adding the slope-advection half.
        self._column: tuple[str, str] | None = None
        self._coords: tuple[str, ...] = ()
        # the terrain slope term's derived halo substitute (V-N2): it
        # multiplies grid.metric slope fields and interpolates u/v onto
        # the b cell, which the halo tracer cannot follow (the core /
        # mapped-advection precedent). None off a mapped grid.
        self._extra_halo: HaloSpec | None = None

    @property
    def field_declarations(
        self,
    ) -> tuple[fr.model.FieldDeclaration, ...]:
        """The buoyancy tracer ``b`` (collocated, TRACER + ADVECTED)."""
        return (
            fr.model.FieldDeclaration.tracer(
                "b", space=fr.spatial.Collocated(),
                long_name="Buoyancy", units="m/s^2"),
        )

    field_references = (
        fr.model.FieldReference(
            "w", hint="buoyancy couples to the diagnosed vertical "
                      "velocity, declared by a hydrostatic core "
                      "(hy.HydrostaticCore)"),
        fr.model.FieldReference(
            "u", hint="the terrain slope-advection term reads the "
                      "zonal velocity, declared by a hydrostatic core "
                      "(hy.HydrostaticCore)"),
        fr.model.FieldReference(
            "v", hint="the terrain slope-advection term reads the "
                      "meridional velocity, declared by a hydrostatic "
                      "core (hy.HydrostaticCore)"),
    )
    parameter_declarations = (
        fr.model.ParameterDeclaration(
            fr.model.params.STRATIFICATION_N2, attr="n2",
            units="1/s^2",
            doc="squared buoyancy frequency N^2"),
    )

    # ================================================================
    #  Bind (capture the terrain column and the slope-term halo)
    # ================================================================
    def bind(self, table: object) -> None:
        r"""Capture the terrain column and derive the slope-term halo.

        Description
        -----------
        Discovers the single-base terrain column on the vertical axis
        through
        :func:`~fridom.hydrostatic.modules.terrain.discover_column`
        (``None`` off a mapped grid — the byte-identical flat path).
        A **terrain + immersed** grid (stage M5) composes: the
        slope-advection term reads the masked contravariant ``w``
        (``hy.HydrostaticCore``) and the min-rule-consistent velocities
        (dead DOFs zeroed by ``MaskState``), so it stays mask-respecting
        without an explicit gate. It requires the Jacobian-weighted
        chart fractions
        (:func:`~fridom.hydrostatic.modules.terrain.require_chart_immersed_order`),
        consistent with the core.
        """
        grid = table.grid  # type: ignore[attr-defined]
        self._coords = tuple(grid.names)
        self._column = discover_column(grid, self._vertical)
        require_chart_immersed_order(grid, self._column)
        if self._column is not None:
            self._extra_halo = self._derive_extra_halo(table)

    def _derive_extra_halo(self, table: object) -> HaloSpec:
        r"""Derive the slope term's ghost width (V-N2) from its rows.

        Description
        -----------
        The terrain slope term interpolates ``u`` / ``v`` onto the
        ``b`` cell (``u.to(b)`` — an ``interpolate`` row on the source
        velocity factor) and multiplies ``grid.metric`` slope fields
        the halo tracer cannot materialize. A single dataflow leg with
        one interpolation per horizontal coordinate reaches 1 there and
        0 on the vertical (no column stencil). A registry override of
        the interpolation moves the value (derived, not a literal —
        the core / mapped-advection precedent).
        """
        registry = table.grid.dispatch  # type: ignore[attr-defined]
        u = table["u"].space  # type: ignore[index]
        v = table["v"].space  # type: ignore[index]
        zonal, meridional = self._horizontal
        slope_leg: dict[str, list[tuple[str, object]]] = {
            zonal: [("interpolate", u.factor(zonal))],
            meridional: [("interpolate", v.factor(meridional))],
        }
        return derive_extra_halo(registry, self._coords, [slope_leg])

    @property
    def extra_halo(self) -> HaloSpec | None:
        """Exempt the terrain slope term from the halo trace.

        Description
        -----------
        On a **terrain** grid the slope-advection term multiplies
        ``grid.metric`` slope coefficient fields (``d<mapped>_d<axis>``)
        the halo tracer's ``_TracerGrid`` cannot materialize — the
        shared core / mapped-advection precedent — so the module
        declares its (order-2) interpolation-stencil halo here (1 per
        horizontal coordinate, 0 on the vertical) instead of being
        traced. Off a mapped grid this is ``None`` — the flat restoring
        stays fully halo-traced, bitwise unchanged.
        """
        return self._extra_halo

    # ================================================================
    #  Tendency term (linear buoyancy restoring)
    # ================================================================
    @fr.model.term(advances=("b",), linear=True)
    def restoring(self, state, ctx) -> dict:  # noqa: ANN001
        r"""``db/dt += -N^2 w`` (w interpolated onto the b cell).

        On a terrain-following column ``w`` is the contravariant volume
        flux ``J\omega``, so the restoring couples ``b`` to the physical
        vertical velocity ``w_true = J\omega + u Z_x + v Z_y`` — adding
        the slope-advection half ``-N^2 (u Z_x + v Z_y)`` (finite metric
        multiplies, no ``1/J`` guard needed; vanishes at rest). On a
        flat grid the expression is byte-identical to ``-N^2 w``.
        """
        n2 = ctx.params[fr.model.params.STRATIFICATION_N2]
        b = state["b"]
        w = state["w"].to(b)
        if self._column is None:
            return {"b": -(n2 * w)}
        mapped = self._column[0]
        zonal, meridional = self._horizontal
        grid = b.grid
        bare = b.function_space.bare
        zx = grid.metric(bare, f"d{mapped}_d{zonal}")
        zy = grid.metric(bare, f"d{mapped}_d{meridional}")
        w_true = w + state["u"].to(b) * zx + state["v"].to(b) * zy
        return {"b": -(n2 * w_true)}
