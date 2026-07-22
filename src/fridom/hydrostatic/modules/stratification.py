r"""Stratification module: the buoyancy tracer and its restoring.

Description
-----------
``ConstantStratification`` registers the buoyancy tracer ``b`` and
contributes the **single** linear restoring term. Unlike the
nonhydrostatic twin it carries **no** ``buoyancy_force`` term
(``+b/delta^2`` in a ``dw/dt`` equation): the hydrostatic model has
no vertical momentum equation, and hydrostatic balance
(``d_z p_hyd = b``, the ``hy.Core`` DIAGNOSE) replaces it. The
energy exchange KE <-> PE flows through the ``p_hyd`` gradient on the
momentum and this restoring on the buoyancy, and is exactly
skew-adjoint under the ``diag(1, 1, 1/N^2, 1/g)`` energy metric
because the diagnosed ``w`` lives on the both-boundary face set
(the surface DOF ``w(0)``) and ``p_hyd`` is the half-cell center form
(``hy.energy``).

The two mutually-exclusive constructor kwarg sets fix the **scaling
variant** (``fr.scaling``) at construction:

- **dimensional** (``n2=`` [1/s^2]): the restoring is the verbatim
  ``-N^2 w`` and the module provides ``stratification.n2``;
- **nondimensional** (``froude_number=``, the internal Froude number
  ``Fr = U/(N H)``): the restoring is ``-(eps/Fr)^2 w`` with the live
  ratio read from ``ctx.params`` at stage time, and the module
  provides ``stratification.froude``. As the ``internal_wave``
  mechanism owner, the assembly aliases ``scaling.nonlinearity`` onto
  this leaf under ``fr.scaling.InternalWave()`` (the ratio then
  self-normalizes to an exact ``1.0``).

The restoring interpolates the diagnosed ``w`` (on the vertical
``Outer`` faces) onto the ``b`` cell centres via ``w.to(b)`` — the
registered ``Outer -> Center`` interpolation, the adjoint of the
half-cell hydrostatic-pressure pairing.

The stored ``w`` is the **physical** vertical velocity on every grid
(``physical_state_components.md`` ruling (b)): on a terrain-following
sigma column ``w = J\omega + u\,Z_x + v\,Z_y`` already carries the
slope-advection terms (added by ``hy.Core._diagnose_w``), so
adiabatic buoyancy is coupled to the physical vertical velocity by
the plain restoring here — no terrain branch.
"""
from __future__ import annotations

import numbers
from functools import partial

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.hydrostatic.units import (
    stratification_factor,
    vertical_extent,
)


@partial(jaxify, dynamic=("n2", "froude_number"))
class ConstantStratification(fr.model.Module):

    r"""Registers ``b``; the linear restoring term (dual variants).

    Parameters
    ----------
    n2 : float | fr.model.Ramp | None, optional
        The constant squared buoyancy frequency ``N^2`` [1/s^2]
        (DIMENSIONAL variant); published as ``stratification.n2``.
        May be an ``fr.model.Ramp`` for a spun-up stratification
        (default: None).
    froude_number : float | fr.model.Ramp | None, optional
        The internal Froude number :math:`\mathrm{Fr} = U/(N H)`
        (NONDIMENSIONAL variant); published as
        ``stratification.froude`` and — as the ``internal_wave``
        mechanism owner — aliased by the assembly onto
        ``scaling.nonlinearity`` under ``fr.scaling.InternalWave()``.
        Must be nonzero (the live ratio divides by it)
        (default: None).
    """

    #: fr.scaling traits: this family owns the internal-wave mechanism
    scaling_mechanism = "internal_wave"
    nonlinearity_attr = "froude_number"

    def __init__(
        self,
        n2: float | fr.model.Ramp | None = None,
        *,
        froude_number: float | fr.model.Ramp | None = None,
    ) -> None:
        """Store the variant's leaf (exactly one kwarg set).

        Raises
        ------
        TypeError
            If both or neither of ``n2``/``froude_number`` are
            given, or ``froude_number`` is exactly zero.
        """
        if (n2 is None) == (froude_number is None):
            raise TypeError(
                "ConstantStratification takes exactly one kwarg "
                "set: DIMENSIONAL n2= (the physical N^2 [1/s^2], "
                "zero scaling ops in the trace) XOR NONDIMENSIONAL "
                "froude_number= (the internal Froude number "
                "Fr = U/(N H), under a nondimensional fr.scaling "
                f"policy); got n2={n2!r}, "
                f"froude_number={froude_number!r}")
        if (isinstance(froude_number, numbers.Number)
                and float(froude_number) == 0.0):
            raise TypeError(
                "ConstantStratification froude_number=0 is refused: "
                "the restoring carries the live ratio (eps/Fr)^2, "
                "which divides by it; pass a nonzero Froude number")
        self.n2 = None if n2 is None else fr.model.leaf(n2)
        self.froude_number = (None if froude_number is None
                              else fr.model.leaf(froude_number))
        self._nondim: bool = froude_number is not None
        # the vertical mesh extent H (the model.units vertical scale;
        # see fridom.hydrostatic.units), captured at bind
        self._vertical_extent: float = 1.0

    def bind(self, table: object) -> None:
        """Capture the vertical mesh extent ``H`` (``model.units``).

        Description
        -----------
        The ``N_dim`` row inverts ``Fr_int = U/(N H)`` for ``N`` with
        ``H`` the vertical mesh extent (the flat-only vertical
        convention of :mod:`fridom.hydrostatic.units`). The module
        takes no ``vertical=`` kwarg, so the vertical is discovered
        from the referenced fields: the one axis the diagnosed ``w``
        (vertical ``Outer`` faces) is staggered against ``b`` on.

        Parameters
        ----------
        table : object
            The binding table (carries the grid and the fields).
        """
        grid = table.grid
        w_space = table["w"].space
        b_space = table["b"].space
        vertical = next(
            (axis for axis in grid.names
             if w_space.factor(axis) is not b_space.factor(axis)),
            None)
        if vertical is None:  # pragma: no cover — w rides the
            # vertical Outer faces on every hydrostatic core
            raise ValueError(
                "ConstantStratification found no vertical axis: the "
                "diagnosed w is not staggered against b on any grid "
                "coordinate")
        self._vertical_extent = vertical_extent(grid, vertical)

    @property
    def unit_factors(self) -> dict[str, fr.model.UnitFactor]:
        """The derived ``N_dim`` row (``model.units``, §D)."""
        return {"N_dim": stratification_factor(self._vertical_extent)}

    @property
    def scaling_variant(self) -> str:
        """The constructor-fixed variant (``fr.scaling`` seam)."""
        return "nondimensional" if self._nondim else "dimensional"

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
                      "(hy.Core)"),
    )

    @property
    def parameter_declarations(
        self,
    ) -> tuple[fr.model.ParameterDeclaration, ...]:
        """``stratification.n2`` (dim) / ``.froude`` (nondim)."""
        if self._nondim:
            return (fr.model.ParameterDeclaration(
                fr.model.params.STRATIFICATION_FROUDE,
                attr="froude_number", units="1",
                doc="internal Froude number (the internal-wave "
                    "mechanism)"),)
        return (fr.model.ParameterDeclaration(
            fr.model.params.STRATIFICATION_N2, attr="n2",
            units="1/s^2",
            doc="squared buoyancy frequency N^2"),)

    # ================================================================
    #  Tendency term (linear buoyancy restoring)
    # ================================================================
    @fr.model.term(advances=("b",), linear=True,
                   linear_params=(fr.model.params.STRATIFICATION_N2,
                                  fr.model.params.STRATIFICATION_FROUDE,
                                  fr.model.params.SCALING_NONLINEARITY))
    def restoring(self, state, ctx) -> dict:  # noqa: ANN001
        r"""``db/dt += -N^2 w`` xor ``-(eps/Fr)^2 w`` (onto the b cell).

        Dimensional: the verbatim ``-N^2 w`` (the stored ``w`` is the
        physical vertical velocity on every grid — flat, stretched,
        terrain — so a single spelling is correct everywhere).
        Nondimensional: the live mechanism ratio
        :math:`(\varepsilon/\mathrm{Fr})^2` (stage-time ``ctx.params``
        reads; under the matching ``InternalWave`` scaling the alias
        row makes the ratio an exact ``1.0`` — the x/x
        self-normalization).
        """
        if not self._nondim:
            n2 = ctx.params[fr.model.params.STRATIFICATION_N2]
            return {"b": -(n2 * state["w"].to(state["b"]))}
        eps = ctx.params[fr.model.params.SCALING_NONLINEARITY]
        froude = ctx.params[fr.model.params.STRATIFICATION_FROUDE]
        ratio = eps / froude
        return {"b": -((ratio * ratio) * state["w"].to(state["b"]))}
