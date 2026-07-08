"""Constant stratification: buoyancy and the linear coupling.

Description
-----------
``ConstantStratification`` registers the buoyancy tracer ``b`` and
contributes **both** linear coupling terms (D1's driving example):
``+b/dsqr`` in the w-equation (buoyancy force) and ``-N^2 w`` in the
b-equation (restoring). It owns the constant ``n2`` leaf and provides
``stratification.n2``; ``dsqr`` is read from ``ctx.params``. The
interpolation stencils are declared through ``extra_halo`` (the
sanctioned raw-``.data`` coefficient-scaling bypass, V-N2).
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import dtype_real, jaxify
from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.model.declarations import (
    FieldDeclaration,
    FieldReference,
)
from fridom.framework2.model.linear_blocks import (
    Coeff,
    Interp,
    LinearBlock,
    apply_linear_blocks,
)
from fridom.framework2.model.module import Module
from fridom.framework2.model.parameters import (
    ParameterDeclaration,
    ParameterReference,
)
from fridom.framework2.model.params import STRATIFICATION_N2
from fridom.framework2.model.space_patterns import Collocated
from fridom.framework2.model.terms import term
from fridom.nonhydro2.params import DSQR

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.fields.scalar_field import ScalarField
    from fridom.framework2.model.context import StepContext


# The two linear coupling blocks (single source of truth). Buoyancy
# force ``+b/dsqr`` divides by the traced ``ctx.params[DSQR]`` scalar
# (``invert``); restoring ``-N^2 w`` scales by the owned ``n2`` leaf.
# Both interpolate across the staggered face (``Interp``) and read
# their symbolic constant off ``model.parameters``.
_BUOYANCY_BLOCKS = (
    LinearBlock("w", "b", Interp(), Coeff(param=DSQR, invert=True)),
)
_RESTORING_BLOCKS = (
    LinearBlock("b", "w", Interp(),
                Coeff(param=STRATIFICATION_N2, sign=-1)),
)


@partial(jaxify, dynamic=("n2",))
class ConstantStratification(Module):

    """Registers ``b``; contributes both linear coupling terms.

    Parameters
    ----------
    n2 : float, optional
        The constant squared buoyancy frequency ``N^2`` (default: 1.0).
    wall_z : bool, optional
        Declare a Dirichlet z boundary on ``b`` (default: False — the
        periodic smoke-test configuration; True for a walled box).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``).
    """

    def __init__(
        self, n2: float = 1.0, *, wall_z: bool = False,
        vertical: str = "z",
    ) -> None:
        """Store the stratification leaf and BC/geometry choices."""
        self.n2 = jnp.asarray(n2, dtype=dtype_real())
        self._wall_z = wall_z
        self._vertical = vertical
        self._coords: tuple[str, ...] = ()

    field_references = (
        FieldReference(
            "w", hint="buoyancy couples to vertical velocity, "
                      "declared by a dynamical core (nh.DynamicalCore)"),
    )
    parameter_declarations = (
        ParameterDeclaration(STRATIFICATION_N2, attr="n2",
                             units="1/s^2",
                             doc="squared buoyancy frequency N^2"),
    )
    parameter_references = (
        ParameterReference(DSQR, hint="declared by nh.DynamicalCore"),
    )

    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """The buoyancy tracer (PROGNOSTIC + TRACER + ADVECTED)."""
        bc = ({self._vertical: BC.DIRICHLET} if self._wall_z
              else None)
        return (
            FieldDeclaration.tracer(
                "b", space=Collocated(bc=bc),
                long_name="Buoyancy", units="m/s^2"),
        )

    def bind(self, table: object) -> None:
        """Capture the grid coordinate names (halo exemption)."""
        self._coords = tuple(
            axis for _, axis in table.velocity().labels)

    @property
    def extra_halo(self) -> HaloSpec:
        """The interpolation stencils (the raw-``.data`` bypass)."""
        return HaloSpec(dict.fromkeys(self._coords, 1))

    @term(advances=("w",), linear=True, blocks=_BUOYANCY_BLOCKS)
    def buoyancy_force(
        self, state: object, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        """``dw/dt += b / dsqr`` (interpolated onto the w face)."""
        return apply_linear_blocks(_BUOYANCY_BLOCKS, state, ctx)

    @term(advances=("b",), linear=True, blocks=_RESTORING_BLOCKS)
    def restoring(
        self, state: object, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        """``db/dt += -N^2 w`` (interpolated onto the b cell)."""
        return apply_linear_blocks(_RESTORING_BLOCKS, state, ctx)
