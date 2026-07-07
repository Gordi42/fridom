"""Coriolis modules: f-plane and beta-plane.

Description
-----------
``FPlaneCoriolis`` owns a constant ``f0`` leaf and **provides**
``coriolis.f0`` (provides-implies-constancy: analytic consumers such as
``nh.eigenmodes.from_model`` rely on the provide as the constancy
check). The linear coupling ``+f v`` / ``-f u`` scales an interpolated
velocity by the scalar ``f0``; the module declares its interpolation
stencil through ``extra_halo`` (the sanctioned raw-``.data`` scaling
bypass, V-N2 — the tracer forbids field-times-traced-scalar and the
landed product dispatch rejects ConstantSpace operands).

``BetaPlaneCoriolis`` holds ``f0``/``beta`` and the genuinely spatial
AUXILIARY field ``f_coriolis = f0 + beta*y`` — it must **not** provide
``coriolis.f0``.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import dtype_real, jaxify
from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.model.declarations import (
    FieldDeclaration,
    FieldReference,
    Lifecycle,
)
from fridom.framework2.model.module import Module
from fridom.framework2.model.parameters import ParameterDeclaration
from fridom.framework2.model.params import CORIOLIS_BETA, CORIOLIS_F0
from fridom.framework2.model.space_patterns import Collocated
from fridom.framework2.model.terms import term

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.fields.scalar_field import ScalarField
    from fridom.framework2.model.context import StepContext

_U_HINT = ("velocities are declared by a dynamical-core module, "
           "e.g. nh.DynamicalCore")


@partial(jaxify, dynamic=("f0",))
class FPlaneCoriolis(Module):

    """Constant-f Coriolis; provides ``coriolis.f0``.

    Parameters
    ----------
    f0 : float, optional
        The (constant) Coriolis parameter (default: 1.0).
    """

    def __init__(self, f0: float = 1.0) -> None:
        """Store the constant Coriolis leaf."""
        self.f0 = jnp.asarray(f0, dtype=dtype_real())
        self._coords: tuple[str, ...] = ()

    field_references = (
        FieldReference("u", hint=_U_HINT),
        FieldReference("v", hint=_U_HINT),
    )
    parameter_declarations = (
        ParameterDeclaration(CORIOLIS_F0, attr="f0", units="1/s",
                             doc="constant Coriolis parameter"),
    )

    def bind(self, table: object) -> None:
        """Capture the grid coordinate names (halo exemption)."""
        self._coords = tuple(
            axis for _, axis in table.velocity().labels)

    @property
    def extra_halo(self) -> HaloSpec:
        """The interpolation stencil (the raw-``.data`` bypass)."""
        return HaloSpec(dict.fromkeys(self._coords, 1))

    @term(advances=("u", "v"), linear=True)
    def coriolis(
        self, state: object, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        """``du/dt += f v``; ``dv/dt += -f u`` (interpolated)."""
        f0 = ctx.params[CORIOLIS_F0]
        v_at_u = state["v"].to(state["u"].function_space)
        u_at_v = state["u"].to(state["v"].function_space)
        return {
            "u": v_at_u.with_data(f0 * v_at_u.data),
            "v": u_at_v.with_data(-f0 * u_at_v.data),
        }


@partial(jaxify, dynamic=("f0", "beta"))
class BetaPlaneCoriolis(Module):

    """Beta-plane Coriolis ``f(y) = f0 + beta*y`` (an AUXILIARY field).

    Description
    -----------
    Declares ``f_coriolis`` on a ``Profile("y")`` space and does
    **not** provide ``coriolis.f0`` (the provides-implies-constancy
    rule); the coupling scales by the interpolated field via
    ``extra_halo``.

    Parameters
    ----------
    f0 : float, optional
        Reference Coriolis parameter at ``y = 0`` (default: 1.0).
    beta : float, optional
        Meridional gradient ``df/dy`` (default: 0.0).
    meridional : str, optional
        The meridional coordinate name (default: ``"y"``).
    """

    def __init__(
        self, f0: float = 1.0, beta: float = 0.0,
        *, meridional: str = "y",
    ) -> None:
        """Store the leaves and the meridional coordinate name."""
        self.f0 = jnp.asarray(f0, dtype=dtype_real())
        self.beta = jnp.asarray(beta, dtype=dtype_real())
        self._meridional = meridional
        self._coords: tuple[str, ...] = ()

    field_references = (
        FieldReference("u", hint=_U_HINT),
        FieldReference("v", hint=_U_HINT),
    )
    parameter_declarations = (
        ParameterDeclaration(CORIOLIS_BETA, attr="beta",
                             units="1/(m s)",
                             doc="meridional Coriolis gradient"),
    )

    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """The ``f(y)`` auxiliary field on a meridional profile."""
        return (
            FieldDeclaration(
                "f_coriolis", space=Collocated(),
                lifecycle=Lifecycle.AUXILIARY,
                default=type(self)._f_default,  # noqa: SLF001
                long_name="Coriolis parameter", units="1/s"),
        )

    def _f_default(self, grid: object, space: object) -> ScalarField:
        """Materialize ``f0 + beta*y`` (unbound owner-method default)."""
        f0, beta, mer = self.f0, self.beta, self._meridional

        def init(x: object, y: object, z: object) -> object:
            return f0 + beta * {"x": x, "y": y, "z": z}[mer]

        return grid.create_field(space, init=init)

    def bind(self, table: object) -> None:
        """Capture the grid coordinate names (halo exemption)."""
        self._coords = tuple(
            axis for _, axis in table.velocity().labels)

    @property
    def extra_halo(self) -> HaloSpec:
        """The interpolation stencil (the raw-``.data`` bypass)."""
        return HaloSpec(dict.fromkeys(self._coords, 1))

    @term(advances=("u", "v"), linear=True)
    def coriolis(
        self, state: object, ctx: StepContext,  # noqa: ARG002
    ) -> dict[str, ScalarField]:
        """``du/dt += f(y) v``; ``dv/dt += -f(y) u``."""
        f = state["f_coriolis"]
        u_space = state["u"].function_space
        v_space = state["v"].function_space
        f_u = f.to(u_space).data
        f_v = f.to(v_space).data
        v_at_u = state["v"].to(u_space)
        u_at_v = state["u"].to(v_space)
        return {
            "u": v_at_u.with_data(f_u * v_at_u.data),
            "v": u_at_v.with_data(-f_v * u_at_v.data),
        }
