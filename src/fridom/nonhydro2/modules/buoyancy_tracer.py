r"""Buoyancy as a bare tracer: ``b`` with no background stratification.

Description
-----------
``BuoyancyTracer`` registers the prognostic buoyancy tracer ``b`` and
contributes the buoyancy force ``+b/delta^2`` in the w-equation —
nothing else. It is the formulation for a buoyancy variable **without**
a background stratification: physically the ``N^2 = 0`` limit of
``ConstantStratification``, but the restoring term ``-N^2 w`` is
absent from the assembly instead of multiplying by zero, so the step
never pays its field arithmetic.

The module is **scaling-neutral** (``fr.scaling``): it carries no
physics kwarg, so it adopts the assembly's variant at bind and provides
no ``stratification.n2`` / ``stratification.froude`` — consumers that
need a background stratification (the ``1/N^2`` energy metric, the
internal-wave eigenmodes) refuse the model through the missing provide
with the taught registry hint.

``b`` is advanced by advection alone: a **linear** assembly
(``advection=False``) leaves ``b`` advanced by no term at all and is
rejected by the D1.4 coverage lint. Like the stratification family,
``b`` is declared BC-free on every grid (topology-driven walls, C8).
"""
from __future__ import annotations

import fridom as fr
from fridom.nonhydro2.params import ASPECT_RATIO


class BuoyancyTracer(fr.model.Module):

    r"""Registers ``b``; the buoyancy force alone (no restoring).

    Parameters
    ----------
    family : str | None, optional
        The discretization family of the buoyancy tracer ``b``
        (FV-D1b): ``"fv"`` declares it on the average family
        (``CellAvg^3``), so its flux-form advection conserves total
        buoyancy to machine zero while the nodal velocity/pressure
        state is untouched; ``"nodal"`` keeps it collocated with the
        pressure cell. None defers to the grid-level default (the
        rest of the model) (default: None).
    """

    def __init__(self, *, family: str | None = None) -> None:
        """Store the tracer's discretization family."""
        self._family = family

    @property
    def field_declarations(
        self,
    ) -> tuple[fr.model.FieldDeclaration, ...]:
        """The buoyancy tracer ``b`` on the requested family."""
        return (
            fr.model.FieldDeclaration.tracer(
                "b", space=fr.spatial.Collocated(family=self._family),
                long_name="Buoyancy", units="m/s^2"),
        )

    field_references = (
        fr.model.FieldReference(
            "w", hint="buoyancy couples to vertical velocity, "
                      "declared by a dynamical core (nh.Core)"),
    )
    parameter_references = (
        fr.model.ParameterReference(
            ASPECT_RATIO, hint="declared by nh.Core"),
    )

    @fr.model.term(advances=("w",), linear=True,
                   linear_params=(ASPECT_RATIO,))
    def buoyancy_force(self, state, ctx) -> dict:  # noqa: ANN001
        """``dw/dt += b / delta^2`` (buoyancy interpolated onto w).

        The one coupling term of the formulation: the aspect ratio
        lives on the core module and is squared at this use site.
        There is deliberately no restoring counterpart — ``b`` carries
        no background stratification.
        """
        delta = ctx.params[ASPECT_RATIO]
        return {"w": state["b"].to(state["w"]) / (delta * delta)}
