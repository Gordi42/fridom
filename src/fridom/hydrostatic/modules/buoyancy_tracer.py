r"""Buoyancy as a bare tracer: ``b`` with no background stratification.

Description
-----------
``BuoyancyTracer`` registers the prognostic buoyancy tracer ``b`` and
contributes **no term at all**: the buoyancy enters the dynamics only
through the hydrostatic balance ``d_z p_hyd = b`` (the ``hy.Core``
DIAGNOSE), which reads ``b`` and drives the momentum through the
``p_hyd`` gradient. It is the formulation for a buoyancy variable
**without** a background stratification, physically the ``N^2 = 0``
limit of ``ConstantStratification`` but with the restoring term
``-N^2 w`` absent from the assembly instead of multiplying by zero, so
the step never pays its field arithmetic.

Unlike the nonhydrostatic twin (``nh.BuoyancyTracer``) it carries **no**
``buoyancy_force`` term either: the nonhydrostatic model forces the
prognostic vertical velocity with ``+b/delta^2``, whereas the
hydrostatic model has no vertical momentum equation and couples
buoyancy through ``p_hyd`` on the core. This module therefore only
declares ``b``.

``b`` is advanced by advection alone: a **linear** assembly
(``advection=None``) leaves ``b`` advanced by no term at all and is
rejected by the coverage lint (use ``ConstantStratification(n2=0.0)``
for a linear barotropic run, where the ``-N^2 w`` restoring is present
but numerically zero). The module is scaling-neutral (it carries no
physics kwarg) and provides no ``stratification.n2`` /
``stratification.froude``, so consumers that need a background
stratification (the ``1/N^2`` energy metric, the internal-wave
eigenmodes) refuse the model through the missing provide.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom as fr

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.declarations import FieldDeclaration


class BuoyancyTracer(fr.model.Module):

    r"""Registers ``b``; no restoring, no force (the ``N^2 = 0`` case).

    Description
    -----------
    A field-only module: it declares the prognostic buoyancy tracer
    ``b`` and contributes no tendency term. The buoyancy reaches the
    flow through the hydrostatic pressure the core diagnoses from it,
    and ``b`` itself is carried by the advection scheme.

    Parameters
    ----------
    family : str | None, optional
        The discretization family of the tracer (FV-D1b): ``"fv"``
        declares it on the average family (``CellAvg`` cell means —
        the space the conservative flux-form advection and the
        flux-form ALE mesh-velocity correction transport), ``"nodal"``
        on the point-value cells. ``None`` — the default — defers to
        the grid-level default, which ``hy.Model`` sets from
        ``hy.Core(family=...)``, so the tracer follows the core
        without being told (default: None).
    """

    def __init__(self, *, family: str | None = None) -> None:
        """Store the tracer's discretization family."""
        self._family = family

    @property
    def field_declarations(
        self,
    ) -> tuple[FieldDeclaration, ...]:
        """The buoyancy tracer ``b`` on the requested family."""
        return (
            fr.model.FieldDeclaration.tracer(
                "b", space=fr.spatial.Collocated(family=self._family),
                long_name="Buoyancy", units="m/s^2"),
        )
