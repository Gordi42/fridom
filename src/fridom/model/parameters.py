"""
Parameter declarations, references, and constructor-slot sentinels.

Description
-----------
Owning class doc: ``design/specs/model/classes/declarations.md``
(sections "ParameterDeclaration", "ParameterReference (and
REQUIRED)", "fr.Param and USE_PROVIDED"). All of these are plain
frozen host objects — never pytrees, never in the carry, never
reaching jit. They are the *vocabulary* consumed by the assembly
pipeline (model.md steps 1-2), which builds the binding table
``{name: (module_slot, attr)}`` and runs every validity check
(one-provider-per-name, provided-parameters-must-be-dynamic, the
dotted-name lint, the ``no_default`` policy); no resolution logic
lives here.
"""
# Wave 2 B: ParameterDeclaration, ParameterReference, REQUIRED,
#           Param, USE_PROVIDED
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, NamedTuple, final

import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.model.time_dependent import TimeDependent


# ================================================================
#  Scalar-leaf coercion
# ================================================================
def leaf(value: float | TimeDependent) -> object:
    r"""
    Coerce a scalar parameter to a dynamic leaf; pass curves through.

    Description
    -----------
    The one shared coercion for module parameter slots: a plain
    number becomes a real-dtype ``jnp`` array (a dynamic leaf), while
    a `TimeDependent` value (an ``fr.Ramp``) rides through untouched
    as its own pytree — so a slot spelled ``fr.leaf(...)`` accepts
    both a swept float and a ramped curve without each module
    re-implementing the branch (and without the ``jnp.asarray``
    inline form silently choking on a Ramp).

    Parameters
    ----------
    value : float | TimeDependent
        The scalar parameter value, or an ``fr.Ramp`` curve.

    Returns
    -------
    object
        The `TimeDependent` value unchanged, or ``value`` coerced to
        a real-dtype array leaf.
    """
    if isinstance(value, TimeDependent):
        return value
    return jnp.asarray(value, dtype=dtype_real())


# ================================================================
#  Sentinels
# ================================================================
@final
class _Sentinel:

    """
    A named singleton marker with a self-describing repr.

    Description
    -----------
    Host-side identity object (compare with ``is``); the module-level
    instances below are the only ones ever created.

    Parameters
    ----------
    name : str
        The importable constant name (round-tripping repr).
    description : str
        One-line semantics shown in the repr.
    """

    __slots__ = ("_description", "_name")

    def __init__(self, name: str, description: str) -> None:
        """Store the name and description; see the class docstring."""
        self._name = name
        self._description = description

    def __repr__(self) -> str:
        """Self-describing repr, e.g. ``<REQUIRED: no default; ...>``."""
        return f"<{self._name}: {self._description}>"


REQUIRED: Final[_Sentinel] = _Sentinel(
    "REQUIRED",
    "no default; an unsatisfied reference is a MissingParameterError "
    "at assembly")
"""Sentinel default of a `ParameterReference`/`Param`: no default."""

USE_PROVIDED: Final[_Sentinel] = _Sentinel(
    "USE_PROVIDED",
    "resolve this constructor slot through the parameter binding "
    "table instead of an owned value")
"""Sentinel forcing a normally-owned slot through the binding table."""


# ================================================================
#  ParameterDeclaration
# ================================================================
@dataclass(frozen=True)
class ParameterDeclaration:

    """
    Publish a module scalar: ``name -> (owner slot, attr)``.

    Description
    -----------
    A live-leaf *accessor*, never a frozen value: ``attr`` names the
    owner attribute holding the live value (a dynamic leaf), read
    from the carry at use time through the assembly-built binding
    table and delivered fresh per stage via ``ctx.params`` (D2.1).
    Scalars (and small static-shape arrays) only — spatially varying
    parameters are AUXILIARY fields. Provides implies constancy: a
    module provides a scalar only when that scalar is the whole
    truth (02_rules). Assembly checks one provider per name and that
    the provided attribute is a dynamic leaf.

    Parameters
    ----------
    name : str
        The canonical dotted name (`ParamName` welcome); parameters
        are dotted by physics concept (assembly lint).
    attr : str, optional
        The owner attribute holding the live value; keyword in use
        (default: "").
    units : str, optional
        Unit string, documentation only (default: "n/a").
    doc : str, optional
        One-line description for ``model.report`` (default: "").
    """

    name: str
    attr: str = ""
    units: str = "n/a"
    doc: str = ""


# ================================================================
#  ParameterReference
# ================================================================
class ParameterReference(NamedTuple):

    """
    A consumer's checked claim on a scalar it does not own.

    Description
    -----------
    The exact twin of ``FieldReference``, with the one divergence: a
    physically-identity ``default``. Declared in
    ``Module.parameter_references``; checked and frozen at assembly.
    An unsatisfied `REQUIRED` reference is a ``MissingParameterError``
    attributed to the requiring module; the hint falls back to the
    canonical-name registry's hint. Defaults are for
    physically-identity values only (``scaling.rossby`` -> 1.0,
    forcing amplitudes -> 0); a default on a registry name marked
    ``no_default`` is an assembly error.

    Parameters
    ----------
    name : str
        The canonical dotted name (`ParamName` welcome).
    hint : str, optional
        Provider hint for the MissingParameterError (default: "").
    default : Any, optional
        Physically-identity fallback value (default: `REQUIRED`).
    """

    name: str
    hint: str = ""
    default: Any = REQUIRED


# ================================================================
#  Param (reference-valued constructor slots)
# ================================================================
@dataclass(frozen=True)
class Param:

    """
    The declaration spelling of a defaulted reference in a slot.

    Description
    -----------
    A constructor slot whose default is
    ``fr.Param("scaling.rossby", default=1.0)`` declares a
    ``ParameterReference(name, default=default)`` **only when the
    caller leaves the slot untouched**; an explicit number or
    ``fr.Ramp`` is an owned value — no reference declared
    (explicit-wins, D2 reconciliation 4). `USE_PROVIDED` is the
    converse sentinel. The binding layer (model.md) performs the
    conversion; module code reads the slot uniformly through
    ``resolve_at``/``ctx.params``.

    Parameters
    ----------
    name : str
        The canonical dotted name (`ParamName` welcome).
    default : Any, optional
        Physically-identity fallback value (default: `REQUIRED`).
    """

    name: str
    default: Any = REQUIRED
