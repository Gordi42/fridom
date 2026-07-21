"""
The canonical parameter-name registry (``fr.params``).

Description
-----------
Owning class doc: ``design/specs/model/classes/declarations.md``
(section "ParamName and the fr.params registry"). ``ParamName`` is a
``str`` subclass carrying registry documentation (units, provider
hint, the ``no_default`` mark), so ``params["coriolis.f0"]`` and
``update_parameters({fr.params.CORIOLIS_F0: f0})`` hit the same
mapping key. The registry constants below are the framework-owned
canonical names; package-specific names (``"nonhydro.dsqr"``, ...)
live in the package's own registry module using this same class.
``fr.params`` is immutable module-level data — no process-global
mutable state.
"""
# Wave 2 B: ParamName + the registry constants
from __future__ import annotations

from typing import Final, Self


class ParamName(str):

    """
    A canonical dotted parameter name — typo-proof, still a string.

    Description
    -----------
    Subclasses ``str`` so both design spellings hash and compare as
    the plain dotted name. The extra attributes are documentation
    for host-side error messages and the reference-defaults policy;
    they never enter equality or hashing. Units are documentation,
    never computed with (D2.2).

    Parameters
    ----------
    name : str
        The canonical dotted name; parameters are dotted by physics
        concept, never by module class (the D2.1 dotted-name lint
        applies at construction).
    units : str, optional
        Unit string, documentation only (default: "n/a").
    hint : str, optional
        Provider hint backing host-side ``MissingParameterError``
        messages (default: "").
    no_default : bool, optional
        When True, a reference default on this name is an assembly
        error — a default here would silently change the physics
        (default: False).
    """

    __slots__ = ("_hint", "_no_default", "_units")

    def __new__(
        cls,
        name: str,
        *,
        units: str = "n/a",
        hint: str = "",
        no_default: bool = False,
    ) -> Self:
        """Intern the dotted name; see the class docstring."""
        if "." not in name:
            raise ValueError(
                "parameter names are dotted by physics concept "
                f"(D2.1 namespace discipline); got {name!r}")
        self = super().__new__(cls, name)
        self._units = units
        self._hint = hint
        self._no_default = no_default
        return self

    # ================================================================
    #  Properties (read-only registry documentation)
    # ================================================================
    @property
    def units(self) -> str:
        """Unit string; documentation only, never computed with."""
        return self._units

    @property
    def hint(self) -> str:
        """Provider hint for MissingParameterError messages."""
        return self._hint

    @property
    def no_default(self) -> bool:
        """True if a reference default on this name is an error."""
        return self._no_default


# ================================================================
#  The canonical registry (framework-owned names)
# ================================================================
# The canonical string "stepper.dt" is a spec concretization
# (declarations.md, open question 5) — confirm at 2.4 when the time
# stepper joins the binding table as its provider.
TIME_STEP: Final[ParamName] = ParamName(
    "stepper.dt",
    units="s",
    hint="provided by the time stepper (its dt leaf) at assembly "
         "step 2, e.g. fr.time_steppers.AdamBashforth(dt=...)",
    no_default=True)

CORIOLIS_F0: Final[ParamName] = ParamName(
    "coriolis.f0",
    units="1/s",
    hint="provided by an f-plane Coriolis module, e.g. "
         "fr.modules.FPlaneCoriolis(f0=...)")

CORIOLIS_BETA: Final[ParamName] = ParamName(
    "coriolis.beta",
    units="1/(m s)",
    hint="provided by a beta-plane Coriolis module, e.g. "
         "fr.modules.BetaPlaneCoriolis(beta=...)")

CORIOLIS_ROSSBY: Final[ParamName] = ParamName(
    "coriolis.rossby",
    units="n/a",
    hint="provided by a nondimensional Coriolis module, e.g. "
         "fr.modules.FPlaneCoriolis(rossby_number=...)")

CORIOLIS_METRIC_RATIO: Final[ParamName] = ParamName(
    "coriolis.metric_ratio",
    units="n/a",
    hint="provided by a nondimensional beta-plane Coriolis module, "
         "e.g. fr.modules.BetaPlaneCoriolis(rossby_number=..., "
         "metric_ratio=...)")

# no_default: a reference default would silently un-stratify a run.
STRATIFICATION_N2: Final[ParamName] = ParamName(
    "stratification.n2",
    units="1/s^2",
    hint="provided by a stratification module, e.g. "
         "ConstantStratification(n2=...)",
    no_default=True)

# The internal-wave Froude number Fr = U/(N H) of a NONDIMENSIONAL
# stratification module (the ``internal_wave`` scaling mechanism);
# no_default for the same reason as ``stratification.n2``.
STRATIFICATION_FROUDE: Final[ParamName] = ParamName(
    "stratification.froude",
    units="1",
    hint="provided by a nondimensional stratification module, e.g. "
         "ConstantStratification(froude_number=...)",
    no_default=True)

# The nonlinearity number epsilon = T_ref / T_adv: under a
# nondimensional scaling the assembly aliases this name onto the
# scaling mechanism's own nonlinearity leaf (fr.scaling); dimensional
# assemblies bind no row (row presence <=> nondimensional).
SCALING_NONLINEARITY: Final[ParamName] = ParamName(
    "scaling.nonlinearity",
    units="n/a",
    hint="bound by the assembly under a nondimensional fr.scaling "
         "policy (aliasing the mechanism module's nonlinearity "
         "leaf, e.g. FPlaneCoriolis(rossby_number=...)); "
         "dimensional models bind no row")
