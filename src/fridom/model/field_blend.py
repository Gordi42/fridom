r"""
Declaration-layer field blends: :math:`p(t) = \sum_i w_i(t)\,P_i`.

Description
-----------
The generic two-endpoint (and, more generally, affine) blend mechanism
of the adiabatic-ramping plan (AR-D2; spec ``08_state_transforms``
§10.9). A field-valued parameter may be declared as an affine
combination of **assembly-materialized ingredient profiles** ``P_i``
with **stage-time scalar weights** ``w_i(t)``,

.. math::
    p(t) = \sum_i w_i(t)\,P_i ,

the two-endpoint case being ingredients ``{p_ref, p_target - p_ref}``
with weights ``{1, lambda(t)}``. The ingredients are static AUXILIARY
fields — their treedefs stay scan-stable and their halos are exchanged
once at assembly, so the pointwise blend adds no halo traffic — and the
weights are ordinary module scalar leaves read at stage time through
``resolve_at`` (any ``fr.Ramp`` drives them, and endpoint/timing sweeps
never recompile).

**Module-author machinery only.** There is deliberately no user-facing
``FieldBlend`` value type in a module constructor: a module *holds* a
``FieldBlend`` (a module-level descriptor), contributes its ingredient
declarations from ``field_declarations`` when a weight is time-dependent,
and evaluates the blend inside its own clock-aware tendency term. The
first consumer is the Coriolis family (``f(t) = f0(t) + beta(t)*y``);
stratification/topography blends are follow-ups (plan §7). We still do
**not** build general time-dependent fields (no SELF_UPDATE rewrite, no
``(coords, t)`` recompute contract) — those stay with the open roadmap
entry, of which this affine blend is the forward-compatible subset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax

from fridom.model.declarations import FieldDeclaration, Lifecycle
from fridom.model.time_dependent import TimeDependent, resolve_at

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.space_patterns import SpacePattern, SpaceRule


def _scalar_weight(weight: object) -> object:
    """
    Demote a concrete 0-d weight to a Python scalar; keep tracers.

    Description
    -----------
    The halo tracer (``trace_halo``) recognizes a *Python* scalar
    operand (``int``/``float``/``complex``) and passes it through the
    stencil unchanged, but a raw 0-d ``jax`` array on the left of
    ``array * tracer`` neither converts nor defers, so the multiply
    raises. During the halo trace the weight is concrete (the module
    leaves are host values, only the state is traced), so it demotes
    cleanly — mirroring the assembly's own ``_as_scalar`` demotion of
    ``ctx.params``. In the jitted step the weight is a tracer, so
    ``float`` raises and the tracer passes through (``ScalarField``
    handles ``tracer * field`` via its reflected operators).
    """
    try:
        return float(weight)
    except (TypeError, ValueError,
            jax.errors.ConcretizationTypeError,
            jax.errors.TracerArrayConversionError):
        try:
            return complex(weight)
        except (TypeError, ValueError,
                jax.errors.ConcretizationTypeError,
                jax.errors.TracerArrayConversionError):
            return weight


@dataclass(frozen=True)
class BlendIngredient:

    r"""
    One ingredient of a :class:`FieldBlend`: a profile ``P_i`` and weight.

    Description
    -----------
    Frozen host data (never a pytree). ``build`` is an **unbound owner
    method** ``(self, grid, space) -> ScalarField`` materializing the
    profile ``P_i`` — the D1.1 owner-method ``default=`` form, so the
    profile is re-materialized on the module's own leaves at assembly
    (and reset). ``weight`` selects ``w_i``: a **module attribute name**
    (a scalar leaf read at stage time via ``resolve_at`` — a plain float
    or an ``fr.Ramp``), or a plain **constant** (the ``1`` weight of a
    two-endpoint ``p_ref`` ingredient).

    Parameters
    ----------
    field : str
        The AUXILIARY field name carrying the materialized profile
        ``P_i`` in the model state.
    weight : str | float
        The module attribute holding the weight leaf ``w_i`` (read at
        stage time), or a constant literal.
    build : Callable
        The unbound owner method ``(self, grid, space) -> ScalarField``
        materializing ``P_i``.
    """

    field: str
    weight: str | float
    build: Callable


class FieldBlend:

    r"""
    An affine blend :math:`p(t) = \sum_i w_i(t)\,P_i` (AR-D2).

    Description
    -----------
    A space-agnostic descriptor a module composes (see the module
    docstring): it declares the ingredient AUXILIARY fields and
    evaluates the stage-time blend. The blend is **active** for a given
    module iff at least one leaf weight resolves to a ``TimeDependent``
    value on it; when no weight is time-dependent the module keeps its
    static single-field fast path (kept bit-identical) and never
    declares the ingredients.

    Parameters
    ----------
    ingredients : Iterable[BlendIngredient]
        The blend ingredients, evaluation order.
    """

    __slots__ = ("_ingredients",)

    def __init__(self, ingredients: object) -> None:
        """Freeze the ingredient tuple."""
        self._ingredients: tuple[BlendIngredient, ...] = tuple(ingredients)

    @property
    def ingredients(self) -> tuple[BlendIngredient, ...]:
        """The blend ingredients (evaluation order)."""
        return self._ingredients

    def is_active(self, module: object) -> bool:
        """
        Whether any leaf weight is time-dependent on ``module``.

        Description
        -----------
        A host-side (static) predicate: a leaf weight's *type*
        (``fr.Ramp`` vs a plain array) is structural, so the module
        may branch its declarations and its term on it without touching
        a traced value. Constant weights and plain-float leaves make no
        ingredient blend necessary.

        Parameters
        ----------
        module : object
            The owning module (its scalar-leaf attributes hold the
            weights).

        Returns
        -------
        bool
            True iff at least one leaf weight is a ``TimeDependent``.
        """
        return any(
            isinstance(ingredient.weight, str)
            and isinstance(getattr(module, ingredient.weight),
                           TimeDependent)
            for ingredient in self._ingredients)

    def field_declarations(
        self,
        *,
        space: SpacePattern | SpaceRule,
        long_name: str = "Unnamed",
        units: str = "n/a",
    ) -> tuple[FieldDeclaration, ...]:
        """
        Return the AUXILIARY ingredient declarations (owner-method).

        Description
        -----------
        One static AUXILIARY declaration per ingredient, materialized by
        its owner-method ``build`` — the halos are exchanged once at
        assembly, and the pointwise blend adds no halo traffic.

        Parameters
        ----------
        space : SpacePattern | SpaceRule
            The shared space of every ingredient profile (the blended
            field's own space).
        long_name : str, optional
            Descriptive nc-style name for the ingredient fields
            (default: "Unnamed").
        units : str, optional
            Physical units annotation (default: "n/a").

        Returns
        -------
        tuple[FieldDeclaration, ...]
            The ingredient declarations, evaluation order.
        """
        return tuple(
            FieldDeclaration(
                ingredient.field, space=space,
                lifecycle=Lifecycle.AUXILIARY, default=ingredient.build,
                long_name=long_name, units=units)
            for ingredient in self._ingredients)

    def evaluate(
        self, module: object, state: object, time: object,
    ) -> ScalarField:
        r"""
        Return the stage-time blend :math:`\sum_i w_i(t)\,P_i`.

        Description
        -----------
        Reads each materialized ingredient ``P_i`` from ``state`` and its
        weight ``w_i`` from ``module`` (a constant, or a leaf resolved at
        ``time`` through ``resolve_at`` — the same seam every ramped
        scalar rides), and accumulates the affine combination as pure
        field arithmetic (a fused multiply-add per ingredient). The scale
        and shift preserve the ingredients' ghost validity, so the result
        is consumed (``.to(...)``) exactly like a materialized profile.

        Parameters
        ----------
        module : object
            The owning module (source of the weight leaves).
        state : Mapping[str, ScalarField]
            The model state (source of the ingredient profiles).
        time : jax.Array | float
            The stage clock time the weights are resolved at.

        Returns
        -------
        ScalarField
            The blended field on the ingredients' shared space.
        """
        total: ScalarField | None = None
        for ingredient in self._ingredients:
            weight = ingredient.weight
            if isinstance(weight, str):
                weight = _scalar_weight(
                    resolve_at(getattr(module, weight), time))
            contribution = weight * state[ingredient.field]
            total = (contribution if total is None
                     else total + contribution)
        return total
