"""
Implicit-operator families (``fr.implicit``).

Description
-----------
``ImplicitOperator`` (the two-capability structural protocol) and
``VerticalDiffusion`` (the framework-owned mergeable tridiagonal
family). Owning class spec:
``notes/framework2/model/classes/declarations.md``; design source
``notes/framework2/model/03_time_stepping.md`` section 5.1.
``SpectralDiagonal`` is designed-for and deliberately not built
(earliest consumer: the sw semi-implicit gravity-wave pair, 2.7).

Implementations are static assembly data riding on terms; live
coefficients are read through UNBOUND callables at trace time (the
D2 aliasing rule applied to coefficients).
"""
# Wave 2 C: ImplicitOperator, VerticalDiffusion
#    (SpectralDiagonal is designed-for)
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Hashable

    from fridom.framework2.grid.fields.scalar_field import ScalarField


# ================================================================
#  The ImplicitOperator protocol
# ================================================================
@runtime_checkable
class ImplicitOperator(Protocol):

    """
    The two-capability implicit-term surface (structural protocol).

    Description
    -----------
    ``apply`` is the forward ``L @ state`` and ``solve`` is
    ``(1 - dt_gamma * L)^{-1}`` — two capabilities, nothing more (the
    minimal surface every IMEX family needs). The forward apply is
    mandatory: the CN solve-only trick is unsound here (recovering
    ``L @ X`` from the previous solve reads the pre-projection state,
    an O(dt) error every step). Coupled blocks are atomic:
    ``fields=("u", "v")`` (semi-implicit Coriolis) is one operator,
    indivisible under by-variable splitting.

    The merge hooks ``merge_key``/``merged_with`` implement the
    signed family-merge rules (mergeable framework families combine
    *exactly*; at most one non-mergeable custom operator per field).
    The hooks themselves are spec-proposed mechanics, confirm at
    first composer use (declarations.md, open question 8).

    Attributes
    ----------
    fields : tuple[str, ...]
        The advanced PROGNOSTIC subset — MANDATORY.
    """

    fields: tuple[str, ...]

    def apply(
        self, module: Any, state: Any, ctx: Any,
    ) -> dict[str, ScalarField]:
        """
        Evaluate ``L @ state`` (the forward evaluation).

        Description
        -----------
        The CNAB right-hand side; also the derived explicit path when
        the owning term omits ``fn`` (write-once).

        Parameters
        ----------
        module : Any
            The live owning module from the carry (unbound-slot
            convention).
        state : Any
            The full assembled state vector.
        ctx : Any
            The per-substage ``StepContext``.

        Returns
        -------
        dict[str, ScalarField]
            Increments keyed by the advanced PROGNOSTIC names.
        """
        ...

    def solve(
        self, module: Any, rhs: dict[str, ScalarField],
        dt_gamma: Any, ctx: Any,
    ) -> dict[str, ScalarField]:
        """
        Solve ``(1 - dt_gamma * L) x = rhs``; keys exactly `fields`.

        Description
        -----------
        Gamma-agnostic: the scheme owns gamma, so ``dt_gamma`` (the
        product gamma*dt) is a TRACED scalar positional supplied by
        the stepper (CN: dt/2, SBDF2: 2dt/3) — warm-up gamma
        switching and adaptive dt never retrace. It is never read
        from ``ctx`` (``ctx.stage_dt`` is not ``dt_gamma``).

        Parameters
        ----------
        module : Any
            The live owning module from the carry.
        rhs : dict[str, ScalarField]
            Right-hand sides keyed by the advanced names.
        dt_gamma : Any
            The traced scalar gamma*dt of the current scheme stage.
        ctx : Any
            The per-substage ``StepContext``.

        Returns
        -------
        dict[str, ScalarField]
            The solved fields, keyed exactly by `fields`.
        """
        ...

    def merge_key(self) -> Hashable | None:
        """
        Return the grouping key for exact merging.

        Returns
        -------
        Hashable | None
            Operators with equal keys merge exactly via
            ``merged_with``; ``None`` marks a non-mergeable operator
            (at most one per field, ``ImplicitCollisionError``).
        """
        ...

    def merged_with(self, other: ImplicitOperator) -> ImplicitOperator:
        """
        Combine exactly with `other` (same ``merge_key`` group).

        Parameters
        ----------
        other : ImplicitOperator
            The operator to merge with; must share this operator's
            ``merge_key``.

        Returns
        -------
        ImplicitOperator
            The exactly-combined operator.
        """
        ...


# ================================================================
#  VerticalDiffusion — the mergeable tridiagonal family
# ================================================================
def _summed_kappa(
    parts: tuple[VerticalDiffusion, ...],
) -> Callable:
    """
    Build the merged coefficient callable of a kappa-summed family.

    Description
    -----------
    The merged coefficient for a field is the sum of the constituent
    coefficients of every part covering that field — the exact
    combination ``(1 - dt_gamma * (L1 + L2))``; sequential opaque
    solves would be Lie splitting inside an IMEX stage
    (order-degrading). The returned callable keeps the family's
    UNBOUND signature ``(module, state, ctx, field_name)``; the
    composer supplies a uniform first argument to every constituent
    (slot pairing is composer territory, wave 3+).

    Parameters
    ----------
    parts : tuple[VerticalDiffusion, ...]
        The constituent operators, in merge order.

    Returns
    -------
    Callable
        The summed coefficient callable.
    """
    def kappa(
        module: Any, state: Any, ctx: Any, field_name: str,
    ) -> Any:
        total = None
        for part in parts:
            if field_name not in part.fields:
                continue
            value = part.kappa(module, state, ctx, field_name)
            total = value if total is None else total + value
        if total is None:
            raise ValueError(
                f"field {field_name!r} is not covered by this merged "
                "VerticalDiffusion operator")
        return total

    return kappa


@dataclass(frozen=True)
class VerticalDiffusion:

    """
    Diffusion along one axis, solved as ONE tridiagonal per field.

    Description
    -----------
    The framework-owned mergeable tridiagonal family: same-axis
    operators touching a field merge by summing kappa into one
    tridiagonal solve per field (Oceananigans' coefficient-merging
    precedent). Static assembly data; kappa is read live through the
    UNBOUND callable at trace time — coefficients read
    ``ctx.params`` / owner leaves at stage time (Ramp-correct).
    State-dependent coefficients are evaluated on the state passed
    into the stage (predictor/lagged values); the solve itself stays
    linear.

    The numerical kernels (``apply``/``solve``) are pending: they
    land with the 2.5 IMEX reference consumer (exact 1D decay and
    stiff-kappa column tests) once the dispatched second-derivative
    operator and the field's declared-space boundary rows are wired
    at assembly. Construction, coefficient math, and the merge logic
    are complete.

    Parameters
    ----------
    axis : str
        The coordinate name of the solve axis.
    fields : tuple[str, ...]
        The advanced PROGNOSTIC subset (one tridiagonal per field).
    kappa : Callable
        UNBOUND ``(module, state, ctx, field_name) ->
        scalar | ScalarField`` — coefficients, not a solver.
        Bound-ness is validated at assembly (the aliasing rule); the
        vocabulary records.

    Raises
    ------
    TypeError
        If `axis` is not a non-empty string or `kappa` is not
        callable.
    ValueError
        If `fields` is empty.
    """

    axis: str
    fields: tuple[str, ...]
    kappa: Callable

    def __post_init__(self) -> None:
        """Normalize `fields` and check local record validity."""
        if not isinstance(self.axis, str) or not self.axis:
            raise TypeError(
                f"axis must be a non-empty coordinate name, got "
                f"{self.axis!r}")
        object.__setattr__(self, "fields", tuple(self.fields))
        if not self.fields:
            raise ValueError(
                "VerticalDiffusion needs at least one field")
        if not callable(self.kappa):
            raise TypeError(
                f"kappa must be callable, got {self.kappa!r}")

    def apply(
        self, module: Any, state: Any, ctx: Any,
    ) -> dict[str, ScalarField]:
        """
        Evaluate ``L @ state`` (kernel pending, see Raises).

        Description
        -----------
        The forward evaluation via the dispatched second-derivative
        operator — a grid-bound registry Operator, negotiated like
        transforms and intercepted generically by the halo tracer
        (no ``.data`` bypasses, no ``extra_halo`` needed).

        Parameters
        ----------
        module : Any
            The live owning module from the carry.
        state : Any
            The full assembled state vector.
        ctx : Any
            The per-substage ``StepContext``.

        Returns
        -------
        dict[str, ScalarField]
            ``{field: L @ state[field]}`` over `fields`.

        Raises
        ------
        NotImplementedError
            Always, in this wave: the numerical kernel requires the
            assembly-wired dispatch registry and lands with the 2.5
            IMEX reference consumer (exact 1D decay + stiff-kappa
            column tests).
        """
        raise NotImplementedError(
            "VerticalDiffusion.apply: the numerical kernel lands "
            "with the 2.5 IMEX reference consumer (it needs the "
            "assembly-wired second-derivative operator)")

    def solve(
        self, module: Any, rhs: dict[str, ScalarField],
        dt_gamma: Any, ctx: Any,
    ) -> dict[str, ScalarField]:
        """
        Solve one tridiagonal per field (kernel pending, see Raises).

        Description
        -----------
        One Thomas solve per field with boundary rows from the
        field's declared space BCs; flux BCs are explicit forcing.
        No assembly-time factorization caching in iteration 1
        (``dt_gamma`` and kappa are traced).

        Parameters
        ----------
        module : Any
            The live owning module from the carry.
        rhs : dict[str, ScalarField]
            Right-hand sides keyed by the advanced names.
        dt_gamma : Any
            The traced scalar gamma*dt (stepper-supplied positional;
            never read from ``ctx``).
        ctx : Any
            The per-substage ``StepContext``.

        Returns
        -------
        dict[str, ScalarField]
            The solved fields, keyed exactly by `fields`.

        Raises
        ------
        NotImplementedError
            Always, in this wave: the numerical kernel requires the
            declared-space boundary rows and lands with the 2.5 IMEX
            reference consumer (exact 1D decay + stiff-kappa column
            tests).
        """
        raise NotImplementedError(
            "VerticalDiffusion.solve: the numerical kernel lands "
            "with the 2.5 IMEX reference consumer (it needs the "
            "declared-space boundary rows)")

    def merge_key(self) -> Hashable:
        """
        Return the family grouping key: same axis, same family.

        Returns
        -------
        Hashable
            ``(type, axis)`` — same-axis instances of one family are
            mergeable (per shared field).
        """
        return (type(self), self.axis)

    def merged_with(
        self, other: VerticalDiffusion,
    ) -> VerticalDiffusion:
        """
        Sum kappa contributions into one solve (exact).

        Description
        -----------
        The merged operator covers the ordered union of both field
        tuples; per field, the merged coefficient is the sum of the
        covering constituents' coefficients — the exact combined
        operator ``(1 - dt_gamma * (L1 + L2))``.

        Parameters
        ----------
        other : VerticalDiffusion
            A same-family, same-axis operator (equal ``merge_key``).

        Returns
        -------
        VerticalDiffusion
            The kappa-summed single operator.

        Raises
        ------
        ValueError
            If `other` has a different ``merge_key`` (different
            family or axis).
        """
        if self.merge_key() != other.merge_key():
            raise ValueError(
                f"cannot merge implicit operators with different "
                f"merge keys: {self.merge_key()!r} vs "
                f"{other.merge_key()!r}")
        merged_fields = self.fields + tuple(
            field for field in other.fields
            if field not in self.fields)
        return VerticalDiffusion(
            axis=self.axis,
            fields=merged_fields,
            kappa=_summed_kappa((self, other)),
        )
