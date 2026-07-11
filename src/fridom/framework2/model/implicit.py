"""
Implicit-operator families (``fr.implicit``).

Description
-----------
``ImplicitOperator`` (the two-capability structural protocol) and
``VerticalDiffusion`` (the framework-owned mergeable tridiagonal
family). Owning class spec:
``design/specs/model/classes/declarations.md``; design source
``design/specs/model/03_time_stepping.md`` section 5.1.
``SpectralDiagonal`` is designed-for and deliberately not built
(earliest consumer: the sw semi-implicit gravity-wave pair, 2.7).

Implementations are static assembly data riding on terms; live
coefficients are read through UNBOUND callables at trace time (the
D2 aliasing rule applied to coefficients).
"""
# Wave 2 C: ImplicitOperator, VerticalDiffusion
#    (SpectralDiagonal is designed-for)
# Wave 5 B: the VerticalDiffusion apply/solve tridiagonal kernel
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.framework2.grid.operators.banded import (
    apply_along_axis,
    second_difference_matrix,
    solve_along_axis,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Hashable

    import jax

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
        Evaluate the forward ``L @ state`` per field (``L = k d2/dz2``).

        Description
        -----------
        The CNAB right-hand-side term. Per field the second-derivative
        stencil along ``axis`` is applied to the true-shape column
        (``ScalarField.data`` — halo/padding stripped) with Neumann
        (zero-flux) boundary rows, then scaled by the live kappa read
        from the unbound callable at stage time (Ramp-correct). The
        result re-enters via ``with_data`` (re-pad + halo-invalidate;
        synced at its next ghost-consuming application), so no manual
        halo/``extra_halo`` bookkeeping is needed. Iteration-1 scope:
        the solve axis must not be distributed across devices (a
        tridiagonal is serial along it) and kappa is constant in the
        column (a scalar); a face-averaged variable-kappa conservative
        form is the follow-up.

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
        """
        result: dict[str, ScalarField] = {}
        for name in self.fields:
            field = state[name]
            operator, axis_index = _diffusion_operator(
                field, self.axis,
                self.kappa(module, state, ctx, name))
            data = jnp.asarray(field.data)
            applied = apply_along_axis(operator, data, axis_index)
            result[name] = field.with_data(applied)
        return result

    def solve(
        self, module: Any, rhs: dict[str, ScalarField],
        dt_gamma: Any, ctx: Any,
    ) -> dict[str, ScalarField]:
        """
        Solve ``(1 - dt_gamma * L) x = rhs``, one tridiagonal / field.

        Description
        -----------
        The linear implicit solve per field along ``axis`` with the
        Neumann (zero-flux) boundary rows of the declared column; flux
        BCs would be explicit forcing (not built). ``dt_gamma`` is the
        stepper-supplied traced positional (CN ``dt/2``, SBDF2
        ``2dt/3``) — never read from ``ctx``, so warm-up gamma
        switching and adaptive dt never retrace; kappa is read live.
        Iteration-1 solves the dense ``(1 - dt_gamma * L)`` system
        (``jnp.linalg.solve`` batched over the off-axis columns) — a
        Thomas/``tridiagonal_solve`` kernel is the production
        optimization; both respect the true-shape ``data`` /
        ``with_data`` halo contract (the solve axis stays
        device-local).

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
        """
        result: dict[str, ScalarField] = {}
        real = dtype_real()
        for name in self.fields:
            field = rhs[name]
            operator, axis_index = _diffusion_operator(
                field, self.axis,
                self.kappa(module, field, ctx, name))
            size = operator.shape[0]
            system = (jnp.eye(size, dtype=real)
                      - jnp.asarray(dt_gamma, dtype=real) * operator)
            data = jnp.asarray(field.data)
            solved = solve_along_axis(system, data, axis_index)
            result[name] = field.with_data(solved)
        return result

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


# ================================================================
#  The tridiagonal kernel (true-shape ``data``; single-device axis)
# ================================================================
def _diffusion_operator(
    field: ScalarField, axis: str, kappa_value: Any,
) -> tuple[jax.Array, int]:
    """
    Build ``L = kappa * d2/dz2`` along ``axis`` (Neumann rows).

    Description
    -----------
    The dense ``(N, N)`` Neumann second-difference band — assembled by
    the shared ``grid.operators.banded`` primitive on the field's
    true-shape column evaluation nodes — scaled by the constant column
    ``kappa``. Returns the matrix and the storage-frame axis index of
    ``axis``.

    Parameters
    ----------
    field : ScalarField
        The column field (true shape via ``.data``).
    axis : str
        The solve coordinate.
    kappa_value : Any
        The live scalar coefficient (a field-valued kappa is the
        variable-coefficient follow-up — not built in iteration 1).

    Returns
    -------
    tuple[jax.Array, int]
        The ``(N, N)`` operator matrix and the axis index.

    Raises
    ------
    ValueError
        If ``axis`` is not a coordinate of the field's space.
    NotImplementedError
        If ``kappa_value`` is field-valued (variable coefficient).
    """
    space = field.function_space
    names = space.bare.names
    if axis not in names:
        raise ValueError(
            f"VerticalDiffusion axis {axis!r} is not a coordinate of "
            f"the field space {names}")
    axis_index = names.index(axis)
    if hasattr(kappa_value, "function_space"):
        raise NotImplementedError(
            "VerticalDiffusion supports a constant (scalar) column "
            "kappa in iteration 1; a face-averaged variable-kappa "
            "conservative form is the follow-up")
    kappa = jnp.asarray(kappa_value, dtype=dtype_real())
    coords = field.grid.evaluation_nodes(space, axis)
    d2 = second_difference_matrix(coords.data)
    return kappa * d2, axis_index
