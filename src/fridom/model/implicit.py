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
from fridom.spatial.operators.banded import (
    tridiagonal_apply_along_axis,
    tridiagonal_solve_along_axis,
    validate_boundary_conditions,
)
from fridom.spatial.spaces.nodal import NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Hashable

    import jax

    from fridom.spatial.fields.scalar_field import ScalarField


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

    The column is **measure-aware** (:func:`_diffusion_bands`): the
    face-averaged conservative flux band reads its cell and face widths
    from ``grid.measure`` — true non-uniform spacing on a stretched
    mesh — and multiplies each width by the column Jacobian
    ``grid.metric`` on a terrain-following grid (the along-``sigma``
    convention, owner-ratified 2026-07-19). ``kappa`` may be a scalar
    or a ``ScalarField`` on the solved field's own space (face-averaged,
    one-sided at the walls); band assembly is linear in ``kappa``, so
    the kappa-summed merge stays exact with field coefficients. On a
    uniform column the band entries are identical to the historical
    ``kappa/dz^2`` second difference (same ``-1`` Neumann / ``-3``
    Dirichlet corners).

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
    bc : tuple[str, str], optional
        The per-side ``(low, high)`` boundary-condition rows of the
        column band, each ``"neumann"`` (zero-flux / free-slip) or
        ``"dirichlet"`` (no-slip). The pair enters :meth:`merge_key`,
        so unlike-BC operators on the same axis never kappa-merge
        (default: ``("neumann", "neumann")``).

    Raises
    ------
    TypeError
        If `axis` is not a non-empty string or `kappa` is not
        callable.
    ValueError
        If `fields` is empty or `bc` is not an accepted per-side pair.
    """

    axis: str
    fields: tuple[str, ...]
    kappa: Callable
    bc: tuple[str, str] = ("neumann", "neumann")

    def __post_init__(self) -> None:
        """Normalize `fields`/`bc` and check local record validity."""
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
        object.__setattr__(
            self, "bc", validate_boundary_conditions(self.bc))

    def apply(
        self, module: Any, state: Any, ctx: Any,
    ) -> dict[str, ScalarField]:
        r"""
        Evaluate the forward ``L @ state`` per field (conservative flux).

        Description
        -----------
        The CNAB right-hand-side term. Per field the measure-aware
        conservative flux band (:func:`_diffusion_bands`) is applied to
        the true-shape column (``ScalarField.data`` — halo/padding
        stripped) along ``axis`` with the per-side boundary rows `bc`,
        the live kappa read from the unbound callable at stage time
        (Ramp-correct). The band stencil is
        ``lower q_{c-1} + diag q_c + upper q_{c+1}`` — the exact
        divergence of the face fluxes
        ``kappa_{c+/-1/2} (q_{c+/-1} - q_c) / dz_{c+/-1/2}`` over the
        cell width ``dz_c``, so the width-weighted column sum telescopes
        to the wall fluxes (zero for Neumann). The result re-enters via
        ``with_data`` (re-pad + halo-invalidate; synced at its next
        ghost-consuming application), so no manual halo/``extra_halo``
        bookkeeping is needed. The solve axis must stay device-local (a
        tridiagonal is serial along it).

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
            lower, diag, upper, axis_index = _diffusion_bands(
                field, self.axis,
                self.kappa(module, state, ctx, name), self.bc)
            data = jnp.asarray(field.data)
            applied = tridiagonal_apply_along_axis(
                lower, diag, upper, data, axis_index)
            result[name] = field.with_data(applied)
        return result

    def solve(
        self, module: Any, rhs: dict[str, ScalarField],
        dt_gamma: Any, ctx: Any,
    ) -> dict[str, ScalarField]:
        r"""
        Solve ``(1 - dt_gamma * L) x = rhs``, one tridiagonal / field.

        Description
        -----------
        The linear implicit solve per field along ``axis`` with the
        per-side boundary rows `bc` of the declared column; flux BCs
        would be explicit forcing (not built). ``dt_gamma`` is the
        stepper-supplied traced positional (CN ``dt/2``, SBDF2
        ``2dt/3``) — never read from ``ctx``, so warm-up gamma switching
        and adaptive dt never retrace; kappa is read live. The system
        bands ``(-dt_gamma * lower, 1 - dt_gamma * diag,
        -dt_gamma * upper)`` of the measure-aware column
        (:func:`_diffusion_bands`) feed
        :func:`~fridom.spatial.operators.banded.tridiagonal_solve_along_axis`
        with the reference ``method="scan"`` (Thomas), chosen over the
        cyclic-reduction kernels (``"pcr"`` / cuSPARSE) deliberately: the
        CN system is only WEAKLY diagonally dominant in the stiff regime
        (its dominance excess is exactly ``1``, so ``diag/|off| -> 1`` as
        ``kappa*dt/dz^2 -> inf``), and cyclic reduction amplifies
        roundoff on a weakly-DD system enough to excite the near-``(-1)``
        highest CN mode over a long run. Thomas back-substitution is
        unconditionally stable for any DD system, needs no pivoting
        (dominance by the ``+1``, ``kappa >= 0``), and is natively
        reverse-mode differentiable (no ``custom_vjp``). The true-shape
        ``data`` / ``with_data`` halo contract holds (the solve axis
        stays device-local).

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
        gamma = jnp.asarray(dt_gamma, dtype=dtype_real())
        for name in self.fields:
            field = rhs[name]
            lower, diag, upper, axis_index = _diffusion_bands(
                field, self.axis,
                self.kappa(module, field, ctx, name), self.bc)
            data = jnp.asarray(field.data)
            # method="scan" (reference Thomas), NOT "auto": the CN system
            # (1 - dt_gamma L) is only WEAKLY diagonally dominant in the
            # stiff regime (dominance excess is exactly 1, so the ratio
            # diag/|off| -> 1 as kappa*dt/dz^2 -> inf). The cyclic-
            # reduction kernels (pcr, and cuSPARSE gtsv2 internally)
            # amplify roundoff on a weakly-DD system, exciting the
            # near-(-1) highest CN mode over a long run (kappa*dt/dz^2 ~
            # 640 blows up by ~step 20). Thomas back-substitution is
            # unconditionally stable for any DD system, exact, and
            # natively reverse-mode differentiable.
            solved = tridiagonal_solve_along_axis(
                -gamma * lower, 1.0 - gamma * diag, -gamma * upper,
                data, axis_index, method="scan")
            result[name] = field.with_data(solved)
        return result

    def merge_key(self) -> Hashable:
        """
        Return the family grouping key: same axis, same BC, same family.

        Description
        -----------
        ``(type, axis, bc)`` — the per-side boundary rows enter the key
        so that a no-slip (Dirichlet-row) leg and a free-slip / no-flux
        (Neumann-row) leg on the same axis are **not** merged:
        kappa-summing them would silently combine two different
        operators (a Dirichlet ``-3`` corner with a Neumann ``-1``
        corner). Same-axis, same-BC instances of one family remain
        mergeable (per shared field).

        Returns
        -------
        Hashable
            ``(type, axis, bc)``.
        """
        return (type(self), self.axis, self.bc)

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
            bc=self.bc,
        )


# ================================================================
#  The measure-aware column band (true-shape data; single-device axis)
# ================================================================
def _axis_slice(
    arr: jax.Array, axis_index: int, start: int, stop: int,
) -> jax.Array:
    """Return ``arr[..., start:stop, ...]`` along ``axis_index``."""
    index: list[slice | int] = [slice(None)] * arr.ndim
    index[axis_index] = slice(start, stop)
    return arr[tuple(index)]


def _zero_axis_end(
    arr: jax.Array, axis_index: int, *, at_start: bool,
) -> jax.Array:
    """Zero the first (``at_start``) or last slice along ``axis_index``."""
    index: list[slice | int] = [slice(None)] * arr.ndim
    index[axis_index] = 0 if at_start else arr.shape[axis_index] - 1
    return arr.at[tuple(index)].set(0.0)


def _face_space(space: Any, axis: str) -> Any:
    """
    Return the wall-including (``Outer``) face sibling of ``space``.

    Description
    -----------
    The ``Outer`` nodal space on the solve-axis mesh: its interior
    entries are the node-to-node dual widths (the ``diff`` denominators
    ``dz_{c+/-1/2}``) and its two boundary entries are the clipped
    node-to-wall half-cells (the physical Dirichlet wall distances).
    One measure query on it therefore serves both the interior
    couplings and the wall rows. The other factors (and the layout) are
    preserved, so the query broadcasts against the field's ``data`` at
    the same storage axis; a single-factor column *is* the face factor.
    """
    outer = space.factor(axis).mesh.nodal(NodeSet.OUTER)
    if len(space.factors) == 1:
        return outer
    return space.replace(**{axis: outer})


def _face_kappa(
    space: Any, axis_index: int, size: int, kappa_value: Any,
    real: Any,
) -> tuple[jax.Array, jax.Array]:
    r"""
    Resolve the ``(kappa_up, kappa_low)`` per-cell face coefficients.

    Description
    -----------
    A scalar ``kappa`` is constant on every face. A ``ScalarField``
    ``kappa`` on the solved field's own space (cell-centered along the
    solve axis) is **arithmetically face-averaged** between adjacent
    cells; the two wall faces take the one-sided corner-cell value (the
    only defined choice — there is no cell beyond the wall). ``kappa_up``
    is the coefficient at the upper face ``c + 1/2`` of each cell and
    ``kappa_low`` at the lower face ``c - 1/2``; both broadcast against
    the field ``data``.

    Raises
    ------
    ValueError
        If a field-valued ``kappa`` is not on the solved field's own
        function space.
    """
    if not hasattr(kappa_value, "function_space"):
        kappa = jnp.asarray(kappa_value, dtype=real)
        return kappa, kappa
    if kappa_value.function_space != space:
        raise ValueError(
            "a field-valued VerticalDiffusion kappa must live on the "
            "solved field's own function space (cell-centered along the "
            f"solve axis) {space!r}; got a field on "
            f"{kappa_value.function_space!r}")
    cells = jnp.asarray(kappa_value.data, dtype=real)
    below = _axis_slice(cells, axis_index, 0, size - 1)
    above = _axis_slice(cells, axis_index, 1, size)
    interior = 0.5 * (below + above)
    bottom = _axis_slice(cells, axis_index, 0, 1)
    top = _axis_slice(cells, axis_index, size - 1, size)
    faces = jnp.concatenate([bottom, interior, top], axis=axis_index)
    kappa_up = _axis_slice(faces, axis_index, 1, size + 1)
    kappa_low = _axis_slice(faces, axis_index, 0, size)
    return kappa_up, kappa_low


def _diffusion_bands(
    field: ScalarField, axis: str, kappa_value: Any,
    bc: tuple[str, str] = ("neumann", "neumann"),
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    r"""
    Build the measure-aware column bands ``(lower, diag, upper, axis)``.

    Description
    -----------
    The face-averaged conservative flux band of
    ``L q = d_z(kappa d_z q)`` along ``axis``, on the field's true-shape
    column (``ScalarField.data`` — halo/padding stripped, so there is no
    padded-zero division anywhere here). Per cell ``c``

    .. math::

        (L q)_c = \frac{
            \kappa_{c+1/2}\,(q_{c+1} - q_c) / \Delta z_{c+1/2}
          - \kappa_{c-1/2}\,(q_c - q_{c-1}) / \Delta z_{c-1/2}}{
            \Delta z_c},

    so ``upper[c] = kappa_{c+1/2} / (dz_{c+1/2} dz_c)``,
    ``lower[c] = kappa_{c-1/2} / (dz_{c-1/2} dz_c)`` and ``diag`` is the
    **negated sum** of the couplings (assembled that way so a Neumann
    row sums to zero exactly and the width-weighted column sum
    telescopes).

    Widths come from ``grid.measure``: the cell width ``dz_c`` is the
    primal measure of the field's own node set (``Center`` or FV
    ``CellAvg``, no hardcoded stagger); ``dz_{c+/-1/2}`` are the dual
    widths of the wall-including ``Outer`` face family — its interior
    entries the node-to-node spacings and its two boundary entries the
    clipped wall half-cells (:func:`_face_space`). On a terrain-
    following grid (``axis in grid.mapping.column_corrections``) every
    width is multiplied by the column Jacobian ``grid.metric`` at the
    same stagger (STATIC params) — the along-``sigma`` physical widths,
    per-column, which is exactly why the caller uses the per-column
    tridiagonal kernels.

    Boundary rows generalize the historical ``-1`` / ``-3`` corners: a
    Neumann side drops the wall coupling (``diag`` loses it, the wall-
    side band entry is 0); a Dirichlet (no-slip, wall value 0) side adds
    the one-sided wall flux ``kappa_wall / (d_wall dz_corner)`` to the
    corner ``diag`` (``d_wall`` the clipped wall half-cell, ``kappa_wall``
    the corner cell's coefficient). On a uniform column the entries are
    identical to ``kappa/dz^2`` with the ``-1`` Neumann / ``-3``
    Dirichlet corners.

    Parameters
    ----------
    field : ScalarField
        The column field (true shape via ``.data``).
    axis : str
        The solve coordinate.
    kappa_value : Any
        The live coefficient: a scalar, or a ``ScalarField`` on the
        field's own function space (face-averaged, one-sided at walls).
    bc : tuple[str, str], optional
        The per-side ``(low, high)`` boundary rows (default:
        ``("neumann", "neumann")``).

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, int]
        The ``lower``, ``diag`` and ``upper`` bands (broadcasting
        against ``field.data``) and the storage-frame axis index.

    Raises
    ------
    ValueError
        If ``axis`` is not a coordinate of the field's space, or a
        field-valued ``kappa`` is on the wrong function space.
    """
    space = field.function_space
    names = space.bare.names
    if axis not in names:
        raise ValueError(
            f"VerticalDiffusion axis {axis!r} is not a coordinate of "
            f"the field space {names}")
    axis_index = names.index(axis)
    grid = field.grid
    real = dtype_real()

    # physical widths: primal cell + wall-including dual face
    m_cell = jnp.asarray(grid.measure(space, axis).data, dtype=real)
    face_space = _face_space(space, axis)
    m_face = jnp.asarray(
        grid.measure(face_space, axis).data, dtype=real)

    # terrain Jacobian (along-sigma) at both staggers, STATIC params
    corrections = getattr(
        getattr(grid, "mapping", None), "column_corrections", {})
    if axis in corrections:
        mapped, base = corrections[axis]
        metric = f"d{mapped}_d{base}"
        m_cell = m_cell * jnp.asarray(
            grid.metric(space, metric, params=None).data, dtype=real)
        m_face = m_face * jnp.asarray(
            grid.metric(face_space, metric, params=None).data,
            dtype=real)

    size = m_cell.shape[axis_index]
    kappa_up, kappa_low = _face_kappa(
        space, axis_index, size, kappa_value, real)
    dz_up = _axis_slice(m_face, axis_index, 1, size + 1)
    dz_low = _axis_slice(m_face, axis_index, 0, size)
    up_coupling = kappa_up / (dz_up * m_cell)
    low_coupling = kappa_low / (dz_low * m_cell)

    # diag as the NEGATED SUM of the kept couplings; a Neumann side
    # drops the wall coupling, a Dirichlet side keeps it (the wall flux)
    low, high = bc
    low_in_diag = (
        low_coupling if low == "dirichlet"
        else _zero_axis_end(low_coupling, axis_index, at_start=True))
    up_in_diag = (
        up_coupling if high == "dirichlet"
        else _zero_axis_end(up_coupling, axis_index, at_start=False))
    diag = -(low_in_diag + up_in_diag)
    lower = _zero_axis_end(low_coupling, axis_index, at_start=True)
    upper = _zero_axis_end(up_coupling, axis_index, at_start=False)
    return lower, diag, upper, axis_index
