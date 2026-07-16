"""
``ScalarField``: (grid, function_space, array) + metadata.

Description
-----------
Owning class doc: ``design/specs/grid/classes/fields.md``. The single
concrete field type: a jaxified pytree whose only leaf is the
storage-shaped ``_data`` array; grid (identity-hashed), space
(interned), and metadata are static aux data — metadata in the
annotation-exempt category (2026-07-08 amendment): it survives
flatten/unflatten but is excluded from aux equality, so treedefs,
jit caching, and scan carries are metadata-insensitive, and jitted
functions return trace-time metadata. Binary arithmetic
implements the strict algebra (rules sections 3.1, 3.3, 3.11): grid
identity first, then the Wave-1 join with the two sanctioned lifts
(constant broadcast, real -> complex promotion).

Dispatch sugar is thin forwarding only (merge decision D3): the
dunders resolve ``("multiply"/"divide"/"power"/"abs", space)``
through ``grid.dispatch`` and apply the registered operator;
``f.diff`` forwards to the seeded ``Dispatched("diff")`` verb; and
``f.to`` is the multi-kind resolver (D3a) reading the conversion
kind from the source/target factor families. There is no elementwise
fallback path — a missing row is a ``DispatchError``.
"""
# Wave 2: ScalarField
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import jaxify
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.errors import (
    GridMismatchError,
    ImmutableStateError,
    SpaceMismatchError,
)
from fridom.spatial.fields.metadata import FieldMetadata
from fridom.spatial.fields.storage import (
    storage_dtype,
    store,
)
from fridom.spatial.operators.base import (
    Dispatched,
    resolve_codomain,
)
from fridom.spatial.operators.registry import DispatchError
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import AverageSpace, CellAvg
from fridom.spatial.spaces.coefficient import CoefficientSpace
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet
from fridom.spatial.spaces.tensor_product import (
    TensorProductSpace,
    join,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax
    import xarray as xr

    from fridom.spatial.decomposition.layout import Layout
    from fridom.spatial.grid import Grid
    from fridom.spatial.scalars import Variance
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike

# Python scalars entering field arithmetic (bool counts as int)
_SCALAR_TYPES = int | float | complex

_DEFAULT_METADATA = FieldMetadata()


@partial(jaxify, dynamic=("_data",), annotation=("_metadata",))
class ScalarField:

    """
    A discrete scalar field on a tensor-product function space.

    Description
    -----------
    Trusting plumbing constructor (jit-hot): takes storage-shaped
    data, performs no validation and no copies. User-facing
    construction goes through ``grid.create_field`` (and the other
    grid factories), which owns validation, padding, and sync.

    Parameters
    ----------
    grid : Grid
        The grid the field was created on.
    function_space : SpaceLike
        The laid-out (product) function space.
    data : jax.Array
        The storage-shaped (halo/stagger-padded), synced local array.
    metadata : FieldMetadata | None, optional
        Annotation metadata; None means the default record
        (default: None).
    halo_valid : HaloSpec | None, optional
        Per-name count of currently valid ghost layers (task 1.8
        halo-validity bookkeeping; internal, static aux data). None
        means zero on every name — claiming fewer valid layers than
        the storage holds is always sound (default: None).
    """

    def __init__(
        self,
        grid: Grid,
        function_space: SpaceLike,
        data: jax.Array,
        metadata: FieldMetadata | None = None,
        halo_valid: HaloSpec | None = None,
    ) -> None:
        """Trusting constructor; see the class docstring."""
        self._grid = grid
        self._function_space = function_space
        self._data = data
        self._metadata = (_DEFAULT_METADATA if metadata is None
                          else metadata)
        self._halo_valid = (
            HaloSpec.zero(tuple(function_space.names))
            if halo_valid is None else halo_valid)

    # ================================================================
    #  Identity (fields have no value equality; section "no
    #  comparisons" of the class doc)
    # ================================================================
    def __eq__(self, other: object) -> bool:
        """Identity: return ``self is other`` (pytree friendly)."""
        return self is other

    def __hash__(self) -> int:
        """Identity hash, matching ``__eq__``."""
        return id(self)

    def __bool__(self) -> bool:
        """Raise TypeError: fields have no truth value."""
        raise TypeError(
            "fields have no truth value; compare arrays explicitly "
            "via f.data")

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def grid(self) -> Grid:
        """The grid this field was created on (section 2.7)."""
        return self._grid

    @property
    def function_space(self) -> SpaceLike:
        """The laid-out (product) function space of the field."""
        return self._function_space

    @property
    def data(self) -> jax.Array:
        """Raw local array at true shape (halo/padding stripped)."""
        return self._grid.decomposition.unpad(
            self._data, self._function_space)

    @data.setter
    def data(self, value: object) -> None:  # noqa: ARG002 — raising stub
        """Reject the write: a field is an immutable pytree (D1.5)."""
        raise ImmutableStateError(
            "ScalarField.data is read-only: a field is an immutable "
            "pytree. Build a new field with f.with_data(new_array) "
            "(or grid.create_field(...)); do not assign f.data = ... "
            "or f.data += ...")

    @property
    def storage(self) -> jax.Array:
        """
        Raw local array in the storage frame (halo/stagger-padded).

        Description
        -----------
        The padded frame the field actually holds — no unpad slice is
        taken. Ghost slots may contain garbage: only ``halo_valid``
        layers of them are meaningful, and a consumer that needs more
        must sync first. Pair with :meth:`with_storage` to round-trip
        a field through a raw-array seam (e.g. a ``lax.scan`` carry)
        without the ``unpad``/``pad`` copies of :attr:`data` /
        :meth:`with_data`.
        """
        return self._data

    @property
    def metadata(self) -> FieldMetadata:
        """Annotation metadata (name/units/nc-attrs)."""
        return self._metadata

    @property
    def name(self) -> str:
        """Shorthand for ``metadata.name``."""
        return self._metadata.name

    @property
    def shape(self) -> tuple[int, ...]:
        """Global true DOF shape, ``function_space.shape``."""
        return self._function_space.shape

    @property
    def dtype(self) -> jnp.dtype:
        """Derived storage dtype (from space scalars + basis; 3.1)."""
        return self._data.dtype

    @property
    def halo_valid(self) -> HaloSpec:
        """
        Per-name count of currently valid ghost layers.

        Description
        -----------
        Internal halo-validity bookkeeping (task 1.8): trace-time
        static aux data, zero runtime cost under jit. Consumers
        (operator applications) sync exactly when the validity is
        below their per-axis requirement; validity participates in
        the pytree treedef, so jit caches key on it.
        """
        return self._halo_valid

    # ================================================================
    #  Functional updates
    # ================================================================
    def with_data(self, data: jax.Array) -> ScalarField:
        """
        Return the field with a new true-shape array.

        Description
        -----------
        The array is routed through ``decomposition.pad``; the
        result claims zero ghost validity and is synced at its
        first ghost-consuming application (task 1.8).

        Parameters
        ----------
        data : jax.Array
            A true-shape array on this field's space.

        Returns
        -------
        ScalarField
            The new field; ``self`` is unchanged.
        """
        space = self._function_space
        stored = store(self._grid.decomposition, space,
                       jnp.asarray(data))
        return ScalarField(self._grid, space, stored, self._metadata)

    def with_storage(self, data: jax.Array) -> ScalarField:
        """
        Return the field with a new storage-frame array.

        Description
        -----------
        The counterpart of :meth:`with_data` for arrays already in
        the storage frame (:attr:`storage`): no pad is performed.
        The result claims zero ghost validity — the canonical state
        of :meth:`with_data`, and always sound, since claiming fewer
        valid layers than the storage holds only costs a sync at the
        first ghost-consuming application.

        Parameters
        ----------
        data : jax.Array
            A storage-shaped array on this field's space.

        Returns
        -------
        ScalarField
            The new field; ``self`` is unchanged.
        """
        return ScalarField(self._grid, self._function_space, data,
                           self._metadata)

    def with_metadata(self, **changes: object) -> ScalarField:
        """
        Return the field with updated metadata.

        Parameters
        ----------
        **changes : object
            ``FieldMetadata.replace`` keyword changes.

        Returns
        -------
        ScalarField
            The re-annotated field; ``self`` is unchanged.
        """
        return ScalarField(self._grid, self._function_space,
                           self._data,
                           self._metadata.replace(**changes),
                           halo_valid=self._halo_valid)

    def retag(self, target: ScalarField | SpaceLike) -> ScalarField:
        """
        Rebuild the field on a BC-sibling space (same nodes, new tag).

        Description
        -----------
        On a walled grid, fields carrying different BC tags on one
        node set must interoperate (nodal operator outputs are
        BC-free; owner decision). ``retag`` adopts the target's BC
        structure without touching the point samples: per axis the
        target factor must differ from the source factor **only** in
        BC structure — same mesh, same node-set class, same shape,
        same scalars — anything else raises ``SpaceMismatchError``.
        The data is unchanged (identical point samples), but halo
        validity resets on the retagged axes (the ghost policy
        changed with the tag) and carries over on the others.

        Parameters
        ----------
        target : ScalarField | SpaceLike
            A field, a full product space, or a single factor space
            (shorthand: retag that factor, keep the rest).

        Returns
        -------
        ScalarField
            The retagged field (``self`` when already on target).
        """
        space = _target_space(self._function_space, target)
        src_bare = self._function_space.bare
        dst_bare = space.bare
        if dst_bare is src_bare:
            return self
        if src_bare.names != dst_bare.names:
            raise SpaceMismatchError(
                f"cannot retag {src_bare!r} onto {dst_bare!r}: "
                "the coordinate names differ",
                left=src_bare, right=dst_bare, operation="retag")
        retagged = []
        for name in dst_bare.names:
            src = src_bare.factor(name)
            dst = dst_bare.factor(name)
            if src is dst:
                continue
            if not _bc_siblings(src, dst):
                raise SpaceMismatchError(
                    f"retag changes BC structure only: at {name!r} "
                    f"the factors {src!r} and {dst!r} differ beyond "
                    "their BC tags (mesh, node-set class, shape and "
                    "scalars must match); use .to for a conversion",
                    left=src, right=dst, operation="retag",
                    mismatched_names=(name,))
            retagged.append(name)
        new_space: SpaceLike = dst_bare
        if self._function_space.layout is not None:
            new_space = dst_bare.with_layout(
                self._function_space.layout)
        halo_valid = HaloSpec({
            name: 0 if name in retagged else self._halo_valid[name]
            for name in new_space.names})
        return ScalarField(self._grid, new_space, self._data,
                           self._metadata, halo_valid=halo_valid)

    def with_variance(self, variance: Variance | None) -> ScalarField:
        """
        Return the field on the variance-tagged space variant.

        Description
        -----------
        A variance tag is a pure claim about the component's role
        (covariant/contravariant, validation section 6.3): the data,
        the staggering, and the ghost validity are untouched — only
        the interned space identity changes, so the strict algebra
        distinguishes (and refuses to mix) differently-tagged
        components. ``None`` strips the tag.

        Parameters
        ----------
        variance : Variance | None
            The component variance to claim; None strips the tag.

        Returns
        -------
        ScalarField
            The retagged field (``self`` when already tagged so).
        """
        space = self._function_space.with_variance(variance)
        if space is self._function_space:
            return self
        return ScalarField(self._grid, space, self._data,
                           self._metadata,
                           halo_valid=self._halo_valid)

    # ================================================================
    #  Scalars (Körper) surface — section 3.1
    # ================================================================
    def as_complex(self) -> ScalarField:
        """Explicit promotion onto the complexified space."""
        if self._function_space.scalars is Scalars.COMPLEX:
            return self
        space = _promoted_space(self._function_space)
        data = self.data.astype(storage_dtype(space))
        stored = store(self._grid.decomposition, space, data)
        return ScalarField(self._grid, space, stored, self._metadata)

    @property
    def real(self) -> ScalarField:
        """The real part, an ``fr.Real`` field (identity if real)."""
        if self._function_space.scalars is Scalars.REAL:
            return self
        _require_no_coefficient(self._function_space, "real")
        space = _real_space(self._function_space)
        stored = store(self._grid.decomposition, space,
                       self.data.real)
        return ScalarField(self._grid, space, stored, self._metadata)

    @property
    def imag(self) -> ScalarField:
        """The imaginary part, an ``fr.Real`` field (zero if real)."""
        if self._function_space.scalars is Scalars.REAL:
            # zeros are the exact imag values everywhere, ghosts
            # included: the operand's valid layers carry over
            return ScalarField(self._grid, self._function_space,
                               jnp.zeros_like(self._data),
                               self._metadata,
                               halo_valid=self._halo_valid)
        _require_no_coefficient(self._function_space, "imag")
        space = _real_space(self._function_space)
        stored = store(self._grid.decomposition, space,
                       self.data.imag)
        return ScalarField(self._grid, space, stored, self._metadata)

    def conj(self) -> ScalarField:
        """Complex conjugate on the same space (identity if real)."""
        if self._function_space.scalars is Scalars.REAL:
            return self
        _require_no_coefficient(self._function_space, "conj")
        # pointwise on the storage frame: valid ghosts stay valid
        return ScalarField(self._grid, self._function_space,
                           jnp.conj(self._data), self._metadata,
                           halo_valid=self._halo_valid)

    # ================================================================
    #  Dispatch sugar — section 3.4 (thin forwarders, D3/D3a)
    # ================================================================
    def diff(self, name: str) -> ScalarField:
        """
        Default derivative along ``name``: (kind="diff", space).

        Description
        -----------
        Thin forwarder to the seeded verb (D3):
        ``fr.operators.diff[name](self)``.

        Parameters
        ----------
        name : str
            The coordinate name to differentiate along.

        Returns
        -------
        ScalarField
            The derivative on the registered operator's codomain.
        """
        return Dispatched("diff")[name](self)

    def to(self, target: ScalarField | SpaceLike) -> ScalarField:
        """
        Convert per axis onto the target's space.

        Description
        -----------
        The multi-kind resolver (D3a): per differing factor it reads
        the conversion kind from the source/target family
        relationship (nodal -> nodal ``"interpolate"``, average
        source ``"reconstruct"``, nodal -> average ``"average"``,
        coefficient -> coefficient ``"interpolate"``), resolves
        ``(kind, source_factor)`` in the grid registry, and applies
        the bound operator. A registered codomain that is a
        BC-sibling of the requested target factor (nodal operator
        outputs are BC-free; owner decision) adopts the requested
        tag via ``retag``; any other disagreement raises
        ``SpaceMismatchError``; nodal <-> coefficient targets raise
        (a ``.to`` is not a transform).

        Parameters
        ----------
        target : ScalarField | SpaceLike
            A field, a full product space, or a single factor space
            (shorthand: convert that factor, keep the rest).

        Returns
        -------
        ScalarField
            The converted field (``self`` when already on target).
        """
        space = _target_space(self._function_space, target)
        src_bare = self._function_space.bare
        dst_bare = space.bare
        if dst_bare is src_bare:
            return self
        if src_bare.names != dst_bare.names:
            raise SpaceMismatchError(
                f"cannot convert {src_bare!r} onto {dst_bare!r}: "
                "the coordinate names differ",
                left=src_bare, right=dst_bare, operation="to")
        result = self
        for name in dst_bare.names:
            src = result.function_space.bare.factor(name)
            dst = dst_bare.factor(name)
            if src is dst:
                continue
            if isinstance(src, ConstantSpace):
                result = _broadcast_factor(result, name, dst)
                continue
            kind = _conversion_kind(src, dst)
            op = self._grid.dispatch.resolve(kind, src)[name]
            resolved = resolve_codomain(
                op, result.function_space).factor(name)
            if resolved is not dst:
                if _bc_siblings(resolved, dst):
                    # nodal operator outputs are BC-free (owner
                    # decision): adopt the requested sibling tag
                    result = op(result).retag(dst)
                    continue
                raise SpaceMismatchError(
                    f"the registered ({kind!r}, {src!r}) operator "
                    f"lands on {resolved!r}, not the requested "
                    f"{dst!r}; use an explicit operator instance "
                    "or a registry override",
                    left=src, right=dst, operation="to")
            result = op(result)
        return result

    def reshard(self, target: Layout) -> ScalarField:
        """
        Explicit layout change (never implicit in arithmetic).

        Description
        -----------
        Thin sugar over the grid-bound ``Reshard`` movement operator
        (section 5.1): the target must be in the negotiated layout
        vocabulary; a matching layout elides the application
        (identity).

        Parameters
        ----------
        target : Layout
            The target layout (member of
            ``grid.decomposition.layouts``).

        Returns
        -------
        ScalarField
            The field in the target layout (``self`` when already
            there).
        """
        if self._function_space.layout == target:
            return self  # identity elision
        from fridom.spatial.operators.movement import (  # noqa: PLC0415 — keep the movement import off the field-core import path
            Reshard,
        )
        return Reshard(self._grid, target)(self)

    def integrate(self, *names: str) -> ScalarField:
        """
        Weighted integral; named factors reduce to ConstantSpace.

        Description
        -----------
        Thin forwarder to the seeded verb (D3): per name,
        ``fr.operators.integrate[name](self)`` resolves
        ``("integrate", factor)`` against ``grid.dispatch`` (rules
        section 3.13). No names integrates every factor; reductions
        along ``ConstantSpace`` factors are the identity; the result
        broadcasts back via ``ConstantSpace`` (section 3.3), so
        ``f - f.integrate("x")`` stays in the strict algebra.

        Parameters
        ----------
        *names : str
            The coordinate names to reduce (default: all).

        Returns
        -------
        ScalarField
            The integral on the reduced space (default metadata).
        """
        space = self._function_space.bare
        result = self
        for name in _reduction_names(space, names):
            factor = space.factor(name)
            if isinstance(factor, ConstantSpace):
                continue  # identity reduction (section 3.13)
            if isinstance(factor, CoefficientSpace):
                raise DispatchError(
                    "no ('integrate', coefficient factor) dispatch "
                    f"entry for {factor!r}: transform back first "
                    "(the zero-mode extraction is designed-for)")
            result = Dispatched("integrate")[name](result)
        return result

    def mean(self, *names: str) -> ScalarField:
        """
        Integral divided by the integrated measure (sugar).

        Description
        -----------
        ``f.integrate(*names)`` scaled by the reciprocal of the
        total measure of the reduced factors (the per-name sums of
        ``grid.measure``), so the mean of a constant is that
        constant on every space family.

        Parameters
        ----------
        *names : str
            The coordinate names to average over (default: all).

        Returns
        -------
        ScalarField
            The mean on the reduced space (default metadata).
        """
        space = self._function_space.bare
        selected = _reduction_names(space, names)
        integral = self.integrate(*selected)
        total = None
        for name in selected:
            factor = space.factor(name)
            if isinstance(factor, ConstantSpace):
                continue
            weight = self._grid.measure(space, name=name)
            length = weight.data.sum()
            total = length if total is None else total * length
        if total is None:
            return integral  # all-constant: identity
        return integral.with_data(integral.data / total)

    def item(self) -> complex | float:
        """
        Return the single value of a one-DOF field (host scalar).

        Description
        -----------
        The host read of a fully reduced field — the tidy end of an
        integral: ``field.integrate().item()`` instead of
        ``float(field.integrate().data.ravel()[0])``. Only defined
        when the field carries exactly one DOF (every factor reduced
        to a ``ConstantSpace``); a real dtype returns ``float``, a
        complex one ``complex``. Forces the computation (a device
        sync).

        Returns
        -------
        complex | float
            The single value.

        Raises
        ------
        ValueError
            If the field carries more than one DOF.
        """
        data = self.data
        if data.size != 1:
            raise ValueError(
                f"item() needs a one-DOF field, got shape "
                f"{tuple(data.shape)} on {self._function_space!r}; "
                "reduce first (e.g. field.integrate())")
        return data.ravel()[0].item()

    # ================================================================
    #  Grid accessor forwarders (a field carries its grid + space)
    # ================================================================
    def nodes(self, name: str | None = None) -> ScalarField:
        """
        Physical coordinates of this field's evaluation nodes.

        Description
        -----------
        Thin forwarder to ``grid.evaluation_nodes`` with this field's
        own function space, so callers need not re-thread grid and
        space (section 2.7). ``name`` may be omitted when the space
        contributes one non-constant name.

        Parameters
        ----------
        name : str | None, optional
            The coordinate to materialize; may be omitted when
            unambiguous (default: None).

        Returns
        -------
        ScalarField
            The per-factor node coordinates as a field.
        """
        return self._grid.evaluation_nodes(self._function_space, name)

    def measure(self, name: str | None = None) -> ScalarField:
        """
        Metric measure proper to this field's node set.

        Description
        -----------
        Thin forwarder to ``grid.measure`` with this field's own
        function space (section 3.9). ``name`` may be omitted when the
        space contributes one non-constant name.

        Parameters
        ----------
        name : str | None, optional
            The coordinate whose measure to materialize; may be
            omitted when unambiguous (default: None).

        Returns
        -------
        ScalarField
            The per-factor measure weights as a field.
        """
        return self._grid.measure(self._function_space, name)

    def wavenumbers(self, name: str | None = None) -> ScalarField:
        """
        Wavenumbers (or mode indices) of this coefficient field.

        Description
        -----------
        Thin forwarder to ``grid.wavenumbers`` with this field's own
        function space (section 3.10). ``name`` may be omitted when
        the space contributes one non-constant name.

        Parameters
        ----------
        name : str | None, optional
            The coordinate whose wavenumbers to materialize; may be
            omitted when unambiguous (default: None).

        Returns
        -------
        ScalarField
            The per-factor wavenumbers as a field.
        """
        return self._grid.wavenumbers(self._function_space, name)

    # ================================================================
    #  Arithmetic — sections 3.1, 3.3, 3.11 (join rule)
    # ================================================================
    def __add__(self, other: ScalarField | complex) -> ScalarField:
        """Linear; join rule. Python scalars are constant fields."""
        if isinstance(other, ScalarField):
            return _linear_combine(self, other, "+",
                                   lambda x, y: x + y)
        if _is_scalar(other):
            return _scalar_shift(self, other, lambda d, s: d + s)
        return NotImplemented

    def __radd__(self, other: complex) -> ScalarField:
        """Scalar + field (fields handle field + field)."""
        if _is_scalar(other):
            return _scalar_shift(self, other, lambda d, s: s + d)
        return NotImplemented

    def __sub__(self, other: ScalarField | complex) -> ScalarField:
        """Linear; join rule."""
        if isinstance(other, ScalarField):
            return _linear_combine(self, other, "-",
                                   lambda x, y: x - y)
        if _is_scalar(other):
            return _scalar_shift(self, other, lambda d, s: d - s)
        return NotImplemented

    def __rsub__(self, other: complex) -> ScalarField:
        """Scalar - field."""
        if _is_scalar(other):
            return _scalar_shift(self, other, lambda d, s: s - d)
        return NotImplemented

    def __neg__(self) -> ScalarField:
        """Negation (a new quantity: default metadata)."""
        # pointwise on the storage frame; negation commutes with
        # every ghost fill (linear-homogeneous), so valid ghost
        # slots stay valid (task 1.8, stage B)
        return ScalarField(self._grid, self._function_space,
                           -self._data,
                           halo_valid=self._halo_valid)

    def __pos__(self) -> ScalarField:
        """Identity."""
        return self

    def __mul__(self, other: ScalarField | complex) -> ScalarField:
        """Scalar: linear scaling. Field: dispatched product (3.11)."""
        if isinstance(other, ScalarField):
            return _dispatched_product(self, other, "multiply", "*")
        if _is_scalar(other):
            return _scalar_scale(self, other, lambda d, s: d * s)
        return NotImplemented

    def __rmul__(self, other: complex) -> ScalarField:
        """Scalar * field (linear scaling on any space)."""
        if _is_scalar(other):
            return _scalar_scale(self, other, lambda d, s: s * d)
        return NotImplemented

    def __truediv__(
        self, other: ScalarField | complex,
    ) -> ScalarField:
        """Scalar: linear scaling. Field: (kind="divide", space)."""
        if isinstance(other, ScalarField):
            return _dispatched_product(self, other, "divide", "/")
        if _is_scalar(other):
            return _scalar_scale(self, other, lambda d, s: d / s)
        return NotImplemented

    def __rtruediv__(self, other: complex) -> ScalarField:
        """Scalar / field: a physical divide (nodal/average only)."""
        if not _is_scalar(other):
            return NotImplemented
        space = self._function_space
        if isinstance(other, complex):
            space = _promoted_space(space)
        op = self._grid.dispatch.resolve("divide", space.bare)
        numerator = _wrap(self._grid, space,
                          jnp.full(space.shape, other))
        return op(numerator, _lift_field(self, space))

    def __pow__(self, exponent: float) -> ScalarField:
        """Physical power, (kind="power", space) (2.5 table)."""
        if not (isinstance(exponent, int | float)
                or _is_0d_array(exponent)):
            return NotImplemented
        space = self._function_space
        op = self._grid.dispatch.resolve("power", space.bare)
        lifted = _wrap(self._grid, space,
                       jnp.full(space.shape, exponent))
        return op(self, lifted)

    def __abs__(self) -> ScalarField:
        """Pointwise modulus, (kind="abs", space); nodal default."""
        space = self._function_space
        op = self._grid.dispatch.resolve("abs", space.bare)
        return op(self)

    # ================================================================
    #  Diagnostics and export
    # ================================================================
    def has_nan(self) -> jax.Array:
        """0-d boolean array: any NaN in the (global) field."""
        return jnp.isnan(self.data).any()

    def block_until_ready(self) -> ScalarField:
        """Wait for async device work; returns self."""
        self._data.block_until_ready()
        return self

    @property
    def xr(self) -> xr.DataArray:
        """
        Export to an ``xarray.DataArray``.

        Description
        -----------
        Thin field-side entry point; the coordinate-label rules
        (average-space cell labels, wavenumber coords) and the
        ``decomposition.gather`` data path live in
        ``fridom.spatial.export``. A lone ``DataArray`` cannot
        collide with sibling variables, so staggered dims export
        under the plain axis names (``x``, not ``x_right``); the
        position stays available as the ``c_grid_axis_shift``
        coordinate attribute. Multi-variable exports
        (``VectorField.xr``, the io stores) keep the xgcm
        position-suffixed names.

        Returns
        -------
        xr.DataArray
            Global true-shape data with labeled coordinates.
        """
        from fridom.spatial.export import (  # noqa: PLC0415 — deferred: keeps optional xarray off the field-core import path
            scalar_to_dataarray,
        )
        return scalar_to_dataarray(self, positions_in_names=False)

    def __repr__(self) -> str:
        """Name, space, shape, dtype summary."""
        return (f"ScalarField({self.name!r}, "
                f"space={self._function_space!r}, "
                f"shape={self.shape}, dtype={self.dtype})")


# ================================================================
#  Shared arithmetic plumbing
# ================================================================
def _reduction_names(
    space: SpaceLike, names: tuple[str, ...],
) -> tuple[str, ...]:
    """
    Normalize a reduction's name selection (integrate/mean sugar).

    Description
    -----------
    No names selects every factor; explicit names are validated
    against the space and deduplicated preserving order (repeated
    reductions along one name are the identity anyway).

    Parameters
    ----------
    space : SpaceLike
        The (bare) operand space.
    names : tuple[str, ...]
        The user-selected coordinate names (possibly empty).

    Returns
    -------
    tuple[str, ...]
        The validated, deduplicated selection.
    """
    if not names:
        return space.names
    unknown = tuple(name for name in names
                    if name not in space.names)
    if unknown:
        raise ValueError(
            f"no factors named {unknown}; this space's names are "
            f"{space.names}")
    return tuple(dict.fromkeys(names))


def _wrap(grid: Grid, space: SpaceLike,
          true_data: jax.Array) -> ScalarField:
    """Build a default-metadata result field from true-shape data."""
    return ScalarField(grid, space,
                       store(grid.decomposition, space, true_data))


def _check_grids(a: ScalarField, b: ScalarField,
                 operation: str) -> None:
    """Raise GridMismatchError unless both fields share one grid."""
    if a.grid is not b.grid:
        raise GridMismatchError(
            f"operands of {operation!r} live on different grids; "
            "fields combine only with fields created on the "
            "identical grid object",
            left=a.grid, right=b.grid, operation=operation)


def _map_factors(
    space: SpaceLike,
    fn: Callable[[FunctionSpace], FunctionSpace],
) -> SpaceLike:
    """Per-factor rebuild (layout and variance preserved)."""
    if isinstance(space, TensorProductSpace):
        new = TensorProductSpace.of(
            *(fn(factor) for factor in space.factors))
    else:
        new = fn(space.bare.with_variance(None))
    if space.variance is not None:
        new = new.with_variance(space.variance)
    if space.layout is not None:
        new = new.with_layout(space.layout)
    return new


def _real_space(space: SpaceLike) -> SpaceLike:
    """Return the per-factor ``fr.Real`` variant of ``space``."""
    return _map_factors(space, lambda factor: factor.as_real())


def _promoted_space(space: SpaceLike) -> SpaceLike:
    """Return the complexified space (guard shape changes)."""
    for factor in space.factors:
        if (isinstance(factor, CoefficientSpace)
                and factor.as_complex().shape != factor.shape):
            raise NotImplementedError(
                "promoting a real-origin half-spectrum coefficient "
                f"factor {factor!r} to complex changes its shape "
                "(Hermitian unfold); not implemented in iteration 1")
    return _map_factors(space, lambda factor: factor.as_complex())


def _require_no_coefficient(space: SpaceLike,
                            operation: str) -> None:
    """Reject complex coefficient factors for value-wise Körper ops."""
    for factor in space.factors:
        if isinstance(factor, CoefficientSpace):
            raise NotImplementedError(
                f"{operation} on complex coefficient-space fields "
                "mixes conjugate modes; not implemented in "
                "iteration 1 (transform back first)")


def _check_lift(from_space: SpaceLike, to_space: SpaceLike) -> None:
    """
    Validate that the sanctioned lifts are realizable elementwise.

    Description
    -----------
    The join already established that the factors are related by the
    two sanctioned lifts; this checks the iteration-1 *realization*:
    constant broadcast is a plain jnp broadcast onto nodal/average
    factors (the seeded ``("broadcast", ConstantSpace)`` default),
    and real -> complex promotion must not change the factor shape.
    """
    pairs = zip(from_space.factors, to_space.factors, strict=True)
    for src, dst in pairs:
        if src is dst:
            continue
        if isinstance(src, ConstantSpace):
            if isinstance(dst, CoefficientSpace):
                raise DispatchError(
                    "no ('broadcast', ConstantSpace -> "
                    f"{dst!r}) dispatch entry: broadcasting a "
                    "constant into a coefficient space is the "
                    "zero-mode update, not implemented in "
                    "iteration 1")
            continue
        if src.shape != dst.shape:
            raise NotImplementedError(
                f"lifting {src!r} to {dst!r} changes the factor "
                "shape (Hermitian unfold); not implemented in "
                "iteration 1")


def _linear_combine(
    a: ScalarField,
    b: ScalarField,
    operation: str,
    data_op: Callable[[jax.Array, jax.Array], jax.Array],
) -> ScalarField:
    """
    Grid check, join, lift, elementwise combine (for +/-).

    Description
    -----------
    Operands already on the joined space combine on the aligned
    **storage frames** and claim the pointwise minimum of the
    operands' ghost validity (the products precedent,
    ``operators/products.py``): every iteration-1 ghost fill
    (periodic wrap, Dirichlet odd/vacant, Neumann even) is
    linear-homogeneous in the interior DOFs, so the combined valid
    ghost slots equal the fill of the combination bitwise, and
    downstream stencil consumers whose reach the claim covers skip
    their sync. The fast path also drops the per-op unpad/re-pad
    round trip of the true-shape route — the full-array passes the
    2026-07-12 512^3 A100 profile priced at ~9.5 ms of the 62 ms
    linear AB3 step. History: Wave 4B measured the storage-frame
    combine WITHOUT the validity claim and reverted it on cpu
    numbers (``benchmarks/RESULTS.md``); with the claim the same
    matched benchmark now *improves* on cpu too (2026-07-12
    re-measurement). Lift cases (constant broadcast, real ->
    complex across spaces) keep the true-shape route.
    """
    _check_grids(a, b, operation)
    joined = join(a.function_space, b.function_space,
                  operation=operation)
    _check_lift(a.function_space, joined)
    _check_lift(b.function_space, joined)
    if a.function_space is joined and b.function_space is joined:
        return type(a)(
            a.grid, joined,
            data_op(a._data, b._data),  # noqa: SLF001 — storage seam
            halo_valid=a.halo_valid.merge_min(b.halo_valid))
    return _wrap(a.grid, joined, data_op(a.data, b.data))


def _is_0d_array(value: object) -> bool:
    """
    Whether ``value`` is a 0-d array (a scalar, not a field).

    Description
    -----------
    A raw 0-d ``jax.Array`` (e.g. a traced ``ctx.params`` leaf such as
    ``dsqr``) is a scalar coefficient. An array with ``ndim >= 1`` is
    **not** a scalar: a field-shaped array carries no space tags and
    must enter as a ``ScalarField``. Mirrors ``Symbol._is_scalar``
    (commit 309bbdd).
    """
    return (not isinstance(value, ScalarField)
            and getattr(value, "ndim", None) == 0)


def _is_scalar(value: object) -> bool:
    """Whether ``value`` enters +/-/*// as a scalar operand."""
    return isinstance(value, _SCALAR_TYPES) or _is_0d_array(value)


def _scalar_shift(
    f: ScalarField,
    value: complex,
    data_op: Callable[[jax.Array, complex], jax.Array],
) -> ScalarField:
    """
    Python scalar in +/-: a constant field entering the join.

    Description
    -----------
    Storage-frame shift (promotion is shape-guarded, so the frames
    stay aligned). The result keeps the operand's ghost claim on
    **periodic** axes only — the wrap fill reproduces constants,
    the bounded fills (Dirichlet odd/vacant) do not, so those axes
    drop to zero and refill at the next consumption (the
    ``apply_staggered`` bounded-axis policy).
    """
    space = f.function_space
    for factor in space.factors:
        if isinstance(factor, CoefficientSpace):
            raise DispatchError(
                "no ('broadcast', ConstantSpace -> "
                f"{factor!r}) dispatch entry: adding a Python "
                "scalar to a coefficient-space field is the exact "
                "zero-mode update, not implemented in iteration 1")
    if isinstance(value, complex):
        space = _promoted_space(space)
    valid = HaloSpec({
        name: (f.halo_valid[name]
               if getattr(factor.mesh, "periodic", False) else 0)
        for factor in space.factors
        for name in factor.names})
    return type(f)(f.grid, space,
                   data_op(f._data, value),  # noqa: SLF001 — storage seam
                   halo_valid=valid)


def _scalar_scale(
    f: ScalarField,
    value: complex,
    data_op: Callable[[jax.Array, complex], jax.Array],
) -> ScalarField:
    """
    Python scalar in * and /: linear scaling on any space.

    Description
    -----------
    Storage-frame scaling (promotion is shape-guarded, so the
    frames stay aligned); scaling commutes with every ghost fill
    (linear-homogeneous), so the operand's ghost claim carries over
    (task 1.8, stage B).
    """
    space = f.function_space
    if isinstance(value, complex):
        space = _promoted_space(space)
    return type(f)(f.grid, space,
                   data_op(f._data, value),  # noqa: SLF001 — storage seam
                   halo_valid=f.halo_valid)


# ================================================================
#  Registry dispatch of the product dunders
# ================================================================
# ``grid.dispatch`` is the (duck-typed) ``OperatorRegistry`` seeded
# by the grid constructor: products resolve
# ``registry.resolve(kind, joined.bare)`` and apply the returned
# binary operator to the lifted operands. Resolution errors are
# ``DispatchError``s (a ``KeyError`` subclass) and propagate
# untouched; there is no elementwise fallback path.


def _lift_field(f: ScalarField, joined: SpaceLike) -> ScalarField:
    """Materialize the lift of ``f`` onto the joined space."""
    if f.function_space is joined:
        return f
    data = jnp.broadcast_to(f.data, joined.shape)
    data = data.astype(
        jnp.promote_types(data.dtype, storage_dtype(joined)))
    return _wrap(f.grid, joined, data)


def _dispatched_product(
    a: ScalarField,
    b: ScalarField,
    kind: str,
    operation: str,
) -> ScalarField:
    """Grid check, join, lift, then registry dispatch."""
    _check_grids(a, b, operation)
    joined = join(a.function_space, b.function_space,
                  operation=operation)
    _check_lift(a.function_space, joined)
    _check_lift(b.function_space, joined)
    op = a.grid.dispatch.resolve(kind, joined.bare)
    return op(_lift_field(a, joined), _lift_field(b, joined))


def _broadcast_factor(
    f: ScalarField, name: str, dst: FunctionSpace,
) -> ScalarField:
    """
    Sanctioned constant broadcast (3.3): a ConstantSpace factor -> dst.

    Description
    -----------
    The ``.to`` realization of the constant-broadcast lift. It reuses
    the eager join-broadcast (``_lift_field``), so an explicit
    ``profile.to(nodal)`` and the implicit lift inside ``profile * f``
    produce the identical field. Halo 0 — a broadcast reads the single
    DOF and adds no ghost demand. Broadcasting into a coefficient factor
    is the zero-mode update, a ``DispatchError`` in iteration 1.

    Parameters
    ----------
    f : ScalarField
        The field carrying a ``ConstantSpace`` factor at ``name``.
    name : str
        The coordinate name of the constant factor to broadcast.
    dst : FunctionSpace
        The requested (bare) target factor.

    Returns
    -------
    ScalarField
        ``f`` broadcast onto the factor-replaced space.
    """
    if isinstance(dst, CoefficientSpace):
        raise DispatchError(
            "no ('broadcast', ConstantSpace -> "
            f"{dst!r}) dispatch entry: broadcasting a constant into a "
            "coefficient space is the zero-mode update, not implemented "
            "in iteration 1")
    space = f.function_space
    if isinstance(space, TensorProductSpace):
        target: SpaceLike = space.replace(**{name: dst})
    elif space.layout is not None:
        target = dst.with_layout(space.layout)
    else:
        target = dst
    return _lift_field(f, target)


def _bc_siblings(src: FunctionSpace, dst: FunctionSpace) -> bool:
    """
    Whether two distinct factors differ only in BC structure.

    Description
    -----------
    The sibling relation of the retag seam: identical space class,
    mesh, shape, and scalars. Under mesh interning two *distinct*
    bare factors agreeing on all of these can only differ in their
    BC structure, so no explicit ``bc`` comparison is needed.
    Restricted to nodal **and average** factors — both now carry
    retaggable BC tags (a Neumann/Dirichlet ``CellAvg`` origin keeps
    shape ``(n,)``, so ``div.retag(neumann_cellavg_sibling)`` in the
    walled FV pressure solve is a pure BC-tag swap), while a
    coefficient factor's BC lives in its origin and is not retagged.

    Parameters
    ----------
    src : FunctionSpace
        The source (bare) factor.
    dst : FunctionSpace
        The requested (bare) target factor.

    Returns
    -------
    bool
        True iff the factors are BC-siblings.
    """
    return (isinstance(src, NodalSpace | AverageSpace)
            and type(src) is type(dst)
            and src.mesh is dst.mesh
            and src.shape == dst.shape
            and src.scalars is dst.scalars)


def _conversion_kind(
    src: FunctionSpace, dst: FunctionSpace,
) -> str:
    """
    Read the per-axis ``.to`` kind off the family matrix (fields.md).

    Parameters
    ----------
    src : FunctionSpace
        The source factor space.
    dst : FunctionSpace
        The requested target factor space.

    Returns
    -------
    str
        The dispatch kind realizing the conversion.
    """
    src_coeff = isinstance(src, CoefficientSpace)
    dst_coeff = isinstance(dst, CoefficientSpace)
    if src_coeff != dst_coeff:
        raise SpaceMismatchError(
            f"a .to from {src!r} to {dst!r} is not a conversion but "
            "a transform; use fr.operators.Fourier(grid, axes=...)"
            ".forward/.backward", left=src, right=dst, operation="to")
    if src_coeff:
        return "interpolate"  # exact inter-origin shift (3.2)
    if _colocated_avg_nodal(src, dst):
        return "deconvolve"  # same-location CellAvg <-> Center (3.9)
    if isinstance(src, AverageSpace):
        return "reconstruct"
    if isinstance(src, NodalSpace):
        if isinstance(dst, AverageSpace):
            return "average"  # quadrature projection (later)
        if isinstance(dst, NodalSpace):
            if (src.node_set is NodeSet.OUTER
                    and dst.node_set is NodeSet.INNER):
                # the both-boundary face set restricts onto its shared
                # interior faces (Outer ⊃ Inner): an exact node drop,
                # not the half-cell average the interpolate kind owns
                return "restrict"
            return "interpolate"
    raise SpaceMismatchError(
        f"no .to conversion is defined from {src!r} to {dst!r}",
        left=src, right=dst, operation="to")


def _colocated_avg_nodal(
    src: FunctionSpace, dst: FunctionSpace,
) -> bool:
    """
    Whether ``src``/``dst`` are the co-located ``CellAvg <-> Center`` pair.

    Description
    -----------
    The routing predicate of the ``"deconvolve"`` kind (rules 3.9): a
    cell average and the point value at the *same* location (the cell
    midpoint) on one mesh — either order. It is exactly this pair the
    staggering ``"reconstruct"``/``"average"`` kinds cannot reach (they
    are committed to the half-cell ``CellAvg -> Right`` / ``Center ->
    FaceAvg`` shifts, and a registry key resolves one codomain). The
    dual ``FaceAvg <-> Right`` pair is designed-for and deliberately
    not matched (FV-D2 option A never instantiates ``FaceAvg``).

    Parameters
    ----------
    src : FunctionSpace
        The source factor space.
    dst : FunctionSpace
        The requested target factor space.

    Returns
    -------
    bool
        True iff one factor is a ``CellAvg`` and the other its
        co-located ``Center`` on the same mesh.
    """
    avg, nod = (src, dst) if isinstance(src, AverageSpace) else (dst, src)
    return (isinstance(avg, CellAvg)
            and isinstance(nod, NodalSpace)
            and nod.node_set is NodeSet.CENTER
            and avg.mesh is nod.mesh)


def _target_space(
    space: SpaceLike,
    target: ScalarField | SpaceLike,
) -> SpaceLike:
    """Resolve a ``to`` target: field/tracer, product, or single factor."""
    # A field-like target (a ``ScalarField``, or a ``HaloTracer`` during
    # a halo trace) carries its own space; duck-type on
    # ``function_space`` so ``x.to(field)`` resolves in both the numeric
    # and the tracing pass. No ``SpaceLike`` exposes ``function_space``.
    resolved = getattr(target, "function_space", None)
    if resolved is not None:
        return resolved
    if (not isinstance(target, TensorProductSpace)
            and isinstance(space, TensorProductSpace)):
        # single-factor shorthand: replace that factor, keep the rest
        return space.replace(
            **{target.names[0]: target.bare})
    return target
