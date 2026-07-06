"""
``ScalarField``: (grid, function_space, array) + metadata.

Description
-----------
Owning class doc: ``notes/framework2/classes/fields.md``. The single
concrete field type: a jaxified pytree whose only leaf is the
storage-shaped ``_data`` array; grid (identity-hashed), space
(interned), and metadata are static aux data. Binary arithmetic
implements the strict algebra (rules sections 3.1, 3.3, 3.11): grid
identity first, then the Wave-1 join with the two sanctioned lifts
(constant broadcast, real -> complex promotion).

Iteration-1 dispatch seam: products (``*``, ``/``) resolve through
``grid.dispatch`` (the duck-typed ``OperatorRegistry``) when the grid
carries one; without a registry they fall back to the clearly marked
elementwise defaults below, which mirror the iteration-1 default
table (nodal/average rows only). The Wave-2 registry merge swaps the
real registry in via ``Grid(dispatch=...)``.
"""
# Wave 2: ScalarField
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import jaxify
from fridom.framework2.grid.errors import GridMismatchError
from fridom.framework2.grid.fields.metadata import FieldMetadata
from fridom.framework2.grid.fields.storage import (
    storage_dtype,
    store,
)
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.average import AverageSpace
from fridom.framework2.grid.spaces.coefficient import CoefficientSpace
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.grid.spaces.nodal import NodalSpace
from fridom.framework2.grid.spaces.tensor_product import (
    TensorProductSpace,
    join,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike

# Python scalars entering field arithmetic (bool counts as int)
_SCALAR_TYPES = int | float | complex

_DEFAULT_METADATA = FieldMetadata()


@partial(jaxify, dynamic=("_data",))
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
    """

    def __init__(
        self,
        grid: Grid,
        function_space: SpaceLike,
        data: jax.Array,
        metadata: FieldMetadata | None = None,
    ) -> None:
        """Trusting constructor; see the class docstring."""
        self._grid = grid
        self._function_space = function_space
        self._data = data
        self._metadata = (_DEFAULT_METADATA if metadata is None
                          else metadata)

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

    # ================================================================
    #  Functional updates
    # ================================================================
    def with_data(self, data: jax.Array) -> ScalarField:
        """
        Return the field with a new true-shape array.

        Description
        -----------
        The array is routed through ``decomposition.pad`` and the
        halo sync, so stored halos are always valid.

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
                           self._metadata.replace(**changes))

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
            return ScalarField(self._grid, self._function_space,
                               jnp.zeros_like(self._data),
                               self._metadata)
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
        return ScalarField(self._grid, self._function_space,
                           jnp.conj(self._data), self._metadata)

    # ================================================================
    #  Dispatch sugar — section 3.4 (operator wiring lands with the
    #  Wave-2 registry merge; the signatures are the it-1 surface)
    # ================================================================
    def diff(self, name: str) -> ScalarField:
        """Default derivative along ``name``: (kind="diff", space)."""
        raise NotImplementedError(
            "f.diff resolves ('diff', factor) through the operator "
            "registry; it is wired in the Wave-2 operators merge")

    def to(self, target: ScalarField | SpaceLike) -> ScalarField:
        """Convert per axis onto the target's space."""
        space = _target_space(self._function_space, target)
        if space.bare is self._function_space.bare:
            return self
        raise NotImplementedError(
            "f.to conversions resolve through the operator "
            "registry; they are wired in the Wave-2 operators merge")

    def reshard(self, target: object) -> ScalarField:
        """Explicit layout change (never implicit in arithmetic)."""
        raise NotImplementedError(
            "reshard is sugar over the Reshard movement operator; "
            "multi-device layouts arrive in Wave 3")

    def integrate(self, *names: str) -> ScalarField:
        """Weighted integral; named factors reduce to ConstantSpace."""
        raise NotImplementedError(
            "f.integrate resolves ('integrate', factor) through the "
            "operator registry; it is wired in the Wave-2 operators "
            "merge")

    def mean(self, *names: str) -> ScalarField:
        """Integral divided by the integrated measure (sugar)."""
        raise NotImplementedError(
            "f.mean is sugar over f.integrate; it is wired in the "
            "Wave-2 operators merge")

    # ================================================================
    #  Arithmetic — sections 3.1, 3.3, 3.11 (join rule)
    # ================================================================
    def __add__(self, other: ScalarField | complex) -> ScalarField:
        """Linear; join rule. Python scalars are constant fields."""
        if isinstance(other, ScalarField):
            return _linear_combine(self, other, "+",
                                   lambda x, y: x + y)
        if isinstance(other, _SCALAR_TYPES):
            return _scalar_shift(self, other, lambda d, s: d + s)
        return NotImplemented

    def __radd__(self, other: complex) -> ScalarField:
        """Scalar + field (fields handle field + field)."""
        if isinstance(other, _SCALAR_TYPES):
            return _scalar_shift(self, other, lambda d, s: s + d)
        return NotImplemented

    def __sub__(self, other: ScalarField | complex) -> ScalarField:
        """Linear; join rule."""
        if isinstance(other, ScalarField):
            return _linear_combine(self, other, "-",
                                   lambda x, y: x - y)
        if isinstance(other, _SCALAR_TYPES):
            return _scalar_shift(self, other, lambda d, s: d - s)
        return NotImplemented

    def __rsub__(self, other: complex) -> ScalarField:
        """Scalar - field."""
        if isinstance(other, _SCALAR_TYPES):
            return _scalar_shift(self, other, lambda d, s: s - d)
        return NotImplemented

    def __neg__(self) -> ScalarField:
        """Negation (a new quantity: default metadata)."""
        return _wrap(self._grid, self._function_space, -self.data)

    def __pos__(self) -> ScalarField:
        """Identity."""
        return self

    def __mul__(self, other: ScalarField | complex) -> ScalarField:
        """Scalar: linear scaling. Field: dispatched product (3.11)."""
        if isinstance(other, ScalarField):
            return _dispatched_product(self, other, "multiply", "*",
                                       lambda x, y: x * y)
        if isinstance(other, _SCALAR_TYPES):
            return _scalar_scale(self, other, lambda d, s: d * s)
        return NotImplemented

    def __rmul__(self, other: complex) -> ScalarField:
        """Scalar * field (linear scaling on any space)."""
        if isinstance(other, _SCALAR_TYPES):
            return _scalar_scale(self, other, lambda d, s: s * d)
        return NotImplemented

    def __truediv__(
        self, other: ScalarField | complex,
    ) -> ScalarField:
        """Scalar: linear scaling. Field: (kind="divide", space)."""
        if isinstance(other, ScalarField):
            return _dispatched_product(self, other, "divide", "/",
                                       lambda x, y: x / y)
        if isinstance(other, _SCALAR_TYPES):
            return _scalar_scale(self, other, lambda d, s: d / s)
        return NotImplemented

    def __rtruediv__(self, other: complex) -> ScalarField:
        """Scalar / field: a physical divide (nodal/average only)."""
        if not isinstance(other, _SCALAR_TYPES):
            return NotImplemented
        space = self._function_space
        if isinstance(other, complex):
            space = _promoted_space(space)
        _require_fallback_entry("divide", space,
                                _POINTWISE_FAMILIES)
        return _wrap(self._grid, space, other / self.data)

    def __pow__(self, exponent: float) -> ScalarField:
        """Physical power, (kind="power", space) (2.5 table)."""
        if not isinstance(exponent, int | float):
            return NotImplemented
        space = self._function_space
        _require_fallback_entry("power", space, _POINTWISE_FAMILIES)
        return _wrap(self._grid, space, self.data ** exponent)

    def __abs__(self) -> ScalarField:
        """Pointwise modulus, (kind="abs", space); nodal default."""
        space = self._function_space
        _require_fallback_entry("abs", space, _ABS_FAMILIES)
        return _wrap(self._grid, _real_space(space),
                     jnp.abs(self.data))

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
    def xr(self) -> object:
        """Export to xarray (label/gather rules: export cluster)."""
        raise NotImplementedError(
            "xarray export arrives with the Wave-4 export cluster")

    def __repr__(self) -> str:
        """Name, space, shape, dtype summary."""
        return (f"ScalarField({self.name!r}, "
                f"space={self._function_space!r}, "
                f"shape={self.shape}, dtype={self.dtype})")


# ================================================================
#  Shared arithmetic plumbing
# ================================================================
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
    """Apply ``fn`` per factor and rebuild (layout preserved)."""
    if isinstance(space, TensorProductSpace):
        new = TensorProductSpace.of(
            *(fn(factor) for factor in space.factors))
    else:
        new = fn(space.bare)
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
                raise KeyError(
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
    """Grid check, join, lift, elementwise combine (for +/-)."""
    _check_grids(a, b, operation)
    joined = join(a.function_space, b.function_space,
                  operation=operation)
    _check_lift(a.function_space, joined)
    _check_lift(b.function_space, joined)
    return _wrap(a.grid, joined, data_op(a.data, b.data))


def _scalar_shift(
    f: ScalarField,
    value: complex,
    data_op: Callable[[jax.Array, complex], jax.Array],
) -> ScalarField:
    """Python scalar in +/-: a constant field entering the join."""
    space = f.function_space
    for factor in space.factors:
        if isinstance(factor, CoefficientSpace):
            raise KeyError(
                "no ('broadcast', ConstantSpace -> "
                f"{factor!r}) dispatch entry: adding a Python "
                "scalar to a coefficient-space field is the exact "
                "zero-mode update, not implemented in iteration 1")
    if isinstance(value, complex):
        space = _promoted_space(space)
    return _wrap(f.grid, space, data_op(f.data, value))


def _scalar_scale(
    f: ScalarField,
    value: complex,
    data_op: Callable[[jax.Array, complex], jax.Array],
) -> ScalarField:
    """Python scalar in * and /: linear scaling on any space."""
    space = f.function_space
    if isinstance(value, complex):
        space = _promoted_space(space)
    return _wrap(f.grid, space, data_op(f.data, value))


# ================================================================
#  Iteration-1 dispatch seam (registry swap-in point)
# ================================================================
# ``grid.dispatch`` is the duck-typed ``OperatorRegistry`` (doc:
# operators_composed.md): when the grid carries one, products resolve
# ``registry.resolve(kind, joined.bare)`` and apply the returned
# binary operator to the lifted operands — resolution errors
# (``DispatchError``, a ``KeyError``) propagate untouched. Without a
# registry the fallback below mirrors the iteration-1 default table:
# elementwise on nodal/average/constant factors (the registered
# ``CollocationProduct``/second-order-shortcut default), no entry —
# hence a ``KeyError`` — on coefficient factors. The Wave-2 registry
# merge replaces the fallback by passing ``dispatch=`` to ``Grid``.

_POINTWISE_FAMILIES = (NodalSpace, AverageSpace, ConstantSpace)
_ABS_FAMILIES = (NodalSpace, ConstantSpace)


def _require_fallback_entry(
    kind: str,
    space: SpaceLike,
    allowed: tuple[type, ...],
) -> None:
    """Mimic the it-1 default table: raise where no row exists."""
    for factor in space.factors:
        if not isinstance(factor, allowed):
            raise KeyError(
                f"no ('{kind}', {factor!r}) dispatch entry: the "
                "iteration-1 defaults cover nodal/average factors "
                "only; coefficient-space products are explicit "
                "operators (e.g. Convolution)")


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
    data_op: Callable[[jax.Array, jax.Array], jax.Array],
) -> ScalarField:
    """Grid check, join, then registry dispatch (or the fallback)."""
    _check_grids(a, b, operation)
    joined = join(a.function_space, b.function_space,
                  operation=operation)
    _check_lift(a.function_space, joined)
    _check_lift(b.function_space, joined)
    registry = a.grid.dispatch
    if registry is not None:
        op = registry.resolve(kind, joined.bare)
        return op(_lift_field(a, joined), _lift_field(b, joined))
    _require_fallback_entry(kind, joined, _POINTWISE_FAMILIES)
    return _wrap(a.grid, joined, data_op(a.data, b.data))


def _target_space(
    space: SpaceLike,
    target: ScalarField | SpaceLike,
) -> SpaceLike:
    """Resolve a ``to`` target: field, product, or single factor."""
    if isinstance(target, ScalarField):
        return target.function_space
    if (not isinstance(target, TensorProductSpace)
            and isinstance(space, TensorProductSpace)):
        # single-factor shorthand: replace that factor, keep the rest
        return space.replace(
            **{target.names[0]: target.bare})
    return target
