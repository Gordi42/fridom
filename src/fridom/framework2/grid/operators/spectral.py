r"""
Spectral (coefficient-space) operators.

Description
-----------
Owning class doc: ``notes/framework2/classes/operators_stencils.md``
("Spectral (coefficient-space) operators"). ``SpectralDerivative`` is
the exact derivative on coefficient spaces — the only ``"diff"``
choice there; ``PhaseShift`` and ``SincShift`` are the exact
inter-origin conversions (rules section 3.2). All three are diagonal
(or index-shifted diagonal) multiplies whose values derive from the
factor's mesh at trace time; ``eigenvalues`` stays the raising base
method until the (designed-for) ``Symbol`` cluster lands.

Coefficient conventions match the transforms (index-based amplitude
convention of ``operators.fourier`` and the sine/cosine mode tables
of ``operators.trig``).
"""
# Wave 3: SpectralDerivative, PhaseShift, SincShift
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.fields.storage import store
from fridom.framework2.grid.operators.base import (
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
)
from fridom.framework2.grid.operators.interned import interned
from fridom.framework2.grid.operators.transform import (
    axis_concat,
    axis_slice,
    axis_vector,
    axis_zeros,
)
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.average import (
    AverageSpace,
    CellAvg,
)
from fridom.framework2.grid.spaces.coefficient import (
    ChebyshevSpace,
    CosineSpace,
    FourierSpace,
    SineSpace,
)
from fridom.framework2.grid.spaces.function_space import FunctionSpace
from fridom.framework2.grid.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    import jax

TWO_PI = 2.0 * jnp.pi

# first-node offset from x_min per nodal node set, in cell widths
_NODE_OFFSETS: dict[NodeSet, float] = {
    NodeSet.CENTER: 0.5,
    NodeSet.LEFT: 0.0,
    NodeSet.RIGHT: 1.0,
}


# ================================================================
#  Wavenumber helpers (grid.wavenumbers delegates here at merge)
# ================================================================
def fourier_wavenumbers(factor: FourierSpace) -> jax.Array:
    """
    Physical wavenumbers of a Fourier factor, in storage layout.

    Description
    -----------
    ``2 pi m / L`` for integer mode ``m``: the half-spectrum layout
    ``m = 0..n//2`` for real origins, the fft layout (positive then
    negative modes) for complex origins.

    Parameters
    ----------
    factor : FourierSpace
        The (bare) Fourier factor space.

    Returns
    -------
    jax.Array
        The wavenumbers along the factor's axis.
    """
    mesh = factor.mesh
    n = factor.origin.shape[0]
    length = mesh.extent[1] - mesh.extent[0]
    if factor.scalars is Scalars.REAL:
        modes = jnp.arange(n // 2 + 1, dtype=dtype_real())
    else:
        modes = jnp.fft.fftfreq(n, 1.0 / n).astype(dtype_real())
    return (TWO_PI / length) * modes


def trig_wavenumbers(
    factor: SineSpace | CosineSpace,
) -> jax.Array:
    """
    Physical wavenumbers of a sine/cosine factor, in mode order.

    Description
    -----------
    ``pi k / L`` for the mode tables of ``operators.trig``: sine
    modes ``k = 1..shape`` (DST-II of ``Center`` origins and DST-I
    of ``Inner`` origins both start at ``k = 1``), cosine modes
    ``k = 0..shape-1`` (DCT-II of ``Center`` and DCT-I of ``Outer``
    origins both start at ``k = 0``).

    Parameters
    ----------
    factor : SineSpace | CosineSpace
        The (bare) trigonometric coefficient factor.

    Returns
    -------
    jax.Array
        The wavenumbers along the factor's axis.
    """
    n = factor.shape[0]
    if isinstance(factor, SineSpace):
        modes = jnp.arange(1, n + 1, dtype=dtype_real())
    else:
        modes = jnp.arange(n, dtype=dtype_real())
    return (jnp.pi / _length(factor)) * modes


def chebyshev_modes(factor: ChebyshevSpace) -> jax.Array:
    """
    Intrinsic mode indices of a Chebyshev factor.

    Description
    -----------
    The Chebyshev basis is not wavenumber-indexed; per rules section
    3.10 ``grid.wavenumbers`` returns the space's own mode indices
    ``k = 0..n``.

    Parameters
    ----------
    factor : ChebyshevSpace
        The (bare) Chebyshev coefficient factor.

    Returns
    -------
    jax.Array
        The mode indices along the factor's axis.
    """
    return jnp.arange(factor.shape[0], dtype=dtype_real())


def _length(factor: FunctionSpace) -> float:
    """Physical interval length of the factor's mesh."""
    extent = factor.mesh.extent
    return extent[1] - extent[0]


def _zero_nyquist(values: jax.Array,
                  factor: FourierSpace) -> jax.Array:
    """Zero the (self-conjugate or sign-ambiguous) Nyquist slot."""
    n = factor.origin.shape[0]
    if n % 2:
        return values
    index = n // 2 if factor.scalars is Scalars.COMPLEX else -1
    return values.at[index].set(0)


def _diagonal_result(f: FieldLike, axis: str,
                     out_factor: FunctionSpace,
                     data: jax.Array,
                     metadata: object) -> FieldLike:
    """Build the result field of a per-factor coefficient op."""
    bare = f.function_space.bare
    if isinstance(bare, FunctionSpace):
        space = out_factor
    else:
        space = bare.replace(**{axis: out_factor})
    stored = store(f.grid.decomposition, space, data)
    return type(f)(f.grid, space, stored, metadata)


def _paired_origin(origin: FunctionSpace, node_set: NodeSet,
                   kind: BC) -> FunctionSpace:
    """Return the bc-flipped partner origin of a trig pair."""
    flipped = origin.mesh.nodal(node_set, bc=kind)
    if origin.scalars is Scalars.COMPLEX:
        flipped = flipped.as_complex()  # pragma: no cover
    return flipped


# ================================================================
#  SpectralDerivative
# ================================================================
@final
@interned
class SpectralDerivative(SeparableOperator):

    """
    Exact derivative in coefficient space (i k multiply/recurrence).

    Description
    -----------
    The three iteration-1 paths (rules sections 3.2/3.4): the Fourier
    diagonal ``i k`` multiply (origin preserved — spectral
    differentiation does not stagger; the sign-ambiguous Nyquist mode
    of even-length spectra is annihilated), the sine/cosine bc-flip
    with the explicit index maps, and the Chebyshev coefficient
    recurrence. The trig index maps (operators_stencils.md): on the
    II-type pair (``Center`` origins) ``d/dx: Sine -> Cosine`` maps
    sine mode k to cosine mode k, annihilates the top sine mode, and
    never populates cosine ``k = 0``; the reverse annihilates the
    constant. On the I-type pair (``Inner`` Dirichlet <-> ``Outer``
    Neumann) ``d/dx: Sine -> Cosine`` lands in cosine ``k = 1..n-1``
    (neither ``k = 0`` nor ``k = n`` is populated); the reverse
    annihilates the constant ``k = 0`` **and** the Nyquist cosine
    ``k = n``. ``eigenvalues`` inherits the raising base until
    ``Symbol`` lands (designed-for).
    """

    dispatch_kind: ClassVar[str | None] = "diff"

    def _intern_key(self) -> tuple:
        """Structural key: no constructor state (D6)."""
        return ()

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        Fourier -> Fourier (same origin); Sine <-> Cosine; Cheb -> Cheb.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D coefficient factor.

        Returns
        -------
        FunctionSpace
            The codomain factor (bc-flipped for the trig pair).
        """
        if isinstance(domain, FourierSpace | ChebyshevSpace):
            return domain
        if isinstance(domain, SineSpace | CosineSpace):
            origin = domain.origin
            node_set = (origin.node_set
                        if isinstance(origin, NodalSpace) else None)
            if isinstance(domain, SineSpace):
                if node_set is NodeSet.CENTER:  # II-type pair
                    return domain.mesh.cosine(_paired_origin(
                        origin, NodeSet.CENTER, BC.NEUMANN))
                if node_set is NodeSet.INNER:  # I-type pair
                    return domain.mesh.cosine(_paired_origin(
                        origin, NodeSet.OUTER, BC.NEUMANN))
            else:
                if node_set is NodeSet.CENTER:  # II-type pair
                    return domain.mesh.sine(_paired_origin(
                        origin, NodeSet.CENTER, BC.DIRICHLET))
                if node_set is NodeSet.OUTER:  # I-type pair
                    return domain.mesh.sine(_paired_origin(
                        origin, NodeSet.INNER, BC.DIRICHLET))
            raise SpaceMismatchError(
                f"no diff signature on {domain!r}: the sine/cosine "
                "derivative covers the II-type (Center origin) and "
                "I-type (Inner Dirichlet <-> Outer Neumann) pairs",
                left=domain, operation="diff")
        raise SpaceMismatchError(
            f"no diff signature on {domain!r}: SpectralDerivative "
            "covers coefficient spaces (nodal spaces dispatch to "
            "FiniteDifference)", left=domain, operation="diff")

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — uniform declaration
    ) -> OperatorRequirements:
        """
        Declare halo = 0, layout "local" (whole-axis mode access).

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        return OperatorRequirements(halo=0, layout="local")

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Differentiate the coefficient factor along ``axis``.

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The derivative field (default metadata: new quantity).
        """
        bare = f.function_space.bare
        factor = bare.factor(axis)
        out_factor = self.codomain(factor)
        index = bare.names.index(axis)
        data = jnp.asarray(f.data)
        if isinstance(factor, FourierSpace):
            k = _zero_nyquist(fourier_wavenumbers(factor), factor)
            out = data * axis_vector(1j * k, data.ndim, index)
        elif isinstance(factor, SineSpace):
            if factor.origin.node_set is NodeSet.CENTER:
                out = _sine_to_cosine(data, index, _length(factor))
            else:
                out = _sine1_to_cosine1(data, index, _length(factor))
        elif isinstance(factor, CosineSpace):
            if factor.origin.node_set is NodeSet.CENTER:
                out = _cosine_to_sine(data, index, _length(factor))
            else:
                out = _cosine1_to_sine1(data, index, _length(factor))
        else:
            out = _chebyshev_derivative(data, index,
                                        _length(factor))
        return _diagonal_result(f, axis, out_factor, out, None)


def _sine_to_cosine(b: jax.Array, axis: int,
                    length: float) -> jax.Array:
    """d/dx of DST-II coefficients: sine mode k -> cosine mode k."""
    n = b.shape[axis]
    k = jnp.arange(1, n, dtype=dtype_real())
    scaled = (axis_slice(b, axis, 0, n - 1)
              * axis_vector(jnp.pi * k / length, b.ndim, axis))
    return axis_concat((axis_zeros(scaled, axis, 1), scaled), axis)


def _cosine_to_sine(a: jax.Array, axis: int,
                    length: float) -> jax.Array:
    """d/dx of DCT-II coefficients: cosine mode k -> sine mode k."""
    n = a.shape[axis]
    k = jnp.arange(1, n, dtype=dtype_real())
    scaled = (axis_slice(a, axis, 1, n)
              * axis_vector(-jnp.pi * k / length, a.ndim, axis))
    return axis_concat((scaled, axis_zeros(scaled, axis, 1)), axis)


def _sine1_to_cosine1(b: jax.Array, axis: int,
                      length: float) -> jax.Array:
    """d/dx of DST-I coefficients: sine k -> cosine k, k = 1..n-1.

    Index map: sine index ``j`` holds mode ``k = j + 1``, cosine
    index ``j`` holds ``k = j``; neither cosine ``k = 0`` nor the
    Nyquist ``k = n`` is populated.
    """
    p = b.shape[axis]  # n - 1 modes
    k = jnp.arange(1, p + 1, dtype=dtype_real())
    scaled = b * axis_vector(jnp.pi * k / length, b.ndim, axis)
    zero = axis_zeros(scaled, axis, 1)
    return axis_concat((zero, scaled, zero), axis)


def _cosine1_to_sine1(a: jax.Array, axis: int,
                      length: float) -> jax.Array:
    """d/dx of DCT-I coefficients: cosine k -> sine k, k = 1..n-1.

    Annihilates the constant ``k = 0`` and the Nyquist cosine
    ``k = n`` (its sine image vanishes at the interior faces).
    """
    m = a.shape[axis]  # n + 1 modes
    k = jnp.arange(1, m - 1, dtype=dtype_real())
    return (axis_slice(a, axis, 1, m - 1)
            * axis_vector(-jnp.pi * k / length, a.ndim, axis))


def _chebyshev_derivative(a: jax.Array, axis: int,
                          length: float) -> jax.Array:
    r"""
    Chebyshev coefficient recurrence for d/dx.

    Description
    -----------
    ``b_k = (2 / cbar_k) \sum_{p > k, p + k odd} p a_p`` on [-1, 1],
    times the affine chain factor ``2 / L``. Vectorized through
    parity-split reversed cumulative sums (no mode loop).
    """
    n = a.shape[axis] - 1
    k = jnp.arange(n + 1, dtype=dtype_real())
    t = a * axis_vector(k, a.ndim, axis)
    even = axis_vector(k % 2 == 0, a.ndim, axis)
    t_even = jnp.where(even, t, 0)
    t_odd = jnp.where(even, 0, t)

    def rev_cumsum_next(arr: jax.Array) -> jax.Array:
        """``S[k] = sum_{p >= k + 1} arr[p]`` along ``axis``."""
        tail = jnp.flip(
            jnp.cumsum(jnp.flip(arr, axis), axis=axis), axis)
        return axis_concat(
            (axis_slice(tail, axis, 1, n + 1),
             axis_zeros(tail, axis, 1)), axis)

    sums = jnp.where(even, rev_cumsum_next(t_odd),
                     rev_cumsum_next(t_even))
    cbar = jnp.where(k == 0, 2.0, 1.0)
    weights = axis_vector(2.0 / cbar, a.ndim, axis)
    return (2.0 / length) * weights * sums


# ================================================================
#  PhaseShift
# ================================================================
@final
@interned
class PhaseShift(SeparableOperator):

    """
    Inter-origin ``e^{i k s dx}`` shift between Fourier spaces.

    Description
    -----------
    The exact ``"interpolate"`` entry in coefficient space (rules
    section 3.2): retags ``Fourier(origin=A)`` to
    ``Fourier(origin=<to>)`` on the same mesh, multiplying by the
    inter-origin phase. On the Hermitian half spectrum of a real
    origin with even n, a half-cell shift is not an exact spectrum
    automorphism: the Nyquist mode is **zeroed** (the one non-exact
    DOF, class-doc caveat). Identity when the origin already matches.

    Parameters
    ----------
    to : NodeSet, optional
        Target origin node set (default: ``NodeSet.CENTER``).
    """

    dispatch_kind: ClassVar[str | None] = "interpolate"

    def __init__(self, to: NodeSet = NodeSet.CENTER) -> None:
        """Shift to the Fourier space of the given origin node set."""
        if to not in _NODE_OFFSETS:
            raise ValueError(
                "PhaseShift targets the periodic nodal node sets "
                f"(CENTER/LEFT/RIGHT), got {to!r}")
        self._to: NodeSet = to

    def _intern_key(self) -> tuple:
        """Structural key: the target origin node set (D6)."""
        return (self._to,)

    @property
    def to(self) -> NodeSet:
        """Target origin node set."""
        return self._to

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        ``Fourier(origin=A) -> Fourier(origin=<to>)``, same mesh.

        Parameters
        ----------
        domain : FunctionSpace
            The bare Fourier factor with a nodal origin.

        Returns
        -------
        FunctionSpace
            The retagged Fourier factor.
        """
        origin = _nodal_fourier_origin(domain, "PhaseShift")
        target = origin.mesh.nodal(self._to)
        if origin.scalars is Scalars.COMPLEX:
            target = target.as_complex()
        return domain.mesh.fourier(origin=target)

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Multiply by the inter-origin phase along ``axis``.

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The retagged field (metadata preserved: same quantity).
        """
        bare = f.function_space.bare
        factor = bare.factor(axis)
        out_factor = self.codomain(factor)
        if out_factor is factor:
            return f
        delta = (_NODE_OFFSETS[self._to]
                 - _NODE_OFFSETS[factor.origin.node_set])
        mult = _origin_shift(factor, delta)
        index = bare.names.index(axis)
        data = (jnp.asarray(f.data)
                * axis_vector(mult, f.data.ndim, index))
        return _diagonal_result(f, axis, out_factor, data,
                                f.metadata)


# ================================================================
#  SincShift
# ================================================================
@final
@interned
class SincShift(SeparableOperator):

    """
    ``sinc(k dx / 2)`` conversion from average to nodal origins.

    Description
    -----------
    Cell-averaging is convolution with a top-hat (rules sections
    3.2/3.9): an average-origin spectrum differs from the nodal one
    by ``sinc(k dx / 2)``, so average -> nodal **divides** (invertible
    on the resolved band: the first sinc zero lies beyond Nyquist).
    Where the origins are offset by half a cell the diagonal composes
    the sinc with the corresponding phase (zeroing the even-n real
    Nyquist mode like ``PhaseShift``). The nodal -> average direction
    is spelled by an explicit transform pair in iteration 1
    (``to`` targets nodal node sets only — ``NodeSet`` has no average
    members, doc deviation reported).

    Parameters
    ----------
    to : NodeSet, optional
        Target nodal origin node set (default: ``NodeSet.CENTER``).
    """

    dispatch_kind: ClassVar[str | None] = "interpolate"

    def __init__(self, to: NodeSet = NodeSet.CENTER) -> None:
        """Convert to the Fourier space of the given node set."""
        if to not in _NODE_OFFSETS:
            raise ValueError(
                "SincShift targets the periodic nodal node sets "
                f"(CENTER/LEFT/RIGHT), got {to!r}")
        self._to: NodeSet = to

    def _intern_key(self) -> tuple:
        """Structural key: the target origin node set (D6)."""
        return (self._to,)

    @property
    def to(self) -> NodeSet:
        """Target origin node set."""
        return self._to

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        ``Fourier(origin=cell/face avg) -> Fourier(origin=nodal)``.

        Parameters
        ----------
        domain : FunctionSpace
            The bare Fourier factor with an average origin.

        Returns
        -------
        FunctionSpace
            The retagged Fourier factor.
        """
        origin = domain.origin if isinstance(
            domain, FourierSpace) else None
        if not isinstance(origin, AverageSpace):
            raise SpaceMismatchError(
                f"no SincShift signature on {domain!r}: the domain "
                "is a Fourier space of an average origin "
                "(nodal origins dispatch to PhaseShift)",
                left=domain, operation="interpolate")
        target = origin.mesh.nodal(self._to)
        if origin.scalars is Scalars.COMPLEX:
            target = target.as_complex()  # pragma: no cover
        return domain.mesh.fourier(origin=target)

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Divide by sinc (and phase-shift) along ``axis``.

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The retagged field (metadata preserved: same quantity).
        """
        bare = f.function_space.bare
        factor = bare.factor(axis)
        out_factor = self.codomain(factor)
        origin = factor.origin
        offset = 0.5 if isinstance(origin, CellAvg) else 1.0
        delta = _NODE_OFFSETS[self._to] - offset
        k = fourier_wavenumbers(factor)
        dx = _length(factor) / origin.shape[0]
        sinc = jnp.sinc(k * dx / TWO_PI)
        mult = _origin_shift(factor, delta) / sinc
        index = bare.names.index(axis)
        data = (jnp.asarray(f.data)
                * axis_vector(mult, f.data.ndim, index))
        return _diagonal_result(f, axis, out_factor, data,
                                f.metadata)


# ================================================================
#  Shared diagonal machinery
# ================================================================
def _nodal_fourier_origin(domain: FunctionSpace,
                          who: str) -> FunctionSpace:
    """Validate a Fourier factor with a staggered nodal origin."""
    if isinstance(domain, FourierSpace):
        origin = domain.origin
        if (isinstance(origin, NodalSpace)
                and origin.node_set in _NODE_OFFSETS):
            return origin
        if isinstance(origin, AverageSpace):
            raise SpaceMismatchError(
                f"no {who} signature on {domain!r}: average "
                "origins dispatch to SincShift",
                left=domain, operation="interpolate")
    raise SpaceMismatchError(
        f"no {who} signature on {domain!r}: the domain is a "
        "Fourier space of a periodic nodal origin",
        left=domain, operation="interpolate")


def _origin_shift(factor: FourierSpace,
                  delta: float) -> jax.Array:
    """
    Build the inter-origin phase diagonal ``e^{i k delta dx}``.

    Description
    -----------
    ``delta`` is the target-minus-source first-node offset in cell
    widths. For real origins with even n and a half-integer
    ``delta`` the shifted Nyquist coefficient has no valid rfft
    layout: the mode is zeroed (rules section 3.2 caveat).
    """
    n = factor.origin.shape[0]
    dx = _length(factor) / n
    k = fourier_wavenumbers(factor)
    phase = jnp.exp(1j * k * (delta * dx))
    if delta == int(delta):
        return phase
    if factor.scalars is Scalars.REAL and n % 2 == 0:
        phase = phase.at[-1].set(0)
    return phase
