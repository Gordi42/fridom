r"""
Spectral (coefficient-space) operators.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_stencils.md``
("Spectral (coefficient-space) operators"). ``SpectralDerivative`` is
the exact derivative on coefficient spaces — the only ``"diff"``
choice there; ``PhaseShift`` and ``SincShift`` are the exact
inter-origin conversions (rules section 3.2). All three are diagonal
(or index-shifted diagonal) multiplies whose values derive from the
factor's mesh at trace time; ``eigenvalues`` returns the matching
``Symbol`` on Fourier factors (Wave 9A) and, since the derived-shift
``Symbol`` alignment landed, on the sine/cosine families too (the
diagonal lives in the codomain slot layout). The Chebyshev
recurrence couples all modes and keeps raising ``EigenbasisError``.

Coefficient conventions match the transforms (index-based amplitude
convention of ``operators.fourier`` and the sine/cosine mode tables
of ``operators.trig``).
"""
# Wave 3: SpectralDerivative, PhaseShift, SincShift
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.spatial.bc import BC
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.storage import store
from fridom.spatial.operators.base import (
    EigenbasisError,
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
    _resolve_axis,
)
from fridom.spatial.operators.interned import interned
from fridom.spatial.operators.symbol import diagonal_symbol
from fridom.spatial.operators.transform import (
    axis_concat,
    axis_slice,
    axis_vector,
    axis_zeros,
)
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import (
    AverageSpace,
    CellAvg,
    FaceAvg,
)
from fridom.spatial.spaces.coefficient import (
    ChebyshevSpace,
    CosineSpace,
    FourierSpace,
    SineSpace,
)
from fridom.spatial.spaces.function_space import FunctionSpace
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.spatial.operators.symbol import Symbol
    from fridom.spatial.spaces.tensor_product import SpaceLike

TWO_PI = 2.0 * jnp.pi

# first-node offset from x_min per nodal node set, in cell widths
_NODE_OFFSETS: dict[NodeSet, float] = {
    NodeSet.CENTER: 0.5,
    NodeSet.LEFT: 0.0,
    NodeSet.RIGHT: 1.0,
}


def _first_node_offset(origin: FunctionSpace) -> float:
    """
    First-node offset of a nodal or average origin, in cell widths.

    Description
    -----------
    The average family enters the periodic staggering-symbol calculus
    at its quadrature point: ``CellAvg`` at the primal-cell midpoints
    (0.5, like ``Center``), ``FaceAvg`` at the dual-cell midpoints —
    the faces (1.0, like ``Right``). Nodal origins read the
    ``Center`` / ``Left`` / ``Right`` offset table. This is the
    symbol-layer twin of ``operators.reconstruct.fv_node_offset``
    (kept local so ``operators.spectral`` stays reconstruct-free).

    Parameters
    ----------
    origin : FunctionSpace
        The (bare) periodic nodal or average origin.

    Returns
    -------
    float
        Distance of the first true DOF from ``x_min`` in cell widths.
    """
    if isinstance(origin, CellAvg):
        return 0.5
    if isinstance(origin, FaceAvg):
        return 1.0
    return _NODE_OFFSETS[origin.node_set]


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
    ``k = n``. ``eigenvalues`` returns the ``i k`` diagonal on
    Fourier factors and the (real, derived-shift) ``±pi k / L``
    diagonal on sine/cosine factors; Chebyshev raises
    ``EigenbasisError``.
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

    def eigenvalues(
        self,
        grid: object,  # noqa: ARG002 — the factor carries the mesh
        space: SpaceLike,
    ) -> Symbol:
        r"""
        Return the exact ``i k`` (Fourier) / ``±pi k / L`` diagonal.

        Description
        -----------
        The spectral derivative is a diagonal multiply on a Fourier
        factor (the Nyquist mode of even-length spectra annihilated,
        matching ``_apply_factor``) and a **derived-shift** diagonal
        on the sine/cosine families: the real ``±pi k / L`` values,
        stored in the codomain slot layout per the ``Symbol``
        convention (``+`` on sine -> cosine, ``-`` on cosine -> sine,
        matching the four apply kernels bitwise; modes the codomain
        lacks are annihilated by the embedding, modes the domain
        lacks multiply structural zero-fills — the cosine ``k = 0``
        entry is an exact zero so inverses/round-trips stay clean).
        The Chebyshev recurrence couples all modes and raises
        ``EigenbasisError`` (deferred to the block layer).

        Parameters
        ----------
        grid : object
            The grid (unused: the Fourier factor carries the mesh).
        space : SpaceLike
            The coefficient factor (or product) space.

        Returns
        -------
        Symbol
            The diagonal on the coefficient factor.
        """
        bare = space.bare
        axis = _resolve_axis(self, bare)
        factor = bare.factor(axis)
        if isinstance(factor, SineSpace | CosineSpace):
            out_factor = self.codomain(factor)
            k = trig_wavenumbers(out_factor)
            leaf = k if isinstance(factor, SineSpace) else -k
            return diagonal_symbol(bare, axis, factor, out_factor,
                                   leaf)
        if not isinstance(factor, FourierSpace):
            raise EigenbasisError(
                "SpectralDerivative has a diagonal symbol only on "
                f"Fourier and sine/cosine factors, got {factor!r}: "
                "the Chebyshev recurrence couples all modes")
        k = _zero_nyquist(fourier_wavenumbers(factor), factor)
        return diagonal_symbol(bare, axis, factor, factor, 1j * k)

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

    def eigenvalues(
        self,
        grid: object,  # noqa: ARG002 — the factor carries the mesh
        space: SpaceLike,
    ) -> Symbol:
        r"""
        Return the inter-origin phase diagonal ``e^{i k delta dx}``.

        Description
        -----------
        Retags ``Fourier(A) -> Fourier(<to>)`` on the same mesh; the
        even-n real Nyquist mode is zeroed for a half-cell shift (the
        one non-exact DOF, matching ``_apply_factor``). Identity when
        the origin already matches.

        Parameters
        ----------
        grid : object
            The grid (unused: the Fourier factor carries the mesh).
        space : SpaceLike
            The coefficient factor (or product) space.

        Returns
        -------
        Symbol
            The phase diagonal on the Fourier factor.
        """
        bare = space.bare
        axis = _resolve_axis(self, bare)
        factor = bare.factor(axis)
        out_factor = self.codomain(factor)
        delta = (_NODE_OFFSETS[self._to]
                 - _NODE_OFFSETS[factor.origin.node_set])
        return diagonal_symbol(bare, axis, factor, out_factor,
                               _origin_shift(factor, delta))

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

    def eigenvalues(
        self,
        grid: object,  # noqa: ARG002 — the factor carries the mesh
        space: SpaceLike,
    ) -> Symbol:
        r"""
        Return the ``sinc(k dx/2)`` average-to-nodal diagonal.

        Description
        -----------
        Average -> nodal **divides** by the cell-averaging sinc factor
        (invertible on the resolved band), composing the corresponding
        inter-origin phase — the exact ``_apply_factor`` diagonal as a
        retagging ``Fourier(avg) -> Fourier(nodal)`` symbol.

        Parameters
        ----------
        grid : object
            The grid (unused: the Fourier factor carries the mesh).
        space : SpaceLike
            The coefficient factor (or product) space.

        Returns
        -------
        Symbol
            The sinc/phase diagonal on the Fourier factor.
        """
        bare = space.bare
        axis = _resolve_axis(self, bare)
        factor = bare.factor(axis)
        out_factor = self.codomain(factor)
        origin = factor.origin
        offset = 0.5 if isinstance(origin, CellAvg) else 1.0
        delta = _NODE_OFFSETS[self._to] - offset
        k = fourier_wavenumbers(factor)
        dx = _length(factor) / origin.shape[0]
        sinc = jnp.sinc(k * dx / TWO_PI)
        leaf = _origin_shift(factor, delta) / sinc
        return diagonal_symbol(bare, axis, factor, out_factor, leaf)

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


# ================================================================
#  Staggering (nodal -> Fourier) symbols
# ================================================================
def periodic_fourier_factor(
    nodal_factor: FunctionSpace, who: str,
) -> FourierSpace:
    """
    Return the Fourier space diagonalizing a periodic nodal factor.

    Description
    -----------
    A staggering stencil (``FiniteDifference``/``LinearInterp``)
    diagonalizes only on a periodic mesh; on a bounded mesh it
    diagonalizes in the sine/cosine basis instead, so this raises
    ``EigenbasisError`` there (the correct boundary of the symbol
    capability, rules section 3.7).

    Parameters
    ----------
    nodal_factor : FunctionSpace
        The (bare) nodal coefficient factor.
    who : str
        The querying operator name, for the error message.

    Returns
    -------
    FourierSpace
        ``mesh.fourier(origin=nodal_factor)``.
    """
    if not (isinstance(nodal_factor, NodalSpace)
            and nodal_factor.mesh.periodic
            and nodal_factor.bc.is_free
            and nodal_factor.node_set in _NODE_OFFSETS):
        raise EigenbasisError(
            f"{who} has a Fourier symbol only on periodic BC-free "
            f"nodal factors (Center/Right/Left), got {nodal_factor!r}: "
            "bounded staggering diagonalizes in the sine/cosine basis")
    return nodal_factor.mesh.fourier(origin=nodal_factor)


def fourier_partner(
    factor: FunctionSpace, who: str,
) -> tuple[FourierSpace, FunctionSpace]:
    r"""
    Resolve the source Fourier factor and its periodic nodal origin.

    Description
    -----------
    Layout-faithful staggering (symbol_stack_design.md decision 3): a
    staggering stencil (``FiniteDifference``/``LinearInterp``) carries
    a Fourier symbol relative to whatever coefficient layout the query
    threads. The eigenvalue query may thread a **nodal** factor (the
    physical operand space — the eigenmode path) or a **Fourier**
    factor (a transformed coefficient space — the spectral solve, where
    one axis of an ``rfftn`` layout is a half spectrum and the rest are
    full). Either way this returns the source Fourier factor — whose
    scalars fix the per-axis mode count that ``fourier_wavenumbers``
    reads — and the periodic nodal origin (carrying the measure).

    Parameters
    ----------
    factor : FunctionSpace
        The threaded coefficient factor (nodal or Fourier).
    who : str
        The querying operator name, for the error message.

    Returns
    -------
    tuple[FourierSpace, FunctionSpace]
        ``(src_fourier, nodal_origin)``.
    """
    if isinstance(factor, FourierSpace):
        origin = factor.origin
        periodic_fourier_factor(origin, who)  # validate the origin
        return factor, origin
    return periodic_fourier_factor(factor, who), factor


def fv_fourier_partner(
    factor: FunctionSpace, who: str,
) -> tuple[FourierSpace, FunctionSpace]:
    r"""
    Resolve the source Fourier factor and its periodic FV origin.

    Description
    -----------
    The average-family sibling of :func:`fourier_partner`: the FV
    flux/face/reconstruct stencils diagonalize only on a **periodic,
    uniform** mesh, in the Fourier basis of their nodal *or average*
    origin. The eigenvalue query may thread the ``Fourier(origin)``
    coefficient factor itself (the spectral solve) or the bare
    periodic origin (the eigenmode path); either way this returns the
    source Fourier factor — whose scalars fix the per-axis mode count
    ``fourier_wavenumbers`` reads — and the origin (nodal or average,
    carrying the measure). Bounded (walled) meshes, mapped/stretched
    meshes (whose non-constant metric breaks translation invariance,
    so no diagonal symbol exists), and non-Fourier coefficient factors
    (sine/cosine/Chebyshev) raise ``EigenbasisError``: FV symbols are
    periodic-and-uniform-only in this iteration (the walled FV
    diagonalizing basis is deferred, scoping study G6 / stage F4).

    Parameters
    ----------
    factor : FunctionSpace
        The threaded coefficient factor (Fourier) or bare periodic
        nodal/average origin.
    who : str
        The querying operator name, for the error message.

    Returns
    -------
    tuple[FourierSpace, FunctionSpace]
        ``(src_fourier, origin)``.
    """
    if isinstance(factor, FourierSpace):
        origin = factor.origin
        src: FourierSpace | None = factor
    elif isinstance(factor, NodalSpace | AverageSpace):
        origin = factor
        src = None
    else:
        raise EigenbasisError(
            f"{who} has a Fourier symbol only on periodic Fourier "
            f"factors and their nodal/average origins, got {factor!r}: "
            "the FV stencils have no diagonalizing basis on "
            "sine/cosine/Chebyshev factors")
    mesh = origin.mesh
    if (not getattr(mesh, "periodic", False)
            or getattr(mesh, "coordinate_map", None) is not None):
        raise EigenbasisError(
            f"{who} has a Fourier symbol only on periodic, uniform "
            f"meshes, got {origin!r}: bounded (walled) and "
            "mapped/stretched average families carry no diagonalizing "
            "basis in iteration 1 (scoping study G6, stage F4/F5)")
    if src is None:
        src = mesh.fourier(origin=origin)
    return src, origin


# ================================================================
#  Staggering (bounded trig) partners and codomain pairing
# ================================================================
# the (basis <- origin node set, BC) rows of the bounded trig
# transforms (mirror of ``grid._TRIG_ORIGIN_CANDIDATES`` and the
# origin tables in ``operators.trig``)
_TRIG_BASIS_ROWS: tuple[tuple[BC, tuple[NodeSet, ...], str], ...] = (
    (BC.DIRICHLET, (NodeSet.CENTER, NodeSet.INNER), "sine"),
    (BC.NEUMANN, (NodeSet.CENTER, NodeSet.OUTER), "cosine"),
)

# constitutive coefficient-side codomains of the order-2 staggering
# stencils on the trig families. Two-representations rule: the
# stencils' NODAL codomains are the BC-free siblings (the input's
# tag governs only the ghost fill), but their COEFFICIENT codomains
# carry BC-tagged origins — the tag is constitutive of the basis.
# ``diff`` flips the family and the BC kind, staggering the node
# set; ``interpolate`` keeps both, staggering the node set (the
# missing DCT-II row would land on "cosine at Inner", which is not
# a grounded family — ``trig_interp_codomain`` raises there).
_TRIG_DIFF_PAIRING: dict[
    tuple[type, NodeSet], tuple[str, NodeSet, BC]] = {
    (SineSpace, NodeSet.CENTER):
        ("cosine", NodeSet.OUTER, BC.NEUMANN),
    (SineSpace, NodeSet.INNER):
        ("cosine", NodeSet.CENTER, BC.NEUMANN),
    (CosineSpace, NodeSet.CENTER):
        ("sine", NodeSet.INNER, BC.DIRICHLET),
    (CosineSpace, NodeSet.OUTER):
        ("sine", NodeSet.CENTER, BC.DIRICHLET),
}
_TRIG_INTERP_PAIRING: dict[
    tuple[type, NodeSet], tuple[str, NodeSet, BC]] = {
    (SineSpace, NodeSet.CENTER):
        ("sine", NodeSet.INNER, BC.DIRICHLET),
    (SineSpace, NodeSet.INNER):
        ("sine", NodeSet.CENTER, BC.DIRICHLET),
    (CosineSpace, NodeSet.OUTER):
        ("cosine", NodeSet.CENTER, BC.NEUMANN),
}


def in_trig_family(factor: FunctionSpace) -> bool:
    """
    Whether a factor diagonalizes in the sine/cosine basis.

    Description
    -----------
    The routing predicate of the staggering ``eigenvalues`` methods
    (``FiniteDifference``/``LinearInterp``): ``True`` on sine/cosine
    coefficient factors and on BC-tagged bounded nodal factors —
    :func:`trig_partner` then validates the exact (node set, BC)
    row. Periodic and BC-free factors return ``False`` and keep the
    Fourier path bitwise untouched.

    Parameters
    ----------
    factor : FunctionSpace
        The threaded coefficient or nodal factor.

    Returns
    -------
    bool
        Whether the trig staggering-symbol path applies.
    """
    if isinstance(factor, SineSpace | CosineSpace):
        return True
    return (isinstance(factor, NodalSpace)
            and not getattr(factor.mesh, "periodic", False)
            and not factor.bc.is_free)


def trig_partner(
    factor: FunctionSpace, who: str,
) -> tuple[SineSpace | CosineSpace, FunctionSpace]:
    r"""
    Resolve the sine/cosine factor and its BC-tagged nodal origin.

    Description
    -----------
    The bounded sibling of :func:`fourier_partner`: on a walled mesh
    a staggering stencil diagonalizes in the sine/cosine basis, and
    the eigenvalue query may thread the **trig coefficient** factor
    itself (a transformed coefficient space — the spectral solve) or
    the **BC-tagged bounded nodal** origin (the physical operand
    space — the eigenmode path). A nodal factor resolves through the
    same (basis <- origin node set, BC) table the trig transforms
    use: Dirichlet ``Center``/``Inner`` -> DST-II/DST-I, Neumann
    ``Center``/``Outer`` -> DCT-II/DCT-I. Anything else — BC-free or
    mixed-tag nodal factors, Chebyshev factors — raises
    ``EigenbasisError`` (no closed staggering diagonal).

    Parameters
    ----------
    factor : FunctionSpace
        The threaded coefficient factor (trig or tagged nodal).
    who : str
        The querying operator name, for the error message.

    Returns
    -------
    tuple[SineSpace | CosineSpace, FunctionSpace]
        ``(trig_factor, nodal_origin)``.
    """
    if isinstance(factor, SineSpace | CosineSpace):
        return factor, factor.origin
    if (isinstance(factor, NodalSpace)
            and not getattr(factor.mesh, "periodic", False)):
        components = factor.bc.components
        for kind, node_sets, family in _TRIG_BASIS_ROWS:
            if (factor.node_set in node_sets
                    and all(c is kind for c in components)):
                return getattr(factor.mesh, family)(factor), factor
    raise EigenbasisError(
        f"{who} has a sine/cosine symbol only on Sine/Cosine "
        "coefficient factors and the BC-tagged bounded trig origins "
        "(Dirichlet Center/Inner, Neumann Center/Outer), got "
        f"{factor!r}: Chebyshev and mixed-tag factors have no "
        "staggering diagonal in iteration 1")


def trig_diff_codomain(
    domain: SineSpace | CosineSpace,
) -> SineSpace | CosineSpace:
    r"""
    Coefficient-side ``diff`` codomain of a trig factor.

    Description
    -----------
    The constitutive pairing of the order-2 staggered derivative
    (module table ``_TRIG_DIFF_PAIRING``): the family **and** the BC
    kind flip while the origin node set staggers —
    ``Sine-II(Center, DIR) <-> Cosine-I(Outer, NEU)`` and
    ``Sine-I(Inner, DIR) <-> Cosine-II(Center, NEU)``.

    Parameters
    ----------
    domain : SineSpace | CosineSpace
        The bare trig coefficient factor.

    Returns
    -------
    SineSpace | CosineSpace
        The flipped-family codomain factor (scalars preserved).
    """
    return _trig_pairing(domain, _TRIG_DIFF_PAIRING, "diff")


def fv_trig_diff_codomain(
    domain: SineSpace | CosineSpace,
) -> SineSpace | CosineSpace:
    r"""
    Coefficient-side ``diff`` codomain of an FV flux/face trig factor.

    Description
    -----------
    The FV-D2 option-A inter-family staggering partner of the walled FV
    C-grid (stage F4): the cell-average pressure gradient
    (``FaceDifference``) pairs the Neumann ``CellAvg`` cosine (DCT-II)
    with the nodal-face Dirichlet ``Inner`` sine (DST-I), and the
    face-flux divergence (``FluxDifference``) pairs the reverse. Unlike
    the nodal :func:`trig_diff_codomain` (whose origins stay in the
    nodal family), the origin **crosses families** — ``CellAvg <->
    Inner`` — because the FV pressure lives on cell averages while the
    velocity lives on the point-value faces (FV-D2). The BC kind and
    the basis flip with it (Neumann cosine <-> Dirichlet sine), and at
    second order the FV stencil is bitwise the nodal ``Center <->
    Inner`` one, so the diagonal is the same ``+-2 sin(k dz/2)/dz``
    derived shift (no ``sinc``, the correction-1 pattern).

    Parameters
    ----------
    domain : SineSpace | CosineSpace
        The bare trig coefficient factor (average- or nodal-face
        origin).

    Returns
    -------
    SineSpace | CosineSpace
        The flipped-family, family-crossing codomain factor.
    """
    mesh = domain.mesh
    origin = domain.origin
    if isinstance(domain, CosineSpace) and isinstance(origin, CellAvg):
        target: SineSpace | CosineSpace = mesh.sine(
            mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET))
    elif (isinstance(domain, SineSpace)
          and isinstance(origin, NodalSpace)
          and origin.node_set is NodeSet.INNER):
        target = mesh.cosine(mesh.average(CellAvg, bc=BC.NEUMANN))
    else:
        raise EigenbasisError(
            f"no FV staggering-diff (sine/cosine) pairing on {domain!r}: "
            "the walled FV C-grid pairs the Neumann CellAvg cosine "
            "(DCT-II, the pressure gradient) with the Dirichlet Inner "
            "sine (DST-I, the flux divergence) only; other trig factors "
            "carry no FV staggering diagonal")
    if origin.scalars is Scalars.COMPLEX:  # pragma: no cover
        target = target.as_complex()
    return target


def fv_trig_interp_codomain(
    domain: SineSpace | CosineSpace,
) -> SineSpace | CosineSpace:
    r"""
    Coefficient-side ``interpolate`` codomain of an FV reconstruct.

    Description
    -----------
    The average-family sibling of :func:`trig_interp_codomain` for the
    walled FV C-grid staggering reconstruction (``LinearReconstruction``,
    stage F5): the two-point mean **keeps** the trig family (sine stays
    sine) and staggers the *origin* between the primal cell average and
    the interior face — ``CellAvg <-> Inner`` — exactly as the nodal
    interp staggers ``Center <-> Inner``. The two grounded rows are the
    Dirichlet sine pair of the walled-vertical eigenmode kit: the
    face -> cell reconstruction ``Sine-I(Inner, DIR) -> Sine-II(CellAvg,
    DIR)`` (the ``ab[z]`` symbol, ``w`` averaged onto the buoyancy cells)
    and the cell -> face reconstruction ``Sine-II(CellAvg, DIR) ->
    Sine-I(Inner, DIR)`` (buoyancy reconstructed onto the ``w`` faces).
    Because the FV stencil is bitwise the nodal ``Center <-> Inner`` one
    at second order, the diagonal is the same ``cos(k dz/2)`` two-point
    mean (no ``sinc``, the correction-1 pattern).

    The Neumann ``CellAvg`` cosine (DCT-II, the ``a[z]`` pressure interp)
    would land on cosine values at the interior faces — not a grounded
    coefficient family — so it raises ``EigenbasisError`` and the eigen
    layer must skip that factor, mirroring the nodal DCT-II case.

    Parameters
    ----------
    domain : SineSpace | CosineSpace
        The bare trig coefficient factor (average- or nodal-face
        origin).

    Returns
    -------
    SineSpace | CosineSpace
        The same-family, origin-staggered codomain factor.
    """
    mesh = domain.mesh
    origin = domain.origin
    if isinstance(domain, CosineSpace) and isinstance(origin, CellAvg):
        raise EigenbasisError(
            "interpolate on the FV DCT-II family (Neumann CellAvg "
            "origin) lands on cosine values at the interior faces — "
            "not a grounded coefficient family; the eigen layer must "
            "skip this factor")
    if isinstance(domain, SineSpace) and isinstance(origin, CellAvg):
        target: SineSpace | CosineSpace = mesh.sine(
            mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET))
    elif (isinstance(domain, SineSpace)
          and isinstance(origin, NodalSpace)
          and origin.node_set is NodeSet.INNER):
        target = mesh.sine(mesh.average(CellAvg, bc=BC.DIRICHLET))
    else:
        raise EigenbasisError(
            f"no FV staggering-interp (sine/cosine) pairing on "
            f"{domain!r}: the walled FV C-grid reconstruction pairs the "
            "Dirichlet CellAvg sine (DST-II, buoyancy) with the "
            "Dirichlet Inner sine (DST-I, the w face) only; other trig "
            "factors carry no FV reconstruction diagonal")
    if origin.scalars is Scalars.COMPLEX:  # pragma: no cover
        target = target.as_complex()
    return target


def trig_interp_codomain(
    domain: SineSpace | CosineSpace,
) -> SineSpace | CosineSpace:
    r"""
    Coefficient-side ``interpolate`` codomain of a trig factor.

    Description
    -----------
    The constitutive pairing of the two-point staggering mean
    (module table ``_TRIG_INTERP_PAIRING``): the family and the BC
    kind are kept while the origin node set staggers —
    ``Sine-I <-> Sine-II`` and ``Cosine-I -> Cosine-II``. The DCT-II
    (Neumann ``Center``) domain would land on cosine values at the
    interior faces — not a grounded coefficient family — and raises
    ``EigenbasisError`` so the eigen layer skips that factor.

    Parameters
    ----------
    domain : SineSpace | CosineSpace
        The bare trig coefficient factor.

    Returns
    -------
    SineSpace | CosineSpace
        The same-family codomain factor (scalars preserved).
    """
    origin = domain.origin
    if (isinstance(domain, CosineSpace)
            and isinstance(origin, NodalSpace)
            and origin.node_set is NodeSet.CENTER):
        raise EigenbasisError(
            "interpolate on the DCT-II family (Neumann Center "
            "origin) lands on cosine values at the interior faces — "
            "not a grounded coefficient family; the eigen layer "
            "must skip this factor")
    return _trig_pairing(domain, _TRIG_INTERP_PAIRING, "interpolate")


def _trig_pairing(
    domain: SineSpace | CosineSpace,
    table: dict[tuple[type, NodeSet], tuple[str, NodeSet, BC]],
    operation: str,
) -> SineSpace | CosineSpace:
    """Resolve one row of a trig staggering pairing table."""
    origin = domain.origin
    node_set = (origin.node_set
                if isinstance(origin, NodalSpace) else None)
    row = table.get((type(domain), node_set))
    if row is None:
        raise SpaceMismatchError(
            f"no {operation} pairing on {domain!r}: the trig "
            "staggering tables cover the DST-II/DST-I/DCT-II/DCT-I "
            "families", left=domain, operation=operation)
    family, target_set, kind = row
    partner = _paired_origin(origin, target_set, kind)
    return getattr(domain.mesh, family)(partner)


def trig_staggering_symbol(
    bare: SpaceLike, axis: str,
    domain: SineSpace | CosineSpace,
    codomain: SineSpace | CosineSpace,
    magnitude: object, top: object,
) -> Symbol:
    r"""
    Sine/cosine diagonal of a bounded staggering stencil.

    Description
    -----------
    The bounded sibling of ``_staggering_symbol``: on a walled mesh
    the order-2 staggering stencils diagonalize in the sine/cosine
    basis with **real** diagonals — the walls kill the periodic
    staggering phase; the half-cell move is absorbed by the family
    flip (``diff``) / half-shifted same-family evaluation
    (``interpolate``) of the codomain basis. The diagonal is
    ``magnitude(pi m / L, dz)`` evaluated on the **codomain** mode
    table (slot ``j`` holds mode ``j + codomain.mode_offset``, where
    the derived-shift ``Symbol`` stores its data), with two exact
    snaps: ``top`` maps the top-mode entry (mode ``n``, half-angle
    ``pi / 2``) to its analytic value — ``cos(pi/2) = 0`` exactly
    for the interp diagonal, the bounded analogue of the periodic
    Nyquist snap — and codomain modes **absent from the domain's
    mode range** (e.g. cosine ``k = 0`` under diff-from-sine) are
    pinned to exact structural zeros: they multiply structural
    zero-fills of the embedding anyway, and exact zeros keep
    ``Symbol.inverse`` and round trips clean.

    Parameters
    ----------
    bare : SpaceLike
        The bare operand space the eigenvalue query threads.
    axis : str
        The coordinate the stencil acts along.
    domain : SineSpace | CosineSpace
        The source trig coefficient factor.
    codomain : SineSpace | CosineSpace
        The paired codomain trig factor (pairing tables above).
    magnitude : object
        Callable ``(k, dz) -> diagonal`` on the codomain mode table.
    top : object
        Callable snapping the top-mode entry to its exact value.

    Returns
    -------
    Symbol
        The real, derived-shift staggering diagonal.
    """
    n = domain.mesh.n_cells
    dz = _length(domain) / n
    leaf = magnitude(trig_wavenumbers(codomain), dz)
    offset = codomain.mode_offset
    slots = codomain.shape[0]
    top_slot = n - offset  # the half-angle pi/2 mode, if present
    if 0 <= top_slot < slots:
        leaf = leaf.at[top_slot].set(top(leaf[top_slot]))
    head = max(0, domain.mode_offset - offset)
    tail = max(0, (offset + slots)
               - (domain.mode_offset + domain.shape[0]))
    if head:  # codomain modes below the domain's mode range
        leaf = leaf.at[:head].set(0.0)
    if tail:  # codomain modes above the domain's mode range
        leaf = leaf.at[slots - tail:].set(0.0)
    return diagonal_symbol(bare, axis, domain, codomain, leaf)


def trig_finite_difference_symbol(
    bare: SpaceLike, axis: str,
    domain: SineSpace | CosineSpace,
    codomain: SineSpace | CosineSpace,
) -> Symbol:
    r"""
    Bounded order-2 staggered-FD diagonal ``±2 sin(k dz/2)/dz``.

    Description
    -----------
    The bounded row of the ``k_hat`` table: the **same magnitude**
    as the periodic ``finite_difference_symbol``, evaluated at
    ``k = pi m / L`` on the codomain mode table, real (no phase),
    with the nodal kernels' family-flip signs — ``+k_hat`` on
    sine -> cosine and ``-k_hat`` on cosine -> sine, exactly as the
    staggered nodal ground truth (and ``_cosine_to_sine``) spell it.

    Parameters
    ----------
    bare : SpaceLike
        The bare operand space the eigenvalue query threads.
    axis : str
        The coordinate the derivative acts along.
    domain : SineSpace | CosineSpace
        The source trig coefficient factor.
    codomain : SineSpace | CosineSpace
        The flipped-family codomain factor.

    Returns
    -------
    Symbol
        The real ``±k_hat`` derived-shift diagonal.
    """
    sign = 1.0 if isinstance(domain, SineSpace) else -1.0
    # the top-mode entry is already exact: sin(pi/2) == 1.0 bitwise
    return trig_staggering_symbol(
        bare, axis, domain, codomain,
        lambda k, dz: sign * (2.0 * jnp.sin(k * dz / 2.0) / dz),
        jnp.real)


def trig_linear_interp_symbol(
    bare: SpaceLike, axis: str,
    domain: SineSpace | CosineSpace,
    codomain: SineSpace | CosineSpace,
) -> Symbol:
    r"""
    Bounded two-point averaging diagonal ``cos(k dz/2)``.

    Description
    -----------
    The bounded row of the ``one_hat`` table: the **same magnitude**
    as the periodic ``linear_interp_symbol`` at ``k = pi m / L`` on
    the codomain mode table, real, family kept. The top-mode entry
    (half-angle ``pi / 2``) is a structural zero — ``cos(pi/2) = 0``
    exactly, so ``Symbol.inverse`` sees it.

    Parameters
    ----------
    bare : SpaceLike
        The bare operand space the eigenvalue query threads.
    axis : str
        The coordinate the interpolation acts along.
    domain : SineSpace | CosineSpace
        The source trig coefficient factor.
    codomain : SineSpace | CosineSpace
        The same-family codomain factor.

    Returns
    -------
    Symbol
        The real ``one_hat`` derived-shift diagonal.
    """
    return trig_staggering_symbol(
        bare, axis, domain, codomain,
        lambda k, dz: jnp.cos(k * dz / 2.0),
        jnp.zeros_like)


def _match_scalars(
    origin: FunctionSpace, src: FourierSpace,
) -> FunctionSpace:
    """Return ``origin`` in ``src``'s Körper (the codomain origin)."""
    if src.scalars is Scalars.COMPLEX:
        return origin.as_complex()
    return origin


def _staggering_symbol(
    bare: SpaceLike, axis: str, src: FourierSpace,
    src_origin: FunctionSpace, codomain_origin: FunctionSpace,
    magnitude: object, nyquist: object,
) -> Symbol:
    r"""
    Fourier diagonal of a periodic-nodal/average staggering stencil.

    Description
    -----------
    The shared body of the periodic staggering ``eigenvalues`` —
    ``FiniteDifference`` / ``LinearInterp`` (nodal) and the FV
    ``FluxDifference`` / ``DualFluxDifference`` / ``FaceDifference`` /
    ``LinearReconstruction`` (average family): the retagging
    ``Fourier(src origin) -> Fourier(codomain origin)`` diagonal
    ``magnitude(k, dx) * e^{i k delta dx}``. The two origins are
    nodal or average — ``delta`` is their first-node offset
    difference (:func:`_first_node_offset`), and the average family
    carries no extra factor at second order (a ``CellAvg`` divergence
    coefficient relates to the ``Right``-flux coefficient by exactly
    the ``i k_hat`` staggering diagonal, no separate sinc — the
    ``sinc`` lives inside ``k_hat = k sinc(k dx / 2)``). ``src`` fixes
    the layout — its scalars select the half/full spectrum (decision
    3) — and the codomain Fourier factor inherits ``src``'s Körper.

    Unlike the pure phase shift (``PhaseShift``/``SincShift``, where a
    half-cell shift of a real even-n Nyquist has no valid rfft layout
    and is zeroed), a staggering **stencil** carries a magnitude that
    combines with the phase into a Nyquist leaf that *is* representable
    — real for the first difference (``2i sin(π/2)/dx · e^{iπ/2} =
    -2/dx``), zero for the two-point average (``cos(π/2) = 0``). The
    honest ``bwd @ fwd`` Laplacian must recover ``-khat^2`` there
    (matching the ``staggered_diff`` kernel exactly, so the pressure
    projection drives the discrete divergence to machine zero), so the
    Nyquist is **kept**, not zeroed.

    The Nyquist entry is **snapped to its exact analytic value** by
    ``nyquist`` (a callable on the computed entry): the interp
    diagonal is a structural zero there (``cos(π/2) = 0`` exactly, so
    ``Symbol.inverse`` sees it) and the first difference is exactly
    real (``exp``/``sin`` round trips leave a spurious ~1e-16
    imaginary part otherwise). Only even-n factors carry a Nyquist
    mode: index ``-1`` on the real half spectrum, ``n // 2`` on the
    complex fft layout (matching ``_zero_nyquist``).
    """
    dst = src.mesh.fourier(origin=_match_scalars(codomain_origin, src))
    dx = _length(src_origin) / src_origin.shape[0]
    k = fourier_wavenumbers(src)
    delta = (_first_node_offset(codomain_origin)
             - _first_node_offset(src_origin))
    leaf = magnitude(k, dx) * jnp.exp(1j * k * (delta * dx))
    n = src_origin.shape[0]
    if n % 2 == 0:  # only even-n factors have a Nyquist mode
        index = n // 2 if src.scalars is Scalars.COMPLEX else -1
        leaf = leaf.at[index].set(nyquist(leaf[index]))
    return diagonal_symbol(bare, axis, src, dst, leaf)


def finite_difference_symbol(
    bare: SpaceLike, axis: str, src: FourierSpace,
    src_origin: FunctionSpace, codomain_origin: FunctionSpace,
) -> Symbol:
    r"""
    Order-2 staggered-FD Fourier diagonal ``2i sin(k dx/2)/dx`` (phase).

    Description
    -----------
    The retagging ``i k_hat`` diagonal, shared by the nodal staggered
    derivative (``FiniteDifference``) and the FV flux/face
    differences (``FluxDifference`` / ``DualFluxDifference`` /
    ``FaceDifference``): the origins are nodal or average, and the
    inter-origin phase (``delta = codomain - src`` first-node offset)
    picks the direction. At second order the FV difference of a
    ``Right`` flux onto a ``CellAvg`` divergence (or ``Center`` onto
    ``FaceAvg``) carries the identical ``i k_hat`` numbers as the
    nodal ``Right -> Center`` / ``Center -> Right`` stencil, differing
    only in the codomain tag.

    Parameters
    ----------
    bare : SpaceLike
        The bare operand space the eigenvalue query threads.
    axis : str
        The coordinate the derivative acts along.
    src : FourierSpace
        The source Fourier factor (fixing the coefficient layout).
    src_origin : FunctionSpace
        The periodic nodal/average origin of ``src`` (measure / node
        offset).
    codomain_origin : FunctionSpace
        The staggered codomain nodal/average factor.

    Returns
    -------
    Symbol
        The retagging ``i k_hat`` diagonal.
    """
    # the Nyquist leaf is exactly real: 2i sin(±π/2)/dx · e^{±iπδ}
    return _staggering_symbol(
        bare, axis, src, src_origin, codomain_origin,
        lambda k, dx: 1j * (2.0 * jnp.sin(k * dx / 2.0) / dx),
        jnp.real)


def linear_interp_symbol(
    bare: SpaceLike, axis: str, src: FourierSpace,
    src_origin: FunctionSpace, codomain_origin: FunctionSpace,
) -> Symbol:
    r"""
    Two-point averaging Fourier diagonal ``cos(k dx/2)`` (one_hat).

    Description
    -----------
    The retagging ``one_hat`` diagonal, shared by the nodal two-point
    mean (``LinearInterp``) and the FV ``LinearReconstruction``: the
    origins are nodal or average, and the inter-origin phase picks the
    direction. At second order the FV cell-average <-> face-value
    reconstruction is the same two-point mean as the nodal
    interpolation — the deconvolution ``sinc`` correction is a
    higher-order effect (this row carries none), so the symbol is the
    plain ``cos(k dx/2)`` averaging diagonal.

    Parameters
    ----------
    bare : SpaceLike
        The bare operand space the eigenvalue query threads.
    axis : str
        The coordinate the interpolation acts along.
    src : FourierSpace
        The source Fourier factor (fixing the coefficient layout).
    src_origin : FunctionSpace
        The periodic nodal/average origin of ``src`` (measure / node
        offset).
    codomain_origin : FunctionSpace
        The staggered codomain nodal/average factor.

    Returns
    -------
    Symbol
        The retagging ``one_hat`` diagonal.
    """
    # the Nyquist leaf is a structural zero: cos(π/2) = 0 exactly
    return _staggering_symbol(
        bare, axis, src, src_origin, codomain_origin,
        lambda k, dx: jnp.cos(k * dx / 2.0),
        jnp.zeros_like)
