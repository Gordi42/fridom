r"""
``Sine`` / ``Cosine``: DST/DCT transforms for BC-structured origins.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_transforms.md``
("Sine / Cosine"). The DST/DCT type is **not a parameter**: it
follows from the origin space (rules sections 3.2/3.5) —

- Dirichlet ``Center``: DST-II, n modes ``k = 1..n`` (array index
  ``j`` holds mode ``k = j + 1``, the scipy layout);
- Dirichlet ``Inner``: DST-I, n - 1 modes ``k = 1..n-1``
  (index ``j = k - 1``);
- Neumann ``Center``: DCT-II, n modes ``k = 0..n-1`` (index
  ``j = k``);
- Neumann ``Outer``: DCT-I, n + 1 modes ``k = 0..n`` (index
  ``j = k``) — Neumann never reduces the origin shape (owner
  decision 2026-07-07), so the n + 1 boundary-inclusive nodes make
  the DCT-I shape-honest.

Coefficients follow the **amplitude (synthesis) convention**: for the
mesh interval of length L,

.. math::

    v(x_j) = \sum_k b_k \sin(k \pi (x_j - x_\min) / L)
    \quad\text{resp.}\quad
    \sum_k a_k \cos(k \pi (x_j - x_\min) / L),

so coefficients of one mode agree across grid resolutions and the
padded variants are plain zero-embeddings at the mode tail. The
DST/DCT-I/DST-I kernels evaluate the transform through length-2n
complex FFTs of the odd/even extensions (correct for complex data
too). DCT-II -- the cell-centered Neumann transform of the walled
pressure solve -- adds the Wave-4 fast paths: real data takes a
half-spectrum ``rfft`` on the same extension and complex data a
single length-n ``fft`` (Makhoul interleave) with a pack-two-reals
unpack, each halving the FFT work; odd lengths keep the extension.
"""
# Wave 3: Sine, Cosine
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

import jax.numpy as jnp

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.operators.transform import (
    Transform,
    TransformStage,
    axis_concat,
    axis_slice,
    axis_vector,
    axis_zeros,
    embed_tail,
)
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.coefficient import (
    CosineSpace,
    SineSpace,
)
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )


def _is_type_two(origin: FunctionSpace) -> bool:
    """
    Whether an origin takes the type-II (cell-midpoint) kernel.

    Description
    -----------
    ``Center`` nodal and ``CellAvg`` origins both sample on the
    cell-midpoint grid, so they take the DST-II / DCT-II kernels; the
    ``Inner`` (DST-I) and ``Outer`` (DCT-I) nodal origins take the
    type-I kernels. A ``CellAvg`` cell average is identified with its
    midpoint value at second order (the FV correction-1 pattern), so
    the trig basis diagonalizes it exactly, with no separate ``sinc``.

    Parameters
    ----------
    origin : FunctionSpace
        The coefficient factor's origin space.

    Returns
    -------
    bool
        True for ``Center`` and ``CellAvg`` origins.
    """
    if isinstance(origin, CellAvg):
        return True
    return (isinstance(origin, NodalSpace)
            and origin.node_set is NodeSet.CENTER)


def _validated_origin(origin: FunctionSpace, node_sets: tuple,
                      family: str, kinds: str) -> None:
    """Reject origins outside the iteration-1 DST/DCT families."""
    if (isinstance(origin, NodalSpace)
            and origin.node_set in node_sets):
        return
    if isinstance(origin, CellAvg) and NodeSet.CENTER in node_sets:
        # cell averages sample on the midpoint grid -> the type-II
        # kernel (Neumann CellAvg -> DCT-II, Dirichlet CellAvg -> DST-II)
        return
    raise SpaceMismatchError(
        f"no {family} signature on {origin!r}: iteration 1 covers "
        f"{kinds}", left=origin, operation="forward")


@final
class Sine(Transform):

    """
    DST transform; type (I/II) selected by the origin node set.

    Description
    -----------
    Grid-bound (see ``Transform``). Dirichlet ``Center`` origins take
    the DST-II (n modes), Dirichlet ``Inner`` origins the DST-I
    (n - 1 modes); coefficient counts equal space shapes by
    construction.

    Parameters
    ----------
    grid : Grid
        The grid to bind.
    axes : tuple[str, ...] | str | None, optional
        Coordinate names to transform along (default: None, all).
    pad : PadFactor | None, optional
        Dealiasing pad factor (default: None).
    """

    _space_family: ClassVar[type] = SineSpace

    def _coefficient_factor(
        self,
        origin: FunctionSpace,
        *,
        half: bool,  # noqa: ARG002 — never a Hermitian stage
    ) -> FunctionSpace:
        """Per-origin sine factor: ``mesh.sine(origin)``."""
        _validated_origin(
            origin, (NodeSet.CENTER, NodeSet.INNER), "DST",
            "Dirichlet Center / CellAvg (DST-II) and Dirichlet Inner "
            "(DST-I) origins")
        try:
            return origin.mesh.sine(origin)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SpaceMismatchError(
                f"no DST signature on {origin!r}: {exc}",
                left=origin, operation="forward") from exc

    def _forward_kernel(self, data: jax.Array,
                        stage: TransformStage) -> jax.Array:
        """Analyze one axis; trim to the coarse modes if padded."""
        axis = stage.index
        modes = stage.coeff.shape[0]
        if _is_type_two(stage.coeff.origin):
            b = _dst2_forward(data, axis)
        else:
            b = _dst1_forward(data, axis)
        if b.shape[axis] == modes:
            return b
        return axis_slice(b, axis, 0, modes)

    def _backward_kernel(self, data: jax.Array,
                         stage: TransformStage) -> jax.Array:
        """Zero-embed the modes if padded; synthesize one axis."""
        axis = stage.index
        points = stage.nodal.shape[0]
        data = embed_tail(data, axis, points)
        if _is_type_two(stage.coeff.origin):
            return _dst2_backward(data, axis)
        return _dst1_backward(data, axis)


@final
class Cosine(Transform):

    """
    DCT transform; type (I/II) selected by the origin node set.

    Description
    -----------
    Grid-bound (see ``Transform``). Neumann ``Center`` origins take
    the DCT-II (n modes ``k = 0..n-1``), Neumann ``Outer`` origins
    the DCT-I (n + 1 modes ``k = 0..n``); coefficient counts equal
    space shapes by construction.

    Parameters
    ----------
    grid : Grid
        The grid to bind.
    axes : tuple[str, ...] | str | None, optional
        Coordinate names to transform along (default: None, all).
    pad : PadFactor | None, optional
        Dealiasing pad factor (default: None).
    """

    _space_family: ClassVar[type] = CosineSpace

    def _coefficient_factor(
        self,
        origin: FunctionSpace,
        *,
        half: bool,  # noqa: ARG002 — never a Hermitian stage
    ) -> FunctionSpace:
        """Per-origin cosine factor: ``mesh.cosine(origin)``."""
        _validated_origin(
            origin, (NodeSet.CENTER, NodeSet.OUTER), "DCT",
            "Neumann Center / CellAvg (DCT-II) and Neumann Outer "
            "(DCT-I) origins")
        try:
            return origin.mesh.cosine(origin)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SpaceMismatchError(
                f"no DCT signature on {origin!r}: {exc}",
                left=origin, operation="forward") from exc

    def _forward_kernel(self, data: jax.Array,
                        stage: TransformStage) -> jax.Array:
        """Analyze one axis; trim to the coarse modes if padded."""
        axis = stage.index
        modes = stage.coeff.shape[0]
        if _is_type_two(stage.coeff.origin):
            a = _dct2_forward(data, axis)
        else:
            a = _dct1_forward(data, axis)
        if a.shape[axis] == modes:
            return a
        return axis_slice(a, axis, 0, modes)

    def _backward_kernel(self, data: jax.Array,
                         stage: TransformStage) -> jax.Array:
        """Zero-embed the modes if padded; synthesize one axis."""
        axis = stage.index
        points = stage.nodal.shape[0]
        data = embed_tail(data, axis, points)
        if _is_type_two(stage.coeff.origin):
            return _dct2_backward(data, axis)
        return _dct1_backward(data, axis)


# ================================================================
#  DST/DCT kernels via length-2n complex FFTs
# ================================================================
# Derivations (module docstring conventions; L drops out — the
# kernels see index space):
#
# DST-II, n half-offset samples v_j at (j + 1/2):
#   u = [v, -flip(v)] (odd about x = L), U = fft(u):
#   U_k = -2i e^{i pi k / (2n)} S_k with
#   S_k = sum_j v_j sin(pi k (j + 1/2) / n), and the analysis
#   b_k = (2/n) S_k (k < n), b_n = (1/n) S_n.
# DCT-II: u = [v, flip(v)] (even), U_k = 2 e^{i pi k / (2n)} C_k,
#   a_0 = C_0 / n, a_k = (2/n) C_k.
# DST-I, n - 1 on-lattice samples v_1..v_{n-1}:
#   w = [0, v, 0, -flip(v)], W_k = -2i S_k,
#   b_k = (2/n) S_k, no phases.
# DCT-I, n + 1 on-lattice samples v_0..v_n:
#   u = [v_0..v_n, v_{n-1}..v_1] (even about both boundaries,
#   length 2n), U_k = v_0 + (-1)^k v_n
#   + 2 sum_{j=1}^{n-1} v_j cos(pi j k / n) — real, no phases;
#   a_0 = U_0 / (2n), a_n = U_n / (2n), a_k = U_k / n else.
# Mirror halves follow from the extension symmetry:
#   DST-II: U_{2n-k} = -e^{-i pi k / n} U_k;
#   DCT-II: U_{2n-k} = +e^{-i pi k / n} U_k;  U_n = 0 for DCT-II,
#   U_0 = 0 for DST-II; DST-I: W_{2n-k} = -W_k, W_0 = W_n = 0;
#   DCT-I: U_{2n-k} = U_k (k = 0 and k = n self-mirrored).
def _dst2_forward(v: jax.Array, axis: int) -> jax.Array:
    """DST-II analysis: n half-offset samples -> modes 1..n."""
    n = v.shape[axis]
    u = axis_concat((v, -jnp.flip(v, axis)), axis)
    big = jnp.fft.fft(u, axis=axis)
    k = jnp.arange(1, n + 1)
    weight = jnp.where(k < n, 2.0 / n, 1.0 / n)
    coef = 0.5j * jnp.exp(-1j * jnp.pi * k / (2 * n)) * weight
    return (axis_slice(big, axis, 1, n + 1)
            * axis_vector(coef, v.ndim, axis))


def _dst2_backward(b: jax.Array, axis: int) -> jax.Array:
    """DST-II synthesis: modes 1..p -> p half-offset samples."""
    p = b.shape[axis]
    k = jnp.arange(1, p + 1)
    s_weight = jnp.where(k < p, p / 2.0, float(p))
    head_coef = -2j * jnp.exp(1j * jnp.pi * k / (2 * p)) * s_weight
    head = b * axis_vector(head_coef, b.ndim, axis)
    mirror_coef = -jnp.exp(-1j * jnp.pi * jnp.arange(1, p) / p)
    mirror = (axis_slice(head, axis, 0, p - 1)
              * axis_vector(mirror_coef, b.ndim, axis))
    spectrum = axis_concat(
        (axis_zeros(head, axis, 1), head,
         jnp.flip(mirror, axis)), axis)
    u = jnp.fft.ifft(spectrum, axis=axis)
    return axis_slice(u, axis, 0, p)


def _dct2_forward_ext(v: jax.Array, axis: int) -> jax.Array:
    """DCT-II analysis via the length-2n complex extension (any n)."""
    n = v.shape[axis]
    u = axis_concat((v, jnp.flip(v, axis)), axis)
    big = jnp.fft.fft(u, axis=axis)
    k = jnp.arange(n)
    weight = jnp.where(k == 0, 1.0 / n, 2.0 / n)
    coef = 0.5 * jnp.exp(-1j * jnp.pi * k / (2 * n)) * weight
    return (axis_slice(big, axis, 0, n)
            * axis_vector(coef, v.ndim, axis))


def _dct2_forward_real(v: jax.Array, axis: int) -> jax.Array:
    """DCT-II analysis of real data via a half-spectrum rfft.

    The even extension is real, so an ``rfft`` returns the ``0..n``
    modes at half the work of the length-2n complex ``fft``; the
    result is real (the sliced modes carry no imaginary tail).
    """
    n = v.shape[axis]
    u = axis_concat((v, jnp.flip(v, axis)), axis)
    big = axis_slice(jnp.fft.rfft(u, axis=axis), axis, 0, n)
    k = jnp.arange(n)
    weight = jnp.where(k == 0, 1.0 / n, 2.0 / n)
    coef = 0.5 * jnp.exp(-1j * jnp.pi * k / (2 * n)) * weight
    return (big * axis_vector(coef, v.ndim, axis)).real


def _axis_stride2(v: jax.Array, axis: int, start: int) -> jax.Array:
    """Take every second slot along ``axis`` from ``start`` (static)."""
    index: list[slice] = [slice(None)] * v.ndim
    index[axis] = slice(start, None, 2)
    return v[tuple(index)]


def _dct_interleave(v: jax.Array, axis: int) -> jax.Array:
    """Makhoul even/odd reorder: evens, then the odds reversed."""
    evens = _axis_stride2(v, axis, 0)
    odds = jnp.flip(_axis_stride2(v, axis, 1), axis)
    return axis_concat((evens, odds), axis)


def _dct2_forward_pack(v: jax.Array, axis: int) -> jax.Array:
    """DCT-II analysis of complex data via one length-n fft (Makhoul).

    The even/odd interleave collapses the length-2n transform to a
    single length-n ``fft``. The two-reals split (``re_spec`` /
    ``im_spec``) recovers the transforms of the real and imaginary
    parts from the Hermitian structure of that one spectrum, so the
    complex analysis costs one ``fft(n)`` rather than ``fft(2n)``.
    """
    n = v.shape[axis]
    big = jnp.fft.fft(_dct_interleave(v, axis), axis=axis)
    conj_rev = jnp.conj(jnp.roll(jnp.flip(big, axis), 1, axis))
    re_spec = 0.5 * (big + conj_rev)
    im_spec = -0.5j * (big - conj_rev)
    k = jnp.arange(n)
    phase = axis_vector(jnp.exp(-1j * jnp.pi * k / (2 * n)),
                        v.ndim, axis)
    weight = axis_vector(jnp.where(k == 0, 1.0 / n, 2.0 / n),
                         v.ndim, axis)
    return weight * ((phase * re_spec).real
                     + 1j * (phase * im_spec).real)


def _dct2_forward(v: jax.Array, axis: int) -> jax.Array:
    """DCT-II analysis: n half-offset samples -> modes 0..n-1."""
    if v.shape[axis] % 2:
        return _dct2_forward_ext(v, axis)
    if jnp.iscomplexobj(v):
        return _dct2_forward_pack(v, axis)
    return _dct2_forward_real(v, axis)


def _dct2_backward_ext(a: jax.Array, axis: int) -> jax.Array:
    """DCT-II synthesis via the length-2n spectrum (any n, complex)."""
    p = a.shape[axis]
    k = jnp.arange(p)
    c_weight = jnp.where(k == 0, float(p), p / 2.0)
    head_coef = 2.0 * jnp.exp(1j * jnp.pi * k / (2 * p)) * c_weight
    head = a * axis_vector(head_coef, a.ndim, axis)
    mirror_coef = jnp.exp(-1j * jnp.pi * jnp.arange(1, p) / p)
    mirror = (axis_slice(head, axis, 1, p)
              * axis_vector(mirror_coef, a.ndim, axis))
    spectrum = axis_concat(
        (head, axis_zeros(head, axis, 1),
         jnp.flip(mirror, axis)), axis)
    u = jnp.fft.ifft(spectrum, axis=axis)
    return axis_slice(u, axis, 0, p)


def _dct2_backward_real(a: jax.Array, axis: int) -> jax.Array:
    """DCT-II synthesis of real modes via a half-spectrum irfft.

    The synthesized samples are real, so the length-2n spectrum is
    Hermitian and an ``irfft`` of its ``0..p`` head reproduces them at
    half the work of the length-2n complex ``ifft``.
    """
    p = a.shape[axis]
    k = jnp.arange(p)
    c_weight = jnp.where(k == 0, float(p), p / 2.0)
    head_coef = 2.0 * jnp.exp(1j * jnp.pi * k / (2 * p)) * c_weight
    head = a * axis_vector(head_coef, a.ndim, axis)
    half = axis_concat((head, axis_zeros(head, axis, 1)), axis)
    u = jnp.fft.irfft(half, n=2 * p, axis=axis)
    return axis_slice(u, axis, 0, p)


def _dct2_backward(a: jax.Array, axis: int) -> jax.Array:
    """DCT-II synthesis: modes 0..p-1 -> p half-offset samples."""
    if a.shape[axis] % 2 or jnp.iscomplexobj(a):
        return _dct2_backward_ext(a, axis)
    return _dct2_backward_real(a, axis)


def _dct1_forward(v: jax.Array, axis: int) -> jax.Array:
    """DCT-I analysis: n + 1 on-lattice samples -> modes 0..n."""
    m = v.shape[axis]
    n = m - 1
    interior = axis_slice(v, axis, 1, n)
    u = axis_concat((v, jnp.flip(interior, axis)), axis)
    big = jnp.fft.fft(u, axis=axis)
    k = jnp.arange(m)
    weight = jnp.where((k == 0) | (k == n), 0.5 / n, 1.0 / n)
    return (axis_slice(big, axis, 0, m)
            * axis_vector(weight, v.ndim, axis))


def _dct1_backward(a: jax.Array, axis: int) -> jax.Array:
    """DCT-I synthesis: modes 0..p -> p + 1 on-lattice samples."""
    m = a.shape[axis]
    n = m - 1
    k = jnp.arange(m)
    weight = jnp.where((k == 0) | (k == n), 2.0 * n, float(n))
    head = a * axis_vector(weight, a.ndim, axis)
    mirror = jnp.flip(axis_slice(head, axis, 1, n), axis)
    spectrum = axis_concat((head, mirror), axis)
    u = jnp.fft.ifft(spectrum, axis=axis)
    return axis_slice(u, axis, 0, m)


def _dst1_forward(v: jax.Array, axis: int) -> jax.Array:
    """DST-I analysis: n - 1 interior samples -> modes 1..n-1."""
    n = v.shape[axis] + 1
    zero = axis_zeros(v, axis, 1)
    w = axis_concat((zero, v, zero, -jnp.flip(v, axis)), axis)
    big = jnp.fft.fft(w, axis=axis)
    return (1j / n) * axis_slice(big, axis, 1, n)


def _dst1_backward(b: jax.Array, axis: int) -> jax.Array:
    """DST-I synthesis: modes 1..p -> p interior samples."""
    n = b.shape[axis] + 1
    head = (-1j * n) * b
    zero = axis_zeros(head, axis, 1)
    spectrum = axis_concat(
        (zero, head, zero, -jnp.flip(head, axis)), axis)
    w = jnp.fft.ifft(spectrum, axis=axis)
    return axis_slice(w, axis, 1, n)
