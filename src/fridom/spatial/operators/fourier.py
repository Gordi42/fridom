"""
``Fourier``: the FFT-family transform.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_transforms.md``
("Fourier"). Scalars drive the realization with no special-casing
(rules section 3.1): ``fr.Real`` origins produce the Hermitian
half-spectrum coefficient space via ``rfft`` (the rfft layout *is*
the shape, section 3.2), ``fr.Complex`` origins the full spectrum via
``fft``. Coefficients follow the **index-based amplitude convention**
``c = fft(v) / n`` (``norm="forward"``): ``v_j = sum_k c_k
exp(2 pi i j k / n)``, so per-origin spaces of one mode differ by the
inter-origin phase (``PhaseShift``). Average origins are ordinary
origins (the origin tag carries the sinc relation, ``SincShift``).

Padding (section 3.12): the padded ``backward`` zero-embeds the
spectrum onto the refined mesh's mode set (splitting the
self-conjugate Nyquist coefficient of even-length spectra) and the
padded ``forward`` trims back, folding the fine ``+/-N/2`` modes onto
the coarse Nyquist slot — so pad-then-trim is exact.
"""
# Wave 3: Fourier
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.storage import hermitian_project
from fridom.spatial.operators.transform import (
    Transform,
    TransformPlan,
    TransformStage,
    axis_concat,
    axis_slice,
    axis_zeros,
)
from fridom.spatial.spaces.coefficient import FourierSpace
from fridom.spatial.spaces.nodal import NodeSet

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike


@final
class Fourier(Transform):

    """
    FFT-family transform to per-origin Fourier coefficient spaces.

    Description
    -----------
    Grid-bound (see ``Transform``). Multi-axis real fields follow the
    planned schedule (section 5.1): only the first stage is
    real-to-complex — its factor keeps the real origin and the half
    spectrum — and every later stage targets the complexified origin
    (full spectrum). ``forward`` re-projects the Hermitian value
    invariant (exactly-real self-conjugate modes) after the kernels.

    Parameters
    ----------
    grid : Grid
        The grid to bind.
    axes : tuple[str, ...] | str | None, optional
        Coordinate names to transform along (default: None, all).
    pad : PadFactor | None, optional
        Dealiasing pad factor (default: None).
    """

    _hermitian: ClassVar[bool] = True
    _space_family: ClassVar[type] = FourierSpace

    def truncation_mask(
        self,
        space: object,
        keep: object = None,
    ) -> object:
        """
        0/1 truncation-filter Symbol on ``space`` (2/3 rule).

        Parameters
        ----------
        space : FunctionSpace | TensorProductSpace
            The coefficient space to mask.
        keep : Fraction, optional
            The retained band fraction (default: 2/3).

        Returns
        -------
        Symbol
            Never returns: ``Symbol`` is designed-for.
        """
        raise NotImplementedError(
            "truncation_mask is designed-for: it returns a Symbol, "
            "which lands with the Symbol cluster")

    # ================================================================
    #  Extension contract
    # ================================================================
    def _coefficient_factor(self, origin: FunctionSpace, *,
                            half: bool) -> FunctionSpace:
        """
        Per-origin Fourier factor: ``mesh.fourier(origin=...)``.

        Description
        -----------
        The half-spectrum stage keeps the real origin; every other
        stage targets the complexified origin (idempotent on complex
        origins).
        """
        target = origin if half else origin.as_complex()
        try:
            return origin.mesh.fourier(origin=target)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SpaceMismatchError(
                f"no Fourier signature on {origin!r}: {exc}",
                left=origin, operation="forward") from exc

    def _project(self, data: jax.Array,
                 codomain: SpaceLike) -> jax.Array:
        """
        Enforce the Hermitian value invariant (section 3.2).

        Description
        -----------
        Delegates to ``storage.hermitian_project``, whose
        ``flat_hermitian_applies`` guard makes the flat projection
        fire only when the half-spectrum factor is the codomain's
        sole complex-carrying factor: with further
        coefficient/complex factors present the invariant is the
        conjugate *pairing* ``c[0, ky] == conj(c[0, -ky])``, which
        the kernels satisfy by construction (the schedule is an
        rfftn) and which a flat imag-zeroing would corrupt. The
        field factory and the random factory share the same guard.
        """
        return hermitian_project(data, codomain)

    def _forward_kernel(self, data: jax.Array,
                        stage: TransformStage) -> jax.Array:
        """(r)fft one axis; trim to the coarse spectrum if padded."""
        axis = stage.index
        n = stage.coeff.origin.shape[0]
        m = stage.nodal.shape[0]
        if stage.half:
            c = jnp.fft.rfft(data, axis=axis, norm="forward")
            if m == n:
                return c
            c = c * _origin_phase(stage, data.ndim, sign=-1)
            return _trim_half(c, axis, n)
        c = jnp.fft.fft(data, axis=axis, norm="forward")
        if m == n:
            return c
        c = c * _origin_phase(stage, data.ndim, sign=-1)
        return _trim_full(c, axis, n, m)

    def _backward_kernel(self, data: jax.Array,
                         stage: TransformStage) -> jax.Array:
        """Zero-embed the spectrum if padded; inverse (r)fft."""
        axis = stage.index
        n = stage.coeff.origin.shape[0]
        m = stage.nodal.shape[0]
        if stage.half:
            if m != n:
                data = (_pad_half(data, axis, n, m)
                        * _origin_phase(stage, data.ndim, sign=1))
            return jnp.fft.irfft(data, n=m, axis=axis,
                                 norm="forward")
        if m != n:
            data = (_pad_full(data, axis, n, m)
                    * _origin_phase(stage, data.ndim, sign=1))
        return jnp.fft.ifft(data, axis=axis, norm="forward")

    def _forward_fused_kernel(
        self, data: jax.Array, plan: TransformPlan,
    ) -> jax.Array | None:
        """
        All-axis ``rfftn``/``fftn`` for unpadded schedules.

        Description
        -----------
        An unpadded all-Fourier schedule is exactly one
        ``jnp.fft.rfftn`` (real origin) or ``jnp.fft.fftn``
        (complex origin) over the stage axes: the per-stage kernels
        add nothing (no trims, no phases). The single n-D call lets
        XLA hand cuFFT one strided batched plan instead of the
        explicit full-spectrum transpose+fft pairs it emits for 1D
        FFTs along non-innermost axes. ``rfftn`` halves the *last*
        axis in ``axes=``, so the half-spectrum stage is passed
        last — reproducing the staged layout (half spectrum on the
        first-transformed axis) up to rounding. Padded schedules
        return None: their trim and phase steps interleave the
        axes, so they keep the staged path.
        """
        stages = _half_last(plan.stages)
        if self._pad is not None or not stages:
            return None
        axes = tuple(stage.index for stage in stages)
        if stages[-1].half:
            return jnp.fft.rfftn(data, axes=axes, norm="forward")
        return jnp.fft.fftn(data, axes=axes, norm="forward")

    def _backward_fused_kernel(
        self, data: jax.Array, plan: TransformPlan,
    ) -> jax.Array | None:
        """
        All-axis ``irfftn``/``ifftn`` for unpadded schedules.

        Description
        -----------
        The synthesis counterpart of ``_forward_fused_kernel``: one
        ``jnp.fft.irfftn`` (Hermitian half-spectrum operand, passed
        last in ``axes=`` with the explicit output shape ``s=``) or
        ``jnp.fft.ifftn`` over the stage axes. Padded schedules
        return None (staged path).
        """
        stages = _half_last(plan.stages)
        if self._pad is not None or not stages:
            return None
        axes = tuple(stage.index for stage in stages)
        if stages[-1].half:
            sizes = tuple(stage.nodal.shape[0] for stage in stages)
            return jnp.fft.irfftn(data, s=sizes, axes=axes,
                                  norm="forward")
        return jnp.fft.ifftn(data, axes=axes, norm="forward")


# ================================================================
#  Fused-schedule helper
# ================================================================
def _half_last(
    stages: tuple[TransformStage, ...],
) -> tuple[TransformStage, ...]:
    """
    Reorder stages so the half-spectrum stage comes last.

    Description
    -----------
    ``jnp.fft.rfftn``/``irfftn`` treat the last entry of ``axes=``
    as the real (halved) axis; the planner schedules it first on
    ``forward`` and last on ``backward``. Full-spectrum stages keep
    their relative order (their mutual order does not affect the
    result).

    Parameters
    ----------
    stages : tuple[TransformStage, ...]
        The planned stages, in execution order.

    Returns
    -------
    tuple[TransformStage, ...]
        The stages with the half-spectrum stage (if any) last.
    """
    full = tuple(s for s in stages if not s.half)
    return full + tuple(s for s in stages if s.half)


# ================================================================
#  Spectrum embedding / trimming (Nyquist-exact, section 3.12)
# ================================================================
# node-set offsets of the first node from x_min, in cell widths
_NODE_OFFSETS = {
    NodeSet.CENTER: 0.5,
    NodeSet.LEFT: 0.0,
    NodeSet.RIGHT: 1.0,
}


def _origin_phase(stage: TransformStage, ndim: int,
                  sign: int) -> jax.Array:
    r"""
    Inter-resolution origin phase of a padded Fourier stage.

    Description
    -----------
    The stored coefficients are index-based relative to the *coarse*
    origin's first node; the refined mesh's same-node-set space
    starts at a different physical offset (``Center``: dx/2 shifts
    with dx). The padded synthesis therefore rotates the embedded
    fine spectrum by ``e^{i k (x0_fine - x0_coarse)}`` (``sign=1``)
    and the padded analysis removes it (``sign=-1``), so
    pad-then-trim stays the exact identity. ``Left`` origins sit on
    ``x_min`` and get the unit phase.

    Parameters
    ----------
    stage : TransformStage
        The padded stage (fine ``nodal``, coarse ``coeff``).
    ndim : int
        Rank of the array the phase multiplies.
    sign : int
        +1 for the backward (synthesis), -1 for the forward (trim).

    Returns
    -------
    jax.Array
        The broadcastable phase diagonal on the fine spectrum.
    """
    origin = stage.coeff.origin
    mesh = origin.mesh
    n = origin.shape[0]
    m = stage.nodal.shape[0]
    length = mesh.extent[1] - mesh.extent[0]
    offset = _NODE_OFFSETS[origin.node_set]
    shift = offset * (length / m - length / n)
    if stage.half:
        modes = jnp.arange(m // 2 + 1, dtype=dtype_real())
    else:
        modes = jnp.fft.fftfreq(m, 1.0 / m).astype(dtype_real())
    phase = jnp.exp(sign * 2j * jnp.pi / length * modes * shift)
    shape = [1] * ndim
    shape[stage.index] = phase.shape[0]
    return phase.reshape(shape)
def _trim_half(c: jax.Array, axis: int, n: int) -> jax.Array:
    """
    Trim a fine half spectrum to ``n // 2 + 1`` coarse modes.

    Description
    -----------
    For even ``n`` the coarse Nyquist slot receives the fold of the
    fine ``+N/2`` and ``-N/2`` modes, ``d + conj(d)`` (the ``-N/2``
    mode is the Hermitian mirror of ``+N/2`` on the fine grid).
    """
    if n % 2:
        return axis_slice(c, axis, 0, n // 2 + 1)
    head = axis_slice(c, axis, 0, n // 2)
    nyquist = axis_slice(c, axis, n // 2, n // 2 + 1)
    return axis_concat((head, nyquist + jnp.conj(nyquist)), axis)


def _pad_half(c: jax.Array, axis: int, n: int, m: int) -> jax.Array:
    """
    Zero-embed a coarse half spectrum into ``m // 2 + 1`` modes.

    Description
    -----------
    For even ``n`` the (real) coarse Nyquist coefficient splits in
    half: the fine ``+N/2`` slot takes ``c / 2`` and the ``-N/2``
    mirror is implied by the Hermitian layout — the exact trigonometric
    interpolant of the coarse data.
    """
    width = m // 2 + 1
    have = n // 2 + 1
    if n % 2:
        return axis_concat(
            (c, axis_zeros(c, axis, width - have)), axis)
    head = axis_slice(c, axis, 0, n // 2)
    nyquist = 0.5 * axis_slice(c, axis, n // 2, n // 2 + 1)
    return axis_concat(
        (head, nyquist, axis_zeros(c, axis, width - have)), axis)


def _pad_full(c: jax.Array, axis: int, n: int, m: int) -> jax.Array:
    """
    Zero-embed a coarse full spectrum into ``m`` modes.

    Description
    -----------
    Positive modes keep their slots, negative modes move to the tail;
    for even ``n`` the shared Nyquist coefficient splits in half onto
    the fine ``+N/2`` and ``-N/2`` slots.
    """
    if n % 2:
        h = (n - 1) // 2
        return axis_concat(
            (axis_slice(c, axis, 0, h + 1),
             axis_zeros(c, axis, m - n),
             axis_slice(c, axis, h + 1, n)), axis)
    nyquist = 0.5 * axis_slice(c, axis, n // 2, n // 2 + 1)
    return axis_concat(
        (axis_slice(c, axis, 0, n // 2),
         nyquist,
         axis_zeros(c, axis, m - n - 1),
         nyquist,
         axis_slice(c, axis, n // 2 + 1, n)), axis)


def _trim_full(d: jax.Array, axis: int, n: int, m: int) -> jax.Array:
    """
    Trim a fine full spectrum to ``n`` coarse modes.

    Description
    -----------
    For even ``n`` the coarse Nyquist slot receives the fold of the
    fine ``+N/2`` and ``-N/2`` modes.
    """
    if n % 2:
        h = (n - 1) // 2
        return axis_concat(
            (axis_slice(d, axis, 0, h + 1),
             axis_slice(d, axis, m - h, m)), axis)
    folded = (axis_slice(d, axis, n // 2, n // 2 + 1)
              + axis_slice(d, axis, m - n // 2, m - n // 2 + 1))
    return axis_concat(
        (axis_slice(d, axis, 0, n // 2),
         folded,
         axis_slice(d, axis, m - n // 2 + 1, m)), axis)
