r"""
``Chebyshev``: Gauss-Lobatto nodal <-> Chebyshev coefficients.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_transforms.md``
("Chebyshev"). The transform pairs the ``ChebyshevMesh``'s Lobatto
(outer) space with its ``ChebyshevSpace``: values at the n + 1
Gauss-Lobatto points and coefficients ``a_k`` of

.. math::

    f(x_j) = \sum_{k=0}^{n} a_k T_k(\xi_j),

where :math:`\xi` is the affine map of the mesh interval onto
[-1, 1]. **Storage order is ascending in x** (index 0 at ``x_min``,
i.e. :math:`\xi_0 = -1`); the kernels flip to the standard Lobatto
order :math:`\xi_j = \cos(\pi j / n)` internally and evaluate the
DCT-I through a length-2n complex FFT of the even extension
(discrete Chebyshev orthogonality: ``a_k = W_k / (n cbar_k)`` with
``cbar_0 = cbar_n = 2``).

Shen-basis codomains (BC-structured Lobatto origins) are
designed-for with the Galerkin spaces; BC-structured origins raise.
Padded variants raise at construction: ``refined()`` is iteration 1
on ``IntervalMesh`` only.
"""
# Wave 3: Chebyshev
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

import jax.numpy as jnp

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.operators.transform import (
    Transform,
    TransformStage,
    axis_concat,
    axis_slice,
    axis_vector,
    embed_tail,
)
from fridom.framework2.grid.spaces.coefficient import ChebyshevSpace
from fridom.framework2.grid.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )


@final
class Chebyshev(Transform):

    """
    Chebyshev transform (Gauss-Lobatto nodes <-> Cheb coefficients).

    Description
    -----------
    Grid-bound (see ``Transform``). ``requirements`` declares layout
    "transpose" like every transform: Chebyshev meshes carry
    transpose-capable decomposition traits (owner directive), so the
    planner treats them exactly like ``Fourier``.

    Parameters
    ----------
    grid : Grid
        The grid to bind.
    axes : tuple[str, ...] | str | None, optional
        Coordinate names to transform along (default: None, all).
    pad : PadFactor | None, optional
        Dealiasing pad factor (default: None; raises — refinement is
        designed-for on ``ChebyshevMesh``).
    """

    _space_family: ClassVar[type] = ChebyshevSpace

    def _coefficient_factor(
        self,
        origin: FunctionSpace,
        *,
        half: bool,  # noqa: ARG002 — never a Hermitian stage
    ) -> FunctionSpace:
        """Per-origin Chebyshev factor: ``mesh.chebyshev(origin)``."""
        if not (isinstance(origin, NodalSpace)
                and origin.node_set is NodeSet.OUTER):
            raise SpaceMismatchError(
                f"no Chebyshev signature on {origin!r}: the origin "
                "is the Gauss-Lobatto (outer) space",
                left=origin, operation="forward")
        if not origin.bc.is_free:
            raise SpaceMismatchError(
                f"no Chebyshev signature on {origin!r}: Shen bases "
                "for BC-structured Lobatto origins are designed-for "
                "(Galerkin cluster)", left=origin,
                operation="forward")
        try:
            return origin.mesh.chebyshev(origin)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SpaceMismatchError(
                f"no Chebyshev signature on {origin!r}: {exc}",
                left=origin, operation="forward") from exc

    def _forward_kernel(self, data: jax.Array,
                        stage: TransformStage) -> jax.Array:
        """Analyze one axis; trim to the coarse modes if padded."""
        axis = stage.index
        modes = stage.coeff.shape[0]
        a = _dct1_forward(data, axis)
        if a.shape[axis] == modes:
            return a
        return axis_slice(a, axis, 0, modes)  # pragma: no cover

    def _backward_kernel(self, data: jax.Array,
                         stage: TransformStage) -> jax.Array:
        """Zero-embed the modes if padded; synthesize one axis."""
        axis = stage.index
        points = stage.nodal.shape[0]
        data = embed_tail(data, axis, points)
        return _dct1_backward(data, axis)


# ================================================================
#  DCT-I kernels via length-2n complex FFTs
# ================================================================
def _dct1_forward(v: jax.Array, axis: int) -> jax.Array:
    """
    Chebyshev analysis: n + 1 Lobatto values -> modes 0..n.

    Description
    -----------
    Flips to standard Lobatto order, builds the even extension
    ``w = [r_0..r_n, r_{n-1}..r_1]`` (length 2n), and reads
    ``a_k = W_k / (n cbar_k)`` off its FFT.
    """
    n = v.shape[axis] - 1
    r = jnp.flip(v, axis)
    interior = axis_slice(r, axis, 1, n)
    w = axis_concat((r, jnp.flip(interior, axis)), axis)
    big = jnp.fft.fft(w, axis=axis)
    k = jnp.arange(n + 1)
    cbar = jnp.where((k == 0) | (k == n), 2.0, 1.0)
    return (axis_slice(big, axis, 0, n + 1)
            / axis_vector(n * cbar, v.ndim, axis))


def _dct1_backward(a: jax.Array, axis: int) -> jax.Array:
    """
    Chebyshev synthesis: modes 0..m -> m + 1 Lobatto values.

    Description
    -----------
    Rebuilds the even-extension spectrum ``W_k = m cbar_k a_k``
    (mirror ``W_{2m-k} = W_k``), inverts, and flips back to the
    ascending-x storage order.
    """
    m = a.shape[axis] - 1
    k = jnp.arange(m + 1)
    cbar = jnp.where((k == 0) | (k == m), 2.0, 1.0)
    head = a * axis_vector(m * cbar, a.ndim, axis)
    interior = axis_slice(head, axis, 1, m)
    spectrum = axis_concat((head, jnp.flip(interior, axis)), axis)
    w = jnp.fft.ifft(spectrum, axis=axis)
    r = axis_slice(w, axis, 0, m + 1)
    return jnp.flip(r, axis)
