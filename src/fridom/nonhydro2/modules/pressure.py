r"""The spectral pressure Poisson solve.

Description
-----------
The nonhydrostatic incompressibility constraint is enforced by the
fractional-step projection ``div(grad p) = div(u*)`` then
``u = u* - grad p`` (03 §5.6, project-the-state). The elliptic operator
is the grid's own ``dsqr``-weighted discrete Laplacian
``Laplacian(metric={z: 1/dsqr})`` — the block matmul
``Div @ Diag(1, 1, 1/dsqr) @ Grad`` whose ``Divergence`` / ``Gradient``
factors expand into whatever staggered difference scheme the grid
dispatches (the C-grid forward / backward differences). The vertical
``1/dsqr`` weight rides ``ctx.params`` as a traced-but-constant 0-d
leaf, carried on the metric's vertical entry.

The solve is delegated to :class:`SpectralSolve`, which resolves the
transform, queries the operator's ``eigenvalues`` on the coefficient
space, pseudo-inverts (the ``k = 0`` mean-pressure gauge), and applies
the realized-map composition ``backward @ inverse @ forward`` (S2). The
``bwd @ fwd`` round-trip recovers the honest ``-k̂²`` on *every* mode
(Nyquist included), so inverting drives the *discrete* divergence to
machine zero.

On a walled grid (bounded axes, rigid lids) the solve is still purely
spectral: the pressure parity at a rigid lid is Neumann (the
parity-even ``Div @ Diag @ Grad`` chain on the cell centers, the
DCT-II / Cosine-II basis), so the solve runs on the **Neumann-tagged
structural sibling** of the divergence space — same mesh, node set,
and shape, only the BC tag differs. The incoming (BC-free) divergence
is retagged onto that sibling (the trig transform rows are keyed on
the BC-tagged origins), the mixed ``Fourier x Fourier x Cosine``
product resolves through ``ComposedTransform``, and the solution is
retagged back onto the caller's BC-free space. The vertical eigenvalue
is the exact trig one, ``k̂_z = 2 sin(pi m dz / (2 L_z)) / dz``, with
the single structural zero at ``(k_x, k_y, m) = (0, 0, 0)`` — the mean
gauge through ``Symbol.inverse``. On a fully periodic grid the sibling
*is* the space itself (interned identity) and no retag happens.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.spatial.bc import BC
from fridom.spatial.operators.composed import (
    Diag,
    Divergence,
    Gradient,
)
from fridom.spatial.operators.spectral_solve import SpectralSolve
from fridom.spatial.spaces.nodal import NodalSpace

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.spaces.tensor_product import SpaceLike


def _neumann_sibling(space: SpaceLike) -> SpaceLike:
    r"""
    Return the Neumann-tagged structural sibling of a solve space.

    Description
    -----------
    The trig transform rows of a walled grid are keyed on the
    BC-tagged nodal origins, and the pressure parity at rigid lids
    is Neumann (the parity-even ``Div @ Diag @ Grad`` chain on the
    cell centers: the DCT-II / Cosine-II origin). Every nodal factor
    on a bounded mesh is swapped for its ``BC.NEUMANN`` sibling —
    same mesh, node set, and shape (a Neumann tag drops no DOF) —
    while periodic factors pass through, so on a fully periodic grid
    the sibling *is* the space itself (interned identity, the
    no-retag fast path).

    Parameters
    ----------
    space : SpaceLike
        The bare (cell-centered) solve space.

    Returns
    -------
    SpaceLike
        The interned sibling (``space`` itself when no factor is
        bounded).
    """
    replacements = {
        factor.names[0]: factor.mesh.nodal(
            factor.node_set, bc=BC.NEUMANN)
        for factor in space.factors
        if isinstance(factor, NodalSpace)
        and not getattr(factor.mesh, "periodic", True)
    }
    if not replacements:
        return space
    return space.replace(**replacements)


def _dirichlet_mid(space: SpaceLike, axis: str) -> SpaceLike:
    r"""
    Declare the Dirichlet parity of one gradient-output space.

    Description
    -----------
    The mid legs of the parity-even chain ``Div @ Diag @ Grad`` on
    the Neumann solve space: the wall-normal gradient of an even
    (Neumann) pressure is odd, i.e. it vanishes **at** the wall —
    the Dirichlet claim on the staggered faces (the free-slip retag
    precedent). Declaring it here keys the divergence legs on the
    BC-structured rows; the BC-free ``Inner -> Center`` rows do not
    exist (R1, boundary_plan.md — a BC-free bounded side defines no
    exterior values). Periodic factors pass through untouched.

    Parameters
    ----------
    space : SpaceLike
        One per-axis gradient codomain (bare).
    axis : str
        The gradient axis of this component.

    Returns
    -------
    SpaceLike
        The interned sibling with the axis factor Dirichlet-tagged
        (``space`` itself on a periodic axis).
    """
    factor = space.factor(axis)
    if (isinstance(factor, NodalSpace)
            and not getattr(factor.mesh, "periodic", True)
            and factor.bc.is_free):
        return space.replace(**{axis: factor.mesh.nodal(
            factor.node_set, bc=BC.DIRICHLET)})
    return space


class SpectralPressureSolver:

    """Grid-bound spectral solve of ``lap(p) = div``.

    Description
    -----------
    Constructed at trace time inside the projection stage from the
    operand's grid; carries no mutable state and is not a pytree leaf.
    Per solve it expands the ``dsqr``-weighted Laplacian
    ``Div @ Diag(1, .., 1/dsqr) @ Grad`` (the live ``1/dsqr`` riding the
    vertical metric entry) and inverts it through :class:`SpectralSolve`.

    On a walled grid the expansion and the solve run on the
    Neumann-tagged sibling of ``space`` (the trig-transform origin of
    the pressure parity); the divergence is retagged onto it and the
    solution retagged back, so the caller's BC-free spaces are
    preserved. On a fully periodic grid the sibling is the space
    itself and no retag happens.

    Parameters
    ----------
    grid : object
        The grid carrying the transform / dispatch registry.
    space : SpaceLike
        The (cell-centered) function space of the divergence operand.
    vertical : str
        The vertical coordinate name (the ``1/dsqr``-weighted axis).
    single_precision : bool, optional
        Run the spectral pressure solve (the ``rfftn`` / spectral
        divide / ``irfftn`` pipeline) in single precision while the
        velocity state stays ``float64`` — a performance option
        forwarded to :class:`SpectralSolve` (see its
        ``single_precision`` doc). The divergence is cast to
        ``float32`` before the transform and the pressure returns on
        ``float64``. Measured (512^3 A100 AB3, 2026-07-12): -12% on
        the linear step, -8% with advection. Accuracy (128^3, 500
        steps): the projected velocity's residual divergence sits at
        the float32 floor (~1e-7 x |u| absolute vs f64 machine zero,
        per step) and the accumulated state error vs the f64 solve
        is ~1.7e-5 relative; no measurable energy drift. Off by
        default: the f64 projection is bitwise preserved
        (default: False).
    """

    def __init__(
        self, grid: object, space: SpaceLike, *, vertical: str,
        single_precision: bool = False,
    ) -> None:
        """Store the grid, spaces, and vertical axis name."""
        self._grid: object = grid
        self._space: SpaceLike = space
        self._vertical: str = vertical
        self._single_precision: bool = bool(single_precision)
        # the space the spectral solve runs on: the Neumann-tagged
        # sibling on a walled grid, the space itself on a periodic one
        self._solve_space: SpaceLike = _neumann_sibling(space.bare)

    def solve(
        self, div: ScalarField, *, dsqr: jax.Array,
    ) -> ScalarField:
        """Return ``p`` with ``div(grad p) = div`` (discrete, exact).

        Parameters
        ----------
        div : ScalarField
            The (cell-centered, real) divergence of the provisional
            velocity.
        dsqr : jax.Array
            The live squared-aspect-ratio leaf.

        Returns
        -------
        ScalarField
            The pressure on the same (cell-centered) space as ``div``.
        """
        solve_space = self._solve_space
        # div @ Diag @ grad expanded leg by leg: the grad rows key
        # on the (Neumann-tagged) solve space, the div rows on the
        # Dirichlet-declared mid parity (see _dirichlet_mid) — the
        # same interned stencil entries the Laplacian builder used
        # to resolve through the BC-free mid rows before R1
        grad_block = Gradient().expand(solve_space, self._grid)
        axes = solve_space.active_axis_names
        mid = tuple(
            _dirichlet_mid(space, axis)
            for axis, space in zip(
                axes, grad_block.codomains(solve_space),
                strict=True))
        div_block = Divergence().expand(mid, self._grid)
        diag = Diag({self._vertical: 1.0 / dsqr}, axes=axes)
        laplacian = (div_block @ diag @ grad_block).scalar()
        solve = SpectralSolve(
            laplacian, self._grid, solve_space,
            single_precision=self._single_precision)
        if solve_space is self._space.bare:
            return solve.solve(div)
        return solve.solve(div.retag(solve_space)).retag(div)
