r"""
Symbolic eigenmodes of a linearized model via ``eigh(iML, M)``.

Description
-----------
Phase H1 of the projection/eigenmode roadmap
(``notes/framework2/blocksymbol_l_assembly.md`` §4): the **symbolic**
``BlockSymbol`` assembly of the linearized system operator ``L(k)`` from
the model's linear-term blocks, the culmination the H0 numeric probe
(``model.eigen``) de-risked. Model-agnostic, host-side (analysis, never
the traced step).

The linear tendency ``L`` is translation-invariant on a constant-
coefficient periodic grid, so per Fourier wavenumber ``k`` it is a small
:math:`m\times m` matrix ``L(k)``. We build it *symbolically*: each
retained linear block (``fr.linear_blocks``) resolves to an
``fr.Operator`` whose ``eigenvalues(grid, src_space)`` is a scalar
:class:`~fridom.framework2.grid.operators.symbol.Symbol` carrying the
per-component staggering phase; scaling by the block's constant
coefficient and scattering the blocks into a
:class:`~fridom.framework2.grid.operators.block_symbol.BlockSymbol`
assembles ``L(k)`` in the model's own staggered spectral basis.

Shallow water assembles the **full** ``A(k)`` this way (``p`` is
prognostic — no constraint). Nonhydro assembles only the **raw** blocks
(Coriolis ``u<->v``, buoyancy ``w<-b``, stratification ``b<-w``); the
diagnostic pressure is eliminated by the Leray projector
:math:`P = I - W\,G\,(\nabla^2)^{-1} D` (``G`` = gradient, ``D`` =
divergence, ``W = diag(1, 1, 1/\delta^2)`` the nonhydrostatic vertical
weighting, :math:`(\nabla^2)^{-1}` a :meth:`Symbol.inverse` of the
``dsqr``-weighted discrete Laplacian ``D W G``), also assembled with the
``BlockSymbol`` algebra. The constrained, energy-conserving operator is
``P @ L_raw @ P``.

Because ``L`` is ``M``-skew-adjoint under the energy metric
(``fr.EnergyMetric``), :math:`H := iML` is Hermitian and
:math:`L q = i\omega q \Leftrightarrow H q = -\omega M q`. We reuse the
H0 generalized Hermitian eigensolve (``model.eigen._generalized_eigh``)
batched over modes, so both tiers share one ``eigh(iML, M)`` seam.

Basis note
----------
The symbolic ``L`` sits in the model's **raw staggered spectral basis**
(each component the DFT amplitude on its own C-grid face), identical to
the H0 numeric probe's basis — so the symbolic spectrum and
eigenvectors reproduce the H0 numeric ones to machine precision (a
Nyquist-mode caveat aside). The analytic ``nh``/``sw`` eigenmode ports
express every component relative to a common collocated reference
carrying explicit interpolation phases, so their spectra match the
symbolic ones (basis-invariant) but their eigenvector *arrays* differ
from the symbolic vectors by per-component staggering phases (the
raw-DFT-vs-staggered-transform mismatch the H0 notes recorded).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import dtype_comp
from fridom.framework2.grid.operators.block_symbol import BlockSymbol
from fridom.framework2.grid.operators.composed import Divergence, Gradient
from fridom.framework2.grid.operators.symbol import Symbol
from fridom.framework2.model.eigen import (
    _generalized_eigh,
    _metric_weights,
)
from fridom.framework2.model.energy import EnergyMetric
from fridom.framework2.model.linear_blocks import linear_blocks
from fridom.framework2.model.time_dependent import resolve_at

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.framework2.grid.spaces.tensor_product import SpaceLike
    from fridom.framework2.model.model import Model

# ----------------------------------------------------------------
#  framework2 core cannot import nonhydro2 (dependency direction), so
#  the pressure constraint is detected by the dotted parameter key
#  directly -- the same iteration-1 coupling ``EnergyMetric.from_model``
#  uses for the ``dsqr`` energy weight.
# ----------------------------------------------------------------
_DSQR = "nonhydro.dsqr"
_VELOCITY = ("u", "v", "w")
_VERTICAL = "w"


class SymbolicEigenmodes:

    r"""
    Per-mode symbolic eigenpairs of a linearized model's operator.

    Description
    -----------
    Holds the batched spectrum, eigenvectors and the assembled
    :class:`BlockSymbol` produced by :func:`symbolic_eigenpairs`. The
    spectrum and eigenvectors are indexed by the model's spectral mode
    grid (leading axes) followed by the ``m``-sized component axes;
    ``components`` names the row/eigenvector order.

    Parameters
    ----------
    omega : jax.Array
        Real frequencies, shape ``(*modes, m)``, sorted ascending per
        mode (``L q = i omega q``).
    q : jax.Array
        Eigenvectors, shape ``(*modes, m, m)``; ``q[..., :, j]`` is the
        M-orthonormal eigenvector for ``omega[..., j]``.
    components : tuple[str, ...]
        The prognostic component names (the ``m`` row order of ``q``).
    weights : tuple[float, ...]
        The energy-metric diagonal ``diag(M)`` in ``components`` order.
    operator : BlockSymbol
        The assembled per-mode operator ``L(k)`` (``P L P`` when the
        model carries a pressure constraint).
    """

    def __init__(
        self,
        omega: jax.Array,
        q: jax.Array,
        components: tuple[str, ...],
        weights: tuple[float, ...],
        operator: BlockSymbol,
    ) -> None:
        """Store the batched spectrum, eigenvectors and operator."""
        self.omega = omega
        self.q = q
        self.components = components
        self.weights = weights
        self.operator = operator

    # ================================================================
    #  Self-consistency diagnostics (validation helpers)
    # ================================================================
    def orthonormality_error(self) -> jax.Array:
        r"""
        Return the max deviation of :math:`q^{H} M q` from the identity.

        Description
        -----------
        The eigenvectors are M-orthonormal by construction; this is a
        residual on that property (a 0-d real array), near machine
        epsilon on a well-conditioned metric.

        Returns
        -------
        jax.Array
            ``max |q^H M q - I|`` over all modes (a 0-d array).
        """
        w = jnp.asarray(self.weights, dtype=dtype_comp())
        gram = jnp.einsum(
            "...ij,i,...ik->...jk", jnp.conj(self.q), w, self.q)
        identity = jnp.eye(len(self.components), dtype=gram.dtype)
        return jnp.max(jnp.abs(gram - identity))


def symbolic_eigenpairs(
    model: Model, *, at_time: float = 0.0,
) -> SymbolicEigenmodes:
    r"""
    Solve ``eigh(iML, M)`` for a linearized model's symbolic eigenpairs.

    Description
    -----------
    The Phase-H1 symbolic path: assembles ``L(k)`` as a
    :class:`BlockSymbol` from the model's linear-term blocks
    (:func:`fr.linear_blocks`), composes the Leray projector for a
    pressure-constrained model (nonhydro), reads the energy metric ``M``
    from :meth:`fr.EnergyMetric.from_model`, and solves the batched
    generalized Hermitian eigenproblem (reusing the H0 eigensolve).
    Restricted to constant-coefficient periodic models (the
    ``EnergyMetric`` / ``linear_blocks`` provides-implies-constancy
    gate); a variable-coefficient (beta-plane) or non-periodic model is
    declined upstream.

    Parameters
    ----------
    model : Model
        An assembled, Fourier-diagonalizable model (nonhydro or shallow
        water in iteration 1).
    at_time : float, optional
        Evaluation time for time-dependent parameters (default: 0.0).

    Returns
    -------
    SymbolicEigenmodes
        The per-mode spectrum, M-orthonormal eigenvectors and operator.
    """
    metric = EnergyMetric.from_model(model, at_time=at_time)
    prog = metric.component_names
    weights = _metric_weights(metric, prog)

    operator = _assemble_raw(model, prog, at_time)
    projector = _leray_projector(model, prog, at_time)
    if projector is not None:
        operator = projector @ operator @ projector

    omega, q = _generalized_eigh(operator.matrix, weights)
    return SymbolicEigenmodes(omega, q, prog, weights, operator)


# ================================================================
#  Raw operator assembly (per-block scalar symbols -> BlockSymbol)
# ================================================================
def _assemble_raw(
    model: Model, prog: tuple[str, ...], at_time: float,
) -> BlockSymbol:
    r"""
    Assemble ``L_raw(k)`` from the model's linear blocks.

    Description
    -----------
    Resolves each linear block to ``coeff . op.eigenvalues(grid,
    src_space)`` (a scalar :class:`Symbol` carrying the staggering
    phase), accumulates blocks sharing an ``(out, src)`` entry through
    the diagonal ``+``, and scatters the ``m x m`` grid into a
    :class:`BlockSymbol` tagged with the prognostic components' physical
    bare spaces (the interned tags the Leray ``@`` composition threads).

    Parameters
    ----------
    model : Model
        The assembled model.
    prog : tuple[str, ...]
        The prognostic component order (the ``L`` rows/columns).
    at_time : float
        The clock time freezing time-dependent coefficients.

    Returns
    -------
    BlockSymbol
        The per-mode raw operator ``(*modes, m, m)``.
    """
    grid = model.grid
    state = model.state
    index = {name: i for i, name in enumerate(prog)}
    grid_of_symbols: list[list[Symbol | None]] = [
        [None] * len(prog) for _ in prog]
    for block in linear_blocks(model, at_time=at_time):
        row, col = index[block.out], index[block.src]
        src_space = state[block.src].function_space.bare
        entry = block.coeff * block.op.eigenvalues(grid, src_space)
        current = grid_of_symbols[row][col]
        grid_of_symbols[row][col] = (
            entry if current is None else current + entry)
    spaces = _component_spaces(model, prog)
    return BlockSymbol.from_blocks(
        tuple(tuple(row) for row in grid_of_symbols), spaces, spaces)


def _component_spaces(
    model: Model, names: tuple[str, ...],
) -> tuple[SpaceLike, ...]:
    """Return the components' (interned) physical bare spaces."""
    return tuple(
        model.state[name].function_space.bare for name in names)


# ================================================================
#  The nonhydro Leray projector (the pressure Schur complement)
# ================================================================
def _leray_projector(
    model: Model, prog: tuple[str, ...], at_time: float,
) -> BlockSymbol | None:
    r"""
    Assemble the Leray projector ``P`` as a ``BlockSymbol``, or ``None``.

    Description
    -----------
    A pressure-constrained model (nonhydro — detected by the
    ``nonhydro.dsqr`` provider, the same by-name coupling
    ``EnergyMetric.from_model`` uses) eliminates the diagnostic pressure
    by :math:`P = I - W\,G\,(\nabla^2)^{-1} D` on the velocity
    ``(u, v, w)``, with ``b`` untouched. ``D`` (divergence) and ``G``
    (gradient) come from the grid operators' ``eigenvalues``; the
    ``dsqr``-weighted discrete Laplacian ``\nabla^2 = D W G`` is inverted
    through :meth:`Symbol.inverse` (its exact structural-zero test
    regularizing the ``k = 0`` gauge). The 3x3 velocity projector is
    assembled with the ``BlockSymbol`` matrix algebra, then embedded
    block-diagonally into the full ``m x m`` state (identity on ``b``).
    Returns ``None`` for an unconstrained model (shallow water).

    Parameters
    ----------
    model : Model
        The assembled model.
    prog : tuple[str, ...]
        The prognostic component order.
    at_time : float
        The clock time freezing ``dsqr``.

    Returns
    -------
    BlockSymbol | None
        The per-mode projector ``(*modes, m, m)``, or ``None``.
    """
    params = model.parameters
    if _DSQR not in params:
        return None
    dsqr = float(resolve_at(params[_DSQR], at_time))

    grid = model.grid
    vel_spaces = _component_spaces(model, _VELOCITY)
    p_space = model.state["p"].function_space.bare
    weight = tuple(
        1.0 / dsqr if name == _VERTICAL else 1.0 for name in _VELOCITY)

    div = Divergence().expand(vel_spaces, grid.dispatch).eigenvalues(
        grid, *vel_spaces)
    grad = Gradient().expand(p_space, grid.dispatch).eigenvalues(
        grid, p_space)
    laplace_inv = _weighted_laplace_inverse(div, grad, weight)

    identity = _diagonal_block(vel_spaces, (1.0,) * len(_VELOCITY))
    weight_block = _diagonal_block(vel_spaces, weight)
    correction = weight_block @ grad @ (laplace_inv @ div)
    p_velocity = identity - correction
    return _embed_velocity(
        p_velocity, _component_spaces(model, prog), prog)


def _weighted_laplace_inverse(
    div: BlockSymbol, grad: BlockSymbol, weight: tuple[float, ...],
) -> BlockSymbol:
    r"""
    Build ``(D W G)^{-1}`` as a ``1 x 1`` block via ``Symbol.inverse``.

    Description
    -----------
    The ``dsqr``-weighted discrete Laplacian ``D W G`` (the exact
    pressure Poisson eigenvalue, ``-(khat_h^2 + khat_z^2/dsqr)``) is a
    scalar per mode; wrapping it as a :class:`Symbol` on the pressure
    coefficient space and calling :meth:`Symbol.inverse` regularizes the
    ``k = 0`` nullspace through the exact structural-zero test (no
    floating tolerance), matching the running model's pressure solve.

    Parameters
    ----------
    div : BlockSymbol
        The divergence symbol (``1 x 3``, velocity -> pressure).
    grad : BlockSymbol
        The gradient symbol (``3 x 1``, pressure -> velocity).
    weight : tuple[float, ...]
        The nonhydrostatic velocity weights (``1, 1, 1/dsqr``).

    Returns
    -------
    BlockSymbol
        The ``1 x 1`` inverse Laplacian on the pressure space.
    """
    w = jnp.asarray(weight, dtype=dtype_comp())
    laplace = jnp.einsum(
        "...ak,k,...kb->...ab", div.data, w, grad.data)[..., 0, 0]
    p_space = div.out_spaces[0]
    inverse = Symbol(p_space, laplace).inverse()
    return BlockSymbol.from_blocks(
        ((inverse,),), (p_space,), (p_space,))


def _diagonal_block(
    spaces: tuple[SpaceLike, ...], diagonal: tuple[float, ...],
) -> BlockSymbol:
    """Build a diagonal ``BlockSymbol`` (constant per-component scale)."""
    size = len(spaces)
    grid_of_symbols = tuple(
        tuple(
            Symbol(spaces[i], jnp.asarray(diagonal[i])) if i == j
            else None
            for j in range(size))
        for i in range(size))
    return BlockSymbol.from_blocks(grid_of_symbols, spaces, spaces)


def _embed_velocity(
    velocity_block: BlockSymbol,
    prog_spaces: tuple[SpaceLike, ...],
    prog: tuple[str, ...],
) -> BlockSymbol:
    r"""
    Embed the velocity projector into the full state (identity on rest).

    Description
    -----------
    Scatters the ``len(velocity) x len(velocity)`` velocity projector
    into the ``m x m`` state matrix and sets the identity on every
    non-velocity component (``b``), so the projector leaves the
    buoyancy untouched — the block-diagonal ``P = P_vel ⊕ I`` structure.
    The full-state space tags (``prog_spaces``) are the same interned
    bare spaces ``L_raw`` carries, so the ``P @ L_raw @ P`` composition
    threads without a space clash.

    Parameters
    ----------
    velocity_block : BlockSymbol
        The assembled velocity projector.
    prog_spaces : tuple[SpaceLike, ...]
        The full prognostic components' bare spaces (row/column tags).
    prog : tuple[str, ...]
        The full prognostic component order.

    Returns
    -------
    BlockSymbol
        The full ``m x m`` projector.
    """
    block = velocity_block.data
    lead = block.shape[:-2]
    size = len(prog)
    data = jnp.zeros(
        (*lead, size, size), dtype=dtype_comp())
    vel_index = [prog.index(name) for name in _VELOCITY]
    for a, i in enumerate(vel_index):
        for b, j in enumerate(vel_index):
            data = data.at[..., i, j].set(block[..., a, b])
    for i in range(size):
        if i not in vel_index:
            data = data.at[..., i, i].set(1.0)
    return BlockSymbol(prog_spaces, prog_spaces, data)
