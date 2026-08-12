r"""
Streamfunction recovery: the BC-aware spectral Laplacian inverse.

Description
-----------
The shared elliptic inverse behind the vorticity-prescribed eddy
initial conditions of every model package. Given a field on a
staggered corner space it returns the field whose discrete negative
Laplacian along the requested axes reproduces it, using the grid's
own operator symbols, so the round trip is exact in the *discrete*
sense and not merely to truncation order.

The one non-obvious part is the boundary condition. A solid wall
forces the wall-normal velocity to vanish, and with a curl
:math:`(u, v) = (\partial_y \psi, -\partial_x \psi)` that reads
:math:`\partial_y \psi = 0` along an x-wall, i.e. the
streamfunction is *constant along each wall*: a Dirichlet condition
on :math:`\psi`, not a Neumann one. The tangential (free-slip or
no-slip) condition plays no part, since it constrains a second
derivative of :math:`\psi` and would overdetermine a second-order
elliptic problem; it is a property of the model's viscous closure,
not of the initial state.

Discretely that condition is already carried by the C-grid. The
wall-normal velocity of a walled axis lives on the ``Inner`` node
set (the n - 1 interior faces) with a ``BC.DIRICHLET`` tag, so the
corner space :math:`\psi` shares -- ``u``'s normal factor tensored
with ``v``'s -- is Dirichlet-``Inner`` on every walled horizontal
axis. That is exactly the DST-I origin, whose basis
:math:`\sin(k \pi (x - x_\min) / L)` vanishes at both walls by
construction. No wall DOF exists to be constrained, and the
inversion needs no retag on those axes at all: the walled
horizontal transform resolves directly.

What does need a retag is a **passive** bounded axis -- a rigid-lid
vertical the inversion does not differentiate along. Its corner
factor is the BC-free cell scalar (``Center`` or ``CellAvg``), which
grounds no trig transform, so :func:`spectral_sibling` tags it. Any
exact transform pair serves there, because the horizontal Laplacian
symbol does not depend on the vertical mode index, so the vertical
round trip is an identity. The tag choice is therefore free, and it
is spent on a constraint of the transform machinery: a trig family
instance is grid-bound over *all* bounded axes, so a
``Sine(x) x Cosine(z)`` product asks the sine transform for a DST
signature on a Neumann origin and raises. The sibling keeps a
single trig family across bounded axes, following whichever family
the active walled axes already commit to.

The gauge is the usual one. On a fully periodic axis set the
constant mode is a structural zero of the symbol and
``Symbol.inverse`` maps it to zero, so the recovered field is
mean-free and the reproduced vorticity is the prescribed one minus
its domain mean. A single walled axis removes the nullspace
entirely -- the sine basis carries no constant -- so there the
recovery is exact mode by mode.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.spatial.bc import BC
from fridom.spatial.spaces.average import AverageSpace
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet
from fridom.spatial.symbols import GridSymbols

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.spaces.function_space import FunctionSpace
    from fridom.spatial.spaces.tensor_product import SpaceLike


def _committed_family(
    factor: FunctionSpace, *, inverted: bool,
) -> BC | None:
    """
    Return the trig family a bounded factor commits to, or None.

    Description
    -----------
    A tagged factor commits to its own tag. An untagged factor on an
    inverted axis commits to Dirichlet, the wall condition on the
    recovered field. An untagged ``Inner`` factor commits to
    Dirichlet whatever its role, since the interior-face node set
    exists only because a Dirichlet condition removed the two wall
    DOFs and there is no Neumann ``Inner`` transform. A passive
    untagged factor is free.

    Parameters
    ----------
    factor : FunctionSpace
        A bounded (non-periodic) factor of the operand space.
    inverted : bool
        Whether the Laplacian differentiates along this factor's
        axis.

    Returns
    -------
    BC | None
        ``BC.DIRICHLET``, ``BC.NEUMANN``, or None when free.
    """
    components = factor.bc.components
    if not factor.bc.is_free:
        return (BC.DIRICHLET if BC.DIRICHLET in components
                else BC.NEUMANN)
    if inverted:
        return BC.DIRICHLET
    if (isinstance(factor, NodalSpace)
            and factor.node_set is NodeSet.INNER):
        return BC.DIRICHLET
    return None


def spectral_sibling(
    space: SpaceLike, *, axes: Iterable[str] = (),
) -> SpaceLike:
    r"""
    Return the trig-tagged structural sibling of a solve space.

    Description
    -----------
    The trig transform rows are keyed on BC-tagged origins, so every
    bounded factor of an operand must carry a tag before a walled
    transform resolves. This is the streamfunction analogue of the
    pressure solver's ``_neumann_sibling``, differing in three ways
    that the wall condition on :math:`\psi` forces.

    First, a factor that already carries a tag is left alone. The
    corner streamfunction is Dirichlet-``Inner`` on a walled
    horizontal axis (see the module docstring) and retagging it
    Neumann would both misstate the physics and resolve no
    transform, since no Neumann ``Inner`` origin exists.

    Second, an untagged bounded factor on an *inverted* axis is
    tagged Dirichlet, not Neumann. That is the wall condition
    :math:`\psi = 0` again, now reached for an operand the C-grid
    did not already tag (a cell-centred one, whose Dirichlet origin
    is the DST-II). Note that a cell-centred inversion axis costs a
    nullspace: the staggered difference annihilates the DST-II top
    mode, so that mode of the operand is dropped by the pseudo
    inverse. The corner (DST-I) space has no such mode.

    Third, the free factors are all tagged with the *same* family,
    taken from whichever family the committed factors use (Neumann
    when nothing commits, matching the pressure convention). A
    per-family transform instance is grid-bound over every bounded
    axis, so a mixed ``Sine x Cosine`` product is not resolvable;
    keeping one family sidesteps that. The tag of a passive axis is
    free anyway, because the inverted symbol does not read its mode
    index.

    Periodic factors and constant (broadcast) factors pass through,
    so on a fully periodic grid the sibling *is* the space itself
    (interned identity, the no-retag fast path).

    Parameters
    ----------
    space : SpaceLike
        The (bare) operand space.
    axes : Iterable[str], optional
        The coordinate names the Laplacian differentiates along; a
        bounded factor on one of them takes the Dirichlet wall
        condition (default: (), every bounded factor passive).

    Returns
    -------
    SpaceLike
        The interned sibling (``space`` itself when nothing needs a
        tag).

    Raises
    ------
    ValueError
        When bounded factors commit to different trig families.
    """
    inverted = set(axes)
    bounded = [factor for factor in space.factors
               if not getattr(factor.mesh, "periodic", True)
               and not factor.is_constant]
    committed = {
        family for family in (
            _committed_family(
                factor, inverted=factor.names[0] in inverted)
            for factor in bounded)
        if family is not None}
    if len(committed) > 1:
        raise ValueError(
            "a mixed Dirichlet x Neumann trig product is not "
            "resolvable: a per-family transform instance is bound "
            "over every bounded axis of the grid, so the sine "
            "transform would be asked for a signature on the "
            "Neumann origin. Hand the inversion an operand whose "
            "bounded factors agree on one family")
    family = committed.pop() if committed else BC.NEUMANN
    replacements: dict[str, FunctionSpace] = {}
    for factor in bounded:
        if not factor.bc.is_free:
            continue
        if isinstance(factor, NodalSpace):
            replacements[factor.names[0]] = factor.mesh.nodal(
                factor.node_set, bc=family)
        elif isinstance(factor, AverageSpace):
            replacements[factor.names[0]] = factor.mesh.average(
                type(factor), bc=family)
    if not replacements:
        return space
    return space.replace(**replacements)


def invert_negative_laplacian(
    field: ScalarField, *, axes: Iterable[str],
) -> ScalarField:
    r"""
    Solve :math:`-\nabla^2_{axes} g = f` spectrally, on any topology.

    Description
    -----------
    The elliptic inverse of the vorticity-prescribed eddies. The
    symbol is assembled from the grid's own staggered difference
    operators,

    .. math::
        \hat{k}^2 = \sum_{i \in axes}
            \left| \hat{D}_i \right|^2,

    so it is the exact eigenvalue of the discrete operator the
    caller will apply afterwards (the C-grid chain
    ``psi -> u, v -> zeta``), not its continuous approximation. The
    round trip is therefore exact to round-off at every resolution,
    with no truncation error of its own.

    The operand is retagged onto its :func:`spectral_sibling` for
    the transform and the solution retagged back, so the caller's
    spaces are preserved; on a fully periodic grid the sibling is
    the space itself and no retag happens.

    ``Symbol.inverse`` maps the structural zeros of
    :math:`\hat{k}^2` to zero. On a fully periodic axis set that is
    the constant mode, i.e. the zero-mean gauge, and the recovered
    field reproduces ``field`` only up to its domain mean. A walled
    axis contributes :math:`\hat{k}_i > 0` on every one of its sine
    modes, so the symbol has no zero at all and the recovery is
    exact -- the wall condition :math:`g = 0` has already fixed the
    gauge.

    Parameters
    ----------
    field : ScalarField
        The right-hand side, on a space the grid grounds a transform
        for (after tagging).
    axes : Iterable[str]
        The coordinate names the Laplacian differentiates along; the
        remaining axes ride along as passive broadcast dimensions.

    Returns
    -------
    ScalarField
        The solution on ``field``'s own function space.
    """
    axes = tuple(axes)
    grid = field.grid
    space = field.function_space.bare
    solve_space = spectral_sibling(space, axes=axes)
    operand = (field if solve_space is space
               else field.retag(solve_space))
    kit = GridSymbols(grid, {"f": solve_space})
    symbol = sum(kit.diff(axis, on="f").magnitude ** 2
                 for axis in axes)
    coeff = kit.forward("f")(operand)
    inverse = jnp.broadcast_to(
        symbol.inverse().data, coeff.data.shape)
    solution = kit.backward("f")(
        coeff.with_data(coeff.data * inverse)).real
    if solve_space is space:
        return solution
    return solution.retag(field)
