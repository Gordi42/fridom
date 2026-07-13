r"""Flux-form advection modules: centered, upwind-biased, and WENO.

Description
-----------
All three modules transport every ``ADVECTED`` component in flux form
``A(v, q) = -div(v q) = -sum_i d_i( interp(v_i) face(q) )``
(divergence-free velocity assumed), scaled by the Rossby number
``scaling.rossby`` (a defaulted reference, so the modules stay
Ro-ignorant — D2 reconciliation 4). The flux for axis ``i`` lives on
``q``'s control-volume face in direction ``i`` (``q`` toggled along
``i``); ``v_i`` is interpolated there with the registered (centered)
``interpolate`` row, multiplied with the scheme's face value of ``q``
(a genuine field-times-field product), and differenced back onto
``q``'s space. The schemes differ only in the face value:

- ``CenteredAdvection``: the registered centered interpolation
  (``q.to(flux_space)``).
- ``UpwindAdvection``: the full linear upwind-biased reconstruction
  (Shu 1998 uniform-mesh rows, the old stack's
  ``upwind_interpolation.py`` numerics), the stencil side selected by
  the sign of the face velocity.
- ``WENOAdvection``: the nonlinear WENO-JS reconstruction of the
  same odd orders (Jiang & Shu 1996; the old stack's
  ``weno_interpolation.py`` numerics via the framework kernel).

**Formal order of the biased schemes (read before choosing one)**:
the *reconstruction* converges at its design order (3rd / 5th), but
the *composite tendency* is formally **2nd order whenever the
advecting velocity varies along the flux axis** — on a periodic,
uniform grid, with exact face velocities, no boundary in sight. The
mechanism is the product rule, not the velocity interpolation: an FV
reconstruction row is a deconvolution, so the two-point difference of
face values is high-order only if the face value is the deconvolved
*flux* :math:`R(v q)`. These schemes form ``v_face * R(q)`` instead,
and the mismatch is the cross term :math:`\sim (h^2/24)\,2\,v'q'`,
which vanishes identically for :math:`v' = 0`. A constant advecting
velocity is therefore the only regime in which the design order is
observable (and the only one a design-order test can use); with a
varying velocity upwind-3, upwind-5 and weno-5 all measure rate 2
(pinned in ``tests/nonhydro2/test_advection.py``). What the biased
schemes are FOR is what they do deliver: dispersion control and the
ENO / non-oscillatory property at fronts — not asymptotic order.

This is a property of the C-grid flux form, not of FRIDOM: the same
algebra runs in Oceananigans, MITgcm, MOM6 and ROMS. Reconstructing
the flux :math:`v q` itself (the Shu-Osher FD form; Mishra,
Pares-Pulido & Pressel, arXiv:1905.13665 — the reference the
``WENOAdvection`` docstring already carries) restores the design
order, and is deliberately NOT taken: it would cost the exact-zero
wall flux (structural today through the Dirichlet-tagged flux space;
truncation-level after) and constancy preservation (a constant ``q``
gives :math:`q\,\nabla\cdot v = 0` exactly today; after, only if the
pressure projection enforced the same wide reconstructed divergence —
a different Poisson operator).

The biased face values are ordinary operator applications (the
module-private ``_BiasedFaceReconstruction`` wrapping the framework
WENO kernel machinery) and the flux-sign selection is the framework
``Where`` select, so the whole path is visible to the halo-accounting
trace: the modules declare **no** ``extra_halo``; negotiation picks
the widened stencil up from the trace.

**Prescribed background flow**: every module takes a ``background=``
mapping of velocity-component names to profiles (callables of
coordinates, or constants), sampled at each component's own
staggered nodes as AUXILIARY fields (the profile-sampling precedent
of ``MeridionalStratification`` / ``GaussianWaveMaker``). With a
background :math:`U` set, the module contributes TWO terms whose sum
telescopes to the module's own scheme at the full advecting velocity
:math:`U + \mathrm{Ro}\,u'`:

- ``background_advection`` (``linear=True``): a linear
  discretization :math:`L(q) = S_\mathrm{lin}(U, q)` of transport by
  :math:`U` — the centered flux for ``CenteredAdvection``, and for
  the biased schemes the **linear** upwind row of the same order
  (WENO's optimal-weight smooth-limit row: a WENO weighting of a
  linear operator is not linear in the state and must never carry
  ``linear=True``), the stencil side selected by the sign of the
  static background face velocity — a state-independent mask, so the
  term is exactly linear in the state and ``fr.model.linearize`` keeps it.
- ``advection`` (nonlinear): :math:`N(u', q) =
  S_\mathrm{full}(U + \mathrm{Ro}\,u', q) - S_\mathrm{lin}(U, q)`,
  the module's own scheme at the full velocity minus the linear
  piece.

**Scaling convention (deliberate)**: the full advecting velocity is
:math:`U + \mathrm{Ro}\,u'` and there is NO outer Rossby factor on
the combined tendency — :math:`U` is an O(1) velocity of the scaled
equations, consistent with the other linear modules (Coriolis,
stratification). With ``background=None`` this reduces exactly to
the single Rossby-scaled term (velocity enters the flux linearly and
upwind selection is invariant under positive scaling, so
:math:`S(\mathrm{Ro}\,u', q) = \mathrm{Ro}\,S(u', q)`), and the
``background=None`` code path is literally the pre-background one.
The OLD stack differed: ``advect_state`` multiplied the whole
advecting velocity (background included) by the nonlinear scaling
factor (``mset.tendencies.advection.scaling = rossby_number``), i.e.
``Ro * S(u' + U, q)`` — old users passed pre-scaled backgrounds.

**Mapped grids (centered scheme, stage C4)**: on a grid whose
``CoordinateMapping`` declares a mapped column ``m = M(b, params)``
(terrain-following ``zp = z * H(x, y)``, boundary-fitted
``yp = y * Y_N(x)``), the honest transport is the **physical** flux
divergence :math:`-\sum_i \partial F_i/\partial x_i|_{\rm phys}` of
the physical velocity fluxes :math:`F_i = v_i\,q`. The module
assembles it per axis from the sketch-4.4 pieces at the module level
(the flux spaces carry wall tags the registry-seeded
``physical_diff`` builder cannot hop across):
``d/dx_i|_phys F_i = d_i F_i - (Z_i/J) interp(d_b F_i)`` on the
coupled axes and ``(1/J) d_b F_b`` along the column, every metric
coefficient derived through ``grid.metric`` **at application** with
the CURRENT dynamic parameter fields threaded through ``params=``
(the ``MovingGeometry`` state; static mapped grids keep the
declaration defaults). This is consistent (2nd order) but not the
J-weighted telescoping conservative form — the mapped pressure
operator keeps that exactness where it is load-bearing (SPD). On a
FLAT grid the divergence helper is literally the pre-C4 expression
``flux.diff(axis).retag(q)`` — zero behavior change. The biased
schemes (``UpwindAdvection``/``WENOAdvection``) reject mapped
geometry at bind with a taught error — **both** surfaces: a mapped
column (``CoordinateMapping``) and a stretched mesh factor
(``MappedIntervalMesh``, whose ``column_corrections`` are empty).
Their order-wide windows are uniform-offset (computational-
coordinate) rows: divided by a two-point measure they stay
consistent but drop to 2nd order (measured: upwind-5 and weno-5 both
5 -> 2 on a wavy-stretched mesh), so they refuse rather than
silently under-deliver — a mapped-aware high-order reconstruction is
future work.

**Walled grids**: all three schemes support bounded mesh factors.
``CenteredAdvection`` does so structurally (next paragraph); the
biased schemes add the **graded near-wall closure** on top of exactly
the same flux bookkeeping. At bind on a walled grid their face
kernels swap to their ``boundary="graded"`` variants
(``_BiasedFaceReconstruction`` / ``_CenteredFaceInterpolation``,
driving the shared ``spatial.operators.graded`` ladder — the nodal
twin of the average family's ``Fallback``): the wide interior windows
stay, and at the ``K`` faces adjacent to each wall progressively
narrower interior-only stencils take over (orders ``min(order, 2d-1)``
at distance ``d``, down to the ``wall=`` bottom rung), so **no
exterior value is ever read** (R1, ``design/plans/active/
boundary_plan.md``). The wall itself stays impermeable *exactly*, not
to truncation: the flux space still adopts the wall-normal velocity's
Dirichlet tag, so the wall flux is a structural zero, and the reduced
rows only ever produce the interior faces. The price is accuracy, not
correctness: the interior keeps the *reconstruction's* design order
(the tendency's own ceiling is the 2nd order above) while the ``K``
near-wall faces drop to their rung's, so the global rate on a walled
axis is the near-wall rung's rate — which is why the bottom rung is
the user's choice: ``wall="upwind1"`` (default) is monotone and
globally 1st order, ``wall="centered2"`` is globally 2nd order but
undissipative on the wall-adjacent face (`UpwindAdvection`). Each
walled axis needs at least ``order + 1`` cells (taught error at bind).
The uniform-mesh refusal is untouched — mapped/stretched factors are
still rejected at bind.

**Walled grids (centered scheme)**: ``CenteredAdvection`` supports
bounded mesh factors (channel walls, rigid lids, and their
combinations) with **no boundary-condition physics choice** — no
free-slip/no-slip knob. The argument is structural: every advective
flux through a wall face carries the wall-normal velocity as a
factor, and that value is an exact zero by impermeability (the
wall-normal velocity lives on the interior ``Inner`` faces with the
Dirichlet wall tag; the wall face is a boundary condition, not a
DOF). The wall-specific work is pure bookkeeping: along a walled
axis the flux space adopts the wall-normal velocity's Dirichlet tag
(a BC-sibling substitution), so the flux divergence closes with the
exact-zero wall flux, and the divergence is retagged back onto the
advected component's space — both identities on periodic axes, so
the periodic path is reproduced bit for bit. The two-point centered
stencils read no beyond-wall values at all: the only wall read is
the wall-normal velocity's own interpolation onto the centers,
whose Dirichlet fill *is* the impermeability zero. A genuine
free-slip choice enters only for PV/vector-invariant schemes (the
shallow-water Sadourny module) and viscous closures — and, notably,
**not** for the biased schemes either: their graded closure needs no
boundary-condition physics choice, because it reads no exterior value
to begin with. With a background flow, ``bind`` additionally
validates that the wall-normal background component vanishes on its
walls (the sampled field is structurally impermeable, so the check
runs on the user's input — the shallow-water background precedent).
"""
from __future__ import annotations

import inspect
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Literal, final

import numpy as np

import fridom as fr
from fridom.model.modules.moving_geometry import mapping_params
from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.scalar_field import (
    _bc_siblings,  # the BC-sibling seam of retag/.to
)
from fridom.spatial.operators.base import (
    OperatorRequirements,
    SeparableOperator,
)
from fridom.spatial.operators.graded import (
    WALL_RUNGS,
    Rung,
    RungSpec,
    apply_graded_walls,
    biased_offset,
    biased_specs,
    centered_ladder,
    centered_offset,
    min_cells,
    spec_offset,
)
from fridom.spatial.operators.interned import interned
from fridom.spatial.operators.reconstruct import (
    apply_fv_staggered,
)
from fridom.spatial.operators.select import Where
from fridom.spatial.operators.staggering import (
    mapped_factor,
    mapped_mesh,
    mapped_order_hint,
)
from fridom.spatial.operators.weno import (
    _shu_row,  # the exact-rational coefficient seam
    weno_reconstruct,
    weno_tables,
)
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from jax import Array

    from fridom.model.context import StepContext
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.operators.base import FieldLike
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )

#: the biased-reconstruction weightings of the module family
_WEIGHTINGS = ("linear", "weno")

#: the odd formal orders grounded by the framework WENO tables
_SUPPORTED_ORDERS = (3, 5)

#: constructor ``boundary`` variants of the module-private nodal
#: kernels (the ``WenoReconstruction`` vocabulary, decision R2):
#: "none" is the periodic-only kernel (today's exact behavior),
#: "graded" adds the bounded-legal graded near-wall closure
_BOUNDARY_MODES = ("none", "graded")

#: the velocity components' staggering axes (the nh C-grid)
_VELOCITY_AXES = {"u": "x", "v": "y", "w": "z"}

#: wall-normal background components must vanish at the wall to this
#: relative tolerance (impermeability; the shallow-water precedent)
_WALL_TOL = 1e-12


def _check_background(
    background: Mapping[str, Callable | float] | None,
) -> dict[str, Callable | float]:
    """
    Normalize and locally validate the ``background=`` mapping.

    Description
    -----------
    Keys must name nh C-grid velocity components (``u``/``v``/``w``);
    values are coordinate callables (parameters named after grid
    coordinates, e.g. ``lambda y, z: ...``) or real constants.
    Missing components are zero. ``None`` and the empty mapping both
    mean "no background".

    Parameters
    ----------
    background : Mapping[str, Callable | float] | None
        The user-facing constructor option.

    Returns
    -------
    dict[str, Callable | float]
        The normalized mapping (empty for no background).
    """
    if background is None:
        return {}
    background = dict(background)
    unknown = sorted(set(background) - set(_VELOCITY_AXES))
    if unknown:
        raise ValueError(
            f"background flow is keyed by the nh velocity "
            f"components {tuple(_VELOCITY_AXES)}, got the unknown "
            f"key(s) {unknown}")
    for name, value in background.items():
        if not callable(value) and not isinstance(value, int | float):
            raise TypeError(
                f"background[{name!r}] must be a coordinate "
                f"callable (e.g. lambda y, z: ...) or a constant, "
                f"got {value!r}")
    return background


def _sample_profile(
    grid: object, space: object, profile: Callable, name: str,
) -> ScalarField:
    """
    Sample a background profile at a space's own node positions.

    Description
    -----------
    The profile-sampling seam of the background flow (the
    ``sample_gaussian_mask`` / ``MeridionalStratification``
    precedent): the ``init`` signature is stamped with the space's
    non-constant coordinate names and the user callable is fed
    exactly the coordinates its own signature declares (constant
    along the rest).

    Parameters
    ----------
    grid : fr.spatial.Grid
        The grid to materialize on.
    space : SpaceLike
        The target function space (the velocity component's space).
    profile : Callable
        The user profile; parameters must name grid coordinates.
    name : str
        Metadata name of the sample field.

    Returns
    -------
    ScalarField
        The sampled background profile.
    """
    names = tuple(
        coordinate for factor in space.factors
        if not isinstance(factor, ConstantSpace)
        for coordinate in factor.names)
    wanted = tuple(inspect.signature(profile).parameters)
    unknown = sorted(set(wanted) - set(names))
    if unknown:
        raise ValueError(
            f"the background profile {name!r} names the "
            f"coordinate(s) {unknown}, which the grid does not "
            f"have (coordinates: {tuple(names)})")

    def init(**coords: Array) -> Array:
        return profile(**{key: coords[key] for key in wanted})

    init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(
            coordinate, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for coordinate in names])
    return grid.create_field(space, init=init, name=name)


def _profile_default(name: str) -> Callable:
    """
    Build the unbound owner-method default for one profile sample.

    Description
    -----------
    The declaration ``default=`` owner-method form (first parameter
    named ``self``), closing over the component name only: the
    profile itself is read off the live module at materialization.

    Parameters
    ----------
    name : str
        The velocity component name (the ``background`` key).

    Returns
    -------
    Callable
        The unbound ``(self, grid, space)`` default.
    """
    def _default(self, grid, space):  # noqa: ANN001, ANN202
        return _sample_profile(grid, space, self._background[name],
                               name=f"background_{name}")
    return _default


def _wall_profile_values(
    grid: object, space: object, fn: Callable, axis: str,
    extent: tuple[float, float],
) -> float:
    """
    Largest ``|fn|`` over both walls of ``axis`` (host-side).

    Description
    -----------
    Evaluates the background callable at the two wall positions of
    the bounded ``axis``, gridded over the tangential evaluation
    nodes it names (the shallow-water wall-check precedent, lifted
    to any number of tangential coordinates via a meshgrid).

    Parameters
    ----------
    grid : fr.spatial.Grid
        The grid supplying the tangential evaluation nodes.
    space : SpaceLike
        The background sample's own space (node positions).
    fn : Callable
        The user profile; parameters name grid coordinates.
    axis : str
        The bounded (wall-normal) coordinate.
    extent : tuple[float, float]
        The wall positions of ``axis``.

    Returns
    -------
    float
        The largest absolute wall value of the profile.
    """
    wanted = tuple(inspect.signature(fn).parameters)
    others = [name for name in wanted if name != axis]
    worst = 0.0
    for wall in extent:
        arrays = [
            np.asarray(grid.evaluation_nodes(space, name).data)
            .ravel() for name in others]
        coords: dict[str, object] = dict(zip(
            others, np.meshgrid(*arrays, indexing="ij"),
            strict=True))
        if axis in wanted:
            coords[axis] = wall
        worst = max(worst, float(np.max(np.abs(
            np.asarray(fn(**coords))))))
    return worst


# ================================================================
#  The linear (optimal-weight) upwind rows
# ================================================================
@cache
def _linear_row(
    order: int, bias: Literal["left", "right"],
) -> tuple[float, ...]:
    """
    Full biased reconstruction row of one linear upwind kernel.

    Description
    -----------
    The optimal-weight combination of the WENO candidate stencils —
    exactly the full ``order``-cell Shu reconstruction row at the
    biased face (the linear-weight consistency identity), i.e. the
    old stack's ``upwind_interpolation.py`` coefficients. Combining
    the cached framework tables keeps the module free of its own
    coefficient computation.

    Parameters
    ----------
    order : int
        The odd formal order (3 or 5).
    bias : Literal["left", "right"]
        The upwind bias side.

    Returns
    -------
    tuple[float, ...]
        The ``order`` static reconstruction coefficients.
    """
    tables = weno_tables(order, bias)
    row = [0.0] * order
    for m, offset in enumerate(tables.offsets):
        for i, coeff in enumerate(tables.coeffs[m]):
            row[offset + i] += tables.optimal[m] * coeff
    return tuple(row)


def _weighted_windows(
    arr: Array, axis: int, row: tuple[float, ...],
) -> Array:
    """
    Static-row weighted sum over the stencil windows (fused).

    Description
    -----------
    The rules-3.5 slice-window kernel: one fused weighted sum over
    the ``len(row)`` stencil windows with static coefficients; the
    axis shrinks by ``len(row) - 1``.

    Parameters
    ----------
    arr : Array
        The input values (halo-extended by the caller).
    axis : int
        The stencil axis; may be negative.
    row : tuple[float, ...]
        The static stencil coefficients.

    Returns
    -------
    Array
        The fused weighted sum.
    """
    out_len = arr.shape[axis] - len(row) + 1
    index: list[slice] = [slice(None)] * arr.ndim
    total = None
    for offset, weight in enumerate(row):
        index[axis] = slice(offset, offset + out_len)
        term = weight * arr[tuple(index)]
        total = term if total is None else total + term
    return total


@cache
def _centered_row(size: int) -> tuple[float, ...]:
    """
    Symmetric even-size interpolation row at the middle interface.

    Description
    -----------
    The cell-average Shu row reconstructing at the window midpoint
    (the old stack's ``PolynomialInterpolation`` coefficients with
    the default ``method="cell_average"``) — the velocity-to-flux
    interpolation the old advection schemes coupled to their order.

    Parameters
    ----------
    size : int
        The even stencil size (2 or 4 for orders 3 and 5).

    Returns
    -------
    tuple[float, ...]
        The ``size`` static interpolation coefficients.
    """
    return tuple(float(c) for c in _shu_row(size, size // 2))


# ================================================================
#  The face-value operators (module-private)
# ================================================================
def _face_codomain(
    domain: FunctionSpace,
    label: str,
    boundary: Literal["none", "graded"] = "none",
) -> FunctionSpace:
    """
    Shared C-grid face signature of the module-private kernels.

    Description
    -----------
    ``Center -> Right`` and ``Right -> Center`` on periodic real nodal
    factors of a **uniform** mesh — the two C-grid flux positions of
    the flux-form advection modules — and, for ``boundary="graded"``,
    their bounded twins ``Center -> Inner`` and ``Inner -> Center``.
    The bounded pair is exactly the ``graded`` cell frame: a
    ``Center`` operand's DOFs are the lattice cells (``shift = 0``, a
    BC-free bounded space, nothing outside its true DOFs exists — R1);
    an ``Inner`` operand's DOFs are the interior faces and the two wall
    faces are its homogeneous-Dirichlet boundary values (``shift = 1``).
    Consequently the ``Inner`` direction is grounded on a Dirichlet tag
    only — the impermeable wall-normal velocity of the nh C-grid; any
    other bounded ``Inner`` tag raises rather than inventing a wall
    value.

    Everything else (average spaces, stretched axes, complex scalars,
    and — under ``boundary="none"`` — any bounded axis) raises.

    Parameters
    ----------
    domain : FunctionSpace
        The bare 1D factor space.
    label : str
        The raising operator's name (error attribution).
    boundary : Literal["none", "graded"], optional
        The kernel's boundary variant (default: "none").

    Returns
    -------
    FunctionSpace
        The flux-position codomain factor.
    """
    if (not isinstance(domain, NodalSpace)
            or domain.scalars is Scalars.COMPLEX):
        raise SpaceMismatchError(
            f"{label} covers real nodal C-grid factors only, got "
            f"{domain!r}", left=domain, operation="reconstruct")
    if mapped_factor(domain):
        raise SpaceMismatchError(
            f"{label} is uniform-mesh only (the biased advection "
            "modules reject stretched meshes at bind) — "
            + mapped_order_hint(
                "the biased face reconstruction rows and their "
                "order-coupled velocity interpolation")
            + f", got {domain!r}",
            left=domain, operation="reconstruct")
    mesh = domain.mesh
    if mesh.periodic:
        if domain.node_set is NodeSet.CENTER:
            return mesh.right
        if domain.node_set is NodeSet.RIGHT:
            return mesh.center
        raise SpaceMismatchError(
            f"{label} maps Center -> Right and Right -> Center (the "
            f"C-grid flux positions), got {domain!r}",
            left=domain, operation="reconstruct")
    if boundary != "graded":
        raise SpaceMismatchError(
            f"{label} is periodic-only in its boundary='none' "
            "variant; the bounded signature is the graded near-wall "
            f"closure (boundary='graded'), got {domain!r}",
            left=domain, operation="reconstruct")
    if domain.node_set is NodeSet.CENTER:
        return mesh.inner
    if domain.node_set is NodeSet.INNER:
        if any(kind is not BC.DIRICHLET
               for kind in domain.bc.components):
            raise SpaceMismatchError(
                f"the graded {label} reconstructs a bounded Inner "
                "factor onto the cell centers only when its wall "
                "faces are homogeneous-Dirichlet boundary values (the "
                "impermeable wall-normal velocity): the near-wall "
                "windows reach the wall face and read its exact zero, "
                f"and no other tag defines one, got {domain!r}",
                left=domain, operation="reconstruct")
        return mesh.center
    raise SpaceMismatchError(
        f"the graded {label} maps Center -> Inner and Inner -> Center "
        f"(the bounded C-grid flux positions), got {domain!r}",
        left=domain, operation="reconstruct")


def _wall_shift(domain: FunctionSpace) -> int:
    """
    Cell-frame shift of a nodal C-grid factor (``graded`` vocabulary).

    Description
    -----------
    1 on the dual (face-staggered) directions — ``Right -> Center``
    (periodic) and ``Inner -> Center`` (bounded) — whose lattice cells
    are the mesh faces, so the operand's true DOFs start one cell in;
    0 on the primal ``Center -> Right`` / ``Center -> Inner``
    directions, whose lattice cells are the operand's own DOFs. It is
    both the window-alignment shift and (on a bounded axis) the count
    of wall cells the graded ladder synthesizes per side.

    Parameters
    ----------
    domain : FunctionSpace
        The bare 1D factor space.

    Returns
    -------
    int
        The cell-frame shift (0 or 1).
    """
    return 1 if domain.node_set in (
        NodeSet.RIGHT, NodeSet.INNER) else 0


@final
@interned
class _CenteredFaceInterpolation(SeparableOperator):

    """
    Even-size symmetric face interpolation (module-private).

    Description
    -----------
    The velocity-to-flux-face interpolation of the biased advection
    schemes, order-coupled as in the old stack (``order - 1``
    centered points): one fused weighted sum with the static
    `_centered_row` coefficients, midpoint-aligned by the FV window
    calculus. At size 2 it coincides with the registered default
    two-point mean.

    ``boundary="graded"`` grows the bounded signature (the walled
    advection path): the interior pass is unchanged and the ``K =
    size // 2 - 1 + shift`` faces adjacent to each wall are rebuilt
    from the graded ladder ``min(size, 2 * d)`` (module ``graded``),
    bottoming out at the two-point mean. On the primal direction
    (``Center -> Inner``, ``shift = 0``) the two-point rung already
    reads interior DOFs only, so ``K = 0`` at ``size = 2`` and the
    walled two-point interpolation is literally the periodic kernel.

    Parameters
    ----------
    size : int
        The even stencil size (2 or 4 for orders 3 and 5).
    boundary : Literal["none", "graded"], optional
        The boundary variant (default: "none", periodic-only).
    """

    dispatch_kind: ClassVar[str | None] = None

    def __init__(
        self,
        size: int,
        boundary: Literal["none", "graded"] = "none",
    ) -> None:
        """Validate the even stencil size and the boundary variant."""
        if size not in (2, 4):
            raise ValueError(
                f"the advective velocity interpolation grounds the "
                f"even stencil sizes (2, 4), got {size}")
        if boundary not in _BOUNDARY_MODES:
            raise ValueError(
                f"boundary must be one of {_BOUNDARY_MODES}, got "
                f"{boundary!r}")
        self._size = size
        self._boundary = boundary

    def _intern_key(self) -> tuple:
        """Structural key: the stencil size and boundary variant (D6)."""
        return (self._size, self._boundary)

    @property
    def size(self) -> int:
        """The even stencil size."""
        return self._size

    @property
    def boundary(self) -> Literal["none", "graded"]:
        """The boundary variant: "none" or "graded"."""
        return self._boundary

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """Resolve the C-grid face codomain (shared signature).

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D factor space.

        Returns
        -------
        FunctionSpace
            The flux-position codomain factor.
        """
        return _face_codomain(domain, type(self).__name__,
                              self._boundary)

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — fixed by the size
    ) -> OperatorRequirements:
        """Declare halo = size // 2, layout "any".

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        return OperatorRequirements(halo=self._size // 2)

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """Interpolate along ``axis`` (midpoint-aligned window).

        Description
        -----------
        The interior pass is the midpoint-aligned fused row; on a
        bounded axis (``boundary="graded"`` only — the plain variant
        has no bounded signature) the ``K`` wall faces per side are
        then overwritten by the graded ladder, which reads interior
        DOFs and the exact-zero Dirichlet wall values only.

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The interpolated field (metadata kept: same quantity).
        """
        size = self._size
        row = _centered_row(size)

        def kernel(arr: Array, axis_index: int) -> Array:
            return _weighted_windows(arr, axis_index, row)

        interior = apply_fv_staggered(self, f, axis, size, kernel,
                                      metadata=f.metadata)
        domain = f.function_space.bare.factor(axis)
        if self._boundary == "none" or domain.mesh.periodic:
            return interior
        shift = _wall_shift(domain)
        rungs = tuple(
            Rung(width, centered_offset(width),
                 _centered_kernel(width))
            for width in centered_ladder(size, shift))
        return apply_graded_walls(f, axis, interior, rungs, shift)


def _centered_kernel(size: int) -> Callable[[Array, int], Array]:
    """
    Array kernel of one centered graded rung (a fused static row).

    Parameters
    ----------
    size : int
        The rung's even window width.

    Returns
    -------
    Callable[[Array, int], Array]
        The ``(window, axis_index) -> face_value`` kernel.
    """
    row = _centered_row(size)

    def kernel(arr: Array, axis_index: int) -> Array:
        return _weighted_windows(arr, axis_index, row)

    return kernel


def _biased_kernel(
    order: int,
    bias: Literal["left", "right"],
    weighting: Literal["linear", "weno"],
) -> Callable[[Array, int], Array]:
    """
    Array kernel of one biased rung (or of the interior pass).

    Description
    -----------
    Order 1 is the single upwind cell (the unit-coefficient Shu row),
    so its kernel is the identity on the size-1 window — and it is the
    same row under either weighting, which is why the wall-adjacent
    rung of a WENO ladder is an ordinary 1st-order upwind value.

    Parameters
    ----------
    order : int
        The rung's odd formal order (1, or a table-grounded 3 / 5).
    bias : Literal["left", "right"]
        The upwind bias side.
    weighting : Literal["linear", "weno"]
        The stencil weighting of the holding scheme.

    Returns
    -------
    Callable[[Array, int], Array]
        The ``(window, axis_index) -> face_value`` kernel.
    """
    if order == 1:
        return lambda arr, _axis: arr
    if weighting == "weno":
        def kernel(arr: Array, axis_index: int) -> Array:
            return weno_reconstruct(arr, axis_index, order=order,
                                    bias=bias)
        return kernel
    row = _linear_row(order, bias)

    def linear(arr: Array, axis_index: int) -> Array:
        return _weighted_windows(arr, axis_index, row)

    return linear


def _rung_kernel(
    spec: RungSpec,
    bias: Literal["left", "right"],
    weighting: Literal["linear", "weno"],
) -> Callable[[Array, int], Array]:
    """
    Array kernel of one graded rung spec (the ``wall=`` seam).

    Description
    -----------
    A "biased" spec carries the odd formal order of its rung (the
    weighted upwind row); a "centered" spec is the ``wall="centered2"``
    bottom rung — the symmetric even-size row, identical under both
    biases, so the module's ``Where`` select returns it whatever the
    sign of the face velocity (the wall-adjacent face loses its upwind
    bias, which is exactly the trade the option offers).

    Parameters
    ----------
    spec : RungSpec
        The ladder rung spec.
    bias : Literal["left", "right"]
        The upwind bias side (inert on a centered rung).
    weighting : Literal["linear", "weno"]
        The stencil weighting of the holding scheme.

    Returns
    -------
    Callable[[Array, int], Array]
        The ``(window, axis_index) -> face_value`` kernel.
    """
    if spec.family == "centered":
        return _centered_kernel(spec.width)
    return _biased_kernel(spec.width, bias, weighting)


@final
@interned
class _BiasedFaceReconstruction(SeparableOperator):

    """
    Upwind-biased nodal face reconstruction (module-private).

    Description
    -----------
    The nodal C-grid twin of the framework's average-family biased
    reconstructions (old-stack parity: point values are treated as
    cell averages, the ``method="cell_average"`` convention of
    ``upwind_interpolation.py`` / ``weno_interpolation.py``): a
    ``Center`` quantity reconstructs onto its right faces
    (``Center -> Right``) and a face-staggered quantity onto the
    cell centers (``Right -> Center``, the dual window — the
    self-advection path of a velocity component). Held directly by
    the advection modules as a left/right pair; never registered
    under a dispatch kind (the sign selection is the ``Where``
    select in the module).

    ``boundary="graded"`` grows the bounded signature — the nodal twin
    of the average family's ``Fallback`` (both drive the shared
    ``operators.graded`` ladder): the wide interior pass is unchanged
    and the ``K = order // 2 + shift`` output faces adjacent to each
    wall are rebuilt from progressively narrower interior-only
    stencils (orders ``min(order, 2*d - 1)`` at distance ``d``,
    bottoming out at the ``wall=`` rung). It reads **no
    exterior value**: on the primal direction (``Center -> Inner``,
    ``shift = 0``) the operand is a BC-free bounded space and the
    ladder stays inside its true DOFs; on the dual direction
    (``Inner -> Center``, ``shift = 1``) the two wall faces are the
    operand's homogeneous-Dirichlet boundary values and the ladder
    synthesizes them as exact zeros rather than reading a ghost slot.
    On a periodic factor the graded variant returns the interior pass
    untouched, so it is bitwise the ``boundary="none"`` kernel there.

    Parameters
    ----------
    order : int
        The odd formal order; the framework tables ground 3 and 5.
    bias : Literal["left", "right"]
        The upwind bias side of the reconstruction.
    weighting : Literal["linear", "weno"]
        "linear" applies the full optimal-weight row (the classic
        linear upwind scheme); "weno" applies the nonlinear WENO-JS
        weighting of the same window.
    boundary : Literal["none", "graded"]
        "none" is the periodic-only kernel; "graded" grows the bounded
        signature by replacing the ``K`` faces adjacent to each wall
        with the graded ladder (see the class Description).
    wall : Literal["upwind1", "centered2"], optional
        The ladder's bottom (wall-adjacent) rung under
        ``boundary="graded"``; inert on the plain kernel (default:
        "upwind1" — the 1st-order upwind cell; "centered2" is the
        two-point mean, accurate but undissipative, see
        ``operators.graded``).
    """

    dispatch_kind: ClassVar[str | None] = None

    def __init__(
        self,
        order: int,
        bias: Literal["left", "right"],
        weighting: Literal["linear", "weno"],
        boundary: Literal["none", "graded"] = "none",
        wall: Literal["upwind1", "centered2"] = "upwind1",
    ) -> None:
        """Validate through the framework tables and store."""
        if weighting not in _WEIGHTINGS:
            raise ValueError(
                f"weighting must be one of {_WEIGHTINGS}, got "
                f"{weighting!r}")
        if boundary not in _BOUNDARY_MODES:
            raise ValueError(
                f"boundary must be one of {_BOUNDARY_MODES}, got "
                f"{boundary!r}")
        if wall not in WALL_RUNGS:
            raise ValueError(
                f"wall must be one of {WALL_RUNGS}, got {wall!r}")
        weno_tables(order, bias)  # validates order and bias
        self._order = order
        self._bias = bias
        self._weighting = weighting
        self._boundary = boundary
        self._wall = wall

    def _intern_key(self) -> tuple:
        """Structural key: order, bias, weighting, boundary, wall (D6)."""
        return (self._order, self._bias, self._weighting,
                self._boundary, self._wall)

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def order(self) -> int:
        """Formal order of the biased reconstruction."""
        return self._order

    @property
    def bias(self) -> Literal["left", "right"]:
        """Upwind bias side of the reconstruction."""
        return self._bias

    @property
    def weighting(self) -> Literal["linear", "weno"]:
        """The stencil weighting: "linear" or "weno"."""
        return self._weighting

    @property
    def boundary(self) -> Literal["none", "graded"]:
        """The boundary variant: "none" or "graded"."""
        return self._boundary

    @property
    def wall(self) -> Literal["upwind1", "centered2"]:
        """The ladder's bottom rung: "upwind1" or "centered2"."""
        return self._wall

    # ------------------------------------------------------------
    #  Signature and requirements
    # ------------------------------------------------------------
    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """Resolve the C-grid face codomain (shared signature).

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D factor space.

        Returns
        -------
        FunctionSpace
            The flux-position codomain factor.
        """
        return _face_codomain(domain, type(self).__name__,
                              self._boundary)

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — fixed by the order
    ) -> OperatorRequirements:
        """
        Declare halo = order // 2 + 1, layout "any".

        Description
        -----------
        The uniform declaration over both biases and both node-set
        directions: the widest window reach beyond the output slot
        is ``order // 2 + 1`` cells.

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        return OperatorRequirements(halo=self._order // 2 + 1)

    # ------------------------------------------------------------
    #  Kernel application (biased window alignment)
    # ------------------------------------------------------------
    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Reconstruct along ``axis`` (biased window-aligned kernel).

        Description
        -----------
        The WENO window alignment (kernel output ``t`` lands on the
        right face of window cell ``order // 2`` for the left bias,
        ``order // 2 - 1`` for the right bias), shifted by one slot
        on the dual ``Right -> Center`` / ``Inner -> Center``
        directions: the right face of the dual cell around face ``j``
        is center ``j + 1`` in the shared storage frame — exactly the
        ``graded`` cell-frame shift. On a bounded axis the graded
        variant then overwrites the ``K`` wall faces per side from the
        ladder (interior DOFs and the exact-zero Dirichlet wall values
        only).

        Parameters
        ----------
        f : FieldLike
            The operand field (storage-shaped ``_data``).
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The reconstructed field (metadata kept: same quantity).
        """
        order = self._order
        bias = self._bias
        weighting = self._weighting
        domain = f.function_space.bare.factor(axis)
        shift = _wall_shift(domain)
        m0 = biased_offset(order, bias) + shift
        kernel = _biased_kernel(order, bias, weighting)
        interior = apply_fv_staggered(self, f, axis, order, kernel,
                                      metadata=f.metadata, align=m0)
        if self._boundary == "none" or domain.mesh.periodic:
            return interior
        rungs = tuple(
            Rung(spec.width, spec_offset(spec, bias),
                 _rung_kernel(spec, bias, weighting))
            for spec in biased_specs(order, shift, self._wall))
        return apply_graded_walls(f, axis, interior, rungs, shift)


# ================================================================
#  The shared flux-form scaffolding (module-private)
# ================================================================
class _FluxFormAdvection(fr.model.Module):

    r"""
    Shared flux-form transport of every ADVECTED component.

    Description
    -----------
    The private scaffolding of the advection family (never exported):
    role selection and the walled-grid vetting at bind, the tendency
    terms, and the per-axis flux loop. Subclasses choose the face
    value of the advected quantity through the `_face_value` /
    `_linear_face_value` hooks (centered by default) and declare
    wall capability through `_supports_walled`: the centered hooks
    work on walled grids through the structural-zero wall flux
    (module docstring), and so do the biased subclasses, whose wide
    stencils swap in their graded near-wall closure at bind.

    Without a background the module contributes the single
    Rossby-scaled ``advection`` term. With ``background=`` set it
    contributes the difference-form split (module docstring): the
    linear ``background_advection`` term
    :math:`L(q) = S_\mathrm{lin}(U, q)` and the nonlinear
    ``advection`` term :math:`S_\mathrm{full}(U + \mathrm{Ro}\,u',
    q) - S_\mathrm{lin}(U, q)`; the background samples are
    AUXILIARY fields ``background_<component>`` on each velocity
    component's own space.
    """

    parameter_references = (
        fr.model.ParameterReference(
            fr.model.params.SCALING_ROSSBY, default=1.0,
            hint="Rossby number (nh.DynamicalCore)"),
    )

    #: whether the scheme's face values work on walled grids (the
    #: centered hooks do structurally; the biased subclasses do
    #: through their graded near-wall closure, installed at bind)
    _supports_walled: ClassVar[bool] = True

    #: whether the scheme is grounded on mapped geometry at all —
    #: both surfaces: a stretched mesh factor (MappedIntervalMesh)
    #: and a mapping-declared mapped column. The centered scheme is
    #: (order-2 stencils over the measure fields / the physical flux
    #: divergence); the biased subclasses opt out — their
    #: uniform-offset windows need a mapped-aware reconstruction,
    #: future work
    _supports_mapped: ClassVar[bool] = True

    def __init__(
        self,
        background: Mapping[str, Callable | float] | None = None,
    ) -> None:
        """Normalize the background mapping; targets resolve at bind."""
        self._advected: tuple[str, ...] = ()
        self._axis_velocity: tuple[tuple[str, str], ...] = ()
        self._background: dict[str, Callable | float] = (
            _check_background(background))
        self._background_axes: tuple[tuple[str, str], ...] = ()
        self._background_by_axis: dict[str, str] = {}
        self._column: tuple[str, str] | None = None
        self._corrections: dict[str, tuple[str, str]] = {}
        self._halo_axes: tuple[str, ...] = ()
        self._walled: tuple[str, ...] = ()

    # ------------------------------------------------------------
    #  Background declarations (AUXILIARY profile samples)
    # ------------------------------------------------------------
    @property
    def field_declarations(self) -> tuple[fr.model.FieldDeclaration, ...]:
        """The background samples on each component's own space.

        One AUXILIARY field ``background_<component>`` per mapped
        component, declared on the velocity template's own pattern
        (staggered along the component axis; on a walled axis the
        topology-conditional wall Dirichlet makes the sample
        structurally impermeable — its wall face is an exact-zero
        boundary condition, not a DOF), so the profile is sampled at
        that component's own staggered nodes when materialized. The
        user value rides the declaration ``default=`` untouched: a
        constant fills, a coordinate callable is discretized by
        ``grid.create_field``.
        """
        return tuple(
            fr.model.FieldDeclaration(
                f"background_{name}",
                space=fr.spatial.Staggered(
                    axis, wall_bc={axis: BC.DIRICHLET}),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=(_profile_default(name)
                         if callable(self._background[name])
                         else self._background[name]),
                long_name=f"Background {name}-velocity",
                units="m/s")
            for name, axis in _VELOCITY_AXES.items()
            if name in self._background)

    @property
    def field_references(self) -> tuple[fr.model.FieldReference, ...]:
        """The checked claims on the mapped velocity components."""
        return tuple(
            fr.model.FieldReference(
                name, hint="the background flow rides the declared "
                           "velocity components (nh.DynamicalCore "
                           "declares u, v, w)")
            for name in _VELOCITY_AXES if name in self._background)

    def bind(self, table: object) -> None:
        """Freeze the advected set and the axis -> velocity mapping.

        Raises
        ------
        NotImplementedError
            On a walled grid (any bounded mesh factor) when the
            scheme opts out through `_supports_walled`: the natural
            downstream failure (an operator dispatch mismatch deep in
            the flux chain) would be cryptic.
        ValueError
            If a background sample does not resolve on its velocity
            component's own space (a component outside the nh
            ``u``/``v``/``w`` staggering vocabulary), or if a
            wall-normal background component does not vanish on its
            walls (impermeability).
        """
        factors = getattr(table.grid, "factors", ())
        walled = tuple(
            name for mesh in factors for name in mesh.names
            if not getattr(mesh, "periodic", True))
        if walled and not self._supports_walled:
            raise NotImplementedError(
                f"{type(self).__name__} does not support walled "
                f"grids (bounded coordinates: {walled}). Use "
                "CenteredAdvection (walled-capable) or a linear "
                "model (advection=False in nh.Model)")
        self._walled = walled
        self._bind_mapping(table.grid)
        self._advected = table.select(fr.model.roles.ADVECTED)
        selector = table.velocity()
        # selector.labels pairs each velocity name with its axis
        self._axis_velocity = tuple(
            (axis, name) for name, axis in selector.labels)
        self._bind_background(table)

    def _bind_mapping(self, grid: object) -> None:
        """Freeze the mapped-column coupling table (stage C4).

        Raises
        ------
        NotImplementedError
            On a grid carrying a **stretched** mesh factor
            (``MappedIntervalMesh``: a per-axis ``coordinate_map``)
            or a mapping-declared **mapped column** when the scheme
            opts out through `_supports_mapped` (the biased
            subclasses — their order-wide uniform-offset windows need
            a mapped-aware reconstruction, future work), or when the
            mapping declares more than one column (mirroring the
            stage-C3 pressure solver support).
        """
        self._require_uniform_factors(grid)
        mapping = getattr(grid, "mapping", None)
        corrections = (mapping.column_corrections
                       if mapping is not None else {})
        if not corrections:
            return
        if not self._supports_mapped:
            raise NotImplementedError(
                f"{type(self).__name__} does not support mapped "
                "grids (the coordinate mapping declares a mapped "
                "column): the biased face reconstructions are "
                "computational-coordinate rows and would silently "
                "misrepresent physical transport — future work. "
                "Use CenteredAdvection (mapped-capable, stage C4) "
                "or a linear model (advection=False in nh.Model)")
        columns = set(corrections.values())
        if len(columns) != 1:
            raise NotImplementedError(
                f"mapped advection supports exactly one mapped "
                f"column, got {sorted(columns)} "
                "(coordinate-systems plan, stage C4)")
        self._column = next(iter(columns))
        self._corrections = dict(corrections)
        self._halo_axes = tuple(grid.names)

    def _require_uniform_factors(self, grid: object) -> None:
        """Reject stretched mesh factors when the scheme opts out.

        Description
        -----------
        The second (and, for a plain stretched grid, the *only*)
        mapped surface of a grid: a ``MappedIntervalMesh`` factor
        carries its own ``coordinate_map`` and needs **no**
        ``CoordinateMapping`` column, so ``column_corrections`` is
        empty and the mapped-column guard below never fires. The
        centered scheme is grounded here (its two-point stencils
        divide by the codomain measure field, order 2); the biased
        subclasses are not — their uniform-offset windows would bind
        happily and silently lose their design order, the exact
        silent-wrongness ``FiniteDifference`` refuses to commit at
        order > 2.

        Raises
        ------
        NotImplementedError
            On a stretched mesh factor when `_supports_mapped` is
            False.
        """
        if self._supports_mapped:
            return
        stretched = tuple(
            name for mesh in getattr(grid, "factors", ())
            for name in mesh.names if mapped_mesh(mesh))
        if not stretched:
            return
        raise NotImplementedError(
            f"{type(self).__name__} does not support stretched "
            f"(mapped) meshes (mapped coordinates: {stretched}): "
            + mapped_order_hint(
                "its biased face reconstructions and their "
                "order-coupled velocity interpolation")
            + ". Use CenteredAdvection (order 2, mapped-capable) or "
            "a uniform mesh (IntervalMesh)")

    #: on a mapped grid the flux divergence multiplies grid.metric
    #: coefficients the halo tracer cannot follow (V-N2): declare
    #: the stencil substitute (diff + interp chains, depth 2 — the
    #: DynamicalCore precedent). None on flat grids: the flat path
    #: stays fully halo-traced, exactly as before stage C4.
    @property
    def extra_halo(self) -> HaloSpec | None:
        """Two halo cells per coordinate on mapped grids only."""
        if self._column is None:
            return None
        return HaloSpec(dict.fromkeys(self._halo_axes, 2))

    def _bind_background(self, table: object) -> None:
        """Freeze the axis -> background-sample mapping.

        Validates that every background sample landed on its
        component's own space (the GaussianWaveMaker precedent) and
        records the per-axis sample names in the flux-loop axis
        order.
        """
        by_axis: dict[str, str] = {}
        for axis, name in self._axis_velocity:
            if name not in self._background:
                continue
            sample = f"background_{name}"
            if (axis != _VELOCITY_AXES.get(name)
                    or table[sample].space is not table[name].space):
                raise ValueError(
                    f"the background sample {sample!r} resolves on "
                    f"{table[sample].space!r} but {name!r} lives on "
                    f"{table[name].space!r}; the background flow "
                    "samples on the nh C-grid staggering (u, v, w "
                    "on their x/y/z faces)")
            by_axis[axis] = sample
        self._background_by_axis = by_axis
        self._background_axes = tuple(by_axis.items())
        self._check_background_walls(table)

    def _check_background_walls(self, table: object) -> None:
        """Impermeability: wall-normal background zero at the wall.

        The sampled field is structurally impermeable (its wall face
        is a Dirichlet zero, not a DOF), so the check runs on the
        *user's input* — the shallow-water background precedent: a
        constant must be zero, a callable must evaluate to zero at
        the wall positions (over the tangential nodes it names).
        """
        grid = table.grid
        for name, axis in _VELOCITY_AXES.items():
            if name not in self._background:
                continue
            mesh = next(
                (factor for factor in grid.factors
                 if axis in factor.names), None)
            if mesh is None or getattr(mesh, "periodic", True):
                continue
            value = self._background[name]
            if not callable(value):
                scale = max(1.0, abs(float(value)))
                worst = abs(float(value))
            else:
                space = table[f"background_{name}"].space
                sample = _sample_profile(
                    grid, space, value, name=f"background_{name}")
                scale = max(1.0, float(np.max(np.abs(
                    np.asarray(sample.data)))))
                worst = _wall_profile_values(
                    grid, space, value, axis, mesh.extent)
            if worst > _WALL_TOL * scale:
                raise ValueError(
                    f"background[{name!r}] does not vanish at the "
                    f"{axis!r} wall (max wall value {worst:.3e}): "
                    "the walls are impermeable, so the wall-normal "
                    "background component must be zero on the wall "
                    "— the sampled field's wall face is a "
                    "structural (Dirichlet) zero and would silently "
                    "disagree with the profile")

    def tendency_terms(self) -> tuple[fr.model.TendencyTerm, ...]:
        """Return the advection term(s) of the module.

        Without a background: the single Rossby-scaled ``advection``
        term (the pre-background code path, literally unchanged).
        With one: the nonlinear ``advection`` difference term plus
        the genuinely separate ``background_advection`` term tagged
        ``linear=True`` so ``fr.model.linearize`` keeps exactly it (V-S3).
        """
        if not self._background:
            return (
                fr.model.TendencyTerm(
                    name="advection", fn=self._advect,
                    treatment=fr.model.Treatment.EXPLICIT,
                    advances=self._advected,
                    transports=self._advected),
            )
        return (
            fr.model.TendencyTerm(
                name="advection", fn=self._advect_perturbation,
                treatment=fr.model.Treatment.EXPLICIT,
                advances=self._advected, transports=self._advected),
            fr.model.TendencyTerm(
                name="background_advection",
                fn=self._advect_background,
                treatment=fr.model.Treatment.EXPLICIT,
                advances=self._advected, transports=self._advected,
                linear=True),
        )

    def _flux_space(
        self, q: ScalarField, v: ScalarField, axis: str,
    ) -> FunctionSpace:
        """
        Flux (control-volume face) space of ``q`` along ``axis``.

        Description
        -----------
        The face space ``q`` toggled along ``axis`` — and, on a
        walled axis where the flux sits on the advecting velocity's
        staggered faces, with the velocity's wall-normal Dirichlet
        factor adopted (a BC-sibling substitution): the flux then
        carries the structural zero-wall-flux claim and its
        divergence closes with the exact-zero wall value
        (impermeability — module docstring). On periodic axes the
        factors are the same interned object and the substitution
        never fires (the periodic path is bitwise unchanged).

        Parameters
        ----------
        q : ScalarField
            The advected quantity.
        v : ScalarField
            The advecting velocity component of ``axis``.
        axis : str
            The advection axis.

        Returns
        -------
        FunctionSpace
            The flux space (layout preserved).
        """
        space = q.diff(axis).function_space
        v_factor = v.function_space.bare.factor(axis)
        factor = space.bare.factor(axis)
        if factor is not v_factor and _bc_siblings(factor, v_factor):
            space = space.replace(**{axis: v_factor})
        return space

    def _geometry_params(self, state: object) -> dict | None:
        """Collect the CURRENT mapping-parameter fields (mapped)."""
        if self._column is None:
            return None
        return mapping_params(state, state[self._advected[0]].grid)

    def _flux_divergence(
        self,
        q: ScalarField,
        flux: ScalarField,
        axis: str,
        params: dict | None,
    ) -> ScalarField:
        r"""
        Physical flux divergence of one axis, on ``q``'s space.

        Description
        -----------
        Flat grids: literally the pre-C4 expression
        ``flux.diff(axis).retag(q)`` (zero behavior change). On a
        mapped column ``m = M(b, params)`` (module docstring):

        - ``axis == b``: ``(1/J) d_b F_b`` — the derivative along
          the column's physical image;
        - coupled ``axis``: ``d_i F_i - (Z_i/J) interp(d_b F_i)``,
          the correction interpolated onto the main term's
          staggering (BC-sibling hops resolved through the
          registry, wall tags re-adopted by ``retag``);
        - uncoupled ``axis``: the plain computational derivative
          (the physical and computational derivatives agree).

        Every metric derives via ``grid.metric`` at application with
        the current ``params`` — nothing cached (rules 2.3/3.8).

        Parameters
        ----------
        q : ScalarField
            The advected quantity (the divergence's target space).
        flux : ScalarField
            The advective flux on ``q``'s control-volume faces.
        axis : str
            The flux axis.
        params : dict | None
            The dynamic mapping-parameter fields (None on flat and
            static-default mapped grids).

        Returns
        -------
        ScalarField
            The flux divergence on ``q``'s (wall-tagged) space.
        """
        if self._column is None:
            return flux.diff(axis).retag(q)
        mapped, base = self._column
        grid = q.grid
        div = flux.diff(axis)
        if axis == base:
            inv_j = grid.metric(
                div.function_space, f"d{base}_d{mapped}",
                params=params)
            return (div * inv_j).retag(q)
        if axis not in self._corrections:
            return div.retag(q)
        dcol = flux.diff(base)
        space = dcol.function_space
        coeff = (
            grid.metric(space, f"d{mapped}_d{axis}", params=params)
            / grid.metric(space, f"d{mapped}_d{base}",
                          params=params))
        corr = coeff * dcol
        registry = grid.dispatch
        for name in (base, axis):
            src = corr.function_space.bare.factor(name)
            dst = div.function_space.bare.factor(name)
            if src is dst or _bc_siblings(src, dst):
                continue
            corr = registry.resolve("interpolate", src)[name](corr)
        return (div - corr.retag(div)).retag(q)

    def _advect(
        self, state: object, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        """Flux-form transport of every advected component (Ro-scaled)."""
        ro = ctx.params[fr.model.params.SCALING_ROSSBY]
        params = self._geometry_params(state)
        out: dict[str, ScalarField] = {}
        for qname in self._advected:
            q = state[qname]
            res = None
            for axis, vname in self._axis_velocity:
                v = state[vname]
                flux_space = self._flux_space(q, v, axis)
                v_face = self._velocity_face(v, flux_space)
                flux = v_face * self._face_value(
                    q, v_face, axis, flux_space)
                # the divergence lands back on q's wall-tagged space
                # (flat grids: literally flux.diff(axis).retag(q))
                divergence = self._flux_divergence(
                    q, flux, axis, params)
                res = -divergence if res is None else res - divergence
            out[qname] = ro * res
        return out

    # ------------------------------------------------------------
    #  The background-split terms
    # ------------------------------------------------------------
    def _advect_perturbation(
        self, state: object, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        r"""
        Nonlinear difference term with a background flow.

        Description
        -----------
        :math:`N(u', q) = S_\mathrm{full}(U + \mathrm{Ro}\,u', q) -
        S_\mathrm{lin}(U, q)`: the module's own scheme at the full
        advecting velocity minus the linear background transport, so
        the two-term sum telescopes to the full-velocity scheme. No
        outer Rossby factor (module docstring, scaling convention).
        """
        ro = ctx.params[fr.model.params.SCALING_ROSSBY]
        return {
            qname: (self._full_transport(state, ro, state[qname])
                    - self._linear_transport(state, state[qname]))
            for qname in self._advected}

    def _advect_background(
        self,
        state: object,
        ctx: StepContext,  # noqa: ARG002 — fixed term signature
    ) -> dict[str, ScalarField]:
        r"""
        Linear transport by the background flow (``linear=True``).

        Description
        -----------
        :math:`L(q) = S_\mathrm{lin}(U, q)`: the linear
        discretization of transport by the static background samples
        — exactly linear in the state (the upwind side selection
        reads only the background field, never the state), so
        ``fr.model.linearize`` keeps this term and drops the nonlinear
        difference.
        """
        return {
            qname: self._linear_transport(state, state[qname])
            for qname in self._advected}

    def _full_transport(
        self, state: object, ro: object, q: ScalarField,
    ) -> ScalarField:
        """
        Flux-form transport of ``q`` by the full velocity.

        Description
        -----------
        The per-axis flux loop of `_advect` with the advecting
        velocity ``U + Ro u'`` (axes without a background sample:
        ``Ro u'``) and no outer Rossby factor.

        Parameters
        ----------
        state : object
            The full state (perturbation + background samples).
        ro : object
            The Rossby number (a traced parameter scalar).
        q : ScalarField
            The advected quantity.

        Returns
        -------
        ScalarField
            The full-velocity transport of ``q``.
        """
        params = self._geometry_params(state)
        res = None
        for axis, vname in self._axis_velocity:
            v = ro * state[vname]
            sample = self._background_by_axis.get(axis)
            if sample is not None:
                v = v + state[sample]
            flux_space = self._flux_space(q, v, axis)
            v_face = self._velocity_face(v, flux_space)
            flux = v_face * self._face_value(
                q, v_face, axis, flux_space)
            divergence = self._flux_divergence(q, flux, axis,
                                               params)
            res = -divergence if res is None else res - divergence
        return res

    def _linear_transport(
        self, state: object, q: ScalarField,
    ) -> ScalarField:
        """
        Linear flux-form transport of ``q`` by the background flow.

        Description
        -----------
        The per-axis flux loop over the background-mapped axes only,
        with the `_linear_face_value` hook (the module's linear row)
        and the face-velocity sign masks read from the static
        background samples.

        Parameters
        ----------
        state : object
            The full state (carries the background samples).
        q : ScalarField
            The advected quantity.

        Returns
        -------
        ScalarField
            The linear background transport of ``q``.
        """
        params = self._geometry_params(state)
        res = None
        for axis, sample in self._background_axes:
            flux_space = self._flux_space(q, state[sample], axis)
            v_face = self._velocity_face(state[sample], flux_space)
            flux = v_face * self._linear_face_value(
                q, v_face, axis, flux_space)
            divergence = self._flux_divergence(q, flux, axis,
                                               params)
            res = -divergence if res is None else res - divergence
        return res

    def _velocity_face(
        self, v: ScalarField, flux_space: object,
    ) -> ScalarField:
        """
        Velocity component on the flux space (subclass hook).

        Description
        -----------
        Default: the registered centered interpolation. The biased
        subclasses override this with the old stack's order-coupled
        symmetric interpolation.

        Parameters
        ----------
        v : ScalarField
            The advecting velocity component.
        flux_space : object
            The flux (control-volume face) space.

        Returns
        -------
        ScalarField
            ``v`` on the flux space.
        """
        return v.to(flux_space)

    def _face_value(
        self,
        q: ScalarField,
        v_face: ScalarField,  # noqa: ARG002 — the biased hooks read it
        axis: str,  # noqa: ARG002 — the biased hooks read it
        flux_space: object,
    ) -> ScalarField:
        """
        Face value of the advected quantity (subclass hook).

        Description
        -----------
        Default: the registered centered interpolation. The biased
        subclasses override this with the upwind-selected
        reconstruction pair.

        Parameters
        ----------
        q : ScalarField
            The advected quantity.
        v_face : ScalarField
            The advecting velocity component on the flux space.
        axis : str
            The advection axis.
        flux_space : object
            The flux (control-volume face) space of ``q`` along
            ``axis``.

        Returns
        -------
        ScalarField
            ``q`` on the flux space.
        """
        return q.to(flux_space)

    def _linear_face_value(
        self,
        q: ScalarField,
        v_face: ScalarField,  # noqa: ARG002 — the biased hook reads it
        axis: str,  # noqa: ARG002 — the biased hook reads it
        flux_space: object,
    ) -> ScalarField:
        """
        Linear face value for the background transport (hook).

        Description
        -----------
        Default: the registered centered interpolation — the
        centered scheme's own flux, which is already linear. The
        biased subclasses override this with the linear
        (optimal-weight) upwind row selected by the sign of the
        static background face velocity.

        Parameters
        ----------
        q : ScalarField
            The advected quantity.
        v_face : ScalarField
            The background face velocity (static in the state).
        axis : str
            The advection axis.
        flux_space : object
            The flux (control-volume face) space of ``q`` along
            ``axis``.

        Returns
        -------
        ScalarField
            The linear face value of ``q``.
        """
        return q.to(flux_space)


# ================================================================
#  The public module family
# ================================================================
class CenteredAdvection(_FluxFormAdvection):

    r"""
    Flux-form centered advection of every ADVECTED component.

    Description
    -----------
    Works on periodic **and walled** grids (channel walls, rigid
    lids, and their combinations): the wall fluxes are structural
    zeros through the wall-normal velocity's Dirichlet fill — no
    boundary-condition physics choice is involved (module
    docstring).

    Parameters
    ----------
    background : Mapping[str, Callable | float] | None, optional
        Prescribed background flow, keyed by velocity component
        (``{"u": lambda y, z: ..., "w": 0.0}``; missing components
        are zero); profiles are sampled at each component's own
        staggered nodes. Adds the linear ``background_advection``
        term; the full advecting velocity becomes
        :math:`U + \mathrm{Ro}\,u'` with no outer Rossby factor —
        unlike the old stack, whose scaling factor multiplied the
        background too (default: None).
    """


class UpwindAdvection(_FluxFormAdvection):

    r"""
    Flux-form advection with linear upwind-biased reconstruction.

    Description
    -----------
    The advected quantity is reconstructed on the control-volume
    faces with the full ``order``-cell upwind-biased Shu row (the
    old stack's ``UpwindAdvection`` numerics), the stencil side
    selected by the sign of the (centered-interpolated) face
    velocity: positive flux velocity reads the left-biased row,
    non-positive the right-biased one — a ``Where`` select, so the
    selection is smooth arithmetic in the traced step (no Python
    branching on traced values).

    As in the old stack, the velocity interpolation onto the flux
    faces is order-coupled: ``order - 1`` symmetric points (the
    two-point mean at the default ``order=3``).

    **Formal order**: the reconstruction converges at ``order``, but
    the tendency this module contributes is formally **2nd order as
    soon as the advecting velocity varies along the flux axis** — the
    product-rule / deconvolution mismatch of the C-grid flux form
    (module docstring): the high-order face quantity is the
    deconvolved flux :math:`R(v q)`, while the scheme forms
    ``v_face * R(q)``, leaving a cross term
    :math:`\sim (h^2/24)\,2\,v'q'` that only vanishes for a constant
    ``v``. Choose the scheme for its dispersion properties and its
    upwind (non-oscillatory) behavior at fronts, not for its
    asymptotic order.

    **Walled grids**: supported. On a grid carrying any bounded mesh
    factor ``bind`` swaps every biased reconstruction and every
    velocity interpolation for its ``boundary="graded"`` variant, so
    the wide windows drop to progressively narrower interior-only
    stencils at the faces adjacent to each wall and read no exterior
    value (class ``_BiasedFaceReconstruction``). The walled flux
    bookkeeping is the centered scheme's: along a walled axis the flux
    adopts the wall-normal velocity's Dirichlet tag (``_flux_space``),
    so the wall flux is a **structural exact zero** and the wall stays
    impermeable to machine precision, not to truncation — under either
    ``wall=`` rung, which only ever writes the *interior* faces. The
    interior keeps the *reconstruction's* design order; the ``K``
    near-wall faces per side legitimately drop to the reduced rungs, so
    the *global* rate on a walled axis is the near-wall rung's — the
    accuracy price of a BC-free bounded closure (R1,
    ``boundary_plan.md``). Each walled axis needs at least
    ``order + 1`` cells.

    **The wall-adjacent rung** (``wall=``, walled grids only) is the
    one genuine choice the closure leaves, and it is the classic
    accuracy-vs-monotonicity trade:

    - ``wall="upwind1"`` (default): the 1st-order upwind cell. Keeps
      the upwind bias — and hence the numerical dissipation —
      everywhere, including on the wall-adjacent face where a boundary
      layer or a front is most likely to sit. Its :math:`O(h)` face
      value costs a full order globally: measured on the walled
      tracer-advection problem of the test suite the max-norm tendency
      error converges at **~1.0** whatever the interior order (which
      stays 3.00 / 5.00 — the interior is untouched).
    - ``wall="centered2"``: the two-point mean of the two cells that
      straddle the wall-adjacent face — the same cells the upwind
      window is a subset of, so it is just as interior-only. Its
      :math:`O(h^2)` face value lifts the global rate to **~2.0**
      (measured), i.e. to the centered scheme's, without touching the
      interior. The price: on that one face per side the left- and
      right-biased reconstructions coincide, so there is **no upwind
      dissipation at the wall**. On a smooth flow this is free
      accuracy; on a front pressed against the wall it rings (measured
      on a wall-adjacent step: ~2x the overshoot of ``upwind1``, and
      WENO's ENO property no longer applies on that face).

    Pick ``centered2`` when the near-wall flow is smooth and the global
    order matters (a resolved boundary layer, a convergence study);
    keep ``upwind1`` when fronts, steps, or under-resolved boundary
    layers may reach the wall.

    Mapped grids stay rejected at bind: the biased rows are
    uniform-offset (computational-coordinate) rows and lose their
    design order on a stretched mesh. Use `CenteredAdvection`
    (mapped-capable, order 2) there.

    Parameters
    ----------
    order : int, optional
        The odd formal order of the biased reconstruction; the
        framework tables ground 3 and 5 (default: 3).
    background : Mapping[str, Callable | float] | None, optional
        Prescribed background flow, keyed by velocity component
        (see `CenteredAdvection`); the linear
        ``background_advection`` term applies the linear upwind row
        of the same order, side-selected by the sign of the static
        background face velocity (default: None).
    wall : Literal["upwind1", "centered2"], optional
        The bottom rung of the graded near-wall ladder on a walled
        grid; inert on a fully periodic one (default: "upwind1" — the
        monotone, globally 1st-order choice; see the class
        Description).
    """

    _weighting: ClassVar[Literal["linear", "weno"]] = "linear"

    #: the order-wide biased windows are legal on a bounded axis
    #: through their graded near-wall closure, installed at bind
    _supports_walled: ClassVar[bool] = True

    #: the biased reconstructions are computational-coordinate rows;
    #: mapped columns need a mapped-aware variant — future work
    #: (taught rejection at bind)
    _supports_mapped: ClassVar[bool] = False

    def __init__(
        self,
        order: int = 3,
        *,
        background: Mapping[str, Callable | float] | None = None,
        wall: Literal["upwind1", "centered2"] = "upwind1",
    ) -> None:
        """Build the biased reconstruction pairs for ``order``."""
        super().__init__(background=background)
        if order not in _SUPPORTED_ORDERS:
            raise ValueError(
                f"{type(self).__name__} grounds the biased "
                f"reconstruction orders {_SUPPORTED_ORDERS} (the "
                f"framework WENO tables), got {order}")
        if wall not in WALL_RUNGS:
            raise ValueError(
                f"{type(self).__name__} grounds the near-wall rungs "
                f"{WALL_RUNGS}, got {wall!r}")
        self._order = order
        self._wall = wall
        self._install_kernels("none")

    def _install_kernels(
        self, boundary: Literal["none", "graded"],
    ) -> None:
        """
        Build the face kernels in one boundary variant.

        Description
        -----------
        A fully periodic grid keeps the plain (``"none"``) kernels
        built at construction — the periodic path is then literally
        the pre-walled one, down to the interned operator objects.
        ``bind`` re-installs the ``"graded"`` variants when the grid
        carries a bounded factor; on the periodic axes of a mixed grid
        the graded kernels return their (identical) interior pass
        untouched, so the periodic numerics are bitwise unchanged
        there too.

        Parameters
        ----------
        boundary : Literal["none", "graded"]
            The variant to install.
        """
        order = self._order
        weighting = self._weighting  # class-attribute lookup
        # the wall rung is inert on the plain kernel: normalized to the
        # default so a periodic grid keeps the pre-``wall=`` interned
        # objects whatever the user asked for (bitwise unchanged)
        wall = self._wall if boundary == "graded" else "upwind1"
        self._left = _BiasedFaceReconstruction(order, "left",
                                               weighting, boundary,
                                               wall)
        self._right = _BiasedFaceReconstruction(order, "right",
                                                weighting, boundary,
                                                wall)
        # the linear-weight pair of the background transport (for
        # weighting == "linear" these are the same interned objects)
        self._lin_left = _BiasedFaceReconstruction(order, "left",
                                                   "linear", boundary,
                                                   wall)
        self._lin_right = _BiasedFaceReconstruction(order, "right",
                                                    "linear", boundary,
                                                    wall)
        self._interp = _CenteredFaceInterpolation(order - 1, boundary)

    def bind(self, table: object) -> None:
        """Install the walled kernels, then widen the provisional halo.

        Description
        -----------
        On a grid carrying any bounded mesh factor the plain kernels
        are swapped for their ``boundary="graded"`` variants (the
        near-wall closure), after checking that every walled axis is
        wide enough to carry the ladder (``order + 1`` cells: the
        widest rung's window must fit the lattice and the two sides'
        reduced faces must not collide).

        The landed assembly then validates every term over real
        zero-valued fields (step 6) *before* the final negotiation
        (step 7), on the grid's provisional halo — wide enough for
        the default registry only. The biased face kernels of
        ``order > 3`` need more, so the widened per-axis demand is
        negotiated here at bind (step 4, host-side, pre-freeze; the
        final negotiation then re-derives at least this width from
        the halo trace). On a frozen grid whose recorded halo is
        narrower this raises the framework's taught
        ``GridFrozenError`` ("assemble the most demanding model
        first").

        Parameters
        ----------
        table : FieldTable
            The resolved field table (base contract).

        Raises
        ------
        NotImplementedError
            If a walled axis carries fewer than ``order + 1`` cells.
        """
        super().bind(table)
        grid = table.grid
        if self._walled:
            self._check_walled_extent(grid)
            self._install_kernels("graded")
        need = self._order // 2 + 1
        current = dict(grid.decomposition.halo.widths)
        axes = tuple(axis for axis, _ in self._axis_velocity)
        if all(current.get(axis, 0) >= need for axis in axes):
            return
        demand = grid.decomposition.halo.merge_max(
            HaloSpec(dict.fromkeys(axes, need)))
        grid.negotiate(halo=demand)

    def _check_walled_extent(self, grid: object) -> None:
        """Reject walled axes too short to carry the graded ladder.

        Raises
        ------
        NotImplementedError
            If a walled axis carries fewer than ``min_cells(order)``
            cells — the ladder's widest rung would then not fit, or
            the two walls' reduced faces would collide, and the
            downstream failure would be a cryptic index error.
        """
        need = min_cells(self._order)
        short = tuple(
            (name, mesh.n_cells)
            for mesh in grid.factors for name in mesh.names
            if name in self._walled and mesh.n_cells < need)
        if short:
            raise NotImplementedError(
                f"{type(self).__name__}(order={self._order}) needs at "
                f"least {need} cells on every walled axis (the graded "
                "near-wall ladder must fit between the two walls), "
                f"got {short}. Use a coarser order, more cells, or "
                "CenteredAdvection")

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def order(self) -> int:
        """Formal order of the biased face reconstruction."""
        return self._order

    @property
    def wall(self) -> Literal["upwind1", "centered2"]:
        """Bottom rung of the graded near-wall ladder."""
        return self._wall

    # ------------------------------------------------------------
    #  The upwind face values
    # ------------------------------------------------------------
    def _velocity_face(
        self, v: ScalarField, flux_space: object,
    ) -> ScalarField:
        """
        Order-coupled symmetric velocity interpolation.

        Description
        -----------
        Old-stack parity: the velocity is interpolated onto the
        flux faces with ``order - 1`` symmetric points, applied
        along every axis where its factor differs from the flux
        space (a genuine operator application on each axis, so the
        halo trace follows).

        The trailing ``retag`` is the walled-grid seam: nodal operator
        outputs are BC-free, so on a walled axis the interpolated
        velocity lands on the BC-free sibling of the flux space's
        wall-tagged factor and adopts the tag here (the exact-zero
        wall flux the divergence closes on). On periodic axes the
        factors are the same interned object and ``retag`` returns
        ``self`` — the periodic path is bitwise unchanged.

        Parameters
        ----------
        v : ScalarField
            The advecting velocity component.
        flux_space : object
            The flux (control-volume face) space.

        Returns
        -------
        ScalarField
            ``v`` on the flux space.
        """
        result = v
        bare = flux_space.bare
        for axis in bare.names:
            if result.function_space.bare.factor(axis) is not (
                    bare.factor(axis)):
                result = self._interp[axis](result)
        return result.retag(flux_space)

    def _face_value(
        self,
        q: ScalarField,
        v_face: ScalarField,
        axis: str,
        flux_space: object,
    ) -> ScalarField:
        """
        Upwind-selected biased face value of ``q``.

        Description
        -----------
        ``v_face + |v_face|`` is nonzero exactly where the face
        velocity is positive, so the ``Where`` select reads the
        left-biased reconstruction there and the right-biased one
        elsewhere (ties at zero velocity take the right-biased
        side, old-stack parity) — all operator applications, fully
        visible to the halo-accounting trace. Both reconstructions
        adopt the flux space's wall tag first (``retag``; a no-op on
        periodic axes), so the three ``Where`` operands agree.

        Parameters
        ----------
        q : ScalarField
            The advected quantity.
        v_face : ScalarField
            The advecting velocity component on the flux space.
        axis : str
            The advection axis.
        flux_space : object
            The flux (control-volume face) space of ``q`` along
            ``axis``.

        Returns
        -------
        ScalarField
            The upwind-biased face value of ``q``.
        """
        positive = v_face + abs(v_face)
        return Where()(positive,
                       self._left[axis](q).retag(flux_space),
                       self._right[axis](q).retag(flux_space))

    def _linear_face_value(
        self,
        q: ScalarField,
        v_face: ScalarField,
        axis: str,
        flux_space: object,
    ) -> ScalarField:
        """
        Linear-row upwind face value for the background transport.

        Description
        -----------
        The same ``Where`` side selection (and the same wall-tag
        ``retag``) as `_face_value`, but with the linear
        (optimal-weight) reconstruction pair — the WENO module's
        smooth-limit row — and the mask read from the static
        background face velocity, so the value is exactly linear in
        the state.

        Parameters
        ----------
        q : ScalarField
            The advected quantity.
        v_face : ScalarField
            The background face velocity (static in the state).
        axis : str
            The advection axis.
        flux_space : object
            The flux (control-volume face) space of ``q`` along
            ``axis``.

        Returns
        -------
        ScalarField
            The linear upwind-biased face value of ``q``.
        """
        positive = v_face + abs(v_face)
        return Where()(positive,
                       self._lin_left[axis](q).retag(flux_space),
                       self._lin_right[axis](q).retag(flux_space))


class WENOAdvection(UpwindAdvection):

    r"""
    Flux-form advection with WENO-JS reconstruction.

    Description
    -----------
    The upwind flux form of `UpwindAdvection` with the nonlinear
    WENO-JS weighting of the same biased windows (Jiang & Shu 1996;
    numerics parity with the old stack's ``WENO`` module through the
    framework kernel): on smooth data the nonlinear weights approach
    the optimal ones and the scheme reduces to the linear upwind
    row; at discontinuities the smoothness indicators suppress the
    oscillatory candidates (the ENO property).

    **Formal order**: as for `UpwindAdvection` — the WENO
    reconstruction converges at ``order`` on smooth data, while the
    tendency is formally 2nd order once the advecting velocity varies
    along the flux axis (the product-rule / deconvolution mismatch of
    the C-grid flux form; module docstring). WENO is used here for the
    non-oscillatory property at fronts, not for asymptotic order. The
    reference below is the route that would restore the design order
    (reconstruct the flux :math:`v q` rather than ``q``); FRIDOM does
    not take it, because it would cost the exact-zero wall flux and
    constancy preservation (module docstring).

    References
    ----------
    .. [1] S. Mishra, C. Pares-Pulido, and K. G. Pressel,
       "Arbitrarily high-order (weighted) essentially
       non-oscillatory finite difference schemes for anelastic
       flows on staggered meshes", *Communications in Computational
       Physics*, 2021.

    Parameters
    ----------
    order : int, optional
        The odd formal WENO order; the framework tables ground 3
        and 5 (default: 3).
    background : Mapping[str, Callable | float] | None, optional
        Prescribed background flow, keyed by velocity component
        (see `CenteredAdvection`). The linear
        ``background_advection`` term applies the linear
        optimal-weight row (the WENO smooth limit), NOT the
        nonlinear WENO weights — a WENO discretization of a linear
        operator is not linear in the state; the nonlinear
        difference term keeps the full WENO weighting of the full
        advecting velocity (default: None).
    wall : Literal["upwind1", "centered2"], optional
        The bottom rung of the graded near-wall ladder on a walled
        grid (see `UpwindAdvection`). Note that ``"centered2"``
        gives up the ENO property on the wall-adjacent face — the
        WENO weighting has nothing to weight there — so a front
        pressed against the wall will ring; it buys the global 2nd
        order the ``"upwind1"`` bottom cannot reach (default:
        "upwind1").
    """

    _weighting: ClassVar[Literal["linear", "weno"]] = "weno"
