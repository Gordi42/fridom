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
(pinned in ``tests/model/modules/test_advection.py``). What the biased
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

**Finite-volume (average-family) tracers (FV-D2 option A)**: an
``ADVECTED`` component declared ``family="fv"`` resolves to the
``CellAvg`` cell-average family and is transported in genuinely
conservative FV flux form — a per-component branch, so one module
carries a mixed nodal-plus-FV advected set (nodal ``u``/``v``/``w``
self-advection beside a ``CellAvg`` buoyancy). Three things change
for a ``CellAvg`` tracer ``q``: the flux sits on the reconstructed
face (``CellAvg -> Right`` periodic / ``Inner`` bounded) rather than
the nodal ``diff`` codomain; the face value is the average-family
reconstruction (the registered ``LinearReconstruction`` for
``CenteredAdvection``, the module-private ``_FVBiasedReconstruction``
— reusing ``spatial.operators.weno`` and the same ``graded`` closure
``Fallback`` wraps — for the biased schemes); and the divergence is
the **exact** ``spatial.operators.flux_diff`` (``Right | Inner ->
CellAvg``, the discrete Gauss theorem) instead of ``diff`` + retag.
Because ``flux_diff`` telescopes to the boundary fluxes — zero on a
periodic wrap, zero at a wall (the ``Inner`` variant pads exact-zero
boundary fluxes) — ``integrate(q)`` is conserved to machine zero over
a run, on periodic and walled axes alike. At second order the FV and
nodal C-grid stencils are the same numbers, so an FV tracer's tendency
is bitwise the nodal one on a periodic box (a retag). On a mapped
terrain-following column (``CenteredAdvection`` only, stage F5) the
flat ``flux_diff`` is replaced by the **J-weighted conservative flux
form** (:meth:`_FluxFormAdvection._mapped_fv_divergence`, mirroring the
mapped pressure operator): ``(1/J) [D_i(J F_i) + D_b(F_b - Z_i
I(F_i))]``, whose ``J``-weighted sum telescopes, so the physical
buoyancy content ``\int q\,\mathrm{d}V = \int J q\,\mathrm{d}x`` is
conserved to machine zero — the FV headline property on genuine
terrain, which the consistent (nodal) mapped divergence does not give.
The biased schemes reject mapped geometry entirely (above). The FV
pressure C-grid is stage F3 (the projection here never touches a
tracer).

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

import jax.numpy as jnp
import numpy as np

import fridom as fr
from fridom.framework.utils import dtype_real
from fridom.model.modules.moving_geometry import mapping_params
from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.scalar_field import (
    _bc_siblings,  # the BC-sibling seam of retag/.to
)
from fridom.spatial.immersed_domain import Slip
from fridom.spatial.operators.base import (
    Operator,
    OperatorRequirements,
    SeparableOperator,
    _ensure_valid,
    _finalize,
    _required_halo,
    resolve_codomain,
)
from fridom.spatial.operators.graded import (
    WALL_RUNGS,
    Rung,
    RungSpec,
    apply_graded_mask,
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
    fv_reach_or,
    wall_slots_addressable,
)
from fridom.spatial.operators.select import Where
from fridom.spatial.operators.staggering import (
    footprint_reach,
    mapped_factor,
    mapped_mesh,
    mapped_order_hint,
)
from fridom.spatial.operators.weno import (
    _shu_row,  # the exact-rational coefficient seam
    _weno_combine,  # the shared nonlinear-weight reduction
    _window_views,  # the union-window slicer of the selected kernel
    weno_reconstruct,
    weno_tables,
)
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import AverageSpace, CellAvg
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet
from fridom.spatial.spaces.trace import Side

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


def _safe_ratio(
    num: ScalarField, den: ScalarField,
) -> ScalarField:
    r"""
    VJP-sealed metric ratio ``num / den`` (the slope factor Z_i / J).

    Description
    -----------
    The coupled-axis slope coefficient of the nodal mapped divergence
    (:meth:`_FluxFormAdvection._flux_divergence`) divides two metric
    fields sharing a space. On a bounded (terrain) axis the denominator
    ``J = d<mapped>_d<base>`` is strictly positive on every valid cell
    but zero-filled in the never-valid storage padding, where the raw
    quotient is a masked singularity: the forward ``0/0`` is discarded
    by the post-application retag/sync, but its reverse VJP is
    ``0/0 -> NaN`` and poisons ``jax.grad`` through nonlinear terrain
    advection (the differentiability-policy poison). The double-
    ``jnp.where`` seals the reverse pass while staying bitwise identical
    on every valid cell (``bad`` covers only the ``J == 0`` padding) —
    the same seal as the staggered measure divide and the mapped
    pressure operator's ``_divide_by_jacobian`` (AGENTS.md diff policy).

    Parameters
    ----------
    num : ScalarField
        The numerator (the slope ``d<mapped>_d<axis>``).
    den : ScalarField
        The denominator (the Jacobian ``d<mapped>_d<base>``).

    Returns
    -------
    ScalarField
        The ratio on the divide's structure, finite (0) in the
        never-valid padding.
    """
    if (getattr(num, "storage", None) is None
            or getattr(den, "storage", None) is None):
        # halo-trace stand-in (no data): the plain quotient flows the
        # space and ghost demand; the VJP seal is a runtime concern,
        # absent here (the H7 slice term traces on the flat, unimmersed
        # path, so this branch fires only under the halo accounting).
        return num / den
    bad = den.storage == 0.0
    safe = jnp.where(bad, 1.0, den.storage)
    ratio = jnp.where(bad, 0.0, num.storage / safe)
    # the field divide fixes the result's structure (space, merged halo
    # validity); its raw quotient data is discarded for the guarded
    # ratio, so the singular divide-VJP is never built.
    return (num / den).with_storage(ratio)


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
        domain: FunctionSpace,
    ) -> OperatorRequirements:
        """Declare reach ``(below, above)``, halo = size // 2.

        Description
        -----------
        The midpoint-aligned interpolation reaches asymmetrically per
        direction (``Center -> Right`` vs ``Right -> Center``); the
        two-sided reach keeps the transport chain from over-provisioning.
        The symmetric ``halo`` stays ``size // 2``.

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        return OperatorRequirements(
            reach=fv_reach_or(self, domain, self._size,
                              self._size // 2))

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
        domain = f.function_space.bare.factor(axis)
        shift = _wall_shift(domain)

        def kernel(arr: Array, axis_index: int) -> Array:
            return _weighted_windows(arr, axis_index, row)

        immersed = getattr(f.grid, "immersed", None)
        if immersed is not None:
            rungs, sel = _centered_mask_ladder(size, shift)
            return _immersed_graded_face(
                self, f, axis, shift, size, centered_offset(size) + shift,
                kernel, rungs, sel, immersed)
        interior = apply_fv_staggered(self, f, axis, size, kernel,
                                      metadata=f.metadata)
        if self._boundary == "none" or domain.mesh.periodic:
            return interior
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


# ================================================================
#  Mask-keyed graded closure (immersed grids)
# ================================================================
#: the halo-synced ``{0, 1}`` wet mask is thresholded back to boolean
#: after the real-dtype exchange (the sync has no boolean negation)
_MASK_WET = 0.5


def _biased_mask_ladder(
    order: int,
    shift: int,
    bias: Literal["left", "right"],
    weighting: Literal["linear", "weno"],
    wall: Literal["upwind1", "centered2"],
) -> tuple[tuple[Rung, ...], tuple[tuple[int, int], ...]]:
    """
    Reduced biased rungs and per-rung union-window selector specs.

    Description
    -----------
    The mask-path ladder of a biased reconstruction: the reduced value
    rungs (`biased_specs`, the same bias-specific windows the wall path
    builds) and the sign-independent **union** selector windows (size
    ``p + 1``, offset ``p // 2``) — one for the interior kernel (index 0)
    and one per reduced rung. A ``wall="centered2"`` bottom rung keeps its
    symmetric window as its own selector.

    Parameters
    ----------
    order : int
        The interior odd formal order.
    shift : int
        The cell-frame shift (0 or 1).
    bias : Literal["left", "right"]
        The upwind bias side of the value rungs.
    weighting : Literal["linear", "weno"]
        The stencil weighting.
    wall : Literal["upwind1", "centered2"]
        The bottom (wall-adjacent) rung.

    Returns
    -------
    tuple[tuple[Rung, ...], tuple[tuple[int, int], ...]]
        The reduced value rungs and the ``K + 1`` union selector specs.
    """
    specs = biased_specs(order, shift, wall)
    rungs = tuple(
        Rung(spec.width, spec_offset(spec, bias),
             _rung_kernel(spec, bias, weighting))
        for spec in specs)
    sel: list[tuple[int, int]] = [(order + 1, order // 2)]
    for spec in specs:
        if spec.family == "centered":
            sel.append((spec.width, centered_offset(spec.width)))
        else:
            sel.append((spec.width + 1, spec.width // 2))
    return rungs, tuple(sel)


def _centered_mask_ladder(
    size: int, shift: int,
) -> tuple[tuple[Rung, ...], tuple[tuple[int, int], ...]]:
    """
    Reduced centered rungs and per-rung selector specs (mask path).

    Description
    -----------
    The mask-path ladder of the centered velocity interpolation: the
    reduced symmetric value rungs (`centered_ladder`) and their selector
    windows (the symmetric window is its own union), plus the interior
    window at index 0.

    Parameters
    ----------
    size : int
        The interior even stencil size.
    shift : int
        The cell-frame shift (0 or 1).

    Returns
    -------
    tuple[tuple[Rung, ...], tuple[tuple[int, int], ...]]
        The reduced value rungs and the ``K + 1`` selector specs.
    """
    widths = centered_ladder(size, shift)
    rungs = tuple(
        Rung(width, centered_offset(width), _centered_kernel(width))
        for width in widths)
    sel: list[tuple[int, int]] = [(size, centered_offset(size))]
    sel.extend((width, centered_offset(width)) for width in widths)
    return rungs, tuple(sel)


def _fill_wall_slots(
    f: FieldLike, axis: str, n_true: int, value: float,
) -> FieldLike:
    """
    Set the two ``axis`` wall ghost slots of ``f`` to ``value``.

    Description
    -----------
    The static-index sibling of ``reconstruct.wall_zeroed_operand`` used
    to exempt the physical-wall structural zeros of a ``shift = 1`` dual
    operand from the selector's wetness demand: the interior-face storage
    keeps its ``n_true`` DOFs at ``[width, width + n_true)`` and the two
    wall faces are the ghost slots ``width - 1`` / ``width + n_true``.
    Gated by the caller on `wall_slots_addressable` (``axis`` local,
    halo >= 1).

    Parameters
    ----------
    f : FieldLike
        The present-mask field on a bounded interior-face factor.
    axis : str
        The bounded coordinate axis.
    n_true : int
        The interior-face count (the operand's true shape).
    value : float
        The fill value (``1.0`` marks the wall slot present/exempt).

    Returns
    -------
    FieldLike
        ``f`` with its two ``axis`` wall ghost slots set to ``value``.
    """
    bare = f.function_space.bare
    axis_index = bare.names.index(axis)
    width = f.grid.decomposition.halo[axis]
    storage = f._data  # noqa: SLF001 — documented storage seam
    ndim = storage.ndim
    left: list[object] = [slice(None)] * ndim
    left[axis_index] = width - 1
    right: list[object] = [slice(None)] * ndim
    right[axis_index] = width + n_true
    storage = storage.at[tuple(left)].set(value).at[tuple(right)].set(
        value)
    return f.with_storage(storage)


def _present_mask(
    op: Operator, immersed: object, operand_space: object,
    axis: str, shift: int,
) -> FieldLike:
    """
    Build the selector present-mask on the operand space (GA-D1).

    Description
    -----------
    The wet-or-exempt mask the union selectors window: ``shift = 0``
    (cell operand) uses the descriptor's own slip (present == wet, no
    exemption); ``shift = 1`` (dual face operand) uses the ``FREE_SLIP``
    (OR) combination so a face adjacent to at least one wet cell is
    present — the wall-side structural zeros are exempt from the wetness
    demand. The mask is materialized zero-padded (``store`` pads, it does
    not sync), so its ghost layers are **halo-synced** here
    (`_ensure_valid`: periodic wrap / shard exchange) before it is
    windowed — otherwise the union products near a periodic or shard
    boundary would read dry ghosts and grade spuriously. On a bounded
    axis the two physical-wall ghost slots are then set present
    (`_fill_wall_slots`, after the sync), reproducing the wall path's
    exempt Dirichlet cells.

    Parameters
    ----------
    op : Operator
        The reconstruction operator (its `_required_halo` triggers and
        sizes the sync).
    immersed : ImmersedDomain
        The grid's immersed descriptor.
    operand_space : SpaceLike
        The (laid-out) operand space.
    axis : str
        The reconstruction axis.
    shift : int
        The cell-frame shift (0 or 1).

    Returns
    -------
    FieldLike
        The real, halo-synced ``{0, 1}`` present-mask on ``operand_space``.
    """
    slip = Slip.FREE_SLIP if shift == 1 else None
    mask = immersed.mask(operand_space, slip=slip)
    present = mask.with_storage(
        mask._data.astype(dtype_real()))  # noqa: SLF001
    present = _ensure_valid(present, _required_halo(op, operand_space))
    if shift == 1:
        factor = operand_space.bare.factor(axis)
        if (not factor.mesh.periodic
                and wall_slots_addressable(present, axis)):
            present = _fill_wall_slots(
                present, axis, factor.shape[0], 1.0)
    # the sync odd-reflects a wall-Dirichlet operand's physical ghosts
    # (interior 1 -> -1 beyond the wall); threshold back to {0, 1} so the
    # union-window product reads a beyond-wall slot as absent (not a
    # sign-cancelling -1) — the exempt wall cell itself was set present
    # above, past the reflected ones
    return present.with_storage(
        (present._data > _MASK_WET).astype(  # noqa: SLF001 — storage seam
            dtype_real()))


def _immersed_graded_face(
    op: Operator,
    f: FieldLike,
    axis: str,
    shift: int,
    size: int,
    m0: int,
    kernel: Callable[[Array, int], Array],
    rungs: tuple[Rung, ...],
    sel_specs: tuple[tuple[int, int], ...],
    immersed: object,
) -> FieldLike:
    """
    Pre-mask, run the interior pass, and select the widest wet rung.

    Description
    -----------
    The shared immersed tail of the biased / centered face kernels
    (GA-D2): the operand is pre-masked to exact zeros at dry DOFs (a
    ``jnp.where``, NaN-safe and VJP-sealing), the interior kernel runs
    over the pre-masked storage, and `graded.apply_graded_mask` selects
    per output face the widest rung whose union window is entirely
    present.

    Parameters
    ----------
    op : Operator
        The reconstruction/interpolation operator.
    f : FieldLike
        The raw operand field.
    axis : str
        The reconstruction axis.
    shift : int
        The cell-frame shift (0 or 1).
    size : int
        The interior stencil size (order for biased, size for centered).
    m0 : int
        The interior window alignment.
    kernel : Callable[[Array, int], Array]
        The interior array kernel.
    rungs : tuple[Rung, ...]
        The reduced value rungs (widest first, bottom last).
    sel_specs : tuple[tuple[int, int], ...]
        The per-rung union selector specs (interior at index 0).
    immersed : ImmersedDomain
        The grid's immersed descriptor.

    Returns
    -------
    FieldLike
        The mask-graded face field on the operator's codomain.
    """
    # sync both the operand and its wet mask to the negotiated ghosts
    # before windowing: the pre-mask reads the mask's ghost layers (a
    # periodic wrap / shard neighbour, dry-exterior at a bounded wall),
    # and ``immersed.mask`` is zero-padded (``store`` never syncs). The
    # mask is cast to real first — the halo exchange has no boolean neg
    required = _required_halo(op, f.function_space)
    raw = immersed.mask(f.function_space)
    wet = _ensure_valid(
        raw.with_storage(raw._data.astype(dtype_real())),  # noqa: SLF001
        required)
    f = _ensure_valid(f, required)
    masked = f.with_storage(jnp.where(
        wet._data > _MASK_WET,  # noqa: SLF001 — storage seam
        f._data,  # noqa: SLF001 — documented storage seam
        jnp.zeros_like(f._data)))  # noqa: SLF001 — storage seam
    interior = apply_fv_staggered(op, masked, axis, size, kernel,
                                  metadata=masked.metadata, align=m0)
    present = _present_mask(op, immersed, f.function_space, axis, shift)
    return apply_graded_mask(op, masked, axis, interior, rungs,
                             sel_specs, shift, present)


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
        domain: FunctionSpace,
    ) -> OperatorRequirements:
        """
        Declare the biased reach ``(below, above)``, halo = order // 2 + 1.

        Description
        -----------
        The biased window is asymmetric and direction-dependent: with
        the same explicit alignment ``m0 = biased_offset + wall_shift``
        the kernel uses, the reach is ``(m0, size - 1 - m0)`` on a
        periodic axis. This is exactly what the runtime consumes, so
        the negotiated width tightens to the true composed footprint
        (``n + 6`` per axis for upwind5, not ``n + 8``) while the
        symmetric ``halo`` (the per-side maximum over both biases) stays
        ``order // 2 + 1``.

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        fallback = self._order // 2 + 1
        try:
            m0 = (biased_offset(self._order, self._bias)
                  + _wall_shift(domain))
            self.codomain(domain)  # SpaceMismatchError on a Fourier row
            reach = footprint_reach(self._order, m0)
        except SpaceMismatchError:
            reach = (fallback, fallback)
        return OperatorRequirements(reach=reach)

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
        immersed = getattr(f.grid, "immersed", None)
        if immersed is not None:
            rungs, sel = _biased_mask_ladder(
                order, shift, bias, weighting, self._wall)
            return _immersed_graded_face(
                self, f, axis, shift, order, m0, kernel, rungs, sel,
                immersed)
        interior = apply_fv_staggered(self, f, axis, order, kernel,
                                      metadata=f.metadata, align=m0)
        if self._boundary == "none" or domain.mesh.periodic:
            return interior
        rungs = tuple(
            Rung(spec.width, spec_offset(spec, bias),
                 _rung_kernel(spec, bias, weighting))
            for spec in biased_specs(order, shift, self._wall))
        return apply_graded_walls(f, axis, interior, rungs, shift)


@final
@interned
class _SelectedFaceReconstruction(Operator):

    """
    WENO selected-input upwind face reconstruction (module-private).

    Description
    -----------
    The one-pass reformulation of the WENO upwind flux value that
    `WENOAdvection._face_value` installs in place of "reconstruct BOTH
    biases, then ``Where``-select the results" (design
    ``research/stencil_lowering.md`` §6). It reads two operands — the
    sign carrier ``positive = v_face + |v_face|`` (on the flux face
    space) and the advected quantity ``q`` (on the cell frame) — and
    runs a SINGLE left-biased WENO reconstruction of the sign-selected
    union window.

    The mirror identity ``recon_right(U) == recon_left(reversed U)``
    (exact for the WENO tables, ``operators.weno``) moves the per-face
    upwind choice to the stencil *inputs*: over the ``order + 1`` cell
    union window ``U`` straddling each output face, tap ``i`` is
    ``where(v_face > 0, U[i], U[order - i])`` for ``i = 0 .. order -
    1`` (ties at ``v_face == 0`` take the right-biased side, old-stack
    parity), and one left reconstruction of the taps is the upwind
    value. The nonlinear-weight arithmetic (smoothness indicators,
    ``order // 2 + 1`` divides per candidate) then runs ONCE, not once
    per bias — measured -39% at 256^3 / -46% at 512^3, exact to
    reversed-summation ulps. The LINEAR upwind path is left on the
    both-then-select spelling (no divides to save; measured slower
    one-path, §6), so this is a WENO-only override.

    The union window's per-side reach equals the biased pair's, so the
    halo demand is unchanged (``order // 2 + 1``): the operator holds
    the interned left `_BiasedFaceReconstruction` and delegates the
    signature, the halo-negotiation trace and the codomain plumbing to
    it (the union kernel's frame *is* that reconstruction's). On a
    bounded (walled) axis only the interior faces take the tap-select;
    the ``K`` faces adjacent to each wall keep the exact both-ladders
    ``Where`` selection (the ladders are ``O(halo)`` slivers, so no
    reconstruction pass is saved there and the byte-identical spelling
    is the cheapest correct one). On a periodic axis no ladder runs —
    the fast path is bitwise the interior tap-select.

    Purity: static structure only (order / boundary / wall; the
    weighting is implicitly ``"weno"``), no Python-side state, and the
    sole ``where`` on the sign is smooth arithmetic — safe in the
    jit-compiled ``mset.tendencies`` container.

    Parameters
    ----------
    order : int
        The odd formal WENO order (the framework tables ground 3, 5).
    boundary : Literal["none", "graded"], optional
        "none" is the periodic-only kernel; "graded" adds the bounded
        near-wall ladder selection (default: "none").
    wall : Literal["upwind1", "centered2"], optional
        The ladder's bottom rung under ``boundary="graded"`` (default:
        "upwind1"; inert on the periodic kernel).
    family : Literal["nodal", "fv"], optional
        The reconstruction family: "nodal" holds the left
        `_BiasedFaceReconstruction`, "fv" the left
        `_FVBiasedReconstruction` of a ``CellAvg`` tracer — the same
        window numbers on the primal cell frame, so the FV/nodal
        bitwise tendency identity of the periodic box survives the
        one-pass spelling (default: "nodal").
    """

    dispatch_kind: ClassVar[str | None] = None

    def __init__(
        self,
        order: int,
        boundary: Literal["none", "graded"] = "none",
        wall: Literal["upwind1", "centered2"] = "upwind1",
        family: Literal["nodal", "fv"] = "nodal",
    ) -> None:
        """Hold the interned left WENO reconstruction of the same frame."""
        recon = (_FVBiasedReconstruction if family == "fv"
                 else _BiasedFaceReconstruction)
        self._recon = recon(order, "left", "weno", boundary, wall)
        self._order = order
        self._boundary = boundary
        self._wall = wall
        self._family = family

    def _intern_key(self) -> tuple:
        """Structural key: order, boundary, wall, family (weno-only; D6)."""
        return (self._order, self._boundary, self._wall, self._family)

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def order(self) -> int:
        """Formal order of the selected-input reconstruction."""
        return self._order

    @property
    def boundary(self) -> Literal["none", "graded"]:
        """The boundary variant: "none" or "graded"."""
        return self._boundary

    @property
    def wall(self) -> Literal["upwind1", "centered2"]:
        """The ladder's bottom rung: "upwind1" or "centered2"."""
        return self._wall

    # ------------------------------------------------------------
    #  Signature and requirements (delegated to the left kernel)
    # ------------------------------------------------------------
    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """Resolve the C-grid face codomain (the left kernel's).

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D factor space of the advected quantity.

        Returns
        -------
        FunctionSpace
            The flux-position codomain factor.
        """
        return self._recon.codomain(domain)

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """Declare halo = order // 2 + 1 (the union window's reach).

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record (the left kernel's).
        """
        return self._recon.requirements(domain)

    # ------------------------------------------------------------
    #  Kernel application (union-window tap select)
    # ------------------------------------------------------------
    def __call__(
        self,
        positive: FieldLike,
        q: FieldLike,
        axis: str,
        flux_space: object,
    ) -> FieldLike:
        """
        Selected-input reconstruction of ``q`` (sign carrier ``positive``).

        Description
        -----------
        The halo-negotiation trace (``HaloTracer`` operands, no
        ``_data``) delegates to the left reconstruction: the union
        window's per-side reach equals it, so the recorded demand and
        the returned codomain tracer are exactly the biased left
        kernel's (faithful, unchanged width). Real operands run the
        union tap-select interior pass through the SAME
        ``apply_fv_staggered`` plumbing as the reconstruction (its
        codomain, alignment ``m0 = order // 2 + shift`` and halo
        accounting), then, on a bounded axis, restore the ``K`` wall
        faces per side from the two graded ladders under the same sign
        select.

        Parameters
        ----------
        positive : FieldLike
            The sign carrier ``v_face + |v_face|`` on ``flux_space``
            (nonzero exactly where the face velocity is positive).
        q : FieldLike
            The advected quantity, on the cell frame.
        axis : str
            The advection axis.
        flux_space : object
            The flux (control-volume face) space the result adopts.

        Returns
        -------
        FieldLike
            The upwind-biased WENO face value of ``q``, on
            ``flux_space``.
        """
        left_op = self._recon[axis]
        if (getattr(q, "_trace_apply", None) is not None
                or getattr(positive, "_trace_apply", None)
                is not None):
            return _to_flux_space(left_op(q), flux_space)
        order = self._order
        u_size = order + 1
        domain = q.function_space.bare.factor(axis)
        # the FV frame is always primal (CellAvg has no nodal node_set)
        shift = 0 if self._family == "fv" else _wall_shift(domain)
        m0 = biased_offset(order, "left") + shift
        tables = weno_tables(order, "left")
        pos_data = positive._data  # noqa: SLF001 — storage seam

        def kernel(storage: Array, axis_index: int) -> Array:
            wins = _window_views(storage, axis_index, u_size)
            length = wins[0].shape[axis_index]
            index = [slice(None)] * pos_data.ndim
            index[axis_index] = slice(m0, m0 + length)
            pos = pos_data[tuple(index)]
            taps = tuple(
                jnp.where(pos, wins[i], wins[order - i])
                for i in range(order))
            return _weno_combine(taps, tables)

        codomain = resolve_codomain(left_op, q.function_space)
        q = _ensure_valid(
            q, _required_halo(left_op, q.function_space))
        interior = apply_fv_staggered(
            left_op, q, axis, u_size, kernel,
            metadata=q.metadata, align=m0)
        interior = _finalize(q, interior, codomain)
        if self._boundary == "none" or domain.mesh.periodic:
            return _to_flux_space(interior, flux_space)
        left_walls = apply_graded_walls(
            q, axis, interior, self._rungs(order, shift, "left"),
            shift)
        right_walls = apply_graded_walls(
            q, axis, interior, self._rungs(order, shift, "right"),
            shift)
        return Where()(positive,
                       _to_flux_space(left_walls, flux_space),
                       _to_flux_space(right_walls, flux_space))

    def _rungs(
        self, order: int, shift: int,
        bias: Literal["left", "right"],
    ) -> tuple[Rung, ...]:
        """Build one wall side's biased graded ladder (WENO rungs)."""
        return tuple(
            Rung(spec.width, spec_offset(spec, bias),
                 _rung_kernel(spec, bias, "weno"))
            for spec in biased_specs(order, shift, self._wall))


@final
@interned
class _FVBiasedReconstruction(SeparableOperator):

    """
    Upwind/WENO reconstruction of a cell average onto its faces.

    Description
    -----------
    The average-family twin of `_BiasedFaceReconstruction`: a
    ``CellAvg`` tracer is reconstructed onto its right faces
    (``CellAvg -> Right`` periodic, ``CellAvg -> Inner`` bounded), the
    face value the FV flux form multiplies by the face-normal
    velocity. Because ``CellAvg`` sits at the primal-cell midpoints
    (``fv_node_offset`` 0.5, exactly like ``Center``), the window
    alignment, the halo, and the graded near-wall ladder are the
    *same numbers* as the nodal primal ``Center -> Right`` / ``Center
    -> Inner`` direction (the FV/nodal stencil identity of the scoping
    study) — so this reuses the module's shared kernels
    (`_biased_kernel`, `_rung_kernel`), ``spatial.operators.weno``
    (the array reconstruction), and the ``spatial.operators.graded``
    ladder (`biased_specs` / `apply_graded_walls`, the same closure
    ``Fallback`` wraps). The cell frame is always the primal one
    (``shift = 0``): the tracer is the cell average, never a
    face-staggered quantity, so there is no dual direction here.

    ``boundary="graded"`` reads **no exterior value** (R1,
    ``boundary_plan.md``): ``CellAvg`` is BC-free and the ladder stays
    inside its true DOFs; on a periodic factor it returns the interior
    pass untouched (bitwise the ``boundary="none"`` kernel). Held
    directly by the advection modules as a left/right pair (the sign
    selection is the module's ``Where`` select); never registered
    under a dispatch kind. Uniform-mesh only (the biased advection
    modules reject stretched meshes at bind).

    Parameters
    ----------
    order : int
        The odd formal order; the framework tables ground 3 and 5.
    bias : Literal["left", "right"]
        The upwind bias side of the reconstruction.
    weighting : Literal["linear", "weno"]
        "linear" applies the full optimal-weight upwind row; "weno"
        applies the nonlinear WENO-JS weighting of the same window.
    boundary : Literal["none", "graded"], optional
        "none" is the periodic-only kernel; "graded" grows the
        bounded ``CellAvg -> Inner`` signature with the near-wall
        ladder (default: "none").
    wall : Literal["upwind1", "centered2"], optional
        The ladder's bottom rung under ``boundary="graded"``; inert
        on the plain kernel (default: "upwind1").
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
        """
        Resolve the FV face codomain: CellAvg -> Right | Inner.

        Description
        -----------
        ``CellAvg -> Right`` on a periodic uniform mesh; the bounded
        ``CellAvg -> Inner`` variant is grounded only under
        ``boundary="graded"`` (the near-wall closure). Average spaces,
        stretched axes, and complex scalars raise.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D factor space.

        Returns
        -------
        FunctionSpace
            The flux-position codomain factor.
        """
        if (not isinstance(domain, CellAvg)
                or domain.scalars is Scalars.COMPLEX):
            raise SpaceMismatchError(
                f"{type(self).__name__} reconstructs a real CellAvg "
                f"tracer onto its faces, got {domain!r}",
                left=domain, operation="reconstruct")
        if mapped_factor(domain):
            raise SpaceMismatchError(
                f"{type(self).__name__} is uniform-mesh only (the "
                "biased advection modules reject stretched meshes at "
                "bind) — "
                + mapped_order_hint(
                    "the biased FV reconstruction rows")
                + f", got {domain!r}",
                left=domain, operation="reconstruct")
        mesh = domain.mesh
        if mesh.periodic:
            return mesh.right
        if self._boundary != "graded":
            raise SpaceMismatchError(
                f"{type(self).__name__} is periodic-only in its "
                "boundary='none' variant; the bounded signature is "
                "the graded near-wall closure (boundary='graded'), "
                f"got {domain!r}",
                left=domain, operation="reconstruct")
        return mesh.inner

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """
        Declare the biased reach ``(below, above)``, halo = order//2+1.

        Description
        -----------
        The FV primal-frame biased window (no dual shift) at the same
        explicit alignment ``m0 = biased_offset`` the kernel uses, so
        the FV upwind/WENO reconstruction tightens to the same width as
        the nodal path and the two stay bitwise-parallel. The symmetric
        ``halo`` (per-side maximum over both biases) stays
        ``order // 2 + 1``.

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        fallback = self._order // 2 + 1
        try:
            m0 = biased_offset(self._order, self._bias)
            self.codomain(domain)  # SpaceMismatchError on a Fourier row
            reach = footprint_reach(self._order, m0)
        except SpaceMismatchError:
            reach = (fallback, fallback)
        return OperatorRequirements(reach=reach)

    # ------------------------------------------------------------
    #  Kernel application (biased window alignment, primal frame)
    # ------------------------------------------------------------
    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Reconstruct along ``axis`` (biased window-aligned kernel).

        Description
        -----------
        The primal cell frame (``shift = 0``): the WENO window
        alignment lands the kernel output on the right face of window
        cell ``order // 2`` (left bias) / ``order // 2 - 1`` (right
        bias). On a bounded axis the graded variant overwrites the
        ``K`` wall faces per side from the ladder, which reads
        interior DOFs only (``CellAvg`` is BC-free).

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
        m0 = biased_offset(order, bias)  # primal frame, shift 0
        kernel = _biased_kernel(order, bias, weighting)
        immersed = getattr(f.grid, "immersed", None)
        if immersed is not None:
            rungs, sel = _biased_mask_ladder(
                order, 0, bias, weighting, self._wall)
            return _immersed_graded_face(
                self, f, axis, 0, order, m0, kernel, rungs, sel,
                immersed)
        interior = apply_fv_staggered(self, f, axis, order, kernel,
                                      metadata=f.metadata, align=m0)
        domain = f.function_space.bare.factor(axis)
        if self._boundary == "none" or domain.mesh.periodic:
            return interior
        rungs = tuple(
            Rung(spec.width, spec_offset(spec, bias),
                 _rung_kernel(spec, bias, weighting))
            for spec in biased_specs(order, 0, self._wall))
        return apply_graded_walls(f, axis, interior, rungs, 0)


def _is_average_space(space: object) -> bool:
    """Whether a (laid-out) space carries any average-family factor.

    Description
    -----------
    The per-component family switch of the flux-form modules: an
    ``ADVECTED`` tracer declared ``family="fv"`` resolves to the
    ``CellAvg`` average family, and its flux mechanics differ from the
    nodal C-grid (the flux lands on the reconstructed face and the
    divergence is the exact ``flux_diff``, not ``diff`` + retag). The
    tracer slice puts a tracer fully on ``CellAvg`` or fully nodal, so
    any average factor marks the FV path.
    """
    return any(isinstance(factor, AverageSpace)
               for factor in space.bare.factors)


def _is_pure_average(space: object) -> bool:
    """Whether *every* non-constant factor is average-family.

    Description
    -----------
    The scalar-tracer discriminant of the mapped FV flux divergence
    (stage F5): a ``CellAvg`` cell-scalar (buoyancy) is average along
    every non-constant axis and takes the J-weighted conservative flux
    form (its physical content is the FV-conserved quantity), whereas a
    C-grid velocity is average only *transversely* (``Right(x) (x)
    CellAvg(y) (x) CellAvg(z)``) — a mixed staggering that takes the
    consistent nodal physical divergence instead (momentum is not
    conserved by the tracer flux form). ``_is_average_space`` (any
    factor) marks the flat FV path; this (all factors) separates a
    pure tracer from a velocity on a mapped column.
    """
    factors = [factor for factor in space.bare.factors
               if not isinstance(factor, ConstantSpace)]
    return bool(factors) and all(
        isinstance(factor, AverageSpace) for factor in factors)


def _is_face_factor(factor: object) -> bool:
    """Whether a bare factor sits on the mesh faces (staggered).

    Description
    -----------
    The staggering discriminant of the biased velocity interpolation:
    a ``Right`` (periodic) or ``Inner`` (bounded) nodal factor lives on
    the mesh faces, half a cell from the ``Center`` / ``CellAvg`` cell
    midpoints. Two factors of the same axis need a genuine half-cell
    interpolation exactly when their face-ness differs; equal face-ness
    means they are co-located (``Center`` and ``CellAvg`` share the
    midpoint) and any difference is family (deconvolve) or BC (retag)
    only.

    Parameters
    ----------
    factor : object
        The bare 1D factor space.

    Returns
    -------
    bool
        True iff the factor is a face-staggered nodal space.
    """
    return (isinstance(factor, NodalSpace)
            and factor.node_set in (NodeSet.RIGHT, NodeSet.INNER))


def _to_flux_space(
    field: FieldLike, flux_space: object,
) -> FieldLike:
    r"""
    Bridge a nodal C-grid stencil output onto the (FV) flux space.

    Description
    -----------
    The biased schemes reconstruct the advected quantity and interpolate
    the advecting velocity on the nodal C-grid frame; both land on a
    ``Center`` factor along the flux axis. On an FV model that axis of
    the flux space is the co-located ``CellAvg`` (the ``diff``
    :class:`FluxDifference` codomain of a velocity's own staggered axis),
    so the output must cross the nodal -> average family there. That
    crossing is the co-located ``Center -> CellAvg`` deconvolution — the
    2nd-order ``LinearDeconvolution`` identity (a one-point pass-through,
    bitwise), so it retypes the factor without moving the numbers and
    the FV tendency stays bitwise the nodal one. A ``retag`` cannot cross
    it (the node-set class changes); a genuine ``.to`` resolves the
    ``"deconvolve"`` kind. Every other axis is either already matching or
    a BC-only difference — the walled adopt-then-strip seam, where a
    BC-free stencil output adopts the flux space's wall Dirichlet tag —
    and stays a ``retag``. On a fully nodal (periodic or walled) flux
    space no axis crosses families, so the loop is empty and this is
    exactly the pre-FV ``retag`` bridge (bitwise unchanged).

    Parameters
    ----------
    field : FieldLike
        The nodal stencil output (a ``ScalarField`` eagerly, a
        ``HaloTracer`` during the halo-negotiation trace).
    flux_space : object
        The flux (control-volume face) space to land on.

    Returns
    -------
    FieldLike
        ``field`` on ``flux_space`` (family crossings deconvolved, the
        rest retagged).
    """
    bare = flux_space.bare
    result = field
    for name in bare.names:
        src = result.function_space.bare.factor(name)
        dst = bare.factor(name)
        if src is dst or _bc_siblings(src, dst):
            continue
        result = result.to(dst)
    return result.retag(flux_space)


def _outer_to_inner(src: object, dst: object) -> bool:
    """Whether ``src -> dst`` is the ``Outer -> Inner`` restriction.

    Description
    -----------
    The hydrostatic diagnosed vertical velocity ``w`` lives on the
    both-boundary ``Outer`` faces, while its vertical flux leg lands on
    the interior ``Inner`` faces. The pair signals the exact restriction
    (``fr.operators.Restriction``): the advecting velocity already sits
    on the flux faces' superset, and dropping its two boundary values is
    the zero-boundary-flux closure. Fires for no existing model — every
    other advecting velocity is on ``Center`` or a face sibling of the
    flux space — so the flux-space and velocity-face branches it gates
    are additive.
    """
    return (isinstance(src, NodalSpace) and isinstance(dst, NodalSpace)
            and src.node_set is NodeSet.OUTER
            and dst.node_set is NodeSet.INNER)


#: H7 surface-flux slice lowering (``_FluxFormAdvection._apply_correction``):
#: ``None`` (the default) resolves per scheme via each class's
#: ``_surface_flux_lowering`` ClassVar — ``"scatter"`` for the centered flux
#: form, ``"embed"`` for the biased/upwind family — because the lowering is
#: scheme-dependent in wall-clock (2026-07-18 single-A100 A/B: scatter wins
#: centered -16% vs off at 2048^2 x 64, embed wins weno5 +5.8% vs off where
#: scatter costs +16%; ``design/plans/active/boundary_trace_plan.md`` §9).
#: ``"scatter"`` — the fully-2D row-scatter-add of the boundary term (the
#: FV-native lowering, and the ``surface_flux`` closure runs FV by default in
#: the hydrostatic model); ``"embed"`` — the sparse-3D ``embed`` of the
#: boundary term plus the pre-slice full-3D ``q * A(1)`` AXPY (nodal only; an
#: FV component's ``embed`` lands on the co-located nodal ``Center``, not
#: ``CellAvg``, so it falls back to ``"scatter"``). Both are exact; setting an
#: explicit string forces one lowering globally (the A/B knob — semantics for
#: explicit strings unchanged).
_SURFACE_FLUX_LOWERING: str | None = None


def _is_surface_seam(v: ScalarField, axis: str) -> bool:
    """Whether ``v`` sits on the both-boundary ``Outer`` face along ``axis``.

    Description
    -----------
    The exact structural condition under which the advective flux drops
    the boundary face (the ``_outer_to_inner`` seam of ``_flux_space``,
    and the FV ``CellAvg -> Inner`` reconstruction): the advecting
    velocity is the diagnosed hydrostatic ``w`` on the vertical ``Outer``
    faces. A static (space-only) predicate, so it branches the tendency
    graph at trace time. Holds today only for the hydrostatic diagnosed
    ``w`` (vertical), so the slice correction is the surface term of that
    one axis; every other advecting velocity is on ``Center`` / a face
    sibling of the flux space and contributes no dropped boundary face.
    """
    factor = v.function_space.bare.factor(axis)
    return isinstance(factor, NodalSpace) and factor.node_set is NodeSet.OUTER


def _is_cell_collocated(space: object) -> bool:
    """Whether every non-constant factor is a cell node set (Center/CellAvg).

    Description
    -----------
    The slice-form eligibility discriminant on an immersed grid: a cell
    scalar (buoyancy / an FV tracer) shares the continuity control volume,
    so its masked ``A(1)`` telescopes to the surface term in every cell
    (the boundary-only property the slice form needs). A staggered C-grid
    velocity (``Right(x) (x) Center(y) (x) Center(z)``) does not: the
    momentum control volume's masked continuity is not discretely
    divergence-free near a cut side wall, so its ``A(1)`` carries genuine
    interior terms the slice form would drop — such a component takes the
    full-3D fallback. Off an immersed grid every component is boundary-only
    (centered interpolation preserves the exact interior divergence-free
    property) and this gate is not consulted.
    """
    factors = [f for f in space.bare.factors
               if not isinstance(f, ConstantSpace)]
    return bool(factors) and all(
        (isinstance(f, NodalSpace) and f.node_set is NodeSet.CENTER)
        or isinstance(f, CellAvg) for f in factors)


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

    The constant-preserving **surface closure** subtracts the correction
    :math:`q\,A(\mathbf 1)` from every ADVECTED component's tendency,
    where :math:`A(\mathbf 1)` is the module's own advective operator
    applied to a constant — a lean divergence of the interpolated face
    velocities. On a bounded vertical axis the diagnosed ``w`` lives on
    the both-boundary ``Outer`` faces and the flux uses only its interior
    ``Inner`` restriction, so :math:`A(\mathbf 1)` is machine-zero in
    every interior cell but ``w(0)/dz`` in the surface cell (the dropped
    surface velocity — the free surface's :math:`\partial_t\eta`).
    Subtracting ``q`` times it advects **through** the surface face with
    the one-sided (top-cell) face value, so :math:`A` annihilates a
    constant in every cell: the Oceananigans-equivalent linear-free-
    surface treatment. Tracer content is then exchanged with the moving
    surface rather than conserved to roundoff (the ``ps`` equation no
    longer carries the whole surface volume flux). Wherever that
    ``A(\mathbf 1)`` is provably boundary-only (a flat, unimmersed or
    cell-collocated component) the correction is evaluated directly as
    the 2D surface trace (``_surface_correction``, the slice form),
    reserving the full-3D accumulation for the cases where the interior
    does not telescope (:meth:`_surface_correction`).

    ``surface_flux`` is tri-state: ``True`` / ``False`` force the
    closure on / off; the default ``None`` **auto-resolves at bind** —
    on iff some advecting velocity sits on the ``Outer`` node set along
    its own flux axis (the ``_outer_to_inner`` seam). So the closure is
    the default for hydrostatic advection (the diagnosed ``w`` on the
    vertical ``Outer`` faces, under *any* construction path) and off —
    bitwise unchanged — for nodal-velocity models (nonhydro2 /
    shallowwater2, every velocity on ``Inner``). The factory knobs
    ``hy.Model(surface_advective_flux=...)`` /
    ``hy.comparison_model(surface_advective_flux=...)`` forward the
    tri-state. The correction covers only the plain ``advection`` term,
    not the ``background_advection`` split.
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

    #: whether the scheme is grounded on immersed (cut-cell) grids
    #: (IP-D4): the centered flux form weights every face flux by the
    #: open-area fraction and divides the divergence by the cell
    #: volume fraction (min-rule faces read two wet cells, so the
    #: two-point stencil never reaches a dry value with nonzero
    #: weight). The biased subclasses opt out — their wide windows
    #: reach across dry cells; the graded-mask closure is designed-for
    #: (IP-D8)
    _supports_immersed: ClassVar[bool] = True

    #: whether the surface-flux slice may relocate the traced surface
    #: velocity onto a staggered (face-collocated) component's flux
    #: column. The slice's :meth:`_surface_boundary_term` moves the
    #: ``Outer`` surface ``w`` onto ``q``'s column with the plain
    #: two-point ``.to`` interpolation; the exactness of the slice's
    #: top-row ``A(1)`` needs that relocation to match the scheme's own
    #: advecting-velocity face interpolation (:meth:`_velocity_face`).
    #: The centered scheme's velocity face *is* the two-point ``.to``,
    #: so it holds; the biased subclasses interpolate the velocity with
    #: an ``(order - 1)``-point centered row (``_CenteredFaceInterpolation``)
    #: that only coincides with ``.to`` at ``order == 3``, so they set
    #: this ``False`` and their staggered momentum takes the exact
    #: full-3D fallback (a cell-collocated tracer needs no horizontal
    #: relocation and keeps the slice regardless — see
    #: :meth:`_slice_valid`).
    _slice_relocation_exact: ClassVar[bool] = True

    #: the per-scheme H7 surface-flux lowering used when the module-level
    #: ``_SURFACE_FLUX_LOWERING`` is ``None``: the centered flux form wins
    #: with ``"scatter"`` (the biased family overrides this to ``"embed"``;
    #: 2026-07-18 A/B, boundary_trace_plan.md §9)
    _surface_flux_lowering: ClassVar[str] = "scatter"

    def __init__(
        self,
        background: Mapping[str, Callable | float] | None = None,
        *,
        surface_flux: bool | None = None,
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
        self._immersed: object = None
        # tri-state config: True/False force, None auto-resolves at bind
        # to on iff the Outer -> Inner boundary-face seam exists
        self._surface_flux: bool | None = surface_flux
        # the resolved per-grid decision (bind); False before bind
        self._surface_flux_on: bool = False

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
        immersed = getattr(table.grid, "immersed", None)
        if immersed is not None and not self._supports_immersed:
            raise NotImplementedError(
                f"{type(self).__name__} does not support immersed "
                "(cut-cell) grids: the wide biased/WENO windows reach "
                "across dry cells, and the graded-mask closure keyed "
                "on the wet region is designed-for (immersed-partial-"
                "cells plan, IP-D8). Use CenteredAdvection (the "
                "supported immersed family — its min-rule faces read "
                "two wet cells) or a linear model (advection=False in "
                "nh.Model)")
        self._immersed = immersed
        if immersed is not None:
            self._halo_axes = tuple(table.grid.names)
        self._bind_mapping(table.grid)
        self._advected = table.select(fr.model.roles.ADVECTED)
        selector = table.velocity()
        # selector.labels pairs each velocity name with its axis
        self._axis_velocity = tuple(
            (axis, name) for name, axis in selector.labels)
        self._bind_background(table)
        self._surface_flux_on = self._resolve_surface_flux(table)

    def _resolve_surface_flux(self, table: object) -> bool:
        """Resolve the tri-state ``surface_flux`` on this grid.

        Description
        -----------
        Explicit ``True`` / ``False`` are forced. The default ``None``
        auto-resolves from **static space metadata only**: on iff some
        advecting velocity sits on the both-boundary ``Outer`` node set
        along its own flux axis — the exact structural condition under
        which ``_outer_to_inner`` fires in ``_flux_space`` and the
        boundary-face flux is dropped. Today that holds precisely for
        the hydrostatic diagnosed ``w`` (on the vertical ``Outer``
        faces), so the constancy-preserving closure is the default there
        while nodal-velocity models (nonhydro2 / shallowwater2, every
        velocity on ``Inner``) resolve off and stay bitwise unchanged.

        Parameters
        ----------
        table : object
            The bind field table (supplies the velocity spaces).

        Returns
        -------
        bool
            Whether the surface closure is active on this grid.
        """
        if self._surface_flux is not None:
            return self._surface_flux
        for axis, vname in self._axis_velocity:
            factor = table[vname].space.bare.factor(axis)
            if (isinstance(factor, NodalSpace)
                    and factor.node_set is NodeSet.OUTER):
                return True
        return False

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
        """Two halo cells per coordinate on mapped or immersed grids.

        On a mapped column the flux divergence multiplies ``grid.metric``
        coefficients; on an immersed grid it multiplies the concrete
        open-area / volume fraction fields (IP-D4) — either way the
        halo tracer cannot follow, so the module declares its (order-2
        centered) FD-stencil halo here rather than being traced. The
        flat, unimmersed path stays fully halo-traced (None), exactly
        as before.
        """
        if self._column is None and self._immersed is None:
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
        q_factor = q.function_space.bare.factor(axis)
        if isinstance(q_factor, AverageSpace):
            # FV: the flux sits on the reconstructed face of the cell
            # average (CellAvg -> Right | Inner), transverse averages
            # untouched. On a walled axis the face adopts the
            # wall-normal velocity's Dirichlet tag (the same BC-sibling
            # substitution as the nodal path), so v.to(flux_space) is
            # the identity on the flux axis (the .to family matrix has
            # no same-node-set BC retag); _flux_divergence then strips
            # the tag before the flux_diff, whose Inner variant imposes
            # the exact-zero wall flux itself. On periodic axes the
            # face and the velocity share the interned BC-free factor,
            # so the substitution never fires (bitwise unchanged).
            recon = q.grid.dispatch.resolve("reconstruct", q_factor)
            space = q.function_space.replace(
                **{axis: recon.codomain(q_factor)})
            v_factor = v.function_space.bare.factor(axis)
            factor = space.bare.factor(axis)
            if factor is not v_factor and _bc_siblings(factor,
                                                       v_factor):
                space = space.replace(**{axis: v_factor})
            return space
        space = q.diff(axis).function_space
        v_factor = v.function_space.bare.factor(axis)
        factor = space.bare.factor(axis)
        if factor is not v_factor and _bc_siblings(factor, v_factor):
            space = space.replace(**{axis: v_factor})
        elif _outer_to_inner(v_factor, factor):
            # the advecting velocity is the diagnosed w on the
            # both-boundary Outer faces (hy.HydrostaticCore): restricting
            # it to the interior Inner flux faces drops its two boundary
            # values, so the advective flux through the top/bottom faces
            # is a structural zero (the fixed-domain closure under a
            # linear free surface). Tag the flux homogeneous-Dirichlet on
            # this axis — the same zero-wall-flux claim the wall-normal
            # velocity's Dirichlet Inner carries above — so its divergence
            # closes on it (and the transported tracer's mass is
            # conserved to roundoff).
            space = space.replace(**{axis: factor.mesh.nodal(
                NodeSet.INNER, bc=BC.DIRICHLET)})
        return space

    def _geometry_params(self, state: object) -> dict | None:
        """Collect the CURRENT mapping-parameter fields (mapped)."""
        if self._column is None:
            return None
        return mapping_params(state, state[self._advected[0]].grid)

    def _immersed_flux(
        self, flux: ScalarField, flux_space: object,  # noqa: ARG002
    ) -> ScalarField:
        r"""
        Weight one face flux by the open-area fraction (IP-D4).

        Description
        -----------
        ``F <- alpha_f (x) F`` with ``alpha_f = fraction(flux space)``
        (the min-rule face fraction of I0) — the cut-cell area weight
        that closes a face with ``alpha = 0`` as a free no-normal-flow
        wall and, on a face-aligned {0, 1} staircase, reproduces the
        walled model bit for bit. The fraction is fetched on the flux's
        own (possibly wall-Dirichlet-tagged) space, so the multiply is a
        plain same-space product. A no-op off an immersed grid.

        Parameters
        ----------
        flux : ScalarField
            The advective flux on the control-volume face.
        flux_space : object
            The flux space (unused; the flux carries its own tag).

        Returns
        -------
        ScalarField
            The open-area-weighted flux.
        """
        if self._immersed is None:
            return flux
        alpha = self._immersed.fraction(flux.function_space)
        return flux * alpha

    def _immersed_scale(
        self, res: ScalarField | None, q: ScalarField,
    ) -> ScalarField | None:
        r"""
        Divide the summed flux divergence by the cell volume fraction.

        Description
        -----------
        The masked divergence tendency ``-(1/(theta_c V_c)) sum_f +/-
        alpha_f A_f F_f``: the summed open-area-weighted flux divergence
        (already ``1/V_c``-scaled by ``flux_diff``) is divided by the
        cell volume fraction ``theta_c = fraction(q space)`` — the
        staggered fraction of the component's own control volume
        (min-rule for momentum, cell for a tracer). Guarded so a dry
        cell (``theta = 0``, numerator identically 0) stays exactly 0.
        The theta-weighted tendency conserves ``sum_c theta_c V_c q_c``
        to machine zero (the flux differences telescope). A no-op off an
        immersed grid.

        Parameters
        ----------
        res : ScalarField | None
            The accumulated ``-sum_axis`` flux divergence on ``q``'s
            space (None when no term contributed).
        q : ScalarField
            The advected component (its space carries the fraction).

        Returns
        -------
        ScalarField | None
            The wet-volume-scaled tendency (``res`` unchanged off an
            immersed grid, None passed through).
        """
        if self._immersed is None or res is None:
            return res
        theta = self._immersed.fraction(q.function_space)
        wet = theta.data > 0.0
        scaled = jnp.where(
            wet, res.data / jnp.where(wet, theta.data, 1.0), 0.0)
        return res.with_data(scaled)

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
        if isinstance(q.function_space.bare.factor(axis), AverageSpace):
            if self._column is not None:
                if _is_pure_average(q.function_space):
                    # mapped FV (stage F5): a pure CellAvg tracer takes
                    # the J-weighted conservative flux form, so its
                    # physical content is conserved to machine zero
                    return self._mapped_fv_divergence(
                        q, flux, axis, params)
                # a mapped C-grid velocity is cell-averaged only
                # transversely (Right(x) (x) CellAvg(y) (x) CellAvg(z));
                # momentum is not the FV-conserved quantity, so its
                # transverse-CellAvg axes take the *consistent* nodal
                # physical divergence below (correct to 2nd order, the
                # nodal-model numbers up to metric round-off)
            else:
                # FV (flat / walled): the exact discrete Gauss theorem.
                # flux_diff maps the face flux back onto the cell average
                # (Right | Inner -> CellAvg), telescoping to the boundary
                # fluxes — zero on a periodic wrap, zero at a wall (the
                # Inner variant pads exact-zero boundary fluxes).
                # integrate(flux_diff(F)) is machine zero, so the tracer
                # mass is conserved by construction. On a walled axis the
                # flux carries the velocity's adopted Dirichlet tag;
                # strip it first (the flux_diff Inner variant is BC-free
                # — it imposes the zero wall flux itself, the wall-face
                # DOFs are never read). The data is unchanged by the
                # retag, so conservation holds. The result already lands
                # on q's space (no retag).
                flux_factor = flux.function_space.bare.factor(axis)
                if not flux_factor.bc.is_free:
                    bcfree = flux_factor.mesh.inner
                    flux = flux.retag(
                        flux.function_space.replace(**{axis: bcfree}))
                    flux_factor = bcfree
                return q.grid.dispatch.resolve(
                    "flux_diff", flux_factor)[axis](flux)
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
        slope = grid.metric(space, f"d{mapped}_d{axis}", params=params)
        jac = grid.metric(space, f"d{mapped}_d{base}", params=params)
        coeff = _safe_ratio(slope, jac)
        corr = coeff * dcol
        registry = grid.dispatch
        for name in (base, axis):
            src = corr.function_space.bare.factor(name)
            dst = div.function_space.bare.factor(name)
            if src is dst or _bc_siblings(src, dst):
                continue
            # family-aware staggering hop (stage F5): the FV velocity
            # is cell-averaged transversely (Right(x) (x) CellAvg(y) (x)
            # CellAvg(z)), so the column correction lands on a nodal
            # face (Inner(z)) that must reduce back onto the *average*
            # cell CellAvg(z) — the "average" reconstruction, not the
            # nodal "interpolate" (-> Center). A nodal-Center target
            # keeps "interpolate" (bitwise unchanged).
            kind = ("average" if isinstance(dst, AverageSpace)
                    else "interpolate")
            corr = registry.resolve(kind, src)[name](corr)
        return (div - corr.retag(div)).retag(q)

    def _mapped_fv_divergence(
        self,
        q: ScalarField,
        flux: ScalarField,
        axis: str,
        params: dict | None,
    ) -> ScalarField:
        r"""
        J-weighted conservative FV flux divergence of one axis.

        Description
        -----------
        The genuinely conservative physical divergence of a
        ``CellAvg`` tracer on a mapped column ``m = M(b, params)``
        (stage F5): the same J-weighted flux form the mapped pressure
        operator uses (``mapped_pressure.py``), reused for advective
        transport so buoyancy content ``\int q\,\mathrm{d}V`` (the
        physical volume ``J`` times the computational measure) is
        conserved to machine zero. Per axis the flux form
        ``J\,\nabla_{\!phys}\!\cdot F = \sum_i D_i(J F_i) +
        D_b(F_b - \sum_i Z_i I(F_i))`` decomposes because every axis
        term telescopes independently under the ``J``-weighted sum
        (``D_i`` and ``D_b`` are exact flux differences ``flux_diff``):

        - ``axis == base``: ``(1/J) D_b(F_b)`` — the column flux
          difference, the walls closed by the Inner ``flux_diff``
          (exact-zero boundary flux, impermeability);
        - coupled ``axis``: ``(1/J)[D_i(J F_i) - D_b(Z_i I(F_i))]`` —
          the diagonal flux difference minus the cross flux
          interpolated onto the column face (:meth:`_mapped_fv_cross`);
        - uncoupled ``axis``: ``(1/J) D_i(J F_i)`` (``J`` and the flux
          difference commute, so it reduces to the plain divergence).

        Every metric derives through ``grid.metric`` at application
        with the current ``params`` (nothing cached, rules 2.3/3.8).
        The outer ``1/J`` (``dz/dzp``) rides the cell space; ``J``
        (``dzp/dz``) the flux (face) space; ``Z_i`` (``dzp/dx_i``) the
        corner. The result lands on ``q``'s space.

        Parameters
        ----------
        q : ScalarField
            The advected ``CellAvg`` tracer (the target cell space).
        flux : ScalarField
            The advective flux ``F_axis = v_axis q_face`` on the
            control-volume face.
        axis : str
            The flux axis.
        params : dict | None
            The dynamic mapping-parameter fields (None on a
            static-default mapped grid).

        Returns
        -------
        ScalarField
            The axis contribution to the physical divergence, on
            ``q``'s space.
        """
        mapped, base = self._column
        grid = q.grid
        registry = grid.dispatch
        # strip a wall Dirichlet tag: the flux_diff Inner variant
        # imposes the exact-zero wall flux itself (the wall face is a
        # structural zero of impermeability, never read from ghosts)
        flux_factor = flux.function_space.bare.factor(axis)
        if not flux_factor.bc.is_free:
            bcfree = flux_factor.mesh.inner
            flux = flux.retag(
                flux.function_space.replace(**{axis: bcfree}))
            flux_factor = bcfree
        if axis == base:
            div = registry.resolve("flux_diff", flux_factor)[axis](flux)
        else:
            jac = grid.metric(
                flux.function_space, f"d{mapped}_d{base}",
                params=params)
            div = registry.resolve(
                "flux_diff", flux_factor)[axis](flux * jac)
            if axis in self._corrections:
                div = div - self._mapped_fv_cross(
                    flux, axis, base, mapped, params)
        inv_j = grid.metric(
            div.function_space, f"d{base}_d{mapped}", params=params)
        return (div * inv_j).retag(q)

    def _mapped_fv_cross(
        self,
        flux: ScalarField,
        axis: str,
        base: str,
        mapped: str,
        params: dict | None,
    ) -> ScalarField:
        r"""
        Cross flux difference ``D_b(Z_i I(F_i))`` on the cell.

        Description
        -----------
        The column cross term of the coupled axis ``axis`` (stage F5):
        interpolate the ``axis``-face flux ``F_i`` onto the cell
        corners along the column (``CellAvg(base) -> Inner(base)``, the
        ``"interpolate"`` reconstruction), contract with the corner
        slope ``Z_i = dm/dx_i``, reduce back onto the column face along
        ``axis`` (``Right(axis) -> CellAvg(axis)``, the ``"average"``
        reconstruction — the FV ``face -> cell`` hop), and take the
        column flux difference (Inner ``flux_diff``, the wall closed by
        the exact-zero boundary flux). Mirrors the mapped pressure
        operator's ``_cross_to_column`` chain (transpose pairing is not
        needed here — advection is not self-adjoint — only telescoping
        conservation, which the flux difference gives for any consistent
        interpolation).

        Parameters
        ----------
        flux : ScalarField
            The ``axis``-face flux ``F_i`` (BC-free on ``axis``).
        axis : str
            The coupled axis.
        base : str
            The mapped column's base coordinate.
        mapped : str
            The mapped physical coordinate.
        params : dict | None
            The dynamic mapping-parameter fields.

        Returns
        -------
        ScalarField
            The cross flux difference on the cell (``CellAvg`` column).
        """
        grid = flux.grid
        registry = grid.dispatch
        # up_b: CellAvg(base) -> Inner(base), onto the cell corners
        corner = registry.resolve(
            "interpolate", flux.function_space.bare.factor(base),
        )[base](flux)
        slope = grid.metric(
            corner.function_space, f"d{mapped}_d{axis}", params=params)
        corner = corner * slope
        # down_i: Right(axis) -> CellAvg(axis), onto the column face
        column = registry.resolve(
            "average", corner.function_space.bare.factor(axis),
        )[axis](corner)
        # gate the reduced cross flux by the column (base) face open
        # fraction on an immersed grid (MI-D5, the composed cross-cell
        # closure): the cross flux is the vertical interpolation of the
        # *horizontal* face flux, so it does not inherit the base face's
        # open-area weight from ``_immersed_flux`` (which weighted the
        # horizontal faces). Left un-gated it stays non-zero at a wet/dry
        # vertical interface, where ``_immersed_scale`` then truncates the
        # column and the telescoping sum leaks (conservation of
        # ``sum theta J V q`` drifts). The base-face fraction is exactly
        # zero across a wet/dry vertical face (the min rule), so gating
        # here makes the cross flux vanish there and the wet-region
        # telescoping close. A no-op without a mask or on an all-wet
        # column (``alpha == 1``), so the mapped-only conservation is
        # byte-identical.
        if self._immersed is not None:
            alpha = self._immersed.fraction(column.function_space)
            column = column * alpha
        # the column flux difference closes the walls itself (BC-free
        # Inner flux_diff pads exact-zero boundary fluxes)
        return registry.resolve(
            "flux_diff", column.function_space.bare.factor(base),
        )[base](column)

    def _advect(
        self, state: object, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        r"""Flux-form transport of every advected component (Ro-scaled).

        When the surface closure is active (``_surface_flux_on``, the
        resolved tri-state) the constancy-preserving correction
        :math:`-q\,A(\mathbf 1)` is added: it restores the boundary-face
        advective flux the ``Outer -> Inner`` restriction drops (module
        docstring), so :math:`A` annihilates a constant in **every**
        cell, the surface cell included. When it is off the branch is
        skipped and the tendency is byte-for-byte the plain flux-form
        one (the nodal-velocity default). The correction rides this
        term's own EXPLICIT stage. **Note:** the background-split terms
        (``_advect_perturbation`` / ``_advect_background``) are *not*
        corrected — a background velocity with a nonzero surface value
        would still drop its boundary face; today no such background
        exists.

        The correction is evaluated as the **direct boundary term**
        (:meth:`_surface_correction`, the slice form) wherever
        :math:`A(\mathbf 1)` is provably boundary-only — its
        ``O(N^2)`` surface trace replaces the ``O(N)`` full-3D
        telescoping sum, reclaiming most of the closure's step
        overhead. Where the boundary-only property does not hold (a
        mapped column, a staggered momentum component on an immersed
        grid, or a forced closure on a grid without the ``Outer`` seam)
        the exact full-3D form is kept (byte-identical to the pre-slice
        behavior).
        """
        ro = ctx.params[fr.model.params.SCALING_ROSSBY]
        params = self._geometry_params(state)
        out: dict[str, ScalarField] = {}
        for qname in self._advected:
            q = state[qname]
            tend = ro * self._immersed_scale(
                self._transport(state, q, params), q)
            if self._surface_flux_on:
                tend = self._surface_correction(
                    state, q, tend, ro, params)
            out[qname] = tend
        return out

    def _transport(
        self, state: object, q: ScalarField, params: dict | None,
    ) -> ScalarField:
        r"""Per-axis flux loop: the advective divergence of ``q``.

        Description
        -----------
        For every advecting axis it forms the flux ``v_face * face(q)``,
        weights it by the open-area fraction (immersed), and differences
        it back onto ``q``'s space, accumulating ``-sum_axis`` — the
        advective divergence, **before** the Rossby and volume-fraction
        scalings the caller applies.

        Parameters
        ----------
        state : object
            The current state (supplies the advecting velocities).
        q : ScalarField
            The transported quantity.
        params : dict | None
            The dynamic mapping-parameter fields (None on flat grids).

        Returns
        -------
        ScalarField
            The accumulated advective divergence on ``q``'s space.
        """
        res = None
        for axis, vname in self._axis_velocity:
            v = state[vname]
            flux_space = self._flux_space(q, v, axis)
            v_face = self._velocity_face(v, flux_space)
            flux = v_face * self._face_value(
                q, v_face, axis, flux_space)
            flux = self._immersed_flux(flux, flux_space)
            # the divergence lands back on q's wall-tagged space
            # (flat grids: literally flux.diff(axis).retag(q))
            divergence = self._flux_divergence(
                q, flux, axis, params)
            res = -divergence if res is None else res - divergence
        return res

    # ------------------------------------------------------------
    #  The constancy-preserving surface closure (H7)
    # ------------------------------------------------------------
    def _surface_correction(
        self,
        state: object,
        q: ScalarField,
        tend: ScalarField,
        ro: object,
        params: dict | None,
    ) -> ScalarField:
        r"""Subtract the surface correction :math:`q\,A(\mathbf 1)`.

        Description
        -----------
        Routes between two exact spellings of the constancy-preserving
        correction:

        - the **slice form** (:meth:`_surface_boundary_term` +
          :meth:`_apply_correction`) — used wherever
          :math:`A(\mathbf 1)` is provably boundary-only, i.e. the
          advecting velocity is discretely divergence-free and drops a
          boundary face (:meth:`_slice_valid`). :math:`A(\mathbf 1)`
          then telescopes to the dropped surface face in the boundary
          row and is machine-zero elsewhere, so the direct 2D trace of
          that surface term is exact and cheap.
        - the **full-3D form** (:meth:`_correction_full`) — the
          pre-slice AXPY ``tend - q * A(1)`` with the un-scaled
          :math:`A(\mathbf 1)` accumulated over the axes, the exact
          scheme-applied-to-a-constant. Kept for the cases where the
          slice's 2D trace is not exact: a mapped column (the physical
          divergence is not the flux-form continuity the diagnosis
          enforces), a staggered momentum component on an immersed grid
          (the momentum control volume's masked continuity is not
          divergence-free near a cut side wall), a staggered momentum
          component under a **biased** scheme (its ``(order - 1)``-point
          velocity interpolation relocates the surface ``w`` onto the
          momentum column differently from the slice's two-point ``.to``,
          so the slice's top-row ``A(1)`` would use the wrong surface
          ``w``), and a forced closure on a grid carrying no ``Outer``
          seam at all (``div(v)`` is a genuine interior field).
        """
        seam = tuple(
            (axis, vname) for axis, vname in self._axis_velocity
            if _is_surface_seam(state[vname], axis))
        if seam and self._slice_valid(q):
            for axis, vname in seam:
                a1 = self._surface_boundary_term(q, state[vname], axis)
                tend = self._apply_correction(tend, q, a1, axis, ro)
            return tend
        corr = self._correction_full(state, q, params)
        return tend - q * (ro * self._immersed_scale(corr, q))

    def _slice_valid(self, q: ScalarField) -> bool:
        r"""Whether ``q``'s slice-form :math:`A(\mathbf 1)` is exact.

        Description
        -----------
        ``True`` for a cell-collocated tracer on any flat grid (it needs
        no horizontal relocation of the surface ``w``, so the slice is
        exact for every scheme), and for a staggered momentum component
        on a flat, unimmersed grid **only** when the scheme relocates the
        surface velocity onto the momentum column exactly as its own
        advecting-velocity face does (:attr:`_slice_relocation_exact` —
        the centered scheme's two-point ``.to``). ``False`` on a mapped /
        moving column, for a staggered momentum component on an immersed
        grid (the masked momentum continuity is not divergence-free near
        a cut side wall), and for a staggered momentum component under a
        biased scheme (its ``(order - 1)``-point velocity interpolation
        does not match the slice's two-point ``.to``, so the slice's
        top-row ``A(1)`` would use the wrong surface ``w`` — a genuine
        constancy break at ``order > 3``). The false branches take the
        exact full-3D fallback — see :meth:`_surface_correction`.
        """
        if self._column is not None:
            return False
        if _is_cell_collocated(q.function_space):
            return True
        return self._immersed is None and self._slice_relocation_exact

    def _correction_full(
        self, state: object, q: ScalarField, params: dict | None,
    ) -> ScalarField:
        r"""Accumulate the un-scaled full-3D :math:`A(\mathbf 1)` fallback.

        Description
        -----------
        Per advecting axis, ``face(1) == 1`` exactly (every
        reconstruction preserves constants), so the correction flux is
        the (immersed-weighted) interpolated velocity face itself: a
        lean divergence of the advecting velocity, one extra
        ``_flux_divergence`` per axis, accumulated as ``-sum_axis``. The
        caller scales it (``ro`` and the wet-volume fraction) and
        subtracts ``q`` times it. Byte-identical to the pre-slice
        in-loop accumulation.
        """
        corr = None
        for axis, vname in self._axis_velocity:
            v = state[vname]
            flux_space = self._flux_space(q, v, axis)
            v_face = self._velocity_face(v, flux_space)
            cflux = self._immersed_flux(v_face, flux_space)
            cdiv = self._flux_divergence(q, cflux, axis, params)
            corr = -cdiv if corr is None else corr - cdiv
        return corr

    def _surface_boundary_term(
        self, q: ScalarField, v: ScalarField, axis: str,
    ) -> ScalarField:
        r"""Build the 2D surface term of :math:`A(\mathbf 1)` on ``q``'s row.

        Description
        -----------
        The boundary row of the vertical ``_flux_divergence`` restricted
        to its dropped surface face: ``alpha_top * w(0) / dz_top``, a 2D
        ``Trace`` on ``q``'s co-located ``Center`` boundary row.

        - ``w(0)`` is the surface value of the advecting velocity
          (``v.trace``, the ``Outer`` boundary DOF), horizontally
          interpolated onto ``q``'s flux column (``.to`` — the trace of
          a tensor-product interpolation commutes with the interpolation
          of the trace, so this is exact). The ``Outer``-parent surface
          face value is relocated onto ``q``'s ``Center`` cell row (the
          face -> cell hop ``_flux_divergence`` performs implicitly)
          through the sanctioned ``as_profile`` / ``adopt`` retag bridge.
        - ``alpha_top`` (immersed only) is the surface-face open-area
          weight, which the diagnosed ``w`` carries as the **top cell**
          fraction (``HydrostaticCore._masked_w_faces`` overrides the
          min-rule ``Outer`` fraction there), so it is
          ``fraction(q.space)`` traced at the wall; it cancels against
          the wet-volume divide of :meth:`_immersed_scale_2d`, leaving
          ``w(0)/dz`` on a wet cell and exactly ``0`` on a dry one.
        - ``dz_top`` is the primal cell width the flux difference divides
          by (``grid.measure`` on ``q``'s cell space — static mesh
          geometry, the constant ``dx`` on a uniform mesh and the
          stretched cell width on a mapped-``z`` mesh), traced at the
          wall. The divide is VJP-sealed (``_safe_ratio``, the
          double-``jnp.where``): mandatory for the masked-singularity
          shape even though the cell width is strictly positive on every
          valid row (AGENTS.md differentiability policy).

        Only reached on a flat grid (``self._column is None``), so no
        mapped ``1/J`` factor and no ``params`` enter (:meth:`_slice_valid`).
        """
        flux_space = self._flux_space(q, v, axis)
        outer_factor = v.function_space.bare.factor(axis)
        outer_space = flux_space.bare.replace(
            **{axis: outer_factor}).with_layout(flux_space.layout)
        w0 = v.trace(axis, Side.HIGH)
        target = outer_space.bare.replace(
            **{axis: w0.function_space.bare.factor(axis)}).with_layout(
            flux_space.layout)
        # horizontal interpolation onto q's flux column (identity for a
        # collocated tracer, Center -> Right for a staggered component),
        # then relocate the Outer surface face onto q's Center cell row
        w0 = w0.to(target).as_profile(axis).adopt(
            axis, NodeSet.CENTER, Side.HIGH)
        num = w0
        if self._immersed is not None:
            num = self._immersed.fraction(q.function_space).trace(
                axis, Side.HIGH) * w0
        dz_top = q.grid.measure(q.function_space, axis).trace(
            axis, Side.HIGH)
        return _safe_ratio(num, dz_top)

    def _apply_correction(
        self,
        tend: ScalarField,
        q: ScalarField,
        a1: ScalarField,
        axis: str,
        ro: object,
    ) -> ScalarField:
        r"""Subtract ``q * ro * scale(A(1)|top)`` via the selected lowering.

        Description
        -----------
        Two exact lowerings: ``"embed"`` materializes ``A(1)|top``
        sparsely back into its parent row and runs the pre-slice full-3D
        ``q * A(1)`` AXPY (nodal only — an FV component's ``embed`` lands
        on the co-located nodal ``Center``, not ``CellAvg``, so it falls
        back to ``"scatter"``); ``"scatter"`` keeps everything 2D and
        row-scatter-adds the negated correction into ``tend``'s boundary
        row. Both give the same tendency. The lowering is chosen
        per-scheme (``scatter`` for the centered flux form, ``embed`` for
        the biased/upwind family) via the class ``_surface_flux_lowering``
        default, globally overridable by the module-level
        ``_SURFACE_FLUX_LOWERING`` (an explicit string forces one lowering
        for all schemes).
        """
        lowering = (self._surface_flux_lowering
                    if _SURFACE_FLUX_LOWERING is None
                    else _SURFACE_FLUX_LOWERING)
        if lowering == "embed" and not _is_average_space(q.function_space):
            # embed lands on the BC-free parent Center row; retag onto q's
            # own (possibly wall-tagged) cell factor for the AXPY (data
            # untouched — the boundary row is a structural interior DOF)
            corr3d = a1.embed(axis)
            corr3d = corr3d.retag(corr3d.function_space.replace(
                **{axis: q.function_space.bare.factor(axis)}))
            return tend - q * (ro * self._immersed_scale(corr3d, q))
        corr2d = q.trace(axis, Side.HIGH) * a1
        corr2d = self._immersed_scale_2d(corr2d, q, axis)
        return fr.spatial.operators.scatter_add(tend, -(ro * corr2d))

    def _immersed_scale_2d(
        self, corr2d: ScalarField, q: ScalarField, axis: str,
    ) -> ScalarField:
        r"""Divide a 2D boundary correction by the wet top-cell fraction.

        Description
        -----------
        The 2D twin of :meth:`_immersed_scale`: the surface-row divide by
        the cell volume fraction ``theta_top = fraction(q.space)`` traced
        at the wall, guarded so a dry top cell (``theta == 0``, numerator
        identically ``0``) stays exactly ``0`` — the double-``jnp.where``
        that seals the VJP under ``jax.grad``. A no-op off an immersed
        grid.
        """
        if self._immersed is None:
            return corr2d
        theta = self._immersed.fraction(q.function_space).trace(
            axis, Side.HIGH)
        wet = theta.data > 0.0
        scaled = jnp.where(
            wet, corr2d.data / jnp.where(wet, theta.data, 1.0), 0.0)
        return corr2d.with_data(scaled)

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
            flux = self._immersed_flux(flux, flux_space)
            divergence = self._flux_divergence(q, flux, axis,
                                               params)
            res = -divergence if res is None else res - divergence
        return self._immersed_scale(res, q)

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
            flux = self._immersed_flux(flux, flux_space)
            divergence = self._flux_divergence(q, flux, axis,
                                               params)
            res = -divergence if res is None else res - divergence
        return self._immersed_scale(res, q)

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
    surface_flux : bool | None, optional
        Tri-state control of the constancy-preserving surface closure
        (advect through the top/bottom boundary faces with the one-sided
        face value; see the ``_FluxFormAdvection`` Description). ``True``
        / ``False`` force it; the default ``None`` auto-resolves at bind
        — on iff an advecting velocity sits on the ``Outer`` faces (the
        hydrostatic diagnosed ``w``), off (bitwise unchanged) otherwise
        (default: None).
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
    surface_flux : bool | None, optional
        Tri-state control of the constancy-preserving surface closure
        (advect through the top/bottom boundary faces with the one-sided
        face value; see the ``_FluxFormAdvection`` Description). ``True``
        / ``False`` force it; the default ``None`` auto-resolves at bind
        — on iff an advecting velocity sits on the ``Outer`` faces (the
        hydrostatic diagnosed ``w``), off (bitwise unchanged) otherwise
        (default: None).
    """

    _weighting: ClassVar[Literal["linear", "weno"]] = "linear"

    #: the order-wide biased windows are legal on a bounded axis
    #: through their graded near-wall closure, installed at bind
    _supports_walled: ClassVar[bool] = True

    #: the biased reconstructions are computational-coordinate rows;
    #: mapped columns need a mapped-aware variant — future work
    #: (taught rejection at bind)
    _supports_mapped: ClassVar[bool] = False

    #: the wide biased windows reach across dry cells, but the graded-mask
    #: near-wall closure keys the ladder on the wet region (GA-D1..D6):
    #: at bind the ``boundary="graded"`` kernels install and the mask
    #: path selects, per output face, the widest rung whose union window
    #: is entirely wet (`graded.apply_graded_mask`)
    _supports_immersed: ClassVar[bool] = True

    #: the biased velocity face is an ``(order - 1)``-point centered
    #: interpolation (:meth:`_velocity_face`), not the two-point ``.to``
    #: the surface-flux slice's trace uses to relocate ``w`` onto a
    #: staggered column, so the slice's top-row ``A(1)`` for a momentum
    #: component would use the wrong surface ``w`` (exact only at
    #: ``order == 3``, where ``order - 1 == 2``). Staggered momentum
    #: therefore takes the exact full-3D correction (:meth:`_slice_valid`);
    #: a cell-collocated tracer keeps the cheap slice (no relocation).
    _slice_relocation_exact: ClassVar[bool] = False

    #: the biased/upwind family (incl. WENO) wins with ``"embed"`` when
    #: ``_SURFACE_FLUX_LOWERING`` is ``None`` -- scatter costs +16% at
    #: 2048^2 x 64 where embed is +5.8% vs off (2026-07-18 A/B, §9).
    #: Provisional: that A/B predates the staggered-momentum reroute
    #: above (only cell-collocated tracers still take the slice here),
    #: so the biased default awaits a post-reroute re-measure
    _surface_flux_lowering: ClassVar[str] = "embed"

    def __init__(
        self,
        order: int = 3,
        *,
        background: Mapping[str, Callable | float] | None = None,
        wall: Literal["upwind1", "centered2"] = "upwind1",
        surface_flux: bool | None = None,
    ) -> None:
        """Build the biased reconstruction pairs for ``order``."""
        super().__init__(background=background, surface_flux=surface_flux)
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
        # the average-family twins, for a CellAvg tracer in the same
        # (possibly mixed) model: CellAvg -> Right | Inner, the same
        # kernels and graded closure on the primal cell frame
        self._fv_left = _FVBiasedReconstruction(order, "left",
                                                weighting, boundary,
                                                wall)
        self._fv_right = _FVBiasedReconstruction(order, "right",
                                                 weighting, boundary,
                                                 wall)
        self._fv_lin_left = _FVBiasedReconstruction(order, "left",
                                                    "linear", boundary,
                                                    wall)
        self._fv_lin_right = _FVBiasedReconstruction(order, "right",
                                                     "linear", boundary,
                                                     wall)

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
        if self._walled or self._immersed is not None:
            # the graded signature also carries the mask-keyed closure
            # on an immersed grid (GA-D4); ``_check_walled_extent`` does
            # not apply on the mask path (any alpha>0 face has two wet
            # neighbours, so the bottom rung is always legal)
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

    @property
    def extra_halo(self) -> HaloSpec | None:
        """The biased kernels' ``order // 2 + 1`` halo on an immersed grid.

        Description
        -----------
        On an immersed grid the mask-keyed reconstruction reads the
        ``order // 2 + 1`` biased window per axis and halo tracing is
        disabled (the concrete pre-mask / selectors the trace cannot
        follow), so the demand is declared here (GA-D3) — wider than the
        base's order-2 centered fraction stencil for ``order = 5``. Off an
        immersed grid the biased schemes reject a mapped column at bind, so
        there is no extra halo (the flat path stays fully halo-traced).
        """
        if self._immersed is None:
            return None
        return HaloSpec(dict.fromkeys(
            self._halo_axes, self._order // 2 + 1))

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

        The trailing bridge is the walled- and FV-grid seam: nodal
        operator outputs are BC-free, so on a walled axis the
        interpolated velocity lands on the BC-free sibling of the flux
        space's wall-tagged factor and adopts the tag (the exact-zero
        wall flux the divergence closes on); on an FV flux axis (a
        velocity's own staggered axis whose flux is the co-located
        ``CellAvg``) the ``Center`` interpolation output crosses to
        ``CellAvg`` through the 2nd-order deconvolve identity
        (`_to_flux_space`). Both are bitwise no-ops on the numbers, so
        the FV velocity face is bitwise the nodal one — the order-coupled
        interpolation is kept whatever the family, unlike a plain ``.to``
        (which would drop to the 2-point row on the flux axis). On a
        periodic nodal axis the factors are the same interned object and
        the bridge returns ``self`` — the periodic path is bitwise
        unchanged.

        On an FV model the velocity's transverse factors are ``CellAvg``;
        where an axis needs a genuine half-cell interpolation (its
        face-ness differs from the flux face) the ``CellAvg`` operand is
        first deconvolved to its co-located ``Center`` (a bitwise
        identity) so the nodal ``_interp`` row applies, then the bridge
        deconvolves back. Co-located and BC-only axes are left to the
        bridge (no interpolation).

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
            src = result.function_space.bare.factor(axis)
            dst = bare.factor(axis)
            if _outer_to_inner(src, dst):
                # the diagnosed w already lives on the flux faces'
                # superset (Outer ⊃ Inner): the exact restriction, not
                # the order-coupled symmetric interpolation — there is
                # nothing to interpolate, the interior faces are shared
                # nodes (hydrostatic vertical leg).
                result = result.to(dst)
                continue
            if _is_face_factor(src) == _is_face_factor(dst):
                # co-located (or BC-only): no half-cell interpolation;
                # the trailing bridge deconvolves / retags it
                continue
            if isinstance(src, CellAvg):
                # bring the FV operand to its nodal skeleton so the
                # order-coupled row applies (a bitwise deconvolve)
                result = result.to(src.mesh.center)
            result = self._interp[axis](result)
        return _to_flux_space(result, flux_space)

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
        left, right = self._biased_pair(q, axis)
        positive = v_face + abs(v_face)
        return Where()(positive,
                       _to_flux_space(left(q), flux_space),
                       _to_flux_space(right(q), flux_space))

    def _biased_pair(
        self, q: ScalarField, axis: str,
    ) -> tuple[object, object]:
        """Return the biased reconstruction pair bound to ``axis``.

        Description
        -----------
        Selects the average-family reconstruction pair
        (`_FVBiasedReconstruction`) for a ``CellAvg`` tracer and the
        nodal pair (`_BiasedFaceReconstruction`) otherwise, so one
        module transports a mixed nodal-plus-FV advected set with the
        family-correct reconstruction per component.
        """
        if isinstance(q.function_space.bare.factor(axis),
                      AverageSpace):
            return self._fv_left[axis], self._fv_right[axis]
        return self._left[axis], self._right[axis]

    def _linear_pair(
        self, q: ScalarField, axis: str,
    ) -> tuple[object, object]:
        """Return the linear-weight pair bound to ``axis``.

        Description
        -----------
        The optimal-weight (smooth-limit) twin of `_biased_pair` for
        the linear background transport: the average-family pair for a
        ``CellAvg`` tracer, the nodal pair otherwise.
        """
        if isinstance(q.function_space.bare.factor(axis),
                      AverageSpace):
            return self._fv_lin_left[axis], self._fv_lin_right[axis]
        return self._lin_left[axis], self._lin_right[axis]

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
        left, right = self._linear_pair(q, axis)
        positive = v_face + abs(v_face)
        return Where()(positive,
                       _to_flux_space(left(q), flux_space),
                       _to_flux_space(right(q), flux_space))


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

    def _install_kernels(
        self, boundary: Literal["none", "graded"],
    ) -> None:
        """Build the biased pair, then the selected-input reconstruction.

        Description
        -----------
        Extends `UpwindAdvection._install_kernels` with the WENO
        one-pass kernel `_SelectedFaceReconstruction` (order / boundary
        / wall matched to the biased `_left` pair it supersedes in
        `_face_value`) and its ``family="fv"`` twin for ``CellAvg``
        tracers; the linear background pair (`_lin_left` /
        `_lin_right`) and the velocity interpolation stay as the base
        builds them, so the linear paths are byte-identical.

        Parameters
        ----------
        boundary : Literal["none", "graded"]
            The variant to install (see the base method).
        """
        super()._install_kernels(boundary)
        wall = self._wall if boundary == "graded" else "upwind1"
        self._selected = _SelectedFaceReconstruction(
            self._order, boundary, wall)
        self._fv_selected = _SelectedFaceReconstruction(
            self._order, boundary, wall, family="fv")

    def _face_value(
        self,
        q: ScalarField,
        v_face: ScalarField,
        axis: str,
        flux_space: object,
    ) -> ScalarField:
        """
        WENO upwind face value via the selected-input reconstruction.

        Description
        -----------
        The one-pass override of `UpwindAdvection._face_value` (design
        ``research/stencil_lowering.md`` §6): ``positive = v_face +
        |v_face|`` is nonzero exactly where the face velocity is
        positive (ties at zero take the right-biased side, old-stack
        parity), and `_SelectedFaceReconstruction` selects the union
        window taps on that sign before ONE left WENO reconstruction —
        exact to reversed-summation ulps against the both-then-select
        pair, and roughly -40% on the WENO step (the nonlinear weights
        run once, not once per bias). Only WENO overrides here: the
        linear `UpwindAdvection._face_value` keeps the both-then-select
        spelling (no divides to save, measured slower one-path), and
        the linear background `_linear_face_value` is inherited
        unchanged. An average-family (``CellAvg``) tracer routes
        through the ``family="fv"`` selected kernel (the
        `_FVBiasedReconstruction` frame, the same window numbers on
        the primal cell frame), so the FV/nodal bitwise tendency
        identity of the periodic box survives the one-pass spelling.

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
            The WENO upwind-biased face value of ``q``.
        """
        if getattr(q.grid, "immersed", None) is not None:
            # the selected-input one-pass optimization is deferred on
            # immersed grids (GA-D2): the both-then-select spelling routes
            # each bias through the mask-keyed reconstruction (the sign
            # select stays outermost, every rung is mask-graded), an exact
            # designed-for perf lever for a later band-restricted pass
            return super()._face_value(q, v_face, axis, flux_space)
        selected = self._selected
        if isinstance(q.function_space.bare.factor(axis),
                      AverageSpace):
            selected = self._fv_selected
        positive = v_face + abs(v_face)
        return selected(positive, q, axis, flux_space)
