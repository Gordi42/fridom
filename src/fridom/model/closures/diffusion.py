r"""
Harmonic and biharmonic diffusion closures (mixing and friction).

Description
-----------
The framework-level generic diffusion closure family (the port of
the old ``fr.modules.closures.HarmonicDiffusion`` /
``BiharmonicDiffusion`` and their per-model mixing/friction wrappers).
The harmonic operator on a target :math:`q` is

.. math::
    \mathcal{H}(q) = \nabla \cdot (\mathbf{A} \cdot \nabla q),
    \qquad \mathbf{A} = \mathrm{diag}(\kappa_1, \dots, \kappa_n),

spelled as pure field arithmetic — per axis, a staggered first
difference, the coefficient inside the flux, and the difference back
onto :math:`q`'s space — so the stencils are halo-traced normally.
Following Griffies & Hallberg (2000), the biharmonic operator
iterates the harmonic one with square-rooted coefficients,

.. math::
    \mathcal{B}(q) = -\mathcal{H}_{\sqrt{\kappa}}(
        \mathcal{H}_{\sqrt{\kappa}}(q)),

so the user-facing :math:`\kappa_i \ge 0` is the true biharmonic
coefficient (units :math:`\mathrm{m^4/s}`).

Targets are role-driven (D1.4, `ClosureBase`): the mixing classes
default to ``fr.roles.TRACER``, the friction classes to the
``Velocity`` family; ``fields=`` / ``exclude=`` override by name. The
old ``ENABLE_MIXING`` / ``ENABLE_FRICTION`` flags are gone.
Coefficients are **provided parameters** (dynamic leaves) published
under the canonical names below, so ``model.update_parameters``
sweeps them without a recompile; each accepts a scalar (or
``fr.Ramp``) or a name-keyed per-field mapping (plain numbers),
validated at bind. Anisotropy is horizontal/vertical: the optional
``*_v`` coefficient acts along the ``vertical`` coordinate (default
``"z"``), the main coefficient along every other axis (and along all
axes when ``*_v`` is omitted) — the old ``(kh, kh, kv)`` /
``(ah, av)`` spellings.

**Walled grids** (bounded mesh factors) are supported per axis. The
two-pass flux chain cannot close on a bounded BC-free face (R1: the
second difference needs the wall-face flux, which a BC-free ``Inner``
does not define), so along a bounded axis the interior flux is
retagged onto its Dirichlet-tagged ``Inner`` sibling — a true
wall-value claim, the wall flux *is* zero — and the closing
difference telescopes with that structural exact-zero wall flux (the
advection flux-space BC-sibling precedent). This is the **no-flux**
tracer wall and the **free-slip** velocity wall (zero tangential wall
stress). The wall-normal velocity component (staggered along the
walled axis, space ``Inner[Dirichlet]``) closes on its own tag with
no slip choice. The **no-slip** velocity wall (``slip="no"`` on the
friction closures) adds the wall-adjacent-cell correction
:math:`-2\,\nu\,u_1/\Delta n^2` — the factor-of-2 ghost against the
zero wall value across the half cell — on top of the free-slip chain.
Periodic axes take the exact periodic path (bit-identical to a
purely periodic grid). The biharmonic closures apply the same wall
treatment on both Laplacian passes (Griffies & Hallberg).

Both discretization families are supported. The **nodal** family
(``Center`` cells, ``Inner`` velocity faces) is above. The
**finite-volume** family (``CellAvg`` cells, the nonhydro2 default) is
the *same chain* in conservative face-flux form: on a walled axis the
FV C-grid ``diff`` profile staggers the cell average onto the nodal
``Inner`` face (``CellAvg -> Inner``, ``FaceDifference``), so the
interior flux lands on the very ``Inner`` face the nodal chain uses.
The identical retag onto the Dirichlet ``Inner`` sibling then closes
the divergence (``Inner[Dirichlet] -> CellAvg``, ``FluxDifference``)
with the structural exact-zero wall flux — the FV no-flux tracer / the
FV free-slip velocity wall — and no-slip adds the same wall-adjacent
correction :math:`-2\,\nu\,u_1/\Delta n^2` with the wall cell's own
measure width (the ghost fills are bit-identical to ``Center``). This
face-exposing FV chain needs the FV C-grid dispatch profile (a
``family="fv"`` model assembly); on a raw grid without it the
collocated ``FVDerivative`` (``CellAvg -> CellAvg``) never surfaces the
flux as a face field and cannot close a wall flux, so a walled
``CellAvg`` target is rejected at bind with a taught error. Mapped /
stretched FV columns take the same **along-sigma** semantics as the
nodal family below (measure-weighted, order 2): the FV cell-width
measure divide gives the metrically-exact physical operator and the
measure-weighted no-flux integral machine zero.

**Mapped / terrain-following grids** (stretched ``MappedIntervalMesh``
factors, or a ``Grid(..., mapping=CoordinateMapping(maps=...))`` chart)
bind and run through the same per-axis chain: the operator is
**along-coordinate** (the ROMS-default ``MIX_S_UV`` convention). Each
``q.diff(axis)`` is the *computational* (along-coordinate) derivative,
which divides by the codomain's own measure (the two-point node
spacing, order 2), so:

- On a **stretched** bounded column the measure *is* the physical cell
  spacing, so the chain is the physical :math:`\partial_z(k\,\partial_z
  q)` and is **metrically exact at order 2** (the same-row-Jacobian
  argument, ``stretched_terrain_combined.md`` §5); the measure-weighted
  telescoping keeps the no-flux tracer integral machine-zero.
- On a **terrain-following** chart the operator acts on
  *constant-coordinate* (constant-:math:`\sigma`) surfaces: the chart
  factor :math:`H(x)` lives in ``grid.metric`` and **never enters**
  ``diff``/``measure``, so the per-axis legs carry no cross terms and
  no :math:`H(x)` coupling — "horizontal" mixing **tilts with the
  terrain**. This is accepted practice for **viscosity** (ROMS
  default), but a physics error for **tracer** mixing over steep
  slopes (spurious diapycnal mixing); the geopotential/rotated
  operator that fixes tracer orientation is future work (record §3.6
  options A/C). The ``vertical=`` split is by **axis name**, so on a
  terrain grid ``nu_v``/``kappa_v`` (with e.g. ``vertical="sigma"``)
  acts along the **column coordinate**, not the physical vertical.

**Immersed (cut-cell) grids** are supported by the **harmonic**
mixing/friction closures only (``_supports_immersed``, CL-D1), in the
IP-D4 fraction spelling: each interface stress flux is weighted by the
open-area fraction :math:`\alpha_f` and the summed divergence is
divided by the wet cell fraction :math:`\theta_c` (sealed). The
min-rule :math:`\alpha_f = 0` across a wet/dry face zeroes the cut-face
stress — **free-slip** at the immersed boundary (CL-D3, the mainstream
mask convention) — so on a face-aligned :math:`\{0, 1\}` staircase the
tendency reproduces the walled model to machine zero, and the
:math:`\theta`-weighted content conserves exactly (the wet-region flux
differences telescope, CL-D4). The immersed :math:`\alpha` acts on the
cut faces while the wall retag acts on domain walls, so the two
compose. The **biharmonic** family, Smagorinsky, and VerticalMixing
keep the per-closure reject (their wide / implicit / nonlinear stencils
are §5 deferrals); ``slip='no'`` on an immersed grid is likewise
rejected (no-slip immersed drag is a §5 deferral, the mask-keyed
side-drag term).

Forward is exact as above, and reverse-mode ``jax.grad`` is finite on
every supported grid kind — the stretched-column boundary-face
measure divide is VJP-sealed in the spatial layer
(``divide_by_codomain_measure``); the mirrored test shards carry the
autodiff regressions.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp

from fridom.framework.utils import jaxify, modify_array
from fridom.model.closures.base import ClosureBase
from fridom.model.errors import AssemblyError
from fridom.model.parameters import (
    ParameterDeclaration,
    leaf,
)
from fridom.model.params import ParamName
from fridom.model.roles import TRACER, Velocity
from fridom.model.terms import TendencyTerm, Treatment
from fridom.model.time_dependent import TimeDependent
from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    from fridom.model.context import StepContext
    from fridom.model.field_table import FieldTable
    from fridom.model.roles import Role
    from fridom.spatial.fields.scalar_field import ScalarField


# ================================================================
#  Canonical parameter names (one provider per name)
# ================================================================
MIXING_KAPPA: ParamName = ParamName(
    "mixing.kappa", units="m^2/s",
    hint="provided by fr.closures.HarmonicDiffusion(kappa=...)")

MIXING_KAPPA_V: ParamName = ParamName(
    "mixing.kappa_v", units="m^2/s",
    hint="provided by fr.closures.HarmonicDiffusion(kappa_v=...)")

MIXING_KAPPA4: ParamName = ParamName(
    "mixing.kappa4", units="m^4/s",
    hint="provided by fr.closures.BiharmonicDiffusion(kappa=...)")

MIXING_KAPPA4_V: ParamName = ParamName(
    "mixing.kappa4_v", units="m^4/s",
    hint="provided by fr.closures.BiharmonicDiffusion(kappa_v=...)")

FRICTION_NU: ParamName = ParamName(
    "friction.nu", units="m^2/s",
    hint="provided by fr.closures.HarmonicFriction(nu=...)")

FRICTION_NU_V: ParamName = ParamName(
    "friction.nu_v", units="m^2/s",
    hint="provided by fr.closures.HarmonicFriction(nu_v=...)")

FRICTION_NU4: ParamName = ParamName(
    "friction.nu4", units="m^4/s",
    hint="provided by fr.closures.BiharmonicFriction(nu=...)")

FRICTION_NU4_V: ParamName = ParamName(
    "friction.nu4_v", units="m^4/s",
    hint="provided by fr.closures.BiharmonicFriction(nu_v=...)")


# ================================================================
#  Coefficient coercion (construction-time, host-side)
# ================================================================
def _coerce_scalar(
    value: object, arg: str, owner: str, *, nonnegative: bool,
) -> object:
    """Coerce one scalar coefficient; validate biharmonic signs."""
    if isinstance(value, TimeDependent):
        return value
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise TypeError(
            f"{owner}: {arg}= takes a number, an fr.Ramp, or a "
            f"name-keyed per-field mapping; got {value!r}") from None
    if nonnegative and numeric < 0.0:
        raise ValueError(
            f"{owner}: {arg}= must be non-negative, got {numeric!r}"
            " (the biharmonic operator is -H(H(q)) with "
            "square-rooted coefficients; a negative coefficient "
            "has no square root)")
    return leaf(value)


def _coerce_coefficient(
    value: object, arg: str, owner: str, *, nonnegative: bool,
) -> object:
    """Coerce a scalar-or-mapping coefficient to dynamic leaves."""
    if not isinstance(value, Mapping):
        return _coerce_scalar(value, arg, owner,
                              nonnegative=nonnegative)
    if not value:
        raise ValueError(
            f"{owner}: {arg}= mapping is empty; give every target "
            "a coefficient, or pass a scalar")
    coerced: dict[str, object] = {}
    for key, entry in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(
                f"{owner}: {arg}= mapping keys are field names, "
                f"got {key!r}")
        if isinstance(entry, TimeDependent):
            raise TypeError(
                f"{owner}: per-field {arg}= mappings take plain "
                "numbers; a ramped coefficient needs the scalar "
                "spelling (one closure per ramped target)")
        coerced[key] = _coerce_scalar(
            entry, f"{arg}[{key!r}]", owner, nonnegative=nonnegative)
    return coerced


# ================================================================
#  Slip validation (construction-time, host-side)
# ================================================================
#: the wall-stress vocabulary of the friction closures
FREE_SLIP: str = "free"
NO_SLIP: str = "no"
_SLIP_VALUES: tuple[str, str] = (FREE_SLIP, NO_SLIP)


def _check_slip_value(value: object, arg: str, owner: str) -> str:
    """Validate one slip choice against the vocabulary."""
    if value not in _SLIP_VALUES:
        raise ValueError(
            f"{owner}: {arg}= must be {FREE_SLIP!r} (zero tangential "
            f"wall stress) or {NO_SLIP!r} (u = 0 at the wall via the "
            f"factor-of-2 ghost), got {value!r}")
    return value  # type: ignore[return-value]


def _coerce_slip(value: object, owner: str) -> str | dict[str, str]:
    """Coerce ``slip=`` to a scalar choice or a per-field mapping."""
    if not isinstance(value, Mapping):
        return _check_slip_value(value, "slip", owner)
    if not value:
        raise ValueError(
            f"{owner}: slip= mapping is empty; give every target a "
            "slip choice, or pass a scalar 'free'/'no'")
    coerced: dict[str, str] = {}
    for key, entry in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(
                f"{owner}: slip= mapping keys are field names, got "
                f"{key!r}")
        coerced[key] = _check_slip_value(entry, f"slip[{key!r}]", owner)
    return coerced


# ================================================================
#  Per-axis wall treatment (bind-time classification)
# ================================================================
#: the three per-axis wall regimes of the flux chain
_PERIODIC: str = "periodic"
_WALL_NORMAL: str = "wall_normal"
_TANGENTIAL: str = "tangential"


def _wall_treatment(
    factor: object, owner: str, name: str, axis: str,
) -> str:
    """Classify one target factor along its coordinate axis.

    Periodic factors take the exact periodic path; a Dirichlet-tagged
    ``Inner`` face (the wall-normal velocity) closes on its own tag; a
    BC-free ``Center`` cell field (nodal tracer / tangential velocity)
    *and* a BC-free ``CellAvg`` cell field (the finite-volume tracer /
    tangential velocity) both take the flux-retag wall closure — the FV
    flux lands on the same nodal ``Inner`` face the nodal chain uses
    (the FV C-grid diff profile staggers ``CellAvg -> Inner``), so the
    two chains are structurally identical. Any other placement/BC — a
    TAGGED (fixed-value) ``CellAvg`` cell wall, a fixed-value
    (Dirichlet) ``Center`` cell wall, a bare face — is out of scope and
    rejected loudly. The face-exposing FV diff row is verified
    separately at bind (:func:`_probe_fv_face`); a BC-free ``CellAvg``
    on a grid without that profile is rejected there.
    """
    if getattr(factor.mesh, "periodic", False):
        return _PERIODIC
    node_set = getattr(factor, "node_set", None)
    has_dirichlet = BC.DIRICHLET in factor.bc.components
    if node_set is NodeSet.INNER and has_dirichlet:
        return _WALL_NORMAL
    if node_set is NodeSet.CENTER and factor.bc.is_free:
        return _TANGENTIAL
    if isinstance(factor, CellAvg) and factor.bc.is_free:
        return _TANGENTIAL
    raise NotImplementedError(
        f"{owner}: target {name!r} has an unsupported wall placement "
        f"along {axis!r} ({factor!r}); the walled closure supports the "
        "nodal family — no-flux / free-slip / no-slip cell fields "
        "(BC-free Center) and the wall-normal velocity face "
        "(Inner[Dirichlet]) — and the finite-volume family (BC-free "
        "CellAvg cell fields, the FV tracer / tangential velocity). A "
        "TAGGED (fixed-value Dirichlet) CellAvg cell wall, a fixed-value "
        "(Dirichlet) Center cell wall, or a bare face is out of scope "
        "(the stage-2e inhomogeneous boundary-data path is future work)")


def _probe_fv_face(
    grid: object, factor: object, axis: str, owner: str, name: str,
) -> None:
    """Verify the grid's FV ``diff`` row exposes a face-located flux.

    Accepting a BC-free ``CellAvg`` factor as tangential is only sound
    when the installed ``diff`` operator staggers the cell average onto
    the nodal interior face (the FV C-grid dispatch profile of a
    ``family="fv"`` model): the flux-retag closure then lands on the
    same ``Inner`` face the nodal chain uses, retags onto its Dirichlet
    sibling, and telescopes with the structural zero wall flux. On a
    raw grid without that profile ``("diff", CellAvg)`` resolves to the
    collocated ``FVDerivative`` composite (``CellAvg -> CellAvg``),
    whose flux never surfaces as a face field, so the wall flux cannot
    close — a taught rejection, never a silently wrong collocated run.
    Only ``CellAvg`` factors pay this probe; nodal ``Center`` targets
    keep the untouched nodal diff and never reach here.

    Raises
    ------
    NotImplementedError
        If the resolved ``diff`` codomain is not a BC-free nodal
        ``Inner`` face.
    """
    op = grid.dispatch.resolve("diff", factor)
    codomain = op.codomain(factor)
    if not (isinstance(codomain, NodalSpace)
            and codomain.node_set is NodeSet.INNER
            and codomain.bc.is_free):
        raise NotImplementedError(
            f"{owner}: target {name!r} is a finite-volume (CellAvg) "
            f"field on the walled axis {axis!r}, but the grid's diff "
            f"operator for it does not expose a face-located flux "
            f"(codomain {codomain!r}). The FV walled closure needs the "
            "face-exposing FV C-grid dispatch profile — a family='fv' "
            "model assembly, which staggers the cell average onto the "
            "nodal interior face. The collocated FVDerivative chain of "
            "a raw grid (no FV C-grid diff profile) cannot close a wall "
            "flux; assemble the model with family='fv'.")


# ================================================================
#  Immersed (cut-cell) fraction weighting (CL-D2, IP-D4)
# ================================================================
# Duplicated from the sw2 ``immersed_weighting`` idiom into the model
# layer: ``model`` must not import from ``shallowwater2`` (package
# layering), and the model advection module already owns its own local
# copies (``_immersed_flux`` / ``_immersed_scale``). Only the two the
# divergence-form closure needs are duplicated here (no ``mask_field``).
def _weight_flux(immersed: object, flux: ScalarField) -> ScalarField:
    r"""Weight one interface stress flux by the open-area fraction.

    Description
    -----------
    :math:`F \leftarrow \alpha_f\,F` with :math:`\alpha_f =
    \mathrm{fraction}(\text{flux space})`, the min-rule face fraction
    (I0): a cut face with :math:`\alpha = 0` carries no stress (the
    free-slip / no-flux closure of CL-D3), and on a face-aligned
    :math:`\{0, 1\}` staircase the weighted flux reproduces the walled
    model. The fraction is fetched on the flux's own (possibly
    wall-Dirichlet-tagged) space, so the multiply is a plain same-space
    product. A no-op off an immersed grid (``immersed is None``), so
    the unimmersed / walled / mapped path stays bit-for-bit the direct
    chain (the parity guard).

    Parameters
    ----------
    immersed : object
        The grid's immersed descriptor, or ``None`` off an immersed
        grid.
    flux : ScalarField
        The interface stress flux on a control-volume face.

    Returns
    -------
    ScalarField
        The open-area-weighted flux (``flux`` unchanged off an
        immersed grid).
    """
    if immersed is None:
        return flux
    return flux * immersed.fraction(flux.function_space)


def _scale_divergence(
    immersed: object, res: ScalarField | None,
) -> ScalarField | None:
    r"""Divide the summed stress divergence by the wet volume fraction.

    Description
    -----------
    The masked tendency :math:`(1/\theta_c)\sum_f \pm\,\alpha_f F_f`
    (each per-face divergence already :math:`1/V_c`-scaled by ``diff``):
    the accumulated open-area-weighted divergence is divided by the cell
    wet fraction :math:`\theta_c = \mathrm{fraction}(\text{res space})`,
    sealed with the double ``jnp.where`` so a dry cell (:math:`\theta =
    0`, numerator identically 0) stays exactly 0 and the reverse pass is
    finite (CL-D5). The :math:`\theta`-weighted tendency conserves
    :math:`\sum_c \theta_c V_c q_c` to machine zero — the flux
    differences telescope over the wet region (CL-D4). A no-op off an
    immersed grid (``res`` returned unchanged, ``None`` passed through).

    Parameters
    ----------
    immersed : object
        The grid's immersed descriptor, or ``None`` off an immersed
        grid.
    res : ScalarField | None
        The accumulated flux divergence on the cell (codomain) space,
        or ``None`` when no axis contributed.

    Returns
    -------
    ScalarField | None
        The wet-volume-scaled tendency (``res`` unchanged off an
        immersed grid, ``None`` passed through).
    """
    if immersed is None or res is None:
        return res
    theta = immersed.fraction(res.function_space)
    wet = theta.data > 0.0
    scaled = jnp.where(
        wet, res.data / jnp.where(wet, theta.data, 1.0), 0.0)
    return res.with_data(scaled)


# ================================================================
#  The shared harmonic pass (pure field arithmetic)
# ================================================================
def _dirichlet_face(flux: ScalarField, axis: str) -> object:
    """Return the Dirichlet BC-sibling of the flux's ``Inner`` factor.

    The first difference of a BC-free ``Center`` field lands on the
    BC-free nodal ``Inner`` face, so the sibling is the same interior
    face with the Dirichlet tag (the wall flux is a structural zero).
    """
    factor = flux.function_space.bare.factor(axis)
    return factor.mesh.nodal(factor.node_set, bc=BC.DIRICHLET)


def _wall_correction(q: ScalarField, k: object, axis: str) -> ScalarField:
    r"""No-slip wall-adjacent tendency correction along ``axis``.

    Adds :math:`-2\,k\,q/\Delta n^2` at both wall-adjacent cells (index
    0 and -1 along ``axis`` — an index test, never a coordinate-value
    test): the factor-of-2 ghost (``q_ghost = -q_1`` across the wall,
    wall value 0) makes the wall-face flux :math:`2 k q_1/\Delta n`,
    whose flux-difference contribution in the wall cell is
    :math:`-2 k q_1/\Delta n^2`. Both lengths are the wall cell's own
    width, read from ``grid.measure`` (so stretched columns are
    correct); the row is static and homogeneous-linear in ``q``, so it
    is jit-pure and reverse-mode differentiable with no singularity.

    Built fresh from ``q``'s (negotiated) grid at evaluation, so the
    weight shares the step's storage frame. The halo tracer carries no
    grid, so the caller skips this pointwise, zero-reach correction
    during negotiation (see :func:`_harmonic`).

    The :math:`1/\Delta n^2` factor is folded into the true-shape
    weight array (cell widths are positive on the primal centres, so
    no division by zero) rather than divided in as a field: a
    field-level ``/ measure`` divides in the never-synced ghost slots
    too, where the zero-weight numerator makes the reverse pass a
    ``0/0`` NaN even though the forward value is masked.
    """
    grid = q.grid
    space = q.function_space
    bare = space.bare
    ax = bare.names.index(axis)
    ndim = len(bare.names)
    indicator = jnp.zeros(space.shape)
    for end in (0, -1):
        where = tuple(end if a == ax else slice(None)
                      for a in range(ndim))
        indicator = modify_array(indicator, where, 1.0)
    width = grid.measure(space, axis).data  # true-shape primal widths
    weight = grid.create_field(space, data=indicator / (width * width))
    return q * weight * (-2.0 * k)


def _harmonic(
    q: ScalarField,
    axes: tuple[tuple[str, object, str], ...],
    no_slip: frozenset[str],
    immersed: object = None,
) -> ScalarField:
    r"""One ``div(A grad q)`` pass with the per-axis wall treatment.

    ``axes`` is ``(axis, coefficient, treatment)`` per coordinate, in
    horizontal-then-vertical order (the float-summation order of the
    purely periodic path, preserved bit-for-bit). Periodic axes take
    the plain two-pass chain; bounded tangential/tracer axes retag the
    interior flux onto its Dirichlet sibling (structural zero wall
    flux) and, on a no-slip axis, add the wall-adjacent correction;
    bounded wall-normal axes close on the target's own tag and retag
    the result back onto it.

    On an immersed grid (``immersed`` not ``None``) each interface
    stress flux is additionally weighted by the open-area fraction
    :math:`\alpha_f` (:func:`_weight_flux`) and the summed divergence
    is divided by the wet cell fraction :math:`\theta_c`
    (:func:`_scale_divergence`, sealed) — the IP-D4 spelling of CL-D2.
    The min-rule :math:`\alpha_f = 0` across a wet/dry face zeroes the
    cut-face stress (free-slip, CL-D3), which composes with the wall
    retag on domain walls (the two act on different faces). Both are
    no-ops off an immersed grid, so the flat / walled / mapped chain is
    untouched.
    """
    res = None
    for axis, k, treatment in axes:
        flux = _weight_flux(immersed, q.diff(axis) * k)
        if treatment == _PERIODIC:
            contribution = flux.diff(axis)
        elif treatment == _WALL_NORMAL:
            contribution = flux.diff(axis).retag(q)
        else:  # tangential / tracer: retag the interior flux
            flux = flux.retag(_dirichlet_face(flux, axis))
            contribution = flux.diff(axis)
            # the correction is pointwise (zero halo reach); skip it on
            # the grid-less halo tracer, materialize it on the real grid
            if axis in no_slip and hasattr(q.grid, "create_field"):
                contribution = contribution + _wall_correction(
                    q, k, axis)
        res = (contribution if res is None
               else res + contribution)
    return _scale_divergence(immersed, res)


def _biharmonic_root(coeff: object) -> object:
    r"""Square-root split of a biharmonic coefficient, AD-safe at zero.

    The biharmonic operator applies ``sqrt(coeff)`` per Laplacian pass.
    Reverse-mode AD: the ``sqrt`` VJP ``0.5 * coeff**-0.5`` is ``inf`` at
    ``coeff = 0``, so a plain ``coeff ** 0.5`` returns ``NaN`` from
    ``jax.grad`` at ``nu4 = 0`` even though the forward value is finite.
    The double ``jnp.where`` guards both branches: the primal is bitwise
    identical for ``coeff > 0`` (the taken branch), and the measure-zero
    subgradient at ``coeff = 0`` is pinned to ``0``.

    Demoted Python scalars (halo trace) keep the plain ``** 0.5`` — the
    ``jnp.where`` would re-promote them to arrays, which the numeric halo
    tracer rejects.
    """
    if isinstance(coeff, int | float | complex):
        return coeff ** 0.5
    pos = coeff > 0
    return jnp.where(pos, jnp.where(pos, coeff, 1.0) ** 0.5, 0.0)


# ================================================================
#  The shared closure machinery
# ================================================================
class _DiffusionClosure(ClosureBase):

    """
    Shared machinery of the four diffusion/friction closures.

    Description
    -----------
    Concrete subclasses declare the class vocabulary (coefficient
    attribute/argument name, canonical parameter names, operator
    order, default targets) and a thin ``__init__``; everything else
    — coercion, bind-time axis resolution, the published parameters,
    and the tendency term — lives here.
    """

    _coeff_attr: ClassVar[str]
    _coeff_param: ClassVar[ParamName]
    _coeff_param_v: ClassVar[ParamName]
    _biharmonic: ClassVar[bool] = False
    _term_name: ClassVar[str]
    _units: ClassVar[str]
    _doc: ClassVar[str]

    def __init__(
        self,
        value: object,
        value_v: object,
        vertical: str,
        fields: Role | type[Role] | str | Iterable[str] | None,
        exclude: str | Iterable[str],
        slip: object = FREE_SLIP,
    ) -> None:
        """Coerce the coefficients and store the geometry names."""
        super().__init__(fields=fields, exclude=exclude)
        cls = type(self)
        owner = cls.__name__
        arg = cls._coeff_attr
        setattr(self, arg, _coerce_coefficient(
            value, arg, owner, nonnegative=cls._biharmonic))
        setattr(self, f"{arg}_v", None if value_v is None
                else _coerce_coefficient(
                    value_v, f"{arg}_v", owner,
                    nonnegative=cls._biharmonic))
        self._vertical = vertical
        self._slip = _coerce_slip(slip, owner)
        self._target_axes: tuple = ()
        self._no_slip_axes: dict[str, frozenset[str]] = {}
        # immersed (cut-cell) bookkeeping, captured at bind (None off
        # an immersed grid — the flat / walled / mapped path)
        self._immersed: object = None
        self._halo_axes: tuple[str, ...] = ()

    # ================================================================
    #  Published parameters and per-field options
    # ================================================================
    @property
    def parameter_declarations(
        self,
    ) -> tuple[ParameterDeclaration, ...]:
        """The coefficient provides (the ``*_v`` one only if set)."""
        cls = type(self)
        declarations = (
            ParameterDeclaration(
                cls._coeff_param, attr=cls._coeff_attr,
                units=cls._units, doc=cls._doc),
        )
        if getattr(self, f"{cls._coeff_attr}_v") is not None:
            declarations += (
                ParameterDeclaration(
                    cls._coeff_param_v,
                    attr=f"{cls._coeff_attr}_v",
                    units=cls._units, doc=f"vertical {cls._doc}"),
            )
        return declarations

    @property
    def per_field_options(self) -> Mapping[str, object]:
        """The coefficient/slip slots, for the bind-time validation."""
        cls = type(self)
        options: dict[str, object] = {
            cls._coeff_attr: getattr(self, cls._coeff_attr)}
        value_v = getattr(self, f"{cls._coeff_attr}_v")
        if value_v is not None:
            options[f"{cls._coeff_attr}_v"] = value_v
        # a scalar slip passes through unchecked; a per-field mapping is
        # validated against the resolved targets (keys + full coverage)
        options["slip"] = self._slip
        return options

    # ================================================================
    #  Bind: per-target axes + wall treatment, vertical + slip check
    # ================================================================
    def bind(self, table: FieldTable) -> None:
        """Resolve targets (base) and freeze the per-target axes.

        Each axis is classified into its wall treatment (periodic,
        wall-normal, or tangential/tracer), stored horizontal-then-
        vertical so the float-summation order matches the purely
        periodic path bit-for-bit.

        Raises
        ------
        NotImplementedError
            If a target factor carries an unsupported wall placement
            (a fixed-value cell wall or a bare face; see
            :func:`_wall_treatment`), or a walled BC-free ``CellAvg``
            target sits on a grid without the face-exposing FV C-grid
            dispatch profile (see :func:`_probe_fv_face`).
        AssemblyError
            If a ``*_v`` coefficient is given but no target carries
            the ``vertical`` coordinate, or a target has no
            coordinate axes at all.
        """
        super().bind(table)
        cls = type(self)
        owner = cls.__name__
        # capture the immersed descriptor (super().bind already passed
        # the per-closure capability gate: only the harmonic family
        # reaches here on an immersed grid). No-slip at an immersed
        # boundary is a §5 deferral (the mask-keyed side-drag term):
        # reject it loudly rather than silently running free-slip.
        self._immersed = getattr(table.grid, "immersed", None)
        if self._immersed is not None and self._requests_no_slip():
            raise NotImplementedError(
                f"{owner}: slip='no' (no-slip) is not supported on "
                "immersed (cut-cell) grids. Stage A ships free-slip "
                "only — a zeroed cut-face stress (CL-D3); no-slip at "
                "the immersed boundary needs the mask-keyed side-drag "
                "term on tangential velocity next to dry cells "
                "(immersed_closures_sadourny_plan §5, the no-slip "
                "immersed drag deferral). Use slip='free' on the "
                "immersed grid, or drop the closure.")
        self._halo_axes = tuple(table.grid.names)
        has_v = getattr(self, f"{cls._coeff_attr}_v")
        target_axes: list[
            tuple[str, tuple[tuple[str, bool, str], ...]]] = []
        for name in self.targets:
            space = table[name].space
            axes = tuple(space.names)
            if not axes:
                raise AssemblyError(
                    f"{owner}: target {name!r} has no coordinate "
                    "axes to diffuse along; exclude it "
                    "(exclude=...) or narrow fields=")
            if has_v is None:
                ordered = [(a, False) for a in axes]
            else:
                ordered = (
                    [(a, False) for a in axes if a != self._vertical]
                    + [(a, True) for a in axes if a == self._vertical])
            spec_axes: list[tuple[str, bool, str]] = []
            for a, is_v in ordered:
                factor = space.bare.factor(a)
                treatment = _wall_treatment(factor, owner, name, a)
                # a bounded BC-free CellAvg is tangential only when the
                # grid's FV diff row exposes a face-located flux (a
                # family='fv' assembly); probe before trusting it (nodal
                # Center targets keep the untouched nodal diff -- they
                # never hit this branch, so pay no probe)
                if (treatment == _TANGENTIAL
                        and isinstance(factor, CellAvg)):
                    _probe_fv_face(table.grid, factor, a, owner, name)
                spec_axes.append((a, is_v, treatment))
            spec = tuple(spec_axes)
            target_axes.append((name, spec))
        if has_v is not None and not any(
                is_v for _, spec in target_axes
                for _, is_v, _ in spec):
            arg = cls._coeff_attr
            raise AssemblyError(
                f"{owner} got {arg}_v= but no target carries the "
                f"vertical coordinate {self._vertical!r}; drop "
                f"{arg}_v= or pass vertical=<coordinate name>")
        self._target_axes = tuple(target_axes)
        self._no_slip_axes = self._resolve_no_slip_axes()

    def _resolve_no_slip_axes(self) -> dict[str, frozenset[str]]:
        """Freeze the bounded tangential axes each target no-slips on.

        The base's :attr:`per_field_options` coverage check guarantees a
        per-field slip mapping keys (and covers) every target.
        """
        by_target = (
            self._slip if isinstance(self._slip, Mapping)
            else dict.fromkeys(self.targets, self._slip))
        return {
            name: frozenset(
                axis for axis, _is_v, treatment in spec
                if treatment == _TANGENTIAL)
            for name, spec in self._target_axes
            if by_target[name] == NO_SLIP}

    def _requests_no_slip(self) -> bool:
        """Whether any target requests the no-slip wall stress.

        Read off the raw ``slip=`` selection (not the wall-resolved
        :attr:`_no_slip_axes`, which is empty on a periodic immersed
        grid with no domain walls): the immersed no-slip deferral is
        about the immersed boundary, not domain walls, so the request
        alone is what the bind-time reject keys on. Mixing closures
        keep ``_slip == 'free'`` (the default), so this is only ever
        ``True`` for a friction closure with ``slip='no'``.
        """
        if isinstance(self._slip, Mapping):
            return NO_SLIP in self._slip.values()
        return self._slip == NO_SLIP

    @property
    def extra_halo(self) -> HaloSpec | None:
        r"""One-cell FD-stencil halo per coordinate on an immersed grid.

        On an immersed grid the harmonic pass multiplies the concrete
        open-area / wet-volume fraction fields (CL-D2), which the halo
        tracer cannot follow, so the closure declares its stencil reach
        here rather than being traced. The single ``div(A grad q)`` pass
        is a two-point-difference chain reaching exactly :math:`\pm 1`
        cell (the min-rule face fraction reads the same two adjacent
        cells), so **one** halo cell per coordinate is the exact reach —
        matching the width the flat harmonic term traces, so an all-wet
        immersed grid stays bit-for-bit the unimmersed run (A-G2). The
        biharmonic family never reaches here (it keeps the per-closure
        immersed reject). The flat / walled / mapped path stays fully
        halo-traced (``None``), bit-for-bit as before.
        """
        if self._immersed is None:
            return None
        return HaloSpec(dict.fromkeys(self._halo_axes, 1))

    @staticmethod
    def _axes(
        spec: tuple[tuple[str, bool, str], ...],
        kh: object,
        kv: object,
    ) -> tuple[tuple[str, object, str], ...]:
        """Bind each axis to its coefficient (vertical picks ``kv``)."""
        return tuple(
            (axis, kv if is_v else kh, treatment)
            for axis, is_v, treatment in spec)

    # ================================================================
    #  The tendency term
    # ================================================================
    def tendency_terms(self) -> tuple[TendencyTerm, ...]:
        """One linear term advancing every resolved target."""
        cls = type(self)
        return (
            TendencyTerm(
                name=cls._term_name, fn=self._apply,
                treatment=Treatment.EXPLICIT,
                advances=self.targets, linear=True),
        )

    def _apply(
        self, state: object, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        """Apply the (bi)harmonic operator to every target."""
        cls = type(self)
        coeff = ctx.params[cls._coeff_param]
        has_v = getattr(self, f"{cls._coeff_attr}_v") is not None
        coeff_v = ctx.params[cls._coeff_param_v] if has_v else None
        out: dict[str, ScalarField] = {}
        for name, spec in self._target_axes:
            q = state[name]
            kh = coeff[name] if isinstance(coeff, dict) else coeff
            kv = None
            if any(is_v for _, is_v, _ in spec):
                kv = (coeff_v[name] if isinstance(coeff_v, dict)
                      else coeff_v)
            no_slip = self._no_slip_axes.get(name, frozenset())
            immersed = self._immersed
            if cls._biharmonic:
                # sqrt split, guarded for reverse-mode AD at coeff=0
                # (plain ** 0.5 keeps demoted Python scalars scalar; the
                # guard adds the jnp.where only for concrete jnp values).
                # Biharmonic keeps _supports_immersed = False, so
                # immersed is None here (rejected at bind) — the pass
                # stays the flat/walled chain.
                kh = _biharmonic_root(kh)
                kv = None if kv is None else _biharmonic_root(kv)
                axes = self._axes(spec, kh, kv)
                inner = _harmonic(q, axes, no_slip, immersed)
                out[name] = -_harmonic(inner, axes, no_slip, immersed)
            else:
                out[name] = _harmonic(
                    q, self._axes(spec, kh, kv), no_slip, immersed)
        return out


# ================================================================
#  The concrete closures
# ================================================================
@partial(jaxify, dynamic=("kappa", "kappa_v"))
class HarmonicDiffusion(_DiffusionClosure):

    r"""
    Harmonic mixing :math:`\nabla \cdot (\kappa \nabla q)` of tracers.

    Description
    -----------
    Targets every ``fr.roles.TRACER`` field by default; provides
    ``mixing.kappa`` (and ``mixing.kappa_v`` when given). On walled
    grids tracers get **no-flux** (zero wall flux) walls structurally
    — there is no slip choice (that is friction-only).

    Parameters
    ----------
    kappa : float | fr.Ramp | Mapping[str, float]
        Mixing coefficient (:math:`\mathrm{m^2/s}`): one scalar (or
        ramp) for every target, or a name-keyed per-field mapping of
        plain numbers covering every target.
    kappa_v : float | fr.Ramp | Mapping[str, float] | None, optional
        Vertical mixing coefficient, acting along ``vertical``;
        ``None`` applies ``kappa`` isotropically (default: None).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``); on a mapped /
        terrain grid this names the **column coordinate** the ``*_v``
        coefficient acts along (e.g. ``"sigma"``), not the physical
        vertical — the along-coordinate split (see the module docstring).
    fields : Role | type[Role] | str | Iterable[str] | None, optional
        Target override; see `ClosureBase` (default: None).
    exclude : str | Iterable[str], optional
        Targets removed from the resolution (default: ()).
    """

    default_targets = TRACER
    _coeff_attr = "kappa"
    _coeff_param = MIXING_KAPPA
    _coeff_param_v = MIXING_KAPPA_V
    _term_name = "mixing"
    _units = "m^2/s"
    _doc = "harmonic mixing coefficient"
    # the divergence-form harmonic operator carries the IP-D4 fraction
    # spelling on immersed grids (CL-D1): min-rule faces read two wet
    # cells, so the two-point stencil never reaches a dry value with
    # nonzero weight
    _supports_immersed = True

    def __init__(
        self,
        kappa: object,
        *,
        kappa_v: object = None,
        vertical: str = "z",
        fields: Role | type[Role] | str | Iterable[str] | None = None,
        exclude: str | Iterable[str] = (),
    ) -> None:
        """Store the mixing coefficients; targets resolve at bind."""
        super().__init__(kappa, kappa_v, vertical, fields, exclude)


@partial(jaxify, dynamic=("kappa", "kappa_v"))
class BiharmonicDiffusion(_DiffusionClosure):

    r"""
    Biharmonic mixing :math:`-\mathcal{H}(\mathcal{H}(q))` of tracers.

    Description
    -----------
    The Griffies & Hallberg (2000) iterated form with square-rooted
    coefficients, so ``kappa`` is the true (non-negative) biharmonic
    coefficient in :math:`\mathrm{m^4/s}`. Targets every
    ``fr.roles.TRACER`` field by default; provides ``mixing.kappa4``
    (and ``mixing.kappa4_v`` when given). On walled grids tracers get
    **no-flux** walls on both Laplacian passes (no slip choice).

    Parameters
    ----------
    kappa : float | fr.Ramp | Mapping[str, float]
        Non-negative biharmonic mixing coefficient
        (:math:`\mathrm{m^4/s}`); scalar, ramp, or per-field mapping.
    kappa_v : float | fr.Ramp | Mapping[str, float] | None, optional
        Vertical biharmonic coefficient along ``vertical``; ``None``
        applies ``kappa`` isotropically (default: None).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``); on a mapped /
        terrain grid this names the **column coordinate** the ``*_v``
        coefficient acts along (e.g. ``"sigma"``), not the physical
        vertical — the along-coordinate split (see the module docstring).
    fields : Role | type[Role] | str | Iterable[str] | None, optional
        Target override; see `ClosureBase` (default: None).
    exclude : str | Iterable[str], optional
        Targets removed from the resolution (default: ()).
    """

    default_targets = TRACER
    _coeff_attr = "kappa"
    _coeff_param = MIXING_KAPPA4
    _coeff_param_v = MIXING_KAPPA4_V
    _biharmonic = True
    _term_name = "mixing"
    _units = "m^4/s"
    _doc = "biharmonic mixing coefficient"

    def __init__(
        self,
        kappa: object,
        *,
        kappa_v: object = None,
        vertical: str = "z",
        fields: Role | type[Role] | str | Iterable[str] | None = None,
        exclude: str | Iterable[str] = (),
    ) -> None:
        """Store the mixing coefficients; targets resolve at bind."""
        super().__init__(kappa, kappa_v, vertical, fields, exclude)


@partial(jaxify, dynamic=("nu", "nu_v"))
class HarmonicFriction(_DiffusionClosure):

    r"""
    Harmonic friction :math:`\nabla \cdot (\nu \nabla u_i)`.

    Description
    -----------
    Targets the ``Velocity`` role family by default (the PROGNOSTIC
    members, V-H2); provides ``friction.nu`` (and ``friction.nu_v``
    when given).

    On a walled grid the wall stress follows ``slip=``. The choice
    applies to every velocity factor that is **tangential** (a cell
    field — nodal ``Center`` or finite-volume ``CellAvg``) along a
    walled axis:
    ``slip="free"`` (the default) sets zero tangential wall stress —
    the structural zero wall flux, no spurious drag on a wall-parallel
    flow; ``slip="no"`` sets ``u = 0`` at the wall via the
    factor-of-2 ghost, adding the wall-adjacent drag
    :math:`-2\,\nu\,u_1/\Delta n^2`. The **wall-normal** component
    (staggered, ``Inner[Dirichlet]`` along the walled axis) ignores
    ``slip=`` — impermeability closes it either way. The default is a
    documented convention (MOM6 / Oceananigans default free-slip;
    MITgcm defaults no-slip — both are valid physics). Per-wall
    refinement is future work (the boundary-data path); a per-field
    mapping picks the slip per velocity.

    Parameters
    ----------
    nu : float | fr.Ramp | Mapping[str, float]
        Viscosity (:math:`\mathrm{m^2/s}`); scalar, ramp, or
        name-keyed per-field mapping covering every target.
    nu_v : float | fr.Ramp | Mapping[str, float] | None, optional
        Vertical viscosity along ``vertical``; ``None`` applies
        ``nu`` isotropically (default: None).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``); on a mapped /
        terrain grid this names the **column coordinate** the ``*_v``
        coefficient acts along (e.g. ``"sigma"``), not the physical
        vertical — the along-coordinate split (see the module docstring).
    slip : str | Mapping[str, str], optional
        Wall stress on tangential velocity factors: ``"free"`` (zero
        wall stress) or ``"no"`` (``u = 0`` at the wall). A scalar
        applies to every target; a name-keyed mapping (covering every
        target) picks per velocity. Static — a change recompiles
        (default: ``"free"``).
    fields : Role | type[Role] | str | Iterable[str] | None, optional
        Target override; see `ClosureBase` (default: None).
    exclude : str | Iterable[str], optional
        Targets removed from the resolution (default: ()).
    """

    default_targets = Velocity
    _coeff_attr = "nu"
    _coeff_param = FRICTION_NU
    _coeff_param_v = FRICTION_NU_V
    _term_name = "friction"
    _units = "m^2/s"
    _doc = "harmonic friction coefficient (viscosity)"
    # free-slip immersed friction rides the same IP-D4 fraction
    # spelling (CL-D1/CL-D3): a zeroed cut-face stress. slip='no' at an
    # immersed boundary is a §5 deferral, rejected at bind
    _supports_immersed = True

    def __init__(
        self,
        nu: object,
        *,
        nu_v: object = None,
        vertical: str = "z",
        slip: str | Mapping[str, str] = FREE_SLIP,
        fields: Role | type[Role] | str | Iterable[str] | None = None,
        exclude: str | Iterable[str] = (),
    ) -> None:
        """Store the viscosities and slip; targets resolve at bind."""
        super().__init__(nu, nu_v, vertical, fields, exclude, slip=slip)


@partial(jaxify, dynamic=("nu", "nu_v"))
class BiharmonicFriction(_DiffusionClosure):

    r"""
    Biharmonic friction :math:`-\mathcal{H}(\mathcal{H}(u_i))`.

    Description
    -----------
    The iterated square-rooted form (see `BiharmonicDiffusion`) on
    the ``Velocity`` role family (PROGNOSTIC members, V-H2);
    provides ``friction.nu4`` (and ``friction.nu4_v`` when given).

    On a walled grid the **same** slip treatment is applied on both
    Laplacian passes (Griffies & Hallberg); see `HarmonicFriction`
    for the ``slip=`` semantics.

    Parameters
    ----------
    nu : float | fr.Ramp | Mapping[str, float]
        Non-negative biharmonic viscosity (:math:`\mathrm{m^4/s}`);
        scalar, ramp, or per-field mapping.
    nu_v : float | fr.Ramp | Mapping[str, float] | None, optional
        Vertical biharmonic viscosity along ``vertical``; ``None``
        applies ``nu`` isotropically (default: None).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``); on a mapped /
        terrain grid this names the **column coordinate** the ``*_v``
        coefficient acts along (e.g. ``"sigma"``), not the physical
        vertical — the along-coordinate split (see the module docstring).
    slip : str | Mapping[str, str], optional
        Wall stress on tangential velocity factors, applied on both
        passes; ``"free"`` (default) or ``"no"`` (see
        `HarmonicFriction`).
    fields : Role | type[Role] | str | Iterable[str] | None, optional
        Target override; see `ClosureBase` (default: None).
    exclude : str | Iterable[str], optional
        Targets removed from the resolution (default: ()).
    """

    default_targets = Velocity
    _coeff_attr = "nu"
    _coeff_param = FRICTION_NU4
    _coeff_param_v = FRICTION_NU4_V
    _biharmonic = True
    _term_name = "friction"
    _units = "m^4/s"
    _doc = "biharmonic friction coefficient"

    def __init__(
        self,
        nu: object,
        *,
        nu_v: object = None,
        vertical: str = "z",
        slip: str | Mapping[str, str] = FREE_SLIP,
        fields: Role | type[Role] | str | Iterable[str] | None = None,
        exclude: str | Iterable[str] = (),
    ) -> None:
        """Store the viscosities and slip; targets resolve at bind."""
        super().__init__(nu, nu_v, vertical, fields, exclude, slip=slip)
