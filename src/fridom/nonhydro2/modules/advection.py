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
  term is exactly linear in the state and ``fr.linearize`` keeps it.
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

**Walled grids are future work**: the advective flux stencils near
rigid walls (bounded, non-periodic mesh factors) are not covered
yet, so ``bind`` rejects walled grids with a taught error — build a
linear model (``advection=False`` in ``nh.Model``) instead. The
background option inherits the restriction.
"""
from __future__ import annotations

import inspect
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Literal, final

import fridom.framework2 as fr
from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.operators.base import (
    OperatorRequirements,
    SeparableOperator,
)
from fridom.framework2.grid.operators.interned import interned
from fridom.framework2.grid.operators.reconstruct import (
    apply_fv_staggered,
)
from fridom.framework2.grid.operators.select import Where
from fridom.framework2.grid.operators.weno import (
    _shu_row,  # the exact-rational coefficient seam (framework2)
    weno_reconstruct,
    weno_tables,
)
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.grid.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from jax import Array

    from fridom.framework2.grid.fields.scalar_field import ScalarField
    from fridom.framework2.grid.operators.base import FieldLike
    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.framework2.model.context import StepContext

#: the biased-reconstruction weightings of the module family
_WEIGHTINGS = ("linear", "weno")

#: the odd formal orders grounded by the framework WENO tables
_SUPPORTED_ORDERS = (3, 5)

#: the velocity components' staggering axes (the nh C-grid)
_VELOCITY_AXES = {"u": "x", "v": "y", "w": "z"}


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
    grid : fr.grid.Grid
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
    domain: FunctionSpace, label: str,
) -> FunctionSpace:
    """
    Shared C-grid face signature of the module-private kernels.

    Description
    -----------
    ``Center -> Right`` and ``Right -> Center`` on periodic real
    nodal factors — exactly the two C-grid flux positions of the
    flux-form advection modules. Everything else (average spaces,
    bounded axes, complex scalars) raises: the modules reject
    walled grids at bind, so a bounded factor here is a genuine
    misuse.

    Parameters
    ----------
    domain : FunctionSpace
        The bare 1D factor space.
    label : str
        The raising operator's name (error attribution).

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
    mesh = domain.mesh
    if not mesh.periodic:
        raise SpaceMismatchError(
            f"{label} is periodic-only (walled grids are rejected "
            f"at module bind), got {domain!r}",
            left=domain, operation="reconstruct")
    if domain.node_set is NodeSet.CENTER:
        return mesh.right
    if domain.node_set is NodeSet.RIGHT:
        return mesh.center
    raise SpaceMismatchError(
        f"{label} maps Center -> Right and Right -> Center (the "
        f"C-grid flux positions), got {domain!r}",
        left=domain, operation="reconstruct")


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

    Parameters
    ----------
    size : int
        The even stencil size (2 or 4 for orders 3 and 5).
    """

    dispatch_kind: ClassVar[str | None] = None

    def __init__(self, size: int) -> None:
        """Validate the even stencil size and store it."""
        if size not in (2, 4):
            raise ValueError(
                f"the advective velocity interpolation grounds the "
                f"even stencil sizes (2, 4), got {size}")
        self._size = size

    def _intern_key(self) -> tuple:
        """Structural key: the stencil size (D6)."""
        return (self._size,)

    @property
    def size(self) -> int:
        """The even stencil size."""
        return self._size

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
        return _face_codomain(domain, type(self).__name__)

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

        return apply_fv_staggered(self, f, axis, size, kernel,
                                  metadata=f.metadata)



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
    """

    dispatch_kind: ClassVar[str | None] = None

    def __init__(
        self,
        order: int,
        bias: Literal["left", "right"],
        weighting: Literal["linear", "weno"],
    ) -> None:
        """Validate through the framework tables and store."""
        if weighting not in _WEIGHTINGS:
            raise ValueError(
                f"weighting must be one of {_WEIGHTINGS}, got "
                f"{weighting!r}")
        weno_tables(order, bias)  # validates order and bias
        self._order = order
        self._bias = bias
        self._weighting = weighting

    def _intern_key(self) -> tuple:
        """Structural key: order, bias side, and weighting (D6)."""
        return (self._order, self._bias, self._weighting)

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
        return _face_codomain(domain, type(self).__name__)

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
        on the dual ``Right -> Center`` direction: the right face of
        the dual cell around face ``j`` is center ``j + 1`` in the
        shared storage frame.

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
        m0 = order // 2 if bias == "left" else order // 2 - 1
        domain = f.function_space.bare.factor(axis)
        if domain.node_set is NodeSet.RIGHT:
            m0 += 1
        if self._weighting == "weno":
            def kernel(arr: Array, axis_index: int) -> Array:
                return weno_reconstruct(arr, axis_index,
                                        order=order, bias=bias)
        else:
            row = _linear_row(order, bias)

            def kernel(arr: Array, axis_index: int) -> Array:
                return _weighted_windows(arr, axis_index, row)
        return apply_fv_staggered(self, f, axis, order, kernel,
                                  metadata=f.metadata, align=m0)


# ================================================================
#  The shared flux-form scaffolding (module-private)
# ================================================================
class _FluxFormAdvection(fr.Module):

    r"""
    Shared flux-form transport of every ADVECTED component.

    Description
    -----------
    The private scaffolding of the advection family (never exported):
    role selection and the walled-grid rejection at bind, the
    tendency terms, and the per-axis flux loop. Subclasses choose
    the face value of the advected quantity through the
    `_face_value` / `_linear_face_value` hooks (centered by
    default).

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
        fr.ParameterReference(
            fr.params.SCALING_ROSSBY, default=1.0,
            hint="Rossby number (nh.DynamicalCore)"),
    )

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

    # ------------------------------------------------------------
    #  Background declarations (AUXILIARY profile samples)
    # ------------------------------------------------------------
    @property
    def field_declarations(self) -> tuple[fr.FieldDeclaration, ...]:
        """The background samples on each component's own space.

        One AUXILIARY field ``background_<component>`` per mapped
        component, declared on the velocity template's own pattern
        (staggered along the component axis; the topology-conditional
        wall Dirichlet is inert on the periodic grids this family
        accepts), so the profile is sampled at that component's own
        staggered nodes when materialized. The user value rides the
        declaration ``default=`` untouched: a constant fills, a
        coordinate callable is discretized by ``grid.create_field``.
        """
        return tuple(
            fr.FieldDeclaration(
                f"background_{name}",
                space=fr.Staggered(
                    axis, wall_bc={axis: BC.DIRICHLET}),
                lifecycle=fr.Lifecycle.AUXILIARY,
                default=(_profile_default(name)
                         if callable(self._background[name])
                         else self._background[name]),
                long_name=f"Background {name}-velocity",
                units="m/s")
            for name, axis in _VELOCITY_AXES.items()
            if name in self._background)

    @property
    def field_references(self) -> tuple[fr.FieldReference, ...]:
        """The checked claims on the mapped velocity components."""
        return tuple(
            fr.FieldReference(
                name, hint="the background flow rides the declared "
                           "velocity components (nh.DynamicalCore "
                           "declares u, v, w)")
            for name in _VELOCITY_AXES if name in self._background)

    def bind(self, table: object) -> None:
        """Freeze the advected set and the axis -> velocity mapping.

        Raises
        ------
        NotImplementedError
            On a walled grid (any bounded mesh factor): the
            advective flux stencils near rigid walls are future
            work, and the natural downstream failure (an operator
            dispatch mismatch deep in the flux chain) would be
            cryptic.
        ValueError
            If a background sample does not resolve on its velocity
            component's own space (a component outside the nh
            ``u``/``v``/``w`` staggering vocabulary).
        """
        factors = getattr(table.grid, "factors", ())
        walled = tuple(
            name for mesh in factors for name in mesh.names
            if not getattr(mesh, "periodic", True))
        if walled:
            raise NotImplementedError(
                f"{type(self).__name__} does not support walled "
                f"grids yet (bounded coordinates: {walled}); the "
                "advective flux stencils near rigid walls are "
                "future work. Build a linear model instead "
                "(advection=False in nh.Model) or drop the "
                "advection module")
        self._advected = table.select(fr.roles.ADVECTED)
        selector = table.velocity()
        # selector.labels pairs each velocity name with its axis
        self._axis_velocity = tuple(
            (axis, name) for name, axis in selector.labels)
        self._bind_background(table)

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

    def tendency_terms(self) -> tuple[fr.TendencyTerm, ...]:
        """Return the advection term(s) of the module.

        Without a background: the single Rossby-scaled ``advection``
        term (the pre-background code path, literally unchanged).
        With one: the nonlinear ``advection`` difference term plus
        the genuinely separate ``background_advection`` term tagged
        ``linear=True`` so ``fr.linearize`` keeps exactly it (V-S3).
        """
        if not self._background:
            return (
                fr.TendencyTerm(
                    name="advection", fn=self._advect,
                    treatment=fr.Treatment.EXPLICIT,
                    advances=self._advected,
                    transports=self._advected),
            )
        return (
            fr.TendencyTerm(
                name="advection", fn=self._advect_perturbation,
                treatment=fr.Treatment.EXPLICIT,
                advances=self._advected, transports=self._advected),
            fr.TendencyTerm(
                name="background_advection",
                fn=self._advect_background,
                treatment=fr.Treatment.EXPLICIT,
                advances=self._advected, transports=self._advected,
                linear=True),
        )

    def _advect(
        self, state: object, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        """Flux-form transport of every advected component (Ro-scaled)."""
        ro = ctx.params[fr.params.SCALING_ROSSBY]
        out: dict[str, ScalarField] = {}
        for qname in self._advected:
            q = state[qname]
            res = None
            for axis, vname in self._axis_velocity:
                v = state[vname]
                flux_space = q.diff(axis).function_space
                v_face = self._velocity_face(v, flux_space)
                flux = v_face * self._face_value(
                    q, v_face, axis, flux_space)
                divergence = flux.diff(axis)
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
        ro = ctx.params[fr.params.SCALING_ROSSBY]
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
        ``fr.linearize`` keeps this term and drops the nonlinear
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
        res = None
        for axis, vname in self._axis_velocity:
            v = ro * state[vname]
            sample = self._background_by_axis.get(axis)
            if sample is not None:
                v = v + state[sample]
            flux_space = q.diff(axis).function_space
            v_face = self._velocity_face(v, flux_space)
            flux = v_face * self._face_value(
                q, v_face, axis, flux_space)
            divergence = flux.diff(axis)
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
        res = None
        for axis, sample in self._background_axes:
            flux_space = q.diff(axis).function_space
            v_face = self._velocity_face(state[sample], flux_space)
            flux = v_face * self._linear_face_value(
                q, v_face, axis, flux_space)
            divergence = flux.diff(axis)
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
    """

    _weighting: ClassVar[Literal["linear", "weno"]] = "linear"

    def __init__(
        self,
        order: int = 3,
        *,
        background: Mapping[str, Callable | float] | None = None,
    ) -> None:
        """Build the biased reconstruction pairs for ``order``."""
        super().__init__(background=background)
        if order not in _SUPPORTED_ORDERS:
            raise ValueError(
                f"{type(self).__name__} grounds the biased "
                f"reconstruction orders {_SUPPORTED_ORDERS} (the "
                f"framework WENO tables), got {order}")
        weighting = self._weighting  # class-attribute lookup
        self._order = order
        self._left = _BiasedFaceReconstruction(order, "left",
                                               weighting)
        self._right = _BiasedFaceReconstruction(order, "right",
                                                weighting)
        # the linear-weight pair of the background transport (for
        # weighting == "linear" these are the same interned objects)
        self._lin_left = _BiasedFaceReconstruction(order, "left",
                                                   "linear")
        self._lin_right = _BiasedFaceReconstruction(order, "right",
                                                    "linear")
        self._interp = _CenteredFaceInterpolation(order - 1)

    def bind(self, table: object) -> None:
        """Bind the flux form, then widen the provisional halo.

        Description
        -----------
        The landed assembly validates every term over real
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
        """
        super().bind(table)
        need = self._order // 2 + 1
        grid = table.grid
        current = dict(grid.decomposition.halo.widths)
        axes = tuple(axis for axis, _ in self._axis_velocity)
        if all(current.get(axis, 0) >= need for axis in axes):
            return
        demand = grid.decomposition.halo.merge_max(
            HaloSpec(dict.fromkeys(axes, need)))
        grid.negotiate(halo=demand)

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def order(self) -> int:
        """Formal order of the biased face reconstruction."""
        return self._order

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
        return result

    def _face_value(
        self,
        q: ScalarField,
        v_face: ScalarField,
        axis: str,
        flux_space: object,  # noqa: ARG002 — fixed by the operator pair
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
        visible to the halo-accounting trace.

        Parameters
        ----------
        q : ScalarField
            The advected quantity.
        v_face : ScalarField
            The advecting velocity component on the flux space.
        axis : str
            The advection axis.
        flux_space : object
            The flux space (fixed by the operator pair here).

        Returns
        -------
        ScalarField
            The upwind-biased face value of ``q``.
        """
        positive = v_face + abs(v_face)
        return Where()(positive, self._left[axis](q),
                       self._right[axis](q))

    def _linear_face_value(
        self,
        q: ScalarField,
        v_face: ScalarField,
        axis: str,
        flux_space: object,  # noqa: ARG002 — fixed by the operator pair
    ) -> ScalarField:
        """
        Linear-row upwind face value for the background transport.

        Description
        -----------
        The same ``Where`` side selection as `_face_value`, but with
        the linear (optimal-weight) reconstruction pair — the WENO
        module's smooth-limit row — and the mask read from the
        static background face velocity, so the value is exactly
        linear in the state.

        Parameters
        ----------
        q : ScalarField
            The advected quantity.
        v_face : ScalarField
            The background face velocity (static in the state).
        axis : str
            The advection axis.
        flux_space : object
            The flux space (fixed by the operator pair here).

        Returns
        -------
        ScalarField
            The linear upwind-biased face value of ``q``.
        """
        positive = v_face + abs(v_face)
        return Where()(positive, self._lin_left[axis](q),
                       self._lin_right[axis](q))


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
    """

    _weighting: ClassVar[Literal["linear", "weno"]] = "weno"
