"""
``ImmersedDomain``: the grid-owned masked-domain descriptor.

Description
-----------
Owning class doc: ``design/specs/grid/classes/grid.md``; rules in
``design/specs/grid/02_rules.md`` section 3.7. A static descriptor of
the wet region (successor of ``WaterMask``): it holds the declaring
callable and static parameters only — no arrays, no fields — and
materializes per-space masks/fractions on demand at trace time,
exactly like ``grid.evaluation_nodes`` (rules section 2.7).

Fractions are the fidelity-ladder point 2 of the masked-domain spec:
cut-cell **volume fractions on cell spaces, area fractions on face
spaces**, the boolean mask staying the ``{0, 1}`` special case. The
default (``order=None``) is the collocation staircase — the indicator
sampled at cell centers and thresholded, ``theta in {0, 1}``, bitwise
``WaterMask`` parity. ``order=q`` (``q >= 2``) opts into genuine
partial cells: ``theta`` is the per-cell ``q``-point Gauss-Legendre
quadrature average of the declared indicator (the small-cell floor
``min_fraction`` applied at materialization). Transition sets and the
level-set generalization are designed-for. Masked domains keep the
full product shape (section 3.7): masked DOFs are stored-but-dead, and
masking correctness is owned by operators and modules, not by the type
system.

Halo note: derived fields are routed through the ordinary storage path
(``pad`` + ``sync``), so periodic wraps and shard-edge exchanges carry
the correct neighboring fraction values; the BC-structured fill of
*physical* boundaries on bounded meshes is dry-exterior here —
mask-aware operators (designed-for) own their own ghost discipline.
"""
# Wave 4: ImmersedDomain, Slip (I0: genuine fractions)
from __future__ import annotations

import copy
import inspect
import numbers
from enum import Enum, auto
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.spatial.bc import BC
from fridom.spatial.fields.metadata import FieldMetadata
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.fields.storage import store
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh
from fridom.spatial.spaces.average import CellAvg, FaceAvg
from fridom.spatial.spaces.coefficient import CoefficientSpace
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet
from fridom.spatial.spaces.tensor_product import TensorProductSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.spatial.coordinate_mapping import CoordinateMapping
    from fridom.spatial.grid import Grid
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike

# the collocation staircase threshold: declared fractions are wet
# above this value (a boolean indicator and a {0, 1} fraction agree)
_WET_THRESHOLD = 0.5

# the smallest point count that leaves the collocation shortcut: an
# ``order`` below this aliases ``None`` (the midpoint sample), so the
# default stays bitwise-identical to plain center collocation
_QUADRATURE_MIN = 2


class Slip(Enum):

    """
    Immersed-boundary structure: staggered-mask combination rule.

    Description
    -----------
    Structure (no-slip vs free-slip) is the *combination rule* used
    to derive staggered masks from the cell mask (rules section
    3.7), a parameter of the derivation, not stored per-space state:
    ``NO_SLIP`` keeps a face wet only when **both** adjacent cells
    are wet (``face = AND(adjacent centers)``); ``FREE_SLIP`` keeps
    it wet when **either** adjacent cell is wet. Fraction staggering
    is slip-independent (the geometric min-transfer, IP-D2).
    """

    NO_SLIP = auto()
    FREE_SLIP = auto()


class ImmersedDomain:

    """
    Static wet-region descriptor; per-space fields derived on demand.

    Description
    -----------
    Declares the wet fraction as a callable of the physical
    coordinates (keyword-matched to the grid's coordinate names,
    like ``create_field(init=...)``). The default (``order=None``)
    derives the collocation staircase: declared values are sampled
    at cell centers and thresholded at ``0.5``, so a boolean
    indicator and a {0, 1} fraction are equivalent. ``order=q``
    (``q >= 2``) opts into genuine partial cells (per-cell
    Gauss-Legendre quadrature). The descriptor is bound to its grid
    at grid construction (``immersed=`` kwarg) or by the pre-freeze
    ``grid.with_immersed`` update; it is fully static (no arrays, not
    a pytree) and every ``fraction``/``mask`` call materializes to
    the local shard at trace time (memoized concrete-only, like
    ``grid._measures``).

    Parameters
    ----------
    init : Callable[..., jax.Array]
        The wet fraction as a function of the physical coordinates:
        an indicator ``in {0, 1}`` or a genuine local fraction (the
        quadrature averages either).
    slip : Slip, optional
        The default staggered-mask derivation rule
        (default: Slip.NO_SLIP).
    order : int | None, optional
        The per-cell Gauss-Legendre quadrature point count for the
        volume fraction: ``None`` (or ``1``) is the collocation
        staircase (``theta in {0, 1}``), ``>= 2`` the genuine
        partial-cell rule, exact for per-cell polynomial averages up
        to degree ``2 * order - 1`` (default: None).
    min_fraction : float, optional
        The small-cell floor (MITgcm ``hFacMin``): after quadrature,
        ``theta < min_fraction / 2 -> 0`` and
        ``theta < min_fraction -> min_fraction``; ``0.0`` disables.
        Boolean {0, 1} fractions are unaffected (default: 0.1).
    """

    def __init__(
        self,
        init: Callable[..., jax.Array],
        *,
        slip: Slip = Slip.NO_SLIP,
        order: int | None = None,
        min_fraction: float = 0.1,
    ) -> None:
        """Declare the wet fraction; see the class docstring."""
        if not callable(init):
            raise TypeError(
                f"init must be a callable of the physical "
                f"coordinates, got {init!r}")
        if not isinstance(slip, Slip):
            raise TypeError(
                f"slip must be a Slip member, got {slip!r}")
        _validate_order(order)
        _validate_min_fraction(min_fraction)
        self._init: Callable[..., jax.Array] = init
        self._slip: Slip = slip
        self._order: int | None = order
        self._min_fraction: float = float(min_fraction)
        self._grid: Grid | None = None
        # concrete-only materialization cache, keyed by (laid-out
        # space, kind, slip, negotiated halo); mirrors grid._measures
        # (host-side, not a pytree — the descriptor stays static aux).
        # The halo is part of the key because the stored array is
        # padded to the storage frame: a model that re-negotiates the
        # grid wider (a module's extra_halo) must re-materialize, not
        # reuse the provisionally-narrower fraction.
        self._cache: dict[
            tuple[object, str, Slip | None, object], jax.Array] = {}

    # ================================================================
    #  Identity (static aux discipline, matching Grid)
    # ================================================================
    def __eq__(self, other: object) -> bool:
        """Identity comparison: ``self is other`` (static aux)."""
        return self is other

    def __hash__(self) -> int:
        """Identity hash, consistent with ``__eq__``."""
        return id(self)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def slip(self) -> Slip:
        """The default staggered-mask derivation rule."""
        return self._slip

    @property
    def order(self) -> int | None:
        """The per-cell quadrature point count (None = collocation)."""
        return self._order

    @property
    def min_fraction(self) -> float:
        """The small-cell floor applied to genuine fractions."""
        return self._min_fraction

    # ================================================================
    #  Grid binding (called by the grid at attachment)
    # ================================================================
    def _bind(self, grid: Grid) -> None:
        """Bind the descriptor to its grid (idempotent per grid)."""
        if self._grid is not None and self._grid is not grid:
            raise ValueError(
                "this ImmersedDomain is already attached to another "
                "grid; descriptors are grid-bound — build a new one "
                "per grid")
        self._grid = grid

    def _bound_grid(self) -> Grid:
        """Return the bound grid, raising when unattached."""
        if self._grid is None:
            raise RuntimeError(
                "this ImmersedDomain is not attached to a grid; "
                "pass it as Grid(..., immersed=...) or attach it "
                "via grid.with_immersed(...)")
        return self._grid

    def _clone_unbound(self) -> ImmersedDomain:
        """
        Return an unbound shallow clone re-attachable to a second grid.

        Description
        -----------
        The re-bind hook ``Grid.coarsened`` uses to attach the same wet
        region to a coarse sibling grid (MG-D6). The descriptor is
        array-free — the declaring callable and static parameters only —
        so a shallow copy shares the declaration by reference; the grid
        binding is reset and the materialization cache (keyed on the old
        grid's laid-out spaces) is dropped, so the coarse grid
        re-quadratures the wet fractions on its own coarse spaces.
        """
        clone = copy.copy(self)
        clone._grid = None  # noqa: SLF001 — re-bind seam (same class)
        clone._cache = {}  # noqa: SLF001 — fresh per-grid materialize cache
        return clone

    # ================================================================
    #  Derived per-space fields (derive-on-demand, section 3.7)
    # ================================================================
    def fraction(
        self,
        space: SpaceLike,
        *,
        fraction: ScalarField | None = None,
    ) -> ScalarField:
        r"""
        Wet fraction ``theta`` on ``space`` (volume/area fraction).

        Description
        -----------
        The fraction transfer is geometric and slip-independent
        (rules section 3.7, IP-D2): cell-positioned axes keep the
        cell fraction ``theta``; face-family axes combine the two
        adjacent cells with ``min`` (dry exterior on bounded meshes,
        wrap on periodic) — the MITgcm ``hFacW = min(hFacC)``
        precedent, whose boolean restriction is exactly the ``AND``
        of the adjacent cell fractions.

        Parameters
        ----------
        space : SpaceLike
            The target space (must resolve every grid coordinate).
        fraction : ScalarField | None, optional
            Explicit cell-fraction data (module-owned state) used
            instead of the static declaration; passed through
            unthresholded, still clipped to [0, 1] and floored
            (default: None).

        Returns
        -------
        ScalarField
            The wet fraction in [0, 1], tagged with ``space``.
        """
        grid = self._bound_grid()
        space = grid._laid_out(space)  # noqa: SLF001 — grid seam
        key = (space, "fraction", None, grid.decomposition.halo)
        if fraction is None:
            cached = self._cache.get(key)
            if cached is not None:
                return ScalarField(
                    grid, space, cached,
                    FieldMetadata.create(name="wet_fraction"))
        cells = self._cell_fraction(space, fraction)
        wet = _derive(space, cells, jnp.minimum)
        stored = store(grid.decomposition, space,
                       wet.astype(dtype_real()))
        stored = _closeable(jnp.clip(stored, 0.0, 1.0))
        if fraction is None and not isinstance(stored, jax.core.Tracer):
            self._cache[key] = stored
        return ScalarField(grid, space, stored,
                           FieldMetadata.create(name="wet_fraction"))

    def mask(
        self,
        space: SpaceLike,
        *,
        slip: Slip | None = None,
        fraction: ScalarField | None = None,
    ) -> ScalarField:
        r"""
        Boolean wet mask on ``space``: staggered ``theta > 0``.

        Description
        -----------
        The cell mask is ``theta > 0``; face axes combine the two
        adjacent cells by the slip rule — ``AND`` under ``NO_SLIP``,
        ``OR`` under ``FREE_SLIP`` (dry exterior on bounded meshes).
        On the collocation default (``order=None``) this is bitwise
        the ``WaterMask`` boolean subset.

        Parameters
        ----------
        space : SpaceLike
            The target space (must resolve every grid coordinate).
        slip : Slip | None, optional
            Override of the constructor's combination rule, for
            mixed-physics diagnostics (default: None).
        fraction : ScalarField | None, optional
            Explicit cell-fraction data (module-owned state) used
            instead of the static declaration (default: None).

        Returns
        -------
        ScalarField
            The boolean wet mask, tagged with ``space``.
        """
        grid = self._bound_grid()
        space = grid._laid_out(space)  # noqa: SLF001 — grid seam
        slip = self._slip if slip is None else slip
        if not isinstance(slip, Slip):
            raise TypeError(
                f"slip must be a Slip member, got {slip!r}")
        key = (space, "mask", slip, grid.decomposition.halo)
        if fraction is None:
            cached = self._cache.get(key)
            if cached is not None:
                return ScalarField(
                    grid, space, cached,
                    FieldMetadata.create(name="wet_mask"))
        combine = (jnp.logical_and if slip is Slip.NO_SLIP
                   else jnp.logical_or)
        cells = self._cell_fraction(space, fraction) > 0.0
        wet = _derive(space, cells, combine)
        stored = store(grid.decomposition, space,
                       wet.astype(dtype_real()))
        result = _closeable(stored > _WET_THRESHOLD)
        if fraction is None and not isinstance(result, jax.core.Tracer):
            self._cache[key] = result
        return ScalarField(grid, space, result,
                           FieldMetadata.create(name="wet_mask"))

    def centroid_offset(
        self,
        space: SpaceLike,
        name: str,
    ) -> ScalarField:
        r"""
        Wet-centroid offset ``delta`` along ``name`` (first moment).

        Description
        -----------
        The signed distance from a cell centre to the physical centroid
        of its wet region along the coordinate ``name`` (PB-D1): the
        first ``name``-moment of the wet region over its volume,

        .. math::

            \delta_c = \frac{\int_{cell}(z - z_c)\,\chi\;\mathrm{d}V}
                            {\int_{cell}\chi\;\mathrm{d}V},

        computed with the same per-cell Gauss-Legendre quadrature as
        :meth:`fraction` (``order >= 2``). A **bottom** cut shifts the
        vertical centroid (``delta != 0``); a **lateral** cut leaves it
        put (the ``name``-symmetric moment vanishes identically), and a
        full or dry cell gives exactly ``0`` — the volume fraction
        alone cannot tell a bottom cut from a lateral one, only the
        first moment can. The divide is sealed
        ``where(theta > 0, moment/(theta*V), 0)`` (never a live 0/0;
        dry cells are exactly ``0``). The collocation staircase
        (``order=None``/``1``) has no partial cells, so ``delta`` is
        identically ``0``. Static geometry: memoized concrete-only like
        :meth:`fraction`, never traced.

        Used by the hydrostatic partial-bottom pressure-gradient
        correction (Pacanowski & Gnanadesikan): a resting stratified
        column with ``b`` sampled at the wet-centroid heights
        ``z_c + delta`` (PB-D4) stays at rest to machine precision on a
        cut z-level grid, where the full-cell ``p_hyd`` integral alone
        differences pressures at mismatched heights.

        Parameters
        ----------
        space : SpaceLike
            The target cell space (all cell-centred factors; must
            resolve every grid coordinate).
        name : str
            The coordinate the moment is taken along (the vertical).

        Returns
        -------
        ScalarField
            The wet-centroid offset ``delta`` on ``space``, tagged with
            it (dry/full/lateral-cut cells exactly ``0``).
        """
        grid = self._bound_grid()
        space = grid._laid_out(space)  # noqa: SLF001 — grid seam
        _validate_factors(space, grid)
        _validate_centroid_space(space, name)
        key = (space, "centroid_offset", name, grid.decomposition.halo)
        cached = self._cache.get(key)
        if cached is not None:
            return ScalarField(
                grid, space, cached,
                FieldMetadata.create(name="wet_centroid_offset"))
        delta = self._centroid_cells(space, name)
        stored = _closeable(store(grid.decomposition, space,
                                  delta.astype(dtype_real())))
        if not isinstance(stored, jax.core.Tracer):
            self._cache[key] = stored
        return ScalarField(
            grid, space, stored,
            FieldMetadata.create(name="wet_centroid_offset"))

    def transition(
        self,
        space: SpaceLike,
        *,
        fraction: ScalarField | None = None,
    ) -> ScalarField:
        """
        Wet/dry transition-set indicator on ``space`` (designed-for).

        Parameters
        ----------
        space : SpaceLike
            The target space.
        fraction : ScalarField | None, optional
            Explicit cell-fraction data (default: None).

        Returns
        -------
        ScalarField
            Never returns in iteration 2.
        """
        raise NotImplementedError(
            "transition-set indicators are designed-for; iteration 2 "
            "implements the fraction/mask subset only")

    # ================================================================
    #  Base cell fraction (the single declared datum, materialized)
    # ================================================================
    def _cell_fraction(
        self,
        space: SpaceLike,
        fraction: ScalarField | None,
    ) -> jax.Array:
        """
        Materialize the float cell fraction in ``space`` factor order.

        Description
        -----------
        Explicit ``fraction`` data passes through (clipped, floored);
        the static declaration is either center-collocated and
        thresholded (``order=None``/``1`` — the {0, 1} staircase) or
        per-cell Gauss-Legendre quadratured (``order >= 2`` — genuine
        partial cells, clipped and floored).

        Parameters
        ----------
        space : SpaceLike
            The (laid-out) target space.
        fraction : ScalarField | None
            Explicit cell-fraction field, or None for the static
            declaration.

        Returns
        -------
        jax.Array
            The float cell fraction, one axis per space factor.
        """
        grid = self._bound_grid()
        factors = space.factors
        _validate_factors(space, grid)
        cell_shape = tuple(
            factor.mesh.n_cells for factor in factors)
        if fraction is not None:
            theta = self._explicit_cells(space, fraction, cell_shape)
            theta = jnp.clip(theta.astype(dtype_real()), 0.0, 1.0)
            return self._floor(theta)
        params = tuple(inspect.signature(self._init).parameters)
        names = tuple(
            name for factor in factors for name in factor.names)
        if set(params) != set(names):
            raise TypeError(
                "the immersed indicator must name exactly the grid "
                f"coordinate names {names}, got {params}")
        if self._order is None or self._order < _QUADRATURE_MIN:
            return self._collocation_cells(factors, cell_shape)
        theta = self._quadrature_cells(space)
        return self._floor(theta)

    def _collocation_cells(
        self,
        factors: tuple[FunctionSpace, ...],
        cell_shape: tuple[int, ...],
    ) -> jax.Array:
        """
        Center-collocate the indicator and threshold (the staircase).

        Parameters
        ----------
        factors : tuple[FunctionSpace, ...]
            The target space factors.
        cell_shape : tuple[int, ...]
            The global cell shape in target factor order.

        Returns
        -------
        jax.Array
            The {0.0, 1.0} cell fraction, one axis per factor.
        """
        coords: dict[str, jax.Array] = {}
        for axis, factor in enumerate(factors):
            shape = [1] * len(factors)
            shape[axis] = cell_shape[axis]
            coords[factor.names[0]] = _cell_centers(
                factor.mesh).reshape(shape)
        values = jnp.asarray(self._init(**coords))
        wet = jnp.broadcast_to(values, cell_shape) > _WET_THRESHOLD
        return wet.astype(dtype_real())

    def _quadrature_cells(self, space: SpaceLike) -> jax.Array:
        r"""
        Per-cell Gauss-Legendre quadrature of the declared indicator.

        Description
        -----------
        Reuses the grid's average-family ``_discretize`` machinery on
        the cell-average space of the target meshes (rules section
        3.10): each cell is quadratured on its **own** physical edges
        (the mesh ``coordinate_map`` seam), so a stretched axis
        averages each cell on its own scale. On a grid whose mapping
        declares column corrections (a terrain/chart column) the
        separable per-axis average is the *chart-parameter* average,
        not the physical wet-volume fraction, so the chart path
        (:meth:`_chart_quadrature_cells`, MI-D1) is taken instead: the
        indicator is sampled at the mapped physical node positions and
        the quadrature is weighted by the column Jacobian
        ``theta = int J chi / int J``. The result is clipped to
        [0, 1] (a declared indicator that overshoots is a user bug,
        not a geometry — the clip keeps ``theta`` a valid fraction).

        Parameters
        ----------
        space : SpaceLike
            The (laid-out) target space.

        Returns
        -------
        jax.Array
            The clipped cell fraction, one axis per factor.
        """
        grid = self._bound_grid()
        meshes = tuple(factor.mesh for factor in space.factors)
        avg = tuple(mesh.cell_avg for mesh in meshes)
        cell_space = (avg[0] if len(avg) == 1
                      else TensorProductSpace.of(*avg))
        cell_space = grid._laid_out(cell_space)  # noqa: SLF001 — seam
        mapping = grid.mapping
        if mapping is not None and mapping.column_corrections:
            theta = self._chart_quadrature_cells(cell_space, mapping)
        else:
            theta = grid._discretize(  # noqa: SLF001 — grid seam
                cell_space, self._init, self._order)
        return jnp.clip(theta, 0.0, 1.0)

    def _chart_quadrature_cells(
        self,
        cell_space: SpaceLike,
        mapping: CoordinateMapping,
    ) -> jax.Array:
        r"""
        Jacobian-weighted physical wet-volume fraction on a chart.

        Description
        -----------
        The designed-for chart path (MI-D1) of the per-cell fraction:
        on a grid whose mapping declares column corrections the
        physical volume element ``J = |partial m / partial b|`` varies
        within a cell, so the wet fraction is the **physical**
        wet-volume average

        .. math::

            theta_c = \frac{\int_{cell} J\,chi\;dV}
                           {\int_{cell} J\;dV},

        a tensor Gauss-Legendre quadrature over the cell whose nodes
        are placed at the mapped physical positions (the indicator is
        an object in physical space) and whose weights carry the
        column Jacobian. The node placement and reference weights come
        from the grid's own per-cell quadrature
        (:meth:`~fridom.spatial.grid.Grid._cell_quadrature_fields`);
        the mapped positions and ``J`` from the mapping's own map
        callable and its exact ``jax.jvp`` derivative
        (:meth:`~fridom.spatial.coordinate_mapping.CoordinateMapping._column_correction`)
        — fraction and operator consume one discretized geometry.
        Normalizing by the quadrature of ``J`` itself keeps
        ``theta in [0, 1]`` and an all-wet chart exactly ``1.0``.

        Parameters
        ----------
        cell_space : SpaceLike
            The laid-out cell-average target space.
        mapping : CoordinateMapping
            The grid's coordinate mapping (column corrections truthy).

        Returns
        -------
        jax.Array
            The physical wet-volume cell fraction, one axis per factor.
        """
        grid = self._bound_grid()
        node_by_name, weight, quad_axes = (
            grid._cell_quadrature_fields(  # noqa: SLF001 — grid seam
                cell_space, self._order))
        physical, jacobian = (
            mapping._column_correction(  # noqa: SLF001 — mapping seam
                node_by_name))
        chi = jnp.asarray(self._init(**physical))
        weighted = jacobian * weight
        numerator = jnp.sum(chi * weighted, axis=quad_axes)
        denominator = jnp.sum(weighted, axis=quad_axes)
        theta = numerator / denominator
        return jnp.broadcast_to(theta, cell_space.shape)

    def _centroid_cells(
        self, space: SpaceLike, name: str,
    ) -> jax.Array:
        r"""
        Per-cell wet-centroid ``name``-offset by first-moment quadrature.

        Description
        -----------
        The genuine per-cell quadrature behind :meth:`centroid_offset`.
        Reuses the grid's unreduced per-cell Gauss-Legendre fields
        (:meth:`~fridom.spatial.grid.Grid._cell_quadrature_fields`, the
        seam the chart fraction MI-D1 uses) so the moment weight
        ``(z - z_c)`` folds in before the sum: with unit-sum reference
        weights the two reduced sums are cell averages, so

        .. math::

            \delta_c = \frac{\langle z\,\chi\rangle}{\langle\chi\rangle}
                       - z_c,
            \qquad z_c = \langle z\rangle,

        the wet-region centroid offset (the common cell measure
        cancels). ``z_c`` is the quadrature of the node positions
        themselves (the symmetric rule's weighted mean = the cell
        centre), so it rides the same stretched per-cell geometry as
        the moment. The divide is sealed on ``theta > 0`` (dry cells
        exactly ``0``). The collocation staircase (``order`` below
        :data:`_QUADRATURE_MIN`) has no partial cells, so the offset is
        identically ``0``. A grid whose mapping declares column
        corrections (a genuine chart) is a taught error (PB-D3): the
        separable per-axis moment is the chart-parameter moment, not
        the physical wet-volume centroid.

        Parameters
        ----------
        space : SpaceLike
            The (laid-out) target cell space.
        name : str
            The coordinate the moment is taken along.

        Returns
        -------
        jax.Array
            The wet-centroid offset, one axis per space factor.
        """
        grid = self._bound_grid()
        factors = space.factors
        cell_shape = tuple(
            factor.mesh.n_cells for factor in factors)
        if self._order is None or self._order < _QUADRATURE_MIN:
            return jnp.zeros(cell_shape, dtype=dtype_real())
        params = tuple(inspect.signature(self._init).parameters)
        names = tuple(
            axis_name for factor in factors
            for axis_name in factor.names)
        if set(params) != set(names):
            raise TypeError(
                "the immersed indicator must name exactly the grid "
                f"coordinate names {names}, got {params}")
        mapping = grid.mapping
        if mapping is not None and mapping.column_corrections:
            raise NotImplementedError(
                "the wet-centroid offset on a chart (mapping with "
                "column corrections) is a taught error: the separable "
                "per-axis first moment is the chart-parameter moment, "
                "not the physical wet-volume centroid (partial-bottom "
                "p_hyd plan, PB-D3). The terrain-chart partial-cell "
                "correction is a recorded follow-up; use a flat / "
                "separable-stretched immersed grid")
        meshes = tuple(factor.mesh for factor in factors)
        avg = tuple(mesh.cell_avg for mesh in meshes)
        cell_space = (avg[0] if len(avg) == 1
                      else TensorProductSpace.of(*avg))
        cell_space = grid._laid_out(cell_space)  # noqa: SLF001 — seam
        node_by_name, weight, quad_axes = (
            grid._cell_quadrature_fields(  # noqa: SLF001 — grid seam
                cell_space, self._order))
        chi = jnp.asarray(self._init(**node_by_name))
        z_node = node_by_name[name]
        w_chi = weight * chi
        total = jnp.sum(weight, axis=quad_axes)
        theta = jnp.sum(w_chi, axis=quad_axes)
        z_chi = jnp.sum(z_node * w_chi, axis=quad_axes)
        # normalize z_c by the weight sum (not assuming it is exactly 1):
        # a full cell (chi == 1) then makes z_chi / theta and z_c the
        # *identical* expression, so delta is exactly 0 there (no FP
        # residue that would misfire the partial-cell activation).
        z_center = jnp.sum(z_node * weight, axis=quad_axes) / total
        wet = theta > 0.0
        theta_safe = jnp.where(wet, theta, 1.0)
        delta = jnp.where(wet, z_chi / theta_safe - z_center, 0.0)
        return jnp.broadcast_to(delta, cell_shape)

    def _floor(self, theta: jax.Array) -> jax.Array:
        r"""
        Apply the small-cell floor (IP-D3), a no-op on {0, 1}.

        Description
        -----------
        ``theta < min_fraction / 2 -> 0`` (the sliver is closed off),
        ``theta in [min_fraction / 2, min_fraction) -> min_fraction``
        (lifted to the stable floor); ``theta >= min_fraction`` and
        exact ``0.0``/``1.0`` are untouched. ``min_fraction == 0.0``
        disables the floor (the collocation staircase never needs
        it).

        Parameters
        ----------
        theta : jax.Array
            The clipped cell fraction.

        Returns
        -------
        jax.Array
            The floored cell fraction.
        """
        mf = self._min_fraction
        if mf == 0.0:
            return theta
        below = theta < mf / 2.0
        between = (theta >= mf / 2.0) & (theta < mf)
        theta = jnp.where(below, 0.0, theta)
        return jnp.where(between, mf, theta)

    def _explicit_cells(
        self,
        space: SpaceLike,
        fraction: ScalarField,
        cell_shape: tuple[int, ...],
    ) -> jax.Array:
        """
        Validate and broadcast explicit cell-fraction data.

        Description
        -----------
        Genuine fractions pass through unthresholded (boolean {0, 1}
        data is the special case); the clip and the floor are applied
        by the caller.

        Parameters
        ----------
        space : SpaceLike
            The (laid-out) target space.
        fraction : ScalarField
            The module-owned cell-fraction field: every non-constant
            factor must be a ``Center``/``CellAvg`` factor of one of
            the target space's meshes.
        cell_shape : tuple[int, ...]
            The cell shape in target factor order.

        Returns
        -------
        jax.Array
            The float cell fraction, one axis per space factor.
        """
        grid = self._bound_grid()
        if fraction.grid is not grid:
            raise ValueError(
                "the explicit fraction field lives on a different "
                "grid than this immersed domain")
        meshes = tuple(factor.mesh for factor in space.factors)
        source = fraction.function_space
        axes: dict[object, int] = {}
        for axis, factor in enumerate(source.factors):
            if isinstance(factor, ConstantSpace):
                continue
            is_center = (isinstance(factor, CellAvg)
                         or (isinstance(factor, NodalSpace)
                             and factor.node_set is NodeSet.CENTER))
            if not is_center or factor.mesh not in meshes:
                raise ValueError(
                    "explicit fraction data lives on the cell "
                    "family (Center/CellAvg factors of the target "
                    f"space's meshes); got {factor!r}")
            axes[factor.mesh] = axis
        data = fraction.data
        # move the source axes into target factor order, inserting
        # broadcast axes for meshes the source is constant along
        order = [axes[mesh] for mesh in meshes if mesh in axes]
        data = jnp.transpose(
            data, (*order, *(i for i in range(data.ndim)
                             if i not in order)))
        shape = tuple(
            cell_shape[i] if mesh in axes else 1
            for i, mesh in enumerate(meshes))
        data = data.reshape(shape)
        return jnp.broadcast_to(data, cell_shape)


# ================================================================
#  Constructor validators
# ================================================================
def _validate_order(order: int | None) -> None:
    """
    Verify ``order`` is None or a positive integer.

    Description
    -----------
    ``None`` (and ``1``) alias the collocation staircase; ``>= 2`` is
    the genuine quadrature. Rejects booleans and non-positive ints.

    Parameters
    ----------
    order : int | None
        The per-cell quadrature point count.
    """
    if order is not None and (
            isinstance(order, bool) or not isinstance(order, int)
            or order < 1):
        raise ValueError(
            "order= is the per-cell quadrature point count: a "
            f"positive integer or None (None/1 = collocation), got "
            f"{order!r}")


def _validate_min_fraction(min_fraction: float) -> None:
    """
    Verify ``min_fraction`` is a real number in ``[0, 1)``.

    Parameters
    ----------
    min_fraction : float
        The small-cell floor.
    """
    if (isinstance(min_fraction, bool)
            or not isinstance(min_fraction, numbers.Real)
            or not (0.0 <= min_fraction < 1.0)):
        raise ValueError(
            "min_fraction= is the small-cell floor: a real number in "
            f"[0, 1) (0.0 disables it), got {min_fraction!r}")


# ================================================================
#  Multi-process close-over guard (replicate static geometry)
# ================================================================
def _closeable(arr: jax.Array) -> jax.Array:
    r"""
    Make a materialized fraction/mask array legal to close over.

    Description
    -----------
    The per-space fraction/mask/centroid arrays are static geometry,
    memoized concrete-only (like ``grid._measures``) and reused by the
    traced step, which bakes each into the jit as a closed-over
    constant. Under a real multi-process launch (``srun -n N`` +
    ``jax.distributed``) ``store`` shards the array across the global
    device mesh, so it spans non-addressable devices and closing over
    it is illegal — ``jax`` raises ``Closing over jax.Array that spans
    non-addressable devices`` at lower time. Replicating such a
    concrete array gathers a full copy onto every process (fully
    addressable), which XLA reshards at each use site; the values are
    unchanged, so the physics is identical. On a single-controller
    launch (one device, or forced host devices) the array is already
    fully addressable and this is a no-op, so those runs — including
    the whole test suite — are byte-for-byte unaffected. A trace-time
    tracer (a query issued under the step jit, never cached) passes
    through untouched.

    Parameters
    ----------
    arr : jax.Array
        A materialized fraction/mask/centroid array, or a trace-time
        tracer.

    Returns
    -------
    jax.Array
        The array, replicated when it was a concrete array spanning
        non-addressable devices; otherwise ``arr`` unchanged.
    """
    if isinstance(arr, jax.core.Tracer) or arr.is_fully_addressable:
        return arr
    replicated = jax.sharding.NamedSharding(
        arr.sharding.mesh, jax.sharding.PartitionSpec())
    return jax.device_put(arr, replicated)


# ================================================================
#  Per-factor staggering transfer (fraction min / mask AND-OR)
# ================================================================
def _validate_factors(space: SpaceLike, grid: Grid) -> None:
    """Reject factor families outside the mask/fraction rules."""
    for factor in space.factors:
        if isinstance(factor, CoefficientSpace):
            # a value error (bad space choice), not a type error
            raise ValueError(  # noqa: TRY004
                f"masks have no coefficient-space representation "
                f"({factor!r}); masked domains on spectral bases "
                "are volume penalization (designed-for, rules "
                "section 3.7)")
        if isinstance(factor, ConstantSpace):
            # a value error (bad space choice), not a type error
            raise ValueError(  # noqa: TRY004
                f"the mask is not resolved along {factor!r}; "
                "constant factors cannot carry the wet region")
    names = tuple(
        name for factor in space.factors for name in factor.names)
    if set(names) != set(grid.names):
        raise ValueError(
            "immersed masks are derived on spaces resolving every "
            f"grid coordinate {grid.names}; this space resolves "
            f"{names}")


def _validate_centroid_space(space: SpaceLike, name: str) -> None:
    """Require a cell-centred space and a resolved moment axis.

    Description
    -----------
    The wet-centroid offset is a per-cell quantity, so every factor
    must be cell-positioned (nodal ``Center`` or FV ``CellAvg``) — no
    face staggering (a moment has no min-transfer). ``name`` must name
    one of the space's coordinates (the moment axis).
    """
    for factor in space.factors:
        is_cell = (isinstance(factor, CellAvg)
                   or (isinstance(factor, NodalSpace)
                       and factor.node_set is NodeSet.CENTER))
        if not is_cell:
            raise ValueError(
                "the wet-centroid offset is a cell quantity: every "
                "factor must be cell-positioned (Center / CellAvg), "
                f"got {factor!r}")
    names = tuple(
        axis for factor in space.factors for axis in factor.names)
    if name not in names:
        raise ValueError(
            f"the moment coordinate {name!r} is not a coordinate of "
            f"this space (resolves {names})")


def _cell_centers(mesh: object) -> jax.Array:
    """
    Materialize the 1D cell-center coordinates of one mesh.

    Description
    -----------
    The cell centers are the collocation points at which the declared
    indicator is sampled (rules section 3.7). The computational
    placement ``s = (i + 0.5) / n`` composes with the mesh's
    ``coordinate_map`` geometry seam (concepts section 2.7), exactly
    like ``grid.evaluation_nodes`` — so masks honor a
    ``MappedIntervalMesh`` stretch rather than assuming the uniform
    placement of a scalar ``dx``. ``coordinate_map is None`` is the
    uniform affine placement read off ``extent`` and ``dx`` (the
    constant special case XLA folds).

    Parameters
    ----------
    mesh : Mesh
        A grid mesh factor (iteration 2: the interval meshes, uniform
        ``IntervalMesh`` or stretched ``MappedIntervalMesh``).

    Returns
    -------
    jax.Array
        The ``n_cells`` cell-center coordinates.
    """
    if not isinstance(mesh, IntervalMesh | MappedIntervalMesh):
        raise NotImplementedError(
            f"immersed masks on {type(mesh).__name__} arrive in a "
            "later wave; iteration 2 covers the interval meshes "
            "(uniform and mapped)")
    n = mesh.n_cells
    steps = jnp.arange(n, dtype=dtype_real()) + 0.5
    mapping = mesh.coordinate_map
    if mapping is None:
        return mesh.extent[0] + steps * mesh.dx
    return jnp.asarray(mapping(steps / n)).astype(dtype_real())


def _derive(
    space: SpaceLike,
    cells: jax.Array,
    combine: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """
    Transfer the cell datum onto the space's staggered node sets.

    Parameters
    ----------
    space : SpaceLike
        The (laid-out) target space.
    cells : jax.Array
        The cell datum (float fraction or boolean mask), one axis per
        factor.
    combine : Callable[[jax.Array, jax.Array], jax.Array]
        The two-neighbor combination rule (``min`` for fractions,
        ``AND``/``OR`` for masks).

    Returns
    -------
    jax.Array
        The transferred datum at the space's true shape.
    """
    arr = cells
    for axis, factor in enumerate(space.factors):
        arr = _stagger_axis(arr, axis, factor, combine)
    return arr


def _take(arr: jax.Array, axis: int, sl: slice) -> jax.Array:
    """Slice ``arr`` with ``sl`` along ``axis``."""
    index: list[slice] = [slice(None)] * arr.ndim
    index[axis] = sl
    return arr[tuple(index)]


def _stagger_axis(
    arr: jax.Array,
    axis: int,
    factor: FunctionSpace,
    combine: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """
    Transfer the cell datum to one factor's node set along ``axis``.

    Description
    -----------
    Cell-positioned factors (``Center``/``CellAvg``) keep the cell
    datum; face-family node sets combine the two adjacent cells with
    ``combine`` — wrapping on periodic meshes, with a dry exterior
    on bounded ones. BC-constrained boundary DOFs are dropped exactly
    like the space shapes drop them.

    Parameters
    ----------
    arr : jax.Array
        The datum, cell-positioned along ``axis``.
    axis : int
        The array axis of this factor.
    factor : FunctionSpace
        The factor space owning the axis.
    combine : Callable[[jax.Array, jax.Array], jax.Array]
        The two-neighbor combination rule.

    Returns
    -------
    jax.Array
        The datum on the factor's node set along ``axis``.
    """
    node_set, membership, components = _axis_geometry(factor)
    if node_set is NodeSet.CENTER:
        staggered = arr
    elif factor.mesh.periodic:
        staggered = _stagger_periodic(arr, axis, node_set, combine)
    else:
        staggered = _stagger_bounded(arr, axis, node_set, combine)
    return _drop_constrained(staggered, axis, membership, components)


def _axis_geometry(
    factor: FunctionSpace,
) -> tuple[NodeSet, tuple[bool, bool], tuple[BC, ...]]:
    """
    Resolve one factor's node set, membership, and BC components.

    Parameters
    ----------
    factor : FunctionSpace
        A nodal or average factor space.

    Returns
    -------
    tuple[NodeSet, tuple[bool, bool], tuple[BC, ...]]
        The node set the datum is derived at, whether the (left,
        right) end DOF sits on the boundary, and the BC structure
        (empty for average factors: no DOF drops).
    """
    if isinstance(factor, CellAvg):
        return NodeSet.CENTER, (False, False), ()
    if isinstance(factor, FaceAvg):
        node_set = (NodeSet.RIGHT if factor.mesh.periodic
                    else NodeSet.INNER)
        return node_set, (False, False), ()
    if isinstance(factor, NodalSpace):
        node_set = factor.node_set
        return (node_set, _BOUNDARY_MEMBERSHIP[node_set],
                factor.bc.components)
    raise NotImplementedError(
        f"immersed masks on {factor!r} are not defined in "
        "iteration 2")


def _stagger_periodic(
    arr: jax.Array,
    axis: int,
    node_set: NodeSet,
    combine: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """
    Combine adjacent cells onto a periodic face node set.

    Parameters
    ----------
    arr : jax.Array
        The datum, cell-positioned along ``axis``.
    axis : int
        The array axis of this factor.
    node_set : NodeSet
        The (non-center) target node set.
    combine : Callable[[jax.Array, jax.Array], jax.Array]
        The two-neighbor combination rule.

    Returns
    -------
    jax.Array
        The datum at the node set along ``axis``.
    """
    if node_set is NodeSet.RIGHT:
        return combine(arr, jnp.roll(arr, -1, axis))
    if node_set is NodeSet.LEFT:
        return combine(jnp.roll(arr, 1, axis), arr)
    # unreachable through the space factories (outer/inner raise on
    # periodic meshes)
    raise NotImplementedError(  # pragma: no cover
        f"node set {node_set} is undefined on periodic meshes")


def _stagger_bounded(
    arr: jax.Array,
    axis: int,
    node_set: NodeSet,
    combine: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """
    Combine adjacent cells onto a bounded face node set.

    Description
    -----------
    The exterior counts as dry: a boundary face has a single wet
    neighbor, combined against a dry value (``0.0`` for fractions,
    ``False`` for masks — the ``arr.dtype`` zero either way).

    Parameters
    ----------
    arr : jax.Array
        The datum, cell-positioned along ``axis``.
    axis : int
        The array axis of this factor.
    node_set : NodeSet
        The (non-center) target node set.
    combine : Callable[[jax.Array, jax.Array], jax.Array]
        The two-neighbor combination rule.

    Returns
    -------
    jax.Array
        The datum at the node set along ``axis``.
    """
    dry_shape = list(arr.shape)
    dry_shape[axis] = 1
    dry = jnp.zeros(tuple(dry_shape), dtype=arr.dtype)
    tail = slice(1, None)
    head = slice(None, -1)
    if node_set is NodeSet.RIGHT:
        right = jnp.concatenate(
            [_take(arr, axis, tail), dry], axis=axis)
        return combine(arr, right)
    if node_set is NodeSet.LEFT:
        left = jnp.concatenate(
            [dry, _take(arr, axis, head)], axis=axis)
        return combine(left, arr)
    if node_set is NodeSet.OUTER:
        return combine(jnp.concatenate([dry, arr], axis=axis),
                       jnp.concatenate([arr, dry], axis=axis))
    if node_set is NodeSet.INNER:
        return combine(_take(arr, axis, head),
                       _take(arr, axis, tail))
    raise NotImplementedError(
        f"immersed masks at {node_set} are not defined in "
        "iteration 2")


def _drop_constrained(
    arr: jax.Array,
    axis: int,
    membership: tuple[bool, bool],
    components: tuple[BC, ...],
) -> jax.Array:
    """
    Drop BC-constrained boundary DOFs, like the space shapes do.

    Parameters
    ----------
    arr : jax.Array
        The datum at the full node set along ``axis``.
    axis : int
        The array axis of this factor.
    membership : tuple[bool, bool]
        Whether the (left, right) end DOF sits on the boundary.
    components : tuple[BC, ...]
        The factor's BC structure (empty: no drops).

    Returns
    -------
    jax.Array
        The datum at the factor's true DOF count along ``axis``.
    """
    start: int | None = None
    stop: int | None = None
    if components:
        left_bc, right_bc = components
        if membership[0] and left_bc is not BC.NONE:
            start = 1
        if membership[1] and right_bc is not BC.NONE:
            stop = -1
    if start is None and stop is None:
        return arr
    return _take(arr, axis, slice(start, stop))


# whether the (left, right) boundary DOF is a member of the node set
# on a bounded mesh (mirrors the shape rule of the nodal spaces)
_BOUNDARY_MEMBERSHIP: dict[NodeSet, tuple[bool, bool]] = {
    NodeSet.CENTER: (False, False),
    NodeSet.LEFT: (True, False),
    NodeSet.RIGHT: (False, True),
    NodeSet.OUTER: (True, True),
    NodeSet.INNER: (False, False),
}
