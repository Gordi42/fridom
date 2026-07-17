"""
``Grid``: the model-agnostic assembly root.

Description
-----------
Owning class doc: ``design/specs/grid/classes/grid.md``. Meshes +
decomposition + dispatch + field factory: ergonomics and wiring only,
all mathematics lives in spaces and operators. Iteration-1 subset:
``__init__`` seeds the default ``OperatorRegistry`` with the
grid-free iteration-1 rows (overridable via ``dispatch=``) and ends
with the provisional negotiation (grid lifecycle step 2) — the halo
is the per-operator maximum over the seeded registry, sound under
the sync-after-every-operator contract. Phase-2 assembly runs
``merge_overrides(...)`` -> ``negotiate(state_spaces=...,
tendency=...)`` -> ``freeze()``; ``freeze()`` records the
negotiation fingerprint, after which mutators raise
``GridFrozenError`` and ``negotiate`` switches to the
demand-satisfaction verify path (model D4/D5).
"""
# Wave 2: Grid -- Wave 3: negotiate/freeze lifecycle -- Phase 2:
#    merge_overrides facade, freeze fingerprint + verify path
from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from fridom.framework.utils import dtype_real
from fridom.spatial.bc import BC, BCStructure
from fridom.spatial.decomposition.decomposition import (
    ReshardingReport,
    _registry_halo,
    negotiate,
)
from fridom.spatial.decomposition.halo import (
    HaloSpec,
    trace_halo,
)
from fridom.spatial.errors import (
    GridFrozenError,
    GridMismatchError,
    SpaceMismatchError,
)
from fridom.spatial.fields.metadata import FieldMetadata
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.fields.storage import (
    factor_axes,
    hermitian_project,
    storage_dtype,
    store,
)
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.structured_1d import (
    StructuredMesh1D,
)
from fridom.spatial.operators.base import OperatorRequirements
from fridom.spatial.operators.chebyshev import Chebyshev
from fridom.spatial.operators.composed import (
    Curl,
    Divergence,
    Gradient,
    Laplacian,
    LowerIndex,
    MetricCurl,
    MetricDivergence,
    MetricGradient,
    MetricLaplacian,
    RaiseIndex,
)
from fridom.spatial.operators.cumulative import CumulativeIntegral
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.spatial.operators.flux_diff import (
    DualFluxDifference,
    FaceDifference,
    FluxDifference,
    FVDerivative,
)
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.operators.integrate import Integral
from fridom.spatial.operators.interp import LinearInterp
from fridom.spatial.operators.mapped import MappedDerivative
from fridom.spatial.operators.products import (
    Abs,
    CollocationProduct,
    Divide,
    Power,
)
from fridom.spatial.operators.reconstruct import (
    LinearDeconvolution,
    LinearReconstruction,
)
from fridom.spatial.operators.registry import (
    DispatchCollisionError,
    LazyEntry,
    OperatorRegistry,
    check_override_key,
)
from fridom.spatial.operators.restrict import Restriction
from fridom.spatial.operators.select import Where
from fridom.spatial.operators.spectral import (
    PhaseShift,
    SincShift,
    SpectralDerivative,
    chebyshev_modes,
    fourier_wavenumbers,
    trig_wavenumbers,
)
from fridom.spatial.operators.trig import Cosine, Sine
from fridom.spatial.random_fields import RandomFieldFactory
from fridom.spatial.scalars import Scalars

# the declaration-tag vocabulary of the ("declared_space", mesh)
# resolver rows (model D1.2); the model layer resolves declared
# patterns through the rows seeded below, so the grid must speak the
# tag enum (a same-layer spatial import of pure vocabulary)
from fridom.spatial.space_patterns import FAMILIES, Dof
from fridom.spatial.spaces.average import (
    AverageSpace,
    CellAvg,
    FaceAvg,
)
from fridom.spatial.spaces.coefficient import (
    ChebyshevSpace,
    CoefficientSpace,
    CosineSpace,
    FourierSpace,
    SineSpace,
)
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet
from fridom.spatial.spaces.tensor_product import (
    TensorProductSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.spatial.coordinate_mapping import (
        CoordinateMapping,
    )
    from fridom.spatial.decomposition.decomposition import (
        Decomposition,
    )
    from fridom.spatial.decomposition.layout import Layout
    from fridom.spatial.immersed_domain import ImmersedDomain
    from fridom.spatial.meshes.mesh import Mesh
    from fridom.spatial.operators.base import Operator
    from fridom.spatial.operators.registry import DispatchKey
    from fridom.spatial.operators.transform import Transform
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike


@dataclass(frozen=True)
class NegotiationFingerprint:

    """
    The negotiation record ``freeze()`` seals (model D4).

    Description
    -----------
    What a subsequent model assembly on the frozen grid verifies its
    demands against (grid.md "Frozen-grid verify path"): the
    verification is demand satisfaction (subset / less-or-equal,
    model D5), never equality, and adopting a new
    ConstantSpace-broadcast state space extends ``state_spaces``
    without reopening negotiation.

    Parameters
    ----------
    state_spaces : frozenset[SpaceLike]
        The bare state spaces of the last pre-freeze negotiation
        (plus post-freeze broadcast adoptions).
    override_keys : frozenset[DispatchKey]
        The normalized dispatch keys merged via ``merge_overrides``.
    halo : HaloSpec
        The negotiated per-name ghost widths.
    layouts : tuple[Layout, ...]
        The negotiated layout vocabulary.
    """

    state_spaces: frozenset[SpaceLike]
    override_keys: frozenset[DispatchKey]
    halo: HaloSpec
    layouts: tuple[Layout, ...]


class Grid:

    """
    Assembly root: meshes, decomposition, dispatch, field factory.

    Description
    -----------
    Fully static: not a pytree container, appears only as
    identity-hashed static aux data in field pytrees. Coordinate
    names come from the meshes (mandatory at mesh construction);
    the constructor validates flat uniqueness and ends with the
    provisional negotiation (grid lifecycle step 2), so a grid is
    fully usable interactively right after construction.

    Parameters
    ----------
    meshes : tuple[Mesh, ...]
        Pre-built, pre-named mesh factors, in coordinate order.
    dispatch : object | None, optional
        The operator dispatch registry (duck-typed
        ``OperatorRegistry``); None seeds the default iteration-1
        registry from the meshes' space families (default: None).
    mapping : CoordinateMapping | None, optional
        The coordinate-mapping descriptor to attach; the grid binds
        it on attachment and seeds the ``"physical_diff"``
        derivative kind for the coordinates it couples
        (default: None).
    immersed : ImmersedDomain | None, optional
        The immersed (masked) domain descriptor to attach; the grid
        binds it on attachment (default: None).
    device_ids : tuple[int, ...] | None, optional
        Indices into ``jax.devices()``; None lets negotiation use
        every available device, falling back to one when nothing is
        shardable (default: None).
    family : str, optional
        The default discretization family the declared-space
        patterns resolve into when they name none (FV-D1b):
        ``"nodal"`` (the ``Center`` / face point-value family) or
        ``"fv"`` (the average family — ``COLLOCATED`` lands on
        ``CellAvg``). A per-field ``SpacePattern(family=...)``
        overrides it, so a mixed model keeps most fields nodal while
        a tracer is FV. The default stays ``"nodal"``; flipping it to
        ``"fv"`` per periodic grid is stage F3 (default: "nodal").
    """

    def __init__(
        self,
        meshes: tuple[Mesh, ...],
        *,
        dispatch: object | None = None,
        mapping: CoordinateMapping | None = None,
        immersed: ImmersedDomain | None = None,
        device_ids: tuple[int, ...] | None = None,
        family: str = "nodal",
        _allow_replicated: bool = False,
    ) -> None:
        """Assemble a grid from pre-built, pre-named mesh factors."""
        meshes = tuple(meshes)
        if not meshes:
            raise ValueError("a grid needs at least one mesh factor")
        if family not in FAMILIES:
            raise ValueError(
                f"the grid-level default family must be one of "
                f"{FAMILIES}, got {family!r}")
        self._default_family: str = family
        names: list[str] = []
        duplicates: list[str] = []
        for mesh in meshes:
            for name in mesh.names:
                if name in names:
                    duplicates.append(name)
                names.append(name)
        if duplicates:
            raise ValueError(
                "duplicate coordinate names across mesh factors: "
                f"{tuple(duplicates)} (the tensor product's flat "
                "namespace requires unique names)")
        self._meshes: tuple[Mesh, ...] = meshes
        self._names: tuple[str, ...] = tuple(names)
        # attach the mapping before seeding: the bind validates its
        # coordinate vocabulary against the names collected above,
        # and the default registry seeds the mapped derivative kind
        # off the coupling table
        self._mapping: CoordinateMapping | None = None
        if mapping is not None:
            mapping._bind(self)  # noqa: SLF001 — attachment seam
            self._mapping = mapping
        self._dispatch: object = (
            _default_registry(self, meshes, mapping)
            if dispatch is None else dispatch)
        self._device_ids: tuple[int, ...] | None = device_ids
        # replicated-fallback flag (MG-D5): only ``Grid.coarsened``
        # sets it, so the public constructor's behavior is unchanged;
        # threaded through both negotiation calls so a coarse level
        # renegotiated in phase B stays replication-capable.
        self._allow_replicated: bool = _allow_replicated
        self._frozen: bool = False
        # negotiation-fingerprint bookkeeping (grid lifecycle;
        # sealed by freeze())
        self._state_spaces: frozenset[SpaceLike] = frozenset()
        self._override_keys: set[DispatchKey] = set()
        self._fingerprint: NegotiationFingerprint | None = None
        self._immersed: ImmersedDomain | None = None
        if immersed is not None:
            self._attach_immersed(immersed)
        # provisional negotiation: halo = per-operator maximum over
        # the registry (grid lifecycle step 2; exact under the
        # iteration-1 sync-after-every-operator contract)
        self._decomposition: Decomposition = negotiate(
            self, self._dispatch, device_ids=device_ids,
            allow_replicated=_allow_replicated)
        self._random: RandomFieldFactory = RandomFieldFactory(self)
        # measure fields are static mesh geometry (no params seam),
        # so they are memoized per (laid-out space, factor name) —
        # identity keys, the spaces are interned. A stable field
        # object also lets the operator-level sync memo hit
        # (operators/base.py), removing the per-application exchange
        # of freshly built measures.
        self._measures: dict[tuple[SpaceLike, str], ScalarField] = {}

    # ================================================================
    #  Identity
    # ================================================================
    def __eq__(self, other: object) -> bool:
        """Identity comparison: ``self is other`` (static aux)."""
        return self is other

    def __hash__(self) -> int:
        """Identity hash, consistent with ``__eq__``."""
        return id(self)

    # ================================================================
    #  Structure
    # ================================================================
    @property
    def factors(self) -> tuple[Mesh, ...]:
        """The Mesh factor objects, in constructor order."""
        return self._meshes

    @property
    def names(self) -> tuple[str, ...]:
        """All coordinate names, collected from the meshes in order."""
        return self._names

    @property
    def default_family(self) -> str:
        """
        The default discretization family for declared patterns.

        Description
        -----------
        The grid-level default (FV-D1b) a ``SpacePattern`` with no
        ``family=`` of its own resolves through: ``"nodal"`` (the
        ``Center`` / face point-value family) or ``"fv"`` (the
        average family). A per-field ``family=`` overrides it.
        """
        return self._default_family

    def set_default_family(self, family: str) -> None:
        """
        Set the grid-level default family (assembly phase; FV-D3).

        Description
        -----------
        The pre-freeze mutation a model assembly uses to make the
        grid's declared patterns resolve into a chosen family (stage
        F3): the nonhydro model factory flips a periodic grid to
        ``"fv"`` here, so every ``family=None`` field of the model
        (velocities, pressure, tracers) follows uniformly — an FV
        model has no accidental nodal field. Parallel to
        ``merge_overrides`` (both are assembly-phase grid mutations);
        on a frozen grid it is a no-op when the family is unchanged
        and raises otherwise (the family was fixed by an earlier
        model).

        Parameters
        ----------
        family : str
            The family to adopt (``"nodal"`` or ``"fv"``).

        Raises
        ------
        ValueError
            If ``family`` is not a known family name.
        GridFrozenError
            If the grid is frozen and ``family`` differs from the
            recorded default.
        """
        if family not in FAMILIES:
            raise ValueError(
                f"the grid-level default family must be one of "
                f"{FAMILIES}, got {family!r}")
        if self._frozen:
            if family == self._default_family:
                return
            raise GridFrozenError(
                "the grid is frozen; set_default_family is legal in "
                "the assembly phase only (an earlier model fixed the "
                f"family to {self._default_family!r})")
        self._default_family = family

    # ================================================================
    #  Hierarchy (coarse sibling grids)
    # ================================================================
    def coarsened(
        self,
        factors: Mapping[str, int] | int,
        *,
        device_ids: tuple[int, ...] | None = None,
    ) -> Grid:
        """
        Assemble the coarse sibling grid (multigrid / regrid levels).

        Description
        -----------
        Rebuilds a grid from per-coordinate coarsened meshes
        (``Mesh.coarsened``, MG-D3) with the same attachments
        (``CoordinateMapping`` / ``ImmersedDomain`` descriptors cloned
        and re-bound; metrics and fractions re-derive on the coarse
        spaces, MG-D6), the same default family, and a fresh default
        registry. Model-level dispatch overrides (merged via
        ``merge_overrides``) do **not** carry over. The result is
        unfrozen with a provisional negotiation over the **same device
        set**, replicating below the shardability floor (MG-D5) instead
        of raising. A missing coordinate name defaults to factor 1
        (semicoarsening, MG-D4); a factor > 1 on a multi-name or
        non-``StructuredMesh1D`` mesh raises.

        Parameters
        ----------
        factors : Mapping[str, int] | int
            Per-coordinate-name integer divisors (missing names default
            to 1), or one uniform divisor applied to every name.
        device_ids : tuple[int, ...] | None, optional
            Override the device set; None inherits this grid's
            (default: None).

        Returns
        -------
        Grid
            The coarse sibling grid (unfrozen, provisionally
            negotiated).

        Raises
        ------
        ValueError
            If a factor is not a positive integer, names an unknown
            coordinate, or exceeds 1 on a multi-name / non-structured
            mesh.
        """
        factor_map = self._normalize_factors(factors)
        new_meshes: list[Mesh] = []
        for mesh in self._meshes:
            mesh_factors = {factor_map[name] for name in mesh.names}
            if mesh_factors == {1}:
                new_meshes.append(mesh)  # pass-through (MG-D4)
                continue
            if (len(mesh.names) != 1
                    or not isinstance(mesh, StructuredMesh1D)):
                raise ValueError(
                    f"cannot coarsen {mesh!r} by "
                    f"{sorted(mesh_factors)}: only single-name "
                    "StructuredMesh1D factors coarsen (a multi-name or "
                    "non-structured mesh must keep factor 1)")
            new_meshes.append(mesh.coarsened(factor_map[mesh.names[0]]))
        mapping = (None if self._mapping is None
                   else self._mapping._clone_unbound())  # noqa: SLF001
        immersed = (None if self._immersed is None
                    else self._immersed._clone_unbound())  # noqa: SLF001
        return Grid(
            tuple(new_meshes),
            mapping=mapping,
            immersed=immersed,
            device_ids=(self._resolved_device_ids()
                        if device_ids is None else device_ids),
            family=self._default_family,
            _allow_replicated=True)

    def _resolved_device_ids(self) -> tuple[int, ...]:
        """
        Return the ``jax.devices()`` indices this grid actually holds.

        Description
        -----------
        Pins the coarse sibling to the fine grid's realized device set
        (MG-D5): an auto-selected (``device_ids=None``) fine grid that
        sharded over every device would otherwise let the coarse level
        *auto-fall-back to one device* — two levels on different device
        meshes, which the transfer refuses. Resolving the fine mesh's
        devices to explicit indices makes the coarse level land on the
        same set and replicate there when it can no longer shard.
        """
        index_of = {dev: i for i, dev in enumerate(jax.devices())}
        devices = self._decomposition.device_mesh.devices.flatten()
        return tuple(index_of[dev] for dev in devices)

    def _normalize_factors(
        self, factors: Mapping[str, int] | int,
    ) -> dict[str, int]:
        """
        Normalize the ``coarsened`` factor argument to a per-name dict.

        Description
        -----------
        One uniform integer applies to every coordinate; a mapping fills
        missing names with 1 (semicoarsening). Every value must be a
        positive integer (booleans rejected), and every mapping key must
        be a grid coordinate name.

        Parameters
        ----------
        factors : Mapping[str, int] | int
            The user divisor argument.

        Returns
        -------
        dict[str, int]
            The per-name integer divisors (every grid name present).
        """
        if isinstance(factors, Mapping):
            unknown = tuple(name for name in factors
                            if name not in self._names)
            if unknown:
                raise ValueError(
                    f"unknown coordinate names {unknown} in factors=; "
                    f"grid coordinates are {self._names}")
            resolved = {name: factors.get(name, 1)
                        for name in self._names}
        elif isinstance(factors, int) and not isinstance(factors, bool):
            resolved = dict.fromkeys(self._names, factors)
        else:
            raise TypeError(
                "factors must be a per-name mapping or one uniform "
                f"integer divisor, got {factors!r}")
        for name, value in resolved.items():
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value < 1):
                raise ValueError(
                    f"coarsening factor for {name!r} must be a positive "
                    f"integer, got {value!r}")
        return resolved

    # ================================================================
    #  Operator dispatch (seam: registry class owned by the
    #  operators cluster; the grid owns the instance)
    # ================================================================
    @property
    def dispatch(self) -> OperatorRegistry:
        """The operator dispatch registry (defaults + overrides)."""
        return self._dispatch

    def merge_overrides(
        self,
        overrides: Mapping[DispatchKey, Operator]
            | Mapping[str, Mapping[DispatchKey, Operator]],
    ) -> None:
        """
        Merge module-local dispatch overrides (pre-freeze only).

        Description
        -----------
        The facade over the pure registry (grid.md "Merge call
        site"): calls ``OperatorRegistry.merge`` — which returns a
        new registry — and swaps the held instance; model assembly
        step 3 calls it exactly once per grid. `overrides` is one
        flat override mapping, or the per-module form
        ``{module_name: {key: operator}}``: there the same resolved
        key contributed by two modules raises
        ``DispatchCollisionError`` naming both (module order never
        silently selects an operator). ``("declared_space", ...)``
        resolver rows are never module-mergeable and are rejected
        in either form (model D1.2).

        Parameters
        ----------
        overrides : Mapping
            Module override entries, flat or keyed by module name.

        Raises
        ------
        GridFrozenError
            If the grid is frozen (grid lifecycle step 3).
        DispatchCollisionError
            If two modules contribute the same resolved key.
        ValueError
            If an override adds or shadows a resolver row.
        """
        if self._frozen:
            raise GridFrozenError(
                "the grid is frozen; merge_overrides is legal in "
                "the assembly phase only (grid lifecycle)")
        flat = _flatten_overrides(overrides)
        merged = self._dispatch.merge(flat)
        self._override_keys |= {
            check_override_key(key) for key in flat}
        self._dispatch = merged

    # ================================================================
    #  Decomposition and lifecycle
    # ================================================================
    @property
    def decomposition(self) -> Decomposition:
        """The negotiated domain decomposition (grid-owned)."""
        return self._decomposition

    def negotiate(
        self,
        *,
        state_spaces: tuple[SpaceLike, ...] | None = None,
        tendency: Callable[..., object] | None = None,
        halo: HaloSpec | None = None,
    ) -> ReshardingReport:
        """
        Renegotiate the decomposition (verify-only once frozen).

        Description
        -----------
        Re-runs the mesh-traits x operator-demands negotiation
        (Phase-2 assembly, grid lifecycle step 3): the halo is the
        pointwise maximum of the traced `tendency` demand and the
        explicit `halo=` extra (declared bypasses) — they combine,
        neither shadows; with neither given, the per-operator
        registry maximum scoped to `state_spaces` applies. Traced
        widths may be capped for shardability (task 1.8: correctness
        is width-independent above the per-application floor).
        Returns the report the model uses to re-``device_put`` its
        live state once; derived arrays need nothing
        (recompute-on-demand).

        On a **frozen** grid nothing is renegotiated: the demands
        are *verified* against the recorded fingerprint — demand
        satisfaction (subset / less-or-equal, model D5), with new
        ConstantSpace-broadcast state spaces adopted into the record
        — and the unchanged-layout report is returned.

        Parameters
        ----------
        state_spaces : tuple[SpaceLike, ...] | None, optional
            The model's state-field spaces (default: None).
        tendency : Callable[..., object] | None, optional
            The tendency to halo-trace (default: None).
        halo : HaloSpec | None, optional
            Extra per-name halo demand, merged per-coordinate max
            into the traced demand (default: None).

        Returns
        -------
        ReshardingReport
            Old/new default layout and whether they differ.

        Raises
        ------
        GridFrozenError
            On a frozen grid, if the demands exceed the recorded
            negotiation fingerprint (diff-style message; "assemble
            the most demanding model first").
        """
        if self._frozen:
            return self._verify_frozen(
                state_spaces=state_spaces, tendency=tendency,
                halo=halo)
        old = self._decomposition.default_layout
        self._decomposition = negotiate(
            self, self._dispatch,
            state_spaces=state_spaces, tendency=tendency, halo=halo,
            device_ids=self._device_ids,
            allow_replicated=self._allow_replicated)
        self._state_spaces = (
            frozenset() if state_spaces is None
            else frozenset(s.bare for s in state_spaces))
        new = self._decomposition.default_layout
        return ReshardingReport(old=old, new=new, changed=old != new)

    def freeze(self) -> None:
        """
        End the assembly phase; record the negotiation fingerprint.

        Description
        -----------
        Seals the grid (grid lifecycle step 3): the state-space set,
        the merged override keys, the negotiated ``HaloSpec``, and
        the layout vocabulary are recorded as the
        ``NegotiationFingerprint`` that post-freeze ``negotiate``
        calls verify against. After ``freeze()``,
        ``merge_overrides`` and ``with_immersed`` raise
        ``GridFrozenError``. Idempotent: freezing a frozen grid
        keeps the existing record (including broadcast adoptions).
        """
        if self._frozen:
            return
        self._fingerprint = NegotiationFingerprint(
            state_spaces=self._state_spaces,
            override_keys=frozenset(self._override_keys),
            halo=self._decomposition.halo,
            layouts=self._decomposition.layouts)
        self._frozen = True

    @property
    def fingerprint(self) -> NegotiationFingerprint | None:
        """The frozen negotiation record; None before ``freeze()``."""
        return self._fingerprint

    def _verify_frozen(
        self,
        *,
        state_spaces: tuple[SpaceLike, ...] | None,
        tendency: Callable[..., object] | None,
        halo: HaloSpec | None,
    ) -> ReshardingReport:
        """
        Verify demands against the fingerprint (model D4/D5).

        Description
        -----------
        The frozen-grid path of ``negotiate``: demand satisfaction
        (subset / less-or-equal), never equality. A demanded state
        space missing from the record is adopted when it belongs to
        the ConstantSpace-broadcast family (zero halo demand by
        construction); all violations are collected into one
        diff-style ``GridFrozenError``.

        Parameters
        ----------
        state_spaces : tuple[SpaceLike, ...] | None
            The demanding model's state-field spaces.
        tendency : Callable[..., object] | None
            The tendency whose traced demand is verified.
        halo : HaloSpec | None
            Extra per-name halo demand (merge_max rule).

        Returns
        -------
        ReshardingReport
            The unchanged-layout report (nothing is renegotiated).
        """
        record = self._fingerprint
        if record is None:  # pragma: no cover — freeze always sets
            raise GridFrozenError(
                "the grid is frozen but carries no negotiation "
                "fingerprint")
        demand = self._demanded_halo(state_spaces, tendency, halo)
        problems = _halo_violations(demand, record.halo)
        adopted: list[SpaceLike] = []
        for space in state_spaces or ():
            layout = space.layout
            if layout is not None and layout not in record.layouts:
                problems.append(
                    f"layout of {space!r} is not in the frozen "
                    "layout vocabulary")
            bare = space.bare
            if bare in record.state_spaces or bare in adopted:
                continue
            if _constant_broadcast(bare):
                adopted.append(bare)
                continue
            problems.append(
                f"state space {bare!r} is not in the frozen record "
                "(and is not ConstantSpace-broadcast adoptable)")
        if problems:
            raise GridFrozenError(
                "the frozen grid cannot satisfy the demanded "
                "negotiation (assemble the most demanding model "
                "first):\n  " + "\n  ".join(problems))
        if adopted:
            self._fingerprint = replace(
                record,
                state_spaces=record.state_spaces | set(adopted))
        layout = self._decomposition.default_layout
        return ReshardingReport(old=layout, new=layout,
                                changed=False)

    def _demanded_halo(
        self,
        state_spaces: tuple[SpaceLike, ...] | None,
        tendency: Callable[..., object] | None,
        halo: HaloSpec | None,
    ) -> HaloSpec:
        """
        Resolve the demanded halo under the merge_max rule.

        Description
        -----------
        The verify-side twin of the negotiation's halo resolution:
        the traced `tendency` demand when supplied (else the
        per-operator registry maximum scoped to `state_spaces`),
        merged per-coordinate max with the `halo=` extra spec.

        Parameters
        ----------
        state_spaces : tuple[SpaceLike, ...] | None
            The demanding model's state-field spaces.
        tendency : Callable[..., object] | None
            The tendency to halo-trace.
        halo : HaloSpec | None
            Extra per-name halo demand.

        Returns
        -------
        HaloSpec
            The demanded per-name ghost widths.
        """
        if tendency is not None:
            if state_spaces is None:
                raise ValueError(
                    "tracing a tendency needs state_spaces= to "
                    "build the tracer state")
            demand = HaloSpec.zero(self._names).merge_max(
                trace_halo(tendency, state_spaces, self._dispatch))
        else:
            demand = _registry_halo(self._names, self._dispatch,
                                    state_spaces)
        if halo is not None:
            demand = demand.merge_max(halo)
        return demand

    def sync(
        self,
        field: ScalarField,
        boundary_data: Mapping[str, ScalarField] | None = None,
        *,
        materialize: bool = False,
    ) -> ScalarField:
        """
        Fill the field's halos (wrap / BC fill / shard exchange).

        Parameters
        ----------
        field : ScalarField
            The field whose halos to fill.
        boundary_data : Mapping[str, ScalarField] | None, optional
            Inhomogeneous ghost-fill data (designed-for; must be
            None in iteration 1) (default: None).
        materialize : bool, optional
            The caller's claim that the synced field is consumed at
            a materialization boundary (``Decomposition.sync``)
            (default: False).

        Returns
        -------
        ScalarField
            The synced field (metadata preserved); its halo validity
            is stamped to the negotiated widths (task 1.8).
        """
        if boundary_data is not None:
            raise NotImplementedError(
                "inhomogeneous ghost fill is designed-for; "
                "iteration 1 is homogeneous only")
        space = field.function_space
        synced = self._decomposition.sync(
            field._data, space,  # noqa: SLF001 — storage seam
            materialize=materialize)
        return ScalarField(
            self, space, synced, field.metadata,
            halo_valid=self._decomposition.halo.over(
                tuple(space.names)))

    # ================================================================
    #  Field factory
    # ================================================================
    def create_field(
        self,
        space: SpaceLike | None = None,
        *,
        init: Callable[..., jax.Array] | None = None,
        init_coeff: Callable[..., jax.Array] | None = None,
        data: jax.Array | None = None,
        order: int | None = None,
        name: str | None = None,
        units: str | None = None,
        metadata: FieldMetadata | None = None,
    ) -> ScalarField:
        """
        Create a field on ``space`` (default: all-Center product).

        Description
        -----------
        The single field factory (rules section 3.10). ``init=``
        discretizes a function of physical coordinates
        (keyword-matched to coordinate names, collocation default);
        on coefficient spaces it composes with the forward transform
        (discretize = transform o discretize-on-origin). ``order=``
        selects the per-cell Gauss-Legendre quadrature used on
        average factors (rules section 3.10 table): ``None`` (the
        default) and ``1`` are the midpoint shortcut — the single
        evaluation-node sample, bitwise-identical to plain
        collocation; ``order >= 2`` is the genuine ``order``-point
        rule, exact for per-cell polynomial averages up to degree
        ``2 * order - 1``.
        ``init_coeff=`` assigns coefficients directly, evaluated at
        ``grid.wavenumbers(space)`` and keyword-matched to the
        wavenumber names (``k<coordinate>``). ``data=`` takes a
        true-shape array (validated, dtype-coerced,
        Hermitian-projected on sole real-origin Fourier factors);
        with none of the three the field is zeros. Bare spaces get
        the decomposition's default layout attached; laid-out spaces
        are honored as given.

        Parameters
        ----------
        space : SpaceLike | None, optional
            The function space; None means the all-Center nodal
            product (default: None).
        init : Callable[..., jax.Array] | None, optional
            Function of the physical coordinates, matched by
            coordinate name (default: None).
        init_coeff : Callable[..., jax.Array] | None, optional
            Function of the wavenumbers/mode indices, matched by
            ``k<coordinate>`` name (default: None).
        data : jax.Array | None, optional
            True-shape array companion; ``init`` / ``init_coeff`` /
            ``data`` are pairwise exclusive (default: None).
        order : int | None, optional
            Per-cell Gauss-Legendre quadrature point count for
            ``init=`` on average factors; ``None``/``1`` keep the
            midpoint shortcut, ``>= 2`` the high-order rule (default:
            None).
        name : str | None, optional
            Metadata name sugar (default: None).
        units : str | None, optional
            Metadata units sugar (default: None).
        metadata : FieldMetadata | None, optional
            Full metadata record; mutually exclusive with the sugar
            (default: None).

        Returns
        -------
        ScalarField
            The new field (padded, synced, on the laid-out space).
        """
        if sum(arg is not None
               for arg in (init, init_coeff, data)) > 1:
            raise ValueError(
                "init=, init_coeff= and data= are pairwise "
                "exclusive")
        _check_order(order)
        if metadata is not None and (name is not None
                                     or units is not None):
            raise ValueError(
                "metadata= is mutually exclusive with the "
                "name=/units= sugar")
        if metadata is None and (name is not None
                                 or units is not None):
            metadata = FieldMetadata.create(
                name=name if name is not None else "unnamed",
                units=units if units is not None else "n/a")
        space = (self._default_space() if space is None
                 else self._laid_out(space))
        if init is not None and any(
                isinstance(factor, CoefficientSpace)
                for factor in space.factors):
            return self._transform_discretize(
                space, init, order, metadata)
        dtype = storage_dtype(space)
        if data is not None:
            arr = jnp.asarray(data)
            if tuple(arr.shape) != tuple(space.shape):
                raise ValueError(
                    f"data= expects the true shape {space.shape}, "
                    f"got {tuple(arr.shape)}")
            if (jnp.iscomplexobj(arr)
                    and not jnp.issubdtype(dtype,
                                           jnp.complexfloating)):
                raise ValueError(
                    "complex data cannot be demoted to the real "
                    f"storage of {space!r}")
            arr = hermitian_project(arr.astype(dtype), space)
        elif init is not None:
            arr = self._discretize(space, init, order).astype(dtype)
        elif init_coeff is not None:
            arr = hermitian_project(
                self._assign_coeff(space, init_coeff).astype(dtype),
                space)
        else:
            arr = jnp.zeros(space.shape, dtype)
        stored = store(self._decomposition, space, arr)
        return ScalarField(self, space, stored, metadata)

    @property
    def random(self) -> RandomFieldFactory:
        """Seeded, sharding-consistent random field generators."""
        return self._random

    # ================================================================
    #  Coordinate accessors
    # ================================================================
    def evaluation_nodes(
        self,
        space: SpaceLike,
        name: str | None = None,
    ) -> ScalarField:
        """
        Physical coordinates of the space's evaluation nodes.

        Description
        -----------
        Returns a field tagged with the querying space, with every
        factor not carrying ``name`` replaced by its
        ``ConstantSpace`` — so it broadcasts exactly under the
        strict algebra (sections 2.7, 3.3). ``name`` may be omitted
        exactly when the space contributes one non-constant name.

        Parameters
        ----------
        space : SpaceLike
            The querying space (mandatory; no default form).
        name : str | None, optional
            The coordinate to materialize; may be omitted when
            unambiguous (default: None).

        Returns
        -------
        ScalarField
            The per-factor node coordinates as a field.
        """
        space = self._laid_out(space)
        name = _pick_factor_name(space, name)
        factor = space.factor(name)
        if isinstance(factor, ConstantSpace):
            # a value error (bad name choice), not a type error
            raise ValueError(  # noqa: TRY004
                f"factor {factor!r} is constant along {name!r}; "
                "constant factors have no evaluation nodes")
        if isinstance(factor, CoefficientSpace):
            raise ValueError(  # noqa: TRY004 — value, not type
                f"coefficient factor {factor!r} has no physical "
                "coordinates; use grid.wavenumbers(space)")
        nodes = _node_vector(factor)
        bare_factors = tuple(
            f if f is factor else f.mesh.constant
            for f in space.factors)
        result: SpaceLike = (
            bare_factors[0] if len(bare_factors) == 1
            else TensorProductSpace.of(*bare_factors))
        # `space` went through _laid_out, so its layout is never None
        result = result.with_layout(space.layout)
        data = nodes.reshape(result.shape)
        stored = store(self._decomposition, result, data)
        return ScalarField(self, result, stored,
                           FieldMetadata.create(name=name))

    def wavenumbers(
        self,
        space: SpaceLike,
        name: str | None = None,
    ) -> ScalarField:
        """
        Wavenumbers (or mode indices) of a coefficient space.

        Description
        -----------
        The coefficient-side coordinate accessor (rules section
        3.10): physical wavenumbers for the wavenumber-indexed bases
        (Fourier ``2 pi m / L`` in storage layout, sine/cosine
        ``pi k / L`` in mode order) and the intrinsic mode indices
        for bases that are not (Chebyshev). Delegates to the
        ``operators.spectral`` helpers. The result is tagged with
        the querying space, all other factors replaced by their
        ``ConstantSpace`` (section 3.3), named ``k<name>``.

        Parameters
        ----------
        space : SpaceLike
            The querying space (mandatory; no default form).
        name : str | None, optional
            The coordinate whose wavenumbers to materialize; may be
            omitted when unambiguous (default: None).

        Returns
        -------
        ScalarField
            The per-factor wavenumbers as a field.
        """
        space = self._laid_out(space)
        name = _pick_factor_name(space, name)
        factor = space.factor(name)
        if isinstance(factor, ConstantSpace):
            # a value error (bad name choice), not a type error
            raise ValueError(  # noqa: TRY004
                f"factor {factor!r} is constant along {name!r}; "
                "constant factors carry no wavenumbers")
        if not isinstance(factor, CoefficientSpace):
            raise ValueError(  # noqa: TRY004 — value, not type
                f"factor {factor!r} carries no wavenumbers; "
                "physical coordinates live on "
                "grid.evaluation_nodes(space)")
        values = _wavenumber_vector(factor)
        bare_factors = tuple(
            f if f is factor else f.mesh.constant
            for f in space.factors)
        result: SpaceLike = (
            bare_factors[0] if len(bare_factors) == 1
            else TensorProductSpace.of(*bare_factors))
        # `space` went through _laid_out, so its layout is never None
        result = result.with_layout(space.layout)
        data = values.reshape(result.shape).astype(
            storage_dtype(result))
        stored = store(self._decomposition, result, data)
        return ScalarField(self, result, stored,
                           FieldMetadata.create(name=f"k{name}"))

    def measure(
        self,
        space: SpaceLike,
        name: str | None = None,
    ) -> ScalarField:
        """
        Materialize the metric measure proper to the node set.

        Description
        -----------
        The staggered ``dx`` field of the factor carrying ``name``:
        the primal cell width on ``Center``/``CellAvg`` (the FV
        integration weight), the dual cell width on the face-family
        node sets (``Right``/``Outer``/``Inner``/``FaceAvg``, the
        ``diff`` denominator), clipped to the domain on bounded
        meshes (a boundary-member node carries the half cell
        ``dx / 2``) — one accessor, the space decides which measure
        it is (rules sections 2.7, 3.9). Structured 1D geometry:
        uniform meshes yield the constant special case XLA folds,
        mapped meshes the staggered differences of the mapped node
        positions. The result is tagged with the querying space,
        all other factors replaced by their ``ConstantSpace``, so
        it broadcasts exactly (section 3.3). Measures are static
        mesh geometry (no ``params=`` seam), so repeated queries
        return one memoized field per (space, name) — the same
        object, which also keeps the operator-level sync memo warm.
        The memo holds **concrete** fields only: a query issued
        under a jax trace (an integral kernel inside the eager-
        operator jit, a traced model stage) returns an uncached
        field, since caching that trace's tracer would leak it into
        every later query (rules 3.8: caches hold static geometry,
        never traced values); the traced query re-derives the
        weights, an XLA constant either way.

        Parameters
        ----------
        space : SpaceLike
            The querying space (mandatory; no default form).
        name : str | None, optional
            The coordinate whose measure to materialize; may be
            omitted when unambiguous (default: None).

        Returns
        -------
        ScalarField
            The per-factor measure weights as a field.
        """
        space = self._laid_out(space)
        name = _pick_factor_name(space, name)
        cached = self._measures.get((space, name))
        if cached is not None:
            return cached
        factor = space.factor(name)
        if isinstance(factor, ConstantSpace):
            # a value error (bad name choice), not a type error
            raise ValueError(  # noqa: TRY004
                f"factor {factor!r} is constant along {name!r}; "
                "constant factors carry no measure")
        if isinstance(factor, CoefficientSpace):
            raise ValueError(  # noqa: TRY004 — value, not type
                f"coefficient factor {factor!r} carries no metric "
                "measure; measures live on nodal/average factors")
        weights = _measure_vector(factor)
        bare_factors = tuple(
            f if f is factor else f.mesh.constant
            for f in space.factors)
        result: SpaceLike = (
            bare_factors[0] if len(bare_factors) == 1
            else TensorProductSpace.of(*bare_factors))
        # `space` went through _laid_out, so its layout is never None
        result = result.with_layout(space.layout)
        data = weights.reshape(result.shape)
        stored = store(self._decomposition, result, data)
        field = ScalarField(self, result, stored,
                            FieldMetadata.create(name=f"d{name}"))
        if not isinstance(stored, jax.core.Tracer):
            self._measures[(space, name)] = field
        return field

    def metric(
        self,
        space: SpaceLike,
        name: str,
        *,
        params: Mapping[str, ScalarField] | None = None,
    ) -> ScalarField:
        """
        Derive a named mapping metric on the requested space.

        Description
        -----------
        Delegates to the attached ``CoordinateMapping`` (argument
        order aligned with ``grid.measure``): the metric is derived
        per staggered space on demand — one owner, so staggered
        consistency (H at u-, v-, w-points) is guaranteed by the
        grid, never per module (rules section 3.8). The result is
        tagged with the querying space, factors the metric does not
        involve replaced by their ``ConstantSpace``, so it
        broadcasts exactly (section 3.3). With ``params=`` given,
        the supplied dynamic fields (module-owned state) **replace**
        the mapping's static defaults; metrics are recomputed from
        the passed values at every query and traced like any field
        arithmetic — no caching anywhere (sections 2.3, 3.8).

        Parameters
        ----------
        space : SpaceLike
            The querying space (mandatory; no default form).
        name : str
            The metric name (one of ``mapping.metric_names``).
        params : Mapping[str, ScalarField] | None, optional
            Caller-supplied parameter fields overriding the static
            defaults (default: None).

        Returns
        -------
        ScalarField
            The metric field, tagged with the querying space.
        """
        if self._mapping is None:
            raise ValueError(
                "this grid has no coordinate mapping; attach one "
                "via Grid(..., mapping=...)")
        return self._mapping.metric(space, name, params=params)

    # ================================================================
    #  Attachments
    # ================================================================
    @property
    def immersed(self) -> ImmersedDomain | None:
        """The immersed (masked) domain descriptor, or None."""
        return self._immersed

    @property
    def mapping(self) -> CoordinateMapping | None:
        """The coordinate-mapping descriptor, or None."""
        return self._mapping

    @property
    def chart_coords(self) -> tuple[str, ...] | None:
        """
        The chart-coupled coordinate family of this grid, or None.

        Description
        -----------
        The public chart-ness signal for model modules (coordinate-
        systems plan, stage C2): the attached mapping's embedding-
        chart base coordinates whenever the chart couples at least
        two coordinates — exactly the condition under which this
        grid seeded the metric-aware ``"grad"`` / ``"div"`` /
        ``"curl"`` / ``"laplacian"`` and ``"raise_index"`` /
        ``"lower_index"`` dispatch kinds. Tendency modules branch on
        it (a static grid property, never a traced value) to select
        their metric-aware path; ``None`` means the flat kinds are
        registered and the Cartesian expressions apply.

        Returns
        -------
        tuple[str, ...] | None
            The chart's base coordinates in signature order, or
            None on chartless (or single-coordinate-chart) grids.
        """
        if self._mapping is None:
            return None
        chart = self._mapping.chart_coords
        if chart is None or len(chart) < 2:  # noqa: PLR2004 — pairs
            return None
        return chart

    def with_immersed(self, immersed: ImmersedDomain) -> Grid:
        """
        Attach the immersed descriptor (pre-freeze only).

        Description
        -----------
        Binds the static descriptor to this grid and returns the
        grid, so already-created fields keep their grid identity
        (the descriptor holds no arrays — there is nothing to
        reshard or invalidate). After ``freeze()`` this raises
        ``GridFrozenError`` (grid lifecycle step 3).

        Parameters
        ----------
        immersed : ImmersedDomain
            The wet-region descriptor to attach.

        Returns
        -------
        Grid
            This grid, carrying the descriptor.
        """
        if self._frozen:
            raise GridFrozenError(
                "the grid is frozen; with_immersed is legal in the "
                "assembly phase only (grid lifecycle)")
        self._attach_immersed(immersed)
        return self

    def _attach_immersed(self, immersed: ImmersedDomain) -> None:
        """Bind and store the immersed descriptor."""
        immersed._bind(self)  # noqa: SLF001 — attachment seam
        self._immersed = immersed

    # ================================================================
    #  Internal helpers
    # ================================================================
    def _default_space(self) -> SpaceLike:
        """Return the all-Center nodal product, laid out."""
        centers = []
        for mesh in self._meshes:
            center = getattr(mesh, "center", None)
            if center is None:
                raise TypeError(
                    f"{mesh!r} has no Center space; pass an "
                    "explicit space to create_field")
            centers.append(center)
        return self._laid_out(TensorProductSpace.of(*centers))

    def _owns_mesh(self, mesh: object) -> bool:
        """Whether a mesh is a grid factor or a refined descendant."""
        current = mesh
        while current is not None:
            if any(current is factor for factor in self._meshes):
                return True
            current = getattr(current, "refined_from", None)
        return False

    def _laid_out(self, space: SpaceLike) -> SpaceLike:
        """Validate factor ownership; attach the default layout."""
        for factor in space.factors:
            if not self._owns_mesh(factor.mesh):
                raise GridMismatchError(
                    f"space factor {factor!r} lives on a mesh that "
                    "is not a factor (or adopted refinement) of "
                    "this grid",
                    left=self, operation="create_field")
        if space.layout is None:
            return space.with_layout(
                self._decomposition.default_layout)
        return space

    def _discretize(
        self,
        space: SpaceLike,
        init: Callable[..., jax.Array],
        order: int | None = None,
    ) -> jax.Array:
        """
        Default of the ``("discretize", space)`` row (rules 3.10).

        Description
        -----------
        Inline iteration-1 default (the registry row proper lands
        with the operators merge): keyword-match ``init`` against
        the non-constant coordinate names, broadcast the per-factor
        node coordinates transiently, and sample. ``order >= 2`` on a
        space that carries an average factor switches to per-cell
        Gauss-Legendre quadrature (``_quadrature_discretize``); the
        midpoint default (``order`` ``None``/``1``, or any pure-nodal
        space) is the plain node-set collocation of this method.
        """
        _check_init_names(space, init)
        if (order is not None and order >= _QUADRATURE_MIN
                and any(isinstance(factor, AverageSpace)
                        for factor in space.factors)):
            return self._quadrature_discretize(space, init, order)
        coords: dict[str, jax.Array] = {}
        ndim = len(space.shape)
        for factor, axis in factor_axes(space):
            if isinstance(factor, ConstantSpace):
                continue
            shape = [1] * ndim
            shape[axis] = factor.shape[0]
            coords[factor.names[0]] = _node_vector(factor).reshape(
                shape)
        values = jnp.asarray(init(**coords))
        return jnp.broadcast_to(values, space.shape)

    def _quadrature_discretize(
        self,
        space: SpaceLike,
        init: Callable[..., jax.Array],
        order: int,
    ) -> jax.Array:
        r"""
        Per-cell Gauss-Legendre quadrature on average factors.

        Description
        -----------
        The genuine per-cell quadrature of the average-family
        ``("discretize", space)`` row (rules section 3.10 table);
        ``order`` is the point count of the tensor-product
        Gauss-Legendre rule, exact for per-cell polynomial averages
        up to degree ``2 * order - 1``. Each non-constant factor
        contributes its own reference rule: an ``AverageSpace``
        factor an ``order``-point rule placed inside every cell by
        the cell's own physical edges (so it scales with each cell's
        width on a stretched axis), a nodal factor the single
        evaluation-node sample (weight 1) — the mixed
        ``Right(x) ⊗ CellAvg(y)`` field therefore stays a point
        value along ``x`` and becomes a quadrature average along
        ``y``. The within-cell reference nodes/weights are static
        host data (there are more of them than DOFs, concepts
        section 2.2); only the per-cell edge positions are traced
        (lazy) mesh arrays.

        The cell edges come from each axis's own 1D geometry
        (``mesh.coordinate_map``): on a chart-mapped grid this is the
        **chart** cell average, not the Jacobian-weighted physical-
        volume average (designed-for, stage F5). ``FaceAvg`` factors
        are guarded (their dual cells need the center-to-center edge
        seam, with periodic wrap and bounded half-width wall cells);
        their midpoint default still works.

        Parameters
        ----------
        space : SpaceLike
            The laid-out target space (>= 1 average factor).
        init : Callable[..., jax.Array]
            Function of the physical coordinates (name-matched).
        order : int
            The Gauss-Legendre point count per cell (``>= 2``).

        Returns
        -------
        jax.Array
            The true-shape per-cell averages.
        """
        ref_nodes, ref_weights = _gauss_legendre_unit(order)
        ndim = len(space.shape)
        movers = [(factor, axis)
                  for factor, axis in factor_axes(space)
                  if not isinstance(factor, ConstantSpace)]
        n_quad_axes = len(movers)
        coords: dict[str, jax.Array] = {}
        weight = jnp.ones((), dtype=dtype_real())
        for q, (factor, axis) in enumerate(movers):
            if isinstance(factor, AverageSpace):
                nodes, weights = _cell_quadrature(
                    factor, ref_nodes, ref_weights)
            else:
                nodes = _node_vector(factor)[:, None]
                weights = jnp.ones(1, dtype=dtype_real())
            node_shape = [1] * (ndim + n_quad_axes)
            node_shape[axis] = nodes.shape[0]
            node_shape[ndim + q] = nodes.shape[1]
            coords[factor.names[0]] = nodes.reshape(node_shape)
            weight_shape = [1] * (ndim + n_quad_axes)
            weight_shape[ndim + q] = weights.shape[0]
            weight = weight * weights.reshape(weight_shape)
        values = jnp.asarray(init(**coords)) * weight
        reduced = jnp.sum(
            values, axis=tuple(range(ndim, ndim + n_quad_axes)))
        return jnp.broadcast_to(reduced, space.shape)

    def _transform_discretize(
        self,
        space: SpaceLike,
        init: Callable[..., jax.Array],
        order: int | None,
        metadata: FieldMetadata | None,
    ) -> ScalarField:
        """
        ``init=`` on coefficient spaces (rules section 3.10).

        Description
        -----------
        discretize = transform o discretize-on-origin: the field is
        collocated on the coefficient factors' origin spaces and
        pushed forward through the per-family transforms (explicit
        per-axes construction, operators_composed.md). When the
        requested space carries a Hermitian half-spectrum factor the
        represented function is real, so the collocation runs on the
        real origins (the rfftn schedule then reproduces the
        requested factor mix); a requested mix the planner cannot
        produce raises ``SpaceMismatchError``. ``order`` rides
        through to the origin discretize, so a high-order rule on an
        average origin (``Fourier(x, origin=CellAvg)``) quadratures
        the origin cell averages before the transform.

        Parameters
        ----------
        space : SpaceLike
            The laid-out target space (>= 1 coefficient factor).
        init : Callable[..., jax.Array]
            Function of the physical coordinates.
        order : int | None
            Per-cell quadrature point count forwarded to the origin
            discretize (average origins only).
        metadata : FieldMetadata | None
            Annotation metadata of the result.

        Returns
        -------
        ScalarField
            The coefficient-space field.
        """
        bare = space.bare
        half = any(
            isinstance(factor, FourierSpace)
            and factor.scalars is Scalars.REAL
            for factor in bare.factors)
        groups: dict[type[Transform], list[str]] = {}
        origin_factors: list[FunctionSpace] = []
        for factor in bare.factors:
            if isinstance(factor, CoefficientSpace):
                groups.setdefault(
                    _transform_family(factor), []).extend(
                    factor.names)
                origin = factor.origin
                if half:
                    origin = origin.as_real()
                origin_factors.append(origin)
            else:
                origin_factors.append(factor)
        origin_space: SpaceLike = (
            origin_factors[0] if len(origin_factors) == 1
            else TensorProductSpace.of(*origin_factors))
        # `space` went through _laid_out, so its layout is never None
        origin_space = origin_space.with_layout(space.layout)
        field = self.create_field(origin_space, init=init,
                                  order=order, metadata=metadata)
        for family, axes in groups.items():
            field = family(self, axes=tuple(axes)).forward(field)
        if field.function_space.bare is not bare:
            raise SpaceMismatchError(
                "init= on a coefficient space composes with the "
                "forward transform, which lands on "
                f"{field.function_space.bare!r}, not the requested "
                f"{bare!r} (multi-axis real transforms put the half "
                "spectrum on the first-listed axis)",
                left=field.function_space.bare, right=bare,
                operation="create_field")
        return field

    def _assign_coeff(
        self,
        space: SpaceLike,
        init_coeff: Callable[..., jax.Array],
    ) -> jax.Array:
        """
        Inline default of the ``("assign_coeff", space)`` row.

        Description
        -----------
        Assignment of coefficients, not projection (rules section
        3.10): keyword-match ``init_coeff`` against the wavenumber
        names ``k<coordinate>``, broadcast the per-factor
        wavenumber/mode vectors transiently, and evaluate.

        Parameters
        ----------
        space : SpaceLike
            The (laid-out) coefficient target space.
        init_coeff : Callable[..., jax.Array]
            Function of the wavenumbers/mode indices.

        Returns
        -------
        jax.Array
            The true-shape coefficient array.
        """
        required: list[str] = []
        for factor in space.factors:
            if isinstance(factor, ConstantSpace):
                continue
            if not isinstance(factor, CoefficientSpace):
                raise TypeError(
                    "init_coeff= assigns coefficients: every "
                    "non-constant factor must be a coefficient "
                    f"space, got {factor!r} (mixed spaces take "
                    "data=)")
            required.append(f"k{factor.names[0]}")
        params = tuple(inspect.signature(init_coeff).parameters)
        if set(params) != set(required):
            raise TypeError(
                "init_coeff= callables must name exactly the "
                f"wavenumber names {tuple(required)}, got {params}")
        coords: dict[str, jax.Array] = {}
        ndim = len(space.shape)
        for factor, axis in factor_axes(space):
            if isinstance(factor, ConstantSpace):
                continue
            shape = [1] * ndim
            shape[axis] = factor.shape[0]
            coords[f"k{factor.names[0]}"] = _wavenumber_vector(
                factor).reshape(shape)
        values = jnp.asarray(init_coeff(**coords))
        return jnp.broadcast_to(values, space.shape)


def _flatten_overrides(
    overrides: Mapping[DispatchKey, Operator]
        | Mapping[str, Mapping[DispatchKey, Operator]],
) -> Mapping[DispatchKey, Operator]:
    """
    Flatten per-module overrides; detect resolved-key collisions.

    Description
    -----------
    ``grid.merge_overrides`` accepts one flat override mapping or
    the per-module form ``{module_name: {key: operator}}`` (values
    are mappings exactly when the form is per-module: operators are
    never mappings). Flattening normalizes each key through the
    merge-call-site guard and raises ``DispatchCollisionError``
    naming both modules on the first resolved key contributed twice.

    Parameters
    ----------
    overrides : Mapping
        Module override entries, flat or keyed by module name.

    Returns
    -------
    Mapping[DispatchKey, Operator]
        The flat (normalized-key) override mapping.
    """
    if not overrides or not all(
            isinstance(entries, Mapping)
            for entries in overrides.values()):
        return overrides
    flat: dict[DispatchKey, Operator] = {}
    owners: dict[DispatchKey, str] = {}
    for module, entries in overrides.items():
        for key, op in entries.items():
            nkey = check_override_key(key)
            other = owners.get(nkey)
            if other is not None:
                raise DispatchCollisionError(
                    f"modules {other!r} and {module!r} both "
                    f"override the resolved dispatch key {nkey!r}; "
                    "module order never silently selects an "
                    "operator")
            owners[nkey] = module
            flat[nkey] = op
    return flat


def _halo_violations(demand: HaloSpec, frozen: HaloSpec) -> list[str]:
    """
    Diff a demanded halo against the frozen record (verify path).

    Description
    -----------
    The less-or-equal check of the frozen-grid verification: a name
    missing from the frozen spec counts as width 0.

    Parameters
    ----------
    demand : HaloSpec
        The demanded per-name ghost widths (merge_max rule).
    frozen : HaloSpec
        The fingerprint's negotiated ghost widths.

    Returns
    -------
    list[str]
        One diff line per name whose demand exceeds the record.
    """
    problems: list[str] = []
    for name, width in demand.widths:
        try:
            frozen_width = frozen[name]
        except KeyError:
            frozen_width = 0
        if width > frozen_width:
            problems.append(
                f"halo[{name!r}]: demanded {width} > frozen "
                f"{frozen_width}")
    return problems


def _constant_broadcast(space: SpaceLike) -> bool:
    """
    Whether a space is in the ConstantSpace-broadcast family.

    Description
    -----------
    The satisfiability relaxation of the frozen-grid verify path
    (grid.md, model validation sign-off): a state space constant
    along at least one coordinate broadcasts over the recorded
    negotiation and carries zero halo demand on its constant
    factors, so it is adopted into the record instead of refused.

    Parameters
    ----------
    space : SpaceLike
        The bare demanded state space.

    Returns
    -------
    bool
        True if at least one factor is a ``ConstantSpace``.
    """
    return any(isinstance(factor, ConstantSpace)
               for factor in space.factors)


def _pick_factor_name(space: SpaceLike, name: str | None) -> str:
    """
    Resolve the coordinate name of a per-factor accessor query.

    Description
    -----------
    Shared by ``evaluation_nodes`` / ``wavenumbers`` / ``measure``:
    ``name`` may be omitted exactly when the space contributes one
    non-``ConstantSpace`` coordinate.

    Parameters
    ----------
    space : SpaceLike
        The (laid-out) querying space.
    name : str | None
        The requested coordinate, or None for the unambiguous case.

    Returns
    -------
    str
        The resolved coordinate name.
    """
    if name is not None:
        return name
    candidates = tuple(
        coord for factor in space.factors
        if not isinstance(factor, ConstantSpace)
        for coord in factor.names)
    if len(candidates) != 1:
        raise ValueError(
            "the coordinate is ambiguous on this space; "
            f"pass name= (one of {candidates})")
    return candidates[0]


def _wavenumber_vector(factor: CoefficientSpace) -> jax.Array:
    """
    Materialize the 1D wavenumber/mode vector of one factor.

    Parameters
    ----------
    factor : CoefficientSpace
        The bare coefficient factor space.

    Returns
    -------
    jax.Array
        The wavenumbers (or mode indices), matching
        ``factor.shape``.
    """
    if isinstance(factor, FourierSpace):
        return fourier_wavenumbers(factor)
    if isinstance(factor, SineSpace | CosineSpace):
        return trig_wavenumbers(factor)
    if isinstance(factor, ChebyshevSpace):
        return chebyshev_modes(factor)
    raise NotImplementedError(
        f"wavenumbers of {factor!r} are not defined in iteration 1")


# ================================================================
#  Node coordinates of the structured 1D spaces
# ================================================================
# offsets of the first node from s = 0, in units of the
# computational cell width 1 / n, and the DOF count offset relative
# to the cell count n
_NODE_OFFSET: dict[NodeSet, tuple[float, int]] = {
    NodeSet.CENTER: (0.5, 0),
    NodeSet.LEFT: (0.0, 0),
    NodeSet.RIGHT: (1.0, 0),
    NodeSet.OUTER: (0.0, 1),
    NodeSet.INNER: (1.0, -1),
}

# whether the (left, right) boundary DOF belongs to the node set on a
# bounded mesh (mirrors the shape rule of the nodal spaces): a
# BC-constrained boundary DOF is dropped only when it is in the set
_BOUNDARY_MEMBERSHIP: dict[NodeSet, tuple[bool, bool]] = {
    NodeSet.CENTER: (False, False),
    NodeSet.LEFT: (True, False),
    NodeSet.RIGHT: (False, True),
    NodeSet.OUTER: (True, True),
    NodeSet.INNER: (False, False),
}


def _node_vector(factor: FunctionSpace) -> jax.Array:
    """
    Materialize the 1D node coordinates of one factor space.

    Description
    -----------
    Structured 1D geometry: nodal spaces at their node sets,
    ``CellAvg`` at the midpoint-quadrature points (cell centers),
    ``FaceAvg`` at the dual-cell midpoints (faces). The
    computational placement ``s = (i + offset) / n`` composes with
    the mesh's ``coordinate_map`` seam (concepts section 2.7);
    ``None`` is the uniform affine placement from ``extent``/``dx``.
    BC-constrained boundary DOFs are dropped exactly like the space
    shapes drop them.

    Parameters
    ----------
    factor : FunctionSpace
        A non-constant, non-coefficient factor space.

    Returns
    -------
    jax.Array
        The node coordinates, matching ``factor.shape``.
    """
    mesh = factor.mesh
    if not isinstance(mesh, StructuredMesh1D):
        raise NotImplementedError(
            f"evaluation nodes on {type(mesh).__name__} are not "
            "defined: node placement needs the structured 1D "
            "geometry seam (coordinate_map)")
    n = mesh.n_cells
    if isinstance(factor, NodalSpace):
        offset, count_offset = _NODE_OFFSET[factor.node_set]
        membership = _BOUNDARY_MEMBERSHIP[factor.node_set]
    elif isinstance(factor, CellAvg):
        offset, count_offset = _NODE_OFFSET[NodeSet.CENTER]
        membership = (False, False)
    elif isinstance(factor, FaceAvg):
        node_set = NodeSet.RIGHT if mesh.periodic else NodeSet.INNER
        offset, count_offset = _NODE_OFFSET[node_set]
        membership = (False, False)
    else:
        raise NotImplementedError(
            f"evaluation nodes of {factor!r} are not defined in "
            "iteration 1")
    count = n + count_offset
    steps = jnp.arange(count, dtype=dtype_real()) + offset
    mapping = mesh.coordinate_map
    if mapping is None:
        nodes = mesh.extent[0] + steps * mesh.dx
    else:
        nodes = jnp.asarray(mapping(steps / n)).astype(dtype_real())
    # drop Dirichlet-constrained boundary DOFs (left, then right),
    # exactly like the space shapes drop them
    start, stop = 0, count
    components = factor.bc.components
    if components:
        left, right = components
        # only Dirichlet eliminates a boundary value DOF (owner
        # decision 2026-07-07, spaces.md shape note)
        if membership[0] and left is BC.DIRICHLET:
            start += 1
        if membership[1] and right is BC.DIRICHLET:
            stop -= 1
    return nodes[start:stop]


# ================================================================
#  Per-cell quadrature (the average-family discretize, rules 3.10)
# ================================================================
# the smallest point count that leaves the midpoint shortcut: one
# point IS the midpoint rule, realized as the node-set sample so the
# default stays bitwise-identical to plain collocation
_QUADRATURE_MIN = 2


def _check_order(order: int | None) -> None:
    """
    Verify ``order=`` is None or a positive integer.

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
            f"positive integer or None, got {order!r}")


def _check_init_names(
    space: SpaceLike,
    init: Callable[..., jax.Array],
) -> None:
    """
    Verify ``init`` names exactly the non-constant coordinates.

    Parameters
    ----------
    space : SpaceLike
        The (laid-out) target space.
    init : Callable[..., jax.Array]
        The physical-coordinate callable.
    """
    required: list[str] = []
    for factor in space.factors:
        if isinstance(factor, ConstantSpace):
            continue
        required.extend(factor.names)
    params = tuple(inspect.signature(init).parameters)
    if set(params) != set(required):
        raise TypeError(
            "init= callables must name exactly the "
            f"non-constant coordinate names {tuple(required)}, "
            f"got {params}")


def _gauss_legendre_unit(order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the ``order``-point Gauss-Legendre unit-interval rule.

    Description
    -----------
    Static host reference data (concepts section 2.2): the
    ``[-1, 1]`` Legendre nodes/weights mapped to ``[0, 1]`` and
    renormalized so the weights sum to 1 — the rule is an **average**
    over the cell, not an integral. Exact for polynomials up to
    degree ``2 * order - 1``.

    Parameters
    ----------
    order : int
        The point count per cell.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The unit-interval nodes and (unit-sum) weights.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _cell_quadrature(
    factor: AverageSpace,
    ref_nodes: np.ndarray,
    ref_weights: jax.Array | np.ndarray,
) -> tuple[jax.Array, jax.Array]:
    """
    Place the reference rule inside each cell of an average factor.

    Description
    -----------
    The per-cell node positions and weights of the average-family
    per-cell quadrature. The cell edges come from the factor's own
    1D geometry (uniform ``extent``/``dx`` or the mesh
    ``coordinate_map``, mirroring ``_node_vector``); every reference
    node ``xi`` lands at ``edge_left + xi * width`` with the cell's
    **own** physical width, so a stretched axis quadratures each cell
    on its own scale. ``FaceAvg`` is guarded: its dual cells span
    center-to-center (periodic wrap, bounded half-width wall cells) —
    a separate edge seam that the midpoint default does not need.

    Parameters
    ----------
    factor : AverageSpace
        A ``CellAvg`` factor of a structured 1D mesh.
    ref_nodes : np.ndarray
        The unit-interval reference nodes (shape ``(order,)``).
    ref_weights : jax.Array | np.ndarray
        The matching unit-sum reference weights.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        The per-cell node positions (shape ``(n_cells, order)``) and
        the (unit-sum) weights (shape ``(order,)``).
    """
    mesh = factor.mesh
    if not isinstance(mesh, StructuredMesh1D):
        raise NotImplementedError(
            f"per-cell quadrature on {type(mesh).__name__} is not "
            "defined: the cell edges need the structured 1D "
            "geometry seam (coordinate_map)")
    if not isinstance(factor, CellAvg):
        raise NotImplementedError(
            f"per-cell quadrature on {factor!r} is designed-for, "
            "not built: FaceAvg dual-cell edges (center-to-center, "
            "with periodic wrap and bounded half-width wall cells) "
            "need their own edge seam; use order=None/1 (the "
            "midpoint default) on FaceAvg")
    n = mesh.n_cells
    steps = jnp.arange(n + 1, dtype=dtype_real())
    mapping = mesh.coordinate_map
    edges = (mesh.extent[0] + steps * mesh.dx if mapping is None
             else jnp.asarray(mapping(steps / n)).astype(dtype_real()))
    left = edges[:-1]
    width = edges[1:] - edges[:-1]
    nodes = (left[:, None]
             + jnp.asarray(ref_nodes)[None, :] * width[:, None])
    return nodes, jnp.asarray(ref_weights)


def _measure_vector(factor: FunctionSpace) -> jax.Array:
    """
    Materialize the 1D measure weights of one factor space.

    Description
    -----------
    Structured 1D geometry (rules sections 2.7, 3.9). Uniform
    meshes (``coordinate_map is None``): the constant cell width
    ``dx`` everywhere, except that on bounded meshes a node sitting
    *on* the boundary (a boundary-member DOF of its node set) owns
    the clipped half dual cell ``dx / 2`` — which makes the
    ``Outer`` weights exactly the trapezoid rule. Mapped meshes:
    the staggered differences of the mapped node positions
    (``_mapped_measure_vector``). BC-constrained boundary DOFs are
    dropped exactly like the space shapes drop them.

    Parameters
    ----------
    factor : FunctionSpace
        A non-constant, non-coefficient factor space.

    Returns
    -------
    jax.Array
        The measure weights, matching ``factor.shape``.
    """
    mesh = factor.mesh
    if isinstance(mesh, ChebyshevMesh):
        raise NotImplementedError(
            "measure fields on ChebyshevMesh await the "
            "Clenshaw-Curtis quadrature weights (rules section "
            "3.13); the staggered cell measures are not that "
            "quadrature")
    if not isinstance(mesh, StructuredMesh1D):
        raise NotImplementedError(
            f"measure fields on {type(mesh).__name__} are not "
            "defined: the staggered cell measures need the "
            "structured 1D geometry seam (coordinate_map)")
    count_offset, membership = _measure_geometry(factor)
    count = mesh.n_cells + count_offset
    mapping = mesh.coordinate_map
    if mapping is None:
        dx = mesh.dx
        weights = jnp.full(count, dx, dtype=dtype_real())
        if not mesh.periodic:
            if membership[0]:
                weights = weights.at[0].set(dx / 2)
            if membership[1]:
                weights = weights.at[-1].set(dx / 2)
    else:
        weights = _mapped_measure_vector(factor, mapping,
                                         membership)
    start, stop = 0, count
    components = factor.bc.components
    if components:
        left, right = components
        # only Dirichlet eliminates a boundary value DOF (owner
        # decision 2026-07-07, spaces.md shape note)
        if membership[0] and left is BC.DIRICHLET:
            start += 1
        if membership[1] and right is BC.DIRICHLET:
            stop -= 1
    return weights[start:stop]


def _measure_geometry(
    factor: FunctionSpace,
) -> tuple[int, tuple[bool, bool]]:
    """
    Resolve the DOF-count offset and boundary membership.

    Parameters
    ----------
    factor : FunctionSpace
        A nodal or average factor space of a structured 1D mesh.

    Returns
    -------
    tuple[int, tuple[bool, bool]]
        The count offset from the cell count and whether the
        (left, right) end DOF sits on the boundary.
    """
    if isinstance(factor, NodalSpace):
        return (_NODE_OFFSET[factor.node_set][1],
                _BOUNDARY_MEMBERSHIP[factor.node_set])
    if isinstance(factor, CellAvg):
        return 0, (False, False)
    if isinstance(factor, FaceAvg):
        return (0 if factor.mesh.periodic else -1), (False, False)
    raise NotImplementedError(
        f"the measure of {factor!r} is not defined in iteration 1")


def _mapped_measure_vector(
    factor: FunctionSpace,
    mapping: Callable[[jax.Array], jax.Array],
    membership: tuple[bool, bool],
) -> jax.Array:
    """
    Staggered measures of a mapped structured 1D factor.

    Description
    -----------
    The rules-section-3.9 staggered differences of the mapped node
    positions: the primal cell width (face-to-face differences) on
    ``Center``/``CellAvg``; the dual center-to-center spacing on
    the face-family node sets — wrapping across the seam on
    periodic meshes (the seam entry adds the domain length) and
    clipped to the wall at boundary-member nodes on bounded ones
    (the stretched trapezoid weights). Pre-BC-drop: the caller
    slices constrained boundary DOFs off.

    Parameters
    ----------
    factor : FunctionSpace
        A nodal or average factor space of the mapped mesh.
    mapping : Callable
        The mesh's coordinate map (``s`` in [0, 1] -> physical).
    membership : tuple[bool, bool]
        Whether the (left, right) end DOF sits on the boundary.

    Returns
    -------
    jax.Array
        The measure weights at the pre-drop DOF count.
    """
    mesh = factor.mesh
    n = mesh.n_cells
    x_min, x_max = mesh.extent
    primal = (isinstance(factor, CellAvg)
              or (isinstance(factor, NodalSpace)
                  and factor.node_set is NodeSet.CENTER))
    if primal:
        faces = jnp.asarray(mapping(
            jnp.arange(n + 1, dtype=dtype_real()) / n))
        return jnp.diff(faces).astype(dtype_real())
    centers = jnp.asarray(mapping(
        (jnp.arange(n, dtype=dtype_real()) + 0.5) / n))
    interior = jnp.diff(centers)
    if mesh.periodic:
        wrap = (centers[0] + (x_max - x_min) - centers[-1])[None]
        if (isinstance(factor, NodalSpace)
                and factor.node_set is NodeSet.LEFT):
            parts = (wrap, interior)  # the seam face sits first
        else:  # Right / FaceAvg hold faces 1..n: seam face last
            parts = (interior, wrap)
        return jnp.concatenate(parts).astype(dtype_real())
    parts = []
    if membership[0]:  # wall node owns the clipped half dual cell
        parts.append((centers[0] - x_min)[None])
    parts.append(interior)
    if membership[1]:
        parts.append((x_max - centers[-1])[None])
    return jnp.concatenate(parts).astype(dtype_real())


# ================================================================
#  Default registry seeding (grid lifecycle step 1; the provisional
#  halo of step 2 lives in decomposition.negotiate)
# ================================================================
# mesh factory attributes of the seeded space families
_NODAL_FACTORIES = ("center", "left", "right", "outer", "inner")
_AVERAGE_FACTORIES = ("cell_avg", "face_avg")

# the BC-tagged nodal origins of the bounded trig transforms
# (DST-II, DST-I, DCT-II, DCT-I) — also the BC-tagged operands the
# staggered stencil rows are seeded for (the tag governs only the
# ghost fill; the stencils' codomains are the BC-free siblings)
_TRIG_ORIGIN_CANDIDATES: tuple[
    tuple[type[Transform], Callable[[Mesh], FunctionSpace]], ...] = (
    (Sine, lambda m: m.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)),
    (Sine, lambda m: m.nodal(NodeSet.INNER, bc=BC.DIRICHLET)),
    (Cosine, lambda m: m.nodal(NodeSet.CENTER, bc=BC.NEUMANN)),
    (Cosine, lambda m: m.nodal(NodeSet.OUTER, bc=BC.NEUMANN)),
)

# coefficient-space family -> transform class (explicit per-axes
# construction is the sanctioned non-registry path)
_TRANSFORM_FAMILY: tuple[tuple[type[CoefficientSpace],
                               type[Transform]], ...] = (
    (FourierSpace, Fourier),
    (SineSpace, Sine),
    (CosineSpace, Cosine),
    (ChebyshevSpace, Chebyshev),
)

# declared requirements of the lazily-seeded transform rows (the
# halo trace reads them off the LazyEntry without materializing)
_TRANSFORM_REQUIREMENTS = OperatorRequirements(
    halo=0, layout="transpose")


def _transform_family(factor: CoefficientSpace) -> type[Transform]:
    """Map a coefficient factor to its transform class."""
    for space_cls, transform_cls in _TRANSFORM_FAMILY:
        if isinstance(factor, space_cls):
            return transform_cls
    raise NotImplementedError(
        f"no transform family is defined for {factor!r} in "
        "iteration 1")


def _family_spaces(
    mesh: Mesh, attrs: tuple[str, ...],
) -> tuple[FunctionSpace, ...]:
    """Collect the factory spaces a mesh actually carries."""
    spaces = []
    for attr in attrs:
        try:
            spaces.append(getattr(mesh, attr))
        except (AttributeError, ValueError, NotImplementedError):
            continue  # factory absent on this mesh type/topology
    return tuple(spaces)


def _default_registry(
    grid: Grid,
    meshes: tuple[Mesh, ...],
    mapping: CoordinateMapping | None = None,
) -> OperatorRegistry:
    """
    Seed the default iteration-1 ``OperatorRegistry``.

    Description
    -----------
    One row per factor space instance of the meshes' space families
    (operators_composed.md default entry table, iteration-1 subset):
    ``("diff", nodal)`` -> ``FiniteDifference(order=2)`` and
    ``("interpolate", nodal)`` -> ``LinearInterp()`` wherever the
    per-factor signature applies (including the BC-tagged trig
    origins of bounded meshes, whose stencil codomains are the
    BC-free siblings); the FV/average family —
    ``("reconstruct", ...)`` -> ``LinearReconstruction()`` (its
    nodal -> average rows additionally seeded under ``("average",
    ...)``, the kind ``f.to`` resolves for that direction, and its
    ``CellAvg``/``FaceAvg`` rows also under ``("interpolate", ...)``
    so the composed metric machinery can hop an average component onto
    the face — G4), the co-located ``("deconvolve", CellAvg/Center)``
    -> ``LinearDeconvolution()`` (the 2nd-order identity ``f.to``
    resolves for a same-location average<->nodal pair — G3),
    ``("flux_diff", face)`` -> ``FluxDifference()``,
    ``("flux_diff", Center/CellAvg)`` -> ``DualFluxDifference()``,
    ``("face_diff", CellAvg)`` -> ``FaceDifference()``, and
    ``("diff", CellAvg)`` -> the ``FVDerivative()`` chain, whose
    ``Dispatched("reconstruct")`` hole is baked by the trailing
    ``merge({})`` (the iteration-1 assembly moment; module override
    merging happens later through ``grid.merge_overrides``);
    ``("integrate", nodal/average/tagged)`` -> one shared
    ``Integral()``; ``("cumint", Center/CellAvg)`` -> one shared
    ``CumulativeIntegral()`` (the running-integral rows, stage H1,
    on the center-valued integrand families only);
    the elementwise ``multiply``/``divide``/``power``/``select``/
    ``abs`` rows on nodal, average *and* BC-tagged trig-origin
    factors (one shared instance per kind — the registry's form-2
    product resolution requires it), each seeded for the real space
    and its complex variant; the kind-only ``"grad"``/``"div"``/``"curl"``/
    ``"laplacian"`` builder rows; the coefficient-space rows —
    ``("diff", coefficient)`` -> ``SpectralDerivative()`` and the
    one-directional ``("interpolate", Fourier(origin != Center))``
    -> ``PhaseShift(to=CENTER)`` / average-origin -> ``SincShift(
    to=CENTER)`` shifts; and the **lazily seeded**
    ``("transform", space)`` rows — one grid-closing ``LazyEntry``
    per family, constructed on first resolve, post-negotiation
    (grid lifecycle step 1). ``ConstantSpace`` factors deliberately
    get no rows.

    Parameters
    ----------
    grid : Grid
        The grid under construction (captured by the lazy
        transform factories only; nothing grid-bound is built
        during seeding).
    meshes : tuple[Mesh, ...]
        The grid's mesh factors.
    mapping : CoordinateMapping | None, optional
        The attached coordinate mapping; a mapping with a
        single-base analytic map seeds the kind-only
        ``"physical_diff"`` row — the constant-physical-coordinate
        derivative builder (rules section 3.8, sketch 4.4) — for
        exactly the coordinates it couples, and a mapping carrying
        an embedding chart (CS-D1, stage C2) seeds the metric-aware
        vector calculus: the ``("integrate", ...)`` rows become
        Jacobian-weighted (``Integral(jacobian=<chart coords>)``,
        rules 3.13), and — for charts coupling at least two
        coordinates — the kind-only ``"grad"`` / ``"div"`` /
        ``"curl"`` / ``"laplacian"`` rows hold the chart builders
        and ``"raise_index"`` / ``"lower_index"`` the explicit
        metric contractions (validation 6.3). Chartless grids keep
        the flat builders untouched (default: None).

    Returns
    -------
    OperatorRegistry
        The seeded default registry (placeholders resolved).
    """
    chart = (mapping.chart_coords
             if mapping is not None else None)
    flux_ops = (FluxDifference(), DualFluxDifference(),
                FaceDifference())
    reconstruct = LinearReconstruction()
    deconvolve = LinearDeconvolution()
    fv_derivative = FVDerivative()
    integral = Integral(jacobian=chart)
    cumint = CumulativeIntegral(jacobian=chart)
    multiply = CollocationProduct()
    divide = Divide()
    power = Power()
    abs_op = Abs()
    select = Where()
    entries: dict[DispatchKey, Operator] = {}
    for mesh in meshes:
        nodal = _family_spaces(mesh, _NODAL_FACTORIES)
        average = _family_spaces(mesh, _AVERAGE_FACTORIES)
        # BC-tagged bounded trig origins get the same stencil rows:
        # the stencils accept them (BC-aware ghost fill, BC-free
        # codomain), so a walled grid dispatches diff/interpolate
        # on Dirichlet/Neumann fields out of the box (C3)
        tagged = _tagged_trig_origins(mesh)
        # the BC-tagged CellAvg origins of the walled FV C-grid: the
        # Neumann DCT-II pressure (F4) and the Dirichlet DST-II
        # buoyancy (F5, the eigenmode analysis origin); empty on
        # periodic meshes
        tagged_avg = _tagged_average_origins(mesh)
        _seed_signature_rows(
            entries, nodal + tagged,
            (FiniteDifference(order=2), LinearInterp(), Restriction()))
        # reconstruct rows, plus (G4) the average family under the
        # "interpolate" kind for the composed metric machinery; the
        # tagged average origins mirror the untagged CellAvg rows, and
        # the tagged Inner (Dirichlet) origin picks up the
        # claim-consuming ("average", Inner) row of the walled
        # stratified w.to(b) seam (F4, scoping §10 correction 5)
        _seed_reconstruct_rows(
            entries, nodal + average + tagged_avg + tagged, reconstruct)
        # flux/face-diff rows, and (G3) the co-located CellAvg <->
        # Center deconvolution (a 2nd-order identity), each seeded via
        # its own dispatch kind where the per-factor signature applies
        _seed_signature_rows(
            entries, nodal + average + tagged_avg, (*flux_ops, deconvolve))
        # The elementwise + integrate rows are seeded on the tagged
        # origins too, so walled-grid fields interoperate (e.g. the
        # flux form ``csqr.to(v) * v`` on a Dirichlet face space).
        # The product keeps the common operand tag. A tag is a
        # *wall-value claim* consumed by the one-layer staggered
        # ghost fills, not a parity statement: keeping Dirichlet is
        # exact whenever one operand vanishes on the wall (the
        # wall-normal velocity, the free-slip vorticity), which
        # covers every product the walled Sadourny advection syncs —
        # including odd*odd ones like ``v * v``, whose exact wall
        # value 0 is precisely what the energy-conserving transpose
        # identities need (see shallowwater2/modules/sadourny.py).
        # A parity-aware product codomain is deliberately NOT added:
        # an even (Neumann) claim carries no wall value at all and
        # would lose the exact zero.
        _seed_elementwise_rows(
            entries, nodal + average + tagged + tagged_avg,
            fv_derivative,
            (integral, multiply, divide, power, select, abs_op))
        _seed_cumint_rows(entries, mesh, cumint)
        resolver = _declared_space_resolver(mesh)
        if resolver is not None:
            entries[("declared_space", mesh)] = resolver
    _seed_transform_rows(grid, meshes, entries)
    if mapping is not None:
        # metric-coefficient derivative kind (stage C1): the row
        # appears only when the mapping couples coordinates
        corrections = mapping.column_corrections
        if corrections:
            entries["physical_diff"] = MappedDerivative(corrections)
            _seed_column_closure(entries, meshes, corrections)
    if chart is not None and len(chart) > 1:
        # metric-aware vector calculus (stage C2): the same kinds,
        # chart-coupled entries (validation 6.3) — seeded only when
        # the mapping's embedding chart couples the coordinates
        entries["grad"] = MetricGradient(chart)
        entries["div"] = MetricDivergence(chart)
        entries["curl"] = MetricCurl(chart)
        entries["laplacian"] = MetricLaplacian(chart)
        # an orthogonal chart drops the off-diagonal contractions:
        # what lets an index move assemble across a bounded chart
        # axis (chart-ergonomics E2). The default keeps the full
        # expansion — correct-or-loud on a non-orthogonal chart.
        diagonal = mapping.orthogonal
        entries["raise_index"] = RaiseIndex(chart, diagonal=diagonal)
        entries["lower_index"] = LowerIndex(chart, diagonal=diagonal)
    entries.setdefault("grad", Gradient())
    entries.setdefault("div", Divergence())
    entries.setdefault("curl", Curl())
    entries.setdefault("laplacian", Laplacian())
    # bake the Dispatched("reconstruct") hole of the FV-derivative
    # chain against the seeded defaults (D4's merge moment, empty
    # override set): day-one `f.diff` on average spaces needs a
    # concrete chain even before any grid.merge_overrides call
    return OperatorRegistry(entries).merge({})


def _seed_column_closure(
    entries: dict[DispatchKey, Operator],
    meshes: tuple[Mesh, ...],
    corrections: Mapping[str, tuple[str, str]],
) -> None:
    """
    Open the BC-free ``Inner -> Center`` hop on mapped columns.

    Description
    -----------
    The near-wall closure of the physical-derivative correction
    chains (coordinate-systems plan, stage C4): on a **bounded**
    mapped column the sketch-4.4 correction differentiates along the
    column (``Center -> Inner``, BC-free) and must interpolate back
    (``Inner -> Center``) — a hop the default closed ``LinearInterp``
    legality rule rejects because the near-wall windows have no
    boundary data. A mapped column *requires* a closure for the
    seeded ``physical_diff`` kind to be usable at all, so the grid
    seeds the explicit one-sided variant
    (``LinearInterp(boundary="one_sided")``, boundary_plan 2d) for
    exactly the column base mesh's BC-free ``Inner`` space — the row
    stage C1 validated as a per-grid override. Flat grids, chart
    mappings, and periodic columns are untouched, and a module
    override of the same key still wins (defaults sit below the
    override layer).

    Parameters
    ----------
    entries : dict[DispatchKey, Operator]
        The default entry table under construction.
    meshes : tuple[Mesh, ...]
        The grid's mesh factors.
    corrections : Mapping[str, tuple[str, str]]
        The mapping's coupling table (coordinate -> (mapped, base)).
    """
    bases = {base for _, base in corrections.values()}
    one_sided = LinearInterp(boundary="one_sided")
    for mesh in meshes:
        if getattr(mesh, "periodic", True):
            continue
        if not any(name in bases for name in mesh.names):
            continue
        inner = getattr(mesh, "inner", None)
        if inner is not None:
            entries.setdefault(("interpolate", inner), one_sided)


def _declared_space_resolver(
    mesh: Mesh,
) -> Callable[[Dof, BC | BCStructure | None, str], FunctionSpace] | None:
    """
    Build the default ``("declared_space", mesh)`` resolver row.

    Description
    -----------
    Seeded per mesh factor at grid construction (model D1.2), so
    declared space patterns (``Collocated()`` / ``Staggered(...)`` /
    ``Profile(...)``) resolve on a bare grid without manual seeding.
    The resolver is family-aware (FV-D1b): its third argument is the
    effective discretization family the pattern resolved to (its own
    ``family=`` or the grid default).

    Under ``"nodal"`` (the default), ``Dof.COLLOCATED`` -> the
    center/nodal family (``ChebyshevMesh``, whose restricted family
    carries no cell centers, uses its outer/Lobatto family instead);
    ``Dof.STAGGERED`` -> the face family: ``Right`` on a periodic
    mesh, ``Inner`` on a bounded one — a C-grid wall-normal velocity
    carries interior faces only, the wall value is a boundary
    condition, not a DOF (an error on ``ChebyshevMesh`` — no face
    spaces exist there).

    Under ``"fv"`` (FV-D2 option A), ``Dof.COLLOCATED`` -> the
    ``CellAvg`` cell-average family (a taught error where it does not
    exist, e.g. a ``ChebyshevMesh``); ``Dof.STAGGERED`` stays the
    nodal face (``Right``/``Inner``), so an FV C-grid keeps
    face-normal velocities on the point-value faces. ``Dof.CONSTANT``
    never reaches a resolver (patterns route it to ``mesh.constant``
    directly; family-agnostic). Meshes without a nodal factory
    (``PointMesh``) get no default row and keep the hinted
    ``DispatchError``.

    Parameters
    ----------
    mesh : Mesh
        The mesh factor to build the resolver for.

    Returns
    -------
    Callable | None
        The ``(tag, bc, family) -> factor space`` resolver, or None
        when the mesh type has no default mapping.
    """
    if isinstance(mesh, ChebyshevMesh):
        node_sets: dict[Dof, NodeSet | None] = {
            Dof.COLLOCATED: NodeSet.OUTER,
            Dof.STAGGERED: None,
        }
    elif isinstance(mesh, StructuredMesh1D):
        node_sets = {
            Dof.COLLOCATED: NodeSet.CENTER,
            Dof.STAGGERED: (NodeSet.RIGHT if mesh.periodic
                            else NodeSet.INNER),
        }
    else:
        return None

    def resolver(
        tag: Dof,
        bc: BC | BCStructure | None,
        family: str = "nodal",
    ) -> FunctionSpace:
        """Resolve one (tag, bc, family) triple to a factor space."""
        if family == "fv":
            return _fv_factor_space(mesh, tag, bc)
        node_set = node_sets.get(tag)
        if node_set is None:
            raise ValueError(
                f"the default ('declared_space', {mesh!r}) resolver "
                f"row maps no {getattr(tag, 'name', tag)!s} "
                "representation on this mesh's space family; seed a "
                "custom grid-level resolver row for it")
        return mesh.nodal(node_set,
                          bc=BC.NONE if bc is None else bc)

    return resolver


def _fv_factor_space(
    mesh: Mesh,
    tag: Dof,
    bc: BC | BCStructure | None,
) -> FunctionSpace:
    """
    Resolve one declared tag to this mesh's FV factor space.

    Description
    -----------
    The FV branch of the default resolver row (FV-D1b / FV-D2 option
    A): ``Dof.COLLOCATED`` -> the ``CellAvg`` cell-average family (a
    BC-free space — average factors carry no boundary structure);
    ``Dof.STAGGERED`` -> the nodal face (``Right`` periodic /
    ``Inner`` bounded), so the FV C-grid keeps face-normal velocities
    on the point-value faces. A taught error where the family cannot
    be produced (a mesh without a ``cell_avg`` factory, e.g. a
    ``ChebyshevMesh``; a BC pinned on the BC-free cell average).

    Parameters
    ----------
    mesh : Mesh
        The mesh factor being resolved.
    tag : Dof
        The declaration tag (``COLLOCATED`` or ``STAGGERED``).
    bc : BC | BCStructure | None
        The matched BC (only meaningful for the staggered face).

    Returns
    -------
    FunctionSpace
        The FV factor space (the cell average or the nodal face).
    """
    if tag is Dof.STAGGERED:
        node_set = NodeSet.RIGHT if mesh.periodic else NodeSet.INNER
        return mesh.nodal(node_set, bc=BC.NONE if bc is None else bc)
    if tag is Dof.COLLOCATED:
        if bc is not None:
            raise ValueError(
                f"pattern pins a BC on a family='fv' collocated "
                f"coordinate of {mesh!r}, which resolves to the "
                "BC-free CellAvg cell average; average factors carry "
                "no boundary structure (topology-driven walls, C8)")
        try:
            return mesh.cell_avg
        except (AttributeError, ValueError, NotImplementedError) as exc:
            raise ValueError(
                f"family='fv' maps a collocated coordinate to the "
                f"CellAvg cell-average family, which {mesh!r} does "
                "not provide (a ChebyshevMesh has no cell averages); "
                "seed a custom grid-level resolver row or keep this "
                "coordinate family='nodal'") from exc
    raise ValueError(  # pragma: no cover — CONSTANT never reaches here
        f"the family='fv' resolver row of {mesh!r} maps no "
        f"{getattr(tag, 'name', tag)!s} representation; seed a "
        "custom grid-level resolver row for it")


def _probe(factory: Callable[[], object]) -> object | None:
    """Call a space factory; None where the family is absent."""
    try:
        return factory()
    except (AttributeError, TypeError, ValueError,
            NotImplementedError):
        return None


def _tagged_trig_origins(mesh: Mesh) -> tuple[FunctionSpace, ...]:
    """
    Collect the BC-tagged nodal trig origins one mesh grounds.

    Parameters
    ----------
    mesh : Mesh
        One grid mesh factor.

    Returns
    -------
    tuple[FunctionSpace, ...]
        The (real) BC-tagged origin spaces; empty on periodic
        meshes and wherever a candidate factory is absent.
    """
    if getattr(mesh, "periodic", False):
        return ()
    return tuple(
        origin for _family, factory in _TRIG_ORIGIN_CANDIDATES
        if (origin := _probe(lambda f=factory, m=mesh: f(m)))
        is not None)


def _tagged_average_origins(mesh: Mesh) -> tuple[FunctionSpace, ...]:
    """
    Collect the BC-tagged average trig origins one mesh grounds (F4/F5).

    Description
    -----------
    The walled FV C-grid closure: the Neumann-tagged ``CellAvg``
    origin of the DCT-II pressure transform (F4) **and** the
    Dirichlet-tagged ``CellAvg`` origin of the DST-II buoyancy
    transform (F5). Both sample on the cell-midpoint grid, so
    ``CellAvg`` maps to the type-II kernels exactly as ``Center``
    does (Neumann ``CellAvg`` → DCT-II, Dirichlet ``CellAvg`` →
    DST-II). The Dirichlet variant is the analysis origin of the
    walled-vertical FV eigenmode kit (``nh.eigenmodes`` on a bounded
    ``b``); F4 grounded only the Neumann sibling because the pressure
    solve was the sole consumer then. Empty on periodic meshes and
    wherever the ``CellAvg`` family is absent (a ``ChebyshevMesh``).

    Parameters
    ----------
    mesh : Mesh
        One grid mesh factor.

    Returns
    -------
    tuple[FunctionSpace, ...]
        The (real) Neumann and Dirichlet ``CellAvg`` origins, or
        empty.
    """
    if getattr(mesh, "periodic", False):
        return ()
    return tuple(
        origin for kind in (BC.NEUMANN, BC.DIRICHLET)
        if (origin := _probe(
            lambda m=mesh, k=kind: m.average(CellAvg, bc=k)))
        is not None)


def _seed_transform_rows(
    grid: Grid,
    meshes: tuple[Mesh, ...],
    entries: dict[DispatchKey, Operator],
) -> None:
    """
    Seed the transform and coefficient-space rows.

    Description
    -----------
    Per transform family, collect the origin spaces its meshes
    ground (Fourier: nodal + average families of periodic meshes,
    real and complex; Sine/Cosine: Dirichlet/Neumann-structured
    origins of bounded meshes; Chebyshev: the Gauss-Lobatto space)
    and seed one shared ``LazyEntry`` under every ``("transform",
    origin)`` key — the grid-bound instance (all family axes) is
    constructed on first resolve, post-negotiation. The families'
    coefficient spaces get the ``("diff", ...)`` ->
    ``SpectralDerivative`` rows (probed per signature; the I-type
    pair is covered since the Neumann shape decision) and the
    one-directional ``("interpolate", ...)`` origin shifts to
    Center.

    Parameters
    ----------
    grid : Grid
        The grid under construction (captured by the factories).
    meshes : tuple[Mesh, ...]
        The grid's mesh factors.
    entries : dict[DispatchKey, Operator]
        The entry table being built (mutated in place).
    """
    spectral = SpectralDerivative()
    phase_shift = PhaseShift(to=NodeSet.CENTER)
    sinc_shift = SincShift(to=NodeSet.CENTER)
    families: dict[type[Transform],
                   tuple[list[FunctionSpace], list[str]]] = {}
    for mesh in meshes:
        for family, origin in _transform_origins(mesh):
            origins, axes = families.setdefault(family, ([], []))
            for name in mesh.names:
                if name not in axes:
                    axes.append(name)
            for variant in (origin, origin.as_complex()):
                origins.append(variant)
                coeff = _probe(
                    lambda v=variant, m=mesh:
                    _coefficient_space(m, v))
                if coeff is not None:
                    _seed_coefficient_rows(entries, coeff, spectral,
                                           phase_shift, sinc_shift)
    for family, (origins, axes) in families.items():
        row = LazyEntry(
            lambda f=family, a=tuple(axes): f(grid, axes=a),
            requirements=_TRANSFORM_REQUIREMENTS)
        for origin in origins:
            entries[("transform", origin)] = row


def _transform_origins(
    mesh: Mesh,
) -> tuple[tuple[type[Transform], FunctionSpace], ...]:
    """
    Collect the (family, origin) pairs one mesh grounds.

    Parameters
    ----------
    mesh : Mesh
        One grid mesh factor.

    Returns
    -------
    tuple[tuple[type[Transform], FunctionSpace], ...]
        The transform family and (real) origin space pairs.
    """
    pairs: list[tuple[type[Transform], FunctionSpace]] = []
    if getattr(mesh, "periodic", False):
        for attr in _NODAL_FACTORIES + _AVERAGE_FACTORIES:
            origin = _probe(lambda a=attr, m=mesh: getattr(m, a))
            if origin is None or _probe(
                    lambda o=origin, m=mesh:
                    m.fourier(origin=o)) is None:
                continue  # family or Fourier signature absent
            pairs.append((Fourier, origin))
        return tuple(pairs)
    trig_origins = [
        (family, _probe(lambda f=factory, m=mesh: f(m)))
        for family, factory in _TRIG_ORIGIN_CANDIDATES]
    # the tagged CellAvg trig origins of the walled FV C-grid: the
    # Neumann DCT-II pressure (F4) and the Dirichlet DST-II buoyancy
    # (F5) — the family follows the tag, exactly as for the nodal
    # Center origins (all-Dirichlet -> Sine/DST, else Cosine/DCT)
    trig_origins += [
        (Sine if all(c is BC.DIRICHLET for c in origin.bc.components)
         else Cosine, origin)
        for origin in _tagged_average_origins(mesh)]
    for family, origin in trig_origins:
        if origin is None or _probe(
                lambda o=origin, m=mesh:
                _coefficient_space(m, o)) is None:
            continue  # origin or its coefficient family absent
        pairs.append((family, origin))
    if hasattr(mesh, "chebyshev"):
        lobatto = _probe(lambda m=mesh: m.outer)
        if lobatto is not None:
            pairs.append((Chebyshev, lobatto))
    return tuple(pairs)


def _coefficient_space(
    mesh: Mesh, origin: FunctionSpace,
) -> CoefficientSpace:
    """Build the family coefficient space of one origin."""
    if getattr(mesh, "periodic", False):
        return mesh.fourier(origin=origin)
    if hasattr(mesh, "chebyshev"):
        return mesh.chebyshev(origin)
    components = origin.bc.components
    if all(kind is BC.DIRICHLET for kind in components):
        return mesh.sine(origin)
    return mesh.cosine(origin)


def _seed_coefficient_rows(
    entries: dict[DispatchKey, Operator],
    coeff: CoefficientSpace,
    spectral: Operator,
    phase_shift: Operator,
    sinc_shift: Operator,
) -> None:
    """
    Seed the rows of one coefficient space.

    Parameters
    ----------
    entries : dict[DispatchKey, Operator]
        The entry table being built (mutated in place).
    coeff : CoefficientSpace
        The interned coefficient space.
    spectral : Operator
        The shared ``SpectralDerivative`` instance.
    phase_shift : Operator
        The shared ``PhaseShift(to=CENTER)`` instance.
    sinc_shift : Operator
        The shared ``SincShift(to=CENTER)`` instance.
    """
    _seed_signature_rows(entries, (coeff,), (spectral,))
    if not isinstance(coeff, FourierSpace):
        return
    origin = coeff.origin
    if isinstance(origin, AverageSpace):
        entries[("interpolate", coeff)] = sinc_shift
    elif origin.node_set is not NodeSet.CENTER:
        entries[("interpolate", coeff)] = phase_shift


def _seed_elementwise_rows(
    entries: dict[DispatchKey, Operator],
    spaces: tuple[FunctionSpace, ...],
    fv_derivative: Operator,
    elementwise: tuple[Operator, ...],
) -> None:
    """
    Seed the FV ``diff`` and the shared elementwise/integrate rows.

    Description
    -----------
    ``("diff", CellAvg)`` -> the collocated ``FVDerivative`` (the FV
    C-grid profile re-points it to ``FaceDifference`` later), and the
    shared ``integrate``/``multiply``/``divide``/``power``/``select``/
    ``abs`` rows on every space and its complex variant. Seeded on the
    BC-tagged origins too (nodal *and* average), so walled-grid fields
    interoperate — the product keeps the common operand tag (a
    wall-value claim, not a parity statement).

    Parameters
    ----------
    entries : dict[DispatchKey, Operator]
        The entry table being built (mutated in place).
    spaces : tuple[FunctionSpace, ...]
        The candidate factor spaces (nodal, average, tagged).
    fv_derivative : Operator
        The shared ``FVDerivative`` chain (``("diff", CellAvg)``).
    elementwise : tuple[Operator, ...]
        The shared ``(integral, multiply, divide, power, select,
        abs)`` instances, in that order.
    """
    integral, multiply, divide, power, select, abs_op = elementwise
    for space in spaces:
        if isinstance(space, CellAvg):
            entries[("diff", space)] = fv_derivative
        for variant in (space, space.as_complex()):
            entries[("integrate", variant)] = integral
            entries[("multiply", variant)] = multiply
            entries[("divide", variant)] = divide
            entries[("power", variant)] = power
            entries[("select", variant)] = select
            entries[("abs", variant)] = abs_op


def _seed_signature_rows(
    entries: dict[DispatchKey, Operator],
    spaces: tuple[FunctionSpace, ...],
    ops: tuple[Operator, ...],
) -> None:
    """
    Seed ``(op.dispatch_kind, space)`` rows where signatures apply.

    Parameters
    ----------
    entries : dict[DispatchKey, Operator]
        The entry table being built (mutated in place).
    spaces : tuple[FunctionSpace, ...]
        The candidate factor spaces.
    ops : tuple[Operator, ...]
        The kernels; a ``codomain`` rejection skips the row.
    """
    for space in spaces:
        for op in ops:
            try:
                op.codomain(space)
            except (SpaceMismatchError, ValueError):
                continue  # no per-factor signature on this space
            entries[(op.dispatch_kind, space)] = op


def _seed_cumint_rows(
    entries: dict[DispatchKey, Operator],
    mesh: Mesh,
    cumint: Operator,
) -> None:
    """
    Seed the running-integral rows on the center-valued families.

    Description
    -----------
    The ``("cumint", ...)`` rows (stage H1) live on the center-valued
    integrand families only — nodal ``Center`` and FV ``CellAvg`` (and
    their complex variants) — carrying one shared
    ``CumulativeIntegral`` (the bottom-up face default; top-down /
    co-located variants are constructed explicitly). A periodic mesh
    gets the row too, so ``f.cumint`` there raises the taught
    bounded-axis error rather than a bare ``DispatchError``.

    Parameters
    ----------
    entries : dict[DispatchKey, Operator]
        The entry table being built (mutated in place).
    mesh : Mesh
        The 1D mesh whose center/cell-average families to seed.
    cumint : Operator
        The shared ``CumulativeIntegral`` (Jacobian-tagged on charts).
    """
    # ``_family_spaces`` skips the families a mesh does not carry
    # (a ``PointMesh`` has no Center, a ``ChebyshevMesh`` no CellAvg)
    for space in _family_spaces(mesh, ("center", "cell_avg")):
        for variant in (space, space.as_complex()):
            entries[("cumint", variant)] = cumint


def _seed_reconstruct_rows(
    entries: dict[DispatchKey, Operator],
    spaces: tuple[FunctionSpace, ...],
    reconstruct: Operator,
) -> None:
    """
    Seed the reconstruction rows of the average family.

    Description
    -----------
    Every space with a reconstruct signature gets a
    ``("reconstruct", space)`` row; the nodal -> average direction is
    additionally seeded under ``("average", space)`` — the kind
    ``f.to`` resolves for that direction (fields.md family matrix),
    realized at second order by the same trapezoid two-point mean.

    The average factors are **also** seeded under the ``"interpolate"``
    kind (G4): ``registry.resolve("interpolate", CellAvg)`` is the key
    the composed vector-calculus machinery hard-codes to move a
    component one staggering hop onto a target component's space
    (``composed._interp_onto``), and the reconstruct instance is that
    hop — landing ``CellAvg -> Right`` (periodic) / ``Inner`` (bounded).
    ``FaceAvg -> Center`` falls out of the identical path for free (the
    decision record invests nothing in ``FaceAvg`` beyond that).
    ``.to`` never reads this kind for an average source (it resolves
    ``"reconstruct"``/``"deconvolve"``), so the row is additive.

    Parameters
    ----------
    entries : dict[DispatchKey, Operator]
        The entry table being built (mutated in place).
    spaces : tuple[FunctionSpace, ...]
        The candidate factor spaces (nodal and average).
    reconstruct : Operator
        The shared ``LinearReconstruction`` instance.
    """
    for space in spaces:
        try:
            codomain = reconstruct.codomain(space)
        except SpaceMismatchError:
            continue  # no per-factor signature on this space
        entries[("reconstruct", space)] = reconstruct
        if isinstance(codomain, AverageSpace):
            entries[("average", space)] = reconstruct
        if isinstance(space, AverageSpace):
            entries[("interpolate", space)] = reconstruct  # G4
