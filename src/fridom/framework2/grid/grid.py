"""
``Grid``: the model-agnostic assembly root.

Description
-----------
Owning class doc: ``notes/framework2/classes/grid.md``. Meshes +
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

import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.framework2.grid.bc import BC, BCStructure
from fridom.framework2.grid.decomposition.decomposition import (
    ReshardingReport,
    _registry_halo,
    negotiate,
)
from fridom.framework2.grid.decomposition.halo import (
    HaloSpec,
    trace_halo,
)
from fridom.framework2.grid.errors import (
    GridFrozenError,
    GridMismatchError,
    SpaceMismatchError,
)
from fridom.framework2.grid.fields.metadata import FieldMetadata
from fridom.framework2.grid.fields.scalar_field import ScalarField
from fridom.framework2.grid.fields.storage import (
    factor_axes,
    hermitian_project,
    storage_dtype,
    store,
)
from fridom.framework2.grid.meshes.chebyshev import ChebyshevMesh
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.meshes.structured_1d import (
    StructuredMesh1D,
)
from fridom.framework2.grid.operators.base import OperatorRequirements
from fridom.framework2.grid.operators.chebyshev import Chebyshev
from fridom.framework2.grid.operators.composed import (
    Curl,
    Divergence,
    Gradient,
    Laplacian,
)
from fridom.framework2.grid.operators.finite_difference import (
    FiniteDifference,
)
from fridom.framework2.grid.operators.flux_diff import (
    DualFluxDifference,
    FaceDifference,
    FluxDifference,
    FVDerivative,
)
from fridom.framework2.grid.operators.fourier import Fourier
from fridom.framework2.grid.operators.integrate import Integral
from fridom.framework2.grid.operators.interp import LinearInterp
from fridom.framework2.grid.operators.products import (
    Abs,
    CollocationProduct,
    Divide,
    Power,
)
from fridom.framework2.grid.operators.reconstruct import (
    LinearReconstruction,
)
from fridom.framework2.grid.operators.registry import (
    DispatchCollisionError,
    LazyEntry,
    OperatorRegistry,
    check_override_key,
)
from fridom.framework2.grid.operators.select import Where
from fridom.framework2.grid.operators.spectral import (
    PhaseShift,
    SincShift,
    SpectralDerivative,
    chebyshev_modes,
    fourier_wavenumbers,
    trig_wavenumbers,
)
from fridom.framework2.grid.operators.trig import Cosine, Sine
from fridom.framework2.grid.random_fields import RandomFieldFactory
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.average import (
    AverageSpace,
    CellAvg,
    FaceAvg,
)
from fridom.framework2.grid.spaces.coefficient import (
    ChebyshevSpace,
    CoefficientSpace,
    CosineSpace,
    FourierSpace,
    SineSpace,
)
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.grid.spaces.nodal import NodalSpace, NodeSet
from fridom.framework2.grid.spaces.tensor_product import (
    TensorProductSpace,
)

# the declaration-tag vocabulary of the ("declared_space", mesh)
# resolver rows (model D1.2); the model layer resolves declared
# patterns through the rows seeded below, so the grid must speak the
# tag enum (an acknowledged upward import of pure vocabulary)
from fridom.framework2.model.space_patterns import Dof

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

    from fridom.framework2.grid.decomposition.decomposition import (
        Decomposition,
    )
    from fridom.framework2.grid.decomposition.layout import Layout
    from fridom.framework2.grid.immersed_domain import ImmersedDomain
    from fridom.framework2.grid.meshes.mesh import Mesh
    from fridom.framework2.grid.operators.base import Operator
    from fridom.framework2.grid.operators.registry import DispatchKey
    from fridom.framework2.grid.operators.transform import Transform
    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


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
    immersed : ImmersedDomain | None, optional
        The immersed (masked) domain descriptor to attach; the grid
        binds it on attachment (default: None).
    device_ids : tuple[int, ...] | None, optional
        Indices into ``jax.devices()``; None lets negotiation use
        every available device, falling back to one when nothing is
        shardable (default: None).
    """

    def __init__(
        self,
        meshes: tuple[Mesh, ...],
        *,
        dispatch: object | None = None,
        immersed: ImmersedDomain | None = None,
        device_ids: tuple[int, ...] | None = None,
    ) -> None:
        """Assemble a grid from pre-built, pre-named mesh factors."""
        meshes = tuple(meshes)
        if not meshes:
            raise ValueError("a grid needs at least one mesh factor")
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
        self._dispatch: object = (
            _default_registry(self, meshes) if dispatch is None
            else dispatch)
        self._device_ids: tuple[int, ...] | None = device_ids
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
            self, self._dispatch, device_ids=device_ids)
        self._random: RandomFieldFactory = RandomFieldFactory(self)

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

    # ================================================================
    #  Operator dispatch (seam: registry class owned by the
    #  operators cluster; the grid owns the instance)
    # ================================================================
    @property
    def dispatch(self) -> object:
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
            device_ids=self._device_ids)
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
            field._data, space)  # noqa: SLF001 — storage seam
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
        (discretize = transform o discretize-on-origin).
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
            return self._transform_discretize(space, init, metadata)
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
            arr = self._discretize(space, init).astype(dtype)
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
        it is (rules sections 2.7, 3.9). Uniform ``IntervalMesh``
        geometry in iteration 1; the result is tagged with the
        querying space, all other factors replaced by their
        ``ConstantSpace``, so it broadcasts exactly (section 3.3).

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
        return ScalarField(self, result, stored,
                           FieldMetadata.create(name=f"d{name}"))

    # ================================================================
    #  Attachments
    # ================================================================
    @property
    def immersed(self) -> ImmersedDomain | None:
        """The immersed (masked) domain descriptor, or None."""
        return self._immersed

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
    ) -> jax.Array:
        """
        Collocation default of the ``("discretize", space)`` row.

        Description
        -----------
        Inline iteration-1 default (the registry row proper lands
        with the operators merge): keyword-match ``init`` against
        the non-constant coordinate names, broadcast the per-factor
        node coordinates transiently, and sample.
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

    def _transform_discretize(
        self,
        space: SpaceLike,
        init: Callable[..., jax.Array],
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
        produce raises ``SpaceMismatchError``.

        Parameters
        ----------
        space : SpaceLike
            The laid-out target space (>= 1 coefficient factor).
        init : Callable[..., jax.Array]
            Function of the physical coordinates.
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
                                  metadata=metadata)
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
#  Node coordinates of the iteration-1 spaces (IntervalMesh)
# ================================================================
# offsets of the first node from x_min, in units of dx, and the DOF
# count offset relative to the cell count n
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
    Uniform ``IntervalMesh`` geometry (iteration 1): nodal spaces at
    their node sets, ``CellAvg`` at the midpoint-quadrature points
    (cell centers), ``FaceAvg`` at the dual-cell midpoints (faces).
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
    if not isinstance(mesh, IntervalMesh):
        raise NotImplementedError(
            f"evaluation nodes on {type(mesh).__name__} arrive in a "
            "later wave; iteration 1 covers IntervalMesh")
    x_min = mesh.extent[0]
    dx = mesh.dx
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
    nodes = x_min + (jnp.arange(count, dtype=dtype_real())
                     + offset) * dx
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


def _measure_vector(factor: FunctionSpace) -> jax.Array:
    """
    Materialize the 1D measure weights of one factor space.

    Description
    -----------
    Uniform ``IntervalMesh`` geometry (iteration 1): the constant
    cell width ``dx`` everywhere, except that on bounded meshes a
    node sitting *on* the boundary (a boundary-member DOF of its
    node set) owns the clipped half dual cell ``dx / 2`` — which
    makes the ``Outer`` weights exactly the trapezoid rule.
    BC-constrained boundary DOFs are dropped exactly like the space
    shapes drop them.

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
    if not isinstance(mesh, IntervalMesh):
        raise NotImplementedError(
            f"measure fields on {type(mesh).__name__} arrive in a "
            "later wave; iteration 1 covers IntervalMesh")
    dx = mesh.dx
    count_offset, membership = _measure_geometry(factor)
    count = mesh.n_cells + count_offset
    weights = jnp.full(count, dx, dtype=dtype_real())
    if not mesh.periodic:
        if membership[0]:
            weights = weights.at[0].set(dx / 2)
        if membership[1]:
            weights = weights.at[-1].set(dx / 2)
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
        A nodal or average factor space of an ``IntervalMesh``.

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


# ================================================================
#  Default registry seeding (grid lifecycle step 1; the provisional
#  halo of step 2 lives in decomposition.negotiate)
# ================================================================
# mesh factory attributes of the seeded space families
_NODAL_FACTORIES = ("center", "left", "right", "outer", "inner")
_AVERAGE_FACTORIES = ("cell_avg", "face_avg")

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
    grid: Grid, meshes: tuple[Mesh, ...],
) -> OperatorRegistry:
    """
    Seed the default iteration-1 ``OperatorRegistry``.

    Description
    -----------
    One row per factor space instance of the meshes' space families
    (operators_composed.md default entry table, iteration-1 subset):
    ``("diff", nodal)`` -> ``FiniteDifference(order=2)`` and
    ``("interpolate", nodal)`` -> ``LinearInterp()`` wherever the
    per-factor signature applies; the FV/average family —
    ``("reconstruct", ...)`` -> ``LinearReconstruction()`` (its
    nodal -> average rows additionally seeded under ``("average",
    ...)``, the kind ``f.to`` resolves for that direction),
    ``("flux_diff", face)`` -> ``FluxDifference()``,
    ``("flux_diff", Center/CellAvg)`` -> ``DualFluxDifference()``,
    ``("face_diff", CellAvg)`` -> ``FaceDifference()``, and
    ``("diff", CellAvg)`` -> the ``FVDerivative()`` chain, whose
    ``Dispatched("reconstruct")`` hole is baked by the trailing
    ``merge({})`` (the iteration-1 assembly moment; module override
    merging happens later through ``grid.merge_overrides``);
    ``("integrate", nodal/average)`` -> one shared ``Integral()``;
    the elementwise ``multiply``/``divide``/``power``/``select``
    rows on nodal *and* average factors (one shared instance per
    kind — the registry's form-2 product resolution requires it),
    ``abs`` on nodal factors only, each seeded for the real space
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

    Returns
    -------
    OperatorRegistry
        The seeded default registry (placeholders resolved).
    """
    fd = FiniteDifference(order=2)
    interp = LinearInterp()
    flux_ops = (FluxDifference(), DualFluxDifference(),
                FaceDifference())
    reconstruct = LinearReconstruction()
    fv_derivative = FVDerivative()
    integral = Integral()
    multiply = CollocationProduct()
    divide = Divide()
    power = Power()
    abs_op = Abs()
    select = Where()
    entries: dict[DispatchKey, Operator] = {}
    for mesh in meshes:
        nodal = _family_spaces(mesh, _NODAL_FACTORIES)
        average = _family_spaces(mesh, _AVERAGE_FACTORIES)
        _seed_signature_rows(entries, nodal, (fd, interp))
        _seed_reconstruct_rows(entries, nodal + average, reconstruct)
        _seed_signature_rows(entries, nodal + average, flux_ops)
        for space in nodal + average:
            if isinstance(space, CellAvg):
                entries[("diff", space)] = fv_derivative
            for variant in (space, space.as_complex()):
                entries[("integrate", variant)] = integral
                entries[("multiply", variant)] = multiply
                entries[("divide", variant)] = divide
                entries[("power", variant)] = power
                entries[("select", variant)] = select
                entries[("abs", variant)] = abs_op
        resolver = _declared_space_resolver(mesh)
        if resolver is not None:
            entries[("declared_space", mesh)] = resolver
    _seed_transform_rows(grid, meshes, entries)
    entries["grad"] = Gradient()
    entries["div"] = Divergence()
    entries["curl"] = Curl()
    entries["laplacian"] = Laplacian()
    # bake the Dispatched("reconstruct") hole of the FV-derivative
    # chain against the seeded defaults (D4's merge moment, empty
    # override set): day-one `f.diff` on average spaces needs a
    # concrete chain even before any grid.merge_overrides call
    return OperatorRegistry(entries).merge({})


def _declared_space_resolver(
    mesh: Mesh,
) -> Callable[[Dof, BC | BCStructure | None], FunctionSpace] | None:
    """
    Build the default ``("declared_space", mesh)`` resolver row.

    Description
    -----------
    Seeded per mesh factor at grid construction (model D1.2), so
    declared space patterns (``Collocated()`` / ``Staggered(...)`` /
    ``Profile(...)``) resolve on a bare grid without manual seeding.
    The default mapping over each mesh's space vocabulary:
    ``Dof.COLLOCATED`` -> the center/nodal family (``ChebyshevMesh``,
    whose restricted family carries no cell centers, uses its
    outer/Lobatto family instead); ``Dof.STAGGERED`` -> the
    face/right family (an error on ``ChebyshevMesh`` — no face
    spaces exist there). ``Dof.CONSTANT`` never reaches a resolver:
    patterns route it to ``mesh.constant`` directly. Meshes without
    a nodal factory (``PointMesh``) get no default row and keep the
    hinted ``DispatchError``.

    Parameters
    ----------
    mesh : Mesh
        The mesh factor to build the resolver for.

    Returns
    -------
    Callable | None
        The ``(tag, bc) -> factor space`` resolver, or None when
        the mesh type has no default mapping.
    """
    if isinstance(mesh, ChebyshevMesh):
        node_sets: dict[Dof, NodeSet | None] = {
            Dof.COLLOCATED: NodeSet.OUTER,
            Dof.STAGGERED: None,
        }
    elif isinstance(mesh, StructuredMesh1D):
        node_sets = {
            Dof.COLLOCATED: NodeSet.CENTER,
            Dof.STAGGERED: NodeSet.RIGHT,
        }
    else:
        return None

    def resolver(
        tag: Dof,
        bc: BC | BCStructure | None,
    ) -> FunctionSpace:
        """Resolve one (tag, bc) pair to this mesh's factor space."""
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


def _probe(factory: Callable[[], object]) -> object | None:
    """Call a space factory; None where the family is absent."""
    try:
        return factory()
    except (AttributeError, TypeError, ValueError,
            NotImplementedError):
        return None


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
    candidates = (
        (Sine, lambda m: m.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)),
        (Sine, lambda m: m.nodal(NodeSet.INNER, bc=BC.DIRICHLET)),
        (Cosine, lambda m: m.nodal(NodeSet.CENTER, bc=BC.NEUMANN)),
        (Cosine, lambda m: m.nodal(NodeSet.OUTER, bc=BC.NEUMANN)),
    )
    for family, factory in candidates:
        origin = _probe(lambda f=factory, m=mesh: f(m))
        if origin is not None:
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
