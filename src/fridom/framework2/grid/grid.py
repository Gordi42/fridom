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
the sync-after-every-operator contract. Phase-2 assembly re-runs
``negotiate(state_spaces=..., tendency=...)`` and ends with
``freeze()``; ``merge_overrides`` stays a stub until the
operator-registry merge lands.
"""
# Wave 2: Grid -- Wave 3: negotiate/freeze lifecycle
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.decomposition.decomposition import (
    ReshardingReport,
    negotiate,
)
from fridom.framework2.grid.errors import (
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
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.finite_difference import (
    FiniteDifference,
)
from fridom.framework2.grid.operators.interp import LinearInterp
from fridom.framework2.grid.operators.products import (
    Abs,
    CollocationProduct,
    Divide,
    Power,
)
from fridom.framework2.grid.operators.registry import OperatorRegistry
from fridom.framework2.grid.random_fields import RandomFieldFactory
from fridom.framework2.grid.spaces.average import CellAvg, FaceAvg
from fridom.framework2.grid.spaces.coefficient import CoefficientSpace
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.grid.spaces.nodal import NodalSpace, NodeSet
from fridom.framework2.grid.spaces.tensor_product import (
    TensorProductSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import jax

    from fridom.framework2.grid.decomposition.decomposition import (
        Decomposition,
    )
    from fridom.framework2.grid.decomposition.halo import HaloSpec
    from fridom.framework2.grid.meshes.mesh import Mesh
    from fridom.framework2.grid.operators.base import Operator
    from fridom.framework2.grid.operators.registry import DispatchKey
    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


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
            _default_registry(meshes) if dispatch is None
            else dispatch)
        self._device_ids: tuple[int, ...] | None = device_ids
        self._frozen: bool = False
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
        self, overrides: Mapping[object, object],
    ) -> None:
        """Merge module-local dispatch overrides (pre-freeze only)."""
        raise NotImplementedError(
            "merge_overrides arrives with the operator-registry "
            "merge (the grid then swaps its held registry instance)")

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
        Renegotiate the decomposition (pre-freeze only).

        Description
        -----------
        Re-runs the mesh-traits x operator-demands negotiation
        (Phase-2 assembly, grid lifecycle step 3): the halo comes
        from the traced `tendency` when supplied, else the
        per-operator registry maximum scoped to `state_spaces`, else
        the explicit `halo=` override. Returns the report the model
        uses to re-``device_put`` its live state once; derived
        arrays need nothing (recompute-on-demand).

        Parameters
        ----------
        state_spaces : tuple[SpaceLike, ...] | None, optional
            The model's state-field spaces (default: None).
        tendency : Callable[..., object] | None, optional
            The tendency to halo-trace (default: None).
        halo : HaloSpec | None, optional
            Explicit per-name halo override (default: None).

        Returns
        -------
        ReshardingReport
            Old/new default layout and whether they differ.
        """
        if self._frozen:
            raise RuntimeError(
                "the grid is frozen; negotiate is legal in the "
                "assembly phase only (grid lifecycle)")
        old = self._decomposition.default_layout
        self._decomposition = negotiate(
            self, self._dispatch,
            state_spaces=state_spaces, tendency=tendency, halo=halo,
            device_ids=self._device_ids)
        new = self._decomposition.default_layout
        return ReshardingReport(old=old, new=new, changed=old != new)

    def freeze(self) -> None:
        """End the assembly phase; further negotiations raise."""
        self._frozen = True

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
            The synced field (metadata preserved).
        """
        if boundary_data is not None:
            raise NotImplementedError(
                "inhomogeneous ghost fill is designed-for; "
                "iteration 1 is homogeneous only")
        space = field.function_space
        synced = self._decomposition.sync(
            field._data, space)  # noqa: SLF001 — storage seam
        return ScalarField(self, space, synced, field.metadata)

    # ================================================================
    #  Field factory
    # ================================================================
    def create_field(
        self,
        space: SpaceLike | None = None,
        *,
        init: Callable[..., jax.Array] | None = None,
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
        ``data=`` takes a true-shape array (validated, dtype-coerced,
        Hermitian-projected on real-origin Fourier factors); with
        neither the field is zeros. Bare spaces get the
        decomposition's default layout attached; laid-out spaces are
        honored as given.

        Parameters
        ----------
        space : SpaceLike | None, optional
            The function space; None means the all-Center nodal
            product (default: None).
        init : Callable[..., jax.Array] | None, optional
            Function of the physical coordinates, matched by
            coordinate name (default: None).
        data : jax.Array | None, optional
            True-shape array companion, mutually exclusive with
            ``init`` (default: None).
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
        if init is not None and data is not None:
            raise ValueError(
                "init= and data= are mutually exclusive")
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
        candidates = tuple(
            coord for factor in space.factors
            if not isinstance(factor, ConstantSpace)
            for coord in factor.names)
        if name is None:
            if len(candidates) != 1:
                raise ValueError(
                    "the coordinate is ambiguous on this space; "
                    f"pass name= (one of {candidates})")
            name = candidates[0]
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
        """Wavenumbers (or mode indices) of a coefficient space."""
        raise NotImplementedError(
            "wavenumbers arrives with the transform cluster "
            "(Wave-2 operators merge)")

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
            if isinstance(factor, CoefficientSpace):
                raise NotImplementedError(
                    "init= on coefficient spaces composes with the "
                    "forward transform (discretize = transform o "
                    "discretize_origin); it arrives with the "
                    "Wave-2 operators merge")
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
    # drop BC-constrained boundary DOFs (left, then right), exactly
    # like the space shapes drop them
    start, stop = 0, count
    components = factor.bc.components
    if components:
        left, right = components
        if membership[0] and left is not BC.NONE:
            start += 1
        if membership[1] and right is not BC.NONE:
            stop -= 1
    return nodes[start:stop]


# ================================================================
#  Default registry seeding (grid lifecycle step 1; the provisional
#  halo of step 2 lives in decomposition.negotiate)
# ================================================================
# mesh factory attributes of the seeded space families
_NODAL_FACTORIES = ("center", "left", "right", "outer", "inner")
_AVERAGE_FACTORIES = ("cell_avg", "face_avg")


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


def _default_registry(meshes: tuple[Mesh, ...]) -> OperatorRegistry:
    """
    Seed the default iteration-1 ``OperatorRegistry``.

    Description
    -----------
    Grid-free entries only (grid lifecycle step 1), one row per
    factor space instance of the meshes' space families
    (operators_composed.md default entry table, iteration-1 subset):
    ``("diff", nodal)`` -> ``FiniteDifference(order=2)`` and
    ``("interpolate", nodal)`` -> ``LinearInterp()`` wherever the
    per-factor signature applies; the elementwise
    ``multiply``/``divide``/``power`` rows on nodal *and* average
    factors (one shared instance per kind — the registry's form-2
    product resolution requires it) and ``abs`` on nodal factors
    only, each seeded for the real space and its complex variant.
    Coefficient and constant factors deliberately get no rows.

    Parameters
    ----------
    meshes : tuple[Mesh, ...]
        The grid's mesh factors.

    Returns
    -------
    OperatorRegistry
        The seeded default registry.
    """
    fd = FiniteDifference(order=2)
    interp = LinearInterp()
    multiply = CollocationProduct()
    divide = Divide()
    power = Power()
    abs_op = Abs()
    entries: dict[DispatchKey, Operator] = {}
    for mesh in meshes:
        nodal = _family_spaces(mesh, _NODAL_FACTORIES)
        average = _family_spaces(mesh, _AVERAGE_FACTORIES)
        for space in nodal:
            for stencil in (fd, interp):
                try:
                    stencil.codomain(space)
                except (SpaceMismatchError, ValueError):
                    continue  # no per-factor signature on this space
                entries[(stencil.dispatch_kind, space)] = stencil
        for space in nodal + average:
            for variant in (space, space.as_complex()):
                entries[("multiply", variant)] = multiply
                entries[("divide", variant)] = divide
                entries[("power", variant)] = power
        for space in nodal:
            for variant in (space, space.as_complex()):
                entries[("abs", variant)] = abs_op
    return OperatorRegistry(entries)
