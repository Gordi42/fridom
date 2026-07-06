# Grid abstraction redesign — Class designs: grid assembly and decomposition

Part of the grid redesign notes; see [`../00_overview.md`](../00_overview.md)
for the document map. Status: draft class design, no implementation.
Signatures are the intended public API for `framework.grid2`; the
numbered concept sections remain the normative reference.

This document owns the **Grid assembly and domain-decomposition
cluster**: the model-agnostic `Grid` root, the cartesian convenience
subclass, the `grid.random` factory, the immersed-domain and
coordinate-mapping attachments, and the per-mesh domain-decomposition
layer. Meshes and function spaces are doc 01, fields are doc 02,
operators / transforms / the `OperatorRegistry` internals are doc 03;
this document only names them at the seams.

Fixed seam anchors used below: space factories `mx.center` / `mx.right`
/ `mx.fourier(origin=...)`, `space_a * space_b -> TensorProductSpace`,
`fr.operators.Fourier(grid, axes=...)` with `.forward` / `.backward`,
`op.eigenvalues(grid, coeff_space) -> Symbol`, `f.diff("x")` dispatching
`("diff", space)` through the grid registry, and fields carrying their
grid.

---

## 1. Package layout summary

Transitional home is `fridom.framework.grid2`; it is renamed to the
canonical `framework.grid` once the old grid is deleted
([overview §8](../00_overview.md#8-migration-strategy)). Docs 01–03
state their own placements; the tree below shows the whole subpackage
so this cluster's modules have an address, with ownership per doc:

```
fridom/framework/grid2/
    __init__.py              # lazypimp re-exports (below)
    scalars.py               # doc 01: fr.Real / fr.Complex, Variance
    bc.py                    # doc 01: fr.BC (iteration 1)
    errors.py                # doc 02: SpaceMismatchError,
                             #   GridMismatchError
    meshes/                  # doc 01: Mesh subclasses (fr.meshes)
    spaces/                  # doc 01: FunctionSpace families,
                             #   ConstantSpace
        tensor_product.py    # doc 02: TensorProductSpace, SpaceLike
    fields/                  # doc 02: ScalarField, VectorField, State
    operators/               # doc 03: Operator hierarchy, transforms,
                             #   Symbol (fr.operators)
        registry.py          # doc 03: OperatorRegistry
    grid.py                  # THIS DOC: Grid (assembly root)
    discretize.py            # THIS DOC: Discretizer protocol
    random_fields.py         # THIS DOC: RandomFieldFactory
    immersed_domain.py       # THIS DOC: ImmersedDomain, Slip
    coordinate_mapping.py    # THIS DOC: CoordinateMapping
    cartesian/
        grid.py              # THIS DOC: cartesian convenience Grid
    decomposition/           # THIS DOC (all of it):
        traits.py            #   HaloStrategy, MeshDecompositionTraits
        halo.py              #   HaloSpec, HaloTracer, trace_halo,
                             #   GhostFill (designed-for)
        layout.py            #   ArrayLayout
        decomposition.py     #   Decomposition (ABC), negotiate(),
                             #   ReshardingReport
        tensor.py            #   TensorDecomposition
        graph.py             #   GraphDecomposition (designed-for)
```

Top-level re-exports (final names left, transitional names right; per
[overview §8](../00_overview.md#8-migration-strategy) the plural
collection namespaces match `fr.modules` / `fr.time_steppers`):

| Final                    | Transitional                | Object |
|--------------------------|-----------------------------|--------|
| `fr.Grid`                | `fr.grid2.Grid`             | assembly root (this doc) |
| `fr.grid.cartesian.Grid` | `fr.grid2.cartesian.Grid`   | convenience subclass (this doc) |
| `fr.meshes`              | `fr.grid2.meshes`           | mesh factors (doc 01) |
| `fr.operators`           | `fr.grid2.operators`        | free-standing operators (doc 03) |
| `fr.Real`, `fr.Complex`  | `fr.grid2.Real`, `fr.grid2.Complex` | scalars / Körper tags (doc 01) |
| `fr.BC`                  | `fr.grid2.BC`               | BC structure enum (doc 01, day one: `DIRICHLET`/`NEUMANN` are exercised by the iteration-1 Sine/Cosine spaces) |
| `fr.ScalarField`, `fr.VectorField`, `fr.TensorField` | `fr.grid2.*` | field types (doc 02) |
| `fr.TensorProductSpace`, `fr.SpaceLike` | `fr.grid2.*`  | product space + space alias (doc 02) |
| `fr.FieldMetadata`       | `fr.grid2.FieldMetadata`    | field metadata record (doc 02) |
| `fr.SpaceMismatchError`, `fr.GridMismatchError` | `fr.grid2.*` | error types (doc 02) |

The table is exhaustive: these are all `fr.*`-level names contributed
by `grid2`. Function spaces get **no** top-level namespace: they are
produced by mesh factories (`mx.center`, `mz.galerkin(...)`). The
decomposition subpackage is *not* re-exported at `fr.*` level — fields
and operators reach it only through the grid
([§5](../04_decomposition.md#5-domain-decomposition)); it is public for
transform/solver authors as `fr.grid2.decomposition`.

The transitional `grid2/__init__.py` realizes the table with the
lazypimp pattern mandated by `AGENTS.md` (at rename time the same
entries move up to the framework `__init__`):

```python
base = "fridom.framework.grid2"

all_modules_by_origin = {
    base: ["meshes", "operators", "cartesian", "decomposition"],
}

all_imports_by_origin = {
    f"{base}.grid": ["Grid"],
    f"{base}.scalars": ["Real", "Complex"],
    f"{base}.bc": ["BC"],
    f"{base}.errors": ["SpaceMismatchError", "GridMismatchError"],
    f"{base}.spaces.tensor_product": ["TensorProductSpace",
                                      "SpaceLike"],
    f"{base}.fields": ["ScalarField", "VectorField", "TensorField",
                       "FieldMetadata"],
    f"{base}.immersed_domain": ["ImmersedDomain", "Slip"],
    f"{base}.coordinate_mapping": ["CoordinateMapping"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
```

Tests mirror the package as `tests/framework/grid2/**`, one test file
per module plus a `test_init.py` per package directory that
parametrizes over the re-exports above (the repo-wide pattern).

---

## 2. Grid assembly

### Grid

The model-agnostic assembly root: meshes + decomposition + dispatch
registry + field factory + attachments. Ergonomics and wiring only —
all mathematics lives in spaces and operators, and model physics
(`omega`, `vec_q`, `vec_p`) has left the grid entirely
([§2.6](../01_concepts.md#26-grid--the-assembly-object)).

- Kind: concrete (also the base class of the cartesian convenience
  subclass; not an ABC — it is fully functional as-is).
- Module: `fridom.framework.grid2.grid`.
- Pytree: **fully static**. The grid is not a pytree container and
  registers **no** dynamic attributes; it appears only as static aux
  data in field pytrees (fields carry their grid). Identity hashing is
  **explicit** (`__eq__` is `self is other`, `__hash__` matches) —
  necessary because fridom's `_values_equal` in `utils/jax_utils.py`
  deep-compares aux objects that keep the default `__eq__`.
- Iteration: 1 (class, factory, coordinate accessors, negotiation);
  `immersed` ships iteration 1 in its boolean subset only; `mapping` /
  `metric` are designed-for.
- Concept refs: [§2.6](../01_concepts.md#26-grid--the-assembly-object),
  [§2.7](../01_concepts.md#27-where-coordinate-data-lives),
  [§3.4](../02_rules.md#34-generic-operator-dispatch),
  [§3.7](../02_rules.md#37-boundaries-ii-immersed-masked-domains),
  [§3.8](../02_rules.md#38-boundaries-iii-terrain-following-boundary-fitted),
  [§3.10](../02_rules.md#310-discretizing-continuous-functions),
  [§5](../04_decomposition.md#5-domain-decomposition).

```python
class Grid:
    """Assembly root: meshes, decomposition, dispatch, field factory."""

    def __init__(
        self,
        meshes: tuple[Mesh, ...],
        *,
        defaults: Mapping[DispatchKey, Operator] | None = None,
        mapping: CoordinateMapping | None = None,
        immersed: ImmersedDomain | None = None,
        device_ids: tuple[int, ...] | None = None,
    ) -> None:
        """Assemble a grid from pre-built, pre-named mesh factors."""
        ...

    # ================================================================
    #  Identity
    # ================================================================

    def __eq__(self, other: object) -> bool:
        """Identity comparison: `self is other` (static aux object)."""
        ...

    def __hash__(self) -> int:
        """Identity hash, consistent with `__eq__`."""
        ...

    # ================================================================
    #  Structure
    # ================================================================

    @property
    def factors(self) -> tuple[Mesh, ...]:
        """The Mesh factor objects, in constructor order."""
        ...

    @property
    def names(self) -> tuple[str, ...]:
        """All coordinate names, collected from the meshes in order."""
        ...

    # ================================================================
    #  Operator dispatch
    # ================================================================

    @property
    def dispatch(self) -> OperatorRegistry:
        """The operator dispatch registry (defaults + merged overrides)."""
        ...

    def merge_overrides(
        self,
        overrides: Mapping[DispatchKey, Operator],
    ) -> None:
        """Merge module-local dispatch overrides (pre-freeze only)."""
        ...

    # ================================================================
    #  Decomposition and lifecycle
    # ================================================================

    @property
    def decomposition(self) -> Decomposition:
        """The negotiated domain decomposition (grid-owned)."""
        ...

    def negotiate(
        self,
        *,
        state_spaces: tuple[TensorProductSpace, ...] | None = None,
        tendency: Callable[..., object] | None = None,
        halo: HaloSpec | None = None,
    ) -> ReshardingReport:
        """Renegotiate the decomposition (pre-freeze only)."""
        ...

    def freeze(self) -> None:
        """End the assembly phase; further merges/negotiations raise."""
        ...

    def sync(
        self,
        field: fr.ScalarField | fr.VectorField,
        boundary_data: Mapping[str, fr.ScalarField] | None = None,
    ) -> fr.ScalarField | fr.VectorField:
        """Fill halos (wrap / BC-structured fill / ghost_fill per axis)."""
        ...

    # ================================================================
    #  Field factory
    # ================================================================

    def create_field(
        self,
        space: TensorProductSpace | FunctionSpace | None = None,
        *,
        init: Callable[..., jax.Array] | None = None,
        init_coeff: Callable[..., jax.Array] | None = None,
        data: jax.Array | None = None,
        name: str | None = None,
        units: str | None = None,
        metadata: FieldMetadata | None = None,
    ) -> fr.ScalarField:
        """Create a field on `space` (default: all-Center nodal product)."""
        ...

    @property
    def random(self) -> RandomFieldFactory:
        """Seeded, sharding-consistent random field generators."""
        ...

    # ================================================================
    #  Coordinate and metric accessors
    # ================================================================

    def evaluation_nodes(
        self,
        space: TensorProductSpace | FunctionSpace,
        name: str | None = None,
    ) -> fr.ScalarField:
        """Physical coordinates of the space's evaluation nodes."""
        ...

    def wavenumbers(
        self,
        space: TensorProductSpace | FunctionSpace,
        name: str | None = None,
    ) -> fr.ScalarField:
        """Wavenumbers (or mode indices) of a coefficient space."""
        ...

    def measure(
        self,
        space: TensorProductSpace | FunctionSpace,
        name: str | None = None,
    ) -> fr.ScalarField:
        """The metric measure (dx) proper to the space's node set."""
        ...

    def metric(
        self,
        space: TensorProductSpace | FunctionSpace,
        name: str,
        *,
        params: Mapping[str, fr.ScalarField] | None = None,
    ) -> fr.ScalarField:
        """A named mapping metric on the requested space (designed-for)."""
        ...

    # ================================================================
    #  Attachments
    # ================================================================

    @property
    def immersed(self) -> ImmersedDomain | None:
        """The immersed (masked) domain descriptor, or None."""
        ...

    @property
    def mapping(self) -> CoordinateMapping | None:
        """The coordinate-mapping descriptor, or None."""
        ...

    def with_immersed(self, immersed: ImmersedDomain) -> Grid:
        """A new grid with the immersed descriptor (pre-freeze only)."""
        ...
```

Semantics and invariants:

- **Constructor.** `meshes` follows sketch
  [4.4](../03_api_sketches.md#44-mixed-grid-uniform-fv-x-chebyshev-galerkin)
  (modulo the removed `names=`): pre-built meshes of any type (sphere,
  unstructured included). This is one of **two constructors, never one
  overloaded signature**
  ([§2.6](../01_concepts.md#26-grid--the-assembly-object)); the
  `shape=`/`extent=` form is the cartesian subclass below. `defaults=`
  seeds the dispatch registry (`DispatchKey` is doc 03's key type,
  `str | tuple[str, SpaceLike]`); `mapping=`/`immersed=` attach the
  static descriptors (see their classes). `device_ids` restricts the
  device set (coupled runs), as today.
- **Coordinate names come from the meshes.** Names are mandatory at
  **mesh construction** (doc 01; `bind_names` is deleted), so the grid
  takes no `names=`: `__init__` collects `mesh.names` from the factors
  in order and validates flat uniqueness, raising on duplicates (the
  tensor product's flat namespace,
  [§2.3](../01_concepts.md#23-tensorproductspace-and-named-coordinates)).
  A mesh's names are fixed for its lifetime, so reusing one mesh in
  two grids is well-defined — always under the same names; safety
  against mixing fields of the two grids comes from doc 02's
  `GridMismatchError`, not from name bookkeeping.
- **Registry placement and lifetime.** The `OperatorRegistry` *class*
  is doc 03's (module `grid2/operators/registry.py`); the grid owns
  the single **instance**, exposed as `grid.dispatch` (matching the
  module-side `self.dispatch` override dicts of sketch
  [4.2](../03_api_sketches.md#42-custom-operator-module-local-override)).
  `grid.merge_overrides` is a **facade** over the pure registry: it
  calls `OperatorRegistry.merge(overrides)` (which returns a new
  registry, doc 03) and swaps the held instance — the successor of the
  removed `diff_module`/`interp_module` slots. Precedence:
  `(kind, space)` entry > kind-only entry > grid default
  ([§3.4](../02_rules.md#34-generic-operator-dispatch)). Seeding order
  and the freeze discipline are in the **Grid lifecycle** subsection
  below. A registry that holds grid-bound operator instances (resolved
  transforms) is **grid-private and never reusable across grids**. The
  exact call site of the merge is the open question tied to the
  Phase 2 composition design (§3.4) — this class specifies the
  mechanism only.
- **Field factory.** `create_field` is the **single factory**
  ([§3.10](../02_rules.md#310-discretizing-continuous-functions)):
  `space` is positional and optional, defaulting to the all-Center
  nodal product; `init=` resolves the `("discretize", space)` entry
  (a `Discretizer`, below), `init_coeff=` the `("assign_coeff",
  space)` entry; `data=` is the direct-array companion; with none of
  the three the field is zeros (sketch
  [4.3](../03_api_sketches.md#43-transform-round-trip-with-per-origin-coefficient-spaces)).
  Construction is pure and traceable — legal inside the jit loop.
  There is no `Field.from_function`, no `f.set(...)`, and no lazy
  field (rejected alternatives, §3.10). `name`/`units` are sugar for
  the two common metadata slots; `metadata=` (doc 02's
  `FieldMetadata`) sets the full record and is mutually exclusive
  with the sugar.
- **Factory validation (normative table).**
  - `init` / `init_coeff` / `data` are pairwise exclusive
    (`ValueError`).
  - `init=` callables must name **exactly** the non-`ConstantSpace`
    coordinate names of the space (checked via `inspect.signature`;
    `TypeError` otherwise); likewise `init_coeff=` against the
    wavenumber / mode-index names.
  - `data=` accepts a **global true-shape** array (sharded by the
    factory via `device_put`) or an **already correctly sharded**
    array (accepted as-is); any other shape raises `ValueError`.
    Storage padding and halo are applied below the factory
    (`decomposition.pad` + `grid.sync`), per the halo/storage
    contract in section 3.
  - dtype is coerced to the space-derived dtype (§3.1/§3.2); only a
    complex-to-real demotion is an error.
  - spaces whose factors are neither grid factors nor adopted
    refinements/boundaries of them raise `GridMismatchError`.
- **Refined-mesh adoption (normative).** Spaces on `mesh.refined(...)`
  results of grid factors (doc 01's `refined_from` link) are
  **adopted**: `create_field` validation accepts them, registry
  seeding extends to them (the same operator instances as the parent
  rows), and negotiation derives their traits and layout from the
  parent factor (same strategy, `min_local_size` scaled by the
  refinement factor). This is what makes the padded-transform
  codomains of dealiasing (§3.12, doc 03 transforms) ordinary spaces
  on the grid.
- **Coordinate accessors.** `evaluation_nodes` / `wavenumbers` are the
  only coordinate surface; the free-floating `grid.coordinates()` and
  the old coordinate-array `get_mesh()` are removed (§3.10, §2.1). The
  `space` argument is deliberately mandatory — no no-argument default
  form. `name` selects the coordinate when the space contributes more
  than one non-`ConstantSpace` name (product spaces, 2D meshes); it may
  be omitted exactly when unambiguous. The result is a `ScalarField`
  **tagged with the querying space**, with every factor not carrying
  `name` replaced by its `ConstantSpace`, so it broadcasts exactly
  under the strict algebra
  ([§2.7](../01_concepts.md#27-where-coordinate-data-lives), §3.3).
  On coefficient spaces whose basis is not wavenumber-indexed,
  `wavenumbers` returns the space's intrinsic mode indices (§3.10).
- **Recompute-on-demand, no persistent array state.** All accessor
  results (nodes, wavenumbers, measures, immersed masks, metrics) are
  recomputed to the **local shard matching field layout** from static
  descriptors via sharded `linspace`/`fftfreq`-style materialization;
  the N-D meshgrid is materialized only transiently inside
  `discretize`, never stored (§2.7). With the fully static grid this
  is not an optimization stance but the **only** mode: the grid holds
  no arrays at all, materialization happens at trace time, and XLA
  constant-folds what is constant. Consequently a renegotiated
  decomposition can never leave a stale shard. XLA CSE / loop-invariant
  code motion is the cost baseline; for a large mapped node array the
  **opt-in performance knob** is materialize-once-outside-scan and
  close over the array — invisible to the abstraction, bound by the
  same re-decomposition-invalidation rule, never a semantic change
  (§2.7; mapped-mesh materialization carries a compile-time benchmark
  item).
- **Measures.** `measure(space)` returns the staggered `dx` field
  proper to the space: the primal **cell width** on `center`/
  `cell_avg` (the FV integration weight), the dual **center-to-center
  spacing** on `right`/`outer`/`face_avg` (the `diff` denominator) —
  one accessor, the space decides which measure it is (§2.7, §3.9).
  On a uniform mesh both collapse to a constant field that XLA
  constant-folds. `name` disambiguates on products, as for
  `evaluation_nodes`; composed volume measures are left to operators
  (weights compose per mesh,
  [§3.13](../02_rules.md#313-reductions-and-integrals)).
- **Negotiation and sync.** The lifecycle subsection below is
  normative for `negotiate`/`freeze`. `grid.sync` is the field-level
  halo-exchange surface: it resolves per-axis fill modes — periodic
  wrap, the **BC-structured homogeneous fill** of the space (see the
  halo/storage contract in section 3), or the designed-for
  `("ghost_fill", space)` dispatch fed by `boundary_data=` — and
  delegates raw-array work to the decomposition. Operators obtain
  halo-extended storage through the grid; nothing above the grid
  touches `jax.sharding` directly.
- **Pytree treatment.** The grid participates in jit cache keys only
  as identity-hashed static aux data on fields. It holds **no**
  dynamic leaves — the earlier dynamic-attachment design is
  **reversed** (a grid-held `ScalarField` leaf creates the pytree
  cycle field -> grid -> fraction field -> grid, duplicates the leaf
  into every field's flatten, and an attachment swapped inside a
  traced step would execute once at trace time and freeze). Every
  array the grid hands out is created on demand and owned by the
  caller; time-dependent geometry is module-owned state threaded
  through the explicit-data accessor overloads (`immersed.mask(space,
  fraction=...)`, `grid.metric(space, name, params=...)`). This keeps
  "structure static, arrays dynamic" exact (§2.2, §2.7).

### Grid lifecycle (normative)

The construction/assembly/freeze sequence, fixing the initialization
order and the mutation windows. It resolves the constructor cycle
(`__init__` -> `Fourier(grid)` -> `grid.decomposition` ->
`negotiate(registry)`), makes the provisional halo sound, and defines
when renegotiation is legal.

1. **`__init__` order:** validate mesh names (flat uniqueness) ->
   intern the factors' space families -> seed the registry with
   **grid-free** entries (stencil operators, discretizers). Rows for
   the `("transform", ...)` kind are seeded as **lazy factories**: the
   registry stores a callable, and the grid-bound transform instance
   (`fr.operators.Fourier(grid, axes=...)`) is constructed on first
   resolve — necessarily post-negotiation. This breaks the cycle:
   nothing grid-bound exists while the registry is being seeded.
2. **`__init__` ends with a provisional negotiation:**
   `negotiate(state_spaces=None, tendency=None)` with halo = the
   per-operator maximum over the default registry. This is **sound
   under the iteration-1 contract** that every operator application
   returns a synced field (halo/storage contract, section 3): chains
   never accumulate, so the single-operator maximum is exact. A grid
   is therefore fully usable interactively right after construction
   (sketch 4.1 without any model).
3. **Phase-2 assembly** then runs `merge_overrides(...)` ->
   `negotiate(state_spaces=..., tendency=...)` -> `freeze()`. After
   `freeze()`, `merge_overrides`, `negotiate`, and `with_immersed`
   raise `RuntimeError`. Renegotiation after **any** jit trace has
   consumed the grid is an error regardless of freeze state — traced
   computations have baked the old layouts in. `negotiate` returns a
   `ReshardingReport`; the model walks its state once and
   `device_put`s each field to `decomposition.sharding(space)` — live
   fields are re-homed exactly once, derived arrays need nothing
   (recompute-on-demand).
4. **Registry reuse:** once any lazy transform row has been resolved,
   the registry holds grid-bound instances and is **grid-private**;
   sharing a registry object between grids is an error.

### grid2.cartesian.Grid

Convenience subclass building uniform `IntervalMesh` factors from
`shape=`/`extent=`/`periodic=` — the day-one constructor
([§10.1](../07_iteration1_api.md#101-building-a-grid)).

- Kind: final concrete subclass of `Grid`.
- Module: `fridom.framework.grid2.cartesian.grid`.
- Pytree: as base (adds nothing).
- Iteration: 1 (it is *the* iteration-1 public constructor).
- Concept refs: [§2.6](../01_concepts.md#26-grid--the-assembly-object),
  sketch [4.1](../03_api_sketches.md#41-uniform-tensor-grid-staggered-derivative).

```python
class Grid(fr.grid2.Grid):
    """Cartesian convenience grid: uniform IntervalMesh factors."""

    def __init__(
        self,
        shape: tuple[int, ...],
        extent: tuple[tuple[float, float], ...],
        periodic: bool | tuple[bool, ...] = True,
        names: tuple[str, ...] | None = None,
        *,
        defaults: Mapping[DispatchKey, Operator] | None = None,
        mapping: CoordinateMapping | None = None,
        immersed: ImmersedDomain | None = None,
        device_ids: tuple[int, ...] | None = None,
    ) -> None:
        """Build uniform IntervalMesh factors and assemble the grid."""
        ...
```

Notes:

- Builds one `fr.meshes.IntervalMesh(shape=n, extent=(a, b),
  periodic=p, names=(name,))` per axis and delegates to the base
  constructor — **no new methods or properties**; the two constructors
  stay split across base and subclass so neither signature silently
  accepts the other's kwarg set (§2.6). No `N`/`L` aliases.
- The subclass keeps `names=` precisely because it *constructs* the
  meshes (names are mandatory at mesh construction); the base root
  takes none. `periodic` broadcasts a single bool to all axes;
  `names` defaults to `("x", "y", "z")[:ndim]` for `ndim <= 3` and is
  required otherwise.
- The keyword-only arguments are forwarded verbatim to
  `fr.grid2.Grid`.

### Discretizer

The operator shape behind the reserved kinds `("discretize", space)`
and `("assign_coeff", space)`: what `create_field` resolves and calls
for `init=` / `init_coeff=`.

- Kind: ABC (small protocol-style base).
- Module: `fridom.framework.grid2.discretize`.
- Pytree: static (stateless strategy objects).
- Iteration: 1 (both default implementations).
- Concept refs:
  [§3.10](../02_rules.md#310-discretizing-continuous-functions),
  [§3.4](../02_rules.md#34-generic-operator-dispatch).

```python
class Discretizer(ABC):
    """Projection of a callable into a space (registry-resolved)."""

    @abstractmethod
    def discretize(
        self,
        grid: Grid,
        space: TensorProductSpace,
        fn: Callable[..., jax.Array],
    ) -> jax.Array:
        """Evaluate `fn` into `space`; return the local storage array."""
        ...
```

Notes:

- Registered like any operator, so the projection is swappable per
  space (§3.10: the true L2/Galerkin projection is "a different,
  registrable operator" — it is a different `Discretizer`).
- Iteration-1 defaults: a collocation discretizer for
  `("discretize", space)` — broadcast the per-factor
  `grid.evaluation_nodes(space)` transiently, keyword-match `fn`, and
  sample; for coefficient spaces it composes with the forward
  transform (`discretize = transform o discretize_origin`, §3.10) —
  and a coefficient assigner for `("assign_coeff", space)` evaluating
  `fn` at `grid.wavenumbers(space)`.
- Returns the raw local array; `create_field` owns field assembly,
  padding, and sync (halo/storage contract, section 3).

### RandomFieldFactory

The `grid.random` accessor: seeded, sharding-consistent random field
generators ([§3.10](../02_rules.md#310-discretizing-continuous-functions)).

- Kind: final concrete.
- Module: `fridom.framework.grid2.random_fields`.
- Pytree: fully static (holds only the grid reference); not a pytree —
  reached only through the static grid.
- Iteration: 1 (`normal`, `phase`; the spectra-IC consumer of `phase`
  is designed-for, the method itself is not).
- Concept refs: §3.10,
  [§5](../04_decomposition.md#5-domain-decomposition), sketch
  [4.9](../03_api_sketches.md#49-random-spectra-initial-condition-spectral-space-construction).

```python
class RandomFieldFactory:
    """Seeded random fields, deterministic across device layouts."""

    def __init__(self, grid: Grid) -> None:
        """Bind the factory to its grid (created by the grid itself)."""
        ...

    def normal(
        self,
        space: TensorProductSpace | FunctionSpace,
        seed: int,
    ) -> fr.ScalarField:
        """Standard-normal field on `space` (complex normal if Complex)."""
        ...

    def phase(
        self,
        space: TensorProductSpace | FunctionSpace,
        seed: int,
    ) -> fr.ScalarField:
        """Unit-modulus field e^{i theta}, theta uniform per DOF."""
        ...
```

Notes:

- Successor of `grid.create_random_array`. The values are a pure
  function of `(space.shape, seed)` over the **global true-DOF index**;
  only the sharding is applied by the grid. Realization: per-shard
  counter-based keying, `fold_in(seed, global_index)` for local DOFs
  only, drawing directly into the shard (`decomposition.local_slice`
  supplies the global index range) — no global materialization,
  layout-independent by construction, superseding the predecessor's
  draw-global-then-slice (§3.10). **Cost note:** per-DOF `fold_in`
  (vmapped fold-in plus one-sample draws) is several times slower
  than a single block draw; layout independence *requires* the
  per-DOF keying, so the block draw is not an option — benchmark
  item: per-DOF keying vs draw-global-then-slice, to quantify what
  determinism costs at IC-construction time.
- The draw covers the space's **true shape** (§3.5), so pad slots are
  outside the index space and random values never land in padding —
  no special rule needed
  ([§5](../04_decomposition.md#5-domain-decomposition)).
- On a real-origin Fourier space the draw covers only the free
  half-spectrum, and the **self-conjugate modes** (`k = 0` and, at
  even lengths, Nyquist) are drawn specially: `normal` draws them
  **real with unit variance**, `phase` draws a **uniform sign ±1**
  there (the unit-modulus reals). The Hermitian structure follows
  from the shape (§3.10/§3.2). `normal` on a `fr.Complex` space draws
  a complex normal (independent real/imag parts).
- **Variance convention (explicit):** draws are **white in
  coefficients** — unit variance per coefficient DOF — *not* unit
  variance of the physical-space field. The two differ by the
  transform normalization and by the sqrt(2) bookkeeping between a
  complex mode and its two real DOFs; spectra-based ICs (sketch 4.9)
  must apply their amplitude on top of the coefficient-white
  convention, or spectral slopes come out wrong.
- Pure and traceable under jit; returns ordinary `(space, array)`
  fields. `space` is mandatory (there is no obvious default and the
  accessors' explicit-space discipline applies). Extension contract:
  additional draws (e.g. uniform) must use the same per-shard
  global-index keying and the same self-conjugate-mode handling;
  anything else silently breaks determinism across device counts or
  reality constraints.

### ImmersedDomain

Grid-owned successor of `WaterMask`: a **static descriptor** of the
wet region plus derive-on-demand per-space masks/fractions
([§3.7](../02_rules.md#37-boundaries-ii-immersed-masked-domains)).

- Kind: final concrete.
- Module: `fridom.framework.grid2.immersed_domain`.
- Pytree: **fully static** — holds the init callable / static
  parameters only, no arrays and no `ScalarField`s. Fractions and
  masks are materialized on demand at trace time, exactly like
  `evaluation_nodes` (G1: the earlier stored-fraction-leaf design is
  reversed).
- Iteration: 1 for the boolean subset (fraction in `{0, 1}`,
  `WaterMask` parity — fidelity-ladder point 1); cut-cell fractions,
  transition sets, ghost-fill and level-set generalizations are
  designed-for. A day-one user does not type it
  ([§10.6](../07_iteration1_api.md#106-deferred-to-later-iterations-not-typed-by-a-day-one-user)).
- Concept refs: §3.7,
  [§2.6](../01_concepts.md#26-grid--the-assembly-object),
  [§2.7](../01_concepts.md#27-where-coordinate-data-lives).

```python
class Slip(Enum):
    """Immersed-boundary structure: staggered-mask combination rule."""

    NO_SLIP = auto()
    FREE_SLIP = auto()


class ImmersedDomain:
    """Static wet-region descriptor; per-space masks derived on demand."""

    def __init__(
        self,
        init: Callable[..., jax.Array],
        *,
        slip: Slip = Slip.NO_SLIP,
    ) -> None:
        """Declare the wet fraction as a function of physical coords."""
        ...

    @property
    def slip(self) -> Slip:
        """The staggered-mask derivation rule (no-slip / free-slip)."""
        ...

    def fraction(
        self,
        space: TensorProductSpace | FunctionSpace,
        *,
        fraction: fr.ScalarField | None = None,
    ) -> fr.ScalarField:
        """Wet fraction on `space` (volume or area weights)."""
        ...

    def mask(
        self,
        space: TensorProductSpace | FunctionSpace,
        *,
        slip: Slip | None = None,
        fraction: fr.ScalarField | None = None,
    ) -> fr.ScalarField:
        """Boolean wet mask on `space`, derived by the slip rule."""
        ...

    def transition(
        self,
        space: TensorProductSpace | FunctionSpace,
        *,
        fraction: fr.ScalarField | None = None,
    ) -> fr.ScalarField:
        """Wet/dry transition-set indicator on `space` (designed-for)."""
        ...
```

Notes:

- **Single declared datum**: the wet volume fraction in `[0, 1]`,
  declared as a callable of physical coordinates and materialized on
  demand onto the **`cell_avg` space** — the fraction *is* a volume
  fraction, a functional over the cell, so the average space is its
  honest home (a `center` sample is the collocation approximation of
  it). The boolean mask is the `{0, 1}` special case, so there is one
  representation, not two; a level-set declaration is the future
  generalization (§3.7).
- **No stored arrays.** The descriptor is bound to its grid at grid
  construction (`immersed=` kwarg) or by the pre-freeze functional
  update `grid.with_immersed(...)`; every `fraction`/`mask`/
  `transition` call materializes to the local shard at trace time,
  and XLA folds the constant result (§2.7). There is no
  binding-time discretization and nothing to reshard on
  renegotiation.
- **Time-dependent geometry is module-owned state.** A moving
  boundary is a prognostic fraction field registered as module state
  (Phase 2) and threaded through the **explicit-data overloads**:
  `immersed.mask(space, fraction=field)` derives the per-space mask
  from the supplied field instead of the static declaration. No
  attachment swapping, no grid-held leaves — module updates are
  ordinary state threading.
- **Derived quantities are grid-mediated and derive-on-demand**,
  mirroring `grid.evaluation_nodes(space)`: `fraction(space)`,
  `mask(space)`, `transition(space)` return fields *tagged with that
  space*, computed from the base fraction by a staggering-transfer
  rule. **Structure is the combination rule**: no-slip is today's
  `face = AND(adjacent centers)`; `slip` is a parameter of the
  derivation, not stored per-space state (the constructor value is
  the default, the `mask(..., slip=...)` override serves
  mixed-physics diagnostics). Fraction transfer at ladder point 2 is
  geometric (area/volume weights), independent of slip. Memoizing
  derived masks is a below-the-operator-layer optimization to
  benchmark, bound by the re-decomposition-invalidation rule; the
  interface is derive-on-demand regardless (§3.7).
- **Masked-operator wrapping contract.** Mask-awareness adds **no
  registry axis** and no wrapper grid type: mask-aware operators are
  ordinary dispatch entries that *consult* `grid.immersed` (fractions
  as weights in `integrate`/`flux`/`reconstruct`), with the same
  grid-materialized-array status as `dx` (§3.7). The ROADMAP 4.4
  phrasing `MaskedGrid(inner_grid)` is superseded by the notes: the
  immersed domain is an *attachment* at `grid.immersed`, not a
  decorator grid — a wrapper would fork the grid identity that spaces,
  fields, and jit keys hang off.
- Masking correctness is not type-checked (masked and unmasked fields
  share a space); it is owned by operators and modules, as today
  (recorded exception, §3.7).

### CoordinateMapping

Grid-attached **static descriptor** of terrain-following / curvilinear
coordinate maps; single owner of the metric *derivation*
([§3.8](../02_rules.md#38-boundaries-iii-terrain-following-boundary-fitted)).

- Kind: final concrete.
- Module: `fridom.framework.grid2.coordinate_mapping`.
- Pytree: **fully static** — map callables and static parameters
  only; no arrays, no `ScalarField`s, no dynamic leaves (G1).
- Iteration: designed-for (nothing in iteration 1 may assume static
  metrics, [§6.5](../05_validation.md#65-terrain-following-vertical-coordinate)).
- Concept refs: §3.8,
  [§2.3](../01_concepts.md#23-tensorproductspace-and-named-coordinates),
  §6.5.

```python
class CoordinateMapping:
    """Static coordinate-transform declaration; metrics on demand."""

    def __init__(
        self,
        maps: Mapping[str, Callable[..., jax.Array]] | None = None,
        *,
        metrics: Mapping[str, Callable[..., jax.Array]] | None = None,
        params: Mapping[str, Callable[..., jax.Array]] | None = None,
    ) -> None:
        """Declare an analytic map (or supplied metrics) and parameters."""
        ...

    @property
    def param_names(self) -> tuple[str, ...]:
        """The named parameters of the map (H, eta, ...)."""
        ...

    @property
    def metric_names(self) -> tuple[str, ...]:
        """The metric names this mapping can supply."""
        ...

    def metric(
        self,
        space: TensorProductSpace | FunctionSpace,
        name: str,
        *,
        params: Mapping[str, fr.ScalarField] | None = None,
    ) -> fr.ScalarField:
        """Derive the named metric on the requested staggered space."""
        ...
```

Notes:

- **Both declaration forms** (§3.8): an *analytic* map per mapped
  coordinate as a function of base coordinates and named parameter
  fields (`maps={"z": lambda sigma, H: sigma * H}`), from which the
  grid derives metric fields (`H_x`, `dz/dsigma`, Jacobians) by
  differentiation through registry operators; **or** user-supplied
  metric callables directly (`metrics=`) for cases with no closed
  form. The descriptor is attached at grid build (`mapping=` kwarg);
  `params` declares the *static* defaults as callables of physical
  coordinates, materialized on demand like everything else — nothing
  is discretized or stored at bind time.
- `grid.metric(space, name, params=...)` delegates here (argument
  order aligned with `grid.measure(space, name=...)`); the accessor
  works uniformly for both forms (supplied metrics are reconstructed
  to the requested space), so **staggered consistency is guaranteed by
  the grid, not per module** — H at u-, v-, w-points comes from one
  owner (§3.8, §6.5).
- **Time-dependent geometry is module-owned state.** Prognostic
  parameters (z*, `H = H_0 + eta(t)`) are state fields registered by
  the owning module (Phase 2) and passed through the explicit-data
  overload: `grid.metric(space, "dz_dsigma", params={"H": h_field})`
  derives the metric from the supplied fields instead of the static
  defaults. Derived metrics are recomputed from the passed values at
  every query — no operator may cache them (§2.3, §3.8) — and module
  updates are ordinary state threading, traced like any other field
  arithmetic.
- Metric-coefficient operator composites (constant-z vs constant-sigma
  derivative kinds) are dispatch entries reading `grid.metric` — doc
  03 territory; this class only owns the declaration and its per-space
  derivation. A mesh-local 1D mapping (stretched vertical) stays mesh
  structure (`MappedIntervalMesh`, doc 01); only **cross-factor**
  mappings live here (§2.1).

---

## 3. Domain decomposition

Design summary, from
[`04_decomposition.md`](../04_decomposition.md) (normative): meshes
declare shardability traits; operators declare per-axis halo /
transform requirements; grid setup collects demands x traits and
chooses the layout(s). There is **no single global halo integer** —
halo is a per-mesh quantity derived by an automatic halo-accounting
trace over the tendency. Staggered/padded storage is owned by this
layer and invisible above it. Fields and operators reach the
decomposition only through the grid, and the transform surface is rich
enough that solvers no longer bypass it (the `RFFTPressureSolver`
lesson).

The class structure: small frozen descriptors (`HaloStrategy`,
`MeshDecompositionTraits`, `HaloSpec`, `ArrayLayout`,
`ReshardingReport`), one negotiation entry point (`negotiate`), one
ABC (`Decomposition`) with the jax-sharding tensor backend
(`TensorDecomposition`, iteration 1) and a named designed-for graph
backend (`GraphDecomposition`), plus the halo-accounting tracer
(`HaloTracer`, `trace_halo`) and the designed-for `GhostFill`
protocol.

### Halo and storage contract (normative)

Jointly owned with doc 02 (field storage) and doc 03 (the operator
base); stated here because the decomposition defines the shapes.

- **Storage vs data.** `ScalarField._data` is **storage-shaped**:
  halo plus stagger padding per the negotiated layout
  (`decomposition.storage_shape(space)`). `.data` is the **true-shape
  view** (pads and halo sliced off). `create_field` and
  `field.with_data` accept true-shape arrays and route them through
  `decomposition.pad` + `grid.sync`; nothing above the factory ever
  constructs storage-shaped arrays.
- **Iteration-1 sync contract.** Operator inputs have valid halos,
  and **every operator application returns a synced field**: the
  operator *base* (doc 03) calls the sync after the kernel — kernel
  authors never do. Under this contract un-synced chains do not
  exist, so the per-operator halo maximum is exact and the
  `HaloTracer` sizes the maximal **single-chain** ghost width.
  **Sync-elision** along traced chains — skipping intermediate
  exchanges and letting depth accumulate, as the accounting semantics
  of [§5](../04_decomposition.md#5-domain-decomposition) permit — is
  the designed-for optimization this contract deliberately leaves on
  the table.
- **Halo-0 paths skip sync structurally.** Symbol application,
  transforms, `Hadamard`, and anything else whose per-axis halo is
  zero performs no exchange — not as an optimization but because the
  width-0 `HaloSpec` makes the sync a no-op by construction.
- **Kernel execution.** Stencil kernels execute per-shard under a
  decomposition-supplied `shard_map`; pad, sync, and kernel form
  **one shard-local region per operator application** (the pattern of
  today's `stencil_view.py`). The shard boundary is owned by the
  operator base plus `Decomposition` and is invisible to kernel
  authors, who write slice-based true-shape stencils (§3.5).
- **Bounded-edge fill is keyed to the space's BC structure.** On
  periodic meshes the halo is filled by wrap-around. On bounded
  meshes the ghost layer is filled per the space's `BCStructure`:
  Dirichlet-structured spaces get the **odd (zero-value) extension**,
  Neumann-structured spaces the **even (mirror) extension**, and
  BC-free spaces (`outer`) a **one-sided extrapolation** consistent
  with the resolved operator's order. Blanket zero-fill is
  **rejected**: it is an undeclared Dirichlet choice that silently
  turns the default bounded Laplacian into a homogeneous-Neumann
  lookalike and corrupts `reconstruct : cell_avg -> outer` at
  boundary faces. Reliance map for the doc 03 default rows: bounded
  `("diff", ...)` / `("laplacian", ...)` stencils rely on the
  odd/even extension matching the space BC; `("reconstruct",
  cell_avg)` and `("interpolate", ...)` rely on the BC-consistent
  extension at boundary faces; coefficient-space rows are halo-0 and
  rely on no fill.
- **Boundary data.** Iteration 1 supports **homogeneous** conditions
  only: the physical-boundary ghost layer carries no user data, and
  iteration-1 kernels must not depend on it beyond the structured
  fill above. The designed-for inhomogeneous surface (§3.6) is:
  - `create_field` accepts products whose factor meshes are
    `m.boundary` of grid factors — trace fields;
  - `grid.sync(field, boundary_data=Mapping[str, ScalarField] |
    None)` threads module-owned trace fields into the exchange;
  - a `GhostFill` protocol, registered under the reserved
    `("ghost_fill", space)` kind and living beside the sync machinery
    in `decomposition/halo.py`:

  ```python
  class GhostFill(Protocol):
      """Space-keyed inhomogeneous ghost fill (designed-for)."""

      def fill(
          self,
          grid: Grid,
          space: TensorProductSpace,
          boundary_data: fr.ScalarField,
      ) -> Mapping[str, jax.Array]:
          """Per-name ghost arrays from a trace field."""
          ...
  ```

### HaloStrategy / MeshDecompositionTraits

The mesh decomposition traits: what a mesh *declares* about the
shardability of its spaces. This cluster owns the single definition of
both types; doc 01 owns the declaring seam on `Mesh` — the per-space
method `Mesh.decomposition_traits(space: FunctionSpace) ->
MeshDecompositionTraits` (per space, not per mesh: nodal and
coefficient spaces of one mesh differ).

- Kind: `HaloStrategy` enum; `MeshDecompositionTraits` final frozen
  dataclass.
- Module: `fridom.framework.grid2.decomposition.traits` (the single
  definition; doc 01 imports from here).
- Pytree: static (hashable descriptors).
- Iteration: 1 (`GHOST`, `TRANSPOSE`, `LOCAL`); `GRAPH` is the
  designed-for unstructured tag.
- Concept refs: [§5](../04_decomposition.md#5-domain-decomposition),
  [§2.1](../01_concepts.md#21-mesh--atomic-factor-of-the-domain),
  [§6.4](../05_validation.md#64-unstructured-horizontal-x-structured-vertical).

```python
class HaloStrategy(Enum):
    """How a factor space can be distributed across devices."""

    GHOST = auto()      # shard with ghost-cell halo exchange
    TRANSPOSE = auto()  # shard via transpose-based transforms
    LOCAL = auto()      # keep this factor on-device
    GRAPH = auto()      # graph partition (designed-for)


@dataclass(frozen=True)
class MeshDecompositionTraits:
    """Shardability declaration for one factor space of a mesh."""

    strategies: tuple[HaloStrategy, ...]
    min_local_size: int = 1
```

Notes:

- `strategies` is ordered by preference, and the declaration is
  **per space**: the nodal spaces of a uniform `IntervalMesh` declare
  `(GHOST, TRANSPOSE)`; its Fourier coefficient spaces declare
  `(LOCAL, TRANSPOSE)` (local-only or distributed FFT, §5); Chebyshev
  meshes declare **`(TRANSPOSE, LOCAL)`** — their recurrences couple
  the whole column, but that demands a *contiguous dimension at
  operator time*, not an unsharded factor: column operators reach a
  pencil layout via transpose, so a tensor product of Chebyshev
  meshes still decomposes. The spaces of an unstructured mesh declare
  `(GRAPH,)`. On `uniform(x, y) ⊗ chebyshev(z)` negotiation still
  picks the cheap outcome — shard x/y with ghosts, keep z local in
  the default layout (§5, §6.2) — but nothing forces it.
- `LOCAL` parallels `OperatorRequirements.layout = "local"` (doc 03).
  There is no separate shardable flag anywhere: doc 01 has dropped
  `mesh.shardable` / `mesh.halo_strategy`, and shardability is
  derivable as `traits.strategies != (LOCAL,)`.
- `min_local_size` guards against shards smaller than a halo (the
  current `local_shape < halo` failure, made a negotiation constraint
  instead of a runtime error).
- Seam: the operator-side counterpart is doc 03's
  `op.requirements(domain) -> OperatorRequirements`, whose `.halo`
  (per-axis widths) and `.layout` (`"local"` vs shardable) fields the
  negotiation below consumes together with the per-space mesh traits
  ([§2.5](../01_concepts.md#25-operator--typed-maps-between-spaces)).

### HaloSpec

Per-coordinate-name halo widths — the negotiated replacement of the
global halo integer.

- Kind: final frozen dataclass.
- Module: `fridom.framework.grid2.decomposition.halo`.
- Pytree: static.
- Iteration: 1.
- Concept refs: [§5](../04_decomposition.md#5-domain-decomposition),
  [§3.5](../02_rules.md#35-shape-is-a-property-of-the-space).

```python
@dataclass(frozen=True)
class HaloSpec:
    """Negotiated ghost-layer widths, one per coordinate name."""

    widths: tuple[tuple[str, int], ...]

    def __init__(self, widths: Mapping[str, int]) -> None:
        """Normalize the mapping to sorted tuple storage (hashable)."""
        ...

    @classmethod
    def zero(cls, names: tuple[str, ...]) -> HaloSpec:
        """A spec with width 0 on every name."""
        ...

    def __getitem__(self, name: str) -> int:
        """Width along `name`."""
        ...

    def grow(self, name: str, by: int) -> HaloSpec:
        """A new spec with `name` widened by `by` (composition chains)."""
        ...

    def merge_max(self, other: HaloSpec) -> HaloSpec:
        """Pointwise maximum of two specs (parallel tendency terms)."""
        ...
```

Notes:

- Keyed by coordinate name, not axis index: names are the stable
  addressing scheme of the flat product (§2.3), and `LOCAL` /
  coefficient factors simply carry width 0.
- Storage is a sorted `tuple[tuple[str, int], ...]`, not a `Mapping`
  field: frozen dataclasses used as jit-cache-key components must be
  hashable, and the constructor accepts any `Mapping[str, int]` and
  normalizes — the same pattern doc 02 uses for `nc_attrs`.
- `grow` and `merge_max` are the two accumulation rules of the
  halo-accounting trace: sequential un-synced applications *add*,
  parallel expression branches *max* (§5).

### HaloTracer / trace_halo

The shape/halo-only stand-in field and the automatic halo-accounting
trace over the tendency.

- Kind: `HaloTracer` final concrete; `trace_halo` module function.
- Module: `fridom.framework.grid2.decomposition.halo`.
- Pytree: not a pytree participant (setup-phase only, never enters
  jit).
- Iteration: 1 (replaces `Module.required_halo`).
- Concept refs: [§5](../04_decomposition.md#5-domain-decomposition),
  [§3.4](../02_rules.md#34-generic-operator-dispatch).

```python
class HaloTracer:
    """Data-free ScalarField stand-in carrying space + halo depth."""

    def __init__(
        self,
        function_space: TensorProductSpace,
        registry: OperatorRegistry,
        depth: HaloSpec | None = None,
    ) -> None:
        """Create a tracer on `function_space` with zero depth."""
        ...

    @property
    def function_space(self) -> TensorProductSpace:
        """The space this tracer pretends to live on."""
        ...

    @property
    def depth(self) -> HaloSpec:
        """Accumulated per-name halo depth since the last sync."""
        ...

    @property
    def data(self) -> NoReturn:
        """Raise TypeError: bypasses must declare Module.extra_halo."""
        ...

    # ScalarField-mimicking surface: arithmetic (`+`, `-`, `*`, ...),
    # `.diff`, `.to`, `.integrate` — every operator application
    # returns a new HaloTracer with grown depth and the operator's
    # codomain space; a sync resets depth to zero. trace_halo wraps
    # components in a VectorField/State-mimicking stand-in so
    # composed operators (Divergence, .map) trace too.


def trace_halo(
    tendency: Callable[..., object],
    state_spaces: tuple[TensorProductSpace, ...],
    registry: OperatorRegistry,
) -> HaloSpec:
    """Dry-run the tendency on tracers; return the max accumulated depth."""
    ...
```

Notes:

- A simple max over operators is **not** enough: halo accumulates
  along un-synced composition chains (`f.diff("x").diff("x")` needs
  `h1 + h2`); the trace is exact and author-effort-free (§5). Under
  the iteration-1 sync-after-every-operator contract (halo/storage
  contract above) chains have length one and the trace reproduces the
  per-operator max; its accumulation semantics are what the
  designed-for sync-elision consumes.
- Interception is **generic**: because the tracer presents the
  `ScalarField` interface, operators run unchanged. Concretely, the
  hook sits in the shared operator application path (doc 03's
  `Operator` base `__call__` / the dispatch shim behind `f.diff`):
  when the operand is a tracer, the operator's
  `requirements(domain).halo` is recorded and the codomain-space
  tracer returned, without touching kernel code. No per-operator
  tracer code, and **no halo bookkeeping on real fields** (field
  metadata stays name/units/nc-attrs, §2.4). The tracer carries the
  **registry reference** its mimicked `.diff`/`.to` surface needs for
  dispatch.
- **Mixed operands:** doc 02's `ScalarField` dunders return
  `NotImplemented` when the other operand is a tracer, so the
  tracer's reflected operations run and the trace survives
  `field + tracer` expressions.
- **`.data` raises** (`TypeError`): a module that drops to raw arrays
  escapes the accounting, so the escape must be *declared* — the
  Phase-2 module surface grows `Module.extra_halo: HaloSpec`, merged
  into the traced result. Strictness is deliberate; a warning would
  make the sizing silently wrong.
- **Vector stand-in:** `trace_halo` feeds the tendency a
  `VectorField`/`State`-mimicking wrapper whose components are
  tracers and which performs **no grid validation** — component
  mapping (`.map`) and composed operators like `Divergence` trace
  through it (the tracer must mimic both field kinds).
- **Transforms reset depth:** `layout_for`/`redistribute` is a global
  data movement, at least as strong as a sync on the moved axes — the
  trace rule is that a transform **resets the accumulated depth on
  its transformed axes** to zero.
- **Un-jitted tracing:** the tendency callable must be traceable
  **without** jit — jit rejects pytrees with unregistered tracer
  leaves. Phase 3's top-level-jit-only architecture makes this
  natural: the tendency is plain Python over fields.
- **Exactness and `("select", ...)`:** doc 03's where-kind keeps
  upwind sign-selection inside the operator layer, which is what
  makes the "exact and author-effort-free" claim true for the shipped
  advection modules — their branches are dispatch entries, not raw
  `jnp.where` on `.data`.
- The trace also yields sync placement information (where depth would
  exceed the chosen ghost width, a sync must be inserted); iteration 1
  sizes ghost layers from the maximum depth and keeps the
  sync-after-every-operator placement.
- Scope: the accounting runs over the registry **as merged**, i.e.
  after module overrides, restricted to operators that can actually
  fire on the model's state-field spaces (§5).

### ArrayLayout

A frozen descriptor of one concrete distribution: which coordinate
names are sharded across which device-mesh axes, with which halo.

- Kind: final frozen dataclass.
- Module: `fridom.framework.grid2.decomposition.layout`.
- Pytree: static (hashable; part of jit cache keys via the grid).
- Iteration: 1.
- Concept refs: [§5](../04_decomposition.md#5-domain-decomposition).

```python
@dataclass(frozen=True)
class ArrayLayout:
    """One assignment of coordinate names to device-mesh axes."""

    device_axes: tuple[tuple[str, str], ...]  # (coord name, device axis)
    halo: HaloSpec

    def __init__(
        self,
        device_axes: Mapping[str, str],
        halo: HaloSpec,
    ) -> None:
        """Normalize the mapping to sorted tuple storage (hashable)."""
        ...

    def is_local(self, name: str) -> bool:
        """Whether the factor carrying `name` is device-local here."""
        ...
```

Notes:

- Purely combinatorial: everything array-shaped (`PartitionSpec`s,
  local slices, storage shapes) is derived by the owning
  `Decomposition`, which holds the `jax.sharding.Mesh`. Layouts are
  values, so transforms can name their pencil schedule (`x-local`
  layout, `y-local` layout) as data.
- Tuple storage with a mapping-accepting constructor, for the same
  hashability reason as `HaloSpec.widths` (the doc 02 `nc_attrs`
  pattern).
- The default layout of a grid shards the preferred `GHOST` factors;
  transform-aware layouts (`TRANSPOSE` strategy) are the generalized
  successors of today's main/alt shardings in `JaxDecomposition`.

### Decomposition / negotiate

The ABC every backend implements, and the per-mesh negotiation entry
point. The grid owns exactly one `Decomposition`; fields and operators
reach it only through the grid.

- Kind: ABC (`abc.ABC`); `negotiate` module function.
- Module: `fridom.framework.grid2.decomposition.decomposition`.
- Pytree: static (structure only; no persistent array state).
- Iteration: 1 (ABC + negotiation; graph backend designed-for).
- Concept refs: [§5](../04_decomposition.md#5-domain-decomposition),
  [§2.7](../01_concepts.md#27-where-coordinate-data-lives),
  [§3.5](../02_rules.md#35-shape-is-a-property-of-the-space).

```python
def negotiate(
    grid: Grid,
    registry: OperatorRegistry,
    *,
    state_spaces: tuple[TensorProductSpace, ...] | None = None,
    tendency: Callable[..., object] | None = None,
    halo: HaloSpec | None = None,
    device_ids: tuple[int, ...] | None = None,
) -> Decomposition:
    """Choose a backend and layouts from mesh traits + operator demands."""
    ...


@dataclass(frozen=True)
class ReshardingReport:
    """What a renegotiation changed (returned by grid.negotiate)."""

    old: ArrayLayout
    new: ArrayLayout
    changed: bool


class Decomposition(ABC):
    """Distribution of a grid's DOFs across devices (grid-owned)."""

    @property
    def halo(self) -> HaloSpec:
        """The negotiated per-name ghost widths."""
        ...

    @property
    def default_layout(self) -> ArrayLayout:
        """The layout fields are created and stepped in."""
        ...

    @property
    def layouts(self) -> tuple[ArrayLayout, ...]:
        """All negotiated layouts (default + transform pencils)."""
        ...

    @abstractmethod
    def sharding(
        self,
        space: TensorProductSpace | FunctionSpace,
        layout: ArrayLayout | None = None,
    ) -> jax.sharding.Sharding:
        """The jax sharding of `space`'s storage under `layout`."""
        ...

    @abstractmethod
    def local_slice(
        self,
        space: TensorProductSpace | FunctionSpace,
        layout: ArrayLayout | None = None,
    ) -> tuple[slice, ...]:
        """Global true-DOF index range of the local shard."""
        ...

    @abstractmethod
    def storage_shape(
        self,
        space: TensorProductSpace | FunctionSpace,
        layout: ArrayLayout | None = None,
    ) -> tuple[int, ...]:
        """Global storage shape: true shape + halo + stagger padding."""
        ...

    @abstractmethod
    def zeros(
        self,
        space: TensorProductSpace | FunctionSpace,
        layout: ArrayLayout | None = None,
    ) -> jax.Array:
        """A zero-filled, sharded, storage-shaped array for `space`."""
        ...

    @abstractmethod
    def pad(
        self,
        arr: jax.Array,
        space: TensorProductSpace | FunctionSpace,
        layout: ArrayLayout | None = None,
    ) -> jax.Array:
        """True-shape local data -> halo/stagger-padded storage."""
        ...

    @abstractmethod
    def unpad(
        self,
        arr: jax.Array,
        space: TensorProductSpace | FunctionSpace,
        layout: ArrayLayout | None = None,
    ) -> jax.Array:
        """Padded storage -> true-shape local data (pads dropped)."""
        ...

    @abstractmethod
    def sync(
        self,
        arr: jax.Array,
        space: TensorProductSpace | FunctionSpace,
        *,
        layout: ArrayLayout | None = None,
        fills: Mapping[str, jax.Array] | None = None,
    ) -> jax.Array:
        """Exchange halos; bounded edges per `fills` / BC-structured."""
        ...

    @abstractmethod
    def layout_for(
        self,
        local_names: tuple[str, ...],
    ) -> ArrayLayout:
        """A negotiated layout in which the named factors are local."""
        ...

    @abstractmethod
    def redistribute(
        self,
        arr: jax.Array,
        space: TensorProductSpace | FunctionSpace,
        src: ArrayLayout,
        dst: ArrayLayout,
    ) -> jax.Array:
        """Transpose an array between two negotiated layouts."""
        ...

    @abstractmethod
    def gather(
        self,
        arr: jax.Array,
        space: TensorProductSpace | FunctionSpace,
        layout: ArrayLayout | None = None,
    ) -> jax.Array:
        """Gather the global true-shape array (I/O, diagnostics)."""
        ...
```

Notes:

- **Negotiation** collects, per factor space in play: the mesh's
  declared `mesh.decomposition_traits(space)` (doc 01 seam, per-space)
  and `op.requirements(domain)` — `.halo` and `.layout` — of every
  registry operator that can fire on the given `state_spaces` (doc 03
  seam). Halo comes from `trace_halo` when a `tendency` is supplied;
  otherwise from the per-operator maximum over the registry — the
  provisional-negotiation path of the Grid lifecycle, **sound under
  the iteration-1 sync-after-every-operator contract** (halo/storage
  contract above) — or from the explicit `halo=` override. The
  backend is chosen from the traits (`GHOST`/`TRANSPOSE`/`LOCAL` ->
  `TensorDecomposition`; any `GRAPH` factor -> `GraphDecomposition`).
  This replaces the old rebuild-on-halo-mismatch logic (§5);
  `grid.negotiate` re-runs it at assembly (pre-freeze only, returning
  the `ReshardingReport` the model uses to re-`device_put` its state
  once), and recompute-on-demand (§2.7) guarantees no derived array
  survives a renegotiation.
- **Solver/transform API (the bypass fix).** The lesson from
  `RFFTPressureSolver` (rfft, axis subsets, custom dct against the raw
  decomposition) becomes supported surface: a grid-bound transform
  (`fr.operators.Fourier(grid, axes=...)`, doc 03) plans its schedule
  as a sequence of (`layout_for(names)`, apply local 1D kernels,
  `redistribute`) steps — any per-axis kernel runs device-local in a
  pencil layout, and the decomposition contributes only layouts and
  transposes. **All coefficient spaces and transforms are iteration
  1**: rfft/fft, the DST/DCT family, and the Chebyshev transform are
  day-one consumers of this transpose machinery (the banded vertical
  solves of §6.2 use the same pencils). The old
  `parallel_forward_transform` wrapper pair disappears; nothing above
  the grid constructs shardings by hand.
- **`jax.sharding` use.** Backends express layouts as
  `jax.sharding.NamedSharding` over a `jax.make_mesh` device mesh;
  `redistribute` is a resharding `jax.device_put` (XLA lowers it to
  all-to-all); `sync` is a `shard_map` + `jax.lax.ppermute` halo
  exchange, as in today's `JaxDecomposition.sync`. The `Sharding`
  return type keeps single-device and multi-host cases uniform.
- **Storage padding** for unevenly sharding staggered pairs (n vs
  n + 1 along a bounded axis) lives in `storage_shape`/`pad`/`unpad`
  and is invisible above this layer; slice-based stencils always
  address the true logical extent, so pad slots never contribute
  (§3.5, §5). `local_slice` is expressed in **global true-DOF
  indices** — the anchor for the random factory's per-shard keying and
  for materializing coordinate shards.
- **Fill modes.** `sync`'s `fills` carries per-name ghost values for
  bounded axes: `None` means periodic wrap (periodic mesh) or the
  space's **BC-structured homogeneous fill** (bounded mesh — odd /
  even / one-sided per the halo/storage contract above, never blanket
  zeros); a supplied array is inhomogeneous ghost-fill data resolved
  by `grid.sync` through the designed-for `("ghost_fill", space)`
  entry (§3.5, §3.6). The decomposition itself never touches the
  registry — the grid resolves, the decomposition moves bytes.
- Reductions need no dedicated methods: operators `unpad` to true
  shape and use `jnp` reductions on sharded arrays under jit
  (§3.13); `gather` covers host-side I/O. **`ConstantSpace` factors
  are replicated in every layout**, and reductions produce
  replicated outputs — the collective sum's output sharding is the
  replicated one, which is exactly what the §3.3 broadcast needs.

### TensorDecomposition

The jax-sharding backend for tensor-product grids — the iteration-1
(and single-device) workhorse.

- Kind: final concrete (`Decomposition` subclass).
- Module: `fridom.framework.grid2.decomposition.tensor`.
- Pytree: static.
- Iteration: 1.
- Concept refs: [§5](../04_decomposition.md#5-domain-decomposition).

```python
class TensorDecomposition(Decomposition):
    """jax.sharding-based decomposition of tensor-product grids."""

    def __init__(
        self,
        meshes: tuple[Mesh, ...],
        names: tuple[str, ...],
        halo: HaloSpec,
        layouts: tuple[ArrayLayout, ...],
        device_ids: tuple[int, ...] | None = None,
    ) -> None:
        """Build the device mesh and shardings (called by negotiate)."""
        ...

    # implements every Decomposition abstract method; no extra
    # public surface — solver code programs against the ABC.
```

Notes:

- Constructed by `negotiate`, not by users. Single-device runs use the
  same class with a one-device mesh (no separate
  `SingleDecomposition`: `jax.sharding` degrades gracefully, and one
  code path means the multi-device tests cover the single-device
  semantics by construction). One sanctioned special case: `sync`
  **branches statically on `n_devices == 1`** and skips the
  `ppermute` self-loop entirely (a Python-level branch on static
  structure, measurable at FRIDOM's overhead-dominated problem
  sizes); trait and layout structure are unchanged by the
  short-circuit.
- Iteration 1 may realize a 1-D device mesh sharding the first
  `GHOST`-capable factor plus one transpose pencil — the direct
  generalization of today's main/alt shardings — but the *interface*
  (named layouts, `layout_for`) already spans n-D device meshes, so
  growing to 2-D pencils is an implementation change, not an API
  change.
- Halo exchange is skipped structurally for names with width 0
  (coefficient axes, `ConstantSpace` factors, `LOCAL` factors), which
  is what makes mixed grids decomposable with zero special cases
  (§5, §6.2).

### GraphDecomposition

Designed-for backend for unstructured factors (graph partitioning,
indirect-neighbor halos).

- Kind: final concrete (`Decomposition` subclass), designed-for.
- Module: `fridom.framework.grid2.decomposition.graph`.
- Pytree: static.
- Iteration: designed-for
  ([§6.4](../05_validation.md#64-unstructured-horizontal-x-structured-vertical)).
- Concept refs: §5, §6.4.

```python
class GraphDecomposition(Decomposition):
    """Graph-partitioned decomposition for unstructured mesh factors."""

    # same ABC surface; `layout_for`/`redistribute` cover only the
    # structured factors of the product — there is no tensor-product
    # transform along the unstructured factor (section 6.4).
```

Notes:

- Exists in this document so the ABC is honest: nothing in
  `Decomposition`'s surface assumes boxed index space (`local_slice`
  generalizes to per-DOF global index arrays; the tuple-of-slices
  return type is the structured specialization and is revisited when
  this class is implemented — flagged under open questions).
- Halo exchange follows partition adjacency (indirect neighbors);
  `sharding` still returns a jax `Sharding` over a flat DOF axis.

---

## 4. Export (`f.xr`)

This cluster owns the xarray-export rules that doc 01/02 point to
(doc 01's "doc 04 territory" pointer lands here). `f.xr` is the field
accessor (doc 02 surface); its semantics are:

- **Axis-position labels are xgcm-style**, derived from the node set
  per factor: `center` / `right` (and `left`) / `outer` / `inner` map
  one-to-one to xgcm staggered-coordinate positions (§2.2's naming
  payoff).
- **Average spaces export coordinate *labels*, not positions**:
  `cell_avg` fields are labeled with the cell-center coordinates,
  `face_avg` fields with the face coordinates — export metadata only,
  since averages have no mathematical position (doc 01's note,
  §2.2). The label carries an attribute marking the DOFs as cell
  means so round-trips do not silently reinterpret them as samples.
- **Data path:** coordinates come from
  `grid.evaluation_nodes(space)`; values are gathered to host via
  `decomposition.gather` (true shape — halo and padding never leave
  the decomposition layer).
- **Not exported in iteration 1:** complex-scalar fields and
  coefficient-space fields (`fr.Complex` storage, wavenumber/mode
  indexing); index-coordinate export for spectra is a later
  iteration. `f.xr` on such fields raises with a pointer to `.data`.

---

## Open questions

- **Merge call site** (inherited from
  [§3.4](../02_rules.md#34-generic-operator-dispatch), stays open):
  which Phase 2 assembly hook calls `grid.merge_overrides`,
  `grid.negotiate`, and `grid.freeze`. This cluster fixes the
  mechanism and the pre-freeze mutation window, not the caller.
- **Per-space refinement of `OperatorRequirements.layout`** (doc 03's
  open question 2, answered on this side as a negotiation detail):
  a `SpectralDerivative`-style operator is `layout="local"` only along
  its own coefficient factor. `negotiate` therefore interprets
  `.layout` per `(operator, factor space)` pair when scoping demands —
  whether doc 03 refines the declared surface to match, or negotiation
  keeps doing the per-space projection itself, is settled at
  implementation time.
- **Product-space `measure` composition**: whether
  `measure(space, name=None)` on a multi-factor product should also
  offer the composed volume measure (the per-factor product), or
  whether that stays operator-internal (current choice: per-factor
  only, `name` required when ambiguous).
- **`local_slice` return type for graph backends**: tuple-of-slices is
  structured-only; the generalization (per-DOF global index array) can
  either widen the ABC signature now or be added as a parallel method
  when `GraphDecomposition` lands.
