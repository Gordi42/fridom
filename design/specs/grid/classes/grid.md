---
status: normative
date: 2026-07-07
---

# Grid abstraction redesign — Class designs: grid assembly and decomposition

Part of the grid redesign notes; see [`../00_overview.md`](../00_overview.md)
for the document map. Status: implemented (Phase 1 landed; kept as the normative reference).
Signatures are the intended public API for `framework2.grid`; the
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

The code lives in `fridom.framework2.grid` (part of the new parallel
`fridom.framework2` package), renamed to `fridom.framework.grid` at
cutover. Docs 01–03
state their own placements; the tree below shows the whole subpackage
so this cluster's modules have an address, with ownership per doc:

```
fridom/framework2/grid/
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
        movement.py          # doc 03: Reshard, Sync (section 5.1)
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
        layout.py            #   Layout (space-level, section 5.1)
        decomposition.py     #   Decomposition (ABC), negotiate(),
                             #   ReshardingReport
        tensor.py            #   TensorDecomposition
        graph.py             #   GraphDecomposition (designed-for)
```

Names contributed by `framework2.grid`. The left column is the clean
top-level re-export (`fr.*`); the right is the subpackage-qualified path
that reaches the same object before the re-export is wired (the plural
collection namespaces match `fr.modules` / `fr.time_steppers`):

| Top-level (`fr.*`)       | Subpackage path             | Object |
|--------------------------|-----------------------------|--------|
| `fr.Grid`                | `fr.grid.Grid`             | assembly root (this doc) |
| — (no top-level alias)   | `fr.grid.cartesian.Grid`   | convenience subclass (this doc) |
| `fr.meshes`              | `fr.grid.meshes`           | mesh factors (doc 01) |
| `fr.operators`           | `fr.grid.operators`        | free-standing operators (doc 03) |
| `fr.Real`, `fr.Complex`  | `fr.grid.Real`, `fr.grid.Complex` | scalars / Körper tags (doc 01) |
| `fr.BC`                  | `fr.grid.BC`               | BC structure enum (doc 01, day one: `DIRICHLET`/`NEUMANN` are exercised by the iteration-1 Sine/Cosine spaces) |
| `fr.ScalarField`, `fr.VectorField`, `fr.TensorField` | `fr.grid.*` | field types (doc 02) |
| `fr.TensorProductSpace`, `fr.SpaceLike` | `fr.grid.*`  | product space + space alias (doc 02) |
| `fr.FieldMetadata`       | `fr.grid.FieldMetadata`    | field metadata record (doc 02) |
| `fr.SpaceMismatchError`, `fr.GridMismatchError` | `fr.grid.*` | error types (doc 02) |

The table is exhaustive: these are all `fr.*`-level names contributed
by `framework2.grid`. Function spaces get **no** top-level namespace: they are
produced by mesh factories (`mx.center`, `mz.galerkin(...)`). The
decomposition subpackage is *not* re-exported at `fr.*` level — fields
and operators reach it only through the grid
([§5](../04_decomposition.md#5-domain-decomposition)); it is public for
transform/solver authors as `fr.grid.decomposition`.

The transitional `framework2/grid/__init__.py` realizes the table with the
lazypimp pattern mandated by `AGENTS.md` (at rename time the same
entries move up to the framework `__init__`):

```python
base = "fridom.framework2.grid"

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

Tests mirror the package as `tests/framework2/grid/**`, one test file
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
- Module: `fridom.framework2.grid.grid`.
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
  is doc 03's (module `framework2/grid/operators/registry.py`); the grid owns
  the single **instance**, exposed as `grid.dispatch` (matching the
  module-side `self.dispatch` override dicts of sketch
  [4.2](../03_api_sketches.md#42-custom-operator-module-local-override)).
  `grid.merge_overrides` is a **facade** over the pure registry: it
  calls `OperatorRegistry.merge(overrides)` (which returns a new
  registry, doc 03) and swaps the held instance. Precedence:
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
  dynamic leaves (a grid-held `ScalarField` leaf creates the pytree
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
   per-operator maximum over the default registry. Under the
   consumption-side contract (task 1.8) this is the exact
   per-application **floor**: any width at or above it is correct
   (chains that exhaust it re-sync mid-chain), wider traced widths
   only save exchanges. A grid
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
   **Amendment (2026-07-08, Phase-2 reconciliation):** this
   report-driven walk is narrower than it reads. `ReshardingReport`
   is `(old, new, changed)` over **layouts only**
   (`decomposition/decomposition.py:77-100`), so a **halo-only**
   renegotiation reads `changed=False` — yet a halo change alters
   **storage** shapes, because padding applies on every axis,
   including unsharded ones (`decomposition/tensor.py:449-451`).
   Fields created under the old negotiation are then stranded: the
   old geometry is gone from the grid after the swap, and sync/unpad
   reject their arrays by shape (`decomposition/tensor.py:505-515`)
   — the root cause of the WENO manual-negotiate finding.
   Report-driven `device_put` therefore **cannot** re-home such
   fields; the sanctioned re-home path is **true-shape re-pad** —
   the `set_fields`/`set_state` path, where values are gathered at
   true shape and re-stored under the live decomposition. Model
   assembly avoids the problem wholesale by allocating all carry
   fields only **after** final negotiation (assembly step 8) and
   forbidding field materialization at `bind`.
4. **Registry reuse:** once any lazy transform row has been resolved,
   the registry holds grid-bound instances and is **grid-private**;
   sharing a registry object between grids is an error.

### framework2.grid.cartesian.Grid

Convenience subclass building uniform `IntervalMesh` factors from
`shape=`/`extent=`/`periodic=` — the day-one constructor
([§10.1](../07_iteration1_api.md#101-building-a-grid)).

- Kind: final concrete subclass of `Grid`.
- Module: `fridom.framework2.grid.cartesian.grid`.
- Pytree: as base (adds nothing).
- Iteration: 1 (it is *the* iteration-1 public constructor).
- Concept refs: [§2.6](../01_concepts.md#26-grid--the-assembly-object),
  sketch [4.1](../03_api_sketches.md#41-uniform-tensor-grid-staggered-derivative).

```python
class Grid(fr.grid.Grid):
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
  `fr.grid.Grid`.

### Discretizer

The operator shape behind the reserved kinds `("discretize", space)`
and `("assign_coeff", space)`: what `create_field` resolves and calls
for `init=` / `init_coeff=`.

- Kind: ABC (small protocol-style base).
- Module: `fridom.framework2.grid.discretize`.
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
- Module: `fridom.framework2.grid.random_fields`.
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
- Module: `fridom.framework2.grid.immersed_domain`.
- Pytree: **fully static** — holds the init callable / static
  parameters only, no arrays and no `ScalarField`s. Fractions and
  masks are materialized on demand at trace time, exactly like
  `evaluation_nodes` (G1).
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
  grid-materialized-array status as `dx` (§3.7). The immersed domain is
  an *attachment* at `grid.immersed`, not a `MaskedGrid(inner_grid)`
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
- Module: `fridom.framework2.grid.coordinate_mapping`.
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

## 4. Export (`f.xr`)

This cluster owns the xarray-export rules that doc 01/02 point to
(doc 01's "doc 04 territory" pointer lands here). `f.xr` is the field
accessor (doc 02 surface); its semantics are:

- **Axis-position labels are xgcm-style**, derived from the node set
  per factor: `center` / `right` (and `left`) / `outer` / `inner` map
  one-to-one to xgcm staggered-coordinate positions (§2.2's naming
  payoff).
- **Single-field exports drop the position suffix** (amended
  2026-07-11, owner review of the docs pilot): `ScalarField.xr`
  exports staggered dims under the plain axis name (`x`, not
  `x_right`) — a lone `DataArray` has no sibling to collide with, and
  plain names give clean plot labels; the position survives in the
  `c_grid_axis_shift` attribute. The suffixed spelling remains the
  `scalar_to_dataarray(..., positions_in_names=True)` default used by
  every multi-variable export (`VectorField.xr`, the io Writer
  store), where two positions of one axis must coexist.
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
- **Coefficient-space export shipped in iteration 1** (amended at
  implementation, 2026-07-07 — originally deferred): dims `k<name>`
  with real wavenumber coords from `grid.wavenumbers` (`mode_index`
  for Chebyshev), a `representation` coordinate attribute, complex
  values passed through to xarray.

---

## Open questions

- **Merge call site** — *resolved by the Phase-2 model design (D4,
  `design/specs/model/04_run_loop_io.md` §6.2)*:
  `grid.merge_overrides` is called **exactly once per grid, by
  `fr.Model` assembly step 3** — after field-declaration collection
  (whose `("declared_space", mesh)` resolution the override keys
  reuse) and before the module `bind` hooks, the composer's
  validation dry run, and `negotiate` — so both the dry run and the
  halo-accounting trace resolve dispatch against the registry *as
  merged* ([§5](../04_decomposition.md)). There is no
  `Module.setup(...)`; module overrides are a constructor-frozen
  `Module.dispatch` mapping keyed by `kind` or
  `(kind, SpacePattern)`, model-resolved to interned spaces; values
  may be lazy factories (the transform-row seeding mechanism)
  exposing `OperatorRequirements` unbound; two modules on one
  resolved key is an assembly error naming both;
  `("declared_space", ...)` entries are never module-mergeable
  (model D1.2). A grid used without a model keeps its provisional
  registry and negotiation.
  **Frozen-grid verify path** (model D4): at `freeze()` the grid
  records a negotiation fingerprint (state-space set, merged
  override keys, `HaloSpec`, layout vocabulary); a subsequent model
  assembly on the frozen grid skips merge/negotiate/freeze and
  *verifies* its demands against the record. **Verification is
  demand-satisfaction (⊆ / ≤), not equality** (model D5): identical
  composition passes trivially, and a model or variant demanding a
  *subset* (e.g. a term-filtered variant — fewer terms, fewer
  operators, smaller-or-equal halo, same state spaces) passes by
  construction, inheriting the recorded layouts/halos; only *larger*
  demands raise `GridFrozenError` ("assemble the most demanding
  model first"). **Satisfiability relaxation** (model validation
  sign-off): a demanded state space that is *new* but carries zero
  halo/layout/override demands (the ConstantSpace-broadcast family
  — e.g. `Profile("y")` on a grid whose record only holds
  `Profile()`) is **adopted into the record** rather than refused;
  adoption never reopens negotiation. Module-*type* sweeps that
  genuinely change demands use a fresh grid per composition (free
  at the jit-cache level — different module tuples are different
  assembly records regardless). One-grid-many-models is the sanctioned
  sweep idiom (fields carry the grid as an identity-hashed static,
  so grid reuse is what makes shared-jit-cache sweeps possible).
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
