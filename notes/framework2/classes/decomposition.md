# Grid abstraction redesign — Class designs: domain decomposition

Part of the framework2 class designs; see [`README.md`](README.md) for the document map. The grid assembly, package layout, and export are in [`grid.md`](grid.md).

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
lesson). The `Layout` is **part of the function space**
([§5.1](../04_decomposition.md#51-layout-is-part-of-the-function-space)):
fields carry laid-out spaces, field arithmetic never reshards, and
`Reshard`/`Sync` (doc 03) are the only data-movement operators.

The class structure: small frozen descriptors (`HaloStrategy`,
`MeshDecompositionTraits`, `HaloSpec`, `Layout`,
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
  operator *base* (doc 03) appends the internal `Sync` node after the
  kernel (§5.1; realized by `Decomposition.sync`) — kernel
  authors never sync, and users never spell `Sync`. Under this contract un-synced chains do not
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
- Module: `fridom.framework2.grid.decomposition.traits` (the single
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
- Module: `fridom.framework2.grid.decomposition.halo`.
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
- Module: `fridom.framework2.grid.decomposition.halo`.
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
- **Reshards reset depth:** a `redistribute` is a global data
  movement, at least as strong as a sync on the moved axes — the
  rule, carried as `Reshard`'s object-level halo rule (§5.1, doc 03),
  is that it **resets the accumulated depth on the moved axes** to
  zero.
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

### Layout

A frozen descriptor of one concrete distribution: which coordinate
names are sharded across which device-mesh axes. **This is the value
that enters the function-space interning key when set**
([§5.1](../04_decomposition.md#51-layout-is-part-of-the-function-space);
doc 02 owns the space-side semantics — `space.layout` / `.bare` /
`.with_layout`).

- Kind: final frozen dataclass.
- Module: `fridom.framework2.grid.decomposition.layout` (imported by
  the space clusters).
- Pytree: static (hashable; part of jit cache keys via the spaces).
- Iteration: 1.
- Concept refs: [§5](../04_decomposition.md#5-domain-decomposition),
  [§5.1](../04_decomposition.md#51-layout-is-part-of-the-function-space).

```python
@dataclass(frozen=True)
class Layout:
    """One assignment of coordinate names to device-mesh axes."""

    device_axes: tuple[tuple[str, str], ...]  # (coord name, device axis)

    def __init__(self, device_axes: Mapping[str, str]) -> None:
        """Normalize the mapping to sorted tuple storage (hashable)."""
        ...

    def is_local(self, name: str) -> bool:
        """Whether the factor carrying `name` is device-local here."""
        ...
```

Notes:

- Purely combinatorial and **semantic-only**: halo widths and stagger
  padding are deliberately *not* part of a `Layout` — they are
  storage the decomposition pairs with it internally (the negotiated
  `HaloSpec`), invisible in the space identity (§5.1: fields with the
  same sharding must add regardless of ghost widths). Everything
  array-shaped (`PartitionSpec`s, local slices, storage shapes) is
  derived by the owning `Decomposition`, which holds the
  `jax.sharding.Mesh`. Layouts are values, so transforms can name
  their pencil schedule (`x-local` layout, `y-local` layout) as data.
- Tuple storage with a mapping-accepting constructor, for the same
  hashability reason as `HaloSpec.widths` (the doc 02 `nc_attrs`
  pattern).
- The default layout of a grid shards the preferred `GHOST` factors;
  transform-aware layouts (`TRANSPOSE` strategy) are the generalized
  successors of today's main/alt shardings in `JaxDecomposition`;
  `redistribute` is `Reshard`'s kernel (§5.1) — user code reaches it
  only through the operator.
- **Closed vocabulary** (§5.1): the layouts negotiated at assembly
  (default + transform pencils + solver-declared) are all the layouts
  there are; `Reshard` targets and the transform planner's
  shortest-path search range over this finite set. New needs are
  declared at negotiation (via `OperatorRequirements`), not minted at
  runtime.

### Decomposition / negotiate

The ABC every backend implements, and the per-mesh negotiation entry
point. The grid owns exactly one `Decomposition`; fields and operators
reach it only through the grid.

- Kind: ABC (`abc.ABC`); `negotiate` module function.
- Module: `fridom.framework2.grid.decomposition.decomposition`.
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

    old: Layout
    new: Layout
    changed: bool


class Decomposition(ABC):
    """Distribution of a grid's DOFs across devices (grid-owned)."""

    @property
    def halo(self) -> HaloSpec:
        """The negotiated per-name ghost widths."""
        ...

    @property
    def default_layout(self) -> Layout:
        """The layout attached to bare spaces at field creation;
        state fields are stepped in it. Transform outputs legally
        stay in their pencils (section 5.1)."""
        ...

    @property
    def layouts(self) -> tuple[Layout, ...]:
        """All negotiated layouts (default + transform pencils)."""
        ...

    @abstractmethod
    def sharding(
        self,
        space: TensorProductSpace | FunctionSpace,
        layout: Layout | None = None,
    ) -> jax.sharding.Sharding:
        """The jax sharding of `space`'s storage under `layout`."""
        ...

    @abstractmethod
    def local_slice(
        self,
        space: TensorProductSpace | FunctionSpace,
        layout: Layout | None = None,
    ) -> tuple[slice, ...]:
        """Global true-DOF index range of the local shard."""
        ...

    @abstractmethod
    def storage_shape(
        self,
        space: TensorProductSpace | FunctionSpace,
        layout: Layout | None = None,
    ) -> tuple[int, ...]:
        """Global storage shape: true shape + halo + stagger padding."""
        ...

    @abstractmethod
    def zeros(
        self,
        space: TensorProductSpace | FunctionSpace,
        layout: Layout | None = None,
    ) -> jax.Array:
        """A zero-filled, sharded, storage-shaped array for `space`."""
        ...

    @abstractmethod
    def pad(
        self,
        arr: jax.Array,
        space: TensorProductSpace | FunctionSpace,
        layout: Layout | None = None,
    ) -> jax.Array:
        """True-shape local data -> halo/stagger-padded storage."""
        ...

    @abstractmethod
    def unpad(
        self,
        arr: jax.Array,
        space: TensorProductSpace | FunctionSpace,
        layout: Layout | None = None,
    ) -> jax.Array:
        """Padded storage -> true-shape local data (pads dropped)."""
        ...

    @abstractmethod
    def sync(
        self,
        arr: jax.Array,
        space: TensorProductSpace | FunctionSpace,
        *,
        layout: Layout | None = None,
        fills: Mapping[str, jax.Array] | None = None,
    ) -> jax.Array:
        """Exchange halos; bounded edges per `fills` / BC-structured."""
        ...

    @abstractmethod
    def layout_for(
        self,
        local_names: tuple[str, ...],
    ) -> Layout:
        """A negotiated layout in which the named factors are local."""
        ...

    @abstractmethod
    def redistribute(
        self,
        arr: jax.Array,
        space: TensorProductSpace | FunctionSpace,
        src: Layout,
        dst: Layout,
    ) -> jax.Array:
        """Transpose an array between two negotiated layouts."""
        ...

    @abstractmethod
    def gather(
        self,
        arr: jax.Array,
        space: TensorProductSpace | FunctionSpace,
        layout: Layout | None = None,
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
- Module: `fridom.framework2.grid.decomposition.tensor`.
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
        layouts: tuple[Layout, ...],
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
- Module: `fridom.framework2.grid.decomposition.graph`.
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

## Open questions

The former open question — where the sharding layout lives — is
**closed** (owner decisions, 2026-07-06), recorded normatively in
[§5.1](../04_decomposition.md#51-layout-is-part-of-the-function-space)
and folded into the cluster docs: `Layout` is an optional defining
attribute of the function space (doc 01/02: `space.layout`, `.bare`,
`.with_layout`; grid-minted, in the intern key only when set); the
join requires layout equality and **field arithmetic never reshards**;
`Reshard` is a user-facing operator and `Sync` an internal-only node
(doc 03), both placed by requirements-driven lowering over the
**closed** negotiated layout vocabulary; multi-axis transforms are
planner-ordered for speed. The forcing counterexample (same logical
coefficient space reached in different pencils depending on transform
order) is kept in §5.1.

- **Per-step sync amplification across tendency modules**
  (owner-flagged, 2026-07-07; important — investigate with the
  Phase-2 model-composition design, ROADMAP 2.1). Under the
  iteration-1 sync-after-every-operator contract, n tendency modules
  each computing a tendency for the same field (advection, Coriolis,
  ...) pay n syncs per step where at most one is needed — tendency
  summation is pointwise, so nothing reads ghost cells between each
  module's last operator and the state update; strictly, the *summed
  tendency* needs valid halos only where the next step's operators
  consume it. This cost is unaffordable at scale (each sync is a
  communication round). Candidate resolutions to investigate:
  1. **Drop auto-sync** — user/model-controlled sync placement.
     Maximum control, but moves the halo-validity invariant onto
     users; the contract's rationale (silent wrongness impossible)
     currently rejects this.
  2. **Tendencies are operators** — each module contributes an
     operator term; the model composes one fused tendency operator
     (an `OperatorSum` of per-module chains), and the
     requirements-driven lowering places a single `Sync` at the end
     (or none, deferring to the state update). The existing
     halo-accounting rules (chains sum, parallel terms max) and the
     designed-for sync-elision machinery already fit exactly this
     shape; this generalizes elision from chains to the whole step.
  Not resolved here; the decision belongs to 2.1 and must be made
  before the module `update` signature is fixed.

