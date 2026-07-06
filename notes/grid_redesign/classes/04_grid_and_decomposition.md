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
    scalars.py               # doc 01: fr.Real / fr.Complex
    bc.py                    # doc 01: fr.BC
    errors.py                # doc 02: SpaceMismatchError
    meshes/                  # doc 01: Mesh subclasses (fr.meshes)
    spaces/                  # doc 01: FunctionSpace families,
                             #   ConstantSpace
        tensor_product.py    # doc 02: TensorProductSpace
    fields/                  # doc 02: ScalarField, VectorField, State
    operators/               # doc 03: Operator hierarchy, transforms,
                             #   Symbol (fr.operators)
        registry.py          # doc 03: OperatorRegistry
    grid.py                  # THIS DOC: Grid (assembly root)
    random_fields.py         # THIS DOC: RandomFieldFactory
    immersed_domain.py       # THIS DOC: ImmersedDomain, Slip
    coordinate_mapping.py    # THIS DOC: CoordinateMapping
    cartesian/
        grid.py              # THIS DOC: cartesian convenience Grid
    decomposition/           # THIS DOC (all of it):
        traits.py            #   HaloStrategy, MeshDecompositionTraits
        halo.py              #   HaloSpec, HaloTracer, trace_halo
        layout.py            #   ArrayLayout
        decomposition.py     #   Decomposition (ABC), negotiate()
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
| `fr.BC`                  | `fr.grid2.BC`               | BC structure enum (doc 01) |

The table is exhaustive: these are all `fr.*`-level names contributed
by `grid2`. Function spaces get **no** top-level namespace: they are
produced by mesh factories (`mx.center`, `mz.galerkin(...)`). The
decomposition subpackage is *not* re-exported at `fr.*` level — fields
and operators reach it only through the grid
([§5](../04_decomposition.md#5-domain-decomposition)); it is public for
transform/solver authors as `fr.grid2.decomposition`.

---

## 2. Grid assembly

### Grid

The model-agnostic assembly root: meshes + names + decomposition +
dispatch registry + field factory + attachments. Ergonomics and wiring
only — all mathematics lives in spaces and operators, and model physics
(`omega`, `vec_q`, `vec_p`) has left the grid entirely
([§2.6](../01_concepts.md#26-grid--the-assembly-object)).

- Kind: concrete (also the base class of the cartesian convenience
  subclass; not an ABC — it is fully functional as-is).
- Module: `fridom.framework.grid2.grid`.
- Pytree: static structure (meshes, names, dispatch registry,
  decomposition, random factory); dynamic children are exactly the
  two data-carrying attachments (`_immersed`, `_mapping`), via
  `@partial(fr.utils.jaxify, dynamic=("_immersed", "_mapping"))`.
  Grids are compared and hashed by identity, like spaces.
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
@partial(fr.utils.jaxify, dynamic=("_immersed", "_mapping"))
class Grid:
    """Assembly root: meshes, decomposition, dispatch, field factory."""

    def __init__(
        self,
        meshes: tuple[Mesh, ...],
        names: tuple[str, ...],
        *,
        defaults: Mapping[str | tuple[str, FunctionSpace], Operator]
            | None = None,
        mapping: CoordinateMapping | None = None,
        immersed: ImmersedDomain | None = None,
        device_ids: tuple[int, ...] | None = None,
    ) -> None:
        """Assemble a grid from pre-built mesh factors."""
        ...

    # ================================================================
    #  Structure
    # ================================================================

    @property
    def factors(self) -> tuple[Mesh, ...]:
        """The Mesh factor objects, in names order."""
        ...

    @property
    def names(self) -> tuple[str, ...]:
        """All coordinate names, in factor order (2D meshes give several)."""
        ...

    # ================================================================
    #  Operator dispatch
    # ================================================================

    @property
    def dispatch(self) -> OperatorRegistry:
        """The operator dispatch registry (defaults + merged overrides)."""
        ...

    def resolve(
        self, kind: str, space: FunctionSpace,
    ) -> Operator:
        """Resolve the default operator for (kind, factor space)."""
        ...

    def merge_overrides(
        self,
        overrides: Mapping[str | tuple[str, FunctionSpace], Operator],
    ) -> None:
        """Merge module-local dispatch overrides (assembly phase only)."""
        ...

    # ================================================================
    #  Decomposition
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
    ) -> None:
        """Renegotiate the decomposition from the merged registry."""
        ...

    def sync(
        self, field: fr.ScalarField | fr.VectorField,
    ) -> fr.ScalarField | fr.VectorField:
        """Fill halos (periodic wrap / homogeneous / ghost_fill per axis)."""
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
        name: str,
        space: TensorProductSpace | FunctionSpace,
    ) -> fr.ScalarField:
        """A named mapping metric on the requested space (designed-for)."""
        ...

    # ================================================================
    #  Attachments
    # ================================================================

    @property
    def immersed(self) -> ImmersedDomain | None:
        """The immersed (masked) domain, or None."""
        ...

    @property
    def mapping(self) -> CoordinateMapping | None:
        """The coordinate mapping (terrain-following), or None."""
        ...
```

Semantics and invariants:

- **Constructor.** `meshes` and `names` follow sketch
  [4.4](../03_api_sketches.md#44-mixed-grid-uniform-fv-x-chebyshev-galerkin):
  pre-built meshes of any type (sphere, unstructured included). A 2D
  mesh contributes several coordinate names; `names` is flat across
  contributions and must be duplicate-free (the tensor product's flat
  namespace, [§2.3](../01_concepts.md#23-tensorproductspace-and-named-coordinates)).
  This is one of **two constructors, never one overloaded signature**
  ([§2.6](../01_concepts.md#26-grid--the-assembly-object)); the
  `shape=`/`extent=` form is the cartesian subclass below. `defaults=`
  seeds the dispatch registry; `mapping=`/`immersed=` attach the
  designed-for objects (both are declarative payloads bound and
  materialized during grid construction — see their classes).
  `device_ids` restricts the device set (coupled runs), as today.
- **Name binding.** The constructor calls `mesh.bind_names(...)` on
  each factor with its slice of `names` — doc 01's **write-once**
  contract. A pre-bound mesh whose bound names conflict with the
  requested ones raises at construction. Recorded consequence: **one
  mesh instance cannot be reused in two grids under different
  coordinate names**; build a second mesh for that. (Whether binding
  stays a bind-time call or becomes mandatory at mesh construction is
  doc 01's open question 2, mirrored below.)
- **Registry placement and lifetime.** The `OperatorRegistry` *class*
  is doc 03's (module `grid2/operators/registry.py`); the grid owns
  the single **instance**, exposed as `grid.dispatch` (matching the
  module-side `self.dispatch` override dicts of sketch
  [4.2](../03_api_sketches.md#42-custom-operator-module-local-override)).
  It is created in `__init__` from the per-mesh space-family defaults
  plus the `defaults=` argument. `grid.merge_overrides` is a **facade**
  over the pure registry: it calls `OperatorRegistry.merge(overrides)`
  (which returns a new registry, doc 03) and swaps the held instance —
  the successor of the removed `diff_module`/`interp_module` slots.
  Precedence: `(kind, space)` entry > kind-only entry > grid default
  ([§3.4](../02_rules.md#34-generic-operator-dispatch)). Swapping is
  legal **only during the setup/assembly phase** (like today's module
  lifecycle); after negotiation the held registry is final, so it can
  serve as part of the grid's static dispatch identity. The exact call
  site of the merge is the open question tied to the Phase 2
  composition design (§3.4) — this class specifies the mechanism only.
- **Field factory.** `create_field` is the **single factory**
  ([§3.10](../02_rules.md#310-discretizing-continuous-functions)):
  `space` is positional and optional, defaulting to the all-Center
  nodal product; `init=` (dispatch kind `("discretize", space)`,
  keyword-matched to coordinate names) and `init_coeff=` (kind
  `("assign_coeff", space)`, matched to wavenumber / mode-index names)
  are **mutually exclusive**; `data=` is the direct-array companion
  (the grid owns sharding/layout validation); with none of the three
  the field is zeros (sketch
  [4.3](../03_api_sketches.md#43-transform-round-trip-with-per-origin-coefficient-spaces)).
  Construction is pure and traceable — legal inside the jit loop. There
  is no `Field.from_function`, no `f.set(...)`, and no lazy field
  (rejected alternatives, §3.10). `name`/`units` seed the shrunken
  field metadata ([§2.4](../01_concepts.md#24-field)); this is a
  **deliberate minimal surface** — the full record (doc 02's
  `FieldMetadata`, incl. nc-attrs) is set on the field after
  construction, and the factory takes no `metadata=` kwarg.
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
  `discretize`, never stored (§2.7). Consequently a renegotiated
  decomposition can never leave a stale shard. Whether a large mapped
  node array is wrapped in a `jax.checkpoint`-style store is an
  **opt-in performance knob** to benchmark — invisible to the
  abstraction, bound by the same re-decomposition-invalidation rule,
  never a semantic change (§2.7).
- **Measures.** `measure(space)` returns the staggered `dx` field
  proper to the space: the primal **cell width** on `Center`/`CellAvg`
  (the FV integration weight), the dual **center-to-center spacing** on
  `Right`/`Outer`/`FaceAvg` (the `diff` denominator) — one accessor,
  the space decides which measure it is (§2.7, §3.9). On a uniform
  mesh both collapse to a constant field that XLA constant-folds.
  `name` disambiguates on products, as for `evaluation_nodes`;
  composed volume measures are left to operators (weights compose per
  mesh, [§3.13](../02_rules.md#313-reductions-and-integrals)).
- **Negotiation and sync.** See section 3 below; `grid.negotiate` is
  the setup-phase entry point that swaps in the final decomposition
  after `merge_overrides`, and `grid.sync` is the field-level
  halo-exchange surface (it resolves per-axis fill modes — periodic
  wrap, homogeneous fill, `("ghost_fill", space)` dispatch — and
  delegates raw-array work to the decomposition). Operators obtain
  halo-extended storage through the grid; nothing above the grid
  touches `jax.sharding` directly.
- **Pytree treatment.** The grid's static structure participates in
  jit cache keys; the only dynamic leaves the grid *holds* are the
  immersed fraction field and the mapping's parameter fields (both
  genuinely data). Everything else the grid hands out is created on
  demand and owned by the caller. This keeps "structure static, arrays
  dynamic" exact (§2.2, §2.7).

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
        defaults: Mapping[str | tuple[str, FunctionSpace], Operator]
            | None = None,
        mapping: CoordinateMapping | None = None,
        immersed: ImmersedDomain | None = None,
        device_ids: tuple[int, ...] | None = None,
    ) -> None:
        """Build uniform IntervalMesh factors and assemble the grid."""
        ...
```

Notes:

- Builds one `fr.meshes.IntervalMesh(shape=n, extent=(a, b),
  periodic=p)` per axis and delegates to the base constructor — **no
  new methods or properties**; the two constructors stay split across
  base and subclass so neither signature silently accepts the other's
  kwarg set (§2.6). No `N`/`L` aliases.
- `periodic` broadcasts a single bool to all axes; `names` defaults to
  `("x", "y", "z")[:ndim]` for `ndim <= 3` and is required otherwise.
- Everything after `names` is forwarded verbatim to `fr.grid2.Grid`.

### RandomFieldFactory

The `grid.random` accessor: seeded, sharding-consistent random field
generators ([§3.10](../02_rules.md#310-discretizing-continuous-functions)).

- Kind: final concrete.
- Module: `fridom.framework.grid2.random_fields`.
- Pytree: static (holds only the grid reference); registered with
  `@fr.utils.jaxify` for containment in jaxified objects.
- Iteration: 1 (`normal`, `phase`; the spectra-IC consumer of `phase`
  is designed-for, the method itself is not).
- Concept refs: §3.10,
  [§5](../04_decomposition.md#5-domain-decomposition), sketch
  [4.9](../03_api_sketches.md#49-random-spectra-initial-condition-spectral-space-construction).

```python
@fr.utils.jaxify
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
  draw-global-then-slice (§3.10).
- The draw covers the space's **true shape** (§3.5), so pad slots are
  outside the index space and random values never land in padding —
  no special rule needed
  ([§5](../04_decomposition.md#5-domain-decomposition)).
- On a real-origin Fourier space the draw covers only the free
  half-spectrum; the Hermitian structure follows from the shape, with
  the real-only `k = 0`/Nyquist DOFs handled as special indices
  (§3.10). `normal` on a `fr.Complex` space draws independent
  real/imag parts.
- Pure and traceable under jit; returns ordinary `(space, array)`
  fields. `space` is mandatory (there is no obvious default and the
  accessors' explicit-space discipline applies). Extension contract:
  additional draws (e.g. uniform) must use the same per-shard
  global-index keying; anything else silently breaks determinism
  across device counts.

### ImmersedDomain

Grid-owned successor of `WaterMask`: the single wet volume-fraction
datum plus derive-on-demand per-space masks/fractions
([§3.7](../02_rules.md#37-boundaries-ii-immersed-masked-domains)).

- Kind: final concrete.
- Module: `fridom.framework.grid2.immersed_domain`.
- Pytree: dynamic leaf is the stored fraction field
  (`@partial(fr.utils.jaxify, dynamic=("_fraction",))`); the slip
  rule and binding are static.
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


@partial(fr.utils.jaxify, dynamic=("_fraction",))
class ImmersedDomain:
    """Wet volume fraction and derived per-space masks/fractions."""

    def __init__(
        self,
        init: Callable[..., jax.Array] | None = None,
        *,
        data: jax.Array | None = None,
        slip: Slip = Slip.NO_SLIP,
    ) -> None:
        """Declare the wet fraction (function of coords, or raw data)."""
        ...

    @property
    def fraction_field(self) -> fr.ScalarField:
        """The stored wet volume-fraction field on the base cell space."""
        ...

    @property
    def slip(self) -> Slip:
        """The staggered-mask derivation rule (no-slip / free-slip)."""
        ...

    def fraction(
        self, space: TensorProductSpace | FunctionSpace,
    ) -> fr.ScalarField:
        """Wet fraction transferred to `space` (volume or area weights)."""
        ...

    def mask(
        self,
        space: TensorProductSpace | FunctionSpace,
        *,
        slip: Slip | None = None,
    ) -> fr.ScalarField:
        """Boolean wet mask on `space`, derived by the slip rule."""
        ...

    def transition(
        self, space: TensorProductSpace | FunctionSpace,
    ) -> fr.ScalarField:
        """Wet/dry transition-set indicator on `space` (designed-for)."""
        ...
```

Notes:

- **Single stored datum**: a wet volume-fraction field in `[0, 1]` on
  the base cell space (`Center`/`CellAvg`), an ordinary full-shape
  dynamic sharded `ScalarField`. The boolean mask is the `{0, 1}`
  special case, so there is one representation, not two; a level-set
  field on the same space is the future generalization (§3.7).
- **Binding.** The constructor takes a declarative payload (`init`
  callable of physical coordinates, or `data`) because the fraction
  field cannot exist before the grid does; the grid discretizes it onto
  the base cell space during grid construction (`immersed=` kwarg).
  After binding, `fraction_field` is the dynamic pytree leaf; on
  renegotiation it is resharded (it is data, not a derived array, so
  the recompute-on-demand rule does not cover it — the one stored
  array in this cluster).
- **Derived quantities are grid-mediated and derive-on-demand**,
  mirroring `grid.evaluation_nodes(space)`: `fraction(space)`,
  `mask(space)`, `transition(space)` return fields *tagged with that
  space*, computed from the base fraction by a staggering-transfer
  rule. **Structure is the combination rule**: no-slip is today's
  `face = AND(adjacent centers)`; `slip` is a parameter of the
  derivation, not stored per-space state (the constructor value is the
  default, the `mask(..., slip=...)` override serves mixed-physics
  diagnostics). Fraction transfer at ladder point 2 is geometric
  (area/volume weights), independent of slip. Memoizing derived masks
  is a below-the-operator-layer optimization to benchmark, bound by
  the re-decomposition-invalidation rule; the interface is
  derive-on-demand regardless (§3.7).
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

Grid-attached declaration of terrain-following / curvilinear
coordinate maps; single owner of the metric data
([§3.8](../02_rules.md#38-boundaries-iii-terrain-following-boundary-fitted)).

- Kind: final concrete.
- Module: `fridom.framework.grid2.coordinate_mapping`.
- Pytree: dynamic leaves are the parameter fields
  (`@partial(fr.utils.jaxify, dynamic=("_params",))`); maps and
  supplied-metric declarations are static.
- Iteration: designed-for (nothing in iteration 1 may assume static
  metrics, [§6.5](../05_validation.md#65-terrain-following-vertical-coordinate)).
- Concept refs: §3.8,
  [§2.3](../01_concepts.md#23-tensorproductspace-and-named-coordinates),
  §6.5.

```python
@partial(fr.utils.jaxify, dynamic=("_params",))
class CoordinateMapping:
    """Coordinate transform declaration and metric-field owner."""

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
    def params(self) -> Mapping[str, fr.ScalarField]:
        """The named parameter fields (H, eta, ...) — dynamic leaves."""
        ...

    @property
    def metric_names(self) -> tuple[str, ...]:
        """The metric names this mapping can supply."""
        ...

    def with_params(self, **params: fr.ScalarField) -> CoordinateMapping:
        """Functionally replace parameter fields (prognostic updates)."""
        ...

    def metric(
        self,
        name: str,
        space: TensorProductSpace | FunctionSpace,
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
  metric fields directly (`metrics=`) for cases with no closed form.
  Like `ImmersedDomain`, declarations are callables bound at grid
  build (`mapping=` kwarg); `params` callables are discretized to
  fields at bind.
- `grid.metric(name, space)` delegates here; the accessor works
  uniformly for both forms (supplied metrics are reconstructed to the
  requested space), so **staggered consistency is guaranteed by the
  grid, not per module** — H at u-, v-, w-points comes from one owner
  (§3.8, §6.5).
- **Time dependence is automatic**: prognostic parameter fields (z*,
  `H = H_0 + eta(t)`) are dynamic pytree leaves; derived metrics are
  recomputed from current parameter values on each query — no operator
  may cache them (§2.3, §3.8). Updates are functional
  (`with_params`), keeping the jax pytree discipline; the grid
  attachment point is swapped by the owning module each step.
- Metric-coefficient operator composites (constant-z vs constant-sigma
  derivative kinds) are dispatch entries reading `grid.metric` — doc
  03 territory; this class only owns the data and its per-space
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
`MeshDecompositionTraits`, `HaloSpec`, `ArrayLayout`), one negotiation
entry point (`negotiate`), one ABC (`Decomposition`) with the
jax-sharding tensor backend (`TensorDecomposition`, iteration 1) and a
named designed-for graph backend (`GraphDecomposition`), plus the
halo-accounting tracer (`HaloTracer`, `trace_halo`).

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
  spaces declare `(LOCAL,)` (recurrences couple the whole column); the
  spaces of an unstructured mesh declare `(GRAPH,)`. This is what
  makes `uniform(x, y) ⊗ chebyshev(z)` decomposable: shard x/y, keep z
  on-device (§5, §6.2).
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

    # ScalarField-mimicking surface: arithmetic (`+`, `-`, `*`, ...),
    # `.diff`, `.to`, `.integrate`, `.data`-free — every operator
    # application returns a new HaloTracer with grown depth and the
    # operator's codomain space; grid.sync resets depth to zero.


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
  `h1 + h2`); the trace is exact and author-effort-free (§5).
- Interception is **generic**: because the tracer presents the
  `ScalarField` interface, operators run unchanged. Concretely, the
  hook sits in the shared operator application path (doc 03's
  `Operator` base `__call__` / the dispatch shim behind `f.diff`):
  when the operand is a tracer, the operator's
  `requirements(domain).halo` is recorded and the codomain-space
  tracer returned, without touching kernel code. No per-operator tracer code, and **no
  halo bookkeeping on real fields** (field metadata stays
  name/units/nc-attrs, §2.4).
- The trace also yields sync placement information (where depth would
  exceed the chosen ghost width, a sync must be inserted); iteration 1
  sizes ghost layers from the maximum depth and keeps today's
  sync-after-every-operator-group placement.
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


@fr.utils.jaxify
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
        """Exchange halos; fill bounded edges per `fills` (else zeros)."""
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
  otherwise from the per-operator maximum (documented as insufficient
  for un-synced composition chains — the trace is the supported path)
  or from the explicit `halo=` override. The backend is chosen from
  the traits (`GHOST`/`TRANSPOSE`/`LOCAL` -> `TensorDecomposition`;
  any `GRAPH` factor -> `GraphDecomposition`). This replaces the old rebuild-on-halo-mismatch
  logic (§5); `grid.negotiate` re-runs it at assembly, and
  recompute-on-demand (§2.7) guarantees no derived array survives a
  renegotiation.
- **Solver/transform API (the bypass fix).** The lesson from
  `RFFTPressureSolver` (rfft, axis subsets, custom dct against the raw
  decomposition) becomes supported surface: a grid-bound transform
  (`fr.operators.Fourier(grid, axes=...)`, doc 03) plans its schedule
  as a sequence of (`layout_for(names)`, apply local 1D kernels,
  `redistribute`) steps — any per-axis kernel runs device-local in a
  pencil layout, and the decomposition contributes only layouts and
  transposes. Day one that kernel set is Fourier-only (rfft/fft); doc
  03's DST/DCT transforms and the banded vertical solves of §6.2 are
  designed-for consumers of the same schedule surface. The old
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
- **Fill modes.** `sync`'s `fills` carries per-name boundary values
  for bounded axes: `None` means periodic wrap (periodic mesh) or
  homogeneous fill (bounded mesh); a supplied array is the ghost-fill
  data resolved by `grid.sync` through the `("ghost_fill", space)`
  dispatch entry (§3.5, §3.6). The decomposition itself never touches
  the registry — the grid resolves, the decomposition moves bytes.
- Reductions need no dedicated methods: operators `unpad` to true
  shape and use `jnp` reductions on sharded arrays under jit
  (§3.13); `gather` covers host-side I/O.

### TensorDecomposition

The jax-sharding backend for tensor-product grids — the iteration-1
(and single-device) workhorse.

- Kind: final concrete (`Decomposition` subclass).
- Module: `fridom.framework.grid2.decomposition.tensor`.
- Pytree: static.
- Iteration: 1.
- Concept refs: [§5](../04_decomposition.md#5-domain-decomposition).

```python
@fr.utils.jaxify
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
  semantics by construction).
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
@fr.utils.jaxify
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

- **Merge call site** (inherited from
  [§3.4](../02_rules.md#34-generic-operator-dispatch), stays open):
  which Phase 2 assembly hook calls `grid.merge_overrides` and
  `grid.negotiate`. This cluster fixes the mechanism and the
  setup-phase-only mutation window, not the caller.
- **Name-binding lifecycle** (doc 01's open question 2, mirrored here
  because `Grid.__init__` is the bind call site): whether
  `mesh.bind_names(...)` stays a bind-time call made by the grid or
  names become mandatory at mesh construction. Either way the
  write-once contract and the no-reuse-across-grids restriction above
  stand.
- **Per-space refinement of `OperatorRequirements.layout`** (doc 03's
  open question 3, answered on this side as a negotiation detail):
  a `SpectralDerivative`-style operator is `layout="local"` only along
  its own coefficient factor. `negotiate` therefore interprets
  `.layout` per `(operator, factor space)` pair when scoping demands —
  whether doc 03 refines the declared surface to match, or negotiation
  keeps doing the per-space projection itself, is settled at
  implementation time.
- **Halo fallback without a tendency**: when `negotiate` gets no
  `tendency` to trace, is the per-operator max acceptable (with a
  documented un-synced-chain caveat), or should the grid refuse and
  require `halo=` explicitly?
- **Renegotiation vs live fields**: fields created before
  `grid.negotiate` hold arrays in the old layout. Derived arrays are
  covered by recompute-on-demand; for field data, is implicit
  resharding on next use acceptable, or should `negotiate` return a
  report the model uses to `device_put` its state once?
- **Product-space `measure` composition**: whether
  `measure(space, name=None)` on a multi-factor product should also
  offer the composed volume measure (the per-factor product), or
  whether that stays operator-internal (current choice: per-factor
  only, `name` required when ambiguous).
- **`local_slice` return type for graph backends**: tuple-of-slices is
  structured-only; the generalization (per-DOF global index array) can
  either widen the ABC signature now or be added as a parallel method
  when `GraphDecomposition` lands.
- **Tracer coverage of non-registry code paths**: `trace_halo` sees
  everything routed through dispatch and operator `__call__`; a module
  that drops to `.data` escapes the accounting. Whether the tracer
  should hard-error on `.data` access (forcing honesty) or warn is an
  implementation-time call.
