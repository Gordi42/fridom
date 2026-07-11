---
status: normative
date: 2026-07-06
---

# Grid abstraction redesign — Domain decomposition

Part of the grid redesign notes; see [`00_overview.md`](00_overview.md)
for the document map. Concepts are in
[`01_concepts.md`](01_concepts.md), rules in
[`02_rules.md`](02_rules.md).

---

## 5. Domain decomposition

Negotiation becomes **per mesh** instead of one global halo integer:

- each *mesh* declares shardability: a uniform axis is shardable with
  ghost-cell halos; a coefficient axis is either local-only or
  shardable via transpose-based transforms (distributed FFT, as
  `DomainDecomposition` already does); an unstructured mesh will need
  graph partitioning (not supported by jaxDecomp — later),
- each *operator* declares per-axis requirements (halo width along
  nodal axes, transform needs),
- grid setup collects (operator demands x mesh traits) and chooses the
  layout; this replaces the "rebuild decomposition on halo mismatch"
  logic and is what makes `uniform(x, y) ⊗ chebyshev(z)` decomposable:
  shard x/y, keep z on-device.

**Halo is a per-mesh quantity the grid derives, not a per-module
integer.** The old `Module.required_halo` scalar (max of the
`diff_module`/`interp_module` submodules) is removed: operators own
their per-axis halo, the module's dispatch overrides are merged into the
grid registry during assembly
([section 3.4](02_rules.md#34-generic-operator-dispatch)), and the grid
aggregates the per-mesh halo from the resolved registry (defaults +
overrides), scoped to the operators that can actually fire on the
model's state-field spaces. A simple max over operators is **not**
enough, because halo accumulates along **un-synced composition chains**:
`f.diff("x").diff("x")` with no intervening exchange needs
`halo_1 + halo_2`, not `max`. The grid therefore runs an **automatic
halo-accounting trace** over the tendency. (Composite operators of the
operator algebra carry these numbers as object-level rules — chains
sum, sums/blocks max — so a registered composite advertises its total
demand without a trace;
[operator design section 3.6](../operator_algebra/02_algebra.md#36-halo-accounting).)

**The trace uses a tracer field type**, analogous to a jax tracer: a
lightweight stand-in that carries only a space (hence shape) and an
accumulated per-axis halo-depth, no data array. It is passed through the
tendency in place of real fields; because it presents the same interface
as a `ScalarField`/`VectorField`, operators run **unchanged** — the
tracer *generically intercepts* each operator application, recording the
kind and growing the halo-depth by the operator's per-axis halo, while a
sync/halo-exchange resets it to zero. A shape/halo-only dry-run of the
tendency thus yields the per-mesh maximum accumulated depth, from which
the grid sizes the ghost layers (and can place the syncs), with no
per-operator tracer code and **no halo bookkeeping on real fields**
(whose metadata stays name/units/nc-attrs,
[section 2.4](01_concepts.md#24-field)). This is exact and
author-effort-free.

Per-space shapes
([section 3.5](02_rules.md#35-shape-is-a-property-of-the-space)) add one
more constraint: staggered
pairs shard unevenly along a bounded axis (n vs n + 1). Padding to a
uniform storage shape is the sanctioned mitigation; it is owned by the
decomposition layer and invisible above it. Slice-based stencils
(section 3.5) always address the **true** logical extent, so pad slots
never contribute; likewise the sharded random draw
([section 3.10](02_rules.md#310-discretizing-continuous-functions)) keys
per-shard by **global true-DOF index**, so its values never land in
padding and are identical for any device count.

The decomposition remains a lower layer that the grid *owns*; fields
and operators reach it only through the grid, and the transform API
must be rich enough that solvers no longer bypass it (lesson from
`RFFTPressureSolver`).

### 5.1 Layout is part of the function space

Owner decisions (2026-07-06), driven by two requirements the layer
split above cannot express: adding two fields in different shardings
must be a caught error, and layout-changing operations are
representation changes that belong in the type system. The forcing
counterexample: per-axis transforms reach the *same* logical
coefficient space in *different* pencils depending on order
(x → y → z vs z → y → x from an x/y-sharded start), so the layout is
not derivable from the logical space and must be carried.

- **`Layout` in the space.** A `Layout` is a pure combinatorial value
  mapping coordinate names to device-mesh axes (unmapped names are
  device-local). It is an *optional defining attribute* of the space,
  entering the interning key only when set (the `variance`
  precedent): mesh factories mint layout-free "bare" spaces; the grid
  mints laid-out variants after negotiation
  (`space.with_layout(...)`, `space.bare`, `space.layout`). Fields
  always live on laid-out spaces; dispatch keys and `codomain`
  resolvers stay bare. Halo widths and stagger padding remain
  decomposition-owned storage — *not* part of the space identity.
- **Strict algebra extension.** The join requires layout equality;
  the sanctioned lifts do not touch layout. The same bare space in
  two layouts raises `SpaceMismatchError` with a reshard hint. **No
  implicit reshard in field arithmetic, ever.** An operator
  *application*, however, may contain reshards demanded by its own
  declared requirements — declared, planned, static, and visible in
  the codomain's layout.
- **Reshard and Sync are operators.** `Reshard` is user-reachable
  and grid-bound: identity on the bare space, codomain = bare space
  with the target layout, kernel = `redistribute`, halo-trace rule =
  reset accumulated depth on moved axes. `Sync` (halo exchange) is an
  algebra node only the operator base / lowering ever inserts — never
  user-spelled, so storage stays invisible above the decomposition.
- **Requirements-driven lowering.** One plan-time pass over a bound
  or registered composite inserts both node kinds from
  `OperatorRequirements` (`.halo` → `Sync` placement, `.layout` →
  `Reshard` insertion). Inserted `Reshard`s carry explicit
  source → target layouts; apply time infers nothing. This is
  *lowering* — a compiled form beside the user-built composite — not
  algebraic rewriting of user graphs, which stays rejected.
- **Closed layout vocabulary.** The layouts negotiated at grid
  assembly (default + transform pencils + solver-declared) are all
  the layouts there are. They form a small graph — nodes = layouts,
  edges = transposes weighted by moved data volume — over which the
  planner picks reshard targets by shortest path with lookahead.
  Exotic layouts are declared at negotiation, not minted at runtime.
- **Transforms plan against this.** A multi-axis transform treats
  `axes` as a *set*: the planner reorders the 1D stages for maximum
  speed (locally-available axes first). For real fields the order
  fixes which factor carries the Hermitian half spectrum, so the
  logical codomain of a multi-axis real transform is
  schedule-determined — still deterministic per grid, since the plan
  depends only on static negotiation results. Users needing a
  specific half-spectrum factor compose single-axis transforms
  (`Fourier(axes="x") @ Fourier(axes="y")` pins the order). A
  transform chain ends in its final pencil; nothing reshards back to
  the default layout implicitly.
