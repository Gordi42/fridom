# Grid abstraction redesign — Rules

Part of the grid redesign notes; see [`00_overview.md`](00_overview.md)
for the document map and motivation. Core concepts are in
[`01_concepts.md`](01_concepts.md).

---

## 3. Rules

### 3.1 Strict space algebra

Combining fields on different function spaces (`f + g`, `f * g`, ...)
is a **hard error**. Conversions are explicit operators
(interpolation, transform, phase shift). Rationale: interpolation is a
physics choice (order, conservation properties); silent coercion hides
energy-conservation bugs. This check is cheap (space identity) and
jit-friendly.

**The scalars (real vs complex) are part of the space.** A function
space is defined over a field of scalars — its *Körper* — either
`fr.Real` or `fr.Complex`, read as `space.scalars` (default `fr.Real`;
[section 2.2](01_concepts.md#22-functionspace--where-discretization-strategy-lives)).
This is the honest home for real-vs-complex, which today is a side
effect of `is_spectral` (a spectral field is stored complex, a physical
field real): the two are now separate — *representation* is the
coefficient space
([section 3.2](#32-coefficient-representations-are-separate-spaces)),
*scalars* are the Körper. Four points fix the semantics:

- **Scalars are the Körper of the represented function, not the storage
  dtype.** The array dtype is *derived* from `(scalars, basis)`: real +
  Fourier stores a **complex, Hermitian half-spectrum** (the reality
  constraint made structural, section 3.2); real + sine/cosine stores
  **real** coefficients; complex + Fourier stores a **full complex
  spectrum**. So a `fr.Real` field can hold a complex array (its
  coefficients) while still denoting a real function.
- **The standard operators preserve scalars.** `diff`, `interpolate`,
  `grad`/`div`/`laplacian`, and transforms are (real-)linear maps that
  commute with conjugation, so they carry the Körper through unchanged
  — a transform changes the *representation*, never the scalars. The
  criterion is exact: **an operator preserves scalars iff it commutes
  with conjugation** (is real-linear). The scalar-*changing* operators
  are exactly the non-conjugation-commuting ones — the archetype is
  extracting a single complex Fourier mode of a real field (`fr.Real`
  half-spectrum -> a `fr.Complex` single-mode coefficient space),
  alongside the explicit `f.as_complex()` — and each declares the
  changed `scalars` in its codomain. There is no enumeration to
  maintain: the conjugation criterion decides membership.
- **`fr.Real` -> `fr.Complex` promotion is implicit and exact.** The
  embedding of the reals into the complexes is canonical and lossless
  (no physics or numerics choice), so mixing scalars in `f + g` / `f *
  g` promotes automatically to the complex space — the second
  sanctioned exception to strict algebra, alongside the `ConstantSpace`
  broadcast (section 3.3). `f.as_complex()` is the explicit form;
  `f.real` and `f.imag` extract the two `fr.Real` parts of a complex
  field.
- **Scalars drive transform dispatch.** Because `space.scalars` is part
  of the hashable key, `("transform", Real Center)` and
  `("transform", Complex Center)` resolve to different defaults — rfft
  vs full fft — with no special-casing, unlike today's
  `RFFTPressureSolver` bypass
  ([section 1](00_overview.md#1-motivation-pain-points-of-the-current-abstraction)).

### 3.2 Coefficient representations are separate spaces

Transforms are operators into **coefficient spaces**. A space admits
many coefficient bases (sine, cosine, Chebyshev, ...) and none is
canonical — "spectral" is only meaningful relative to an operator
whose eigenbasis the transform diagonalizes (Fourier for d/dx on
periodic meshes, eigenfunctions of the discrete Laplacian for a
pressure solver). Two rules make this unambiguous:

1. **The origin is constitutive of the coefficient space**: a
   coefficient space is *defined* by (basis, origin space) — e.g.
   `Fourier(origin=Center)` — so the inverse transform is never
   ambiguous, and no `is_spectral` flag exists. The origin even
   determines the shape (DST-I vs DST-II below).
2. **The default forward transform is a dispatch entry**
   `("transform", space)` (section 3.4), not a space property; other
   transforms are applied as explicit operators.

Examples:

- periodic `Center` -> `Fourier(origin=Center)`; periodic `Right` ->
  `Fourier(origin=Right)`. The two are distinct spaces related by the
  phase shift `e^{i k dx/2}` — exact on every mode except, for real
  origins with even n, the Nyquist mode (its shifted coefficient is no
  longer the rfft of a real field; the class design zeroes it and
  documents the one non-exact DOF, see
  [`classes/03_operators.md`](classes/03_operators.md)). Adding `u_hat`
  and `w_hat` without the shift is a caught error instead of a silent
  bug — the strict algebra pays off in coefficient space too.
- on bounded meshes, the BC structure selects *compatible* bases (not
  a unique one): Dirichlet `Center` -> DST-II coefficients (n modes);
  Dirichlet `Inner` -> DST-I coefficients (n - 1 modes — coefficient
  counts equal space shapes, section 3.5, which is why coefficient
  spaces must be per-origin); Neumann `Center` -> DCT-II
  coefficients.
- average spaces have their own coefficient spaces: cell-averaging is
  convolution with a top-hat, so the coefficient space of `CellAvg`
  carries a `sinc(k dx / 2)` factor relative to that of `Center` —
  another instance of "the coefficient space remembers its origin"
  (see also section 3.9).
- coefficient spaces of origins with **`scalars = fr.Real`**
  (section 3.1) carry the rfft-style half-spectrum layout as their
  *shape* (section 3.5), so the Hermitian constraint between paired
  modes is structural, not conventional — it is simply the reality of
  `fr.Real` scalars made into a shape, the coefficient-space analogue
  of "BC-constrained spaces silently project" (section 3.10). One
  caveat: the *realness* of the self-conjugate modes (k = 0 and, for
  even n, Nyquist) is a value constraint the shape cannot encode; the
  field factory projects it on assignment and operators must preserve
  it (class docs 02/04).
- Operator codomains express what today are metadata mutations: the
  bc-flipping hack of `SpectralDiff` becomes the honest signature
  `d/dx : SineCoeff -> CosineCoeff`.

### 3.3 `ConstantSpace` replaces `topo`, with automatic broadcast

A field that is constant along an axis has a `ConstantSpace` factor
(one DOF) on that mesh. Broadcasting a constant factor against a full
factor is **exact and unambiguous**, so it is one of two sanctioned
exceptions to the strict algebra (the other being `fr.Real` -> `fr.Complex`
promotion, section 3.1): `f_2d * f_3d` broadcasts automatically. Operators
along a `ConstantSpace` axis are identity or trivially defined; the
current family of "partial topo not supported" gaps disappears
structurally.

### 3.4 Generic operator dispatch

`f.diff("x")` resolves a default operator from the key
`(kind="diff", space_of_f_along_x)` in a registry:

- the **grid holds the default table** (e.g. nodal spaces -> FD of
  configured order; coefficient spaces -> the spectral derivative,
  which is the only choice),
- **modules can carry local overrides** (generalizing today's
  `module.interp_module = ...` pattern): a module holds a
  dispatch-override dict (sketch
  [4.2](03_api_sketches.md#42-custom-operator-module-local-override))
  that is **merged into the grid registry during model assembly** — the
  successor of the removed `diff_module`/`interp_module` submodule
  slots. An override entry may be keyed **either** by kind alone
  (`dispatch["reconstruct"] = op`, applying to every space that kind can
  fire on) **or** by `(kind, space)`
  (`dispatch[("reconstruct", space)] = op`, one space); the
  space-specific entry wins over the kind-only entry, which in turn wins
  over the grid default. Halo is then a per-mesh quantity the grid
  derives from the merged registry, not a module-owned integer
  ([section 5](04_decomposition.md#5-domain-decomposition)). (The exact
  hook that performs the merge — a `Module.setup(...)` method or another
  assembly step — is **an open question tied to the Phase 2 composition
  design** (ROADMAP 2.1/2.3), which removes `ModelSettings`; the notes
  describe only the merge *mechanism*, not its call site.)
- `grad` / `div` / `laplacian` are generic **dispatch kinds**, not
  special slots: their default entry on separable grids is a
  composition over `diff`/`interpolate`, but on non-separable meshes they
  are registered as primitive **mesh-level** operators (metric-aware
  grad/div on a sphere,
  [section 6.3](05_validation.md#63-sphere--curvilinear); `div:
  edge-normal -> cell` on an unstructured mesh,
  [section 6.4](05_validation.md#64-unstructured-horizontal-x-structured-vertical)).
  **A dispatch kind's default entry *is* an operator object**: the
  standalone `fr.operators.Laplacian(order=2)` (sketch
  [4.6](03_api_sketches.md#46-operator-eigenvalues-for-exact-spectral-solvers))
  and the `"laplacian"` kind resolved by dispatch are the same thing —
  the object is what the registry holds — so "not a special slot" means
  it has no privileged plumbing, not that no object exists. Registered
  objects may be composites/sums/blocks of the operator algebra, with
  kind placeholders resolved at assembly time
  ([operator design section 3.10](../operator_design/02_algebra.md#310-dispatch-integration)),
- arbitrary kinds are allowed (`"grad"`, `"div"`, `"laplacian"`,
  `"interpolate"`, `"reconstruct"`, `"filter"`, `"transform"`,
  `"integrate"`, `"discretize"`, `"assign_coeff"`, ...).

On a mixed grid the same `f.diff("z")` call therefore does the right
thing per axis — this is the point of the whole design.

**`f.to(target)` is the generic space-conversion sugar.** It brings a
field to another space, taking either a field or a space
(`g.to(f)` uses `f`'s space; `g.to(space)` also works), and dispatches
**per axis**. The registry key stays the single-space `(kind, space)`
used everywhere else: `.to` reads the conversion *kind* from the
source→target relationship along each axis — `"interpolate"` for nodal
spaces, `"reconstruct"` for the average/FV family (section 3.9), the
exact phase shift for coefficient spaces of differing origin
(section 3.2) — then resolves `(kind, source_space)`, whose registered
operator already fixes its own codomain (that codomain must equal the
target's factor, else it is a space error). There is **no separate
space-pair key**. Those kinds remain the underlying operators for
explicit control; `.to` is the concise,
readable spelling for the common case (sketches
[4.1](03_api_sketches.md#41-uniform-tensor-grid-staggered-derivative),
[4.7](03_api_sketches.md#47-worked-example-finite-volume-on-a-c-grid)).

### 3.5 Shape is a property of the space

Different spaces on one mesh generally have **different DOF counts**;
equal shapes are an artifact of periodicity that the current design
silently exploits (FACE fields are stored with n points, the boundary
face wrapped or masked). On a bounded `IntervalMesh` with n cells:

| Space                               | True DOF count |
|-------------------------------------|----------------|
| `Center` (nodal, cell centers)      | n              |
| `Outer` (all faces, BC-free)        | n + 1          |
| `Inner` (interior faces)            | n - 1          |
| `Left` / `Right` (periodic faces)   | n              |
| `CellAvg` (primal-cell averages)    | n              |
| `FaceAvg` (interior dual-cell avgs) | n - 1          |
| DST-I coeffs (of Dirichlet `Inner`) | n - 1          |
| DST-II coeffs (of Dirichlet `Center`) | n            |
| Chebyshev-Shen basis with 2 BCs     | n - 1          |

Rules:

- `space.shape` is the **true DOF count** and the only shape the
  abstraction knows; the shape of a product space is the concatenation
  of its factor shapes. The mesh only knows its cell count.
- **The BC decides boundary-DOF membership**: spaces with baked-in BCs
  exclude constrained DOFs. This makes the shape of a space and the
  coefficient count of its transforms *the same fact* (Dirichlet
  interior faces: n - 1 values <-> DST-I: n - 1 modes, section 3.2).
  BC-free
  spaces (`Outer`) keep boundary DOFs for unconstrained data such as
  prescribed boundary fluxes.
- Operator signatures encode shape changes exactly
  (`diff : Outer(n+1) -> Center(n)`); the strict space algebra
  (section 3.1) already rejects shape mismatches, so no additional
  user-facing rule is needed.
- **Storage padding is permitted below the operator layer** as a
  decomposition optimization (e.g. padding staggered pairs to a common
  storage shape for even sharding). It is invisible to the
  abstraction — like halos today. Operators are always written against
  true shapes.
- **Stencils are slice-based over halo-extended storage, not
  roll-based.** Today's stencils use `jnp.roll` and preserve shape
  (they assume periodicity). A true-shape operator instead reads its
  input at stencil offsets as **slice windows into the halo-extended
  local array** and writes the true-shape output, so a shape change
  (`diff : Outer(n+1) -> Center(n)`) is simply the differing slice
  lengths. **Periodicity stops being a stencil special-case**: on a
  periodic axis the halo is filled by wrap-around during sync (slicing
  it reproduces the old roll), on a bounded axis the same halo is
  filled by BC/ghost values ([section 3.6](#36-boundaries-i-conforming-bc-structure-vs-boundary-data)) —
  one slicing implementation, the difference living only in the
  halo-fill mode. The halo width is the operator's per-axis halo, and
  composition (`diff o diff`) slices into a wider halo per the
  halo-accounting rule
  ([section 5](04_decomposition.md#5-domain-decomposition)). Padded
  storage stays below the operator: it always slices against the true
  logical extent, and pad slots never contribute (the same dead-slot
  discipline as immersed dead-DOFs, section 3.7).

### 3.6 Boundaries I: conforming (BC structure vs boundary data)

A boundary condition has two parts with different lifetimes:

1. **Structure** (Dirichlet-type, Neumann-type, periodic): determines
   which DOFs are free, the space's shape (section 3.5), and the
   compatible coefficient bases (section 3.2). Static — lives in the
   space.
2. **Data** (the prescribed values: an inflow profile g(y, z, t), a
   surface flux): per-field, dynamic, often time-dependent. Lives
   outside the space.

Only the structure can live in the space, for two independent
reasons:

- fields satisfying an inhomogeneous BC (u = g on the boundary) form
  an affine set, not a linear space;
- spaces are static, interned jit-cache keys
  ([section 2.2](01_concepts.md#22-functionspace--where-discretization-strategy-lives)) —
  time-varying data baked into the space would force recompilation
  every step.

**Boundary data are trace fields.** Since `mesh.boundary` is itself a
mesh
([section 2.1](01_concepts.md#21-mesh--atomic-factor-of-the-domain)),
boundary data are ordinary fields on trace
product spaces — an inflow profile u(x=0, y, z, t) is a field on
`boundary(x) ⊗ Center(y) ⊗ Center(z)` — reusing the entire
field/space/operator machinery. A trace space is **not** a
`ConstantSpace` (section 3.3): `boundary(x)` is a located boundary
*mesh* (with coordinates and an area element, generally >1 DOF — two
points for a bounded interval, empty for a periodic axis, a closed
curve for a 2D mesh) restricted to the boundary manifold, whereas a
`ConstantSpace` is a geometry-less bulk reduction ("constant along x")
that broadcasts into the interior — precisely what boundary data must
not do.

Enforcement is one concept with a per-discretization realization:

| Discretization | Applying boundary data                          |
|----------------|--------------------------------------------------|
| FD (nodal)     | ghost-point fill during sync                     |
| FV (flux form) | boundary-face DOFs of a flux field on an `Outer` space (the slot provided by section 3.5) |
| Galerkin       | **extended basis**: homogeneous modes + boundary modes in one space; boundary values are ordinary coefficients (shenfun `BCGeneric` precedent; pure lifting and tau methods are the rejected alternatives) |

Decisions:

- **Boundary data is module-owned**: an `OpenBoundary`/`FluxForcing`
  module holds and updates the trace fields; prognostic fields stay
  `(space, array)`. Consistent with the rule that dynamic/stateful
  concerns live in modules (radiation BCs, sponges, and nudging are
  modules anyway).
- The strict algebra **permits** `u1 + u2` doubling boundary data:
  discretely well-defined; the BC-owning module defines physical
  correctness. No affine-space type is introduced.

Inhomogeneous-BC machinery (the three realizations made concrete):

- **The extended (inhomogeneous) Galerkin space is a distinct space.**
  A homogeneous Shen space with `k` baked-in BCs has shape `n - k`
  (section 3.5); its inhomogeneous variant is a *separate* space
  `(n - k) homogeneous modes ⊕ k boundary modes` of shape `n`, whose
  `k` extra DOFs are boundary modes carrying the prescribed boundary
  values as ordinary coefficients (shenfun `BCGeneric`). It is a
  distinct jit/dispatch key related to the homogeneous space by an
  **inclusion operator**; nothing is conditional on a flag, so the
  static-space-key design (section 2.2) is preserved. Assigning
  inhomogeneous BC data is assigning the boundary-mode coefficients.
- **Ghost-fill is a halo-fill mode, not a separate mechanism.** For
  nodal spaces the inhomogeneous BC is applied by `("ghost_fill",
  space)`, which writes the ghost layer from the module-owned trace
  field during sync — the third fill mode beside periodic-wrap and
  homogeneous fill (section 3.5). It reuses the halo machinery
  directly.
- **One `OpenBoundary`/`FluxForcing` module interface, three dispatched
  enforcements.** The module holds the module-owned trace fields on
  `boundary(mesh) ⊗ ...` spaces plus an update rule (radiation,
  nudging, flux forcing) and hands the *same* trace field to whichever
  enforcement the **target space's discretization** dispatches: nodal
  -> `ghost_fill`, FV -> boundary-face DOFs on the `Outer` space,
  Galerkin -> boundary-mode coefficients (the table above). The module
  is discretization-agnostic; the enforcement is dispatched by the
  space.

### 3.7 Boundaries II: immersed (masked domains)

Coastlines and topography define a boundary that is a curved
(n-1)-manifold cutting *through* cells — staircase-approximated on
structured grids. It differs structurally from the conforming case:
the wet/dry interface is not a product of factor boundaries, so no
trace product space exists.

**The mask is data, not space structure.** A "wet-DOFs-only" space
would be shape-honest but would destroy everything the tensor product
buys (separable operators, transforms, uniform sharding) — it would
turn every masked cartesian grid into an unstructured one. Therefore:

- `space.shape` counts the DOFs of the **product domain**, not the
  wet subset. Masked DOFs are stored-but-dead. This is a conscious,
  recorded exception to the spirit of section 3.5: the mask is a
  projection operator on the full space, not a space.
- Masked and unmasked fields share a space, so masking correctness is
  **not type-checked**; it is owned by operators and modules (as
  today).

Immersed BC structure and data:

- **Structure** (no-slip vs free-slip) = the choice of *derived
  staggered masks* (today's face mask = AND of adjacent centers is
  one such choice) plus mask-aware operator behavior; optionally
  ghost/mirror-fill operators for higher-order immersed-boundary
  methods.
- **Data** (e.g. river inflow through a coastal cell): the "trace" is
  the wet/dry transition set — not a product space — so data is
  represented as an ordinary **full-shape field supported on the
  transition set** (zero elsewhere). The mask machinery exposes
  transition-set indicator fields per space. Ownership is
  module-side, as in section 3.6.

Fidelity ladder (all designed-for; iteration 1 implements boolean
masks only):

1. **boolean mask** — first-order staircase;
2. **cut-cell fraction fields** — wet volume fractions on center
   spaces, wet area fractions on face spaces, entering integrals and
   flux operators as weights; the boolean mask is the special case
   fraction in {0, 1};
3. **ghost-cell immersed-boundary fill** — an operator kind;
4. **volume penalization** (Brinkman) — a pure forcing module;
   notably the standard route for masked domains on spectral bases.

Immersed-boundary integrals (e.g. boundary flux diagnostics) have no
trace space to integrate over; they are volume integrals with
indicator/fraction weights (level-set style; a level-set
representation of the mask is a natural future extension).

**The immersed-domain object.** The grid-owned successor of `WaterMask`
is an `ImmersedDomain`, reached as `grid.immersed`, designed to mirror
the coordinate-data machinery
([section 2.7](01_concepts.md#27-where-coordinate-data-lives)) so it adds
no new mechanism:

- **Single stored datum:** a wet **volume-fraction field** in `[0, 1]`
  on the base cell space (`Center`/`CellAvg`), an ordinary full-shape
  dynamic sharded `ScalarField`. The boolean mask is the `{0, 1}`
  special case (fidelity ladder point 1 is a special case of point 2),
  so there is one representation, not two. A level-set (signed
  distance) field on the same cell space is the future generalization
  from which fractions become derived.
- **Derived per-space masks/fractions are grid-mediated**, like
  `grid.evaluation_nodes(space)`: `grid.immersed.fraction(space)`,
  `grid.immersed.mask(space)`, and `grid.immersed.transition(space)`
  (the wet/dry indicator feeding module-owned immersed-BC data) return
  fields *tagged with that space*, computed from the base fraction by a
  staggering-transfer rule. **Structure (no-slip vs free-slip) is the
  combination rule** used to derive staggered masks (today's
  `face = AND(adjacent centers)` is the no-slip choice), a parameter of
  the derivation, not stored state. Whether these derived staggered
  masks are memoized is a below-the-operator-layer optimization to
  benchmark (like storage padding, section 3.5), bound by section 2.7's
  re-decomposition-invalidation rule; the interface is derive-on-demand
  regardless.
- **Mask-awareness adds no registry axis.** Mask-aware operators are
  ordinary dispatch entries (`reconstruct`, `flux`, `integrate`) whose
  registered operator simply *consults* `grid.immersed.fraction(space)`
  as a weight — the same grid-materialized-array status as `dx`
  (section 2.7), so cut-cell fractions enter integrals and flux
  operators as weights (ladder point 2) with zero registry changes.

Precedent: Oceananigans' `ImmersedBoundaryGrid`
(`GridFittedBottom`/`PartialCellBottom`) — a grid-owned immersed
domain consulted by mask-aware operators; this is the `grid.immersed`
object above
([section 2.6](01_concepts.md#26-grid--the-assembly-object)).

### 3.8 Boundaries III: terrain-following (boundary-fitted)

The complementary, equally first-class route for topography: map the
vertical coordinate so the boundary becomes conforming, e.g.
z = sigma * H(x, y). Both routes coexist and may be combined (e.g.
sigma coordinates over smooth topography plus partial cells for steep
features — standard ocean-model practice).

- The mapping depends on other factors' coordinates, so it is
  **grid-level metric data**, not mesh structure
  ([section 2.1](01_concepts.md#21-mesh--atomic-factor-of-the-domain)): the
  *topological* product survives (DOF layout, shapes, staggering,
  sharding — sections
  [2.3](01_concepts.md#23-tensorproductspace-and-named-coordinates),
  3.5), while the *geometric* product is
  coupled through metric fields (H, dH/dx, Jacobians) living on
  product spaces.
- Physical-space operators become **sums of separable operators with
  field coefficients**, e.g.
  `d/dx|_z = d/dx|_sigma - (sigma H_x / H) d/dsigma`.
  "Derivative at constant z" and "derivative at constant sigma" are
  *different dispatch kinds*; the generic registry (section 3.4)
  resolves both. This is the same machinery the sphere needs
  ([section 6.3](05_validation.md#63-sphere--curvilinear)), appearing on
  cartesian grids.
- At the mapped boundary, BCs are conforming again: the bottom is a
  coordinate surface, so BC structure returns to the space picture
  (section 3.6) — the payoff that motivates the route.
- Time dependence (z*, free-surface-following) is inherited from the
  general rule in
  [section 2.3](01_concepts.md#23-tensorproductspace-and-named-coordinates):
  metric fields may be dynamic; no
  operator may assume static metrics.

**The coordinate-mapping object.** A grid-attached `CoordinateMapping`
declares the transform and is the single owner of the metric data:

- **Declaration accepts both forms.** Either an *analytic* coordinate
  map as a function of base coordinates and named parameter fields
  (`z = sigma * H(x, y)`), from which the grid derives the metric
  fields (`H_x`, `dz/dsigma`, Jacobians) by differentiation, **or**
  user-supplied metric fields directly for cases with no closed form.
  The mapping is attached at grid build (`meshes=..., mapping=...`).
- **Metric fields are grid-derived per staggered space on demand**:
  `grid.metric(name, space)` returns the metric on whatever staggered
  space an operator asks for (H at u-, v-, w-points), mirroring
  `grid.evaluation_nodes(space)` — grid-mediated, per-space,
  derive-on-demand, with the same recompute-vs-cache benchmark caveat
  as the immersed masks (section 3.7). The accessor works uniformly
  whether the source is an analytic map or supplied fields (which it
  reconstructs to the requested space), so **staggered consistency is
  guaranteed by the grid, not per module**
  ([section 6.5](05_validation.md#65-terrain-following-vertical-coordinate)).
- **Time dependence is automatic**: when a parameter field is
  prognostic (free-surface `H = H_0 + eta(t)`, z*), the derived metrics
  are dynamic fields recomputed from the current parameter values each
  step; no operator caches them (section 2.3). The metric-coefficient
  operator composites (constant-z vs constant-sigma dispatch kinds
  above) read them through `grid.metric`.

### 3.9 Finite-volume semantics: the average family and the FV derivative

The FV strategy gets its own space family and its own derivative
factorization.

**The average family.** `CellAvg` holds averages over primal cells;
`FaceAvg` holds averages over dual cells (the intervals between cell
centers, centered on faces). The family parallels the nodal
`Center`/`Right` pair; relations between its members are operator
signatures, as always
([section 2.2](01_concepts.md#22-functionspace--where-discretization-strategy-lives)).
Along one axis, two distinct
1D spaces therefore coexist *at* a face: `Right`/`Outer` (point value
at the face) and `FaceAvg` (average over the dual cell around the
face) — the same nodal-vs-average distinction as `Center` vs
`CellAvg`, shifted by half a cell.

**The FV derivative factors into exact + approximate.** By the
fundamental theorem of calculus, the cell average of a derivative is
a flux difference of point values at the faces:

```
(1/dx) int_cell_i du/dx dx = ( u(x_{i+1/2}) - u(x_{i-1/2}) ) / dx
```

so the natural FV derivative is a composition:

```
reconstruct : CellAvg(n) -> Outer(n+1)   # APPROXIMATE: order,
                                         # upwinding, WENO live here
flux_diff   : Outer(n+1) -> CellAvg(n)   # EXACT: discrete Gauss
                                         # theorem

diff = flux_diff o reconstruct
```

(spelled `flux_diff @ reconstruct` in the operator algebra, where the
composed symbol verifies the exactness identity below —
[operator design sections 3.2/3.7](../operator_design/02_algebra.md#32-composition-c--a--b))

Consequences:

- **Conservation is visible in the type**: summing the output of
  `flux_diff` telescopes to the boundary fluxes by construction. The
  mimetic property is carried by the exact operator; all
  approximation error lives in the `reconstruct` dispatch kind —
  which is why advection schemes override reconstruction, not diff
  (sketch [4.2](03_api_sketches.md#42-custom-operator-module-local-override)).
- **The eigenvalue machinery verifies exactness**
  ([section 2.5](01_concepts.md#25-operator--typed-maps-between-spaces)):
  cell-averaging has Fourier eigenvalue `sinc(k dx / 2)`; the
  face-difference eigenvalue is `2 i sin(k dx / 2) / dx
  = i k sinc(k dx / 2)` — identically the eigenvalue of
  "average of d/dx".
- **The measures are themselves staggered metric fields.** The cell
  width `flux_diff` divides by is a field on `CellAvg`/`Center`, while
  the center-to-center spacing a `Center -> Right` derivative divides
  by is a field on `Right`/`Outer` — the same quantity `dx` lives on
  different spaces depending on which measure it is
  ([section 2.7](01_concepts.md#27-where-coordinate-data-lives)). Uniform
  meshes collapse both to one constant.
- **Conservation-minded schemes are choices, not defaults**:
  energy-conserving Coriolis/advection discretizations (Sadourny,
  Arakawa-Lamb) are specific operator combinations selected by
  modules; the framework provides type-correct plumbing (the
  section 3.1 rationale — "interpolation is a physics choice" —
  applied to staggering conversions).

### 3.10 Discretizing continuous functions

Setting a field from a function of physical coordinates
(`lambda x, y: x * y`) is **projection into the space** — an operator
like any other, with dispatch kind `("discretize", space)`:

| Space kind  | Default discretization                              |
|-------------|------------------------------------------------------|
| nodal       | evaluate at the node set (collocation)               |
| average     | per-cell quadrature (midpoint rule is the 2nd-order shortcut, section 3.9) |
| coefficient | evaluate at the *origin space's* nodes, then apply the forward transform |

Rules:

- Every space has **evaluation nodes** as an intrinsic property
  ([section 2.2](01_concepts.md#22-functionspace--where-discretization-strategy-lives)):
  its own nodes (nodal), quadrature points (average),
  or delegated to the origin (coefficient spaces), where
  `discretize = transform o discretize_origin` — composition; the
  "origin is constitutive" rule of section 3.2 at work. This is how a
  Fourier-space field is initialized from a physical-coordinate
  function.
- Evaluation happens at the **physical** coordinates of the
  evaluation nodes: staggering (face-x/center-y nodes for
  `Right(x) ⊗ Center(y)`), terrain-following metrics (section 3.8),
  and unstructured per-DOF coordinates are all handled by the same
  mechanism. Under jit these are ordinary sharded arrays; a function
  written with jnp operations traces through.
- **Function arguments are matched to coordinate names by keyword**
  (`def u_ini(x, y)`), independent of axis order; `ConstantSpace`
  factors simply do not appear in the signature.
- **Construction is functional** (jax style):
  `grid.create_field(space, init=u_ini)` is the **single field
  factory** (`space` is a positional, optional argument defaulting to
  the all-`Center` nodal space). A separate `Field.from_function`
  classmethod was rejected as a redundant second spelling, and a
  mutating `f.set(...)` was rejected too. `discretize` is a **pure,
  traceable operator**, so
  construction is legal *inside* the model's jit loop (e.g. re-seeded
  forcing each step). There is **no lazy/deferred field**: a field is
  always `(space, array)`, a uniform pytree. "Compute only what is
  needed" is delivered by XLA dead-code elimination (an `init` whose
  output feeds nothing is pruned at trace time), and "recompute instead
  of store" by jax rematerialization — not by a stateful field that
  caches on first use, which would reintroduce the rejected mutation
  and destabilize the pytree.
- **`init=` is the single path for discretizing continuous functions
  of physical coordinates — but not the only way to make a field.**
  Two companions exist; they take different *input kinds*, they are
  not competing discretization semantics:
  - **direct data**: `grid.create_field(space, data=arr)`, for any
    space. The grid factory owns sharding/layout validation.
    Operators construct fields from arrays internally anyway; this is
    merely the public spelling of that constructor.
  - **coefficient-native init**: for coefficient spaces,
    `grid.create_field(space, init_coeff=lambda kx, ky: ...)`
    evaluates the function at `grid.wavenumbers(space)`, keyword-matched
    to the wavenumber coordinate names — the exact mirror of nodal
    evaluation. Semantics: **assignment of coefficients, not
    projection**; its own dispatch kind `("assign_coeff", space)`
    alongside `("discretize", space)`. `init=` and `init_coeff=` are
    mutually exclusive, and reusing `init=` with keyword-name
    matching (`kx` vs `x`) was rejected as too implicit. On
    coefficient spaces whose basis is not wavenumber-indexed
    (Chebyshev, Shen), the matched coordinates are whatever
    `grid.wavenumbers(space)` generalizes to — the coefficient space's
    own **mode indices**, an intrinsic coordinate of the space. These
    are
    distinct from an operator's **eigenvalues**, which are a *derived
    per-operator field* over that index
    ([section 2.5](01_concepts.md#25-operator--typed-maps-between-spaces));
    fixing this generalization is part of that eigenvalue machinery.
- **Randomness is neither** a physical function nor a coefficient
  function: seeded per-DOF data enters as
  `grid.random.normal(space, seed)` / `grid.random.phase(space, seed)`
  — grid methods returning fields on the given space (grid-mediated,
  like every sharded-array accessor,
  [section 2.7](01_concepts.md#27-where-coordinate-data-lives)). The
  *values* are a pure function of `(space.shape, seed)` over the global
  DOF index, hence layout-independent; only the *sharding* is applied
  by the grid — which is exactly what makes them **deterministic and
  sharding-consistent**. This is the successor of
  `grid.create_random_array`; random-spectra initial conditions are
  the motivating consumer
  (sketch [4.9](03_api_sketches.md#49-random-spectra-initial-condition-spectral-space-construction)).
  The efficient realization is a **sharded draw with per-shard PRNG
  keying by global true-DOF index**: because jax threefry is
  counter-based, each device keys `fold_in(seed, global_index)` for its
  local DOFs only and draws directly into its shard — no global
  materialization, and layout-independent by construction (the key is
  the global index, not the local position). The draw is over the space's
  **true shape** (section 3.5), so pad slots are outside the index
  space and **random values never land in padding**
  ([section 5](04_decomposition.md#5-domain-decomposition)) with no
  special rule. `random.normal` on a `fr.Complex` space draws a complex
  normal (independent real/imag); `random.phase` draws unit-modulus
  `e^{i theta}` with uniform `theta`; on a real-origin Fourier space
  the draw covers only the free half-spectrum and the Hermitian
  structure follows from the shape (section 3.2), the real-only
  `k = 0`/Nyquist DOFs handled as special indices. The draw is pure and
  traceable under jit, returning a `(space, array)` field.
- The default is **collocation projection**, which aliases high
  wavenumbers; the true L2/Galerkin projection is a different,
  registrable operator.
- **BC-constrained spaces silently project**: discretizing a function
  that violates the baked-in BCs discards the incompatible part (the
  function-space analogue of "masking is not type-checked",
  section 3.7).
- **`grid.coordinates()` and coordinate-field algebra are removed.**
  Returning coordinate *fields* (`x, y = grid.coordinates();
  f = x * x + fr.sin(y)`, as in the presentation slides) was
  considered and rejected: it is space-ambiguous (coordinates on
  which space?) and meaningless in coefficient spaces (projecting `x`
  into a Fourier basis is Gibbs nonsense). `discretize` via `init=`
  functions is the **single, uniform discretization path for every
  space** (with `data=`/`init_coeff=` as the non-discretizing
  companions above). Coordinates survive only as grid-mediated
  accessors —
  `grid.evaluation_nodes(space)` (nodal/average spaces) and
  `grid.wavenumbers(space)` (coefficient spaces) — public, for solver
  and diagnostic writers. These are **`ScalarField`s tagged with the
  querying space** (`.data` for the raw array), consistent with
  operator eigenvalues
  ([section 2.5](01_concepts.md#25-operator--typed-maps-between-spaces));
  the rejected construct is the *free-floating, grid-level*
  `grid.coordinates()`, not a space-keyed coordinate field. They are
  **dynamic, device-sharded fields the grid recomputes on demand** to
  the local shard matching field layout, not static properties stored
  on the space
  ([section 2.7](01_concepts.md#27-where-coordinate-data-lives)).
  **The `space` argument is deliberately mandatory** — there is no
  no-argument default-space convenience form. Even casual coordinate
  access (plotting, diagnostics) must name the space, because the whole
  point of removing `grid.coordinates()` is to force the "coordinates on
  *which* space?" question to be answered explicitly; a silent default
  would reintroduce exactly the ambiguity being removed.

### 3.11 Field operations: linear ops and the product problem

Field arithmetic splits into two regimes by whether the operation
commutes with discretization.

**Linear operations are representation-independent.** `f + g`, `f - g`,
and scalar multiplication are linear, and sampling/transforms are
linear, so they commute with discretization: adding nodal values and
adding Fourier coefficients denote the same field. Their meaning is
therefore identical in every space, and the only rule is the strict
algebra (section 3.1) — operands must share a space, checked by
identity. This is *why* `f + g` needs no per-space realization.

**Multiplication does not commute with discretization**, so it is not
free in the same way:

- nodal collocation: elementwise multiply *aliases* (the product of two
  band-limited fields carries higher wavenumbers than either factor);
- coefficient space: the pointwise product of two functions is a
  *convolution* of their coefficients, not the elementwise product;
- average/FV space: the product of cell averages is not the cell
  average of the product (section 3.9).

Choosing among these realizations (aliasing, dealiasing rule,
quadrature order) is a numerics choice, so — like interpolation
(section 3.1) and staggering conversions (section 3.9) — products are
**explicit operators, never a silent array multiply**.

**Products are binary operators.** The operator signature
([section 2.5](01_concepts.md#25-operator--typed-maps-between-spaces))
generalizes from unary `(domain -> codomain)` to binary
`(domain_a, domain_b) -> codomain`; operators are callable
*instances*, constructed then applied, so a product is spelled
`CollocationProduct()(f, g)`, `Convolution(t)(f, g)`, or
`Hadamard()(f, g)` — never as a class call (this keeps operators
uniformly constructed-then-applied static structure; class-design doc
[`classes/03_operators.md`](classes/03_operators.md)). The dispatch
registry (section 3.4) grows the
operand: `("multiply", space)` resolves the default product for fields
on that space.

**`*` is the physical (pointwise) product**, the
representation-independent *meaning* ("the field whose continuous
representation is `f g`"), realized by dispatch per space:

| Space kind  | Operator resolved by `*`                        |
|-------------|-------------------------------------------------|
| nodal       | `CollocationProduct` (aliased unless dealiased) |
| coefficient | `Convolution` (a padded-transform composite, section 3.12) |
| average     | quadrature product (reconstruct, then multiply) |

So `f * g` means the same thing in every space, exactly as `f + g`
does; only the realization is dispatched. "Same meaning" is at the
*continuous* level: the product evaluated on nodal fields (aliased) and
on coefficient fields (dealiased) yield different arrays, just as
FD-`diff` and spectral-`diff` differ discretely while both denoting
`d/dx`.

Two rules keep `*` uniform with `+`:

- **Same-space requirement.** `f * g` demands identical spaces, like
  `+`; the only implicit exceptions are the `ConstantSpace` broadcast
  (section 3.3) and `fr.Real` -> `fr.Complex` promotion (section 3.1).
  A staggered product (`u` on `Right(x)` times `b` on
  `Center(x)`) is illegal until an explicit interpolation brings both
  to a common space; in coefficient space, multiplying
  `Fourier(origin=Right)` by `Fourier(origin=Center)` requires an
  explicit phase shift first (section 3.2). The conversion is the
  physics choice, made visible.
- **Dealiasing is carried by the bracketing transforms, not the
  product** (the 2/3 and 3/2 rules): the product itself stays a plain
  `CollocationProduct`, computed on a grid the padded transforms make
  fine enough. This is spelled out in section 3.12.

**The coefficient-wise product is a different operator.** Elementwise
multiplication of two fields on the *same* coefficient space is
`Hadamard`, not `*`: it is the transform dual of the physical product —
elementwise in coefficients corresponds to *convolution* in physical
space — and models filtering, eigenvalue application, and
amplitude/phase scaling of a spectrum (sketch
[4.9](03_api_sketches.md#49-random-spectra-initial-condition-spectral-space-construction)).
Because it has no representation-independent meaning, it is always
named explicitly (`Hadamard()(f, a)`) and never spelled `*`. The two
products coincide on nodal spaces — where the DOFs are local and
elementwise multiply *is* the collocation product — and diverge only on
coefficient and average spaces, where a single output DOF depends on
many input DOFs.

**Diagonal operators are `Symbol`s, applied as `Hadamard`.** An
operator's eigenvalues (section 2.5), a spectral filter, the 2/3
truncation mask (section 3.12), an inter-origin phase shift, and the
`sinc(k dx / 2)` averaging factor (section 3.2) are all *diagonal
operators* in coefficient space. They are a distinct `Symbol` type, not
`ScalarField`s: on a `Symbol` the operations `*`, `**`, `+`, and
`1 / .` are the diagonal (elementwise) algebra — composition and
inverse of diagonal operators — so `k ** 2` and `1 / lap` are
well-defined, whereas the same spellings on a field would mean the
physical product/convolution. Applying a `Symbol` to a field *is* the
`Hadamard` multiply, so the spectral solve `p_hat = (1 / lap)(-div_hat)`
(sketch [4.6](03_api_sketches.md#46-operator-eigenvalues-for-exact-spectral-solvers))
stays inside this section's algebra.

### 3.12 Dealiasing

Nonlinear products alias: the pointwise product of two fields
band-limited to N modes carries content up to 2N modes, which folds
back onto the resolved wavenumbers on an N-point grid (section 3.11).
More generally, a degree-p product needs a grid of at least
`(p + 1)/2 * N` points to be computed without aliasing.

**Dealiasing is a property of the transform, not of the product.** The
physical product is always the same operation (`CollocationProduct`);
aliasing is controlled by *which grid the product is computed on* and
*what is kept afterward* — i.e. by the transforms that bracket it. The
degree-p pad factor `(p + 1)/2` parameterizes a padded transform; the
two classical rules are the two strategies at degree p = 2:

- **3/2 rule (pad).** The padded inverse transform maps into a *finer*
  nodal space, `Fourier(N) -> Center((p+1)/2 * N)`; the product is an
  ordinary `CollocationProduct` on that finer space (exact, no
  aliasing); the forward transform trims back to `Fourier(N)`. The
  finer space is a genuine first-class space — its shape is honest
  (section 3.5) — being the codomain of the padded transform.
- **2/3 rule (truncate).** Instead of padding up, shrink the retained
  band to `2N/(p+1)` modes with a spectral **truncation filter** (a
  fixed 0/1 diagonal on `Fourier(N) -> Fourier(N)`, a `Hadamard`); the
  top band the aliasing error folds into is discarded. Cheaper (no
  larger transform), lossy (fewer active modes).

Both strategies reuse primitives the design already has — a padded
transform (3/2) or a truncation filter (2/3) — so dealiasing adds no
new primitive. `Convolution` (the coefficient-space realization of `*`,
section 3.11) is therefore *defined* as the composite

```
Convolution = trim_transform o CollocationProduct o pad_inverse_transform
```

reading its pad factor from the padded transforms. (In the operator
algebra this is a literal composite — `t.forward @ CollocationProduct()
@ tp.backward`, with binary pre-composition applying `tp.backward` to
each operand;
[operator design section 3.8](../operator_design/02_algebra.md#38-binary-operators-in-chains).)
The default factor
is a grid-level dispatch entry (section 3.4); modules override it
locally (the successor of today's `SpectralAdvection(padding=...)`).

**Efficiency is a scheduling concern.** The performant idiom transforms
a whole state to the finer space **once**, computes many plain
`CollocationProduct`s there, and transforms the results back once —
paying one transform pair rather than one per product (sketch
[4.10](03_api_sketches.md#410-dealiased-pseudospectral-product-transform-once)).
`Convolution` gives the same result in a single call but re-transforms
shared operands.

**Scheduling is explicit, via a combinator.** The transform-once
schedule across a multi-term right-hand side is delivered by a reusable
**transform-once combinator** — a pure higher-order operator that
formalizes sketch 4.10: it does one padded `backward` per distinct
coefficient operand into the finer nodal space, runs a user function
that computes plain `CollocationProduct`s there, and does one
`forward`+trim per named output. It is **eager** (no lazy field) but
boilerplate-free and transform-once by construction. **Automatic
scheduling is rejected**: a fully automatic scheduler needs a deferred
coefficient-expression graph, which conflicts with section 3.10's
already-made decision that there is no lazy/deferred field — the same
reason `f.set(...)` and self-caching fields were rejected. XLA
common-subexpression elimination is the backstop that de-duplicates
any `backward` a module still writes redundantly, so the combinator is
about *guaranteeing* the fused schedule, not the only thing preventing
recompute.

### 3.13 Reductions and integrals

Reductions are ordinary dispatch kinds (section 3.4 already lists
`"integrate"`), distinguished by what they contract against and the
codomain they land in.

- **`integrate` reduces a factor to `ConstantSpace`.** `f.integrate("x")`
  has signature `Center(x) ⊗ Center(y) -> ConstantSpace(x) ⊗ Center(y)`;
  the result broadcasts back via section 3.3, so `f - f.integrate("x")`
  (remove the x-mean) stays inside the strict algebra. A global integral
  reduces every factor to `ConstantSpace`. Reductions along a
  `ConstantSpace` axis are identity.
- **Quadrature weights come from the space, reusing the metric-measure
  fields** of sections
  [2.7](01_concepts.md#27-where-coordinate-data-lives) and 3.9 — no new
  data. `integrate` contracts the field against the space's
  quadrature-weight field: uniform `dx` for `Center`/`CellAvg`,
  Clenshaw-Curtis for Chebyshev, the stretched cell-width field for
  mapped meshes, Jacobian weights (`dz = H dsigma`) for
  terrain-following coordinates (section 3.8). It is **exact on average
  spaces** (sum of average x cell-measure) and the node-set quadrature
  rule on nodal spaces. Weights **compose per mesh**
  ([section 6.2](05_validation.md#62-uniform-fv-x-chebyshev-galerkin)).
- **Only the weighted integral is a field operator.** There is no
  separate unweighted `sum` operator; a raw unweighted DOF sum is an
  array escape hatch (`f.data.sum()`), not a space-tagged reduction.
- **Cumulative integrals are operators too** (`cumsum`/`cumint`); the
  staggering change they induce (e.g. a running integral landing on a
  face space, the discrete-FTC partial inverse of `flux_diff`,
  section 3.9) is the operator's decision and is left to the operator,
  not fixed by this section.
