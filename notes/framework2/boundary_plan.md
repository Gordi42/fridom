# Boundary-closure plan — resolving the two open boundary problems

**Status: options analysis + staged plan, awaiting owner sign-off on
R1–R4 below (2026-07-07).** The two problems, opened as
[`classes/spaces.md`](classes/spaces.md) open questions 4 and 1:

1. **BC-free bounded spaces** — exterior values must be untouchable
   (analysis in [`bc_free_boundaries.md`](bc_free_boundaries.md));
   what replaces the extrapolation fill.
2. **Robin / mixed BCs** — what BC structure enters the static space
   key vs what stays dynamic data (recorded constraints: float
   parameters in the key mean recompile-per-value under the Phase-3
   single jit and foreclose autodiff through / module updates of BC
   parameters).

They are one problem seen from two sides, and one principle resolves
both.

## 1. The principle

**The space key carries only the boundary's *structure* — the
per-side closure shape that determines DOF counts, fill stencils,
and row legality. Every boundary *value* is dynamic data. The
storage layer never invents values: each closure is (a) structural
(mirror fills), (b) data-parameterized (declared boundary data
through a dispatch row), or (c) an explicitly chosen one-sided
scheme. Nothing else exists.**

The machinery is largely in place: `BCStructure` is already a
normalized **per-boundary-component tuple** of kinds inside the
intern key (mixed D/N per side is expressible today), fills already
dispatch per side (`_boundary_geometry(factor, side)`), operators
already carry dynamic leaves (`ScaledOperator`,
`jaxify(dynamic=("coeff",))`), the `("ghost_fill", space)` registry
kind and `grid.sync(boundary_data=...)` are designed-for seams, and
requirements already express axis-locality demands
(`layout="local"`).

## 2. Problem 1 — what replaces the BC-free fill

### R1: adopt the legality rule (recommended)

Remove the BC-free bounded extrapolation fill; an operator row
exists on a BC-free bounded operand iff its true-shape output needs
no exterior values (`Center -> Inner`, `Outer -> Center` stay;
`Inner -> Center`, `Center -> Outer` demand BC structure). Missing
rows raise `DispatchError` with a hint naming the BC-structured
alternatives. Rationale and the `d²` counterexample:
[`bc_free_boundaries.md`](bc_free_boundaries.md). Rejected
alternatives (shrinking codomains, defined-region masks) are
recorded there.

### R2: one-sided stencil rows as the explicit opt-in closure

Where no BC is physically available (open boundaries, diagnostics),
the honest replacement is a **one-sided boundary scheme the user
chooses**: interior windows unchanged, the boundary-adjacent
outputs patched from one-sided windows over true DOFs only.

- **Spelling** — options:
  - (a) constructor knob: `FiniteDifference(order=2,
    boundary="one_sided")`, a distinct interned variant registered
    as an override row; **recommended** — it reuses dispatch, keeps
    the default rows closed under R1, and the choice is visible in
    the operator's repr;
  - (b) separate op classes (`OneSidedFD`) — same effect, more
    surface;
  - (c) a per-space "closure mode" in the key — rejected: the
    closure is per-operator numerics, not a property of the field's
    space (that conflation is exactly the extrapolation-fill
    mistake).
- **Multi-device**: one-sided kernels must know they are at a
  *physical* edge. Rather than making kernels shard-position-aware
  (the design forbids it), the row declares
  `OperatorRequirements(layout="local")` — negotiation keeps that
  axis undistributed (or the lowering inserts a `Reshard`), and the
  kernel patches the two ends of the local axis unconditionally.
  Bounded axes are typically the small vertical one, so the
  locality demand is cheap in practice.
- **Scope**: FD and interp first; reconstruct/WENO one-sided
  variants only when a concrete model needs them.
- **Validity claim**: reset on the applied axis (one-sided patches
  don't commute with any fill) — no change to the task-1.8
  machinery.

### Consequence for sync on BC-free bounded axes

After R1 the physical-boundary ghost slots of BC-free spaces are
never filled and never readable: sync fills interior shard edges
only. Per-axis validity stays as is (claims already reset on
bounded axes); the side-aware refinement is stage 2f, not a
prerequisite.

## 3. Problem 2 — Robin and mixed BCs

### R3: structure in the key, kinds only; all BC data dynamic

- `BC` gains `ROBIN`. The key holds per-side **kinds only** — never
  the Robin coefficient α, never inhomogeneity values. Shape rule
  extends per side: a side drops its boundary DOF iff its kind is
  DIRICHLET on a member node set (Robin keeps DOFs, like Neumann —
  the boundary value is not known a priori).
- Mixed kinds per side (`bc=(BC.DIRICHLET, BC.NEUMANN)` — ocean
  vertical) are already expressible in `BCStructure`; grounding
  them is verification + rows, not new structure.
- **Fills**: homogeneous Dirichlet/Neumann mirrors exist. The
  homogeneous Robin fill is affine in the near-boundary DOF with
  coefficients from (α, dx) — with α a *dynamic scalar*, the fill
  is jit-stable across α values and autodiff-able, satisfying both
  recorded constraints. Robin fills reach the sync through the
  data path below, not as a new hardcoded mirror.

### The boundary-data path — options

How do dynamic values (Robin α, inhomogeneous g) reach a sync that
fires *implicitly* inside an operator application?

- (a) **BC data as field leaves** — fields on BC-structured spaces
  carry their boundary data as extra dynamic pytree leaves.
  Rejected: it fattens every field, and arithmetic must define how
  BC data combines (fine for linear ops, undefined for products) —
  a correctness trap baked into the core type.
- (b) **`("ghost_fill", space)` rows holding dynamic leaves** —
  **recommended**. A grid-bound `GhostFill` operator instance per
  BC-structured space, registered at assembly, carrying (α, g) as
  dynamic jaxified leaves (the `ScaledOperator` precedent). The
  consumption-side sync dispatches it exactly where the
  designed-for seam already says; modules own and update the
  values (model D-decisions: parameters live in modules; the grid
  joins the traced state in ROADMAP 2.3, making updates traceable).
  `grid.sync(boundary_data=...)` stays as the explicit override of
  the same row.
- (c) **Homogeneous-only grid + lifting** — inhomogeneity handled
  by writing `u = u_h + u_lift` at model level. Kept as the
  *documented pattern* it is (it composes with (b) and needs no
  mechanism), but rejected as the only path: it cannot express
  Robin's α, and it pushes boilerplate onto every user.

### Robin derivatives: flux-form, not ghost chains

The derivative of a Robin field has no local mirror structure, so
its codomain is BC-free and — under R1 — exterior-needing
follow-ups are correctly blocked. This is mathematically honest,
and the supported route for Robin diffusion is **flux form**: the
boundary *flux* is derivable from the Robin data
(`u' = (g − u)/α` at the wall), so the flux-difference row closes
the boundary exactly — the `FluxDifference`-INNER precedent
generalized to data-parameterized boundary closures. Codomain
structure mapping for the mirror kinds (new rows, existing math):
diff maps Dirichlet→Neumann-structured and Neumann→Dirichlet-
structured per side; interp preserves the kind.

### Out of scope (designed-for)

Robin coefficient-space pairs (no classical fast transform; later
via Chebyshev tau/Galerkin); mixed-kind transform pairs (quarter-
wave DST/DCT variants); Robin/mixed on 2-D unstructured boundaries
(blocked on the spaces.md boundary-mesh question).

## 4. Validity claims per closure kind (task-1.8 interaction)

| closure on the applied axis | claim | why |
|---|---|---|
| periodic wrap | consume (shipped) | copies commute bitwise |
| homogeneous Dirichlet/Neumann mirror | consume — **stage 2f refinement** | mirror symmetry commutes with the stencil families; extending the claim gate from `mesh.periodic` to "fill commutes" is the recorded decision-record refinement |
| Robin / inhomogeneous (data-bearing) | reset | affine-in-data fills don't commute across space changes |
| one-sided rows | reset | no fill to commute with |
| BC-free | n/a | exterior-needing rows don't exist (R1) |

## 5. Work breakdown

Grid-local stages (no model-agent coordination needed): 2a–2d.
Stage 2e touches the module/assembly seam — schedule against the
model branch.

- **2a. Per-side grounding.** `BC.ROBIN` member; per-side shape
  rule audit (mixed tuples through `space_key`, member-Dirichlet
  drops per side); fill dispatch verification for mixed kinds;
  tests for every (node set × kind-pair) geometry. Files: `bc.py`,
  `spaces/nodal.py`, `meshes/structured_1d.py`,
  `decomposition/tensor.py`.
- **2b. BC-structured operator rows.** The coverage matrix for FD /
  interp / flux_diff / reconstruct over BC-structured domains with
  the codomain-kind mapping (D→N, N→D, interp preserves, per
  side); registry seeding in `grid.py`. This retires the
  `phase1_findings` BC-row API gap.
- **2c. The R1 flip.** Remove the BC-free bounded fill
  (`tensor.py`); prune/redirect exterior-needing BC-free rows into
  `DispatchError`s with BC-structured hints; migrate the bounded
  validation tests to declared BCs (they become *better* tests —
  yesterday's `d²` case asserts an exact −2 only by accident of
  the extrapolation's single-consumption exactness).
- **2d. One-sided rows** (R2): `boundary="one_sided"` variants for
  FD/interp with `layout="local"` requirements and reset claims;
  registry override examples + docs.
- **2e. Dynamic boundary data** (R3b): materialize the
  `("ghost_fill", space)` kind — `GhostFill` operators with
  dynamic (α, g) leaves, module-owned; `grid.sync(boundary_data=)`
  resolves through the same row; homogeneous Robin fill; flux-form
  Robin closure row. **Gated on model 2.2/2.3** (module parameter
  ownership, grid-in-state); coordinate the registry-seeding seam
  with the model branch.
- **2f. Mirror-claim extension** (perf, optional): claim gate from
  `mesh.periodic` to fill-commutes for homogeneous D/N; side-aware
  validity if profiling of bounded chains warrants it. Exchange-
  count gates extended to bounded chains.

Ordering: 2a → 2b → 2c (the flip must not land before the
BC-structured path is usable — R1 inverts which path is supported);
2d parallel to 2c; 2e after model 2.2/2.3; 2f last or deferred.

## 6. Gates

- Full suite + forced-4 at every stage; the R1 flip (2c) must keep
  the multi-device bitwise gate and the migrated validation cases
  green.
- New validation cases: mixed D/N diffusion (vertical-ocean
  geometry) against the analytic solution; Robin diffusion
  (flux-form) against analytic; inhomogeneous Dirichlet solved both
  ways (lifting vs ghost_fill row) agreeing to rounding; a
  one-sided-row case pinning its documented order.
- jit stability: sweeping α over values compiles **once** (compile-
  counter test) and `jax.grad` through α runs — the two recorded
  Robin constraints, as tests.

## 7. Decisions for owner sign-off

- **R1** — adopt the legality rule; remove the BC-free bounded
  extrapolation fill (bc_free_boundaries.md).
- **R2** — one-sided closures as per-operator opt-in rows
  (`boundary="one_sided"`), multi-device via `layout="local"`
  demands.
- **R3** — kinds-only per-side structure in the space key
  (`BC.ROBIN` added); all boundary values dynamic, delivered
  through `("ghost_fill", space)` rows with dynamic leaves
  (module-owned); lifting stays a documented pattern; Robin
  derivatives are supported flux-form.
- **R4** — sequencing: 2a–2d grid-local now-ish, 2e after model
  2.2/2.3, 2f optional; the Robin "do not resolve yet" directive
  on spaces.md question 1 is superseded by R3 only when signed.
