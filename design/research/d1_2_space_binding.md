---
status: frozen
date: 2026-07-07
---

# D1.2 — How module field declarations get their function spaces

Research report (see [`README.md`](README.md) for status).

## 0. Framing: what the decision actually has to reconcile

Three inherited facts collide:

1. **Spaces are grid-bound.** A `TensorProductSpace` is built from
   mesh factories (`mx.right * my.center * mz.center`); factors are
   interned per mesh, and meshes exist only once a grid does
   (`grid.factors`).
2. **Modules are grid-free until assembly.** They are constructed
   before the grid is known (`ConstantStratification(n2=1e-4)`) and
   must be reusable across grids — the same module on 2D (x,z) and
   3D (x,y,z), on uniform-FV and Chebyshev verticals.
3. **Assembly needs the spaces before fields exist.** The D4
   pipeline collects `state_spaces` for `grid.negotiate(...)`
   *before* `grid.create_field` runs.

So the question is precisely: *where does the grid enter the
declaration, and in what form does the module express staggering
before it does?*

## 1. The options

### Option A — declaration hook receives the grid

```python
class NonhydroCore(fr.Module):
    def field_declarations(self, grid) -> tuple[FieldDeclaration, ...]:
        mx, my, mz = grid.factors
        return (FieldDeclaration("u", space=mx.right * my.center * mz.center), ...)
```

Called once at assembly. Maximally expressive: anything the mesh
factories can spell, a module can declare. The "moral twin of
`op.eigenvalues(grid, space)`" argument.

### Option B — declarations are pure symbolic data (staggering descriptors)

```python
FieldDeclaration("u", stagger=("right", "center", "center"))   # positional, Oceananigans-style
FieldDeclaration("u", stagger={"x": "right"})                  # name-keyed, xgcm-style
```

The model resolves tags against the grid. Two sub-variants matter
enormously: **positional tuples** (the Oceananigans ceiling) vs
**name-keyed dicts with a default** (xgcm-style). And the tag
vocabulary matters: **concrete node-set names** (`"right"`) vs
**semantic tags** (`"staggered"`).

### Option C — first-class space-pattern objects with a resolution protocol

```python
FieldDeclaration("u", space=fr.Staggered("x"))       # staggered along x, collocated elsewhere
FieldDeclaration("b", space=fr.Collocated())
```

A `SpacePattern` is a static, frozen, hashable, reprable object with
one method the *model* calls at assembly:
`pattern.resolve(grid) -> TensorProductSpace`. Escape hatch: a
pattern wrapping a full callable, `fr.SpaceRule(lambda grid: ...)`,
which *is* Option A per-field.

These are not really three disjoint options: **C is B with a
resolution protocol and an A-shaped escape hatch.** The genuine axes
of decision are: (i) does the *hook* see the grid, or is the
declaration pure data; (ii) is the staggering vocabulary positional
or name-keyed; (iii) is it concrete node sets or semantic tags
resolved per mesh.

## 2. Trade-off analysis against the stress cases

| Stress case | A (hook gets grid) | B (positional / concrete tags) | B/C (name-keyed semantic pattern) |
|---|---|---|---|
| **Dimension generality** (same module, 2D & 3D) | Hand-rolled per module: loops over `grid.factors`, `if "y" in grid.names` — boilerplate repeated in every module, easy to get wrong | Positional 3-tuple **breaks outright** on (x,z); Oceananigans only survives because its world is always 3-slot (`Flat` topology, `Nothing` locations) — a move FRIDOM's variable-ndim, named-coordinate grids cannot make | **Free.** Names absent from the grid are simply unmatched: `Staggered("y")` on an (x,z) grid resolves to all-collocated — exactly the right space for `v` in a 2D rotating setup |
| **Mixed discretizations** (uniform FV horizontal × Chebyshev-Galerkin vertical) | Module must branch on mesh type (`isinstance(mz, ChebyshevMesh)`) — physics module accreting discretization knowledge; the exact scope creep D1.2 fears | Concrete tags (`"center"`) **cannot express it at all** — there is no `Center` on a Galerkin vertical; this is the ceiling the grid notes cite | Handled **below the module**: the semantic tag (`COLLOCATED`) is resolved per mesh through the grid's dispatch registry ("all default choices are dispatch entries", §2.2/§3.4). The *grid builder*, not the module, decides that collocated-on-`mz` means `mz.galerkin(bc=...)` |
| **Coefficient-space fields** (eigenmode-adjacent aux fields) | Trivially expressible | Inexpressible | Not in the tag vocabulary — deliberately. Covered by the `SpaceRule` escape hatch. (The eigenmode objects themselves are model-side consumers of `(grid, params)` per §2.5 and never pass through declarations.) |
| **Inspectable before fields exist** (negotiate needs `state_spaces` first) | OK — hook runs at assembly step 1; but inspection *requires a grid* | OK, pure data | OK — pure data pre-grid, resolved to bare interned spaces at assembly step 1, before `negotiate` mints layouts |
| **Scope creep** | High temptation: the hook receives the whole grid — decomposition queries, coordinate materialization, *parameter-dependent space choices* (catastrophic: spaces must be static and fixed pre-negotiate) | None | None for patterns; confined to the escape hatch, auditable per declaration |
| **Testability without a grid** | Cannot even *list* a module's fields without constructing a grid | Full | Full: `assert decl.space == fr.Staggered("x")` is plain equality on frozen data; resolution is tested once per mesh family in framework tests, not per module |
| **Serialization / repr / debugging** | A bound method — opaque | Good | Good: `Staggered('x', bc={'x': Dirichlet})` round-trips, hashes, prints; assembly can log a `name -> pattern -> resolved space` table |

Two further points against pure Option A:

- **The `op.eigenvalues(grid, space)` analogy is imperfect.** An
  operator receiving the grid computes *values*; a declaration hook
  receiving the grid chooses *static structure* (jit/dispatch keys,
  treedef, negotiation input). Letting arbitrary module code choose
  static structure with the full grid in hand invites exactly the
  "static structure holds no values / no late choices" tension the
  grid notes police everywhere else.
- **A is not actually more expressive where it matters.** The hard
  case (Galerkin vertical) is not solved by A — it is merely
  *relocated into every module* as an `isinstance` ladder. The right
  owner of "what space does a collocated variable get on this mesh"
  is the mesh/grid (a dispatch default), not each physics module.

Against pure Option B as the scaffold rejected it: that was the
*positional, concrete-node-set* variant — correctly rejected; the
rejection does not transfer to the name-keyed semantic variant.

## 3. Recommendation

**Hybrid, C-shaped: declarations are pure data; the space slot holds
a `SpacePattern` — name-keyed semantic tags with a default tag and
per-coordinate BC structure — resolved by the model against the grid
at assembly step 1 via per-mesh dispatch entries; a callable-wrapping
pattern (`SpaceRule`) is the escape hatch and recovers Option A's
full power per field.**

```python
class Dof(Enum):
    COLLOCATED = auto()   # the mesh's default "cell" representation
    STAGGERED  = auto()   # the mesh's default dual/face representation
    CONSTANT   = auto()   # ConstantSpace factor (old topo=False)

@dataclass(frozen=True)
class SpacePattern:
    default: Dof = Dof.COLLOCATED
    tags: tuple[tuple[str, Dof], ...] = ()        # keyed by coordinate name
    bc:   tuple[tuple[str, fr.BC], ...] = ()      # keyed by coordinate name

    def resolve(self, grid) -> TensorProductSpace:
        factors = []
        for mesh in grid.factors:
            tag = self._tag_for(mesh.names)       # default if no name matches
            bc  = self._bc_for(mesh.names)        # fr.BC.NONE if absent
            factors.append(grid.dispatch[("declared_space", mesh)](tag, bc))
        return prod(factors)                      # flat, interned

# sugar (what modules actually type)
fr.Collocated(bc=...)          == SpacePattern()
fr.Staggered("x", bc=...)      == SpacePattern(tags={"x": Dof.STAGGERED})
fr.Profile("z")                == SpacePattern(default=Dof.CONSTANT, tags={"z": Dof.COLLOCATED})
fr.SpaceRule(fn)               #  escape hatch: fn(grid) -> TensorProductSpace
```

- **The hook is grid-free.** `Module.field_declarations` is a plain
  property/attribute returning `tuple[FieldDeclaration, ...]` — no
  arguments. The *model* resolves at assembly step 1, yielding the
  bare `state_spaces` tuple handed to `grid.negotiate`.
- **Resolution is a dispatch entry, not a space property.**
  Design-consistent with the rejection of `space.shift()`: the
  staggering partner is ambiguous *as a space-level relational
  property*, but "what space does a STAGGERED declaration get on
  this mesh" is a per-mesh *default choice*, and §2.2/§3.4 already
  rule that default choices are dispatch entries. Seeds: uniform
  `IntervalMesh` nodal → `center`/`right` (`outer` with the declared
  BC on bounded axes); FV-flavored grid → `cell_avg`/`face_avg`;
  `ChebyshevMesh` → Lobatto collocation by default, overridable at
  grid construction to `galerkin(bc=...)`. The grid builder writes
  that override **once**; every module's `COLLOCATED` then lands on
  the Galerkin space without any module knowing.
- **BC structure rides in the declaration** because BCs are baked
  into the space (shape + interning key) and the declaring module is
  the natural owner of its field's BC structure — replacing
  `FieldMetadata.bc_types` as static per-coordinate data keyed by
  name.
- **The escape hatch is a pattern, not a different mechanism.**
  `SpaceRule(lambda grid: ...)` satisfies the same `resolve(grid)`
  protocol. Contract: pure, returns bare (pre-layout) spaces built
  from `grid.factors`' factories, runs before `negotiate`, may not
  consult the decomposition.

### Why this beats the hook (Option A)

The declaration is the one place where module structure meets grid
structure, and it is consumed by *machinery* (collision checks,
negotiate, treedef construction, debugging reports, eventually
run-configuration serialization). Machinery wants data, not code. A
keeps the 95% case (staggered/collocated/profile) as opaque
per-module code and duplicates dimension-generality and mesh-family
logic into every physics module; the pattern form makes the 95% case
declarative, inspectable, and testable, while `SpaceRule` preserves
100% of A's expressiveness for the 5%.

### Precedents, mapped

- **Firedrake/UFL** — strongest precedent *for*: `FiniteElement`
  descriptors are mesh-free symbolic data; `FunctionSpace(mesh,
  element)` binds at assembly; `FunctionSpace(mesh, "CG", 1)`
  resolves a *semantic family string* against the mesh's cell type —
  precisely the semantic-tag-resolved-per-mesh move. Reusable solver
  components traffic in element descriptors, never mesh-bound
  spaces. ([functionspace source](https://www.firedrakeproject.org/_modules/firedrake/functionspace.html),
  [arXiv:1501.01809](https://arxiv.org/pdf/1501.01809))
- **Oceananigans.jl** — demonstrates the appeal and the ceiling of
  symbolic-but-positional-and-concrete: `Field{Face, Center,
  Center}` works only because the framework forces an always-3D
  world (2D = `Flat` topology, one point; reduced fields get
  `Nothing` locations). FRIDOM's variable-dimension, named-coordinate,
  non-interval-factor grids cannot make that move.
  ([grids/topology](https://clima.github.io/OceananigansDocumentation/stable/model_setup/legacy_grids/))
- **Dedalus v3** — fields from `dist.Field(bases=...)`: the
  everything-grid-bound world; fine for problem *scripts*, yields no
  reusable module library. The shape Option A drifts toward.
- **xgcm** — the name-keyed nuance: staggered positions are metadata
  *per named axis* (`{"X": "left"}`), not positional tuples.

## 4. Risks and open questions

1. **Vocabulary ceiling of the semantic tags.**
   `COLLOCATED/STAGGERED/CONSTANT` covers A/B/C-grid nodal, FV, and
   Galerkin cases (B-grid: `tags={"x": STAG, "y": STAG}`), but a 2D
   unstructured factor breaks the *name-keyed* premise: a triangular
   C-grid velocity lives on the edge-normal space of the whole
   factor, not "staggered along lon". Resolution keying is per *mesh
   factor* (good — `resolve` iterates `grid.factors`), but the tag
   for a multi-name factor needs a per-factor spelling, and the tag
   set may need mesh-family members (`EDGE_NORMAL`). Designed-for;
   the escape hatch covers it meanwhile. Record as an open thread.
2. **Silent non-match of coordinate names.** `Staggered("y")` on an
   (x,z) grid resolving to all-collocated is the *feature*, but a
   typo (`Staggered("X")`) resolves the same way. Mitigation: the
   assembly step logs/returns the `name → pattern → resolved space`
   table; patterns could grow an optional `require=("x",)` field.
   Decide with D4's error-reporting story.
3. **Who may override the resolver.** Resolution entries
   (`("declared_space", mesh)`) should be **grid-level only** —
   seeded per mesh type, overridable in the grid's `defaults=`, but
   *not* mergeable from `Module.dispatch` overrides: a module
   silently changing how every other module's `COLLOCATED` resolves
   is cross-module action at a distance. Needs a one-line rule in
   the D4 merge-call-site decision.
4. **BC agreement across modules.** The declaring module fixes BC
   structure (it enters the interning key); a *referencing* module
   inherits it. A module that *requires* `b` with a specific BC has
   no way to assert it — probably fine (it can check the resolved
   space at bind); flag for D2.
5. **Escape-hatch discipline.** `SpaceRule` callables must be pure,
   static-only, pre-negotiate, bare-space-returning. Worth an
   explicit contract sentence; possibly a debug-mode double-resolve
   identity check (cheap — spaces are interned).
6. **Naming.** `Dof.COLLOCATED/STAGGERED` deliberately avoids the
   removed "position" vocabulary. `CONSTANT` replaces `topo=False`
   and matters for auxiliary parameter fields (`N2(z)` as
   `Profile("z")` — constant along whatever horizontal coordinates
   the grid happens to have: another dimension-generality win).

## 5. Illustrative sketch

```python
class NonhydroCore(fr.Module):
    field_declarations = (
        fr.FieldDeclaration("u", space=fr.Staggered("x", bc={"x": fr.BC.DIRICHLET}),
                            roles=..., long_name="u - velocity", units="m/s"),
        fr.FieldDeclaration("v", space=fr.Staggered("y", bc={"y": fr.BC.DIRICHLET}),
                            roles=..., long_name="v - velocity", units="m/s"),
        fr.FieldDeclaration("w", space=fr.Staggered("z", bc={"z": fr.BC.DIRICHLET}),
                            roles=..., long_name="w - velocity", units="m/s"),
    )

class ConstantStratification(fr.Module):
    def __init__(self, n2: float = 1e-4):
        self.n2 = n2
    field_declarations = (
        fr.FieldDeclaration("b", space=fr.Collocated(bc={"z": fr.BC.DIRICHLET}),
                            roles=..., long_name="Buoyancy", units="m/s²"),
    )
```

Absent: dimension count, mesh types, the grid. Class-level constants
— printable, hashable, testable, collectable before any field
exists.

**Resolution on a 3D uniform grid** (names `("x","y","z")`, default
resolvers — nodal: collocated→`center`; staggered→`right` periodic /
`outer`+BC bounded):

| Field | Resolved space |
|---|---|
| `u` | `Right(x) ⊗ Center(y) ⊗ Center(z)` |
| `v` | `Center(x) ⊗ Right(y) ⊗ Center(z)` |
| `w` | `Center(x) ⊗ Center(y) ⊗ Outer(z, bc=Dirichlet)` |
| `b` | `Center(x) ⊗ Center(y) ⊗ Center(z, bc=Dirichlet)` |

**Same modules, unchanged, on a 2D (x,z) grid:**

| Field | Resolved space |
|---|---|
| `u` | `Right(x) ⊗ Center(z)` |
| `v` | `Center(x) ⊗ Center(z)` — `"y"` matches nothing; exactly right for a rotating 2D slice |
| `w` | `Center(x) ⊗ Outer(z, bc=Dirichlet)` |
| `b` | `Center(x) ⊗ Center(z, bc=Dirichlet)` |

**Same modules, unchanged, on a mixed grid** (uniform x,y ×
`ChebyshevMesh` z, grid builder overriding the z-resolver to
Galerkin): `u`,`v`,`w`,`b` land on the appropriate Shen/Galerkin
z-factors per the resolver's BC entries. The physics modules were
edited zero times across the three grids; the one hard choice
(Galerkin vertical) was made once, by the grid builder, at the grid
boundary.
