# D1.5 — Access surface, collisions, namespacing, functional updates

Research report (see [`README.md`](README.md) for status).

## What today's usage actually looks like (calibration)

- **User scripts** touch canonical fields as attributes: `mz.z.u`,
  `mz.z.b`, `mz.z.etot`, and mutate ICs through the *array*, not the
  property setter: `z.b.arr += (mesh[2] < 0.5)`
  (`rayleigh_taylor_instability.py`).
- **User tracers** already go through dict access: `mz.z["dye"]`
  (`tracers_and_eddies.py`) — nobody writes `z.dye` today. The
  precedent for "canonical names get attributes, extras get strings"
  is already established in FRIDOM's own user base.
- **Tendency modules** mutate the accumulator:
  `mz.dz.u += interp(z.v, z.u.position) * f` (`linear_tendency.py`).
- **Derived diagnostics** (`ekin`, `pot_vort`, `cfl`) are `State`
  properties reading `self.mset` parameters — D2 already leans
  toward evicting the parameter-dependent ones from `State`.

## 1. Access surface

Options: (a) dict access only; (b) hand-written properties on
per-package State subclasses; (c) `__getattr__` fallback; (d)
generated/typed accessors.

**The `__getattr__`-on-pytree footgun, honestly.** A *disciplined*
implementation is mechanically safe (guard via
`object.__getattribute__(self, "__dict__").get("_components")`,
raise a real `AttributeError`): this avoids the classic failures —
infinite recursion during unflattening (jaxify builds instances via
`object.__new__` and sets `_components` afterward),
copy/pickle/deepcopy probing, jax's `hasattr` feature probes. The
costs that remain are the ones that don't go away:

1. **Property-error masking** — the nasty one given FRIDOM's design.
   `State` subclasses keep computed properties (`ekin`,
   parameter-free diagnostics). If a property's *body* raises
   `AttributeError` (a typo inside `ekin`, API drift), Python
   swallows it and falls through to `__getattr__`, which reports
   "State has no component 'ekin'" — a wrong, deeply confusing
   message masking a real bug. Every pytree class combining
   properties with `__getattr__` hits this eventually.
2. **Static opacity** — no completion, no type for `z.dye`;
   pyright/mypy flag every dynamic attribute unless the class is
   typed `Any`-leaky (which stops catching real typos class-wide).
3. **Typos become plausible-looking code** — `z.psi` looks
   statically checked but isn't.
4. **Read/write asymmetry** — `z.dye` reading dynamically while
   `z.dye = ...` must raise means `__setattr__` needs a matching
   guard; more magic surface.

Verdict: survivable but buys three characters per access at the
price of the masking bug and typing opacity. Reject — concurs with
the already-normative VectorField note ("no dynamic `__getattr__`
fallback"), so no amendment needed.

**Generated/typed accessors (d).** To give IDEs anything, codegen
must happen before type-check time (stub generation on disk);
assembly-time `type()` synthesis is invisible to pyright and *worse*
than `__getattr__` (same opacity plus a dynamically-created class
per model instance, breaking pytree registration-per-class
assumptions and treedef caching). Reject.

**The per-model-State-subclass vs generic-Model tension (b).** Under
D1.3-B the component set is fully module-determined — so does a
`nonhydro.State` subclass even exist? Yes, and the tension dissolves
once the subclass's job is restated: it is **not a structural
entity** (declares no fields, no assembly logic, inherited
constructor) — it is a **vocabulary class**: a bundle of curated
accessors and parameter-free diagnostics for the names that are
canonical *in that model family*, whether or not the current module
list produced them:

```python
class State(fr.VectorField):                      # fridom.nonhydro.state
    @property
    def b(self) -> fr.ScalarField:
        """Buoyancy (present when a stratification module is assembled)."""
        return self._component("b", hint="add a stratification module, e.g. "
                                          "nh.modules.ConstantStratification")
```

`VectorField._component(name, hint=...)`: return the component or
raise `MissingComponentError("State has no component 'b'. " + hint +
" Components present: u, v, w, dye.")`. The property doesn't care
*who* declared `b`, only that the name is canonical vocabulary. Who
instantiates the subclass: the generic `Model` takes a
state-class hook at assembly, supplied by the preset / core module
(D1.3). One registered class, not synthesized — treedef stability
untouched; a bare generic `Model` gets plain `VectorField` and dict
access, losing nothing but sugar.

### Recommendation (access)

- **Primary, and the only form module code uses: `z["b"]`.** Modules
  are generic over State classes; this also makes the halo-trace dry
  run independent of which State subclass is live.
- **Allowed convenience: hand-written properties on per-package
  vocabulary State subclasses** (`z.u/z.v/z.w/z.b` nonhydro), each
  with a curated `hint=`. No setters. User tracers and
  module-private fields get no properties: `z["dye"]`.
- **Reject** `__getattr__` and generated accessors.
- **Error-message contract** for `MissingComponentError`: name
  what's missing, list what's present, and — from a curated property
  or a `requires` failure — say which module family provides it.
  Optionally `difflib.get_close_matches` typo suggestions in
  `__getitem__` (host-side only; the error path never traces).

## 2. Collisions and the declare-vs-reference distinction

Options: (a) any duplicate name errors; (b) identical duplicates
silently merge; (c) single-owner declarations plus a separate
**reference** mechanism; (d) references that auto-create with a
default declaration.

Precedents (verified): Oceananigans buoyancy —
`validate_buoyancy` checks `required_tracers ⊆ tracers` and
**errors** (reference-checked, not auto-added). Oceananigans
biogeochemistry — `validate_biogeochemistry` **auto-adds**
`required_biogeochemical_tracers` when the user specified none, but
auxiliary fields must exist or it errors. MOM6 — flat
`register_tracer` registry that **locks** after registration ("to
prevent the addition of more tracers") — the exact analogue of
"component set frozen before negotiation".

Analysis: two modules *declaring* `b` is almost always a wiring bug
(two stratification modules), and silent merge would have to
reconcile spaces, roles, metadata, and — worst — *tendency
ownership* (both modules contributing `db/dt` believing they own
it). But the shared-need case is real (a closure acting on `b`; a
BGC module reading temperature declared by an EOS module). The clean
factorization: **exactly one owner per name; everyone else
references**:

```python
class FieldReference(NamedTuple):
    name: str
    hint: str = ""          # "add a stratification module (ConstantStratification, ...)"
    # optionally: expected roles the referenced field must carry
```

Assembly rules: (1) two declarations of one name →
`FieldCollisionError` naming both modules; (2) unsatisfied reference
→ `MissingFieldError` using the reference's hint; (3) freeze
(MOM6-style lock) before halo tracing / negotiation.

No auto-creation: a reference that carries a space *is* a
declaration in disguise — it reintroduces merge-reconciliation
through the back door. When there is genuinely no natural owner, the
model *preset* adds a minimal declaring module (a
`Tracer("T", space=...)` one-liner — which is also how plain user
tracers enter, replacing `mset.custom_state_fields`). Cost: one
extra line in a preset; benefit: "who owns `T`" always has a unique
answer.

**Interaction with roles (D1.4).** Keep two query kinds distinct: a
**reference** is by-name, existence-mandatory, assembly-checked; a
**role query** is by-role and may match zero fields (advection with
no advected tracers is a no-op, not an error). A reference may
optionally assert roles on the referenced field — assembly-checked.
The reference mechanism is the *field-level* face of D2's requires
machinery; D2 should adopt the same pattern so module authors learn
one idiom.

## 3. Namespacing

**Flat, with prefix-by-convention.** All surveyed systems are flat:
MOM6's registry (generic-tracer packages use conventional prefixes),
MITgcm ptracers, Oceananigans tracer symbols (OceanBioME's bgc
tracers merge into the same flat tuple). Structural namespacing
(`"bgc.no3"`) would fight the rest of the decision: it breaks the
shared-field story (consumers *shouldn't* know the owner — that's
the point of references), uglifies IO (netCDF names are flat), and
kills the vocabulary-property sugar. The single-owner +
collision-error rule already provides the guarantee namespacing
would exist to provide, at assembly time with a good message. A BGC
package shipping twenty fields prefixes by convention (`"bgc_no3"`)
— a documentation guideline, not a mechanism.

## 4. The functional-update surface for tendency code

Old idiom: `mz.dz.u += ...` — in-place accumulation into a shared
mutable `dz`. Options: (a) raw `replace`
(`dz = dz.replace(b=dz["b"] + term)` — name three times per term);
(b) an accumulator method (`dz = dz.add(b=term_b)`); (c) **modules
return contributions** (a name→field mapping) and the composer sums.

**Take the D3 constraint seriously:** D3 tags terms
`EXPLICIT`/`IMPLICIT` and steppers consume the *partition*; IMEX
needs the implicit contributions separately; by-variable sweeps need
per-term, per-field contributions. **A shared threaded accumulator
erases exactly the structure D3 needs** — once module A has added
into `dz`, module B's contribution is inseparable. So (c) is not an
ergonomics choice; it is the shape D3 requires. It is also
friendliest for module authors (they never see other modules'
state), for testing (a term is a pure function
`(z, params) -> contributions`), and for the halo tracer (each term
independently traceable).

Concretely: a term returns a plain `dict[str, ScalarField]`
(jax-transparent; keys validated at assembly-time trace — unknown
key → error naming the module). The composer sums, with
`VectorField.add` as the primitive:

```python
def add(self, **contributions: ScalarField) -> Self:     # on VectorField
    """Functional accumulate: replace(**{k: self[k] + v}); unknown
    name -> MissingComponentError with the component list."""
```

`add` is worth having publicly anyway (multi-part modules, user
forcing hacks, the degenerate one-callable mode). `__setitem__` is
not implemented except as a raising stub. Example term:

```python
class ConstantStratification(fr.Module):
    n2: float = 2.5e-5
    field_references = (fr.FieldReference("w"),)   # couples to vertical velocity

    def tendency(self, z, params) -> dict[str, fr.ScalarField]:
        return {
            "w":  z["b"].to(z["w"]),               # +b in the w-equation
            "b": -self.n2 * z["w"].to(z["b"]),     # -N² w in the b-equation
        }
```

Composer (framework-side, once):

```python
dz = zero_like(z)
for term in explicit_terms:
    dz = dz.add(**term.tendency(z, params))
```

**Position on the fields.md open question:** ports go **fully
functional immediately** — the contribution-return shape means
module ports don't even need a mutation-lookalike; a "temporarily
working" shim has no constituency in module code. `replace` remains
the idiom for *overwriting* (projection stage, ICs).

## 5. Migration shim

The old mutation surface is three-headed: (i) `z.u = field` via
property setters (rare); (ii) `z.u.arr += ...` — the dominant IC
pattern, which **bypasses State entirely**, so no State-level shim
can make it "temporarily work"; (iii) `mz.dz.b += term` in modules,
which has no syntactic successor under contribution-return. A
"temporarily working" mutation shim would mutate `_components` in
place — silently wrong under jit (leaked tracers, stale carries) —
and *still* wouldn't catch (ii). So:

**Raising guidance shims, one deprecation cycle, never-working
mutation.**

- Vocabulary property **setters raise** `ImmutableStateError`: "use
  `z = z.replace(u=...)` (or `z = z.add(u=...)` to accumulate)."
- `VectorField.__setitem__` exists only as the same raising stub.
- `ScalarField.data` gets an explicit raising setter so
  `z.u.data = ...` gets guidance ("fields are immutable; use
  `f.with_data(...)` / `grid.create_field(space, init=...)`")
  instead of a bare `AttributeError`.
- Port all examples in the same change; the shims are for *users'*
  scripts, not the repo's.

## Risks and open questions

1. **Property-vocabulary drift**: hints name the *concept* plus one
   example module; keep them one line.
2. **`add` and space joins**: a term on the wrong staggering raises
   `SpaceMismatchError` inside the composer loop; the composer
   should catch and re-raise with the module/term name attached, or
   error quality regresses versus today's in-module `+=`.
3. **Contribution-dict typing**: stringly-typed at the term
   boundary; the assembly-time dry run (already required for halo
   tracing) must validate keys so typos fail at assembly, not step
   40 000 — makes un-jitted traceability load-bearing for error
   quality too.
4. **Zero-initialized `dz` cost**: jax usually fuses; the composer
   could tree-sum per name without a zero base — implementation
   freedom, not a design commitment.
5. **Deferred**: role assertions in `FieldReference` (recommend
   name-only for it-1); diagnostics writers select by name or role
   (D1.4/D2); error-type naming alignment with the grid notes'
   error registry.

Sources: Oceananigans `Biogeochemistry.jl` / `BuoyancyFormulations`,
Oceananigans tracers docs, MOM6 `MOM_tracer_registry` API docs,
OceanBioME.jl.
