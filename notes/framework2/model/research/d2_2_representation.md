# D2.2 — Parameter representation and time dependence

Research report (see [`README.md`](README.md) for status).

## 1. The placement rule

The old dual representation (`f0` scalar + `f_coriolis` field,
setters syncing) conflates three things: the module's **inputs**
(what the user specifies), the **traced read surface** (what
tendencies consume pointwise), and the **host-side read surface**
(what eigenmodes/ICs consume as closed-form scalars). The rule
separates them; each parameter has exactly one writable source of
truth.

**R1 — field iff consumed pointwise.** A parameter is an AUXILIARY
field iff consumers combine it *pointwise with state fields* in the
field algebra (products, staggering interpolation, gradients). A
parameter consumed only as an algebraic coefficient — scaling a
term, entering a solver symbol or eigenvalue formula — is a plain
dynamic leaf on its owner. Test: `f_coriolis` is interpolated to the
`u` position and multiplied into `v` → field; `dsqr` divides whole
terms and enters the pressure symbol and every eigenvector formula →
scalar leaf (it is not a function of space in any configuration).

**R2 — always-field within a family (no consumer branching).**
Where a physics family has constant and varying members (coriolis,
stratification, sw `csqr`), **every member declares the field**, the
constant member on all-`ConstantSpace` factors: `FPlaneCoriolis`
declares `f_coriolis` on `fr.Profile()` — ConstantSpace on every
axis, **one DOF**. The grid notes give this for free (ConstantSpace
broadcast is exact and a sanctioned strict-algebra exception;
operators along a ConstantSpace axis are identity), so the identical
consumer line handles f-plane and beta plane. The alternative —
consumers branching scalar-vs-field — is the `isinstance` ladder
disease D1.2 killed for spaces. (Oceananigans gets uniformity via
Julia dispatch per coriolis type; FRIDOM's consumers are generic
single-path field expressions, so the field *is* the dispatch
mechanism, and ConstantSpace makes it cost-free.)

**R3 — inputs are the source of truth; the field is owner-derived
state.** Constructor inputs (`f0`, `beta`, an `n2` scalar/Ramp) are
the single writable surface (dynamic leaves). The AUXILIARY field is
*derived*: materialized by the owner at allocation, re-materialized
host-side at the run boundary when a parameter changes (the
one-directional, owner-confined successor of `_update_coriolis`),
recomputed per step in `self_update` only when time-dependent. No
setter on the field, ever. Parameter changes go through a functional
update API (`model.update_parameters(...)`) which triggers
re-materialization — same treedef, new leaves, **no recompile**.
Direct leaf-poking is out of contract.

**R4 — scalar provides for host-side consumers.** Modules
additionally *provide* named scalars for host-side analytic
consumers (eigenmodes need `f0`, `n2`, `dsqr` closed-form). A second
**read** surface, not a second source of truth. A module that cannot
supply the scalar doesn't provide it — `BetaPlaneCoriolis` provides
`f0`/`beta`, and a consumer requiring "constant f" fails at
resolution with a hint, which is physics-correct.

### Decision table

| Old parameter | New placement | Owner |
|---|---|---|
| `f0`, `beta` | dynamic leaves; provided scalars | coriolis module |
| `f_coriolis` | AUXILIARY field, **always declared**: `fr.Profile()` (1 DOF) f-plane, `fr.Profile("y")` beta plane; derived at allocation | coriolis module |
| `stratification_n2` (+field) | AUXILIARY field `"n2"`: `fr.Profile()` constant, `fr.Profile("z")` profile variant; provided scalar only from the constant module | stratification module |
| `dsqr` | dynamic leaf + provided scalar — never a field | nh core |
| `csqr` (+`csqr_field`) | AUXILIARY field (`fr.Profile()` constant / horizontal space varying); provided scalar in the constant configuration | sw core |
| `rossby_number` | dynamic leaf + provided scalar (§4) | core module |
| diffusivities | dynamic leaves on the closure; a varying-closure variant declares a field per R1; the biharmonic auto-coefficient becomes an assembly-time derivation from required `rossby`/`dsqr` | closure |

Structural corollary: the declared space is static, so f-plane vs
beta-plane is **two module types** (a `Coriolis(f0, beta)` factory
can paper over ergonomics), matching Oceananigans' distinct
`FPlane`/`BetaPlane` types.

## 2. Traced vs static discipline

Keep explicit `jaxify(dynamic=(...))` declaration — the equinox
value-based alternative (`eqx.is_array` filtering) has the worse
default failure mode: a Python-float `f0` silently lands static and
recompiles per change. Hardened rules:

1. **Dynamic leaves are arrays**: `jaxify` coerces declared-dynamic
   attributes through `jnp.asarray` on assignment; a structural
   value declared dynamic fails coercion at construction — loud.
2. **Static attributes are hashable non-arrays**: asserted at
   registration (arrays in the treedef = hash failure or per-value
   recompile keys). Structural parameters (advection order, stencil
   choice, a Ramp's shape function) are static by default; changing
   one recompiles — correct and visible.
3. **Provided parameters must be dynamic**: resolution errors on a
   static provided scalar — closes the silent-recompile-per-sweep
   hole for exactly the parameters that get swept.
4. **Dev-mode recompile lint** (02_rules): `jax_log_compiles`; a
   sweep logging more than one compile is a static-leak bug.

## 3. `fr.Ramp` — the time-dependent value

**A frozen callable pytree value, not a module.** It occupies any
scalar slot and is evaluated by the owning/consuming module inside
the trace at the traced clock time.

```python
@partial(fr.utils.jaxify, dynamic=("start", "period", "v0", "v1"))
class Ramp(TimeDependent):
    shape: Callable[[Array], Array]   # STATIC: named curve or user fn
    def __call__(self, t):            # branch-free; valid for all t
        theta = jnp.clip((t - self.start) / self.period, 0.0, 1.0)
        return self.v0 + (self.v1 - self.v0) * self.shape(theta)

def resolve_at(value, t):
    return value(t) if isinstance(value, TimeDependent) else value
```

- **Split verdict**: the *shape* is static (changing the curve is
  different math — one recompile, correct); *endpoints/timing* are
  dynamic leaves (sweeping ramp targets never recompiles). Mirrors
  diffrax: term structure static, coefficient data traced.
- The old `Ramper`'s Python `if time < start` is untraceable;
  `clip` + `shape(0)=0, shape(1)=1` subsumes both branches; the
  exponential curve's guards port to `jnp.where`.
- Scalar↔Ramp swap in a slot changes the treedef → one recompile +
  restart-fingerprint change; documented.
- **Composition**: `Ramp * scalar` etc. return `TimeDependent`s
  (needed by OptimalBalance). Ramp deliberately does **not**
  implement field/array arithmetic — a consumer that forgets
  `resolve_at` fails loudly in the assembly dry run.
- **Universal idiom**: traced code reads physics-scalar slots
  through `resolve_at(self.x, clock.time)` — identity on plain
  scalars, so every scalar parameter is Ramp-able with zero
  consumer changes. Requires the clock in the tendency signature —
  fed forward to D3.

**Boundary with `self_update`:**

> A Ramp describes a time curve; it never writes anything.
> `self_update` is the only writer of owner state, and it may
> consume Ramps. Parameters consumed as **scalars** get time
> dependence via `resolve_at` at the point of use (default).
> Parameters consumed as **fields** (R1/R2) get it via the owner's
> `self_update` recomputing the AUXILIARY field, because the
> consumers' read surface is the field.

Ramped constant N²(t): the aux field is 1 DOF (`fr.Profile()`);
`self_update` writes `resolve_at(self.n2, clock.time)` into that
one number per step. Assembly schedules a module's `self_update`
**only if** any input is `TimeDependent` (structural, decided at
assembly) — a static model pays nothing. Slot position owed to D3
(thread 3e).

## 4. The `rossby_number` successor

Full consumer inventory (grep-verified): advection scaling (3
packages), OptimalBalance's mutation of `advection.scaling`,
nh `pot_vort`/`linear_pot_vort`/`local_rossby_number`, sw
`ekin`/`pot_vort`/`local_rossby_number`, biharmonic-closure
coefficients, IC recipes (jet with `rossby_number=0.1`),
hydrostatic.

**Recommendation: the dynamical-core module owns and provides
`rossby_number`** (dynamic leaf, default 1 = unscaled), alongside
`dsqr`/`csqr` which D1.3 already assigned there. Ro is a property of
the *nondimensionalized equation set* (the docstrings write
`∂t u + Ro u·∇u = ...` with (Ro, δ²) as the scaled pair), not of the
advection scheme, and it reaches diagnostics, closures, eigenmodes,
and ICs. Rejected: a dedicated `NondimensionalScaling` module (a
fieldless one-number module duplicated per package — the core slot
already exists); advection-local-only (misses five of seven
consumers); preset-kwargs-only (presets forward `rossby_number=` to
the core constructor, but storage is the core).

Mechanism: generic framework modules stay Ro-ignorant via
**reference-valued slots** — `fr.Param("scaling.rossby",
default=1.0)` as constructor default, the parameter-level analogue
of `FieldReference`: resolved at assembly, hinted error if
required-without-default, and the slot equally accepts an explicit
number or a Ramp (escape hatch A). A globally ramped nonlinearity
(the old Ramper's headline use) is
`DynamicalCore(rossby_number=fr.Ramp(...))` — every consumer
reading through `resolve_at`/params gets the ramp for free.

## 5. Units/validation

**Carry the string, never compute with it.** AUXILIARY-field
parameters get units/long_name via `FieldDeclaration` →
`FieldMetadata`; provided scalars get the same optional metadata.
Payoff: the run-header parameter table and provenance metadata
reproduce for free (ClimaParams precedent). Full unit algebra:
rejected (dead weight in a nondimensional-flagship framework).

## 6. Sketch

```python
@partial(fr.utils.jaxify, dynamic=("f0",))
class FPlaneCoriolis(fr.Module):
    def __init__(self, f0: float): self.f0 = f0
    def field_declarations(self):
        return (fr.FieldDeclaration("f_coriolis",
            space=fr.Profile(),                    # ConstantSpace everywhere -> 1 DOF
            lifecycle=Lifecycle.AUXILIARY,
            default=lambda: self.f0, units="1/s"),)

@partial(fr.utils.jaxify, dynamic=("f0", "beta"))
class BetaPlaneCoriolis(fr.Module):
    def __init__(self, f0, beta): self.f0, self.beta = f0, beta
    def field_declarations(self):
        return (fr.FieldDeclaration("f_coriolis",
            space=fr.Profile("y"),
            lifecycle=Lifecycle.AUXILIARY,
            default=lambda y: self.f0 + self.beta * y, units="1/s"),)

@partial(fr.utils.jaxify, dynamic=("n2",))
class ConstantStratification(fr.Module):
    def __init__(self, n2: float | fr.Ramp): self.n2 = n2
    def field_declarations(self):
        return (fr.FieldDeclaration("n2", space=fr.Profile(),
            lifecycle=Lifecycle.AUXILIARY,
            default=lambda: fr.resolve_at(self.n2, t=0.0), units="1/s^2"),)
    def self_update(self, clock):   # scheduled ONLY if n2 is TimeDependent
        return {"n2": fr.resolve_at(self.n2, clock.time)}
    def tendency(self, state, clock):   # reads the field; identical for profile variants
        n2, w, b = state["n2"], state["w"], state["b"]
        return {"b": -n2 * w.to(b), "w": b.to(w)}
```

## 7. Risks and open questions

1. **Leaf-staleness residue (R3)**: direct leaf mutation between
   runs leaves derived fields stale; contract = changes go through
   the functional update API, which re-materializes. Needs a D4
   lifecycle hook + 02_rules entry — the one place the old bug
   survives in weakened, documented form.
2. **`resolve_at` discipline** is conventional — mitigated by Ramp
   refusing field arithmetic (loud dry-run failure); possible lint:
   dry-run each Ramp-able slot with a sentinel.
3. **scalar↔Ramp treedef change**: restart fingerprint must treat
   the parameter *spec* (shape) as structure, endpoints as leaves —
   owed to the D1 restart section.
4. **Clock in tendency signature** + **self_update slot**: fed to D3.
5. **Dual read surface rule**: inside the trace the field is
   authoritative; provided scalars are assembly/host-side. A traced
   consumer requiring `"f0"` instead of reading
   `state["f_coriolis"]` should fail resolution or lint.
6. **One-vs-two coriolis modules**: two types (structural spaces),
   optional factory for ergonomics.
7. **Eigenmode constancy checks** ride on provides: a profile
   module simply doesn't provide the scalar, so failure is at
   resolution with a hint.

Precedents: Oceananigans FPlane/BetaPlane as distinct types;
equinox static-field lore (never arrays static; value-filtering
silently statics Python floats); diffrax static-structure /
dynamic-coefficient split; ClimaParams name+units+description
registry.
