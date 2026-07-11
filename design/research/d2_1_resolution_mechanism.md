---
status: frozen
date: 2026-07-07
---

# D2.1 — Cross-module parameter resolution: mechanism design

Research report (see [`README.md`](README.md) for status).

**Recommendation: B + C complementary, A as a bounded escape hatch —
with one decisive refinement: resolution binds *names to static
accessors* at assembly, and delivers *values fresh per
stage-evaluation* through a `params` argument computed inside the
trace.** Consumers never hold provider objects or frozen values.
This is the only reference semantics that simultaneously survives
pytree flattening, respects dynamic leaves, and composes with
`fr.Ramp`.

## 1. The mechanism

### 1.1 Provides

```python
ParameterDeclaration(
    name="coriolis.f0",       # canonical dotted name; collision unit
    attr="f0",                # the module attribute holding the live value (a dynamic leaf)
    units="1/s",
    doc="constant Coriolis frequency (f-plane)",
)
```

collected from `Module.parameter_declarations` at assembly (D4 step
2, alongside field collection). Deliberate choices:

- **`attr`, not `value`.** The declaration names *where the value
  lives*, it does not carry the value — the value is read from the
  live carry at use time, so changing a leaf between runs (no
  retrace) and holding a `Ramp` in the slot both work with zero
  extra machinery. A snapshot-at-assembly `value` slot is the
  frozen-copy bug by construction.
- **Scalars (and small static-shape arrays) only.** Spatially
  varying parameters are AUXILIARY fields, not table entries (§1.3).
- **One provider per name** — duplicate is a
  `ParameterCollisionError` naming both modules (the
  `FieldCollisionError` contract). No rename-on-conflict.

### 1.2 Requires and delivery

```python
class ParameterReference(NamedTuple):
    name: str
    hint: str = ""              # falls back to the canonical-name registry's hint
    default: Any = REQUIRED     # REQUIRED sentinel -> MissingParameterError
```

declared in `Module.parameter_references`, checked at assembly with
field references, frozen before halo tracing. The `default` slot is
the one divergence from `FieldReference` (which forbids
auto-creation because fields need space/role reconciliation): a
scalar default has no reconciliation problem, and it lets advection
require `scaling.rossby` with `default=1.0` and work in dimensional
models with no provider present.

**What the consumer receives: values, per evaluation, via an
argument.** Resolution produces a Model-owned **binding table**
`{name: (module_slot, attr)}` — pure static data. At run time the
composer evaluates, *inside the trace*:

```python
def eval_params(modules, t) -> Params:          # composed once at assembly
    return Params({name: _resolve(getattr(modules[slot], attr), t)
                   for name, (slot, attr) in binding_table.items()})

def _resolve(value, t):
    return value(t) if isinstance(value, fr.TimeDependent) else value
```

and passes the result to every traced module entry point
(`tendency(self, state, params)`, stages, self-update). `Params` is
a thin frozen mapping; unknown-name lookups are caught by the
assembly-time dry run like unvalidated contribution keys. Cost:
a handful of scalar ops per stage; XLA dead-code-eliminates unused
reads.

**Ramp composition and stage times.** Because `eval_params` is
`(modules, t) -> values`, the stepper calls it at each stage time —
an RK stage at `t + c·dt` sees the ramp evaluated at `t + c·dt`,
for free. Time-dependent *scalars* need no self-update hook; the
hook remains only for evolving AUXILIARY *fields*.

**Host-side consumers.** `model.parameters` evaluates the same
binding table on the host, returning a concrete mapping with
units/doc attached — the surface IC recipes and eigenmode objects
consume. Host-side lookup of an unprovided name raises the same
`MissingParameterError` with the registry hint.

**Assembly-time reads.** A module reading a parameter at `bind()`
to precompute coefficients may — but reading a `TimeDependent`
there raises unless spelled `at_time(0.0)`. Rule encoded in the
error: split the precomputation — grid factor at bind (static),
parameter factor inside the step (traced multiply, free). This
kills the `BiharmonicClosure._on_setup` stale-coefficient bug class
(it bakes `rossby_number` into `_hor_diff_coeff` at setup today).

### 1.3 The C half: AUXILIARY fields

Spatially-varying parameters are **not in the parameter table**:
they are AUXILIARY-lifecycle fields, declared via
`FieldDeclaration`, referenced via `FieldReference`, read as
`state["f_coriolis"]`. Decision rule:

> **Varies in space → declare/reference a field. Scalar (possibly
> time-dependent) → declare/reference a parameter.**

Both faces may be offered when analytically meaningful. The split
is *physically load-bearing*: an eigenmode object requiring the
scalar `coriolis.f0` correctly **fails** on a beta-plane setup —
the namespace split turns an analytic-validity assumption into an
assembly-checked requirement. No field-valued table entries (shape
polymorphism, duplicate read path, merge questions).

## 2. The jax aliasing analysis

**The problem:** pytrees are trees, not DAGs. A consumer storing a
provider module as a dynamic attribute while the provider sits in
the modules tuple puts it in the carry twice: flatten emits its
leaves twice, **unflatten reconstructs two independent objects**,
and the copies silently diverge (the provider's self-update writes
its tuple-slot copy; the consumer reads its stale embedded copy).
This is the documented motivation for `eqx.nn.Shared` in Equinox,
whose fix is precisely accessors-not-objects.

| Semantics | Verdict |
|---|---|
| Consumer stores provider object (dynamic attr) | Broken: duplicate leaves, silent divergence — wrong physics with no error |
| Consumer stores provider object (static attr) | Broken: treedef/hashability, frozen at trace time |
| Resolve to frozen value at assembly | Broken for leaves: defeats between-run updates, can't hold a Ramp; acceptable only as the explicit `at_time(0)` assembly read |
| Mirror values into a carry-resident params namespace | Works but redundant: every value twice + a sync step (Flax-linen-shaped) |
| **Static accessor table + per-stage evaluation into an argument** | **Chosen**: single source of truth, no duplicate leaves, no staleness, treedef-stable; Ramp evaluation is part of the read |

Oceananigans independently confirms the delivery half: tendency
kernels receive `coriolis`/`buoyancy`/`closure` as per-step
arguments threaded by the model.

**Guard rail:** assembly runs an identity-based duplicate-node scan
over the composed model pytree and errors on any jaxified Module
instance appearing twice.

## 3. Namespace rules

- **Dotted physics-concept prefixes, mandatory**: `coriolis.f0`,
  `stratification.n2`, `nonhydro.dsqr`, `scaling.rossby`. Dotted by
  *concept*, never by module class (`fplane.f0` would break
  provider substitutability).
- **Two visually disjoint namespaces**: field names are dot-free
  (state/netCDF); parameter names must contain a dot
  (assembly-lint). `state["f_coriolis"]` vs `params["coriolis.f0"]`
  self-document the mechanism in play.
- **Canonical-name registry**: `fr.params.CORIOLIS_F0 =
  ParamName("coriolis.f0", units="1/s", hint="provided by Coriolis
  modules, e.g. nh.FPlaneCoriolis(f0=...)")` — mirrors D1.4's Role
  markers; typo-proof interop; registry hints back host-side
  errors. User packages prefix by convention.

## 4. Option A coexistence; the `rossby_number` successor

**A coexists, bounded by where the object lives**: passing module
objects directly is fully supported for consumers **outside the
traced carry** (host-side diagnostics, IC helpers, analysis). For
in-carry modules, A degenerates to constructor *values* (an owned
copy, no linkage claimed) or is forbidden (object-storing — caught
by the duplicate-node lint).

Ownership rule of thumb: **constructor-arg what you own; provide
what others consume; require what you don't own; field-reference
what varies in space.**

`rossby_number` successor (as proposed here): a
`NondimensionalScaling` module providing `scaling.rossby`;
consumers require it with `default=1.0`; ramped spin-up =
`NondimensionalScaling(rossby=fr.Ramp(...))`; constructor override
remains (explicit-wins). *(Reconciliation: the resolved design
instead puts ownership on the dynamical core, per the d2_2 consumer
inventory — the mechanism here is unchanged.)* `dsqr`: owned by
`nh.DynamicalCore`, read directly by its own projection stage
(self-owned, no reference), required by closures. Shallowwater
`csqr` showcases field-or-scalar: the old `csqr`/`csqr_field` pair
dissolves into the two-table rule with no synchronizing setters.

## 5. Risks and open questions

1. **`params` threading through D3**: the stepper must call
   `eval_params(modules, stage_time)` per stage — fed forward; the
   exact signature packaging is D3's.
2. **Default-valued references** weaken the hinted-error idiom if
   overused. Rule: defaults only for physically-identity values
   (scaling → 1, forcing amplitude → 0); registry names can be
   marked `no_default=True` (a default on `stratification.n2`
   would silently un-stratify a run).
3. **Explicit-wins mechanism** for constructor-set values
   suppressing a declared reference: proposal — the reference is
   declared only when the constructor arg is left at a
   `USE_PROVIDED` sentinel. Small API-sketch item.
4. **Scalar-vs-field genericity pressure**: if a "field view of any
   provided scalar" helper is ever needed, provide it host-side
   (`model.parameters.as_field(name, space)`), never as table
   unification.
5. **Duplicate-node lint** targets jaxified Module instances only
   (interned static objects are legitimately shared).
6. **Small non-scalar parameters** (rotation axis, tidal tables):
   allowed as static-shape array leaves; the boundary is "does it
   live on the mesh?", not size.
7. **Units are documentation** (matching D1); a future units lint
   is designed-for (both sides carry `units`), not built.

## 6. Sketch

```python
@fr.jaxify(dynamic=("f0",))
class FPlaneCoriolis(fr.Module):
    def __init__(self, f0: float = 1e-4):
        self.f0 = f0                                  # dynamic leaf; may hold fr.Ramp
    parameter_declarations = (
        fr.ParameterDeclaration(fr.params.CORIOLIS_F0, attr="f0", units="1/s"),)
    field_references = (fr.FieldReference("u"), fr.FieldReference("v"))

@fr.jaxify
class PotentialVorticity(fr.Module):
    parameter_references = (
        fr.ParameterReference(fr.params.CORIOLIS_F0,
            hint="add a Coriolis module, e.g. nh.FPlaneCoriolis(f0=...)"),
        fr.ParameterReference(fr.params.STRATIFICATION_N2,
            hint="add a stratification module, e.g. nh.ConstantStratification(n2=...)"),)
    def diagnose(self, state, params):
        f0, n2 = params["coriolis.f0"], params["stratification.n2"]
        ...

# failure mode: no stratification module ->
#   MissingParameterError at assembly, attributed to PotentialVorticity,
#   with the hint and the list of provided parameters.

f0 = model.parameters["coriolis.f0"]                  # host-side resolved view
modes = nh.Eigenmodes(grid, params=model.parameters)  # fails correctly on beta planes
```

Precedents: Equinox `eqx.nn.Shared` (accessors, duplicates removed);
Oceananigans per-step argument threading; Flax linen path-keyed
collections / NNX graph split-merge (the mirror-namespace cost);
Gusto `Configuration` objects; MITgcm/MOM6 namelists (flat stringly
globals with no provider identity — what the collision/hint
machinery here fixes).
