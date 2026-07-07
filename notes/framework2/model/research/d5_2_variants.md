# D5.2 — model.variant() and the term-predicate vocabulary

Research report (see [`README.md`](README.md) for status).

## 1. The predicate vocabulary

**ClosureBase — designed, since the old code has none**
(`SmagorinskyLilly` and `BiharmonicClosure` subclass `Module`
directly; only the harmonic/biharmonic family shares
`HarmonicDiffusion`). Ruling: introduce **`fr.closures.ClosureBase`**
— an abstract Module subclass that earns its keep beyond the
predicate: it hosts D1.4's role-target resolution boilerplate
(friction → Velocity family, mixing → TRACER, name-keyed overrides),
so closures subclass it for the ergonomics and the isinstance
predicate follows free. A parallel `Module.category` tag axis
rejected (second classification vocabulary with one consumer; as
forgettable as the base class). Foreign closures stay reachable via
`owned_by(TheirClass)` / `named(...)`.

**Five leaves, three combinators** — `term_filter` is a
keep-predicate `matches(term, owner) -> bool`, host-side, never
traced:

| Predicate | Matches | Why |
|---|---|---|
| `fr.terms.linear` | the declared `linear` tag | the motivating case; trust inherits the tag's strict, JVP-lintable contract |
| `fr.terms.explicit` / `implicit` | treatment | backward runs drop implicit diffusion (backward diffusion is ill-posed); solver debugging |
| `fr.terms.owned_by(Type)` | isinstance(owner, Type) | the family axis (`~owned_by(ClosureBase)`) |
| `fr.terms.named(*keys)` | exact "Module/term" keys (from the report) | surgical escape; **unknown keys error at build** (kills the silent-typo footgun); globs designed-for |
| `fr.terms.advancing(*fields)` | intersection with `advances` | splits one module's terms (Smagorinsky: stress advancing u,v,w vs κ mixing advancing tracers) |

**Rejected**: `transporting(...)` — duplicates `advancing` at term
granularity while *suggesting* per-field masking it cannot deliver
(an advection scheme is one term advancing many fields; "freeze one
tracer" is a term rewrite, not a filter — documented); bare
lambdas (unfingerprintable; `fr.terms.where(fn, token=)`
designed-for). Combinators `& | ~`; the identity filter is
`term_filter=None`.

**Predicate objects**: frozen dataclass expression trees, not
pytrees, hashable, canonically reprable; each node has
`fingerprint_token()` — `OwnedBy` tokens use qualified class names
(stable across processes; the filter enters the restart-fingerprint
source record). No simplification.

## 2. `model.variant(term_filter=..., updates=..., name=...)`

Re-assembly on the parent's **frozen grid** (the one-grid-many-models
idiom) + two ingredients: the term filter at step 5, parameter
seeding at step 8. Returns a plain `fr.Model`.

1. Same grid (verify path, never renegotiates).
2. Same module tuple; **declarations unfiltered** — filters act on
   terms, never modules/declarations; stages survive. Consequence:
   the variant's FieldTable, State treedef, shapes, halos, and
   layouts are **identical to the parent's** — parent↔variant state
   exchange is treedef-exact and copy-free. (The load-bearing
   property of every Tier-2 inner loop.)
3. Terms filtered at step 5.
4. **`updates=`** resolves through the binding table — and, because
   this is an *assembly*, may change a parameter's **value spec**
   (scalar → `fr.Ramp`, dt sign): OB's backward variant requires it.
   The *carry* treedef may then differ; the load-bearing identity is
   the **State** treedef.

**Parameter sharing: snapshot at variant creation.** Live sharing is
*impossible*, not merely forbidden — carries are functionally
replaced every chunk, so "shared" module objects are stale after the
parent's first step; within one pytree, aliasing is the D2
double-flatten bug. Snapshot-at-creation is deterministic, free
(immutable arrays — structural sharing), matches the fingerprint
moment, and satisfies
`variant(f, u) ≡ fresh-assembly(current leaves) + u`. Rejected:
pristine-inputs rebuild (ignores pre-variant parent updates);
deferred-to-call-time (assembly in OB's hot loop; frozen config
becomes a moving target — violates law 1). Refresh is a one-liner
(identical binding tables); `Transform.resync()` designed-for. The
variant is a full lifecycle citizen (reset/set_state/
update_parameters/snapshots).

## 3. Assembly integration

- Steps 1–4 unchanged (re-run semantics; record reuse is an
  optimization). Step 3: override keys verified identical, not
  re-merged (the frozen path).
- **Step 5 — the filter acts here**, post-bind, before everything
  that consumes terms; implicit merging sums only surviving κ
  contributors (exact).
- Step 6 — dry run on the filtered set only, errors named for the
  variant.
- **Step 7 — verify path, ⊆ ruling**: filtered demands are always a
  subset (fewer terms → fewer operators → ≤ halo; same state-space
  set; same override keys; updates add no operators). **The verify
  contract is demand-satisfaction (⊆/≤), not equality** — amendment
  owed to the grid notes if worded as equality. The variant
  *inherits* the parent's layouts/halos: possibly oversized, always
  correct — a feature (copy-free state exchange), not a compromise.
- **Coverage lint**: error → info under any filter (a linear variant
  leaves `b` un-advected by design; the info line is the audit
  trail); double-transport stays an error (filtering only relaxes
  it).
- `supported_treatments` checked against the filtered set (IMEX
  stepper over an all-explicit filtered set passes).
- **Fingerprint: the filter is structure** (canonical token +
  updates specs); variant snapshots never silently load into the
  parent — the mismatch diffs. Same filter/specs on identical
  parents share the jit cache.
- **Empty filter result / unknown `named` key: build-time error.**
- Stage filtering: designed-for only (seed position stands).

## 4. Empty implicit partition: graceful, ruled

The IMEX driver's implicit side is a loop over merged operators —
with the mixing term filtered, the set is empty and the solve loop
**is not emitted at trace time**; buffers already partition by
treatment; γ becomes dead static data. CNAB2 minus L is **textbook
AB2** (documented nuance: without the AB stepper's `eps` — the
degenerate variant is mathematically the same order, not bitwise
identical to `AdamBashforth`; document, don't inject eps). SBDF2
likewise. The reverse case (implicit term surviving under an
explicit parent stepper) cannot arise (parent assembly already
errored).

## 5. Old-code archaeology and the port table

Verified line-by-line: **GTA** deepcopied mset, disabled *only
advection* (closures — including nonlinear Smagorinsky — kept
running in the "linear" twin!) and the writer container; per pass:
dt sign flips, reset, N steps, running mean. **OB**: two internal
models (the docstring's "negative viscosity" `mset_backwards` is
**never used anywhere in the repo**, grep-confirmed); per-step
host mutation `advection.scaling = ramp(θ)·Ro`; reset per leg.

| Old | New |
|---|---|
| deepcopy(mset) + internal Model | `model.variant(...)` (snapshot semantics; verify path) |
| GTA advection.disable() | `term_filter=fr.terms.linear` — **behavior delta**: correctly also drops Smagorinsky (old kept it — arguably a bug); harmonic/biharmonic survive (linear). Pin in cutover tests |
| diagnostics.disable() | nothing — `advance` is IO-free |
| per-step scaling mutation | `updates={"scaling.rossby": fr.Ramp(...)}` — the callback dies |
| backward: 1−θ ramps, dt<0 | Ramp endpoints are leaves (swap via update_parameters, no recompile); dt = TIME_STEP leaf flip |
| per-leg reset | `reset()` (clock reset restarts Ramps, re-warms) |
| mset_backwards "negative viscosity" | `backward_filter=~owned_by(ClosureBase) & ~implicit` — the cleaner physics (drop dissipation rather than sign-flip it) |
| two persistent models | two variants, one per Propagator (isolation law) |

## 6. Ergonomics / naming

`fr.linearize(model)` ≡ `variant(term_filter=fr.terms.linear)`,
named `"{parent}/linear"`. Transforms take the user's model as a
spec and build their variant internally (`filter=` kwarg; default
per transform); passing a pre-built model is **ownership transfer**.
Variant names default to `"{parent}/variant({predicate})"`
(truncated + hash suffix); full predicate/updates/lint-downgrades in
`model.report`; attributed errors carry the variant name.

## 7. Risks / open questions

Verify-path ⊆ wording (grid-notes amendment); qualname fingerprint
collisions under interactive redefinition (accepted, 02_rules note);
term granularity documented (no per-field masking); snapshot
staleness (documented + designed-for resync); `linear`-tag honesty —
**prioritize the JVP debug lint in 2.5** now that linearize consumes
the tag; one blessed Ramp endpoint-swap spelling
(`Ramp.reversed()`?) — small follow-up; variant memory = the known
Tier-2 carry cost.

## 8. Sketches

```python
inviscid_linear = fr.terms.linear & ~fr.terms.owned_by(fr.closures.ClosureBase)
average = nh.transforms.TimeAverage(model, max_period=2*np.pi/f0, n_ave=4,
                                    filter=inviscid_linear)

Ro = model.parameters["scaling.rossby"]
fwd = model.variant(term_filter=~fr.terms.owned_by(fr.closures.ClosureBase),
                    updates={"scaling.rossby": fr.Ramp(0.0, Ro, period=T, curve="exp")},
                    name="run/ob-forward")
bwd = model.variant(term_filter=(~fr.terms.owned_by(fr.closures.ClosureBase)
                                 & ~fr.terms.implicit),      # backward diffusion ill-posed
                    updates={"scaling.rossby": fr.Ramp(Ro, 0.0, period=T, curve="exp"),
                             fr.params.TIME_STEP: -dt},
                    name="run/ob-backward")
```
