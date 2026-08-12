---
status: frozen
date: 2026-08-12
---

# Units and field-metadata investigation

Owner-reported, 2026-08-12, two problems:

1. **Units are not scaling-correct.** "For a dimensional setup, the unit
   for time is s, for distance m, for velocity m/s etc. However, for a
   nondimensional model, all these units are 1. At the moment, this is
   true for some of the dimensions (e.g. time) but not for others (e.g.
   velocity). We need a system so that the units are correct reliably."
2. **Derived fields carry missing or wrong metadata.** "Some derived
   fields (for example the `rel_vort_z`) have no units or wrong one.
   Other attributes such as the long name etc. are also missing
   sometimes."

Method: four parallel worktree-isolated agents on `dev` (`3ffdab03`) —
a metadata-surface audit, a scaling-architecture design, a
derived-field/propagation design, and an external prior-art survey —
plus direct verification of the load-bearing claims. Measurements the
agents ran (rather than read) are marked **[m]**. Scope is the **new
stack** only (`spatial/`, `model/`, `nonhydro2/`, `shallowwater2/`,
`hydrostatic/`); the old stack is being retired.

## 1. Verdict

Both reports are real, and they are **two independent defects that
happen to share a data structure**. Neither is a numerical problem.

- Problem 1 is a **seam** problem: the writer already owns a
  scaling-aware rendering step, and variables do not pass through it.
- Problem 2 is a **declaration** problem: field algebra resets metadata
  by design, and the derived sites that need to re-declare it mostly
  do not — while seven of them actively *inherit the wrong record*.

Neither needs a units library, and neither needs dimensional analysis
through the field algebra (§6.6 — that option was prototyped and is
disqualified on measured evidence).

## 2. Problem 1 root cause: three seams, three different answers

The nondimensional-vs-dimensional decision is made in **three separate
places in `io/writer.py`**, which independently chose three different
conventions for the same situation:

| quantity | where its `units` is decided | nondimensional result |
|---|---|---|
| **time axis** | `io/writer.py:915-923` — explicit `if self._units.nondimensional:` → `units = "1"`, calendar anchor dropped | **correct** |
| **coordinates** | `io/writer.py:900-904` — stamps the unit *only* `if not nondimensional` | `units` key **absent** |
| **variables** | `io/writer.py:950-953` — `zattrs.update(layout.attrs)` straight from `FieldMetadata`, then the `dimensional_*` stamp on top; **never** overrides `units` | **wrong** — `u` still claims `"m/s"` |

So nothing is special about time. Time is *not a field*: its unit string
is synthesized at write time, after `Writer.bind(model)` (`io/writer.py:774`)
has put `self._units.nondimensional` in scope. Velocity *is* a field:
its `units="m/s"` is frozen at the declaration site
(`nonhydro2/modules/core.py:532`), folded by
`FieldDeclaration.field_metadata()` (`model/declarations.py:408-425`)
into the `FieldRecord` at assembly step 1 (`model/field_table.py:206`,
`model/assembly.py:1972-1978`) — **none of which consults the scaling,
even though `assemble()` itself receives `scaling=`**
(`model/assembly.py:1941`).

The variable branch is the method immediately after the time branch,
with `self._units.nondimensional` already in scope. The existing
mechanism is directly reusable.

**Provenance.** This state is not an oversight, it is a half-applied
ruling. `design/plans/active/nondimensionalization_plan.md` §D records
the 2026-07-21 decision that *"NetCDF output stays in model units; the
writer stamps ... per-variable `dimensional_factor` (+ target unit
string)"*, and separately the **CF option (b)** ruling that gave the
time axis its `"1"` (plan lines 520-525). Option (b) was applied to
time and never extended to variables or coordinates. Problem 1 is the
request to finish it.

## 3. Problem 2 root cause: reset-by-default plus an identity foot-gun

Metadata propagation through the algebra, measured in full **[m]**:

- **KEEPS:** unary `+`; `.to()`; `.retag` / `.with_variance`;
  `.real / .imag / .conj / .as_complex`; `.with_data` / `.with_storage`;
  and the whole interp/reconstruct/restrict family.
- **RESETS to blank:** `f±g`, `f±scalar`, `f*g`, `f/g`, `scalar*f`,
  `f/scalar`, `f**n`, `abs(f)`, unary `-`, `.diff()`,
  `.integrate/.mean/.max/.min`, `FiniteDifference`.
- **REPLACES:** `.nodes/.measure/.wavenumbers` → `name=x/dx/kx`.

One line: **conversion-shaped ops keep metadata, value-computing ops
reset it.** That rule is defensible and is *not itself the defect* —
the local inconsistencies it contains (`-f` resets but `+f` keeps;
`2.0*f` resets on `ScalarField` but keeps on `VectorField`, whose
`_keep_metadata` re-attaches the incumbent by key at
`spatial/fields/vector_field.py:630-662`) cause no user-visible harm.

Two things do cause harm:

**(a) Derived sites do not re-declare.** `nonhydro2/state.py:118`
`rel_vort_z` is `v.diff("x") - u.diff("y").to(...)` — two resetting ops
— and declares nothing, so it reaches the store as
`long_name: "Unnamed"`, `units: "n/a"`.

**(b) `with_data` borrows identity along with the space.** Seven
diagnostics build their result as `state["p"].with_data(...)` to reuse
pressure's *function space* (`nonhydro2/diagnostics.py:90` for `ekin`,
`:124` for `epot`, and the sw2 analogues). `with_data` preserves
metadata by contract, so they inherit **pressure's record**. `ekin`
reaches zarr as `long_name: "Pressure"`. The output *variable name* is
unaffected — the writer keys off the `fields=`/`derived=` key, not
`metadata.name` (`io/writer.py:938`) — but `.xr` is worse: **[m]**
`diagnostics.ekin().xr` is a DataArray literally named `"p"`.

The inherited `units` is a coin flip rather than a uniform error:
`m^2/s^2` is *correct* for specific kinetic energy and *wrong* for
`linear_pot_vort`. That is the signature of borrowed identity, not of a
units bug, and is why "just fix the unit strings" does not address it.

## 4. Inventory (measured)

Across nh2 / sw2 / hy, **47 user-visible quantities** (state fields,
State properties, bound diagnostics, coordinates, time) **[m]**:

- **3 (6%)** correct in both variants — all three are the time axis.
- **22 (47%)** carry a physical unit string that is false under
  nondimensional scaling.
- **22 of 47** have no usable `long_name` (15 blank in both variants,
  7 carrying a borrowed wrong one).
- **Derived quantities: 17 total — 7 correct, 7 actively wrong**
  (labelled `p` / "Pressure" / `m^2/s^2`), **3 blank**
  (`nh.rel_vort_z`, `nh.etot`, `sw.etot_full`).
  `hydrostatic` is **4/4 correct** — the house style already exists,
  it is simply not applied uniformly.
- **45 `FieldDeclaration` sites all declare both** `long_name` and
  `units`. None is partial. But **all 23 AUXILIARY sites lose them at
  allocation** (§5.2).
- **0 of 5 coordinates** carry a `long_name`.

## 5. Collateral defects found (each independent of the two reports)

### 5.1 `lon`/`lat` in radians are stamped `units: "m"` — wrong in the *dimensional* variant

`UnitFactor.unit` is the unit of `factor × value`, **not** the unit of
the stored value. `shallowwater2/units.py:156` returns
`UnitFactor(unit="m", expr="L", kind="coordinate")` for *both*
horizontal coordinates, and `io/writer.py:904` stamps that as a CF
claim on a dimensional model. On `fr.spatial.spherical.Grid` the stored
values are radians — measured range `[0.196, 6.087]` **[m]**.

Consequences: CDFViewer's radian→degree remap never fires, and
`--xunit km` would divide radians by 1000. Found independently by two
agents; the mechanism is confirmed in the source. **This is the one
finding here that is wrong today in the variant the owner uses most,
and it is not what was reported.**

Note the trap it sets: any design that promotes `model.units` to *the*
CF source of truth blesses this bug by construction.

### 5.2 AUXILIARY declarations silently discard their annotations

`RematerializationEntry` (`model/assembly.py:907-941`) has **no
`metadata` field at all**, so everything the module author wrote is
dropped at allocation (`model/model.py:1503-1509` →
`model/assembly.py:1179-1222`). `csqr` declares
`"Squared phase speed" / "m^2/s^2"` and ships `Unnamed / n/a`. 23 sites:
`f_coriolis`, `csqr`, `n2`, relaxation targets, source patterns,
background velocities. A 12-line fix was prototyped with tests green **[m]**.

### 5.3 `units="n/a"` is written to NetCDF verbatim

`spatial/export.py:295`, pinned by `tests/spatial/test_export.py:77`.
`"n/a"` is not udunits-parseable. (Amusingly `"N/A"` *does* parse — as
newton·ampere.) `b_relax_target` ships this way
(`model/modules/relaxation.py:216-217`).

### 5.4 `grid.create_field` cannot set a long name

The public factory has `name=` / `units=` sugar but **no `long_name=`**
(`spatial/grid.py:1028-1104`). A user cannot annotate their own field
fully through the supported surface.

### 5.5 Two unit sources — mostly complementary, not duplicated

`FieldMetadata.physical_units` and `UnitFactor.unit` were recorded
here as "two hand-maintained copies of the same fact". Re-measured
after the fix landed **[m]**, that framing is wrong and the correction
matters, because it rules out the obvious remedy:

| | `physical_units` | `UnitFactor` row |
|---|---|---|
| `u v w p b` | `m/s`, `m^2/s^2`, `m/s^2` | **identical** |
| `f_coriolis` (AUX) | `1/s` | no row |
| `x y z` | not fields | `m` |
| `t`, `T_ref`, `N_dim`, `f_dim` | not fields | `s`, `1/s` |
| derived (`ekin`, `rel_vort_z`, …) | declared | **no row** |

The genuine overlap is the **five component rows**. Everything else is
disjoint by construction, and for coordinates the two provably mean
different things (`rad` stored vs `m` after conversion, §5.1). So
making either side authoritative would force one concept to
impersonate the other — which is the bug of §5.1, generalized. What
was actually actionable split three ways:

1. **Five duplicated strings, unchecked.** Closed 2026-08-12: a
   per-package drift lint compares `physical_units` against
   `target_unit` for `component` rows that annotate a state field,
   with `coordinate` / `curated` / `constant` / `time` excluded and
   the reason stated in the test.
2. **`UnitFactor.unit` was misnamed.** It is the unit of
   `factor * value`, not of the stored value; reading it as a CF
   claim is what labelled radians metres. Renamed `target_unit`
   throughout (rows, `FactorEntry`, the writer stamp) 2026-08-12.
   The on-disk attribute was already correctly named
   `dimensional_units` and is unchanged.
3. **Derived quantities have no conversion row — still open.**
   Scoped 2026-08-12; see §5.6.

### 5.6 Converting derived quantities back (the open item)

Demonstrated on a `Rotational(L=2e3, U=0.5)` nonhydro store **[m]**,
writing `fields=["u"]` and `derived={"vort": …rel_vort_z}`:

| variable | `units` | `long_name` | `dimensional_factor` |
|---|---|---|---|
| `u` | `1` | Zonal velocity | `0.5` (= `U`) |
| `vort` | `1` | Vertical relative vorticity | **absent** |

Both are honest; only one is recoverable. Four findings bound the fix.

**(a) The factor cannot be derived from the unit string — it must be
declared.** If the nondimensionalization were a consistent `(L, T)`
rescaling, `m/s` would imply `L/T_ref`. Measured against the declared
rows on that model **[m]**:

| | declared | from dimensions | ratio |
|---|---|---|---|
| `u` `[m/s]` | 0.5 | 2 | `eps` |
| `p` `[m^2/s^2]` | 1 | 4 | `eps` |
| `b` `[m/s^2]` | 0.001 | 0.002 | `delta` |

The discrepancy is not even a single consistent factor: each variable's
amplitude follows the simplest-linear-operator rule independently, and
the vertical scale `delta*L` is a second length. Dimensional analysis
under-determines the answer, so per-quantity rows are the only route.

**(b) No new machinery is needed.** Rows arrive through the duck-typed
`module.unit_factors` mapping, which the cores already use for
components and coordinates; `UnitFactor` reads live bound parameters
(`params={"eps": SCALING_NONLINEARITY}` + `fn`), so `eps`- and
`delta`-dependent factors are expressible; and `UnitsView` marks a row
unresolvable rather than failing when its inputs are unbound — which
is what a diagnostic that needs an absent module should do.

**(c) The stamp is keyed by the wrong name.** `io/writer.py` looks the
row up by **output name**, which the user chooses (`derived={"vort":
…}`), while a package row can only be keyed by the canonical name
(`rel_vort_z`). The fix is a fallback to `field.metadata.name`, which
the per-package name-identity gate now pins to the canonical key — a
second use for that gate beyond the one it was built for.

**(d) Ten of the seventeen are mechanical; the rest are not.**

| factor | quantities |
|---|---|
| `U/L` | `nh.rel_vort_z`, `hy.rel_vort_z`, `hy.hor_divergence`, `sw.rel_vort`, `sw.divergence` |
| `U^2` | `ekin`, `epot`, `etot` in nh / sw / hy |

`nh.epot` is worth spelling out because it looks scaling-dependent and
is not: `factor_b^2 / factor_N^2` = `[U^2/(eps*delta*L)]^2 *
[Fr*delta*L/U]^2` = `U^2*(Fr/eps)^2`, and the nondimensional definition
carries `N^2_eff = (eps/Fr)^2`, which cancels it back to `U^2` — so it
agrees with `ekin` and `etot` is well defined.

The remaining five need deriving individually, not pattern-matching:

- **`nh.linear_pot_vort` = `U/(eps*L)`, not `U/L`.** Its nondimensional
  definition multiplies through by `eps` (`diagnostics.py:191`), so it
  is *not* the plain vorticity factor despite sharing the unit `1/s`.
  This is the same quantity whose borrowed unit was wrong by four
  powers; it punishes pattern-matching twice.
- **`sw.epot_full` = `(U^2/eps)^2`** from `0.5*p^2`.
- **`sw.ekin_full`, `sw.etot_full`, `sw.pot_vort`** — unresolved here.
  `ekin_full` is thickness-weighted (`0.5*h_bar*u^2`) and the shipped
  `thickness` row is `(U/Fr)^2` while `p` is `U^2/eps`; I could not
  reconcile those into one factor that also makes `etot_full =
  ekin_full + epot_full` consistent under a non-`GravityWave` sw
  scaling without reading further into the sw geopotential
  conventions. **Owner call / further derivation needed** — flagged
  rather than guessed.

**(e) A package row can never cover a user's own quantity.**
`derived={"my_thing": lambda ms: …}` is opaque to every table, so
full coverage additionally needs a way for the writer's derived
channel to accept a declared factor alongside the callable.

**Options.** (A) package rows for the shipped seventeen + the
`metadata.name` keying fallback; (B) a user-declarable factor on the
`derived=` channel, for ad-hoc quantities; (C) both; (D) none —
document that derived quantities are unconvertible and let users
apply the component factors themselves. A alone leaves ad-hoc
quantities out; B alone leaves the shipped diagnostics needing
hand-declaration at every call site.

**Recommendation: C, staged** — accepted and **shipped**
2026-08-12 (`feat/derived-unit-rows`), except the `sw` family.

- A new raw row kind `derived` (identity on a dimensional model, as
  a diagnostic computed in SI already is), contributed by each core
  alongside its component and coordinate rows — no new machinery.
- 14 rows: `U/L` for `nh.rel_vort_z`, `hy.rel_vort_z`,
  `hy.hor_divergence`, `sw.rel_vort`, `sw.divergence`; `U^2` for
  `ekin`/`epot`/`etot` in all three; `U/(eps*L)` for
  `nh.linear_pot_vort`; `(U^2/eps)^2` for `sw.epot_full`. Each is
  pinned to its arithmetic in the package's `test_units.py`.
- `sw.ekin_full` / `etot_full` / `pot_vort` stay **absent**, with a
  test asserting the absence and naming the open call, so nobody
  reads it as an oversight.
- The writer falls back from the user-chosen output key to
  `layout.name` (the canonical name the identity gate pins), so
  `derived={"vort": …rel_vort_z}` picks up the `rel_vort_z` row.
- `Writer(unit_factors={...})` declares a row for an ad-hoc
  quantity, resolved through the new public
  `UnitsView.resolve(name, factor)` — same scaling switch, same live
  reads, same unresolvable marking as a collected row.
- The per-package derived-metadata gate additionally asserts that a
  quantity's row `target_unit` equals its declared physical unit —
  the second overlap between the two unit sources, now pinned like
  the component one (§5.5).

Measured end to end **[m]**: on the `Rotational` store that opened
this section, `vort` now carries `dimensional_factor = U/L = 2.5e-4`
beside `u`'s `U = 0.5`, and an ad-hoc `mine` row declared at the
writer resolves to `U^2/L`.

## 6. Constraints that bound the design

**6.1 Metadata does not retrace — the design space is wider than assumed.**
`ScalarField` is jaxified `annotation=("_metadata",)`
(`spatial/fields/scalar_field.py:82`), which excludes it from aux
equality (`framework/utils/jax_utils.py:255-266`). Measured: one trace
across differing units **[m]**, already pinned by
`tests/spatial/fields/test_metadata_treedef.py`. Metadata may vary
freely at no compilation cost. The corollary is that traced outputs
carry **stale first-trace metadata**, so metadata must never be
load-bearing for correctness.

**6.2 The `Grid` does not know the scaling.** Fields are built by
`grid.create_field(...)` and by module bind code; scaling is a `Model`
concept. Whoever renders the unit string must sit at or above assembly.

**6.3 Field algebra runs inside the traced step** (~57 metadata-touching
ops in a nonhydro2 step body **[m]**) — but metadata is a *concrete host
object* at trace time, so a metadata check costs nothing at runtime.

**6.4 Nevertheless, the algebra must never raise or warn.** A raise is
fatal to compilation over an annotation concern; it is unsound given
6.1; and it would not fire usefully — with dimensions fully wired,
**33/33** additive combinations in a real step were already poisoned to
blank, 0 genuine mismatches **[m]**. Loud failure belongs host-side
(`BoundDiagnostic`, `Writer.bind`, tests), all of which are eager.

**6.5 CF: the dimensionless spelling is `"1"` or omission — never `""`.**
`"1"` is canonical (571 of 5071 standard-name entries); CF states a
variable with no `units` is assumed dimensionless; udunits maps `""` to
its **`unknown`** sentinel, *not* to dimensionless. `"1"` is already the
house spelling (`io/writer.py:922`, `model/params.py:144`). CDFViewer
renders it `[1]` and never rescales it, whereas a nondimensional
coordinate labelled `"m"` *would* be silently divided by `--xunit km`.
Dissent worth recording: NASA ESDSWG discourages `units="1"` and
prefers omission; iris reads omission as `unknown`. Every spelling has
an objection; `"1"` has the fewest and is already in use.

**6.6 CF forbids `standard_name` on a nondimensional variable.**
Conformance requires a variable's units be physically equivalent to its
standard name's canonical units. `standard_name` and nondimensional
mode must switch off together — which rules out "derive units from CF
standard names" as a single mechanism. (Also: CF has **no** standard
name for buoyancy, buoyancy frequency in sea water, horizontal
divergence, or volumetric KE density, so `standard_name` must be
optional regardless.)

**6.7 Dimensional analysis through the algebra is disqualified.**
Prototyped at 112 lines with all 271 field tests passing and
`u*u → m^2/s^2` correct **[m]** — then disqualified on three findings:
(1) undimensioned Python constants make it **confidently wrong** —
`0.5*b**2/n2` yields `m^2/s^4`, which is worse than `n/a` because it
looks authoritative; (2) no mesh carries a unit, so `.diff()` cannot
subtract a length; (3) the operator layer resets metadata by design
(`spatial/operators/products.py:55`), so nothing propagates in a real
model anyway. Independently, xarray ≥ v2025.11.0 now preserves attrs
with `drop_conflicts`, and `u / v` with both `m/s` retains
`units: 'm/s'` **[m]** — blind inheritance is *worse* than the current
blank. Iris resolves this exact case by deriving units dimensionally
and then setting the name to `unknown`, because the name is not
derivable. **Dimensional analysis can never supply `long_name`**, which
is half of what was reported.

## 7. Prior art, condensed

Of 17 simulation codes surveyed, only **veros** switches annotation on
the run configuration (its `units` is a callable of settings).
**Oceananigans considered and explicitly rejected** a nondimensional
switch (issue #1462). **pyqg has FRIDOM's exact bug**: hardcoded unit
literals on a code that runs nondimensionally. Most (Dedalus, Nek/nekRS,
SpectralDNS, AMReX, PyClaw, FourierFlows, Gusto) annotate nothing;
Channelflow declares `Conventions="CF-1.0"` and writes zero units.

No simulation code records reference scales by design — but two
standards outside CF solve it properly, and FRIDOM's existing
`dimensional_factor` stamp is already the same idea:
**openPMD** (`unitSI` + a 7-element `unitDimension` exponent vector) and
**CGNS** (`DataClass_t` = `Dimensional` / `NormalizedByDimensional` /
`NormalizedByUnknownDimensional`). `scale_factor` is *not* the tool for
this — ESDSWG calls it a misuse of the packing standard, and
xarray/zarr re-scale symmetrically, so an open→write cycle corrupts
data **[m]**.

Python units libraries, judged against a jax codebase: **pint** cannot
wrap a jax array under `jit` (rejected as not a valid JAX type) but is
excellent *host-side* (`.dimensionality`, and cf-xarray's `:cf`
formatter emits exact CF strings); **unyt** cannot wrap a jax array at
all and costs 3.25 s to import; **astropy** silently pulls to numpy —
the worst outcome. **saiunit** (Nature Comms 2025) is the interesting
one: it puts the mantissa as the single dynamic leaf and the unit in
the treedef, which is structurally identical to where `FieldMetadata`
already lives, and jit/grad/vmap all worked **[m]** — evidence the
placement is sound, though it is v0.5.x and untested against
scan/remat/sharding. For FRIDOM's actual need none of them is
warranted.

Failure policy is unanimous where units *are* modelled: every library
**raises** on mismatched `+` (pint, astropy, unyt, saiunit, Iris);
`*` and `/` are always allowed; no warn-mode exists and none is being
requested. Degrading to *dimensionless* has no precedent and is the
dangerous option. astropy's `UnrecognizedUnit` is the model worth
copying: it serializes but refuses to participate in arithmetic.

Note the tension with 6.4: the consensus applies to libraries whose
units are an *arithmetic invariant*. In nondimensional mode a strict
checker degenerates to a no-op, since everything is dimensionless —
which argues for treating units as an **I/O annotation**, not an
invariant, and keeps the loud failures host-side.

## 8. Options — problem 1 (scaling-correct units)

| | A: writer patch | B: render at assembly | C: `UnitsView` sole truth | **D: hybrid** |
|---|---|---|---|---|
| src files | 1 | 3 | 8+ | 5 |
| declaration sites edited | 0 | 0 | 43 deleted | 0 |
| fixes `.xr` / in-memory | no | yes | yes | yes |
| fixes coordinates | partly | no | yes (but see 5.1) | yes |
| public API | none | `+scaled_units=` | **removes** `units=` | additive ×2 |
| hook for problem 2 | no | no (destroys the physical string) | no | yes |
| tests changed | ~2 | ~3 | ~50 | ~4 |

**Recommended: D.** One function
`stored_units(native, *, nondimensional, scaled)` in `model/units.py`,
applied at assembly for fields and reused by the writer for coordinates
and time, so all three seams of §2 answer from one place.
`FieldMetadata` gains a `physical_units: str = ""` slot so the physical
string survives rendering — that slot is what lets a nondimensional run
still know what a quantity *would* be dimensionally, and it is the seam
problem 2 needs.

**Not C.** It looks free (declared-vs-row agreement measured 18/18), but
it forces rendering after bind (hydrostatic's rows are bind-captured,
`hydrostatic/modules/core.py:465`), obliges masks and relaxation targets
to own conversion rows, and — decisively — makes `model.units` the CF
authority, which blesses the radians-as-metres bug of §5.1.

**Not A alone.** Correct on disk while `field.metadata.units` and `.xr`
still say `"m/s"` is not "reliably".

## 9. Options — problem 2 (derived metadata)

| | A declare | B dimensional analysis | C registry | D write seam | E `new_quantity` |
|---|---|---|---|---|---|
| fixes wrong units | yes | yes* | structurally | writer only | wrong→blank |
| fixes `long_name` | yes | **never** | yes | writer only | no |
| fixes `.xr` | yes | partly | yes | **no** | partly |
| enforceable | with a name-identity gate | no | **yes** | at bind | no |
| ad-hoc user lambda | manual | free but **wrong** | no help | **natural home** | n/a |

**Recommended: A + E, gated, with D for ad-hoc quantities; not B.**
Declare metadata explicitly at the ~10 derived sites; add
`ScalarField.new_quantity(data)` as the space-borrowing spelling that
does *not* also borrow identity, and convert the 7 `with_data` sites to
it; give the `derived={...}` writer channel a natural place to name a
user's ad-hoc quantity.

**Enforcement is the part that answers "reliably."** A naive
"is it default?" check finds only **3 of 10** defects; keying on
`metadata.name == registry key` finds **10/10 with 0 false positives**
**[m]**. That gate belongs in a per-package
`tests/<pkg>/test_derived_metadata.py`, plus a standing invariant test
that a nondimensional model reports no physical unit anywhere.

## 10. Recommended shape

Ordered so each phase is independently mergeable and independently
useful:

0. **AUX metadata stamp** — §5.2, 12 lines, prototyped green. Standalone.
1. **`lon`/`lat` radian units** — §5.1. Standalone, and it is a live bug
   in the dimensional path.
2. **`stored_units()` in `model/units.py`** + `physical_units` slot;
   apply at assembly; repoint all three writer seams at it. Fixes
   problem 1 including `.xr`.
3. **`new_quantity` + declare at the derived sites** (7 + ~10 edits).
   Fixes problem 2.
4. **Gates**: name-identity test per package; nondimensional-purity
   invariant test; replace the `"n/a"` default with a CF-legal spelling
   (§6.5) and repoint `tests/spatial/test_export.py:77`.
5. **`long_name=` on `grid.create_field`** (§5.4).

## 11. Owner rulings (answered 2026-08-12)

1. **In-memory spelling** — `field.metadata.units` reads `"1"` on a
   nondimensional model. The record stores `physical_units` plus a
   `nondimensional` flag and derives `units` from the pair, so there
   is one stored truth and no stale physical claim to leave behind.
2. **Dimensionless spelling** — `"1"`, per §6.5.
3. **The `"n/a"` default** — `"unknown"`, the iris/cf-units spelling
   (`Unit("unknown").is_unknown()` true, `is_dimensionless()` false;
   [SciTools/iris#935]). It serializes and is never a CF claim.
   Consequence adopted with it: an **undeclared** unit stays
   `"unknown"` under *either* scaling — nondimensionalizing an
   unknown quantity yields an unknown quantity, not a dimensionless
   one.
4. **`standard_name`** — not adopted. It must switch off under
   nondimensional scaling (§6.6) and does not cover four of our
   quantities, so it cannot be the mechanism; revisit separately if
   CF interchange ever becomes a goal.
5. **§5.1** — in scope, fixed here.

## 12. What shipped

Option **D** for problem 1 and **A + E with the name-identity gate**
for problem 2, as recommended.

- `FieldMetadata` stores `physical_units` + `nondimensional` and
  derives `units`; `units=` remains the writing sugar on `create()`
  / `replace()`, so **none of the 45 declaration sites changed**.
  The assembly stamps the flag once from the scaling
  (`model/assembly.py` step 1 -> `FieldRecord.from_declaration`).
- The three seams of §2 now answer from that one rule: variables and
  `.xr` render from the metadata, coordinates render at the writer
  (`"1"` instead of an omitted attribute), time keeps its option-(b)
  branch with the shared constant.
- `ScalarField.new_quantity` — borrows the space, not the identity;
  inherits only the scaling flag, so a diagnostic declares its
  physical unit once and reports correctly under either scaling.
- §5.1: the fix went to the **chart**, not the factor row.
  `CoordinateMapping` gained `coordinate_units=` (declared as radians
  by `lonlat_sphere`), surfaced as `grid.coordinate_units` and applied
  at export; the writer fills only a unit the layout did not declare.
  This keeps the two facts apart — what the stored values *are* versus
  what the factor converts them *to* — which is the confusion that
  produced the bug.
- §5.2 (AUXILIARY annotations) and §5.4 (`long_name=` on
  `grid.create_field`) fixed. The remat table takes its metadata from
  the field record, so AUX fields carry the scaling stamp too,
  including across `update_parameters` re-runs.
- **The scaling frame now rides through the algebra** (closed
  2026-08-12, `fix/scaling-flag-propagation`). This shipped first as
  an accepted gap: value-computing ops reset the *whole* metadata
  record, so six re-declaring sites that do not land on a state
  field's space passed `nondimensional=` by hand — which cut against
  "declaration sites spell the physical unit and never the scaling",
  and left an unregistered derived quantity able to ship a false
  physical claim. `FieldMetadata.cleared()` now expresses the right
  rule: the identity (name, long name, unit, nc-attrs) is dropped
  because the operands say nothing about the result, while
  `nondimensional` survives because it describes the **value
  system** the numbers live in, not the quantity they measure.
  Applied at every reset site — the elementwise products path, the
  two staggered-kernel chokepoints (`reconstruct`/`staggering`,
  covering `diff`, flux-diff and the graded ladder), `__neg__`,
  `abs`, the linear-combine fast path, and both scalar
  shift/scale paths — and all six hand-threadings deleted.
  Verified across `+ - * / ** abs -` , `diff` and `to`.
  One genuine bug fell out: `_lift_field` (the constant-space
  broadcast a sanctioned lift performs) dropped metadata entirely,
  so `f_coriolis` lost its annotation the moment it entered an
  expression — `sw.pot_vort` was the visible casualty. A lift is a
  conversion, not a computation, so it now keeps the record.

- §5.5 (the two hand-maintained unit copies) is **not** addressed:
  `FieldMetadata.physical_units` and `UnitFactor.unit` remain
  independent. They now mean provably different things for
  coordinates (§5.1), so a drift lint would need to compare only the
  component rows — left as a follow-up.

[SciTools/iris#935]: https://github.com/SciTools/iris/issues/935
