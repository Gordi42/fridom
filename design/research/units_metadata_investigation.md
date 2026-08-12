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

### 5.5 Two hand-maintained copies of the same fact

`FieldMetadata.units` and `UnitFactor.unit` duplicate each unit string.
They agree today at **18/18** across nh/sw/hy **[m]**; nothing checks
it. Derived quantities exist only in the first, coordinates only in the
second.

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

## 11. Owner rulings needed

1. **What does `field.metadata.units` read in-memory on a nondimensional
   model — `"1"`, or the physical string with `"1"` rendered only at
   I/O?** Option D assumes the former (it matches the report's wording:
   "for a nondimensional model, all these units are 1").
2. **Dimensionless spelling: `"1"` or omit the attribute?** §6.5 —
   recommendation `"1"`, already the house spelling.
3. **Replacement for the `"n/a"` default** (§5.3) — `"1"` is wrong
   (claims dimensionless), omission loses the "nobody set this" signal.
   An `UnrecognizedUnit`-style sentinel that serializes but is never a
   CF claim is the prior-art answer.
4. **Is `standard_name` wanted at all?** It must switch off under
   nondimensional scaling (§6.6) and does not cover four of our
   quantities.
5. **Is §5.1 in scope for this work** or a separate fix? It is a
   dimensional-path bug and unrelated to what was reported.
