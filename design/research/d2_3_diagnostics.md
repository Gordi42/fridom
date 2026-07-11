---
status: frozen
date: 2026-07-07
---

# D2.3 — Where physics diagnostics live, and their API

Research report (see [`README.md`](README.md) for status).

## 1. Taxonomy: every old State diagnostic, verified

Verified against the property bodies in `nonhydro/state.py` and
`shallowwater/state.py`.

**nonhydro** (u, v, w, b):

| Diagnostic | Verified formula (as coded) | Needs beyond state+grid | Class |
|---|---|---|---|
| `velocity`, `tracers` | positional slices `self[:3]`, `self[3:]` | none (D1.4 `table.velocity()` replaces) | accessor |
| `ekin` | `0.5(u² + v² + dsqr·w²)` | **dsqr** | **(b) param-dependent** — *not* plain ½(u²+v²+w²)! |
| `epot` | `0.5 b²/N²` if `n2 != 0`, else `b·z_mesh` | **N² field** + a host-side branch | (b) + structural branch |
| `etot` | ekin + epot | dsqr, N² | (b) |
| `rel_vort_x` | `dsqr·∂y w − ∂z v` | **dsqr** | **(b)** |
| `rel_vort_y` | `∂z u − dsqr·∂x w` | **dsqr** | **(b)** |
| `rel_vort_z` | `∂x v − ∂y u` | none | **(a) parameter-free** |
| `pot_vort` | full Ertel PV with Ro, f0, N² | f0, N² field, Ro, dsqr | (b) |
| `linear_pot_vort` | `Ro(f0/N²·∂z b + ζz)` | f0, N², Ro | (b) |
| `local_rossby_number` | `Ro·ζz/f_coriolis` | Ro, f_coriolis **field** | (b) |
| `cfl` | `max(|u|dt/dx, ...)`, dt from `mset.time_stepper.dt` | **stepper dt** + spacing | **(c) model context** |

**shallowwater** (u, v, p): `ekin` = `0.5 Ro² h_full (u²+v²)` with
`h_full = csqr_field + Ro·p` → **(b)**; `epot`, `etot`, `pot_vort`,
`local_rossby_number` → (b); `spectral_ekin`, `rel_vort` → (a);
`cfl` → (c).

**Decisive finding: the parameter-free class is nearly empty** —
only `rel_vort_z` (nh) and `rel_vort`/`spectral_ekin` (sw). Even
`ekin` carries `dsqr`/`Ro²csqr` scaling in both packages. Any design
treating State properties as the primary diagnostic home designs for
an empty set. **Functions must be the primitive.** Also: the
field-valued parameters these consume (`f_coriolis`, `N²`, `csqr`)
are AUXILIARY components — but a bare State (eigenmode output,
snapshot slice) may not carry them, so diagnostics must not
*require* them to be in the state vector.

## 2. Recommendation: hybrid, functions as the primitive

**R1 — Primitive: pure, explicit diagnostic functions**, one flat
namespace per package (`nh.diagnostics`, `sw.diagnostics`):

```python
def pot_vort(state, *, f0, n2, rossby_number=1.0, dsqr=1.0) -> fr.ScalarField
```

Explicit params (scalars **or** ScalarFields — the field algebra
handles both); no model back-reference; works on any State from
anywhere; jit-traceable; unit-testable. Each function carries a
declarative requirement annotation — the parameter-side twin of
`FieldReference` (`@requires(ParamRef("n2", hint=...))`) — consumed
only by the binding layer, invisible when called directly.

**R2 — Bound form: `model.diagnostics`, built at assembly.** The
core module that supplies `state_type` also supplies the diagnostics
namespace (the D1.3 commitment-4 channel). The model wraps each
function with parameters resolved through D2:

```python
model.diagnostics.pot_vort()        # on the current carry state
model.diagnostics.pot_vort(state0)  # on an explicit state
```

Binding is **lazy per call**: a hinted `MissingParameterError` fires
only when a diagnostic actually needing an absent provider is
invoked (mirrors the D1.5 vocabulary-property contract).
Ramp-valued provided values resolve at the carry clock at call time.

**R3 — Writer seam: named output expressions, evaluated at output
cadence** (the Oceananigans `outputs=` pattern):

```python
writer = fr.io.Writer(
    fields=("u", "b"),
    derived={"pv": model.diagnostics.pot_vort,
             "zeta": nh.diagnostics.rel_vort_z},
    trigger=fr.every(hours=10),
)
```

The seam's only contract: every output is a pure
`(model_state) -> ScalarField | scalar`. Where it evaluates
(in-trace under `lax.cond` feeding `io_callback` vs host-side at
chunk boundaries) is owned by 2.6 — both satisfy the seam.

**R4 — Reject diagnostic-provider modules as the default**
(always-on per-step cost, carry growth, treedef churn, warm-start
staleness confusion; MITgcm's registry is the anti-pattern).
**Escape hatch retained**: a diagnostic genuinely needing
step-frequency state (time averages, accumulated budgets) is a real
module owning an AUXILIARY field via `self_update` — nothing new.

**R5 — Parameter-free ones** additionally appear as State
vocabulary properties (`state.rel_vort_z`) — thin wrappers
delegating to the same functions. A courtesy, not the architecture.

## 3. Parameter-plumbing interface (assumption for D2.1)

Needs exactly: (1) **`model.parameters`** — a read-only, flat,
host-reachable mapping of resolved provided values (the same
surface IC recipes and eigenmodes require — diagnostics are just a
third host-side consumer); (2) a **`ParamRef(name, hint)`**
requirement object — one idiom with `FieldReference`; (3)
field-valued parameters resolve to their AUXILIARY component/owner
so the bound wrapper fetches the current value from the carry.
Explicitly *not* assumed: any traced-side registry, State→model
back-reference, or resolution inside jit.

## 4. Traced-run vs interactive

- **Traced run**: derived outputs evaluated only at output cadence;
  the step carries zero diagnostic cost. jax makes Oceananigans'
  lazy AbstractOperations unnecessary — an eager field-algebra
  function traced under jit fuses anyway.
- **Notebook**: `model.diagnostics.etot().xr.plot()` — successor of
  `model.z.etot.xr.plot()`; one pair of parens more, honest about
  computing.
- **Offline on written zarr**: diagnostic functions are field
  algebra (staggered diff/interp) — they need Fields. Blessed path:
  re-assemble the model (cheap — the producing script has the
  recipe), load a snapshot, call
  `model.diagnostics.pot_vort(state)`. Pure-xarray postprocessing
  stays possible by convention (`.xr` export is xgcm-compatible),
  not machinery.

## 5. Energy / budget scalars

Domain-integrated scalars are the same primitive at rank 0:
`nh.diagnostics.total_energy(state, *, dsqr, n2)` built on the
grid-owned `integrate` — exactly where `norm_l2`/`dot` were evicted
to. Time series are a different **sink**, not a different contract:
`fr.io.TimeSeries(outputs={"E": model.diagnostics.total_energy},
trigger=...)` — designing the sink is 2.6's job. Term-wise budget
closure (enabled by contribution dicts) is step-frequency
accumulation → the R4 module escape hatch, deferred.

## 6. Sketches

```python
# (a) notebook, live model
model.run(hours=5)
model.diagnostics.pot_vort().xr.sel(z=0).plot()          # bound via D2
nh.diagnostics.pot_vort(model.state, f0=2e-4, n2=1e-5)   # pure form, overrides

# (b) pot_vort to zarr every 10 model-hours
fr.io.Writer(fields=("u","v","w","b"),
             derived={"pv": model.diagnostics.pot_vort},
             trigger=fr.every(hours=10))

# (c) parameter-free vocabulary property
class State(fr.VectorField):
    @property
    def rel_vort_z(self):
        return nh.diagnostics.rel_vort_z(self)   # ∂x v − ∂y u only
# ekin is deliberately NOT here: it needs dsqr -> model.diagnostics.ekin()
```

## 7. Risks and open questions

1. **`epot`'s N²=0 branch**: the old silent formula switch dies —
   `epot` requires `n2` with a hinted error; the unstratified `b·z`
   form is user field algebra or a separate `epot_unstratified`.
2. **`cfl` and dt**: dt lives with D3's stepper; the bound
   `model.diagnostics.cfl()` needs read access to it — if dt is
   traced carry, cfl reads it from model_state like the clock.
3. **Metadata on results**: diagnostic functions must end with
   `with_metadata(...)` (strict-algebra default-metadata rule) —
   cheap; 02_rules entry owed.
4. **User-extensible namespace**: deferred — users call their own
   functions with `model.parameters`; registration sugar is not
   load-bearing.
5. **In-trace vs chunk-boundary evaluation** for cadences that
   don't divide the chunk — owned by 2.6; the seam supports both.
6. **D4 constructor alignment**: the `diagnostics=(...)` kwarg in
   the D4 sketch predates this decision — should become IO/writer
   config or be dropped.

Precedents: Oceananigans output writers (names → Field |
`func(model)`, computed on schedule) and Oceanostics.jl (a package
of functions consuming the assembled model, handed to writers) —
the shape validated end-to-end; MITgcm diagnostics registry as the
anti-pattern.
