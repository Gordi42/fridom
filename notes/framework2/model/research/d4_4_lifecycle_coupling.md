# D4.4 — Post-assembly lifecycle, sweep/multi-device/coupling proofing

Research report (see [`README.md`](README.md) for status).

## 0. Two-layer spec

Every lifecycle operation is specified as (i) a **pure carry
transformer** `op(assembly, carry, ...) -> carry` (never in-place,
never treedef-changing) and (ii) a **driver-method spelling**
(`model.set_fields(...)` swaps the held carry reference) — robust to
the D4.1 Model-status ruling either way; the mutating spelling
matches D1.1 and is recommended.

## 1. The mutation surface (lifecycle table)

| Phase | Allowed | Forbidden |
|---|---|---|
| Pre-assembly | everything (modules plain objects; grid pre-freeze ops) | — |
| Assembly | snapshots derived data; builds FieldTable/binding/**re-materialization** tables, schedule, dry run; merge→negotiate→freeze (or the frozen-grid verify path); carry born sharded | constructor handles go stale after |
| Post-assembly (host, chunk boundaries) | `set_fields` (PROGNOSTIC-only; incoming fields re-homed to `decomposition.sharding(space)`); `set_state` (PROGNOSTIC subset; AUX/DIAG in the input ignored with a debug log); `apply_constraints()` (one host-side CONSTRAINT pass — the initial-projection residual, opt-in); `update_parameters(...)` (§1.1); `reset()` (§1.2); `snapshot`/`load_snapshot`; reads (`state`, `parameters`, `module()`, `diagnostics.*`); **`advance(steps)`** (IO-free pure chunk primitive) and `run()` (sugar over it + IO) | attribute pokes (`ImmutableParameterError`); frozen-grid ops; anything treedef-changing (module add/remove/**enable/disable**, Ramp static-spec change, treatment flips) → re-assemble |
| In-run (traced) | only sanctioned dynamics: Ramp via resolve_at, owner self_update, stepper advance, gated stage writes | structurally impossible |

### 1.1 `update_parameters(updates, *, rewarm=True)`

1. Resolve dotted names through the assembly-frozen binding table
   (registry constants as typo-proof keys). **New ruling: the
   stepper joins the binding table as provider of
   `fr.params.TIME_STEP`** — backward legs flip the dt sign through
   the same hook (no private stepper pokes; `run_backward` stays
   dead), and dt sweeps use it too.
2. Validate structure preservation (asarray coercion; a Ramp may
   replace a Ramp of identical static spec; scalar↔Ramp is a
   treedef change → re-assemble). Same treedef → jit cache intact.
3. Write functionally (`replace` on the module in the carry).
4. **Re-materialize owner-derived AUX fields** — the model tracks no
   fine-grained dependencies; **the owner re-runs its declaration
   defaults**. Mechanism: `FieldDeclaration.default` additionally
   accepts an **unbound method** of the owner
   (`default=type(self).materialize` — the TendencyTerm.fn aliasing
   pattern), called with the *live* module; the closure may read
   **only the owner's own leaves** (02_rules entry; cross-module
   AUX derivation is self_update or re-assembly). Assembly retains
   these closures in a static **re-materialization table**
   (D1.1 amendment: AUX default closures are retained in the
   assembly record, not discarded). **Allocation and
   update_parameters share one code path** (the defaults rule
   becomes "evaluated with the owner's *current* leaves; same path
   both times") — re-materialization is correct by construction.
   Invalidation is per-owner, conservative (all AUX declarations of
   any module whose leaves changed).
5. Solver/operator precomputes need **no hook**: D2.1's "grid factor
   at bind, parameter factor in-step" already forbids baking
   parameter values — cross-referenced as load-bearing for D4.
6. Stale-buffer policy applies (§2). DIAGNOSTIC warm-starts keep
   old-parameter values for one substage (harmless; documented).
7. `update_parameters` does **not** clear the panic flag
   (`load_snapshot`/`reset`/`set_*` do) — changing ν after a NaN is
   not a resume path.

### 1.2 `reset()` — exactly the OptimalBalance need

`stepper_state ← stepper.init(...)` (re-warm; RK no-op);
`clock ← (start, 0, 0)` — **load-bearing: Ramps are functions of
clock.time, so the clock reset is what restarts a ramp leg**;
`panicked ← False`; DIAGNOSTIC components → declared defaults.
PROGNOSTIC and AUXILIARY **untouched** (state re-init is
`set_state`'s job; zeroing `f_coriolis` would be a bug). Testable
invariant: `reset(); set_state(z)` ≡ fresh assembly + `set_state(z)`
bitwise.

## 2. Stale-buffer ruling: auto re-ramp, opt-out

`update_parameters(..., rewarm=True)` resets the warm-up counter to
0 (buffers need no zeroing — the zero-padded warm-up rows never
weight old entries before they shift out; RK structurally no-op).
Why auto: the s−1 buffered tendencies embody *old physics* — an O(1)
discontinuity that AB extrapolation can amplify beyond the O(dt)
warm-up cost — and it buys the invariant
`update_parameters(p); run ≡ fresh-assembly(p) + set_state; run`.
`rewarm=False` serves knowingly-epsilon changes. Flipping the dt
sign **requires** the re-ramp (backward legs must not consume
forward history) — the default covers it.

## 3. The sweep workflow, verified — and the grid-reuse ruling

Shared jit cache requires: (1) **the same grid object** — fields
carry the grid as identity-hashed static aux, so a fresh grid per
sweep point is a different static → recompile (this *forces*
one-grid-many-models); (2) identical treedef (module types/order,
declarations, Ramp static specs, treatments); (3) same static
stepper config (dt is a leaf — dt sweeps share); (4) swept values
dynamic (guaranteed by D2's rule); (5) **a shared framework-level
jitted entry** `step_chunk(assembly_record, carry, n)` with the
assembly record hashable static — per-model closures silently defeat
the cache (implementation obligation + compilation-count regression
test).

**Grid-reuse answered from the lifecycle rules**: nothing forbids a
second Model *reading* a frozen grid — what is forbidden is
re-running merge/negotiate/freeze. **Ruling: one-grid-many-models is
the sanctioned sweep idiom, via a frozen-grid verify path**: at
freeze the grid records a negotiation fingerprint (state-space set,
merged override keys, HaloSpec, layout vocabulary); a subsequent
assembly on a frozen grid skips merge/negotiate/freeze and
**verifies** its demands against it — identical composition passes
by construction; different overrides / new spaces / larger halo /
new layouts → `GridFrozenError` ("assemble on a fresh grid, or
assemble the most demanding model first"). Small grid.md amendment
owed (fingerprint record + verify entry; merge call site = assembly
step 3, **first model on the grid**). Also document the cheaper
sweep: leaf-only changes need no re-assembly at all
(`update_parameters` + `reset` + `set_fields`).

## 4. The multi-device walk

Allocation happens **after** negotiation → every carry field is
**born sharded**; the `ReshardingReport` walk applies to
externally-produced leaves entering later (`set_fields`/`set_state`/
`load_snapshot` re-home per `decomposition.sharding(space)`) — the
precise landing spot of the grid lifecycle's "walks its state once"
sentence. Module code contains zero device awareness (patterns +
extra_halo only). D4 specifies exactly: allocation-after-negotiation,
the entry-point device_put walk, gather at the host/IO boundary —
everything else is decomposition-owned. **Restart across device
counts** requires snapshots to store true-shape (gathered,
layout-independent) arrays, re-homed on load — fed to d4_3. CI:
forced-host-devices lifecycle tests (AGENTS.md pattern) asserting
shardings, set_fields re-homing, sweep cache sharing at 4 devices,
snapshot portability 4→1 and 1→4, any-device-count result equality.

## 5. Coupling proofing (minimal hook set)

1. **`advance(steps)` is the public IO-free primitive; `run()` is
   sugar** — an outer coupler loop interleaves
   `a.advance(Na); exchange(); b.advance(Nb)`; since `step_chunk` is
   pure and jit-composable, 3.2 may later fuse without D4 changes.
   (Also dissolves OptimalBalance's old `disable_diagnostic`.)
2. The carry is a name-addressable pytree with functional component
   replace; exchanged data enters model B as an **AUX field owned by
   B's coupler module** (owner-mediated — the "AUX written only by
   its owner" rule preserved verbatim).
3. Stage bodies are arbitrary pure functions — already D3.
4. **Clocks stay per-model**; dt ratios = per-model chunk lengths
   per exchange window (`Na·dt_a == Nb·dt_b`), exact integers under
   steps-primary runs — no float-drift coupling hazard.
5. No process-global mutable state in any lifecycle op (audited).
6. Regridding: cross-grid arithmetic already raises
   `GridMismatchError`; the Phase-3 `Regrid` operator slots in as a
   grid-pair-bound operator.

## 6. The OptimalBalance walk — composes with no remaining gaps

Forward model with `rossby_number=fr.Ramp(t=(0,T), v=(0,Ro))`;
backward model on the **same grid** with `dt=-dt` (leaf) and the
Ramp over `t∈(−T,0)`, `v=(0,Ro)` (value Ro at leg start, 0 at −T) —
or **one** model serving both legs via `update_parameters`
(TIME_STEP sign + Ramp endpoints). Per leg:
`reset()` (re-warm + clock reset = Ramp restart) → `set_state(z)` →
`advance(N)` → read state. Base points:
`em = nh.eigenmodes.from_model(model)` — needs f0/n2/dsqr, **not**
Ro, so the Ramp-valued scaling never triggers `at_time=`;
`P = em.projector("geostrophic")` host-side between legs. The three
hooks this walk *added*: TIME_STEP in the binding table; the IO-free
`advance`; `reset()`'s clock reset being normative. Backward
`stage_dt` sign stays the parked 2.7 test item.

## 7. Risks / open questions

- D4.1 coordination: the two-layer spec is the hedge on Model
  status (resolved: host driver — the mutating spellings stand).
- **D1.1 amendment owed**: AUX default closures retained; extended
  `default=` forms; the rewritten defaults rule (same path, owner
  leaves only).
- Shared-runner discipline: an assembly lint for unhashable statics
  + a compilation-count regression test.
- Grid fingerprint/verify path: new grid-side surface (grid.md
  amendment).
- d4_3 coordination: snapshots true-shape/layout-independent; the
  fingerprint covers Ramp specs + treatments (already in 02_rules).
- Ramp-fed AUX allocation values are placeholders (self_update owns
  them in-run) — document.

## 8. Sketches

**Sweep idioms**: (A) re-assemble per point on the same frozen grid
(verify path; identical treedef → no recompile); (B) one assembly +
`update_parameters({fr.params.CORIOLIS_F0: f0}); reset();
set_fields(...)` per point (cheapest).

**Re-materialization (owner side)**:

```python
@fr.utils.jaxify
class FPlaneCoriolis(fr.Module):
    f0: jax.Array                              # dynamic leaf, source of truth
    def field_declarations(self):
        return (fr.FieldDeclaration("f_coriolis", space=fr.Profile(),
            lifecycle=fr.Lifecycle.AUXILIARY,
            default=type(self).materialize),)   # UNBOUND — re-run with the live module
    def materialize(self, grid, space):         # owner leaves only
        return grid.create_field(space, data=jnp.asarray(self.f0))
```

**OptimalBalance (host driver over assembled models)**:

```python
def optimal_balance(z, *, grid, ramp_period, dt, Ro, max_it=3, tol=1e-9):
    T, N = ramp_period, int(ramp_period / dt)
    fwd = nh.Model(grid=grid, core=nh.DynamicalCore(
              rossby_number=fr.Ramp("exp", t=(0., T), v=(0., Ro))), ...,
          time_stepper=fr.time_steppers.AB3(dt=dt))
    bwd = nh.Model(grid=grid, core=nh.DynamicalCore(
              rossby_number=fr.Ramp("exp", t=(-T, 0.), v=(0., Ro))), ...,
          time_stepper=fr.time_steppers.AB3(dt=-dt))    # leaf: no recompile
    P = nh.eigenmodes.from_model(fwd).projector("geostrophic")
    z_base, z_res, prev = P(z), z, np.inf
    for it in range(max_it):
        bwd.reset(); bwd.set_state(z_res); bwd.advance(steps=N)
        z_lin = P(bwd.state)
        fwd.reset(); fwd.set_state(z_lin); fwd.advance(steps=N)
        z_new = fwd.state - P(fwd.state) + z_base
        err = (z_new - z_res).norm(); z_res = z_new
        if err < tol or err > prev: break
        prev = err; z_base = P(z_res)
    return z_res
```
