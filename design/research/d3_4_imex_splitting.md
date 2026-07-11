---
status: frozen
date: 2026-07-07
---

# D3.4 — Split integrators: IMEX by term, Gauss-Seidel by variable

Research report (see [`README.md`](README.md) for status).

Context note: the old hydrostatic package is essentially a stub
(settings + a bare ModuleContainer — implicit mixing and the free
surface were never built), so this design is greenfield grounded in
MITgcm/ROMS/MOM6/Oceananigans practice. The old nonhydro projection
chain projects the *tendency* (see d3_3 for the resolution).

## 1. IMEX families vs the minimal implicit-term surface (feeds D3.1)

Unifying form (Dedalus's split): `∂t X = F(X,t) + L·X` — F = summed
EXPLICIT contributions, L = the IMPLICIT terms' linear operator.

| Family | `solve(·, dt·γ)` | Forward apply `L·X` | Explicit F history | State history | eval_params |
|---|---|---|---|---|---|
| AB1–4 | — | only if forced explicit | s−1 past F | — | 1/step |
| FB Euler / SBDF1 | γ=1 | — | current F | — | 1 |
| **CNAB2** | γ=½ | **yes — one `L·Xⁿ`/step** | Fⁿ, Fⁿ⁻¹ | — | 1 |
| **SBDF2/3** | γ=⅔ / 6⁄11 | no | Fⁿ…Fⁿ⁻ˢ⁺¹ | **Xⁿ…Xⁿ⁻ˢ⁺¹** | 1 |
| **IMEX-RK** (ARS-type) | one diagonal γ | optional (stage derivative recoverable post-solve) | per stage, within-step | — (self-starting) | **per stage, t+cᵢdt** |

Worked forms: CNAB2 `(1 − dt/2·L)Xⁿ⁺¹ = Xⁿ + dt(3/2 Fⁿ − 1/2 Fⁿ⁻¹)
+ dt/2·L·Xⁿ`; SBDF2 `(3/2 − dt·L)Xⁿ⁺¹ = 2Xⁿ − 1/2 Xⁿ⁻¹ +
dt(2Fⁿ − Fⁿ⁻¹)` (L only at the new level — solve-only by
construction). ARS(2,2,2) stage times c = (0, γ, 1) — exactly why D2
mandated eval_params at stage times; a Ramp forcing is correct under
IMEX-RK with zero extra machinery.

**The CN solve-only trick is unsound here**: recovering
`L·Xⁿ = (Xⁿ − r)/(dtγ)` from the previous solve fails because
constraint stages (projection, barotropic correction) overwrite
velocities *after* the solve — the recovered quantity is L of the
pre-projection state, an O(dt) inconsistency every step. (For
IMEX-RK, recording the stage derivative immediately post-solve,
before constraints, IS sound — an internal optimization.)

**Requirements fed to D3.1**:

1. **An implicit term IS a TendencyTerm** — its tendency() returns
   the forward apply `L·X` as an ordinary contribution (the author
   already knows how to compute it; CNAB gets `L·Xⁿ` with no special
   protocol; a term can be forced EXPLICIT and every explicit
   stepper handles it; the halo trace sees it through the same
   callable).
2. **`solve(fields, dt_gamma, params/ctx) -> fields`** — γ-agnostic
   (the scheme owns γ; dt_gamma traced so warm-up γ-switching and
   adaptive dt never retrace); linear in the rhs; operates on the
   declared advanced subset.
3. **Not required**: L as a matrix; γ-specific factorizations as API
   (in-trace Thomas solve is the honest baseline — κ may be Ramp-ed
   or spatial; internal caching keyed on statics is allowed);
   solve-history protocols.
4. **Lint: at most one implicit operator per prognostic component**
   (sequential solves are first-order Lie splitting that silently
   degrades SBDF2+; make the failure loud). *(Reconciled with d3_1:
   framework families merge by coefficient-summing — the lint
   applies to non-mergeable operators.)*
5. Terms declare `advances`; per-column-independent vs coupled-block
   is one designed-for flag; coupled blocks are atomic under
   by-variable splitting.

## 2. Multistep + IMEX buffer partitioning (feeds D3.2)

- **Explicit ring buffer**: length s−1 of the **summed** explicit
  contribution (per-term history is never needed — the partition is
  consumed at accumulation time). Same carry cost as the old
  dz_list.
- **Implicit side: no tendency history, ever** (SBDF has L only at
  the new level; CN's `L·Xⁿ` is computed fresh — buffering it across
  the projection boundary is exactly the staleness bug of §1).
- **State history**: SBDF2/3 need Xⁿ⁻¹(, Xⁿ⁻²) — a new buffer class;
  D3.2's ring machinery parameterized by (buffered: tendency|state,
  depth), instantiated per ADVANCE stage.
- **Ownership per ADVANCE stage**, not global — the barotropic inner
  integrator owns its own buffers over its own subset. All buffers
  traced carry → bitwise restart free.
- **Warm-up**: order-ramp within the family (SBDF1→2→3; semi-implicit
  Euler→CNAB2), selected by the carried counter; **γ changes across
  warm-up levels** (SBDF1 γ=1 → SBDF2 γ=2/3), so the switch selects
  (explicit weights, state weights, γ) tuples — reinforcing traced
  dt_gamma. IMEX-RK is self-starting (no warm-up/buffers) — an
  operational advantage noted for later.
- **Params × buffers**: buffered Fʲ carry their own step-time
  eval_params — Ramp-correct automatically. `update_parameters`
  between runs leaves s−1 slightly-stale buffered tendencies —
  document; option: re-ramp the counter on parameter update (D4
  lifecycle note).

## 3. Split by variable: stages, grounded in the split-explicit free surface

**Abstraction**: a step is an ordered list of stages; each declares
kind, `advances`/`overwrites` (named PROGNOSTIC subsets), and reads
the **latest** state (Gauss-Seidel — D1.5's most-recent-write rule
extended to PROGNOSTIC across ADVANCE stages). Assumed kind order:
`SELF_UPDATE → DIAGNOSE → TENDENCY → ADVANCE(term-integrator) →
ADVANCE(by-variable…) → CONSTRAINT` *(reconciled with d3_3's
vocabulary in the consolidated design)*. Stages after TENDENCY
receive the **StepContext carrying the per-treatment tendency sums**
— the second consumer of the contribution-dict partition, and how
the split-explicit stage gets its slow forcing.

**Grounding** (MOM6/ROMS/Oceananigans): one slow baroclinic step;
inside it N (static, CFL-derived) fast substeps of a 2D subsystem —
η and barotropic transports U, V — driven by fast terms refreshed
per substep + a **frozen slow forcing** `G_U = ∫(slow u-tendency)dz`
read once from the StepContext; the fast trajectory is
**time-filtered**; a **barotropic correction** overwrites the
depth-mean of the 3D velocities.

**Nested mini-model vs ordinary PROGNOSTIC fields — verdict:
ordinary fields, decisively.** The FreeSurface module declares
`eta, U, V` as PROGNOSTIC (2D — constant-along-z spaces) and owns
two stages: ADVANCE({eta,U,V}) containing a `lax.scan` over N
substeps (filter weights in the scan carry), and CONSTRAINT({u,v})
for the correction. Free consequences: η visible to IO/diagnostics
through the one state vector; restart via ordinary carry; halo
negotiation through the traced stage; **lint amendment required:
stage `advances` claims count as "advanced"** (else eta,U,V fail the
coverage lint). Nested mini-model rejected: duplicates assembly for
3 fields, hides η, nested-carry restart plumbing, breaks the one-
carry treedef discipline — BUT its *shape* survives as the rule **"a
stage body is an arbitrary pure function of (state, ctx) that writes
its declared subset"** (also what a Phase-3 Coupler stage needs).

**The `table.velocity()` ambiguity** (D1 residual) resolves: `U, V`
get **no Velocity role** — they are diagnostically-slaved
transports; no closure should friction them, no scheme should
transport them; `table.velocity()` keeps returning the baroclinic
trio. The `group=` qualifier stays the designed-for escape hatch.

## 4. The composed hydrostatic step

Modules: `hs.DynamicalCore` (u, v; DIAGNOSE stage writing
`p_hyd = ∫ᶻ b dz`), Coriolis, stratification (b + coupling),
advection, `VerticalMixing` (IMPLICIT, per-column tridiagonal over
{u,v,b,tracers}), `SplitExplicitFreeSurface` (eta,U,V + the two
stages), driven by CNAB2. Schedule — **no hand-written stepper**:

1. eval_params(tⁿ); SELF_UPDATEs.
2. DIAGNOSE: `p_hyd` written; the pressure-gradient term reads it
   under most-recent-write.
3. TENDENCY: explicit sums; `g∇η` is NOT among the slow terms — it
   is the FreeSurface module's fast term inside the subcycle (the
   double-counting guard; ROMS's consistency correction is 3.1
   numerics, not abstraction).
4. ADVANCE (CNAB2 over {u,v,b,…}): rhs from F-history +
   dt/2·L·Xⁿ (forward apply), then mix.solve(rhs, dt/2) →
   provisional u*, v*, b*.
5. ADVANCE (barotropic): slow forcing = depth-integral of the
   explicit sum from ctx (+ optionally the implicit increment — a
   constructor knob, not framework policy); lax.scan over N FB
   substeps; filtered ⟨η⟩,⟨U⟩,⟨V⟩ written.
6. CONSTRAINT (correction): `u ← u* + (⟨U⟩ − ∫u* dz)/H` — reads the
   latest u* (Gauss-Seidel).
7. Roll buffers, tick.

The **non-split alternative** falls out of the same surface: an
implicit free surface (MITgcm default) is `grad η` as an IMPLICIT
term whose solve is a 2D Helmholtz solve — no new abstraction.

**Where composition honestly ends** (escape hatch = custom
Stage/Integrator, never a Model subclass): (a) IMEX-RK ×
split-explicit has no production precedent — assembly error
(multistep outer drivers only; the state of the art's own
restriction); (b) which pieces enter the fast forcing is per-model
numerics — module constructor knob; (c) ROMS-style pipelined LF-AM3
interleaving is below stage granularity — a custom integrator
against the same term surface.

## 5. Projection placement

**Project-state** (d3_3 confirms with the equivalence proof and
Oceananigans verification): multistep — one CONSTRAINT per step;
IMEX-RK — per stage (projected stage velocities, standard
incremental fractional-step; the integrator records implicit stage
derivatives before constraints run). History buffers store
unprojected explicit tendencies.

## 6. Iteration-1 scope

Freeze the abstractions now (treatment tags, two-capability implicit
surface, stage kinds + advances claims, per-stage eval_params,
StepContext, per-stage buffers); ship incrementally:

- **Ship in 2.5**: AB1–4 + eps at parity (+ explicit RK if trivial);
  **one generic `IMEXMultistep` driver with CNAB2 and SBDF2
  coefficient sets** (CNAB2 exercises the forward apply, SBDF2 the
  state-history buffers and γ-switching warm-up); a reference
  vertical-diffusion consumer validated against an exact 1D decay
  solution and a stiff-κ column test. Deliberately more than "defer
  IMEX to 3.1": debug the surface on a toy before the hydrostatic
  port depends on it.
- **Design-freeze only**: IMEX-RK tableaux slots (ARS(2,2,2)/
  ARS(4,4,3) — build 2.7/3.1); the split-explicit stage (build at
  3.1; freeze now: stage-body-as-pure-function, advances-claims lint
  amendment, no-Velocity-role rule, ctx per-treatment sums,
  multistep-outer restriction); SBDF3; the `group=` qualifier.

## 7. Risks / open questions

1. In-trace tridiagonal cost vs factorization caching — term-internal
   optimization if profiling demands; no surface change.
2. One-implicit-per-field forecloses implicit mixing + implicit
   Coriolis on `u` simultaneously; fractional-step composition
   deferred with its order caveat recorded.
3. Stale buffers across `update_parameters` — document; optional
   counter re-ramp.
4. Split-explicit consistency details (double-counting guard, filter
   weights) — 3.1 numerics against ROMS/MOM6 references.
5. Diagnostics consumed by *fast* substeps must be refreshed inside
   the subcycle — stage-body responsibility, 3.1 note.
6. Adaptive dt not precluded (dt_gamma traced; adaptive substep
   count stays host-side static N).
7. **Coverage-lint amendment is load-bearing** — coordinate with
   D3.1/D3.3.

## 8. Sketches

```python
# CNAB2 + implicit vertical mixing, nonhydro
mix = fr.closures.VerticalMixing(kappa=1e-3, treatment=fr.IMPLICIT)
model = fr.Model(grid=grid,
    modules=(nh.DynamicalCore(pressure_solver=..., dsqr=...),
             nh.ConstantStratification(n2=2.5e-5),
             fr.closures.CenteredAdvection(), mix),
    time_stepper=fr.steppers.CNAB2(dt=30.0))
# step: eval_params(tⁿ) → TENDENCY: F = Σ explicit
#   → ADVANCE: rhs = Xⁿ + dt(3/2 Fⁿ − 1/2 Fⁿ⁻¹) + dt/2·mix.tendency(Xⁿ)
#              X* = mix.solve(rhs|{u,v,w,b}, dt/2, ctx)
#   → CONSTRAINT: Xⁿ⁺¹ = P X* (writes p) → roll F-buffer

# Hydrostatic step skeleton (3.1 target)
model = fr.Model(grid=grid,
    modules=(hs.DynamicalCore(...),            # u, v; DIAGNOSE: p_hyd = ∫ b dz
             fr.BetaPlaneCoriolis(...), hs.ConstantStratification(...),
             fr.closures.CenteredAdvection(),
             fr.closures.VerticalMixing(..., treatment=fr.IMPLICIT),
             hs.SplitExplicitFreeSurface(substeps=32,
                                         filter=fr.PowerLawFilter())),
    time_stepper=fr.steppers.CNAB2(dt=600.0))
# SELF_UPDATE → DIAGNOSE(p_hyd) → TENDENCY
#   → ADVANCE(CNAB2 {u,v,b}, implicit mixing solve)
#   → ADVANCE(barotropic subcycle {eta,U,V})
#   → CONSTRAINT(barotropic correction {u,v})
```
