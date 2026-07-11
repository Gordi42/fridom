---
status: frozen
date: 2026-07-07
---

# C3 — Coupling architecture pre-design

Research report (see [`README.md`](README.md) for status); feeds
[`../specs/model/09_coupling_designfor.md`](../specs/model/09_coupling_designfor.md).

## The recommended Phase-3 shape

**A `CoupledModel` facade layered on the driver pattern** — sugar
over `advance()` + `set_aux()`, owning N models + a declarative
exchange spec + the ops surface (ensemble progress/NaN-abort/
coordinated snapshots/walltime/report); normative test: facade run ≡
hand-written driver, bitwise (the run()-over-advance() discipline
carried up one level). **Coupler-as-module-owning-stages-in-both
models is rejected by construction** (contradicts one-carry-per-
model, the stage signature, and Model-not-a-pytree simultaneously);
the sanctioned per-model piece is the receiver-side Coupler module
(host_writable AUX + flux terms). **A mediator that is itself a
degenerate 2D fr.Model** (exchange-grid style) is *not precluded* —
it is D1.3's module-only-models property — needing one lint guard
(empty-PROGNOSTIC models legal).

**Concurrency, honestly**: two models on disjoint device_ids get
overlap from jax async dispatch **only under dispatch-then-sync
ordering** — with today's blocking `advance()`, the per-chunk
panic-flag read blocks the host thread before the second model
dispatches: `atm.advance(); ocn.advance()` **serializes entirely**.
The fix is small: an invariant ("the panic read is advance's only
host sync; the committed carry may hold pending arrays") + a
reserved non-blocking spelling. Multi-host: explicitly not promised
(process-divergent submesh programs are deadlock-fragile; host-side
Regrid breaks outright on non-addressable shards — that is the
traceable-Regrid precondition set).

**Regrid seam**: host-side (host-driven; still jax ops, so async)
for 3.2; traceable-inside-a-super-step is designed-for only, with
named preconditions (cross-mesh resharding over a super device
mesh; Regrid demands entering both negotiations pre-freeze; an
exchange seam in the traced schedule). D5 already leaves the door
open: signatures carry their own grid identity; a cross-grid Tier-1
Regrid transform is representable today.

**Iterated/Schwarz coupling**: expressible as D5 algebra over a
**product state space** — `FixedPoint(CoupledPropagator)` where the
propagator's domain/codomain is a `ProductSignature`; §10.1
forecloses nothing (signatures are opaque compared values; the
laws' call semantics generalize from reset-to-zero to
restore-to-checkpoint, which requires the in-memory carry
save/restore hook). Product norm = a 3.2 choice on the existing
`norm=` kwarg.

**Time axis**: one designated **master dt**, all others derived by
exact division (generalizing V-C11 beyond pairwise — pairwise
derivation dephases once a third clock exists); ramped coupling
strength is free by D2 (a Ramp in the coupler module's coefficient
slot); asynchronous per-direction windows are driver-expressible.

## The class-spec hook list

**Include now**: H1 no-hidden-sync invariant on `advance`; H2
reserve `advance(sync=False) -> PendingAdvance`; H3
`model.panicked` cheap property; H4 `RunResult` plain/aggregatable
with a shared status enum; H5 trigger lowering as a pure reusable
function; H6 `set_aux` accepts device-resident Fields (no host
round-trip); H7 master-clock wording; H8 in-memory carry *read*
(`model.carry` opaque snapshot value — the disk-free twin of
`snapshot()`); H9 empty-PROGNOSTIC models legal; H10 keep
StateTransform product-ready (no single-grid assumptions, no
isinstance-State in the base, endo-ness asserted only where
required); H11 snapshots never embed run-loop state.

**Leave open**: PendingAdvance full semantics; the CoupledModel
surface (exchange spec object, ensemble report, panic-propagation
policy); the Regrid operator spec; the Coupler-module template +
mediator preset; CoupledPropagator/ProductSignature + product norm;
the sanctioned carry *setter*.

**Explicitly not promised**: multi-host in 3.2; fused cross-model
super-steps; threads/MPMD; automatic conservative-flux machinery.

## Rejected alternatives

Cross-model stages (three resolved structures at once); one
super-model (grids/treedefs are per-model; D3 already rejected
nested mini-models and named coupling as the separate-model home);
bare-driver-only (loses the ops surface; users would hand-roll the
serializing dispatch order); facade-as-new-primitive (breaks the
bitwise test); blocking-only advance (provably serializes); per-model
host threads (async dispatch already overlaps; thread-safety not
granted).

## Deferred to 3.2

Conservation policy (finer-grid vs exchange grid; mediator in it-1
of coupling or later); parallel vs sequential/Gauss-Seidel window
schedule semantics; ensemble panic policy + coupled RunResult;
coupled manifest format; product-state norm; Regrid demand timing
vs frozen grids; whether the mediator needs a stepper; multi-host
program-divergence discipline + consensus (with 3.3).
