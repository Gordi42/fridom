# C2 — Concrete atmosphere–ocean walk (adversarial)

Research report (see [`README.md`](README.md) for status); feeds
[`../09_coupling_designfor.md`](../09_coupling_designfor.md).

Configuration: 2D shallowwater atmosphere (dt=60 s, AB3) over the
3D nonhydro ocean (dt=600 s), different horizontal resolutions
(real regrid), one process; window = 1 ocean step = 10 atm steps;
τ down (window-mean), SST up.

## Findings

| id | severity | finding |
|---|---|---|
| **F1** | **DECISION** | **Windowed accumulation is a seam gap**: an S6 DIAGNOSTIC stage can accumulate in-trace (read-own-previous + replace is legal per the §5.2 read rule) and the host reads the sum at chunk boundaries — but **no surface resets it**: `set_aux` is AUX-only by the signed rule; `reset()` nukes clock+warm-up. Chunk-boundary driver reads can't see interior steps; `advance(1)` loops kill chunking. |
| **F2** | DECISION | D2.3's blessed accumulator route (`self_update`) runs **per substage** → multi-counts at stage times under RK3; correct under AB3 only by accident. `cadence=STEP` is recorded-not-built. |
| **F3** | DOC | Surface-concentrated forcing has two expressible routes (owner-declared indicator-AUX via `default=f(coords)` — works today — vs the grid notes' §3.6 trace-field flux BC); no rule picks; the coupler-AUX space (Profile-broadcast vs trace space) is unpinned; `set_aux` on trace-space AUX unexercised. |
| **F4** | **DECISION** | `update_parameters` re-materialization **silently zeroes host-written AUX** (τ, SST — defaults are zeros; an AUX accumulator's partial sum unrecoverably). The re-materialization rule and the `set_aux` rule contradict on source of truth. |
| **F5** | DECISION | Coupled-driver ops: `advance()` has no typed status/panic contract; `RunResult`, walltime prediction, resubmit, progress are all locked inside `run()` — unreachable when interleaving. |
| **F6** | DOC | Coupled restart: per-model snapshots suffice (exchange fields + accumulators are carry leaves — half-exchanged data is snapshot-consistent) **iff snapshots happen at window boundaries**; missing: a machine-readable manifest header (clock/it without leaf IO) for the cross-model consistency assertion. |
| **F7** | OK-NOTED | Iterated/Schwarz coupling is **not** a D5 FixedPoint (single-grid signatures; rewind needs full carries — `set_state` is PROGNOSTIC-only). Needs public in-memory carry save/restore (exists privately for `debug_nan`); product-state transforms are Phase-3 design. |
| **F8** | OK-NOTED | Conservation achievable: accumulate-then-regrid-once (one regrid of the mean per window); demand a conservative `Regrid` variant + pinnable accumulator dtype (space `scalars` — implied by D1.1, never stated for this use). |
| **F9** | DOC | Window 0 runs on default-zero exchange fields unless the driver primes exchanges before the first advance — silently unforced. Protocol rule: exchange-then-advance. |
| **F10** | DOC | The SST path needs a host-side **surface-restriction (trace) operator** (3D→2D at the boundary) before regrid — on no operator list. |

## Resolutions (recommended)

- **F1+F2 bundle**: `host_writable=True` becomes lifecycle-polymorphic
  (AUX ∪ DIAGNOSTIC) and `set_aux` writes any consented
  non-PROGNOSTIC component; bless the **S6 accumulation idiom**
  normatively (read-own-previous + replace); **re-point D2.3's
  escape hatch from `self_update` to S6 DIAGNOSTIC stages**; reserve
  `cadence=STEP` on self_update; optional
  `fr.modules.WindowAccumulator` preset (host-reset or in-trace
  modulo-reset variants). Restart correctness free (carry leaf).
- **F4**: host-writable components are **exempt from
  re-materialization** (host write is the source of truth; the
  default is initialization-only).
- **F5**: `advance()` returns a status object + raises a typed
  `PanicError`; walltime predictor + progress renderer become
  free-standing utilities `run()` composes.
- **F3/F9/F10/F6**: pin Profile-broadcast + indicator-AUX for it-1
  (trace-space designed-for); coupler-protocol page
  (exchange-then-advance; window-boundary snapshots; manifest
  header; cross-model clock assertion); trace-restriction operator
  on the Phase-3 list.

## Constraint list for the class specs

(1) `host_writable` lifecycle-polymorphic; `set_aux` not hard-coded
to AUX. (2) DIAGNOSTIC write gate permits accumulation; S5-before-S6
guaranteed. (3) `self_update` reserves `cadence=`; per-substage
hazard documented. (4) re-materialization table carries the
host_writable flag (exemption implementable). (5) `advance()` typed
return + typed panic; `run()` implementable purely on top. (6) ops
utilities free-standing. (7) machine-readable snapshot-manifest
header. (8) room for public in-memory carry save/restore. (9)
`Regrid` admits `conservative=True` with the discrete-integral
contract. (10) surface/trace restriction operator listed; coupler
AUX space policy pinned. (11) accumulator dtype pinnable via space
scalars. (12) the coupler protocol doc page.
