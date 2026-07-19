---
status: frozen
date: 2026-07-19
---

# WENO momentum z-shard seam — the one-pass selected kernel under-traces its halo

The forced-4 hydrostatic parity test
`tests/hydrostatic/test_z_shard_parity.py::test_weno_z_shard_parity_buoyancy`
(8×8×16, `WENOAdvection(order=5)`, z sharded 4 planes/shard) leaves a
`~1e-5` seam in `u`/`v` at every z-shard boundary (z-planes 3|4, 7|8,
11|12) after 8 steps. `b` passes only because the test IC gives it a
0.01 amplitude; the seam is present in `b` too, amplitude-scaled below
the 1e-5 tolerance. This is a **silent wrong-physics** bug on a bounded
sharded axis, not an error — the runtime bounds guard does not fire.

## Mechanism

`WENOAdvection` builds its upwind face value with the one-pass
**selected-input** kernel `_SelectedFaceReconstruction`
(`model/modules/advection.py:1530`, the -39%/-46% optimization of
`research/stencil_lowering.md` §6): instead of reconstructing both
biases and `Where`-selecting the results, it slices one **union
window** of `order + 1` cells straddling each output face, per-tap
selects `where(v_face>0, U[i], U[order-i])`, and runs a *single* left
WENO reconstruction (`__call__`, lines 1722–1739:
`u_size = order + 1`, `m0 = biased_offset(order,"left") + shift`).

The halo negotiation never sees that union window. `_SelectedFaceReconstruction`
delegates its signature, halo-trace and codomain to the interned
**left** `_BiasedFaceReconstruction` (line 1608), and under the halo
trace `__call__` short-circuits to `left_op(q)` (lines 1717–1720). The
trace therefore records the *left* kernel's two-sided footprint
`footprint_reach(order, m0)` (advection.py:1466), **size = order**, not
the size = order+1 union:

| direction (order 5)          | m0 | left kernel `footprint_reach(5,m0)` | runtime union `footprint_reach(6,m0)` |
|------------------------------|----|--------------------------------------|----------------------------------------|
| primal `Center→face`, shift 0| 2  | **(2, 2)** → sym 2                    | **(2, 3)** → sym 3                      |
| dual `face→Center`,   shift 1| 3  | (3, 1) → sym 3                        | (3, 2) → sym 3                          |

The union's extra cell lands entirely on the side **opposite** the left
bias (the "above" side). That extra cell is exactly the right bias's
reach: the biased *pair* per-side max is `max(left(2,2),right(1,3)) =
(2,3)`, which equals the union window's reach. The one-pass kernel
reads the pair's window but declares only the left half of the pair.

Only the **primal** direction (`Center→face`, shift 0) is under-declared
(2 vs the needed 3). The **dual** direction (shift 1) already declares
sym 3, because the +1 shift pushes the left footprint to (3,1). So the
bug fires only where a *cell-centered* advected quantity is reconstructed
onto a face along the **sharded** axis with **no** face-staggered
quantity dual-reconstructed on that same axis — precisely the
hydrostatic vertical:

- **x, y** get halo 3: `u` is x-face / `v` is y-face, so their self-
  advection along their own axis is the *dual* reconstruction (shift 1,
  sym 3). Correctly provisioned.
- **z** gets halo 2: `u`, `v` (and `b`) are z-*centered*, so their
  vertical advection is the *primal* reconstruction (shift 0). The
  left-only trace records sym 2; the runtime reads sym 3. **One short.**

The runtime `apply_fv_staggered` bounds guard (reconstruct.py:359) does
not catch it: on the bounded z axis `reach_right = (n_out−n_in) + size−1
− m0 = (15−16) + 6−1−2 = 2 ≤ width 2`. The global codomain/domain
deficit (`Center→Inner` staggering, −1) masks the extra union cell — the
same `window_reach` vs `footprint_reach` distinction the requirements
were switched to `footprint_reach` for (`storage_halo_width.md` §1), but
the internal guard still uses the `window_reach` form. On a **periodic**
sharded axis the guard would compute `reach_right = 3 > 2` and *raise*,
so the silent failure is specific to a **bounded** sharded axis.

This is the same family as `halo_sharding_invariants.md` §2 (trace-only
wide advection stencils invisible to the registry floor). §2 made the
tracer record each application's reach; the gap here is that the
one-pass kernel traces only *one* of the two biased applications, so
the pair's binding reach never enters the trace.

## Evidence

All under `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4
FRIDOM_TEST_FORCED_DEVICES=4`.

- **Negotiated `{'x':3,'y':3,'z':2}`**, sharded `['z']`. z-demand table
  (traced per-application reach on z): `_BiasedFaceReconstruction z=(2,2)`
  is the widest; `Restriction z=(0,1)`, `FiniteDifference z=(0,1)/(1,0)`,
  `LinearInterp z=(0,1)`. Max traced z-reach = 2.
- **Seam** (many vs 1-device, 8 steps): `u` 9.14e-6, `v` 5.65e-6, all at
  z-planes {3,4},{7,8},{11,12} (shard seams), decaying away from the
  first; `b` 9.64e-8 at the same planes; `ps` 5.1e-9.
- **b is not genuinely tight.** b-seam / u-seam tracks b's IC amplitude:
  0.012 at `b_amp=0.01`, 0.76 at `b_amp=1.0` (b-seam 7.06e-6). b is
  TRACER+ADVECTED (stratification.py:111); its z-face value is the same
  under-provisioned primal weno reconstruction as u/v, merely scaled by
  amplitude below the test tolerance.
- **Scope.** `WENOAdvection` (one-pass selected) → z-halo 2;
  `UpwindAdvection` (both-then-select, traces *both* biased kernels →
  floor merges to sym 3) → z-halo 3. `nonhydro2` weno5 (prognostic w;
  w self-advects dual in z → sym 3) → z-halo 3 already, both bounded and
  periodic z: the bug is **latent** everywhere w is prognostic, exposed
  only in the hydrostatic model where w is diagnosed. `CenteredAdvection`
  → z-halo 1 (unaffected).
- **Fix probe** — declare the union footprint (`footprint_reach(order+1,
  m0)`) on the traced left kernel: z-halo 2→**3**; u-seam 9.14e-6→4.6e-16,
  v→1.1e-16, **b→5.2e-18** (b's seam vanishes with u/v — its own path is
  cured, confirming the same mechanism), ps→2.4e-15. z **stays sharded**
  (16/4=4-plane shards hold halo 3, needs 4).

## Defective sites

- `model/modules/advection.py:1717-1720` — `_SelectedFaceReconstruction.__call__`
  trace branch applies `left_op(q)` only. **This is the site the
  negotiation reads** (the module is trace-fed, not registered; patching
  `requirements` alone has no effect — verified). It must record the
  union / biased-pair reach.
- `model/modules/advection.py:1654-1669` — `_SelectedFaceReconstruction.requirements`
  delegates to the left kernel (reach (2,2) primal); dead for the trace
  but wrong on its face.
- `model/modules/advection.py:1560-1564` and 1688-1690 — the false
  premise in the docstring: "the union window's per-side reach equals
  the biased pair's, so … delegates … to it [the left kernel]". The
  union reach equals the *pair's* (2,3), but the left kernel alone is
  (2,2) — half the pair.
- Same latent defect in the FV arm: `_FVBiasedReconstruction.requirements`
  (advection.py:1966, `footprint_reach(order, m0)`), reached via
  `_SelectedFaceReconstruction(family="fv")`.

Note the task's premise: `WenoReconstruction.requirements` (weno.py:823)
does declare symmetric `halo = order//2+1 = 3`, but that is the
standalone FV average-family operator, **not** on the nodal
momentum/tracer path. The momentum path traces the nodal
`_BiasedFaceReconstruction` two-sided `reach` (2,2). "Declares 3" and
"negotiates 2" are two different operators.

## Fix options (priced)

1. **Trace the pair / union reach (recommended).** In the `__call__`
   trace branch, record the union footprint (or apply both `left_op` and
   `right_op`, mirroring `UpwindAdvection`) — and override
   `requirements` to match. Trace-only, zero runtime cost. Effect:
   z-halo 2→3 on any grid where a cell-centered weno-advected quantity
   is primal-reconstructed on the sharded axis with no dual there
   (hydrostatic weno with z sharded; nonhydro2 already at 3, no change;
   centered unaffected). Comm-volume: z-halo layers 2→3, +50% on the
   z-halo exchange for that axis only.
2. **Per-side asymmetric halo (2 below / 3 above).** The true reach is
   asymmetric (2,3). Storage is symmetric (`HaloSpec.symmetric()`
   collapses to max), so this is not supported and buys nothing over
   option 1 (sym 3 either way).
3. **Re-sync between applications.** N/A: the union window is a *single*
   application; invariant (1) forbids capping a single stencil's reach
   and there is no intermediate sync to insert.
4. **Also tighten the `apply_fv_staggered` guard** (reconstruct.py:359)
   to the `footprint_reach` form so a future bounded under-provisioning
   *raises* instead of going silent. Defensive, orthogonal to the fix;
   would have turned this into a loud error.

### De-shard question

The union reach (3) is a per-application floor — uncappable under the
`fc2a3b66` semantics (`_cap_for_sharding` respects the floor). On the
test's 16/4 = 4-plane shards, halo 3 fits (extent ≥ halo+1 = 4), so the
fix **does not de-shard the test grid** — it only corrects z-halo 2→3
and collapses the seam. A borderline grid at exactly 3 planes/shard
(e.g. n_z=12, P=4) **de-shards** under the fix (a 3-plane shard cannot
hold a 3-wide stencil; `_shardable_names` rejects it, falling back to a
single device if nothing else shards). That is the honest outcome — the
current behaviour shards it and computes wrong physics — but it is a
capability change for such grids the owner should weigh.

## Owner ruling (2026-07-19) — SHIPPED

Ruled in session: ship option 1 plus option 4. Landed as dev merge
`c8ba82b8` (`fix/weno-selected-union-reach`), all three parts: the
trace branch applies the biased *pair* (mirroring `UpwindAdvection`;
the pair max equals the union footprint exactly for every
order/shift/family since `m0_left = m0_right + 1`), `requirements`
declares the union footprint for both families, the
`apply_fv_staggered` guard moved to the per-side footprint form (a
synthetic under-provisioned bounded application now raises), and the
parity test asserts u/v/b/ps tight at atol 1e-11 with O(1) buoyancy
plus a negotiated z-halo >= 3 assertion. Verified forced-4 post-fix:
z-halo 2→3 (z stays sharded), seams to the FP floor (u 3.3e-16,
v 1.4e-16, b 1.1e-16), all other configs' negotiated widths
unchanged. Roadmap entry retired to
[`../roadmap/done.md`](../roadmap/done.md).
