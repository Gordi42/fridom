---
status: frozen
date: 2026-08-12
---

# Wide stencils on thin axes — halo fill, flat-axis elision, options

Research report (see [`README.md`](README.md) for status). Question:
why does high-order advection (WENO5, upwind5) fail on a grid with a
single cell along one direction, and what are the options — repair the
halo fill, or treat a flat axis as halo-free? Method: root-cause read
of the fill machinery, an exhaustive index-equivalence check, a
monkeypatched end-to-end verification (gradient and bitwise
replication), a measured blast-radius sweep over every new-stack
component, and a storage/step-time study. Three parallel
investigations, cross-checked; scripts were throwaway, every
load-bearing number is inlined here.

## Verdict

**The restriction is a spelling limitation of the periodic wrap, not a
numerical or structural one. It is ~14 lines in one file to lift, and
the result is bitwise correct and exactly differentiable. Flat-axis
elision is a separable performance question that cannot serve as the
correctness fix.**

- The bug is **halo wider than the axis**, not "size-1 axis". WENO5
  (halo 3) fails at `n = 1` *and* `n = 2`, and works from `n = 3`. Any
  rule keyed on `n == 1` leaves `n = 2` broken.
- The correct periodic fill is modular indexing. It is **bitwise
  identical** to the shipped fast path wherever that path is defined,
  and extends to arbitrary width.
- A **closed** flat axis is degenerate at the function-space level and
  is out of scope: the wall-normal velocity space is empty. Periodic
  is the "2-D direction" idiom.
- Elision buys an exact 3x/5x/7x on stored field traffic but measured
  only 5–13% of step time on cpu at 128². Its gpu payoff is unmeasured
  and is the whole case for doing it.

## 1. Root cause

`_axis_map`
([`tensor.py`](../../src/fridom/spatial/decomposition/tensor.py)
~1584–1649), periodic branch:

```python
if factor.mesh.periodic:
    if width > n or trail > n:
        raise NotImplementedError(
            f"periodic wrap with halo {width} wider than the "
            f"axis length {n} is not supported")
    src[:width] = np.arange(n, n + width)
    src[width + n:] = np.arange(width, width + trail)
```

The storage frame on one shard is `[width ghosts | n true | trail
ghosts]` with `trail == width` (`_axis_storage`, ~430). The leading
ghosts read source range `[n, n + width)`, which lies inside the true
region `[width, width + n)` **iff `width <= n`**. Outside that regime
the arithmetic would silently source *ghost* slots — for `n = 1,
width = 3`, slots 1 and 2 are ghosts and only slot 3 is the DOF. The
guard is honest; it refuses rather than iterating. One wrap is not
enough, and the fill is spelled as exactly one wrap.

A twin guard sits in `_write_axis` (~1788–1798), the `materialize=True`
spelling used at the scan carry
([`model.py`](../../src/fridom/model/model.py) ~548). A model *run*
hits both; a fix must touch both.

Call chain: `assembly.assemble` -> `composer.dry_run` ->
`WENOAdvection` -> `base.py` `_ensure_valid` (~1663) -> `movement.py`
`Sync` (~209) -> `grid.py` `sync` (~1008) -> `tensor.py` `sync`
(~996) -> `_fill_axis` -> `_axis_map`.

### Provenance

The guards date from `063e25c6` (2026-07-07, wave 2C), present in the
very first `_fill_axis`; the commit message never mentions a
width/length restriction. `49b19f1e` (2026-07-19) duplicated the
periodic guard into the second spelling.

This repo has a legible convention for deliberate deferrals: the raise
names the design doc. Three sibling raises **in the same file** do
exactly that (~986 points at `decomposition.md` 104–127; ~1508 and
~1527 point at `boundary_plan.md`). The two length-vs-halo guards
carry **no doc pointer, no design entry, no roadmap bullet, and no
test**. The governing spec
([`decomposition.md`](../specs/grid/classes/decomposition.md), "Halo
and storage contract (normative)") specifies fill semantics and states
no assumption about axis length versus halo width. Every length
constraint on record is scoped to **shards**, not axes
(`min_local_size`; `multigrid_pathway_plan.md` 265–266, "≥ negotiated
halo + 1 *when the axis is sharded*"), and `_cap_for_sharding`
explicitly exempts never-shardable axes — so an unsharded 1-cell axis
is unguarded by negotiation and walks into the raise.

**These are defensive asserts from wave 2C, not a ratified deferral.**
Lifting them overturns no recorded decision.

## 2. Blast radius (measured, not read)

Negotiated `halo.widths` were measured after assembly for every
constructible component. **Only the biased/WENO advection family
requests more than 1.**

| component | halo | breaks on periodic axis | breaks on walled axis |
|---|---|---|---|
| `CenteredAdvection` | 1 | never | never |
| `UpwindAdvection(3)` / `WENOAdvection(3)` | **2** | `n <= 1` | `n_cells <= 3` |
| `UpwindAdvection(5)` / `WENOAdvection(5)` | **3** | `n <= 2` | `n_cells <= 5` |
| `HarmonicFriction` / `HarmonicDiffusion` | 1 | never | never |
| `BiharmonicFriction` / `BiharmonicDiffusion` | **1** | never | never |
| `SmagorinskyLilly`, `ConstantStratification`, Coriolis | 1 | never | never |
| spectral pressure solve | 1 total | never | never |
| `cumulative`, `integrate`, `extremum`, transposes | 0 | never | never |
| `restrict` (multigrid) | 1 | never | never |
| immersed masking | 0 | never | never |
| `MeshVelocityCorrection` (ALE) | **2** | `n <= 1` | `n_cells <= 1` *(inferred)* |
| `FiniteDifference(order>=4)` (registry override) | `order//2` | `n <= order//2 - 1` | *(inferred)* |

**The biharmonic result is the surprise and it narrows the problem.**
Expected halo 2, measured 1: the staggered `diff`/`interpolate` pairs
telescope and bounded applied axes reset the traced depth. So
`dancing_eddies` runs `BiharmonicFriction` on its flat z today without
complaint, and the flat-axis problem is essentially **a single-feature
problem — high-order advection** — plus one unexercised ALE corner.

### What works today

`shape=(16,16,nz)`, x/y periodic, construction only:

| scheme | halo | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|---|
| centered | 1 | ok | ok | ok | ok |
| upwind3 / weno3 | 2 | **fail** | ok | ok | ok |
| upwind5 / weno5 | 3 | **fail** | **fail** | ok | ok |

The demand is applied to **every** advected axis including the flat
one ([`advection.py`](../../src/fridom/model/modules/advection.py)
~4065–4072 merges `need` over all of `self._axis_velocity` with no
per-axis length awareness). Nothing narrows the flat axis's demand.

## 3. The fix — modular indexing

```python
src[i] = width + ((i - width) % n)
```

- **Equivalence verified exhaustively.** For all `n <= 11` and every
  `width, trail <= n`: **zero mismatches** against the shipped two-slice
  result. Proof: for `i < width`, `i - width` lies in `[-n, 0)` so the
  modulo gives `i - width + n`, i.e. `n + i` — exactly the first slice.
  For the trailing side, `(n + j) % n = j` for `j < trail <= n`, giving
  `width + j` — exactly the second.
- **`_fill_axis` needs no change.** It consumes `src` as a gather
  (`jnp.take(arr, src, axis=axis, mode="clip")`), not as slice
  assignments. `neg`/`zero` are applied as `jnp.where`. The `mode="clip"`
  clamp still folds away.
- **Cost is nil.** `_axis_map` is host-side numpy at trace time; both
  spellings emit the same `jnp.take` over a same-length constant, with
  *identical constant values* in the `width <= n` regime.
- **`_write_axis` must keep its fast path** as a static branch: its
  slice + `dynamic_update_slice` spelling is a documented perf contract.
  Only the wide branch is new — gather the O(halo) slab with the same
  modular index, then write in place exactly as before.
- Add an `n < 1` guard: numpy's `% 0` warns and yields 0 rather than
  raising.

### Verification under the patch (monkeypatched; no source edited)

- `(16,16,1)` + `WENOAdvection(order=5)` + AB3 assembles with halo 3 in
  every direction, z storage **7** — as predicted. Runs.
- Green across `(16,16,1)` orders 3 and 5, `periodic=(False,True,True)`,
  `(16,1,16)`, `(1,16,16)`, `(16,16,2)`.
- **Bitwise replication.** `(8,8,1)` vs `(8,8,4)` with a z-replicated
  IC, 4 AB3 steps: `|nz1 - nz4[:,:,0]| = 0.000e+00` for u, v, w, p, and
  the deep run stays exactly z-uniform. Independently reproduced at
  `(16,16,·)` over 20 steps.
  *Refinement from the shipped test:* with coriolis, stratification and
  the CG pressure projection all active, u/v/p still match **bitwise**
  and the deep run is still exactly z-uniform, but `w` lands at
  ~1e-20 rather than identically 0 — the projection is not bit-exact.
  The regression test asserts bitwise on the horizontal state and
  machine-zero on `w` (`tests/nonhydro2/test_thin_axis.py`).
- **Differentiability** (per the AGENTS.md policy): `jax.grad` of a
  sum-of-squares loss through `Model.propagator(wrt=("u",), steps=4)` on
  `(8,8,1)`: finite, **0 NaNs**, directional derivative
  `0.091789539599` vs central FD `0.091789539597` — ~2e-11 relative.

### WENO on a constant column is safe

The concern was a masked singularity — finite forward value, singular
VJP. It does not arise.
[`weno.py`](../../src/fridom/spatial/operators/weno.py) ~445 is
`alpha_m = d_m / (beta + eps)**2`: the epsilon is added **before**
squaring, so the denominator is `>= 1e-20` and the normalization at
~485 is **never 0/0**. No `sqrt`, nothing needing a double-`where`.
On a z-constant window `beta = 0` exactly for 5 of 6 rows and `1.2e-32`
for row `(3,-4,1)` — 22 orders below `WENO_EPS = 1e-10`. Normalized
weights come out at exactly the optimal `d_m = (0.1, 0.6, 0.3)`; output
equals the constant to 1 ulp at order 5, and since all windows are
bitwise identical the flux difference is **exactly** zero.

The VJP factor `d alpha/d beta = -2 d_m/(beta+eps)^3 ~ -2e30 d_m` is
large but finite, and multiplies `d beta/d cells = 0` exactly. Measured:
reverse-mode grad equals the linear-scheme gradient, forward-mode `jvp`
is `[1,1,1]`, `jax.hessian` is finite. No `custom_vjp` anywhere, so
forward mode stays available.

*Caveat (pre-existing, not thin-axis-specific):* under
`jax_enable_x64=False` the `2/eps^3 ~ 2e30` factor leaves only ~8 orders
of f32 headroom. fridom enables x64 at import, so this is a note.

## 4. Multi-device — no interaction

**A size-1 axis is never sharded, at any device count.** Two
independent gates: `_shardable_names`
([`decomposition.py`](../../src/fridom/spatial/decomposition/decomposition.py)
~1060–1084) computes `last = 1 - (P-1) < 1` for `P >= 2` and skips the
axis, so it never enters a `Layout`; and `_cells_per_shard`
(`tensor.py` ~353) rejects it independently.

Verified on 4 forced host devices, `(32,32,1)` WENO5: layouts are x and
y pencils plus the replicated one, z never appears; `u` storage is
`(60, 38, 7)` sharded `P('devices', None, None)`; 2 steps run finite.
The thin axis always takes the `shards == 1` static branch and
`_exchange_block` never sees it. **The fix touches only the
single-shard path.**

## 5. The closed flat axis is degenerate — and stays refused

`mesh.n_cells == 1` does **not** imply one DOF per space. Measured:

| space | periodic | bounded |
|---|---|---|
| `Center` / `Left` / `Right` / `CellAvg` | (1,) | (1,) |
| `Outer` | n/a | **(2,)** |
| `Inner` | n/a | **(0,)** |
| `FaceAvg` | (1,) | **(0,)** |

- **Periodic flat: the premise holds exactly.** Every space family has
  1 DOF, the wrap makes every neighbour identical, interpolation is
  identity and every derivative is identically 0 — confirmed by the
  bitwise flat-vs-deep run above. `nonhydro2`'s staggered `w` lives on
  `Right(z)` with 1 DOF; no exception.
- **Bounded flat: "identity/zero" would be silently WRONG.** `Outer`
  has 2 DOFs; `Outer(2) -> Center(1)` differencing of `[1, 3]` returns
  **2**, not 0. A rule keyed on `n_cells == 1` would destroy a genuine
  vertical flux divergence in a 1-cell column. Separately, a bounded
  `CellAvg(1)` with a Dirichlet tag gets ghost `= -u`, so its two-point
  derivative is `2u/dz != 0`.
- Bounded flat is already consistent by degeneracy for the physics that
  matters: `w` lands on `Inner(z)` with **0 DOFs**, which is right (no
  normal flow at both walls leaves no interior face). And bounded flat
  plus a wide stencil is already refused with a *taught* error —
  `_check_walled_extent` (`advection.py` ~4073) via `graded.min_cells`
  (~395): *"needs at least 4 cells on every walled axis (the graded
  near-wall ladder must fit between the two walls)"*.

**So the safe predicate for any flat rule is
`factor.mesh.periodic and factor.mesh.n_cells == 1` — never
`n_cells == 1` alone, and never `shape[0] == 1` alone** (a bounded
`Center(1)` passes the shape test and fails the semantics).

The two-wall reflection fold was worked out for completeness (infinite
dihedral group; agrees with the shipped `_ghost_slot` table on all 391
accepted configurations and extends to the 121 it refuses), but it
buys nothing: the bounded thin axis fails for the DOF-count reason
above, which no fill rule repairs. **Do not generalize the bounded
fill.**

Incidental hazard: on a walled `nz=1` assembly, `Inner(z)` arrives at
`_axis_map` with **n = 0** DOFs and survives only because it is
`BC.NONE`; `Inner(z, bc=DIRICHLET)` also arrives at n = 0 and survives
only via the `sign == 0` vacant-slot rule. Zero-DOF spaces already flow
through the fill undefended.

## 6. Flat-axis elision — the performance option

### Where halo is negotiated

Per-axis and two-sided throughout. `OperatorRequirements`
([`base.py`](../../src/fridom/spatial/operators/base.py) ~77–124) is
per applied factor, carrying `reach=(below, above)`. Chains sum per
side (~1731); sums take the per-side max (~1075). The currency is
`HaloSpec` ([`halo.py`](../../src/fridom/spatial/decomposition/halo.py)
~59–388), keyed by coordinate name. `trace_halo` collapses to
symmetric at ~1298. `negotiate` -> `_negotiated_halo`
(`decomposition.py` ~611–816) = registry max `merge_max` trace
`merge_max` explicit. The storage choke point is
`TensorDecomposition._width` (`tensor.py` ~287–304), which **already
has structural per-axis zero overrides** for `CoefficientSpace` and
`factor.collapses_axis`.

### Two clamps, one of them dangerous

- **Clamp the negotiated `HaloSpec`:** `_width` -> 0 and `sync` skips
  the axis structurally. But it breaks the stencil guards *loudly*
  (`apply_staggered` ~808 and `apply_fv_staggered` ~348 read
  `decomposition.halo[axis]` and raise), churns `halo_valid` (the field
  claims `(0,0)`, so `covers` never satisfies and `_ensure_valid`
  rebuilds a `Sync` node at every application), and hard-fails
  `require_solver_halo`
  ([`halo_demand.py`](../../src/fridom/model/halo_demand.py) ~146–196),
  which asserts every solved axis carries `halo >= 1` because the
  immersed/mapped pressure cores use hand-rolled `jnp.roll` diagonals.
- **Clamp only `_width`: silent wrong answers.** The six direct readers
  of `decomposition.halo[axis]` would still see the negotiated width, so
  the guards pass while the storage axis is 1 long. Traced through
  `apply_fv_staggered`: the kernel output has length 0, the slice is
  empty, and `jnp.pad` fills the result with **exact zeros**. No error.
  **Do not do this.**

The existing width-0 overrides in `_width` are safe **only because
operators never run on those axes** (`resolve_codomain` and
`SeparableOperator._apply` both short-circuit `ConstantSpace` first).
So **elision and operator short-circuit are one change, not two** —
that is the real cost driver.

### Where a short-circuit would go

Every halo>0 stencil operator routes through one of two shared tails:
`staggering.apply_staggered` (~743–847; `FiniteDifference`,
`LinearInterp`, `Restriction`) and `reconstruct.apply_fv_staggered`
(~270–400; `LinearReconstruction`, `LinearDeconvolution`,
`WenoReconstruction`, the `FluxDifference` family, `UpwindOne`,
`Fallback`, the graded rungs, the three private advection
reconstructions). Everything else is halo-0 and needs no flat
treatment.

The best spelling is **repeat-and-run**: on a flat axis, `jnp.repeat`
the single storage slot to a `size`-long window, run the operator's own
kernel unchanged, take slot 0. Because the input is then constant,
every *consistent* scheme automatically yields identity (coefficients
sum to 1) or zero (coefficients sum to 0), linear or nonlinear, with no
per-operator declaration and no silent-wrong failure mode. It keeps
`halo_valid` honest and confines the `1 + 2H` expansion to the kernel
while the stored field stays 1 slot.

The alternative — a `flat_action: ClassVar[Literal["retag","zero"]]` on
`SeparableOperator` — is ~12 one-line declarations, but each is a new
correctness invariant with a silent-wrong failure mode if mis-declared,
inherited by every future stencil operator. Rejected.

### Measured payoff

Storage inflation is exact and is a real FLOP multiplier, not just
memory: same-space field arithmetic runs on the **storage-shaped**
`_data`, not the true-shape view
([`scalar_field.py`](../../src/fridom/spatial/fields/scalar_field.py)
~1375). `(128,128,nz)` periodic, `u`:

| config | true | storage | inflation |
|---|---|---|---|
| centered (halo 1), nz=1 | (128,128,1) | (130,130,**3**) | 3.09x |
| WENO-3 (halo 2), nz=1 | (128,128,1) | (132,132,**5**) | 5.32x |
| WENO-5 (halo 3), nz=1 | — | z = **7** | 7x |
| WENO-3, nz=4 | (128,128,4) | (132,132,8) | 2.13x |

Step time (cpu, single device, 128², AB3, best of 3): the flat run
costs 68–72% of a 16-level run while resolving 6% of its DOFs — but
most of that is *fixed* per-step cost. Fitting `t = a + b·N_stored` on
the nz=4/nz=16 WENO points gives `a = 103.3 ms`, `b = 0.222 us` per
stored element, and predicts nz=1 **out of sample** at 122.7 ms vs
122.9 measured (0.2%). Extrapolating to a halo-0 z: **107.2 ms, a 12.8%
saving.** The centered+biharmonic pair gives ~5%.

A synthetic bandwidth-bound proxy at 514² with z-extent 1/3/5/7 gives
**1.00x / 4.85x / 11.80x / 15.33x** — *worse* than proportional,
because a flat axis with ghosts makes the fastest-varying storage axis
3 long, a pathological SIMD vector length, whereas at width 0 XLA
collapses it and vectorizes along y. The project's own perf notes
target exactly this regime, so *(inferred)* the gpu payoff for a large
flat-axis run is far closer to the 3–5x field-traffic figure than to
the 5–13% measured on cpu. **Unverified — no gpu on the investigation
box, and agents do not submit gpu jobs.**

The merely-small axis, which no flat rule covers: n=2/halo 2 -> 3.0x,
n=4/halo 2 -> 2.0x, n=4/halo 3 -> 2.5x, n=8/halo 3 -> 1.75x. All legal
today, none helped.

## 7. Prior art

The current old stack does **not** support this either, and fails
harder: `domain_decomposition.py` ~97–102 raises `ValueError: Local
shape ... is smaller than halo ...` at construction, uniformly over all
dims — which rejects even the biharmonic closures on a flat axis, where
the new stack copes. (`examples/nonhydro/convection_and_closures.py`
builds `shape=(512,1,512)` with biharmonic closures and raises today on
the old stack.)

**But the real prior art is a deleted fallback.** Before `be682e5f`
(2026-03-21, "use concatenate instead of `.at[].set` for halo
exchanges"), `_sync_periodic_axis` was:

```python
if self.shape[axis] < self.halo:
    return pad(arr[ics], pad_width, mode='wrap')
else:
    ...  # sliced exchange
```

The MPI variant carried the same fallback and additionally forced
`n_procs[i] = 1` whenever `n_global[i] == 1`, so a flat axis was never
split. **It was removed for spelling uniformity, not because it was
broken** — `np.pad(x, (6,6), mode='wrap')` tiles correctly when the pad
width exceeds the axis length. A working implementation of the fix
shipped once and was lost to a refactor.

The old stack also has a `flat_axes`/`topo` concept
(`domain_decomposition.py` ~486–515, sync skip at ~183–216) that skips
padding and exchange entirely — structurally exactly the elision
option — but it is *field topology* (a field that does not extend along
an axis), and never keys off `shape[i] == 1`. The new stack's
`TraceSpace`/`ConstantSpace` are the same idea.

## 8. In-tree users

Nine sites with a size-1 axis; **exactly one runs advection on it**,
and it carries the workaround in reader-facing prose —
`examples/nonhydro/dancing_eddies.py`, `shape=(nx,ny,1)`, periodic z,
`CenteredAdvection`: *"A thin vertical rules out the wide
reconstruction stencils, so the advection here is the centered
scheme."* Its flat axis is z and z **is** periodic, so it is in scope
for the fix.

The rest (`wave_package`, `internal_wave_maker`,
`multiple_wave_makers`, the `nonhydro2` initial-condition fixtures) are
flat-y with `advection=False` — linear physics, not a workaround.
Benchmarks have no flat grids. `shallowwater2` is *genuinely* 2-D
(`fr.spatial.Grid((mx, my))`), not a 3-axis grid with a flat third
axis; the nonhydro "2-D" setups are the latter, in both stacks.

**Nothing in the tree runs WENO or upwind on a flat axis** — because it
cannot.

## 9. Recommendation

1. **Ship the correctness fix alone.** *(Shipped — see below.)* Modular
   form in `_axis_map`, tiled wide branch in `_write_axis` behind the
   preserved fast path.
2. **Do not couple elision to it.** Different change, different risk:
   a new predicate, a clamp, a flat branch in both shared tails, an
   exemption in `require_solver_halo`, and a `halo_valid` decision — with
   a silent-wrong failure mode and a semantics trap (bounded `Outer`)
   waiting for a contributor who reads the flat rule and not this
   report. **2–4 days plus a per-family test matrix, for 5–13% on cpu.**
3. **If elision is done,** use repeat-and-run in the two shared tails,
   gated on `factor.mesh.periodic and factor.mesh.n_cells == 1`, with
   `_width` returning 0 for such factors and `decomposition.halo` left
   untouched.
4. **Measure on gpu before deciding.** The case rests entirely on
   whether the step is bandwidth-bound; it is not on cpu at 128²–512²,
   and the proxy says it very much is in the regime the project's perf
   work targets. One `(512,512,1)` WENO run on an A100 settles whether
   this is a 5% cleanup or a 4x win. Owner's call to submit.
5. **Possibly worth more than either: the asymmetric halo.** Two-sided
   reaches are already computed and threaded through the entire trace,
   then discarded in one line (`halo.py` ~1298). WENO-5's true reach is
   (2,3) but storage is 3+3, so every grid pays an extra plane on every
   axis — not just the flat ones.

## 10. What shipped

`fix/thin-axis-halo-wrap`, 2026-08-12 — recommendation 1 only.

- `_axis_map`: the two slice copies become
  `src = width + (arange(size) - width) % n`, plus an `n < 1`
  `ValueError` (numpy's `% 0` warns and yields 0 rather than raising).
- `_write_axis`: a `wide = width > n or trail > n` predicate selects
  between the existing slices and a new `_tiled` helper; the `if
  width:` / `if trail:` structure, the DUS writes and the side
  sequencing are unchanged, so the perf contract holds (the
  `temp_size_in_bytes` guard in the mirrored test still passes).
- Tests: the two negative tests
  (`test_periodic_wrap_wider_than_the_axis_raises`, the wide-wrap half
  of `test_materialized_sync_mirrors_the_map_edge_cases`) became
  positive tiling assertions; added a fast-path equivalence guard over
  `n` in 2..6 with `width <= n`, and the `n < 1` raise — each
  parametrized over both spellings. New
  `tests/nonhydro2/test_thin_axis.py` covers orders 3/5 on `nz` 1/2,
  the z-replication invariant, `w` at machine zero, and the
  `jax.grad`-vs-FD regression.
- One measurement worth recording: the FD check needs `eps = 1e-7`
  (9e-2 relative error at 1e-4, 2e-5 at 1e-7) — WENO's nonlinear
  weights make the loss only piecewise smooth, so a coarse step is
  dominated by truncation error, not by an AD defect. And the gradient
  on the flat axis's ghost slots is **exactly 0**, confirming the
  tiled fill is a pure function of the true DOFs.

Not shipped: flat-axis elision (recommendation 2/3, parked in
[`../roadmap/deferred.md`](../roadmap/deferred.md) pending the gpu
measurement) and the asymmetric halo (recommendation 5).

## 11. Follow-ups noted in passing

- The `dancing_eddies` prose must change when the fix lands (docs
  content -> owner review, per AGENTS.md).
- ~~`_check_walled_extent` keys `min_cells` on cells while the bounded
  fill guard keys on DOFs; `OUTER/NEUMANN` has 2 DOFs at `n_cells = 1`
  yet still trips at width 2. Worth aligning.~~ **Closed as a
  misreading** — see §12.9: the Neumann `_MEMBER` rule excludes the
  reflected-about boundary node, so the DOF-keyed guard reduces to
  `width <= mesh.n_cells` uniformly and the two currencies already
  agree. `_check_walled_extent` is strictly stricter and always fires
  first with a taught message.
- ~~Zero-DOF spaces flow through `_axis_map` undefended (§5).~~
  **Closed** by `fix/walled-thin-axis-fill`, 2026-08-12. The map was
  never the problem — it honours the `sign == 0` vacant-slot rule and
  returns exact zeros. Its documented twin `_write_axis` did not: via
  `_ghost_values` it read `dof(rank)` purely to obtain a *shape* for
  that slot, so a walled `nz = 1` model assembled and then raised
  `NotImplementedError` from the first step (`w` lands on
  `Inner(DIRICHLET)` with 0 DOFs). `_bounded_ghosts` now builds the
  slot from the storage frame (`zero_slot`) and passes it in; values
  are unchanged for every `n >= 1`, and the multi-device caller
  (`_exchange_block`, always `n = width + 1 >= 2`) is byte-identical.
  Semantics, owner's call: a walled 1-cell axis **runs**, stepping a
  genuinely empty wall-normal velocity — no assembly-time refusal, no
  interior face to carry flow. The twin invariant is now asserted over
  the whole bounded matrix in
  `tests/spatial/decomposition/test_tensor_flat_axis.py`.
- `done.md` ~2370 claims "The weno5 momentum z-seam residual is the one
  item left open in this family (`open.md`)", but no such entry exists
  in `open.md` today.

## 12. Corrections — the elision campaign (2026-08-12)

The follow-up campaign (four parallel investigations plus a working
prototype) **overturns §6's payoff estimate by an order of magnitude
and retires two of its recommendations.** Method: the recommended
repeat-and-run shape was actually built, then A/B'd against its own
branch point by three independent harnesses. Every number below is a
direct measurement, not a fit.

### 12.1 §6's "5–13% on cpu" is wrong; the real figure is 2.2x–7.7x

| config | base | elided | speedup | z storage |
|---|---:|---:|---:|---|
| 128², centered+biharmonic | 13.37 ms | 6.15 ms | **2.17x** | 3 -> 1 |
| 128², WENO5 | 30.02 ms | 3.88 ms | **7.73x** | 7 -> 1 |
| 512², WENO5 | 308.23 ms | 41.38 ms | **7.45x** | 7 -> 1 |

Three harnesses, one prototype, agreeing on ratios (WENO5 5.8–7.7x,
centered+biharmonic 1.9–2.2x) while absolute times spread ~2x with box
load. `u` sum-of-squares after 20 steps is **identical to the last
digit** across arms in every pair.

**Why §6 was wrong.** It had no prototype, so it extrapolated
`t = a + b·N_stored` from two anchors (nz=4, 16) down to nz=1 — a 5x
extrapolation below the lower anchor. The variable coefficient
reproduces (`b ≈ 0.19` vs §6's `0.222 µs`/element); the **fixed term
does not**. Refits on real prototype data give `a ≈ 10.6 ms`, and
across configurations `a` swings from **−91 ms to +84 ms** — a
two-point fit cannot separate fixed cost from curvature. §6's
`a = 103.3 ms` (84% of its 122.9 ms step) is ~10x too large, and
because the saving is `variable / (a + variable)`, a 10x-too-large `a`
makes the same saving look 10x smaller. The fit passed its own
out-of-sample nz=1 check, which is why §6 trusted it: interpolating
one step below the lower anchor is easy, extrapolating 5x below it is
not.

**The step is bandwidth-bound, which is what carries the result to
gpu.** Post-fusion `cost_analysis()` gives arithmetic intensity
**0.13–0.44 FLOP/byte** (elided 0.20–0.67) against an A100-80GB fp64
machine balance of 4.76 — bandwidth-bound by 11–37x at every size, in
both arms. Achieved bandwidth is near-constant at 10.6–15.2 GB/s
across schemes, sizes and arms, which is why the byte ratio predicts
wall time. The same grid at nz=**16** reads 11.0 FLOP/byte and is
compute-bound: ghost planes along a *constant* axis carry bytes and no
useful arithmetic. Byte/FLOP ratios are size-invariant to 3 s.f.
(3.40x/7.69x bytes, 2.29x/4.98x flops — FLOPs drop less because
repeat-and-run recomputes the widened window, harmless precisely
because the step is bandwidth-bound).

### 12.2 §6's synthetic proxy is an artifact — discard it, keep the conclusion

The 514² proxy's 4.9x/11.8x/15.3x at z-extent 3/5/7, and the reading
that "a 3-long fastest-varying storage axis is a pathological SIMD
width", do not survive a direct sweep (one process per point, 4
samples, forward and reverse z order):

| n=2048, ns/element | z=1 | 2 | 3 | 4 | **5** | 6 | **7** | **8** | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | 2.61 | 3.06 | 2.99 | 2.40 | **12.27** | 3.59 | **12.48** | **12.61** | 3.05 | 3.28 |

The cliff is at z ∈ {5,7,8}, ~5x, reproducible; **z=3 shows only
1.14–1.33x** and z=6/12/16 are fine. Slow-5 / fast-6 / slow-7 / slow-8
/ fast-12 cannot come from "3 is a pathological vector length". There
is also **no collapse bonus at z=1** (2.61 ns/element vs 2.40 at z=4),
so the "XLA collapses the axis and vectorizes along y at width 0"
mechanism is absent from the proxy. §6's figures land on a codegen
cliff. The real step's speedup never exceeds its byte ratio (7.73x
measured vs 7.69x bytes), so the proxy was **over**-optimistic, not
optimistic in the right direction. The case no longer needs it.

*Unresolved and immaterial:* whether a length-1 trailing axis earns a
small bonus beyond the storage ratio. The microbenchmark says no; a
fit residual on the real model says yes; the one measurement marginally
above the byte ratio is within box noise.

### 12.3 The asymmetric halo saves nothing — retired

§6's closing bullet ("plausibly worth more than flat-axis elision")
was a **per-row claim presented at tendency scope**. Instrumenting the
input to `HaloSpec.symmetric` at both collapse sites (`halo.py` ~1298,
`halo_demand.py` ~143) across nine configurations — nh centered,
weno3, weno5, upwind5, weno5 walled-z, advection=False, sw advective
and linear — records **zero asymmetric specs**: WENO5 arrives at the
collapse as `(3,3)`, not `(2,3)`.

The mechanism is structural. Individual applications *do* record
asymmetry (per axis the recorder sees `(1,3)`x6 **and** `(3,1)`x2,
`(1,0)`x10, `(0,1)`x4) and that asymmetry is already exploited per row
(`advection.py` ~1479: "tightens to the true composed footprint,
`n + 6` per axis for upwind5, not `n + 8`"). But a complete tendency
leg symmetrizes: staggered schemes pair a forward difference with a
backward one, and upwinding keeps both biased reconstructions resident
because the flux sign selects at runtime. Storage is one array, so it
holds the per-side max — exactly `storage_halo_width.md` §1's own
derivation, `[-1,0] ⊕ [-2,+3] = [-3,+3]`. Ceiling had it existed:
**4.4% / 2.3% / 1.15%** of step bytes at 64³/128³/256³. Entry moved to
[`../roadmap/declined.md`](../roadmap/declined.md).

**The lever it was reaching for is real but different:** on a *mapped*
grid, centered negotiates halo **2** while the trace demands only 1 —
a shape-unconditional one-plane-per-axis narrowing worth ~8% (64³) /
~4% (128³) of step bytes. Already flagged in
[`storage_halo_width.md`](storage_halo_width.md) §1 as
`DynamicalCore.extra_halo = 2`; now measured on the mapped path.

### 12.4 §5's bounded-axis warning holds, and the trap is worse than recorded

The predicate `factor.mesh.periodic and factor.mesh.n_cells == 1`
survives scrutiny and every load-bearing §5 claim reproduces
(`Outer(2) -> Center(1)` of `[1,3]` is exactly `2` at `dz = 1`; a
bounded `CellAvg(1)`+Dirichlet column syncs to `[-1, 1, -1]`, giving
`2u/dz`). Two corrections, both making the trap **worse**:

1. **§5's DOF table is the untagged column only.** Under a Dirichlet
   tag the *member* boundary DOFs are dropped, so at `n_cells = 1`
   `Outer`, `Left` and `Right` all collapse to **0** DOFs. So
   `mesh.n_cells == 1` spans DOF counts **0, 1 and 2** — a three-way
   spread, not §5's two-way one.
2. **The sharpest trap is at `n_cells = 2`, which §5 never discusses.**
   A walled `nz = 2` `nonhydro2` model — shipped, running today — puts
   `w` on `Inner(z, DIRICHLET|DIRICHLET)` whose z factor has
   `shape[0] == 1`, and its vertical divergence is `±1/dz`, **nonzero
   for a constant column**. A contributor keying a flat rule on the
   field's own shape rather than the mesh's cell count breaks a live
   model where no `n_cells == 1` check and no "flat axis" intuition
   would warn them.

A *sound* bounded predicate does exist — key on the realized fill being
a pure copy of the single DOF (`all(src == width)`, no sign flips, no
zeroed slots) rather than on the mesh — and it additionally admits
bounded `Center/NEUMANN` and `CellAvg/NEUMANN` at width 1. It is not
worth promoting to the gate: on a bounded 1-cell axis the staggered
codomain is *empty* (`Center(1) --diff-> Inner(0,)`), so the win is one
operand's 3x; it re-opens the per-factor-vs-per-axis halo split that
seven storage-index sites depend on; and its dangerous sibling is one
keystroke away (`Center/DIRICHLET` has the same DOF count *and* the
same `src` indices as the Neumann pair, differing only in `neg`).
Keep it as an assertion **inside** the gate, never as the gate.

**The one-sentence form for a future contributor:** a size-1
*periodic* axis has one DOF and a wrap fill, so its whole storage
column is that one DOF and every stencil along it is identity or zero;
a size-1 *bounded* axis has zero, one or two DOFs depending on the BC
tag and its ghost fill is a BC extension, not a copy. Never key a flat
rule on `n_cells == 1` or on `shape[0] == 1`.

### 12.5 §6's "silent wrong answers" is mostly wrong — the failure is loud

Forcing `_width -> 0` with both tails untouched does **not** silently
zero. It raises at assembly `dry_run` for every configuration:
`TermEvaluationError: ConstantStratification/buoyancy_force: ...
ValueError: axis 2 of length 1 is shorter than the 2-point stencil`
(`stencil_kernels.py` ~215, twin at `weno.py` ~369).

The silent zero needs `m0 <= 0` **plus** an unguarded slicer, and
exactly one existed: `_weighted_windows` (`advection.py` ~585), which
returned shape `(4,4,0)` on a length-1 axis for
`apply_fv_staggered`'s `jnp.pad` to fill with exact zeros. Pinned
configuration: size 3 with `m0 = 0`, i.e. `UpwindAdvection(order=3)`'s
right-biased rung (`graded.biased_offset = order//2 - 1 = 0`) —
`[0.]` where the answer is `2.0`, no diagnostic. Latent in shipped
code (storage is always `1 + 2H >= stencil`), reachable under elision.
**Guard shipped** on `dev` as `fix/weighted-windows-guard`.

### 12.6 The function-space retag route is dead

Reusing `ConstantSpace` for a flat *physical* axis fails on four
independent grounds, three of them the defining properties of the class
being reused. `collapses_axis` bundles halo-free storage, device
replication, per-factor dispatch skip and squeeze-out-of-output; a flat
physical axis wants the first two and must **not** have the last two.
Measured: the pressure core's C-grid pair discovery (`nonhydro2/modules/
core.py` ~580) finds the face factor as "the velocity factor that
differs from the pressure factor", so retagging collapses the pair and
a bare `next` raises `StopIteration` at bind — no model builds. The
registry's real `FiniteDifference` bound to a `Constant(z)` returns
`d/dz(1) = 1.0` (identity, where a flat axis needs **zero**) —
**silently**. `evaluation_nodes` and `measure` raise, `.xr` drops the z
dim and coord, and `Center(z) * Right(z)` stops raising
`SpaceMismatchError`. The honest contribution is a *third* predicate
(storage-only), not a retag — which is option 1's storage half under a
better name.

### 12.7 Costs and traps of the real implementation

~40 executable lines across 5 files (+204/−0). **One of ~12 stencil
families needed individual attention** — repeat-and-run's central claim
held for FiniteDifference, LinearInterp, Restriction, the three FV
reconstructions, WenoReconstruction, the FluxDifference family,
UpwindOne, Fallback, the graded rungs and both private face
reconstructions.

**The exception is the campaign's genuine unknown unknown.**
`_SelectedFaceReconstruction` (`advection.py` ~1790) closes over
`pos_data = positive._data` and slices it with the *rebuilt* window's
length. `jnp.repeat` widens only the array the tail is handed, never
what a kernel **captured**. **Repeat-and-run is transparent to kernels
but not to kernel closures.** §6 did not anticipate this.

**Correction (the merge-ready pass): the co-operand trap is not
reliably loud.** The prototype broke with `ValueError: Incompatible
shapes for broadcasting: shapes=[(14,14,0),(14,14,2),(14,14,2)]` only
because WENO5's `m0 = 2` slices the stale one-slot capture to length
**0**. At `m0 = 0` it slices to length **1**, which *broadcasts*
against the rebuilt window instead of erroring — a **silently wrong**
answer. Discovered by writing the negative test, which failed to raise.
So a `co_operands=` declaration is necessary but not sufficient: the
shipped implementation adds `staggering.flat_captures`, which on the
flat path inspects the kernel's closure for a `jax.Array` of the
operand's rank holding one slot on the flat axis and refuses, naming
the free variable and `co_operands=`. Zero false positives across 13
configs, and not a proof — an array inside a custom object escapes it,
which its docstring states.

Verified non-issues, contradicting §6's predictions: `require_solver_halo`
needs **no exemption** (the CG/immersed/mapped cores read `.data`, not
raw storage, and their `jnp.roll` wraps are length-1-correct — proven by
walled-x CG and multigrid-preconditioner runs); the `apply_staggered` /
`apply_fv_staggered` reach guards pass unchanged and still teach,
because `decomposition.halo` is untouched; `halo_valid` shows **no
churn** (compile counts identical, zero extra compiles on re-run) when
the flat axis's claim is left unconsumed; §4's multi-device conclusion
holds under elision.

Elision is *more* exact than the deep run: §10's `w`-at-1e-20 residual
comes from stenciling wrap-filled ghosts, and with no ghosts to stencil
the vertical flux difference cancels identically — `w` and `b` become
**exactly 0**.

**Correction: "`u`/`v`/`p` bitwise" is not an invariant.** Across 13
configs on the merge-ready implementation, the `nz = 2` control and
every flat-**y** config are fully bitwise (all fields, `w` and `b`
included), and 6 of 10 flat-**z** configs are bitwise on `u`/`v`/`p` —
but the other four (WENO5, FV, immersed, multigrid, walled-x CG) differ
by **exactly 1 ulp** (rel 1e-16 on `u`/`v`, ~2e-15 on `p`). Traced
rather than assumed: stepping WENO5 at 16², steps 1–3 are bitwise and
the 1 ulp appears at step 4. The *deep* run's `w` residual grows
1.0e-19 -> 3.3e-19 while the flat run's stays exactly 0, that residual
feeds `b` through `w·N²` and then the projection, and the two
trajectories separate at the 1-ulp level once it is large enough. The
elided run is the cleaner of the two — the divergence is the baseline's
own roundoff, not an elision error. The prototype's "bitwise `u`/`v`/`p`"
held at its step counts and sizes; it does not generalize. The
load-bearing checks are stronger and do hold bitwise:
`test_flat_run_reproduces_a_z_replicated_deep_run` on `u`/`v`/`p`, and
every advection tendency at orders 3 and 5 against a replicated deep
run. One real coverage loss: with elision, `_axis_map`'s
tiling branch (§10) is only reachable for `2 <= n < width`, so
`test_periodic_wrap_wider_than_the_axis_tiles` needs retargeting rather
than deleting.

### 12.8 A live bug found in passing, unrelated to elision

A **walled 1-cell axis** assembles and then raises at the first step:
`NotImplementedError: the bounded halo fill of width 1 reaches deeper
than the 0 DOFs along the axis` (`tensor.py` ~1496 `_bounded_ghosts` ->
~1616 `_ghost_values` -> ~1490 `dof`). `nz=2` and `nz=4` are fine.

The two fill spellings **disagree at `n == 0`**: `_axis_map` honours the
Dirichlet `_VACANT` `sign == 0` rule and returns exact zeros, while
`_ghost_values` takes the same branch as `jnp.zeros_like(dof(rank))` —
calling `dof` purely for a *shape* — and `dof` raises when `k > n`. So
`_write_axis`'s docstring promise of "the same values" is false for a
zero-DOF Dirichlet factor. It fires at the scan carry, which is why it
survives assembly and every dry run. **Predates the thin-axis fix**
(`baf6bfd0` left the bounded path byte-unchanged). Owner ruling
(2026-08-12): make it run with a genuinely empty wall-normal velocity —
the `sign == 0` slot takes its shape from the storage frame instead of
from `dof`.

### 12.9 §11 follow-up resolved: `_check_walled_extent` is consistent

§11 suspected a cells-vs-DOFs misalignment. There is none: the
Neumann `_MEMBER` rule returns `rank = k + 1` because the boundary node
is a true DOF the even extension reflects *about*, so the extra `Outer`
node is consumed exactly and the DOF-keyed guard reduces to
**`width <= mesh.n_cells`** uniformly for every grounded iteration-1
space (measured: first-admitting `n_cells` per width 1..5 is
`[1,2,3,4,5]` for BC-tagged spaces). The two currencies already agree.
The guards also do not race — `_check_walled_extent`'s `min_cells` is
strictly stricter and always fires first with a taught message (order 3
needs 4 cells but the fill admits from 2; order 5 needs 6, fill admits
from 3), so the fill guard is unreachable on that path. Worth doing
only: state the `width <= n_cells` law in `_axis_map`'s docstring, and
name `mesh.n_cells` in the `rank > n` message (reachable via a
`FiniteDifference(order>=4)` registry override, which has no taught
pre-flight).
