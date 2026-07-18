# The sharded-periodic channel remainder: anatomy and solution paths

**Date:** 2026-07-18. **Status:** investigation complete; the
half-axis re-designation (§3.1) **landed** the same day (merge
`feade7fa`, coverage follow-up `964a9117`) — the patches in
[`artifacts/eigen_remainder/`](artifacts/eigen_remainder/) are the
archived evidence of what landed. The 2-D gather path (§3.2) remains
recommended-not-implemented.
**Context:** follow-on to the fused distributed contraction (merge
`e60259de`, `multidevice_test_faults.md` §"Item 1, GPU mechanism —
real fix shipped"). All GPU numbers: 4×A100, jax/jaxlib 0.10.2.

## 1. Why a remainder exists

The fused contraction (`spatial/operators/distributed_contract.py`)
serves one geometry: the 3-D channel on a 1-D device mesh whose layout
shards a periodic axis that is **not** the engine's half (`rfft`) axis.
Its central move is a role split between the two periodic axes: the
half axis `b` (`em.periodic_axis`, pinned to the last periodic axis)
must be *local* when its `rfft` runs (`rfft` consumes the full real
extent), and the sharded axis `a` gets its full `fft` only after one
`all_to_all` re-parks the shardedness onto `b`'s coefficient extent.
Each remainder case breaks one precondition:

1. **2-D channel** (one periodic + one bounded axis): the `rfft` cannot
   run locally and there is no second periodic axis to absorb the
   shardedness. The only other axis is the bounded one — and the
   contraction *reduces over it* (the stacked mode index `d`), so
   parking shardedness there changes the algebra: `amp = Σ_d conj(q)·M·z`
   becomes a per-shard partial sum needing a `psum`. Decline:
   `len(periodic) != 2` (`distributed_contract.py`).
2. **Half axis itself sharded**: the first stage (local `rfft` on `b`)
   is impossible, and the one available reshard happens before any
   transform — leaving `a` sharded when *its* `fft` must run. The
   obstacle is convention, not math: which periodic axis carries the
   half spectrum is a free choice, but `q`, labels, doubling weights,
   slices and synthesis are all *built* half-on-last-axis. Decline:
   `a_name == periodic_axis`.
3. **Non-1-D mesh**: one transpose can never make both FFT axes local;
   the classic fix is a pencil pipeline. Decline:
   `len(mesh.axis_names) != 1`.

Left to plain GSPMD, all three hit the upstream XLA:GPU distributed-FFT
fault (c64 twiddles against c128 data), hence the taught
`NotImplementedError` (`_reject_sharded_projection`).

## 2. Exposure (who actually hits each case)

Default-layout rule (`decomposition.py` `_shardable_names` /
`_shard_rank`): every negotiated layout is a single-axis
`Layout({name: "devices"})` — **the mesh is always 1-D** — and the
sharded axis is the lowest-rank shardable axis, grid order breaking
ties: rank 0 = periodic & divisible by P, rank 1 = other divisible,
rank 2 = indivisible. Extent size never enters. No public API forces a
layout or mesh (`Grid` exposes only `device_ids`). Survey
(`artifacts/eigen_remainder/layout_survey.py`, 11 configs):

- **2-D channel: the DEFAULT outcome.** Any 2-D channel (e.g. a
  shallowwater x-periodic/y-walled strip) on >1 device shards its
  single periodic axis (periodic outranks bounded). Highest-exposure
  remainder; the taught error is what every multi-device 2-D channel
  user sees.
- **Half-axis-sharded: reachable only via extents.** The last periodic
  axis wins the ranking only when every earlier periodic axis is
  worse-ranked — first periodic indivisible by P while the half axis is
  divisible (e.g. x=30, y=32, z walled, P=4 → shards y). Never happens
  on all-divisible (power-of-2) grids. Low, by-accident exposure.
- **Non-1-D mesh: unreachable.** `_build_device_mesh`
  (`tensor.py`) raises `NotImplementedError` on >1 distinct device-axis
  names at decomposition-build time; nothing in src/tests/benchmarks
  constructs one. The eigen engine can never see a 2-D mesh — its
  remainder clause is purely defensive.
- **Bounded-axis-sharded (not a remainder): served today** by plain
  GSPMD (all periodic FFTs local per shard). Verified on 4×A100:
  max |many − one| = 9.9e-14.

## 3. Solution paths, probed

### 3.1 Half axis sharded → **layout-aware half-axis re-designation** (prototyped, green)

The half-axis choice is an engine convention. Designating a periodic
axis the layout does **not** shard as the half axis makes the *shipped
kernel* serve the layout unchanged with roles swapped (a = sharded
ex-half axis, full spectrum; b = new local half axis) — zero new
collective code, 2 all-to-alls per apply, and `q` is *built* directly
in the chosen frame (reorder the `rfftn` `axes=` in `_probe_block`), so
no runtime re-layout of the basis exists at all.

- **Frame freedom proven** (`probe_1b_frame_freedom.py`, real engine +
  real nh labeler in both frames, single device): projections in
  half-on-z vs half-on-x frames agree to ≤ 2.5e-14; forced-frame
  idempotency 2.4e-15; `inverse_l` spectral function 1.5e-13.
- **Assumption inventory:** `em.periodic_axis` is set in exactly one
  place (`eigen_channel.py` `channel_eigenpairs`) and honored by all
  plane/synthesis helpers and by the distributed kernel (already a
  parameter). Hidden last-axis assumptions: `fourier_ops`' transform
  order, and the nh Leray labeler `_constrained_column` (plane axis and
  half extent hardcoded — the one non-trivial site). Blast radius:
  moderate — 3 source sites.
- **Prototype implemented and validated** (branch commits preserved as
  `artifacts/eigen_remainder/0001-*.patch` + `0002-*.patch` against
  base `13f2e082`; 152+/18− over `model/eigen_channel.py`,
  `model/_eigenbasis.py`, `nonhydro2/channel_eigenmodes.py` + one
  GPU-scoped regression test): `_designate_half_axis` keeps the last
  periodic axis unless the layout shards it, then picks the first local
  periodic axis — byte-identical whenever the last periodic axis is
  local (single device and all currently-served layouts). Gates on
  4×A100, default layout sharding the last periodic axis (x=10
  indivisible, z=12 divisible, walled y): many-vs-one 8.5e-15,
  idempotency 1.4e-14, HLO all-to-all present / all-gather absent,
  existing distributed-eigen + transform test files 34 passed (the one
  error is the pre-existing setup `GridFrozenError` on the n=8 fixture,
  reproduced identically at the base commit), ruff clean.
- **Rejected alternative — keep-frame kernel variant:** keep the half
  spectrum on the sharded axis; forward = local `fft(a)` → all_to_all →
  local full `fft(b)` → slice to half (identity `rfftn = fftn` sliced,
  order-independent — verified to 3e-15); backward needs Hermitian
  reconstruction of the upper `b` half via a *flip across the sharded
  a-coefficient axis* (one `collective_permute`), then local inverses.
  Full chain verified to ~1e-14 including non-closed contractions
  (`probe_sol2_keep_frame.py`) — mathematically sound, but strictly
  worse: 2 all-to-all **+ 1 collective_permute** per apply plus a new
  backward kernel stage, versus re-designation's reuse of the shipped
  kernel verbatim. Documented as the fallback if a layout ever *needs*
  the half spectrum pinned to a sharded axis.

Landing note (doc hygiene): with re-designation landed, the "half
axis itself sharded" clauses in `distributed_contract.py` and
`_reject_sharded_projection` are unreachable for the 3-D channel;
they were trimmed to defensive notes in the landing merge
(`feade7fa`), the taught error now naming only the reachable
remainder (2-D channel, non-1-D mesh).

### 3.2 2-D channel → **gather fallback preferred; psum kernel proven but shelved**

Both candidate mechanisms were probed on 4 GPUs (f64):

- **Bounded-partner psum kernel** (`probe_2d_psum_kernel.py`): pad the
  bounded extent → all_to_all (bounded → periodic) → local `rfft` →
  partial contraction over the local `d` shard → `psum(amp)` → output
  einsum → `irfft` → all_to_all back. Exact on divisible /
  indivisible-bounded / indivisible-sharded variants (≤ 2.5e-14); HLO
  all-to-all + all-reduce, no all-gather. It works — but its one
  structural edge (per-device basis sliced ×P) only pays in a regime
  the engine cannot reach:
- **The dense engine bounds 2-D sizes first.** `q[n_kx, D, D]` with
  `D = 3·n_z`: 0.56 GiB at 512×128, 4.5 GiB at 1024×256, 72 GiB at
  4096×512 — where the batched `eigh` that *builds* it is already
  infeasible. At every buildable size the full basis fits one device
  and the contracted fields are megabytes.
- **Gather fallback** (`probe_gather_fallback.py`): `all_gather` the
  sharded periodic axis inside a shard_map, run the existing
  single-device contraction per device, `dynamic_slice` the local shard
  back out. Exact (1.6e-14); HLO all-gather only. For the 2-D channel
  its usual costs vanish: the engine already stores `em.q` fully
  replicated, so no *new* basis memory, and the gathered field is MB.

**Verdict:** serve the 2-D channel with the gather path (it makes the
highest-exposure remainder just work, exactly, reusing the tested
single-device contraction), and keep the psum kernel as a recorded
probe. Caveat carried from the fallback analysis: gather-as-default is
acceptable *here* because there is no faster alternative to hide and
the cost is objectively negligible at every buildable size; as a
*universal* silent fallback (3-D cases included) it would hide a ×P
memory cliff and a ~P/2× bandwidth penalty behind a fast-looking API —
contrary to the prefer-explicit-over-auto-magic ruling — so it stays
scoped to 2-D (owner ratification before implementing).

### 3.3 Non-1-D mesh → **keep the decline; pencil primitive proven for the future**

Unreachable today (§2), so no code is warranted. The primitive a
pencil generalization needs was still probed
(`probe_pencil_primitive.py`, 2×2 forced-CPU mesh): per-mesh-axis
`jax.lax.all_to_all` inside one `shard_map` over both axes composes
cleanly — round-trip 7.9e-16, forward equal to replicated `fftn` to
0.0, HLO all-to-all only. The generalization is feasible and low-risk
the day a 2-D decomposition backend lands; until then the defensive
declines stay.

## 4. Summary of recommendations

| case | exposure | recommendation | status |
|---|---|---|---|
| half axis sharded | low (indivisible-first-periodic extents only) | layout-aware half-axis re-designation | **landed** (merge `feade7fa`) |
| 2-D channel | default for 2-D on >1 device | gather path scoped to 2-D | probed both mechanisms; needs owner ratification + implementation |
| non-1-D mesh | unreachable | keep defensive decline | pencil primitive proven, recorded |

Artifacts: patches + all probe scripts in
[`artifacts/eigen_remainder/`](artifacts/eigen_remainder/). The
prototype branch (`chore/eigen-remainder-investigation`) was deleted
after extracting the patches (worktree-hygiene rule); base commit
`13f2e082`.
