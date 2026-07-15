---
status: frozen
date: 2026-07-15
---

# Indivisible-extent sharding — probe evidence

Research report (see [`README.md`](README.md) for status); feeds
[`../plans/active/indivisible_shard_plan.md`](../plans/active/indivisible_shard_plan.md).

Four probes, run 2026-07-15 on the 4×A100 node (jax 0.10.2, x64) and
forced-4 CPU (`XLA_FLAGS=--xla_force_host_platform_device_count=4`;
collective *structure* is backend-independent). Probe scripts lived in
the session scratchpad (evaporates); every load-bearing spelling and
repro recipe is inlined here.

## P1 — HLO forensics: what exactly explodes on a walled sharded axis

Whole-step compiled HLO of the nh linear step (N=32, forced-4; N=64
verified identical structure), collectives attributed to source lines
via the HLO stack-frame table.

| opcode | x-walled | z-walled | periodic |
|---|---|---|---|
| collective-permute | **274** | 35 | 34 |
| dynamic-slice | **158** | 5 | 6 |
| all-gather | **4** | 0 | 0 |

z-walled ≈ periodic: wall physics off the shard axis costs nothing.
The x-walled excess attributes **entirely** to the decomposition's
*global* reblock path — `tensor.py` `_take` (unpad), `_scatter_axis` /
`_gather_axis` (pad/unpad bodies) — reached from **true-frame
excursions in the hot loop**:

- `adam_bashforth.py:563` — `_weighted` uses
  `f.with_data(w * f.data)`: `.data` = unpad, `.with_data` = pad,
  per state field per step (108 of the 274 permutes);
- the coriolis space-lifts (`_lift_field` / `_broadcast_factor` /
  `.to`, `scalar_field.py`);
- once per chunk: `_scrub_ghost_storage`, `_all_finite` (`model.py`).

The stencil operators are storage-frame-only and clean; the spectral
solve's reblocks are on center fields and take the fast plan — **the
solve is not implicated in the walled case.**

Per-op microscope (x-walled; u = 31-extent staggered leg, b = 32
center control):

| op | u (31) | b (32) |
|---|---|---|
| `f * 2`, `f + f` (storage path) | clean | clean |
| `f.with_data(w * f.data)` (AB `_weighted`) | **ds:16 cp:32** | clean |
| `f.with_storage(w * f._data)` | clean | clean |
| `grid.sync(f)` | +local BC fill only | clean |

Sharding trace: `u._data` (storage frame) is block-sharded
`P('devices',…)`; **`u.data` (true frame) comes back REPLICATED** —
a 31-extent array cannot block-align with the 13-slot storage blocks,
so unpad materializes a replicated gather and the following pad
re-scatters it. That pair is the per-excursion collective cost. The
count scales with excursions × shards, not N — which is why it
dominates wall time at every grid size.

**Why the 31-leg takes the global path:** the reblock fast-path gate
(`tensor.py:525`)

```python
divisible = n_cells is not None and n_cells % shards == 0
if n > shards * cells or (divisible and n != shards * cells):
    return None          # → global slice/concat reblock
```

excludes the *divisible staggered-deficit* leg (`n = P·cells − 1` when
`n_cells % P == 0` — exactly the walled staggered velocity). The
non-divisible mild case (e.g. center 257) already takes the validated
padded-even plan; the walled leg was left on the legacy path.

**H2 — the decisive experiment.** Monkeypatching the gate to reject
only `n > shards * cells` routes the deficit leg through the existing
padded-even plan (`surplus = 1`). Results:

- storage placement **byte-for-byte identical** to the global path;
  `unpad(pad(true)) == true`;
- eager `pad(unpad(u))`: `{ds:16, cp:32}` → **`{}`** (collective-free;
  at `n_cells ≡ 0 mod P` the frames align exactly);
- **whole-step x-walled: collective-permute 274 → 34 (== periodic),
  dynamic-slice 158 → 19, all-gather 4 → 0.**

Residual vs periodic (ds 19 vs 6, dus 79 vs 72) is the local,
non-collective BC-structured halo fill — inherent to having a wall.

Intervention A (independent cross-check): spelling AB `_weighted` as
`with_storage(w * f._data)` is bitwise-identical and removes exactly
the AB share (cp 274→178) — confirming the excursion mechanism, but
fixing only one call site. The gate fix covers all sites at the choke
point.

**Deficit/surplus alignment arithmetic.** For `n = n_cells − 1`,
`n_cells ≡ 0 mod P`: canonical `ceil(n/P)` blocks equal the storage
cells → trim collective-free. `n_cells ≡ 1 mod P` mis-aligns by one →
≤1 collective-permute (neighbour shift), per the uneven-shard-padding
record. The surplus leg (`n = n_cells + 1`, Neumann outer) also falls
through the current gate (`n > P·cells`); its canonical frame
(`ceil(65/4)=17`: 17,17,17,14) vs storage true counts (16,16,16,17)
shifts rows only to the right neighbour → ≤1 collective-permute via
the same plan machinery. Not yet empirically validated (the nh linear
step carries no surplus-leg excursion); validate during
implementation.

## P2 — padded all-to-all transpose for an indivisible split axis

The distributed slab transpose (`lax.all_to_all`) requires the split
axis divisible by P, so the planner declines 257³ and the solve
replicates. Hypothesis: only **local** pad/slice is needed. Validated
exactly.

Working spelling (P=4; frame A = global `(260, 257)` sharded
`P('s', None)`, 65-blocks, true 257; forward A→B):

```python
def fwd(a):                       # inside jax.shard_map,
    a = jnp.fft.fft(a, axis=1)    # local (65, 257)
    a = jnp.pad(a, ((0, 0), (0, 3)))          # local, 257→260
    a = jax.lax.all_to_all(a, "s", split_axis=1,
                           concat_axis=0, tiled=True)  # (260, 65)
    a = a[:257]                                # local slice
    return jnp.fft.fft(a, axis=0)  # FFT at TRUE length
```

Inverse mirrors (`ifft`, pad axis 0, `all_to_all` back, slice axis 1).

- **Numerics:** forward vs single-device `fft2` of the true array:
  0.0 (CPU) / 9.0e-16 (GPU); round trip 9.3e-16; zero pad-lane leak.
- **Collectives:** forward = exactly 1 all-to-all; round trip = 2;
  **no all-gather, no collective-permute**; pads/slices lower local.
- **GPU timing** (c128 n×n, 4×A100, padded shard_map vs replicated
  all-gather+fft2): n=257: 1.03× (launch-latency bound); n=1025:
  **1.76×**; n=2049: **2.10×**.
- Gotchas: `tiled=True` mandatory; `all_to_all` refuses an indivisible
  split axis (`ValueError … has to be divisible`) — hence pad *before*,
  slice *after*; `jax.shard_map` requires inputs already sharded to
  match `in_specs` (`device_put` with the `NamedSharding` first).

**Unrepresentability (load-bearing).** In jax 0.10.2 a global
true-extent indivisible array cannot exist on the mesh:
`device_put` of a 257-axis onto 4 devices raises (`does not evenly
divide`); `x[:257]` on a sharded 260 raises
`NotImplementedError: slicing on sharded dims where out dim (257) is
not divisible …`. Re-blocking a sharded axis to a different block size
(65→66 blocks) costs 1 collective-permute. The padded-even storage
frame is therefore not an optimization but the **only representable
form**; true extent must be logical metadata.

## P3 — what padding costs ("can we pad without copying?")

A100, f64 512³ (1 GiB), padded (515,512,512); block_until_ready,
median ≥20 after warmup:

| case | time | finding |
|---|---|---|
| elementwise, no pad | 1.394 ms | baseline (1541 GB/s) |
| `pad` → same op, one jit | 1.405 ms | **1.01× — pad fuses, ≈0 extra HBM** |
| standalone `pad` (own jit) | 1.423 ms | materializes: full-bandwidth copy |
| DUS into padded buf, no donate | 2.588 ms | clones whole buffer first |
| DUS into padded buf, **donated** | 1.328 ms | **in-place** (HLO aliased, temp=0) |

Donating a `(512,…)` input against a `(515,…)` output emits
`UserWarning: Some donated buffers were not usable` and does **not**
alias — XLA input/output aliasing requires identical shape/layout.

**Verdict:** there is no XLA mechanism to grow a buffer in place;
`pad` always allocates. What *is* free: a pad consumed inside the same
jit (fused), and donated same-shape in-place updates. The storage
doctrine follows: **allocate padded once, keep the padded shape for
the buffer's whole life, write true-extent payloads via donated DUS
(or produce data born-padded), and never pad/unpad across a jit
boundary in the hot path** (each such pad ≈ 1.4 ms/GiB).

## P4 — prior art

- **Pad-and-mask is the established idiom.** GSPMD itself ceil-pads
  uneven tiles internally and masks reductions
  ([arXiv:2105.04663](https://arxiv.org/abs/2105.04663)); P3DFFT's
  `USEEVEN` pads send buffers to use the fast *balanced* all-to-all
  instead of alltoallv ([arXiv:1905.02803](https://arxiv.org/abs/1905.02803));
  cuDecomp supports uneven pencils via NCCL/MPI alltoallv; **jaxDecomp's
  pure-jax backend is `lax.all_to_all(tiled=True)` and simply requires
  divisible grids** — no jax-native prior art does uneven natively.
- **`jax.shard_map` hard-requires even blocks** (no ragged support);
  `with_sharding_constraint` on an indivisible dim can **silently
  replicate** ([jax#26946](https://github.com/jax-ml/jax/issues/26946))
  — never rely on WSC for these extents.
- **`jax.lax.ragged_all_to_all`** is the only jax alltoallv: MoE-shaped
  (offset tables), HLO-only (not StableHLO), experimental on GPU, and
  still divisibility-constrained on its offset arrays. Not a fit for a
  pencil transpose; padding + balanced all-to-all is lower-risk and
  portable.
- **Buffer donation** aliases only same-shape/layout outputs
  ([jax buffer-donation docs](https://docs.jax.dev/en/latest/buffer_donation.html))
  — consistent with P3's measurement.

## Method notes

Collective counting: count opcode *definition* lines in
`.lower(...).compile().as_text()`; GPU spells async collectives
`all-to-all-start/-done` (fold pairs); a naive substring count on CPU
over-counts ~6× (tuple reads, `op_name` metadata). On CPU, XLA
decomposes `all-to-all` into `collective-permute` — compare totals,
not opcode mixes, across backends. Attribution uses the HLO
stack-frame table (`source_file`/`source_line` on each instruction).
Whole-step programs via the model's chunk-executable cache. 4-GPU runs
need `XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion`
(jax-ml/jax#39100).
