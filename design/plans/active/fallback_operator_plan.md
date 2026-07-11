---
status: active
date: 2026-07-08
---

# Graded-order boundary fallback operator — implementation plan

**Status: plan, ready to implement (2026-07-07).** Delivers the
"reduce order near the wall, using interior DOFs only" closure the
Oceananigans `buffer_scheme` chain implements — the designed-for path
`operators_stencils.md:283-287` flagged for bounded WENO. Cross-refs:
graded-vs-one-sided distinction and the numerical settled design in
this session's analysis; legality tie-in in
[`bc_free_boundaries.md`](bc_free_boundaries.md) (R1) and
[`boundary_plan.md`](boundary_plan.md) (R2).

## 1. Goal and non-goals

**Goal.** A `Fallback` separable operator that runs a wide high-order
kernel in the interior and, at the `K` output rows adjacent to each
physical wall, substitutes progressively narrower interior-only
stencils (order p → p−2 → … → 1). It **reads no exterior values and
needs no BC**, so it makes a wide bounded reconstruction R1-legal *by
construction* and retires WENO's periodic-only restriction
(`weno.py:584-590`). Primary target: `WenoReconstruction(order=5,
boundary="graded")`; the operator is order/kind-agnostic so linear
FD/interp graded variants fall out later.

**Non-goals (designed-for, out of scope here).**
- Immersed-boundary graded reconstruction — the reduced-order zone is
  a geometry mask, not a static edge slice, so it needs a
  `jnp.where`/gather path, not this static-slice one.
- Order-*preserving* one-sided WENO (one-sided smoothness indicators).
  Graded *reduces* order; that is the easy, shipping choice.
- The R1 flip / width-aware legality guard on the *default* rows
  (separate work item; this operator is opt-in, so it doesn't depend
  on it).
- Sharding beyond `layout="local"` (the decomposition-layer
  physical-end patch that keeps the interior `layout="any"` is a
  later refinement).

## 2. Numerics (settled)

Static-slice partition, **not** masking. `K` = number of reduced rows
per side = `interior.halo − boundary_slack`, a compile-time constant
(WENO-5: halo 3, one interior face of slack ⇒ K = 2). Output face `k`
counting from the wall is computed by the widest interior-only stencil
that fits — the interior kernel covers faces `[K, n−1−K]`, rung `k`
covers wall-adjacent faces `k` and `n−1−k`. Each face is computed
**once**; cost ≈ one interior pass + `2K` thin boundary slabs. No
candidate is computed-then-discarded per point; the interior-vs-
boundary split is a static index partition (`arr[:K]`, `arr[K:-K]`,
`arr[-K:]`), the array-language twin of Oceananigans' compile-time
`ifelse` recursion. Linear members collapse further to a single
position-dependent static stencil band (one matmul); WENO cannot
(data-dependent weights) so it uses the slice partition.

## 3. Architecture fit (seams, with file:line)

- **Base contract** (`base.py:585-742`): a `SeparableOperator`
  implements `codomain(1D factor)`, `requirements(domain)`,
  `_apply_factor(f, axis)`. The `__call__` template handles tracer
  interception, consumption-side sync, and layout re-attach — `Fallback`
  writes none of that.
- **Sibling to `SeparableComposite`** (`base.py:861`): a new
  `SeparableOperator` subclass, **not** built by `@`; interned on its
  static structure via `_ALGEBRA_TABLE.intern(key, build)`
  (`base.py:69`, pattern at `:1604`).
- **Application tail** `apply_fv_staggered` (`reconstruct.py:185-248`):
  `full = kernel(storage, axis)` → slice valid interior `piece` →
  `jnp.pad(piece, (lo, s_out-hi))` zero-fills the edge slots → claim
  resets to zero on bounded axes (`:243-246`). **Constraint:
  reconstruct.py is read-only for this cluster** (`weno.py:6-9`), so
  `Fallback` must NOT modify `apply_fv_staggered`. It reuses it for the
  interior pass and does the boundary-slot fill itself (the pad slots
  are exactly the `K` rows to overwrite).
- **Exact coefficient generators** (reuse, do not re-derive):
  `_shu_row(cells, face)` (`weno.py:107`) — any-size FV reconstruction
  coeffs; `weno_tables`/`weno_reconstruct(order, bias)`
  (`weno.py:226,397`) — reduced-order WENO for a rung;
  `staggered_diff_weights`/`_solve_exact` (`stencil_kernels.py:47-131`)
  — nodal one-sided/narrow stencils for the linear case.
- **Registration**: `dispatch_kind` ClassVar + registry rows; graded is
  a **module-override** row (opt-in), like WENO today — *not* a default
  table row.

## 4. The `Fallback` design (target API)

```python
@final
class Fallback(SeparableOperator):
    """Graded near-boundary order reduction over interior DOFs only."""
    dispatch_kind = "reconstruct"          # inherits the interior's kind

    def __init__(self, interior, boundary):  # boundary: tuple, wall→in? see note
        ...  # static structure; interned on (interior, boundary)

    def codomain(self, domain):
        # accept BOTH periodic and bounded; return interior.codomain(domain)
        # but on a bounded mesh the wide interior alone would raise —
        # Fallback overrides that to the interior-face codomain
        # (CellAvg -> Inner) and asserts every ladder rung agrees.
        ...

    def requirements(self, domain):
        # halo = interior.requirements(domain).halo (interior dominates);
        # layout = "local" on a bounded axis (iteration 1), else "any".
        ...

    def _apply_factor(self, f, axis):
        # 1. interior pass via apply_fv_staggered(interior, ...) → valid
        #    interior faces [K, n-1-K]; edge slots come back zero-padded.
        # 2. for k in range(K): compute rung_k on the wall-side window at
        #    each end, write into physical-end output slot k / n-1-k.
        # 3. concat; bounded claim reset is inherited from the tail.
        ...
```

**Ladder builder / knob.** `WenoReconstruction(order=5,
boundary="graded")` returns `Fallback(WenoReconstruction(5),
(WenoReconstruction(3), UpwindOne()))`. Innermost rung is 1st-order
upwind (`_shu_row(1, face)` = the single wall cell). `boundary="graded"`
sits beside R2's `boundary="one_sided"` on the same operators.

**Bias at the wall.** The upwind pair is selected by flux sign
(`Where`, unchanged). Each biased instance grades independently; the
rung windows are the wall-side interior cells for that bias. The plan's
implementer must confirm the left/right window alignment against
`weno_reconstruct`'s `align=m0` (`weno.py:651-659`).

## 5. Verification (the decisive gates)

- **No exterior read (the key correctness gate).** On a 1-D bounded
  grid, poison the exterior halo with NaN, apply the graded operator,
  assert the output is entirely finite. This *proves* interior-only
  access without inspecting indices.
- **Interior order.** On a smooth field, convergence rate at interior
  faces = p (5 for WENO-5), and graceful (monotone, ≥1) reduction at
  the wall rows — no order *blow-up* or NaN.
- **Bounded legality.** `WenoReconstruction(5, boundary="graded")` on a
  bounded `CellAvg` factor resolves `CellAvg -> Inner` (no raise);
  plain `WenoReconstruction(5)` still raises on bounded (unchanged).
- **jit single-compile.** Sweeping grid values compiles once (static K,
  static slots).
- **Multi-device forced-4.** `layout="local"` keeps the bounded axis
  undistributed; bitwise-identical to single-device.
- **Linear d² legality (when linear graded lands).** The
  `f=y(1−y)` / `d²=−2` case (`bc_free_boundaries.md:22-35`) computed via
  a graded linear stencil, no extrapolation fill.

## 6. Work breakdown (staged; file ownership to avoid collisions)

- **F1 — `Fallback` core.** New file
  `src/fridom/framework2/grid/operators/fallback.py`: the operator, the
  graded ladder builder, the reduced-order rung kernels (via `_shu_row`
  / `weno_reconstruct`), and the boundary-slot assembly in
  `_apply_factor`. New test file `tests/framework2/.../test_fallback.py`
  with the NaN-poison gate + a small bounded convergence smoke test.
  Owns only new files. **No edits to reconstruct.py / base.py.**
- **F2 — the `boundary="graded"` knob.** Edit `weno.py`: add the knob to
  `WenoReconstruction.__init__` (returns/dispatches to a `Fallback`);
  make `codomain` accept bounded when graded. Registration example.
  Depends on F1.
- **F3 — validation suite.** Convergence, jit-single-compile,
  forced-4 multi-device, bounded-legality. Separate test files.
  Depends on F1+F2.
- **F4 (follow-on) — linear graded.** `finite_difference.py` /
  `interp.py` graded knobs reusing `Fallback`; the d² legality case.

Order: F1 → F2 → F3; F4 optional/after. F1 is the crux and is done by
one agent (the boundary-index arithmetic must stay internally
consistent — do not split it).

## 7. Decisions taken (flag if the owner disagrees)

- Graded is **module-override only**, not a default bounded row
  (matches WENO; keeps the default table closed under R1).
- Iteration 1 = **WENO-graded** primary (highest value, retires
  periodic-only); linear graded is F4.
- `layout="local"` on bounded now; decomposition-layer physical-end
  patch (interior stays `any`) is designed-for.
- Innermost rung = 1st-order upwind cell value; confirm it equals the
  `LinearReconstruction` wall result under both biases.

## 8. F6 — distributed bounded axis (drop `layout="local"`)

**Status: designed, ready to implement (2026-07-07).** Removes the
iteration-1 non-goal: make the graded `Fallback` bitwise-correct when
the bounded axis it grades is sharded, so `layout="local"` becomes
`"any"`. Backed by two converging lines of evidence — the framework's
own `_exchange_block` and Oceananigans' per-rank topology dispatch
(`RightConnected`/`LeftConnected` grade only wall-touching ranks;
`FullyConnected` interior ranks run full-order+halo). Full research:
this session's two agent reports.

### Mechanism (settled)

The stencil kernel path is **SPMD-on-global**, no `shard_map` (that
lives only in `sync`). Storage is a *blocked* global array; `sync`
already fills each block's halos (interior edges `ppermute`, physical
walls `axis_index`-masked BC fill), so the **interior wide pass is
already bitwise-correct on a sharded bounded axis** — identical to the
sharded periodic case (proven device-count-invariant). Only the two
physical-wall blocks' `K` faces need the rung patch. The current patch
is wrong solely because it does single-block index arithmetic on a
blocked global array (the right-wall write `width+(n-d-1)` with global
`n` lands in an interior block — F3's ~0.19 bug).

Fix: move the patch behind a decomposition-layer seam that mirrors
`_exchange_block` — `shard_map` + `axis_index` + masking — so
`Fallback` stays shard-blind (the design rule: kernels are not
shard-position-aware; the decomposition layer is). Rejected
alternatives: global-index `.at[].set`/`scatter` (all-gathers or
miscomputes today, jax#23052); operator-owned `shard_map` (leaks block
geometry, violates the rule).

### Change list

- **`decomposition/decomposition.py`** — add abstract
  `patch_physical_ends(out_arr, in_arr, out_space, in_space, axis,
  patch, *, layout=None)` beside `sync`. `patch` is a **pure,
  shard-blind** local-block callback
  `patch(in_block, out_block, side, width_in, t_in, width_out,
  t_out) -> out_block` overwriting the `K` wall faces of one
  `[width|true|trail]` block from its wall-side interior cells.
- **`decomposition/tensor.py`** — implement it: `shards == 1` static
  branch (both walls on the single block, `t = n` — bitwise-identical
  to today); `shards >= 2` one `jax.shard_map` over the axis with
  `s = axis_index`, `t = where(s==shards-1, n-(shards-1)*cells, cells)`
  (the `_exchange_block:957-959` idiom), compute left+right patches on
  every shard, then `where(s==0, patched_left, block)` /
  `where(s==shards-1, patched_right, block)`. Reuse `_cells_per_shard`,
  `_geometry`, `_take`/`_set`/`dynamic_slice_in_dim`.
- **`operators/fallback.py`** — `requirements` returns
  `layout="any"` always (delete the bounded branch); `_apply_factor`
  keeps the interior `apply_fv_staggered` pass + periodic early-return,
  and replaces the wall loop with one `patch_physical_ends` call. The
  rung/`_set_slot` logic becomes the `patch` closure, **re-indexed off
  the local `t`** instead of global `n` (the one real code change).
- **`decomposition/traits.py`** — only for **order ≥ 9**: declare
  `min_local_size = order-2` on the graded space. **No change for
  WENO-3/5/7**: `order-2 ≤ width+1 = order//2+2` holds through order 7,
  and even cell-sharding (`n_cells % shards == 0`) is already
  negotiated — so the graded region never straddles a shard and the
  two walls always live in different shards when `shards ≥ 2`.
- **`tests/.../decomposition/test_fallback_multi_device.py`** — flip
  the gate: genuinely shard the bounded axis (1-D bounded
  `IntervalMesh` forced-4, or a layout sharding `y`); assert forced-4
  bitwise device-count invariance (per-shard **and** gathered) + finite
  output; add the NaN-poison exterior-halo gate under sharding.

Do **not** touch `reconstruct.py` / `base.py` (read-only for this
cluster) or `sync`/`_exchange_block` (the operand sync is already
correct). Watch-item: two co-sharded arrays (input `CellAvg` vs output
`Inner`, different block sizes, same device-mesh axis + cells-per-shard)
inside one `shard_map` — pinned by the forced-4 bitwise + gathered gate.
