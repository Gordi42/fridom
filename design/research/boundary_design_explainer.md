---
status: frozen
date: 2026-07-11
---

# Boundary handling near walls: dev vs `framework2-boundaries`

A self-contained comparison of the two divergent designs for "what
happens when a staggered operator needs a value outside a walled
domain." Written 2026-07-11 to ground the owner decision recorded as
BLOCKED in `../plans/active/boundary_plan.md`. Every mechanism is
quoted from the actual code on each side.

---

## 1. The shared problem

On a bounded (walled) axis, the staggered operators live on the nodal face-set family: `Center`, `Inner` (interior faces only), `Outer` (interior faces **plus** the two wall faces). A first derivative or interpolation maps between these:

| signature | needs values outside the domain? |
|---|---|
| `Center → Inner` (diff/interp) | **no** — every inner face lies between two real cell centers |
| `Outer → Center` | **no** — the wall faces are true DOFs of the input |
| `Center → Outer` | **yes** — the two wall faces |
| `Inner → Center` | **yes** — recovering the wall-face values (the ghost slots) |

Both sides agree on this table (`../plans/active/bc_free_boundaries.md`, the "legality split" section, present on both). The disagreement is: **when a field carries no BC structure (`BC.NONE`, "BC-free") and an operator's true-shape output needs those exterior wall values, what does the machinery do?**

The R1–R4 sign-off records this precisely. Quoting the branch plan (`git show framework2-boundaries:notes/framework2/boundary_plan.md`), which marks them **SIGNED (owner, 2026-07-07); stages 2a–2d IMPLEMENTED**:

- **R1** (the legality rule): "Remove the BC-free bounded extrapolation fill; an operator row exists on a BC-free bounded operand **iff its true-shape output needs no exterior values** (`Center -> Inner`, `Outer -> Center` stay; `Inner -> Center`, `Center -> Outer` demand BC structure). Missing rows raise `DispatchError` with a hint naming the BC-structured alternatives."
- **R2** (one-sided rows): "Where no BC is physically available (open boundaries, diagnostics), the honest replacement is a **one-sided boundary scheme the user chooses**… interior windows unchanged, the boundary-adjacent outputs patched from one-sided windows over true DOFs only," spelled as a constructor knob `FiniteDifference(order=2, boundary="one_sided")`.
- **R3** (Robin/mixed): "`BC` gains `ROBIN`. The key holds per-side **kinds only** — never the Robin coefficient α, never inhomogeneity values… a side drops its boundary DOF iff its kind is DIRICHLET on a member node set (Robin keeps DOFs, like Neumann)." All BC *values* are dynamic data delivered through `("ghost_fill", space)` rows (stage 2e, still gated).
- **R4** (sequencing): "2a–2d grid-local now-ish, 2e after model 2.2/2.3, 2f optional."

The dev copy of the same file is byte-identical through §7 but stops at **"awaiting owner sign-off"** and has **no §8** — dev never took the implementation path. The two designs are exactly R1-adopted (branch) vs R1-declined (dev).

The design rationale for R1 is in `../plans/active/bc_free_boundaries.md`: the extrapolation fill is "a *hidden numerical scheme*: consumed once it is a defensible second-order one-sided stencil, but composed it is inconsistent — the extrapolation kills curvature, so `f.diff("y").diff("y")` on `f = y(1−y)` returns **0 instead of −2** at the boundary cells… An implicit closure that is fine once and garbage twice, applied silently by the storage layer, is the 'silent wrongness' class the framework exists to forbid."

---

## 2. Dev's current design — keep the fill, keep the table total

### The fill itself

`src/fridom/framework2/grid/decomposition/tensor.py`, `_bounded_ghosts` (line 844), still contains the `BC.NONE` branch (lines 897–903):

```python
if kind is BC.NONE:
    if n < 2:
        raise NotImplementedError("… needs at least two DOFs …")
    ghosts = [(1.0 + k) * dof(1) - float(k) * dof(2)
              for k in range(1, width + 1)]
```

`dof(1)`/`dof(2)` are the two nearest true cells. Ghost slot `k` is the **linear extrapolation** `(1+k)·u₁ − k·u₂`. Dirichlet gets the odd mirror `-dof(k)`, Neumann the even mirror. So on dev a BC-free walled edge is silently given a value the storage layer invents.

### What the BC tag does on dev

Commit `0041c30` ("BC-aware bounded staggering and Staggered→Inner on walls") added BC-tag *acceptance* but pointedly **kept BC-free codomains**. `finite_difference.py` `codomain` (lines 109–165) and `interp.py`:

> "BC-tagged bounded domains resolve through the same table; the codomain is the **BC-free** sibling (nodal outputs are BC-free — the input's tag governs only the ghost fill)."

So on dev, `mesh.nodal(Center, bc=DIRICHLET)` still diffs to plain `Inner` (BC-free); the tag only selects which mirror fills the ghost. There is no `Center → Outer` BC-structured codomain and no kind-flip.

The same commit's "Staggered→Inner" is in `grid.py` `_declared_space_resolver` (lines 1679–1698): `Dof.STAGGERED` resolves to `NodeSet.RIGHT` on a periodic mesh but **`NodeSet.INNER` on a bounded mesh**. The commit message explains why: "Right was wrong for the C-grid: the wall-normal velocity's boundary faces are not DOFs." So a declared staggered (velocity-point) field on a wall lands its codomain on `Inner` — the interior faces — and the wall faces are ghost slots filled by the machinery above.

### Reconstruction has a *second*, different mechanism: graded fallback

For finite-volume reconstruction (`CellAvg → Inner`, WENO), dev does **not** use extrapolation. `weno.py` refuses bounded axes ("bounded-axis boundary biasing … is designed-for and raises"; lines 19–20), and `src/fridom/framework2/grid/operators/fallback.py` owns the bounded signature via `Fallback` + `UpwindOne`:

- `graded_ladder(order)` builds `WenoReconstruction(order) → WenoReconstruction(order-2) → … → UpwindOne()`.
- `Fallback._apply_factor` runs the wide interior kernel, then at the `K = (O−1)//2` faces adjacent to each wall **overwrites** them with progressively narrower **interior-only** stencils (`_rung_value`, using `weno_reconstruct` on a window that "lies in the true region, so no exterior cell is read"). The wall-adjacent face drops all the way to `UpwindOne` (1st order, the single upwind cell).

This is "graded order reduction": the scheme reads no exterior value and needs no BC — it just loses formal order near the wall (order `2d−1` at distance `d`). It is loud in the sense that the order reduction is a static, compile-time index partition, not a silent invention — but it is a *different* boundary philosophy from the FD/interp extrapolation. Note `weno.py` lines 76–77 explicitly say the one-sided-stencil variant (R2) "is designed-for and **not yet a mode** here" — dev never built R2.

### Dev's full data path for the wall scenario

A BC-free `Center` field `u`, first derivative wanted at/near the wall:
1. `FiniteDifference(order=2)["y"](u)` resolves `Center → Inner` (always legal, no ghost read for the interior faces).
2. Anything that needs the **wall face** — a second derivative `Inner → Center`, a `Center → Outer` interp (`interp.py` `target=OUTER` variant: "boundary faces by one-sided extrapolation through the BC-free ghost"), or an order-4 near-wall face — reads a ghost slot, and `tensor.py` **silently fills it by linear extrapolation**.
3. No error is ever raised. The table stays *total*: every bounded FD/interp signature is defined.

So on dev, **a BC-free bounded edge gets an invented linear-extrapolated value**, and the operators that reduce order/accuracy near the wall are (a) FD/interp implicitly (via the extrapolation), and (b) reconstruction explicitly (via `Fallback`'s graded ladder).

---

## 3. The branch's design — R1 flip: refuse to invent, offer explicit closures

### The fill is removed

On the branch, `_bounded_ghosts` replaces the `BC.NONE` branch with:

```
BC-free sides return **None** — exterior values are undefined
…
if kind is BC.NONE:
    return None
```

Sync then **skips** BC-free walled sides — the ghost slots stay padded and unreadable. Dirichlet/Neumann/mixed mirror fills are unchanged; `BC.ROBIN` raises `NotImplementedError` pointing at the 2e data path.

### The codomain becomes a legality gate

The branch's `finite_difference.py` `codomain` (lines 99–165) has three bounded branches:

- **BC-free**: `Center → Inner`, `Outer → Center` stay; but `Inner → Center` (unless `boundary="one_sided"`) **raises `SpaceMismatchError`**: *"Inner → Center needs the wall faces, which a BC-free bounded space does not define — declare BC structure … or opt into `FiniteDifference(boundary='one_sided')`."* This is R1: the row un-seeds itself, and dispatch yields a `DispatchError` with a legality hint.
- **BC-structured**: from centers the derivative now lands on **`Outer`**, kind-flipped (`staggered_codomain` with `DIFF_KIND` mapping D→N, N→D). The §8 finding (i) explains: "the wall values are exactly computable from the domain's mirror fill, so they are true DOFs of the kind-flipped codomain (an Inner(Neumann) codomain would need the underdetermined vacant-Neumann fill instead); the return leg `Outer → Center` is then exterior-free." An `Inner` domain with Neumann/Robin structure raises (no grounded wall-slot fill).
- Products/nonlinear pointwise ops drop BC structure to the BC-free sibling (`as_bc_free()`), because the product of two odd extensions is even — keeping the tag would fill ghosts with the wrong sign (§8 "Design corollary").

`staggered_codomain` (`staggering.py` line 128) is the shared BC-aware tail: it maps each side's kind through `kind_map` and mints `mesh.nodal(target, bc=structure)`.

### The explicit one-sided closure (R2)

New machinery, **branch-only** (absent from dev's `operators/`):

- `stencil_kernels.py` `one_sided_weights(offsets, derivative)` (line 135): solves the exact moment system `Σⱼ wⱼ xⱼ^m = m!·[m==derivative]` over the rationals, so `p` nodes reproduce polynomials up to degree `p−1`.
- `staggering.py` `patch_one_sided_edges` (line 177): after the standard kernel runs, "every output whose window reaches beyond the true region of a bounded BC-free axis is recomputed from the `points` nearest **true** DOFs, with exact moment-solved weights at the output's actual offset. Interior outputs are untouched." FD uses `points = order + 1` (so order-2 → 3 points, quadratic-exact).
- `require_local_axis` (line 285) + `requirements(... layout="local")`: the patch writes static physical-edge indices, so the axis must be undistributed; the kernels stay shard-position-blind.

### The branch's full data path

Same BC-free `Center` field `u`:
1. `u.diff("y")` → `Center → Inner`: legal, identical numbers to dev.
2. A second `.diff("y")` (`Inner → Center`) on the default operator **raises `SpaceMismatchError`** with the hint. The chain stops loudly.
3. To proceed the owner must either (a) **declare BC** — then the derivative lives on the kind-flipped `Outer` codomain with mirror-filled, exactly-computable wall values; or (b) opt into `FiniteDifference(order=2, boundary="one_sided")` — then `patch_one_sided_edges` overwrites the wall-adjacent outputs with an explicit, visible one-sided stencil over true DOFs only.

---

## 4. Worked micro-example: `du/dx` at the wall

Take `f = y(1−y)` on `[0,1]`, `Center` field, `n = 4` cells (`dx = 0.25`; this reproduces the numbers cited in `bc_free_boundaries.md` so you can cross-check). True `f'(y) = 1 − 2y`, so **`f'(0) = 1.0`** at the wall; `f'' = −2` everywhere. Cell values: `u₀ = 0.109375, u₁ = u₂ = 0.234375, u₃ = 0.109375`.

The staggered order-2 diff stencil is `(−1, +1)/dx` across the two nodes straddling the face (`staggered_diff_weights(2) = (−1, 1)`, verified in `stencil_kernels.py`).

**(a) Dev's extrapolation fill — wall face `Center → Outer`.** Ghost cell at `y = −0.125` is `2u₀ − u₁ = −0.015625`. Wall-face derivative `= (u₀ − ghost)/dx = (u₁ − u₀)/dx = 0.125/0.25 = **0.5**`. True is `1.0`.
This is exactly the interior slope `(u₁−u₀)/dx`, which is `f'` at `y = dx`, not at `y = 0` — a **first-order (O(dx))** estimate of the wall derivative, wrong here by a factor of 2. **Silent.** Its composed failure: because the extrapolation forces the wall-face derivative to equal the first *inner*-face derivative (both `0.5`), a second `Outer → Center` difference cancels their difference to **0 instead of −2** at the boundary cell (the `bc_free_boundaries.md` counterexample). Caveat verified in code: the *other* routing, `Center → Inner → Center`, extrapolates the *derivative* field (which is linear for this `f`, so extrapolation is exact) and recovers `−2` correctly — this is the branch plan's "the `d²` case asserts an exact −2 only by accident of the extrapolation's single-consumption exactness." **The failure mode is data-dependent and invisible at the call site:** the fill silently linearizes at the wall, exact only when the field being extrapolated is itself locally linear there.

**(b) Dev's graded `Fallback`.** **Does not apply** to this case — `Fallback`/`UpwindOne` are `reconstruct`-kind (`CellAvg → Inner`), not FD on a `Center` field. If this were a WENO-5 flux reconstruction on a walled axis, the wall-adjacent face would be computed by `UpwindOne` (1st order, single upwind cell), the next by WENO-3, etc. — **formally reduced order, but reads no exterior value and raises nothing.** Loud only in the sense that the reduction is a static, documented index partition.

**(c) The branch's one-sided row.** `FiniteDifference(order=2, boundary="one_sided")`, `points = 3`. `one_sided_weights((0.5, 1.5, 2.5), derivative=1)` solves the 3×3 moment system → weights `(−2, 3, −1)/dx`. Wall-face derivative `= (−2u₀ + 3u₁ − u₂)/dx = 0.25/0.25 = **1.0**` — exact, because a 3-point stencil is quadratic-exact and `f` is quadratic. Formal order **2** (exact on quadratics, second-order convergence including the patched cells; §8 finding iv). The branch test `test_one_sided_fd_reopens_inner_to_center` pins exactly this: the default `Inner → Center` raises with a `"one_sided"` hint, and the one-sided `d²` returns `full(16, −2.0)`.

| design | wall `du/dx` (`f=y(1−y)`, n=4) | formal order at wall | failure mode |
|---|---|---|---|
| (a) dev extrapolation | 0.5 (true 1.0) | 1 (data-dependently exact) | **silent** O(1)/curvature-loss when consumed twice |
| (b) dev graded Fallback (reconstruction only) | n/a for FD; UpwindOne=1st order | reduced (1 at wall) | **quiet** reduced order, no error |
| (c) branch one-sided row | 1.0 (exact) | 2 | none for degree ≤2; else honest reduced-but-declared order |
| (branch default, no opt-in) | — | — | **loud** `DispatchError`/`SpaceMismatchError` + hint |

---

## 5. Consequences comparison

| axis | dev (fill kept) | branch (R1 flip) |
|---|---|---|
| Accuracy near walls (FD/interp) | 1st-order, silently linearized; O(1) if consumed twice through a curved field | either exact BC-structured `Outer` codomain, or explicit 2nd-order one-sided patch; default refuses |
| Accuracy near walls (reconstruction) | `Fallback` graded reduction (order `2d−1`) | same `Fallback` exists; R2 one-sided reconstruct "only when a concrete model needs them" (not built) |
| Energy/conservation | walled Sadourny advection (`4b4bc85`) achieves machine-precision energy conservation (2.6e-17…1.5e-16) via **structural zeros + Dirichlet-sibling adoption**, *not* via the extrapolation fill — it doesn't lean on the fill | R1 does not touch the structural-impermeability argument; the fill it removes is the FD/interp `BC.NONE` fill, orthogonal to the flux-loop zeros |
| Silent vs raised | BC-free exterior reads are **silent** (invented value) | BC-free exterior reads **raise** with a hint; nothing invented |
| Robin readiness | **no `BC.ROBIN` at all** (dev `bc.py` has NONE/DIRICHLET/NEUMANN only); no data path | `BC.ROBIN` present + per-side grounding; flux-form route designed; dynamic (α,g) path is stage 2e, gated on model 2.2/2.3 |
| Complexity | one universal filler keeps the table total; second, unrelated mechanism (`Fallback`) for reconstruction | more surface: per-side kind-flip codomains, `staggered_codomain`, `patch_one_sided_edges`, `one_sided_weights`, `require_local_axis`, `layout="local"` negotiation |
| Implication for walled model work on dev | the whole bounded operator table is **total** — walled advection / rigid-lid modules never hit a missing row; extrapolation is the safety net that keeps every bounded diff/interp defined | the R1 flip **inverts which path is supported**: bounded chains must declare BC or opt into one-sided; existing walled-advection code paths would need to route through declared BCs or `boundary="one_sided"` |

Note the tension the two docs record: dev's own `bc_free_boundaries.md` calls the fill "garbage twice"; the branch plan's §8 notes the current `d²` test passes only "by accident." Both agree the fill is *safe exactly once*; they disagree on whether a silent single-use scheme is acceptable.

---

## 6. What is salvageable regardless of the decision

- **`BC.ROBIN` enum + per-side grounding** — **design-independent.** It is a new enum member plus per-side shape/fill dispatch in `bc.py`/`tensor.py`/`staggering.py`; it does not depend on the R1 flip. Dev lacks it entirely, so it is pure addition. (The dynamic-value data path, stage 2e, is gated on the model layer under *either* design.) The branch tests `test_robin_fill_points_at_the_data_path` and `test_robin_field_creation_and_arithmetic_work` pin only the structural/creation behavior, not R1.
- **One-sided rows machinery** (`one_sided_weights`, `patch_one_sided_edges`, `require_local_axis`, `layout="local"`) — **design-independent as a capability.** dev's `weno.py` already calls the one-sided variant "designed-for." It can be added to dev as an *opt-in* without removing the fill. It becomes *load-bearing* only under R1 (where it is the sanctioned replacement for the default path); under dev it would be a convenience alongside the fill.
- **The boundary-closures test suite** (`git show framework2-boundaries:tests/framework2/grid/test_boundary_closures.py`) — **split.** Design-independent tests: `test_mixed_pair_fills_per_side`, `test_neumann_dirichlet_flipped_pair`, `test_*_d2_is_analytic_to_truncation` (Dirichlet/Neumann/mixed), `test_interp_preserves_the_kind_roundtrip`, `test_products_drop_bc_structure`, `test_linear_arithmetic_preserves_bc_structure`, the Robin creation tests. R1-dependent tests (would fail on dev by construction): `test_one_sided_fd_reopens_inner_to_center` (asserts the default `Inner → Center` **raises**), `test_one_sided_interp_grounds_the_outer_target`, `test_one_sided_variants_demand_a_local_axis`, `test_one_sided_is_an_override_row_never_a_default`, and the `Center → Outer` BC-structured codomain assertions.

Roughly: the BC-structured machinery, Robin structure, and the one-sided *capability* survive either way; only the *legality flip* (default BC-free exterior reads raise instead of extrapolate) is genuinely fork-specific.

---

## The decision, stated precisely

**Do BC-free exterior reads on a walled axis raise (branch/R1: the default `Inner → Center` / `Center → Outer` rows un-seed themselves, and the user must declare a BC structure or explicitly opt into `boundary="one_sided"`), or do they continue to be filled by a silent linear extrapolation (dev: every bounded FD/interp signature stays total, at the cost of a hidden first-order closure that is exact only when consumed once on locally-linear data)?**

Everything else — `BC.ROBIN`, per-side mirror fills, the one-sided stencil machinery, the BC-structured codomain rows — can coexist under either answer; the flip is the only mutually exclusive choice.

**Dev's implicit rationale for keeping extrapolation** (from the code, not a recommendation): `bc_free_boundaries.md` states plainly that "today BC-free is the only fully usable bounded path, which is *why* the extrapolation is load-bearing." Dev optimized for a **total** bounded operator table so the model layer never hits a missing row — commit `0041c30` deliberately keeps BC-free codomains and lets the tag "govern only the ghost fill," and the walled-advection work (`4b4bc85`) leans on structural flux-loop zeros rather than confronting the fill. The extrapolation is retained as the universal filler that keeps every bounded `diff`/`interp` defined; R1 would invert that, making BC-structured (or explicitly one-sided) the supported path and turning today's silent fills into loud dispatch errors.
