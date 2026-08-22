# Upwind / WENO advection on a thin walled axis (`nz = 1`)

Status: **resolved 2026-08-22** (`fix/thin-walled-axis-advection`,
`done.md` "Honest halo on short walled axes"). The investigation below
is kept as written; the route taken is none of its A/B/C but the
owner's sharper diagnosis, recorded here first.

## Resolution

The declared `order // 2 + 1` reach of the biased family along a
walled axis is a lie whenever every face of that axis is a ladder
face: a graded kernel then reads no ghost slot at all (R1). So the fix
is an **honest requirement** — `graded.fully_patched(n_cells, rows,
shift)` consulted by every graded kernel's `requirements` and by
`UpwindAdvection.bind`'s explicit demand — plus the ladder's per-face
assignment to the nearer wall (mechanism 1 below, the only genuine
bug), with the shared tails skipping their halo guard and an unfitting
window on a fully patched axis. No fill fold (mechanism 2 is never
reached once nothing asks for the wide halo), no elision. The one-cell
hydrostatic column runs WENO-3/5 on **three** z storage layers (the
flux divergence's width 1, as the centered scheme) and reproduces the
six-layer run to 5.6e-16. What remains of the storage question (3
layers vs the linear model's 1) is the flux divergence reading two
structural-zero wall slots, a different and smaller lever.

## The question

`hy.Model(..., advection=fr.model.modules.WENOAdvection(order))` on the
geostrophic-adjustment grid `shape=(nx, ny, 1)`,
`periodic=(True, False, False)` refuses to assemble:

```
NotImplementedError: WENOAdvection(order=3) needs at least 4 cells on
every walled axis (the graded near-wall ladder must fit between the two
walls), got (('z', 1),). Use a coarser order, more cells, or
CenteredAdvection
```

`CenteredAdvection` assembles and steps on the same grid, and a
*periodic* one-cell axis works under every scheme. Physically a walled
one-cell axis is the simpler object — it has no interior face at all —
so the refusal is structural, not numerical. Two independent mechanisms
are involved; WENO itself is not one of them.

## Mechanism 1: the two walls' ladders are patched independently

`UpwindAdvection._check_walled_extent` (`model/modules/advection.py`,
`bind`) demands `min_cells(order) = order + 1` cells on every walled
axis (`spatial/operators/graded.py`). The number encodes two
constraints of `graded.apply_graded_walls`: the widest rung's window
must fit the lattice, and the `K = order // 2 + shift` reduced faces
of the **left** wall must not overlap those of the **right** wall
(`n_faces >= 2K`). The second one is an artifact of the implementation:
`patch` runs once per wall with the rung chosen by the distance to
*that* wall only, and the right wall's patch runs last. Where the two
sets of faces overlap, the face keeps the right wall's rung, which may
be wider than the distance to the left wall allows — an exterior read,
i.e. an R1 violation (`boundary_plan.md`).

NaN-poison probe (the `test_graded_rows_read_no_exterior_value` idiom:
every ghost slot NaN, ghosts claimed valid) on a `Center` operand,
linear weights, `wall="upwind1"`, `_check_walled_extent` bypassed:

| cells `n` | order | left bias | right bias |
|---|---|---|---|
| 2 | 3 | finite | finite |
| 3 | 3 | finite | finite |
| 4 | 3 | finite | finite |
| **3** | **5** | **NaN at face 1** | finite |
| 4 | 5 | finite | finite |
| 5, 6 | 5 | finite | finite |

So the collision is real exactly where `min_cells` says it is
(`n = 3 < 6` at order 5: the right patch writes face 1 with the
distance-2 order-3 rung, whose left-biased window starts at cell −1),
and order 3 is clean down to `n = 2` because its single reduced face
per wall cannot collide in a harmful way (both walls agree on the
order-1 rung).

At `n = 1` there are **zero** interior faces. The patch loop then
writes only into the output field's ghost slots (the
`dynamic_update_slice` clamps), which nothing reads before the next
sync refills them — harmless, and the proof of concept below confirms
it bit for bit. So `n = 1` is not blocked by mechanism 1 at all.

**Correct rule.** The rung at a face must be the narrowest of the two
walls' ladders: order `min(order, 2 d_L - 1, 2 d_R - 1)` (biased),
size `min(size, 2 d_L, 2 d_R)` (centered), with `d_L`, `d_R` the
distances to the two walls (`d = 1` wall-adjacent). Both patches then
compute the same value in the overlap, every window stays inside the
true DOFs for either bias (checked by the offset arithmetic of
`biased_offset`), and the ladder degenerates gracefully: one face gets
the bottom rung, zero faces means no patch. `n_faces` is static (the
mesh), so this is still a compile-time index partition.

## Mechanism 2: the bounded halo fill cannot reflect past the far wall

With the guard bypassed, `nz = 1` fails one layer deeper, at the first
dry evaluation of the term:

```
NotImplementedError: the bounded halo fill of width 2 reaches deeper
than the 0 DOFs along the axis
```

(`spatial/decomposition/tensor.py`, `_axis_map`; its multi-device twin
`_bounded_ghosts.dof` raises the same way). The hydrostatic vertical
leg forms its flux on the `Inner` faces (the `Outer -> Inner`
restriction of the diagnosed `w`); on a one-cell axis that space has
**0 DOFs**, and the biased scheme negotiates a halo of `order // 2 + 1`
(2 for order 3, 3 for order 5) on every advecting axis. The
Dirichlet-structured fill maps ghost slot `k` to the odd reflection of
true DOF `rank(k)` about the near wall; once `rank > n` the reflection
has passed the *far* wall and the code gives up. The same limit bites
WENO-5 on `nz = 2` (width 3 against `1 + 1`).

This is precisely the case the **periodic** fill already handles: its
branch wraps modulo `n` "whenever the halo is *wider* than the axis —
a thin periodic direction (`n = 1` is the flat 2-D direction) under a
wide stencil" (comment in `_axis_map`). The bounded analogue is the
reflected continuation: when the reflected rank overshoots the far
wall by `k'`, reflect again about the far wall with *its* own
`(kind, distance)` rule (`_ghost_slot`), accumulating the sign, and
keep going until the rank lands inside `1..n` or hits a `_VACANT`
zero slot. The odd/even extension about both walls is periodic
(period `2(n + 1)` for dropped-Dirichlet node sets, `2n` for the cell
frame), so the walk terminates; it is the unique extension consistent
with both boundary conditions, and it is the bounded axis's
counterpart of the modular wrap:

- `n = 0`, Dirichlet `_VACANT` both sides (the `Inner` flux space on a
  one-cell axis): every slot is exact zero — the odd extension of the
  empty function.
- `n = 1`, `_VACANT` both sides: `[0, -q, 0 | q | 0, -q, 0]`.
- `n = 1`, Dirichlet `_OFFSET` (cell-centred odd reflection):
  `[-q, +q, -q | q | -q, +q, -q]`.
- A BC-free far side (`BC.NONE`) has no exterior values, so a fold
  through it stays undefined and should keep raising.

The fill is a host-side static index map (jit constants), so the fold
has no trace or autodiff footprint.

## Evidence: proof of concept

Scratch script `poc_weno_nz1.py` (session scratchpad, not in the repo)
monkeypatches the fold into `tensor._axis_map` and bypasses
`_check_walled_extent`, nothing else. On the example grid
(`192 x 64 x 1`, 6000 x 2000 km, depth 40 m, `buoyancy=None`,
explicit free surface, AB3, 72 h):

| run | finite | eddy centre | max \|η − linear\| | max \|η − same scheme on 6 identical layers\| |
|---|---|---|---|---|
| WENO-3, `nz = 1` | yes | 0.5064 m | 1.37e-3 m | **5.6e-16 m** |
| WENO-5, `nz = 1` | yes | 0.5064 m | 1.35e-3 m | **5.6e-16 m** |

The six-layer runs are the workaround used earlier today (barotropic,
so the vertical leg is exact); agreement to roundoff shows that the
fold is the only thing standing between the one-cell grid and WENO,
that the vacuous ladder at `n_faces = 0` is harmless, and that the
surface closure (`surface_flux=None` → on for the hydrostatic `Outer`
`w`) composes correctly with the biased family on a one-cell column.
The 1.4 mm gap to the linear run is physics (`η/H = 2.5 %`,
`Ro ≈ 3e-3`), identical for both orders and for the centered scheme.

## Proposed change

Three source files, all host-side structure:

1. `spatial/decomposition/tensor.py` — the fold in `_axis_map`
   (single-shard map) and in `_bounded_ghosts` / `_ghost_values`
   (multi-device physical-boundary fill); the two are documented twins
   and must not disagree. Keep the raise for a BC-free far side.
2. `spatial/operators/graded.py` — `apply_graded_walls` takes the
   per-face minimum over both walls' rungs and returns `interior`
   untouched when the axis has no interior face; `min_cells` drops to
   1 (or is deleted with its callers). Under `shard_map` the left
   block's faces are global faces `1..K` and the right block's
   `t_out - d + 1` sit at global distance `n_faces - d + 1` from the
   left wall, so the rule needs only the static global face count — but
   a wall block narrower than `K` faces would not hold all the faces
   it must patch. Thin axes are never sharded in practice; either keep
   a taught refusal for "sharded and `n_faces < 2K`" or prove
   `patch_physical_ends` masks it correctly.
3. `model/modules/advection.py` — remove (or narrow to the sharded
   corner) `_check_walled_extent`; the three docstring statements that
   every walled axis needs `order + 1` cells follow. The FV family
   (`operators/fallback.py`), `_SelectedFaceReconstruction` and
   `_CenteredFaceInterpolation` all run the same tail, so they are
   covered by (2).

Tests (mirrored files):

- `tests/model/modules/test_advection_walls.py`:
  `test_walled_axis_too_short_for_the_ladder_is_taught` flips into
  "a short walled axis reads no exterior value" — the poison gate
  parametrized over `n ∈ {1, …, order + 1}` × order × bias × node set
  (the `n = 3`, order-5, left-bias cell is the regression); plus a
  cheap "hydrostatic WENO on `nz = 1` equals the six-layer column to
  roundoff" (`16 x 8 x 1` vs `16 x 8 x 6`, a few steps); plus the
  policy autodiff regression through a short `nz = 1` WENO run.
- `tests/spatial/decomposition/test_tensor.py` (and
  `test_tensor_flat_axis.py`): the fold patterns above for Dirichlet
  `_OFFSET` / `_VACANT` and Neumann `_OFFSET` / `_MEMBER`, the all-zero
  `n = 0` case, and single-shard vs multi-device agreement under
  forced devices.
- `tests/spatial/operators/test_graded.py`: the overlap rule against a
  hand-built narrowest-rung reference, and the `n_faces = 0` no-op.

## Storage inflation, and what the fold does *not* buy

The fill's cost is nil either way: fold and zero-fill are both a static
`(src, neg, zero)` index map evaluated inside the same sync, and on
this model their *values* are never even consumed (the one-cell flux
leg has no interior face; the flux divergence reads only the wall
slot, which is zero under both; the `Outer` `w` is restricted, not
stenciled). The cost that matters is the **halo storage the biased
scheme negotiates on the thin axis**, which the fold leaves untouched.
Measured on the one-cell grid (48 x 16 x 1, `u` storage):

| scheme | negotiated halo (x, y, z) | `u` storage | z layers |
|---|---|---|---|
| linear (`advection=False`) | 1, 1, 0 | (50, 18, 1) | 1 |
| `CenteredAdvection` | 1, 1, 1 | (50, 18, 3) | 3 |
| Upwind-3 / WENO-3 + fold | 2, 2, 2 | (52, 20, 5) | 5 |
| WENO-5 + fold | 3, 3, 3 | (54, 22, 7) | 7 |

Every 3-D field and every x/y kernel pass carries those layers.
`thin_axis_halo_investigation.md` §12.1 measured exactly this
inflation on the periodic thin axis before elision: **2.2x** wall time
for 3 -> 1 layers and **7.5–7.7x** for 7 -> 1 (WENO-5), bandwidth-bound
(0.13–0.44 FLOP/byte), so the byte ratio is the wall-time ratio and the
figure carries to gpu.

**Reconciliation with that record's §5 ("do not generalize the bounded
fill").** §5 evaluated the two-wall fold as a route to *elision* and
rightly found it buys nothing there, because a bounded one-cell axis
carries 0, 1 or 2 DOFs per factor and its ghosts are a BC extension,
not copies — the `is_flat` repeat rule cannot apply. This note uses the
fold for a different purpose: *running at all* with the halo kept. The
two conclusions are compatible; §5's "do not generalize" should be read
as "not as a flat predicate".

Three routes, by cost and payoff:

- **A. Fold + per-face ladder rule (this note's proposal).** Correct,
  local, ~three files. Storage stays inflated: 5 / 7 z layers on a
  one-cell column, i.e. the §12.1 multiples relative to the linear
  model.
- **B. A plus an empty-codomain short-circuit.** On the one-cell axis
  the biased z leg maps `Center(1) -> Inner(0)`: zero output DOFs. If
  the shared tails (`staggering.apply_staggered`,
  `reconstruct.apply_fv_staggered`) returned the empty field without
  running the kernel when the codomain has no true DOF along the
  applied axis, nothing along z would read a ghost, and the z halo
  demand could drop to the flux divergence's width 1 — **3 layers**,
  as the centered scheme, a 1.7–2.3x saving over A. Two caveats: the
  halo tracer records an operator's *declared* reach
  (`HaloTracer._trace_apply` -> `_record_reach(self._reach(op))`), so it
  needs the same exemption it already has for `ConstantSpace` factors;
  and at `nz = 1` the fold is then no longer needed (width 1 admits,
  §12.9's `width <= n_cells` law), though WENO-5 on `nz = 2` still is.
- **C. Bounded flat-axis elision.** Generalize `staggering.flat_widen`
  from `jnp.repeat` of one slot to *applying the fill's static map* to
  the 0 / 1 / 2 stored slots. That is exact by construction (it rebuilds
  precisely the non-elided storage), so §5's objection to identity/zero
  *rules* does not apply to it — but §12.4's traps do: the per-factor
  DOF split 0 / 1 / 2 on one mesh axis against the per-axis halo that
  "seven storage-index sites" assume, the empty staggered codomain, and
  the kernel-closure audit (`flat_captures`). It is a campaign of the
  shipped periodic elision's size (§12.7), with the full 7.7x payoff on
  WENO-5 relative to A (~2.3x relative to B).

Recommendation: A is the correctness fix; B is the cheap performance
fix if WENO on one-cell columns is to be *used* rather than merely
allowed; C only if such columns become a real workload.

## Decisions for the owner

1. **Fold vs. zero-fill beyond the first reflection.** Model results
   are identical either way — the graded family discards every row
   that reads those slots and the flux divergence reads only the wall
   zero — so this is a semantics choice. The fold is the principled
   "same as periodic" answer and keeps the ghost contents meaningful
   for any future non-graded consumer; zero-fill is a two-line change
   that documents the slots as don't-care.
2. Keep a guard for the sharded thin-axis corner, or not.
3. Whether to do this now: the geostrophic-adjustment example does not
   need it (linear is physically right there), so the motivation is
   completeness of the biased family on one-cell columns, which the
   centered family already has.
