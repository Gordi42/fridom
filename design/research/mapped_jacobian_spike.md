---
status: frozen
date: 2026-07-16
---

# Mapped-Jacobian spike — which divisor grounds the wide rows

Research report (see [`README.md`](README.md) for status); answers the
Jacobian spike of
[`../plans/active/high_order_mapped_plan.md`](../plans/active/high_order_mapped_plan.md)
(§3) and decides the divisor for the option-(i) full lift. Run
2026-07-16 on cpu (x64). The spike script was throwaway (per the plan:
no production edits) and lived in a worktree; every load-bearing
spelling, kernel choice and result is inlined here.

## The question

Option (i) reconstructs in computational space and divides by an
order-matched discrete Jacobian. Two candidates for the divisor:

- **DA — discrete, same-row**: the same wide linear row that produces
  the face values, applied to the (seam-unwrapped) center-node
  coordinates; the flux-form width is the staggered difference of the
  reconstructed coordinate, `W_i = x̂_{i+1/2} − x̂_{i−1/2}`. For a
  collocated derivative row: `J_i = (D_ξ x)_i`.
- **DB — analytic**: the mesh map's autodiff Jacobian,
  `W_i = (dx/ds)(s_i) / n`.

Baseline **D0** is the current C0 behavior: the production divisor
placement, the staggered flux difference divided by the codomain
measure `grid.measure(mesh.center)` (two-point face-to-face width).

One charter question dissolved during recon: `grid.metric` is **not**
reachable from a stretched `MappedIntervalMesh` — it raises unless a
`CoordinateMapping` is attached (the terrain-following column path).
The analytic candidate for the refusal set is autodiff of
`mesh.coordinate_map`, which is unconditionally available
(`MappedIntervalMesh` requires a callable map; explicit node arrays
raise `TypeError`). So availability does not decide the choice; the
numerics below do.

## Setup

- 1D periodic, wavy map `x(s) = s + a·sin(2πs)/(2π)` on [0, 1) with
  a = 0.1 — the map the pre-guard 5→2 order-loss figures were measured
  on; robustness rerun at a = 0.25. Uniform `x(s) = s` as control.
- Constant advecting velocity c = 1 (the only regime where design
  order is observable; the varying-velocity composite is 2nd order on
  any mesh — plan §5). Profile `u₀(x) = sin(2πx)`; **semi-discrete**
  measurement: tendency of the initial data vs the exact −c·u₀′(x) at
  the centers (no time integration, so no time-error contamination).
- Scheme structure mirrors production `_FluxFormAdvection`: biased Shu
  face reconstruction of the state, flux = c·q̂ at the faces, staggered
  two-point difference, divided by the candidate width. c > 0 selects
  the production upwind branch (`Where(v+|v|, left, right)` → "left"),
  so all rows are left-biased.
- Norms: L∞ and measure-weighted L2; weno-5 additionally under the
  production critical-point mask (`|exact| > 0.3·max|exact|`). WENO
  eps = 1e-10 (production `WENO_EPS`). Order fits: pairwise log2 over
  N = 32…1024 plus a least-squares slope; errors below 5e-12 dropped
  as roundoff floor (the flux form divides O(h) by O(h); the uniform
  control's exact order-5 schemes bottom out at ~2.5e-12 at N = 1024,
  and the guard sits at 2× that).

Pre-flight checks, all exact: the spike's weno-5 kernel vs production
`WenoReconstruction` on a uniform mesh — max abs diff **0.0**
(bit-exact; halo 2/2, no roll); the spike's D0 width vs
`grid.measure(center)` on the wavy mesh — **0.0** (the baseline IS the
production divisor).

The load-bearing constructions:

```python
def linear_row(order, bias="left"):
    # optimal-weight combination of the Shu sub-stencil rows, taken
    # straight from weno_tables(order, bias) (coeffs, optimal,
    # offsets) => bit-identically the production linear rows:
    # upwind-3 left = (-1, 5, 2)/6, upwind-5 left = (2,-13,47,27,-3)/60
    tables = weno_tables(order, bias)
    row = np.zeros(order)
    for m, off in enumerate(tables.offsets):
        for j in range((order + 1) // 2):
            row[off + j] += tables.optimal[m] * tables.coeffs[m][j]
    return row

# DA: x is NOT periodic (it grows by L per period) — the halo uses the
# unwrapped extension, and the seam width gets L added back:
halo = np.concatenate([xc[-pad:] - L, xc, xc[:pad] + L])
xhat = apply(linear_row(order), halo)        # x̂_{i+1/2}
W_DA = xhat - np.roll(xhat, 1); W_DA[0] += L

# DB: analytic width from the mesh's own map
W_DB = jax.vmap(jax.grad(mesh.coordinate_map))(s_centers) / n
```

## E1 — free-stream residual (u ≡ 1, N = 128)

Exactly `0.0` for every scheme × divisor (upwind-3/5, weno-5 ×
D0/DA/DB) — not merely roundoff. Reconstruction of a constant is
exact, so the flux numerator vanishes before any divisor enters:
**the 1D map cannot show the multi-D free-stream failure**, confirming
the plan's de-risking claim. In 1D the identity that does discriminate
is E2.

## E2 — the linear-preservation identity (the discriminator)

Residual `max_i |(same-row flux difference of x)_i / W_i − 1|` on the
wavy (a = 0.1) mesh; collocated analog `max_i |(D_ξ x)_i / J_i − 1|`.

| row | N | DA | DB | D0 |
|---|---|---|---|---|
| upwind-3 (flux) | 32 / 64 / 128 | 0.0 / 0.0 / 0.0 | 6.3e-05 / 7.9e-06 / 9.9e-07 | 1.8e-04 / 4.5e-05 / 1.1e-05 |
| upwind-5 (flux) | 32 / 64 / 128 | 0.0 / 0.0 / 0.0 | 4.9e-07 / 1.5e-08 / 4.8e-10 | 1.8e-04 / 4.5e-05 / 1.1e-05 |
| FD-4 (colloc.) | 32 / 64 / 128 | 0.0 / 0.0 / 0.0 | 5.5e-06 / 3.4e-07 / 2.1e-08 | 7.0e-04 / 1.8e-04 / 4.5e-05 |
| FD-6 (colloc.) | 32 / 64 / 128 | 0.0 / 0.0 / 0.0 | 4.5e-08 / 7.1e-10 / 1.1e-11 | 7.1e-04 / 1.8e-04 / 4.5e-05 |

DA is **exactly zero at every resolution** — its divisor is the
numerator, so the composite derivative of the coordinate is exactly 1
by construction. DB misses at the design order of the row (O(h³) /
O(h⁵) / O(h⁴) / O(h⁶)); D0 at O(h²). This is the 1D shadow of the
discrete metric identity: the property that becomes free-stream
preservation where it actually bites (multi-D charts, mapped columns).

## E3 — convergence order (wavy a = 0.1; LS slopes, L∞ / L2w)

| scheme | D0 | DA | DB |
|---|---|---|---|
| upwind-3 | 2.50 / 2.49 (pairwise 2.90 → 2.00) | **3.00 / 3.00** | 3.00 / 3.00 |
| upwind-5 | 2.00 / 2.00 | **4.99 / 4.99** | 4.99 / 4.99 |
| weno-5 (masked) | 2.00 / 2.01 | **5.00 / 5.00** | 5.01 / 5.00 |
| weno-5 (unmasked) | 2.00 / 2.03 | 3.28 / 3.93 | 3.28 / 3.93 |

- D0 reproduces the documented trap: upwind-5 and weno-5 exactly 2,
  upwind-3 pre-asymptotic 2.5 trending to 2 (matching the pre-guard
  "5.0 → 2.0, 3.0 → ~2.6" figures).
- **DA and DB both restore design order and are numerically
  indistinguishable** — errors agree at truncation level and differ
  only at O(h^p). E3 cannot choose between them; E2 does.
- Robustness at a = 0.25: same picture (upwind-5 DA 4.98, weno-5
  masked DA 4.98, D0 → 2).
- The unmasked weno-5 slopes (~3.3 L∞, ~3.9 L2w) are the standard
  WENO-JS critical-point order loss, not a mapped-mesh effect: the
  mask *keeps* the periodic seam (|cos| = 1 at x ≈ 0) and recovers
  order 5, so the loss sits at the interior critical points and the
  seam unwrap is independently confirmed. It hits DA and DB
  identically and is uniform-mesh-visible in principle (the uniform
  control shows ~5 under the same mask); relevant to the option-(ii)
  trigger, not to the divisor choice.

## E4 — collocated FiniteDifference analog (L∞ LS slopes)

| row | D0 | DA | DB |
|---|---|---|---|
| FD-4 | 1.99 | **3.99** | 3.99 |
| FD-6 | 2.00 | **5.97** (pairwise 5.93 → 6.00, then floor) | 5.97 |

The same divisor retires the `FiniteDifference(order > 2)` refusal.

## E5 — controls

Uniform mesh: the three divisors coincide to machine precision and
every scheme shows design order — upwind-3 3.00, upwind-5 5.00 (floor
at N = 1024), weno-5 masked 5.09–5.10. With the kernel cross-check
bit-exact, the mapped measurements rest on production rows.

## E6 — integrated sanity (optional, run)

upwind-5 + DA, RK4 at CFL 0.1, one full period: L∞ 3.49e-05 → 1.10e-06
→ 3.45e-08 (N = 32/64/128), orders 4.99 / 5.00. Stable and consistent;
E3 is the order result (RK4 time error would contaminate finer grids).

## The answer: DA — the same-row discrete Jacobian

Both candidates restore design order; only DA satisfies the discrete
metric identity, and it does so exactly, at every order and
resolution, by construction. DB silently misses it at O(h^p) — in 1D
that costs nothing observable, but it is precisely the invariant that
becomes free-stream preservation in the multi-D and mapped-column
settings this work targets, and "silently" is the operative failure
mode the guards exist to prevent. Two practical properties seal it:

1. **The widths are static.** The weno-5 flux used nonlinear weights
   while its DA width used the *linear* row — and design order still
   restored (E3) with free-stream exact (E1). So the width never needs
   the nonlinear weights: one data-independent field per
   (space, order, bias), materialized once like the measure fields.
2. **No analytic map needed.** DA is built from node coordinates
   alone, so it generalizes to any future mesh construction; DB exists
   only where a differentiable map callable does.

Full-lift scoping notes surfaced by the spike (inputs to the plan, not
spike scope):

- **Branch consistency under upwind selection.** The spike ran a
  single upwind branch (c > 0). With sign-varying velocity the width
  must use the same branch as the flux face-by-face: reconstruct x̂
  with both biases (two static face fields) and `Where`-select with
  the same predicate as the flux before differencing — the identity
  then holds exactly per cell with purely static inputs.
- **Order 2 is genuinely the boundary of the old rule.** The same-row
  width of the order-2 scheme is the centered center-difference, not
  the face-to-face measure; both are 2nd-order Jacobians, which is why
  "divide by the measure" was exactly grounded at order 2 and only
  there.
