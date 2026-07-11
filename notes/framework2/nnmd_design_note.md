# NNMD design note (P0 of nnmd_rewrite_plan.md)

**Status: draft for owner review, 2026-07-11.** Scope: formulation +
user surface. Owner reviews these two; framework2 internals are not
review material (standing preference).

## 1. Formulation

### 1.1 Setting

System on the state space with energy metric M:

```
∂t φ = L φ + N(φ),      N(φ) = B(φ, φ)
```

with L anti-self-adjoint under M (iML Hermitian — the property every
framework2 eigenmode tier is built on) and B the symmetric bilinear
form of the quadratic term, computed by polarization

```
B(z1, z2) = ½ ( N(z1+z2) − N(z1) − N(z2) ),
N(z) = model.variant(term_filter=nonlinear).tendency(z)
```

(3 tendency evaluations; 1 when z1 is z2).

**Inputs**: a conjugation-closed spectral selection splitting the
modes into *slow* (projector V) and *fast* (W = I − V), with

- (i) every fast mode satisfies |ω| ≥ Ω > 0 (so `L_w = LW` is
  invertible on ran W with ‖L_w⁻¹‖ ≤ 1/Ω);
- (ii) slow rates — both the slow linear frequencies |ω_slow| and the
  advective rate — are ≤ εΩ with ε ≪ 1.

The slow set is NOT required to be stationary (Lφ ≠ 0 on ran V is
allowed): β-plane Rossby bands, topographic-Rossby bands under
varying c², etc. qualify via (ii).

### 1.2 The recursion (unscaled, generalized)

Balanced state of z at order N, with base point v = Vz:

```
z_b = Σ_{n=0}^{N} φ_n ,   φ_0 = v ,   Vφ_n = 0 for n ≥ 1
```

Slow derivatives of the base point (see closure choice, §1.4):

```
φ_0^(1) = L φ_0 + V B(φ_0, φ_0)
φ_0^(k) = L φ_0^(k−1) + V Σ_{m=0}^{k−1} C(k−1,m) B(φ_0^(m), φ_0^(k−1−m))
```

Fast corrections, doubly indexed (series order n, derivative order k):

```
φ_{n+1}^(k) = L_w⁻¹ W [ φ_n^(k+1) − Σ_{m=0}^{k} C(k,m) I_n^(m, k−m) ]
I_n^(a,b)   = Σ_{j=0}^{n} B(φ_j^(a), φ_{n−j}^(b))
```

for n ≥ 0, where φ_n^(k+1) for n = 0 is the slow recursion above.
The balanced state uses the k = 0 column. Well-founded exactly like
v1: order N needs φ_0^(N) at the deepest point; ~N²/2 distinct
bilinear evaluations.

**Only three operator ingredients**: V/W (projectors), L_w⁻¹ (inverse
on the fast subspace), and L on the slow subspace (for the Lφ_0^(k)
terms — equivalently `fr.linearize(model).tendency`, or the spectral
applicator of §2.C). No per-mode ±branch tracking anywhere; the ±ω
pairing is inside L_w⁻¹, which is real-safe because the selection is
conjugation-closed. Fast modes with structurally zero ω cannot enter
W by assumption (i) — the constraint/Leray family is excluded by the
selection, with a structural guard, not a 1/ω floor.

### 1.3 Collapse to v1 (verified)

If LV = 0 (stationary slow space, the v1 setting), the Lφ_0^(k) terms
vanish and the recursion is the stationary-eigenspace vault note.
Componentwise on eigenmode s with `∂t z = −iA_v1 z`: L = −iA_v1 has
eigenvalue ∓iλ on s = ±1, so L_w⁻¹ acts as ±i/λ — exactly the v1
`-1j * sign * one_over_omega` factors, derivative recursion included.
Same method, coordinate-free.

### 1.4 Open math point — slow-tendency closure (owner Option D)

The slow recursion in §1.2 drives all derivatives along the
*leading-order* slow flow v̇ ≈ Lv + VB(v,v). This is what v1 and the
stationary note do (v1 structurally has no z_{0,n≥1,k} slots). The
exact slaving (Warn et al.) closes v̇ with the wave feedback
VB(φ, φ) ⊃ 2VB(v, s(v)) + ..., which enters φ_n^(k) two orders up —
i.e. it may be REQUIRED for the residual slope ε^{N+1} at N ≥ 3, or
may only change constants. This is precisely testable in the toy
harness (T1): if the leading-order closure saturates at slope ~3, the
order-consistent closure is needed; its cost is extra bilinear terms
I_n with slow-slot entries φ_0-corrections (same recursion shape,
z_{0,n,k} slots reinstated).

R1 literature verdict (2026-07-11, `nnmd_literature.md` §2): Warn
et al.'s superbalance hierarchy is order-consistent by construction;
Tribbia (1979) shows the term the leading-order closure misses is
exactly an O(ε²) slow drift; nobody has tested leading-order closure
at N ≥ 3 (Chouksey et al. used model finite-difference derivatives at
n ≥ 2, silently full-closure). Expect the empirical gate to trigger
at N ≥ 3 — order-consistent is likely the shipping default, with the
toy slopes as confirmation.

### 1.5 Telescoping variant

Upgraded after R1 (2026-07-11): the `A^{−(n+1)}` reorganization is
**published prior art — Tribbia (1984)** — and in Temperton's
implicit form it removed residual large-scale low-frequency modes
that the first-order direct scheme leaves behind (see
`nnmd_literature.md` §5). Serious contender, not a curiosity: the
toy-harness M2/R2 comparison decides on data whether it ships as a
second scheme.

### 1.6 Diagnostics

Two cheap residuals, both exposed:

- `residual_series(z)` — ‖φ_{N+1}‖_M / ‖z_b‖_M (next-order term; the
  T2 ε-slope quantity).
- `residual_fast(z)` — ‖W F(z_b) − ds[v̇]‖_M with the slaved
  prediction by finite difference (T3 differential residual).

## 2. User surface — DECIDED (owner, 2026-07-11)

- **A — name & placement: `fr.transforms.BalanceExpansion`.** The
  NNMD acronym is dropped from the class name; same home as
  `OptimalBalance`; takes the model and dispatches eigenmode tiers
  internally (`fr.eigenbasis`/`from_model`). No package factories.
  Docstring keeps the literature anchor ("nonlinear normal mode
  decomposition / initialization, Machenhauer 1977, Warn et al. 1995,
  Eden et al. 2019") so the method stays findable by its established
  name.
- **B — slow space: family string or predicate**, same grammar as
  `eb.projector`: `slow="vortical"` (default) or
  `slow=lambda omega, labels: ...` (the β-band case). The fast set is
  the complement minus structural-zero families (constraint/Leray
  excluded automatically). The operator-injected core stays private
  (toy harness uses it internally; can be promoted later without
  breakage).
- **C — f(L) applicator: general `em.function(f, sel)` /
  `eb.function(f, sel)`** returning a StateTransform applying
  `Σ P_s · f(ω_s)` (analytic tier) / `Q diag(f(ω)·mask) Q^H M`
  (channel tier). BalanceExpansion uses `f = ω ↦ 1/(iω)`; free
  byproducts: spectral filters, `exp(iωt)` propagators, the deferred
  `omega_field`. Structural-zero guard: singular f on a selected
  zero-ω mode is an error, not a floored division.
- **D — closure: empirical gate.** Implement the leading-order
  closure (v1 bookkeeping); the toy harness measures both convergence
  slopes; promote the order-consistent closure (same recursion shape)
  only if leading-order saturates below the advertised order.

### Decided sketch

```python
bal = fr.transforms.BalanceExpansion(
    model, order=3,
    slow="vortical",              # family | predicate
    nonlinear=~fr.terms.linear,   # quadratic term filter (linted)
    at_time=0.0,
)
z_bal = bal(z)
r = bal.residual_series(z)
```

Dropped from v1 (decided): `use_model` FD-derivative branch,
`enable_dealiasing` flag (dealiasing wired for real through
`PadFactor` in the nonlinear evaluation, not a flag that does
nothing).
