---
status: frozen
date: 2026-07-11
---

# NNMD literature sweep (R1 of nnmd_rewrite_plan.md)

**Status: research report, 2026-07-11.** Sources verified against the
full texts of Temperton 1988, Warn et al. 1995, Vanneste 2013,
Chouksey et al. 2023, Tribbia 1979, Errico 1983 (PDFs read in full or
in relevant part); everything else from abstracts/secondary citation.

## 1. Implicit NMI in bounded domains (Temperton line)

**Findings.** Temperton, "Implicit Normal Mode Initialization", MWR
116, 1013–1031 (1988), doi:10.1175/1520-0493(1988)116<1013:INMI>2.0.CO;2
is the closest published relative of our projector formulation.
Machenhauer's per-mode update `Δc_G = iΛ_G⁻¹ δ_t c_G` is written
compactly as `Δx = i E_G Λ_G⁻¹ E_Gᵀ δ_t x` (his 2.15); premultiplying
by `A = EΛEᵀ` gives the **fundamental implicit equation (3.5)**

```
A Δx = i E_G E_Gᵀ δ_t x        (our  L_w Δx = W F(x),  Δx ∈ ran W)
```

valid even when A is singular (slow modes = ker A; the RHS is
orthogonal to it; uniqueness by constraining Δx to the fast
subspace). So implicit NMI needs exactly two capabilities — his
words: (1) split a tendency vector into slow/fast parts, (2) solve
one linear system `AΔx = y`. That is precisely our `W` and `L_w⁻¹`.

How he realizes them without eigenmodes, on a **wall-bounded**
finite-element domain (Staniforth–Mitchell quasi-hemispheric model,
solid wall, BCs ψ=0, ∂χ/∂n=0):

- **W implicitly**: two structural properties read off the linearized
  equations — slow modes are stationary, nondivergent, and satisfy
  the linear balance `∇²φ_R = 𝓕ψ_R` (`𝓕 = ∇·f∇`); fast modes have
  zero *linearized PV* `Z' = Φ∇²ψ − 𝓕∇_n⁻²(φ/m²)`. Divergence
  tendency is entirely fast; the fast parts of (ζ, φ) tendencies
  follow from these two conditions via elliptic solves (his
  5.15–5.20). Our analogue: ran V / ran W characterized by family
  membership + constraint-family exclusion, not by mode loops.
- **L_w⁻¹ implicitly**: solving (3.5) reduces to variable-coefficient
  **Helmholtz problems** `(∇² − f²/(m²Φ))(·) = rhs` — the operator is
  the spatial symbol of ω² on the gravity branch — solved by
  Concus–Golub CG with a constant-coefficient Helmholtz kernel.
- **Tendencies by one forward model timestep** (Andersen 1977 /
  Williamson & Temperton, MWR 109, 744–757, 1981 trick), with the
  caveat that semi-implicit time-averaging distorts the fast
  tendencies and *drastically slows convergence* unless Δt is small
  (his Fig. 3: Δt=600 s stalls, Δt=60 s ≈ Δt=6 s).
- **Walls**: §5b–d prove that with the geostrophic boundary
  conditions (∂φ/∂x = fV etc.) the boundary fluxes in the elliptic
  problems cancel and — crucially — the **slow and fast subspaces are
  orthogonal in the energy inner product**
  `⟨x₁,x₂⟩ = ∬ {φ₁φ₂/m² + Φ(U₁U₂+V₁V₂)}` (his 5.40–5.52). This is
  exactly framework2's M-orthogonality of eigenmode families on
  walled domains.
- Convergence: BAL (Σ|dc_n/dt|² over fast modes) drops 4 orders of
  magnitude in 2–3 iterations, then stalls; height-field fast
  oscillations go from tens of m to ~1 m.

Context: Temperton, MWR 117, 436–451 (1989) extends to spectral
models. Predecessors doing the same elliptic-inversion NMI more ad
hoc: Bourke & McGregor, MWR 111, 2285–2297 (1983) (vertical-mode NMI,
horizontal elliptic solves, "Filtering Condition B"
`δ_t D = δ_t(fζ̃ − ∇²φ) = 0` — which Temperton shows his scheme
reproduces exactly, eq. 4.36); Juvanon du Vachat, MWR 114, 2478–2487
(1986) (normal modes as eigenfunctions of an elliptic operator,
rigorous derivation of Bourke–McGregor); Brière, MWR 110, 1166–1186
(1982) (limited-area NMI with sine modes); Ballish (PhD thesis, Univ.
Maryland, 1980) (physical-space NMI ↔ bounded-derivative link). Daley,
*Atmospheric Data Analysis* (CUP, 1991) is the standard textbook
treatment of NMI including the implicit form.

**Implication.** Our formulation is the coordinate-free restatement of
implicit NMI; nothing in it is unpublished except the combination with
a non-stationary slow space (§6). Temperton's published caveats map to
requirements: (i) his equivalence proof *assumes stationary,
nondivergent slow modes* (LV = 0) — our Lφ₀ terms go beyond it; (ii)
consistent BCs are what make V/W M-orthogonal on walls — in framework2
the channel eigenbasis provides this by construction, replacing his
hand-built Helmholtz machinery with `Q diag(f(ω)) Qᴴ M`; (iii) his
Δt-sensitivity is an argument for our analytic B over FD tendencies.

## 2. Slow-tendency closure in slaving expansions

**Findings.** Warn, Bokhove, Shepherd & Vallis, QJRMS 121, 723–739
(1995), doi:10.1002/qj.49712152313, read in full. Setup: `∂s/∂t =
S(s,f;ε)`, `∂f/∂t + Γf/ε = F(s,f;ε)`, |ω_ν| ≥ 1 on ran Γ. Slaving
`f = U(s;ε)` inserted into the fast equation *using the slow equation*
gives the **superbalance equation (their 16)**:

```
𝔅(U) ≡ ε 𝒯 S(s, U; ε) + Γ U − ε F(s, U; ε) = 0,   𝒯 = ∂U/∂s.
```

The chain-rule term `𝒯 S(s,U;ε)` contains the FULL slow tendency,
including the wave feedback of the slaved field into S. Their modified
expansion (19)–(23) expands only f and generates the hierarchy

```
O(εⁿ):  f = εΓ⁻¹{F(s,f;ε) − 𝒯S(s,f;ε)}|_{O(ε^{n−1})},
```

i.e. the closure is **order-consistent by construction**: the slave
relation at order n differentiates along `ṡ = S(s, f)` with f
truncated at order n−1, not along the leading-order flow. The
iterative alternative (their 24, = Allen 1993's iterated geostrophic
intermediate models, JPO 23) is `U_{n+1} = εΓ⁻¹{F(s,U_n) − 𝒯_n
S(s,U_n)}` — also order-consistent; expansion and iteration differ
only in higher-order cross-terms (§4d: one extra O(ε²) term) and
"have the same formal accuracy".

Accuracy claims: **none proven**. The series is required only to be
asymptotic (`‖u − Σᵐ εᵏu⁽ᵏ⁾‖/εᵐ → 0` as ε→0); they explicitly note
"for a fixed but small ε it often happens in practice that the error
decreases with m up to some order and thereafter increases" (optimal
truncation), and point to Kreiss's bounded-derivative method for the
rigorous results. Exact slaving is called unlikely (Lorenz, JAS 43,
1547–1557, 1986; Vautard & Legras, JAS 43, 1986; Warn & Menard 1986).
Lorenz, JAS 37, 1685–1699 (1980) coined the superbalance equation.

Originals: Machenhauer, Beitr. Phys. Atmos. 50, 253–271 (1977)
iterates with the forcing held constant (`r_n` frozen), i.e. **drops
𝒯S entirely**. Baer & Tribbia, MWR 105, 1536–1539 (1977) and Baer,
Beitr. Phys. Atmos. 50, 350–366 (1977) use a two-timescale expansion;
Tribbia, MWR 107, 704–713 (1979) shows (his 2.6b) that the O(ε²) term
is *precisely* the slow drift `dy⁽¹⁾/dT` of the slaved field, "not
obtained in any order iteration in the Machenhauer scheme". So the
literature's canonical higher-order schemes all differentiate along an
order-consistently closed slow flow.

**Implication.** Our design-note §1.4 leading-order closure (`v̇ ≈ Lv
+ VB(v,v)`) is *stronger* than Machenhauer (we keep the leading
drift) but *weaker* than Warn et al./Baer–Tribbia. Order-counting in
Warn's hierarchy says the feedback `VB(v, s(v))` enters the slave
relation two orders up — consistent with our expectation that
leading-order closure is exact through N = 2 and may cap the residual
slope at N ≥ 3. No published experiment isolates this (see §4:
Chouksey et al. computed higher slow derivatives by finite-differencing
the full model, which silently includes the full closure). Option D's
empirical gate (T1) is therefore genuinely needed; expect the gate to
trigger at N ≥ 3.

## 3. Divergence and optimal truncation

**Findings.** Vanneste, "Balance and spontaneous wave generation in
geophysical flows", Annu. Rev. Fluid Mech. 45, 147–172 (2013),
doi:10.1146/annurev-fluid-011212-140730, read in relevant part.
Slaving coefficients typically grow factorially, `F⁽ⁿ⁾ ∝ n!`; the
series is divergent because no invariant slow manifold exists for
non-dissipative systems (MacKay 2004) — divergence is fundamental,
not an artifact of the procedure. **Optimal truncation**: minimize
`εᴺF⁽ᴺ⁾ ∝ εᴺN!` ⟹ `N_ε ∝ 1/ε`, residual at the optimum
`~ γε^β exp(−α/ε)` (α > 0 an O(1) constant). Past N_ε the error
*increases* (Bender–Orszag standard). Warn, Atmosphere–Ocean 35,
135–145 (1997): "fuzzy slow manifold" of exponentially small width.
Rigor: Nekhoroshev/Neishtadt-type bounds (Cotter & Reich 2006,
Wirosoetisno 2004); for dissipative hydrostatic PEs Temam &
Wirosoetisno (2007–2011) prove exp(−α/ε^{1/3}). Explicit
exponential-asymptotic amplitude computations: Vanneste & Yavneh, JAS
61, 211–223 (2004); Vanneste, JAS 65, 1622–1637 (2008).

Practical takeaway for Ro 0.1–0.5: with `N_ε ≈ α/ε` and α = O(1),
the optimum is ≳ 2–10 across that range, and the Chouksey et al.
(2023) experiments (§4) confirm empirically that orders up to 4 still
*help* at Ro ≤ 0.5; degradation with order appears only as **Ro → 1**
("for Ro getting close to 1, the residuals start growing when the
order is increased … we expect that for Ro approaching 1 the optimal
truncation is of rather low order"). At Ro ~ 0.1 the observed
limitation of orders 3–4 was *numerical truncation error* (floors
~1e-6–1e-7 in their double-precision 255² runs), not divergence.

**Implication.** A genuine "order 3 worse than order 2" at Ro ≤ 0.5 is
a bug or a numerics floor, not series divergence — corroborating plan
§0.2's forensic conclusion. T2 should (a) assert slopes only above the
numerics floor, (b) include one Ro ≈ 0.7–1 point to *see* the expected
order-inversion, as a physics sanity check rather than a pass/fail.

## 4. Balance-method comparison protocols

**Findings.** Chouksey, Eden, Masur & Oliver, "A comparison of methods
to balance geophysical flows", JFM 971, A2 (2023),
doi:10.1017/jfm.2023.602, read in full. Single-layer rotating SW,
doubly periodic 2π domain, 255² points, two codes (pseudospectral
A-grid; C-grid Sadourny FD), AB3.

- **Methods**: higher-order balance `B_n` (Warn-et-al.-style NNMD,
  orders 0–4, our exact method) vs optimal balance `B_opt` (Masur &
  Oliver, GAFD 114, 429–452, 2020; exponential ramp, T = 0.5–4,
  backward–forward nudging per Masur, Mohamad & Oliver, Multiscale
  Model. Simul. 21, 624–640, 2023; theory Gottwald, Mohamad & Oliver,
  MMS 15, 1404–1422, 2017; Cotter, Phil. Trans. R. Soc. A 371,
  20120300, 2013; original OPV balance Viúdez & Dritschel, JFM 521,
  343–352, 2004).
- **Slow derivatives in B_n**: `∂_s z₁^±` analytically from the QG
  (leading-order) tendency; `∂_s z₂^±` and higher **by
  finite-differencing the nonlinear model** over a few steps from a
  balanced IC (following Kafiabad & Bartello, JFM 847, 614–643, 2018;
  Eden, Chouksey & Olbers, JPO 49, 2393–2406, 2019a). No published
  analytic order-consistent closure at n ≥ 2.
- **Diagnostic**: *diagnosed imbalance* — balance a base point
  `z*⁰ = P⁰z`, evolve to t′ = 0.5/Ro (random IC) or 4/Ro (jet IC),
  rebalance, and take the relative difference
  `I(u) = ‖u′ − u″‖ / (½(‖u′‖+‖u″‖))`, **separately for u and h**
  (an energy norm mixes their different Ro-scalings). Robustness of
  this proxy: von Storch, Badin & Oliver (2019, Energy Transfers
  book ch., pp. 53–85).
- **Slopes** (their Fig. 3, random IC): `I ∝ Ro` for B₀, `Ro²` for
  B₁, `Ro³` for B₂, `Ro⁴` for B₃/B₄ (the last two only above the
  numerics floor). Approximate magnitudes at Ro = 0.1 (read off the
  figure): B₀ ≈ 5e-2, B₁ ≈ 2e-3, B₂ ≈ 3e-5, B₃ ≈ 5e-6, B₄ ≈ 1e-6;
  at Ro = 0.5 all orders cluster at ~1e-2–1e-1. Floors ≈ 1e-7–1e-6.
- **B₄ ≈ B_opt** in magnitude and pattern; cross-balancing (balance
  with one, rebalance with the other) gives residuals ≈ max of the
  two — both find the *same* balanced state.
- **Scheme-consistency**: using A-grid analytic eigenvectors to
  balance the C-grid model injects an O(1)-in-Ro zeroth-order error
  (Fig. 8, orange): "the notion of balance in the discrete case is
  fundamentally tied to a particular scheme" (also Mohebalhojeh &
  Dritschel, QJRMS 126, 669–688, 2000). Sensitivity: "small details
  in the numerical coding affect the residual drastically" (also
  Eden et al. 2019a).
- Wave-emission companions: Chouksey, Eden & Brüggemann, JPO 48,
  1709–1730 (2018): first-order residual still dominated by slaved
  modes; true wave signal emerges only at **third/fourth order**
  (Eden et al. 2019a). Chouksey, Eden & Olbers, JPO 52, 1351–1362
  (2022): significant emission only near Ro ~ 1 with
  symmetric/convective instability. Chouksey PhD thesis (Univ.
  Hamburg) collects the JPO line. Recent extension of optimal
  balance with time-averaging to realistic flows: Rosenau et al.,
  JAMES (2026), doi:10.1029/2025MS005477.

**Implication.** Adopt their protocol wholesale for T5 (it is our v1
benchmark, now with published reference numbers) and their norm choice
(separate u/h relative norms) for T2/T3. The order-1/2/3/4 comparison
we planned **exists** (their Fig. 3) — our acceptance targets below
are lifted from it. Their scheme-consistency finding elevates
framework2's "eigenmodes from the discrete operator" (channel tier)
from nicety to requirement, and warns that the analytic tier is only
valid for spectrally-discretized models.

## 5. Resummed / telescoped variants

**Findings.** The telescoping variant **is published**: Tribbia, "A
simple scheme for higher-order nonlinear normal mode initialization",
MWR 112, 278–284 (1984). His derivation (reproduced in Temperton 1988
§7): Taylor-expand the nonlinear forcing on a fast mode, `r(t) ≈ r₀ +
r₁t + …`; killing the oscillatory part of the solution gives

```
c_n(0) = i r₀/ν_n + r₁/ν_n² + …   ⟺   φ = Σ_k (iA)^{−(k+1)} W N^{(k)}
```

— powers of the inverse linear operator on successive time
derivatives of the nonlinearity, i.e. exactly our §1.5 telescoped
partial-sum. Tribbia states it is *asymptotically equivalent* to
Baer–Tribbia at matching order. Temperton 1988 §7 gives the implicit
form `(iA)²Δx = −E_G E_Gᵀ δ_tt x` (second time derivative from two
forward steps) and finds order 2 strictly better than order 1
operationally: BAL₂ decreases monotonically for ≥ 10 iterations, and
the second-order step removes the residual **large-scale,
low-frequency** gravity modes that first order leaves ("the
large-scale gravity modes are the most difficult to initialize" —
Tribbia 1984). No published result shows telescoping degrading at
higher order; equally, nobody published order ≥ 3 of it.

Related reorganizations, none a resummation: the bounded-derivative
hierarchy sets `∂_t^N f = 0` directly (Kreiss; Browning, Kasahara &
Kreiss, JAS 37, 1424–1436, 1980; review Kasahara, Rev. Geophys. 20,
385–397, 1982) — Tribbia's order-N condition is its NMI twin;
Mohebalhojeh & Dritschel (JAS 58, 2411–2426, 2001) δ–γ balance
hierarchies; McIntyre & Norton (JAS 57, 1214–1235, 2000) PV-inversion
hierarchy. **No Padé/Borel resummation of the slaving series exists in
the geophysical literature** (searches return only QFT-style generic
resummation); the exponential-asymptotics line (Vanneste & Yavneh
2004; Vanneste 2008) computes the beyond-all-orders remainder directly
instead of resumming the series.

**Implication.** Upgrade the telescoping variant from "parked
curiosity" to "published contender with operational wins at order 2"
(Machenhauer-iterated + telescoped = exactly Temperton's §7 scheme).
The M2/T1 comparison stays decisive for N ≥ 3, where the literature is
silent. A resummation study would be publishable novelty, not a
reimplementation.

## 6. Non-stationary slow space

**Findings.** The Baer–Tribbia formalism handles it from the start:
in Tribbia 1979 (eq. 2.4b, following Baer 1977) the slow modes obey
`dx/dt + εΛ_x x = εG_x(ξ,ξ)` with `Λ_x = O(ε)` — the **slow linear
rotation is retained in the slow equation at the same order as the
nonlinearity**, and the O(ε²) slaved term contains `dy⁽¹⁾/dT` driven
by both. That is precisely our `Lφ₀^{(k)}` term. Applications with
genuinely rotating slow modes:

- **Tribbia, MWR 107, 704–713 (1979), equatorial β-plane**: no true
  spectral gap for ultralong waves; he picks a cutoff frequency
  σ̄ = 0.35 (between O(ε) = 0.02 and O(1), "approximately halfway on
  a log scale"), putting Rossby modes with small nonzero σ (and the
  low-σ part of the mixed mode) in the slow set. Result: BT
  initialization still works; it smooths the highest-frequency modes
  best; for zonal wavenumber 1 (σ_R ~ σ_g, no separation) smoothing
  is weak but "not disastrous"; balanced gravity modes barely affect
  rotational trajectories (≤ 5% amplitude differences, his Table 2).
- **Convergence limit**: Errico, MWR 111, 2214–2223 (1983) and
  Ballish (1981): the Machenhauer iteration `g^{l+1} = g^l −
  (i/ω)(dg/dt)^l` diverges for mode combinations whose advective
  frequency shift is comparable to the linear frequency (ratio of
  advecting velocity to modal phase speed ≳ 1 — his "phase speed more
  than doubles" criterion); in practice divergence occurs when
  initializing slow/shallow gravity modes (small equivalent depth).
  Remedies used operationally: frequency cutoff on the initialized
  set (Williamson & Temperton 1981) and under-relaxation (Kitade
  1982). Tribbia 1979 makes the same remark: the iteration "suffers
  when the elements of Λ_f approach order ε".
- Eden, Chouksey & Olbers, JPO 49, 291–308 (2019b, appendix B):
  first-order perturbation formula for the finite (but ≪ |ω±|)
  Rossby-wave frequency under slowly varying f — the β-band slow
  space in our target setting, with the explicit remark that the
  formalism carries over as long as ω_Rossby ≪ ω_gravity.
- Temperton 1988's equivalence proof, by contrast, **assumes**
  stationary nondivergent slow modes; no published *implicit* NMI
  with a rotating slow space exists.

**Implication.** Assumption (ii) (|ω_slow| ≤ εΩ and advective rate
≤ εΩ) is exactly the published operating envelope: Tribbia's σ̄
cutoff is our frequency-gap predicate, and Errico's divergence
criterion is the quantitative version of "the gap must beat the
nonlinearity". Our M1 recursion (slow rotation kept in φ₀^{(k)}) is
the Baer–Tribbia treatment made operator-explicit; combining it with
the implicit (eigenmode-free… or discrete-eigenbasis) machinery on
walls is the genuinely unpublished part — P4 is novel as a
combination, each ingredient separately has 1977–1988 pedigree.

## Acceptance targets (from the literature)

- **Slopes (T2, diagnosed-imbalance or next-order-term norm)**: order
  N residual ∝ Ro^{N+1}; demand clean slopes for N ≤ 2 over a decade
  of Ro; for N = 3–4 demand slope Ro⁴+ only above the numerics floor
  (Chouksey et al. 2023, Fig. 3).
- **Magnitudes (relative, separate u/h norms, random IC, Ro = 0.1)**:
  order 0 ≈ 5e-2, order 1 ≈ 2e-3, order 2 ≈ 3e-5, order 3 ≈ 5e-6,
  order 4 ≈ 1e-6; expect a floor at ~1e-7–1e-6 (double precision,
  ~256² grid). At Ro = 0.5 all orders ≈ 1e-2–1e-1.
- **Cross-method (T5)**: `BalanceExpansion(order≈3-4)` vs
  `OptimalBalance` residuals within a small factor of each other, and
  cross-balancing residual ≈ max of the two (same balanced state).
- **Order inversion**: higher order must *help* for Ro ≤ 0.5 and may
  legitimately hurt only for Ro ≳ 0.7–1 (optimal-truncation physics).
- **Iteration-flavored checks**: fast-tendency norm (BAL analogue)
  should drop ~4 orders of magnitude between order 0 and the first
  nontrivial correction (Temperton 1988).

## Surprises / risks

- **Balance is discretization-specific.** Eigenvectors must come from
  the *discrete* linear operator; mixing continuum eigenmodes with a
  staggered-grid model injects an O(1) zeroth-order error (Chouksey
  et al. 2023 Fig. 8). Analytic-tier BalanceExpansion is only valid
  for spectrally-consistent models; the channel tier's discrete Q is
  a requirement, not a bonus.
- **Nobody has tested the leading-order closure at N ≥ 3.** Chouksey
  et al. sidestepped it with FD model derivatives (which contain the
  full closure); our all-analytic recursion with leading-order
  closure enters untested territory at exactly the orders where the
  ordering argument says the feedback may bite. The T1 gate is
  load-bearing, not paranoia.
- **Numerical floors mimic saturation.** "Small details in the
  numerical coding affect the residual drastically" (Eden et al.
  2019a); slope tests must not interpret the 1e-6–1e-7 floor as a
  closure failure — and dealiasing (PadFactor) belongs to the method,
  not to test hygiene.
- **FD tendencies interact with time-stepping.** If model-stepped
  derivatives are ever reintroduced, semi-implicit/averaging schemes
  corrupt fast tendencies and stall convergence (Temperton 1988,
  Fig. 3); forward micro-steps only.
- **Telescoping won operationally at order 2** (Temperton 1988 §7,
  monotone BAL₂, removes large-scale low-frequency residuals) — the
  in-house "worse at order ≥ 3" result has no literature support and
  a broken baseline (plan §0.2); M2/R2 could plausibly land *for*
  telescoping.
- **Low-frequency fast modes are the hard ones** (Tribbia 1984;
  Errico 1983): convergence/accuracy degrades exactly at the
  slow-fast boundary. The β-band predicate must leave a real gap
  (Ω margin), and the structural-zero guard should be complemented by
  a *gap-ratio* diagnostic (max slow rate / Ω) exposed to the user.
