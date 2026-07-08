# Projection / eigenmode build roadmap (dependency-ordered)

**Status: build-order synthesis, 2026-07-08.** Sequences the two
untracked design notes —
[`operator_symbols_plan.md`](operator_symbols_plan.md) (Symbol
substrate S1/S2) and
[`projection_eigenmode_plan.md`](projection_eigenmode_plan.md) (energy-
metric ladder P1–P4) — against the committed wave plan
([`model/implementation_plan.md`](model/implementation_plan.md), wave 7
= ROADMAP 2.8). Grounded in a full audit of the current tree.

## 0. Where we stand

Between **wave 6 (ROADMAP 2.7 — model ports, done)** and **wave 7
(ROADMAP 2.8 — state transforms, unstarted)**. Wave 6 shipped the
pressure projection and eigenmode *data* **hand-rolled**, deliberately
bypassing the not-yet-built Symbol layer.

**Built & mature (reuse):** all four transform families
(Fourier/Sine/Cosine/Chebyshev) in `grid/operators/` + dispatch;
`grid.wavenumbers` / `grid.measure` (trapezoid-correct on bounded
axes); all coefficient/nodal/average/constant spaces; the operator
algebra with a **dormant** eigenvalue-composition path
(`_chain_eigenvalues` product, `OperatorSum` sum, `ScaledOperator`
scale — all coded, all waiting on a `Symbol`); the model layer
(`model.parameters`, `TimeDependent`/`Ramp`, `resolve_at`,
`advance`/`reset`); analytic eigenmode data ports; a dense banded solve
in `model/implicit.py`; `ScalarField.integrate`/`conj`/`mean`.

**Stub / missing / designed-only (build):** `symbol.py` (9-line stub —
the one node blocking the whole spectral half); every per-operator
`eigenvalues` (all raise); `SpectralSolve`; `grid/operators/banded`;
`BlockSymbol` (`Block.__init__` raises); the whole `StateTransform`
algebra (`transforms/` package empty); `Propagator`/`TimeAverage`/
`OptimalBalance`; **any inner-product / norm / energy on State — the
single load-bearing gap** (SW/nonhydro2 eigenmodes fake it with a bare
mode-wise sum); SW energy diagnostics; non-`IntervalMesh` (Chebyshev)
quadrature and Shen/Galerkin BC-structured Chebyshev.

## 1. The dependency graph

Three **independent foundations** (parallelizable), then convergence:

```
 Track M (metric)      Track S (symbol)        Track T (transforms)
 ────────────────      ────────────────        ────────────────────
 A EnergyMetric        C Symbol + eigenvalues  E StateTransform algebra
 + State inner prod    │  (lights the dormant  │  (transforms/ pkg,
 │                     │   base.py algebra)     │   @/+/complement,
 ▼                     ▼                        │   Identity/Shift/FixedPoint)
 B eigenmodes p = M q  D SpectralSolve          │
 │  (analytic Tier-0)  │  + banded (lift from   │
 │                     │    model/implicit.py)  │
 │                     │                        │
 └─────────┬───────────┴────────────┬───────────┘
           ▼                        ▼
 F Tier-1 projections        H numeric BlockSymbol
   (Vortical/Wave/Div         eigenmodes  eigh(H,M)
    as StateTransforms)       needs C + A
   needs B + E                        │
           │                          ▼
           ▼                  I non-periodic / vertical
 G Tier-2 dynamical            general eigenmodes
   (Propagator, TimeAverage,   (structure fns / banded eigh;
    OptimalBalance)            walls ⇒ Shen-Cheb + Cheb measure)
   needs E (+ F for OB base)   needs H + D
```

Edges that matter: **A blocks every projection** (nothing has an inner
product today). **C blocks the entire spectral half** (D, H) and lights
up the already-written composition algebra the moment it exists.
**E blocks all transform-wrapped projections** (F, G). B needs only A;
F needs B **and** E; H needs C **and** A.

## 2. The phases

Each: what · depends · delivers · gates. Map column ties to the source
notes.

### Phase A — `EnergyMetric` + State inner product  · *(P1a)*
- **What.** An `⟨a,b⟩_M` on `State`/`VectorField`: component sum of
  measure-weighted `integrate(conj(a_c) · (w_c · b_c))`, in two forms —
  physical (`grid.measure` + `integrate`) and spectral/Parseval (the
  `norm="forward"` amplitude convention × interval length, since
  `integrate` refuses coefficient factors). `EnergyMetric` holds the
  per-field weights (`diag(1,1,δ²,1/N²)` nonhydro; `diag(1,1,1/c²)` SW),
  sourced from the model energy. Port SW `ekin`/`epot` and nonhydro2
  `epot` as bound diagnostics so the weights have one home.
- **Depends.** Nothing new (`ScalarField.integrate`/`conj`,
  `grid.measure`, `model.parameters` all exist).
- **Delivers.** The missing load-bearing surface; norms for every
  downstream phase.
- **Gates.** Decision 2 (metric first-class).

### Phase B — eigenmodes `p = M q`  · *(P1b)*
- **What.** Rewrite nonhydro2 `_vec_p`/`_pair` and SW `_p_arrays` to
  *derive* `p = M(q)` via Phase A; keep the analytic `q`. Lift the
  nonhydro2 projector from raw-dict to `State` (SW parity). Update the
  biorthonormality tests to normalize under the energy inner product
  (the SW test currently asserts the *unweighted* sum = 1).
- **Depends.** A.
- **Delivers.** No hand-written `p`; validated analytic Tier-0 projector.
- **Gates.** Confirms decision 2 end-to-end; pure refactor, lowest risk.

### Phase C — scalar `Symbol` + per-operator `eigenvalues`  · *(S1)*
- **What.** Implement `Symbol` (`@final`) in `symbol.py`:
  `@`,`+`,`*`,`**`,`inverse`,`conj`,`__call__` — the `@`/`+`/`*` contract
  the dormant `base.py` composition already calls. Fill each operator's
  `eigenvalues` (SpectralDerivative, FiniteDifference, PhaseShift,
  SincShift, LinearInterp, flux/reconstruct family, `Fourier`
  truncation mask) → the composition algebra and the Laplacian symbol
  light up for free.
- **Depends.** Existing wavenumber helpers only.
- **Delivers.** Pure-diagonal spectral solve capability; the substrate
  `BlockSymbol` extends.
- **Gates.** Sign-offs A (`@final`), B (eager materialization), E
  (`inverse` exact `== 0`).

### Phase D — `SpectralSolve` + banded primitive  · *(S2; DONE, with a follow-up)*
- **Done (round 2 / Wave 9B):** banded primitive lifted to
  `grid/operators/banded.py`; `SpectralSolve` (pure-diagonal) built;
  pressure inverts via `Symbol.inverse` bitwise-identical.
- **Follow-up (D′ — the pressure-solver refinement,
  [`symbol_stack_design.md`](symbol_stack_design.md)):** make
  `eigenvalues` **layout-faithful** (read `grid.wavenumbers(space,
  axis)`) + add `Symbol × field` (constant-in-transformed-axes) +
  build the pressure `∇² = Div @ Diag(1,1,1/dsqr) @ Grad` so
  `SpectralSolve(∇²)` **retires the hand-rolled `discrete_laplace_symbol`**
  in `nonhydro2/modules/pressure.py`. Bounded, near-term. `dsqr` scales
  at the symbol level (traced-but-constant leaf).
- **Deferred:** mixed Fourier×Chebyshev via the diagonal/banded
  partition (folds into Phase I's `Banded`).

### Phase E — `StateTransform` algebra  · *(wave 7 A / ROADMAP 2.8-A)*
- **What.** Populate `transforms/`: `StateTransform` base,
  `StateSignature`, `TransformInfo`, `@`/`+`/`.complement`, the Tier-1
  pytree vs Tier-2 host split, `Identity`/`Shift`/`FixedPoint`,
  `relative_l2`/`assert_idempotent`. Plus the wave-7-A model hooks
  (`model.tendency`, `model.variant`, `fr.linearize`, `fr.terms`) — all
  currently stubs that raise "lands at wave 7."
- **Depends.** State + model (exist). Independent of A–D.
- **Delivers.** The composition algebra all projections compose in.

### Phase F — Tier-1 projections  · *(wave 7 C / P4a)*
- **What.** Wrap the Phase-B energy-metric projectors as
  `StateTransform`s: `VorticalProjection`, `WaveProjection = P(+1) +
  P(−1)`, `DivergenceProjection = (P_vortical + P_wave).complement`.
  forward-transform → diagonal project → inverse-transform; expose
  `nh.transforms` / `sw.transforms`.
- **Depends.** B **and** E. (Not Symbol — wraps the analytic eigenmodes.)
- **Delivers.** The user-facing analytic projections, composable.

### Phase G — Tier-2 dynamical projections  · *(wave 7 B / P4b)*
- **What.** `Propagator` (wrap `model.advance`), `TimeAverage`,
  `OptimalBalance = forward @ base @ backward` (`Ramp.reversed` exists).
  NNMD stays **descoped** (signed off).
- **Depends.** E + `Ramp`/`Model` (exist); OB needs a Phase-F base
  projection.
- **Delivers.** The variable-coefficient / ramping fallback (Tier 2),
  incl. the Rayleigh-quotient eigenvalue diagnostic.

### Phase H — numeric eigenmodes `eigh(iML, M)`  · *(P2)*
Decision 4 **resolved: assemble `L` from the operator-algebra
`BlockSymbol`** (not a hand `linear_operator()`). Split into two
backends behind one `Eigenmodes.from_operator(L, M, grid)` seam — see
[`blocksymbol_l_assembly.md`](blocksymbol_l_assembly.md).
- **H0 — numeric probe bootstrap.** `L(k)` by `jax.jvp` of
  `fr.linearize(model).tendency(·, constraints=True)` on spectral unit
  inputs (`constraints=True` applies the nonhydro Leray projection
  numerically — pressure handled for free); `eigh(iML, M)`. Verify it
  reproduces the analytic Tier-0 modes.
  - **Depends.** A (metric `M`) + the wave-7 `linearize`/`tendency`
    surface (Phase E hooks) + `jvp`. **Not** Symbol/BlockSymbol.
- **H1 — symbolic `BlockSymbol` (committed target).** `BlockSymbol`
  type + `BlockMatrix.eigenvalues()`; assemble `L(k)` from block-placed
  per-term symbols; **for nonhydro compose the Leray `Symbol.inverse`.**
  Swap behind the unchanged `from_operator` seam.
  - **Depends.** SW: C. **Nonhydro: C → D** (the constraint elimination
    is a scalar `Symbol.inverse` = the pressure Poisson solve). **Plus a
    term-system change** — a `(writes, reads)`/block signature on
    `TendencyTerm` + re-authoring the four linear terms as retained
    block operators (the genuinely missing metadata; larger than the
    `BlockSymbol` type itself).
- **Gates.** Decision 1 (`eigh(iML, M)`, not `eig`).
- **Delivers.** Numeric spectral eigenmodes where no closed form exists.

### Phase I — variable-coefficient / wall-bounded eigenmodes  · *(P3; redesigned)*
Now scoped by [`symbol_stack_design.md`](symbol_stack_design.md) — the
`Banded` + nesting + mixed-representation tier (not "needs Chebyshev").
**Realized as S3–S4 of
[`composition_refactor_plan.md`](composition_refactor_plan.md)** (on the
consolidated realized-map layer, after the S0–S2 composition-core work).
- **What.** (a) The **`Banded`** operator type (promote
  `grid/operators/banded.py` to a first-class diagonal-in-transformed /
  banded-in-one-axis operator with matvec + Thomas solve). (b) The
  **mixed `Fourier ⊗ Nodal`** representation + partial transforms (from
  D′). (c) **`Symbol × field`** for variable coefficients (`f(y)`,
  `N²(z)`; from D′). (d) **`BlockSymbol` of `Banded`** nesting for the
  general system, densified only at the `eigh` boundary. Delivers the
  β-plane / boundary-trapped / vertical-structure modes (the
  `boundary_emission` / `Adiabatic-Coriolis-Ramping` generality).
- **Depends.** H + D′ (layout-faithful `eigenvalues`, `Symbol × field`,
  mixed transforms). Chebyshev/Shen is now just *one* `Banded` instance
  (dense bandwidth); the earlier "Shen + Chebyshev quadrature"
  prerequisites are needed only for the spectral-vertical variant, not
  for the FD-vertical / structure-function path.
- **Delivers.** The fully general eigen/projection reach.

## 3. Recommended serialization

Parallel-friendly (three foundations at once): **A · C · E** →
then **B, D** → then **F** → **G, H** → **I**.

If built by one hand, prioritizing the projection design first:

1. **A** — inner product / `EnergyMetric` (smallest, unblocks all
   projections, fixes the missing surface).
2. **C** — scalar `Symbol` + eigenvalues (highest-leverage substrate;
   lights the dormant algebra; unblocks D and H). Start alongside A.
3. **B** — eigenmodes `p = M q` (the headline refactor; validates the
   whole energy-metric idea against the existing regression tests).
4. **E** — `StateTransform` algebra (the committed wave 7; gate for
   F/G).
5. **D** — `SpectralSolve` + banded (retires hand-rolled pressure code).
6. **F** — Tier-1 projections in the algebra.
7. **G** — Tier-2 dynamical (Propagator / OptimalBalance).
8. **H** — numeric `BlockSymbol` eigenmodes (`eigh(H,M)`).
9. **I** — non-periodic / vertical general eigenmodes.

Phases **A–B–F–G** deliver the complete energy-metric projection story
on the analytic (Fourier + sine/cosine collocated) path — the user's
headline goal — **without** the Symbol substrate. **C–D** are the
parallel substrate track that retires hand-rolled wave-6 code; **H–I**
are the general (numeric, non-Fourier, wall-bounded) reach that build on
both.

## 4. Decision gates (sign-off before the phase that needs them)

- Before **C/D**: `operator_symbols_plan.md` §7 A–E (Symbol `@final`,
  eager materialization, banded lift across model/grid, separable-only,
  exact-zero `inverse`).
- Before **B**: decision 2 (metric first-class, `p = M q`).
- Before **H**: decision 1 (`eigh(iML, M)` not `eig`). Decision 4
  **resolved** → `BlockSymbol`-from-algebra
  ([`blocksymbol_l_assembly.md`](blocksymbol_l_assembly.md)); the
  `TendencyTerm` block-signature + linear-term re-authoring feeding H1
  is scoped in
  [`linear_term_blocks_plan.md`](linear_term_blocks_plan.md).
- Before **F/G**: the wave-7 named oracles
  (`model/implementation_plan.md:143-145` — bitwise-twin regression,
  the info law, `assert_idempotent`, the three signed behavior deltas);
  NNMD descoped.
