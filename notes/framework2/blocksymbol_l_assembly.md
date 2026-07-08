# Assembling `L` as a `BlockSymbol` from the operator algebra

**Status: research + decision record, 2026-07-08.** Resolves decision 4
of [`projection_eigenmode_roadmap.md`](projection_eigenmode_roadmap.md)
§4 / [`projection_eigenmode_plan.md`](projection_eigenmode_plan.md) §7:
**`Eigenmodes.from_model` builds the linearized system operator `L`
from the operator algebra (a `BlockSymbol`), not a hand-provided
`linear_operator()`.** This note says what that requires and what it
implies. Backed by an audit of the tendency/term system, the
`Block`/`BlockMatrix` code, and the `A(k)` derivations.

## 1. What `L` is, per model

`L` is the linearized tendency, per Fourier mode an `m×m` matrix `A(k)`
(`∂_t z = L z`, the modules supply `L = −iA`). Its block structure
splits sharply:

- **Shallow water — a pure `BlockSymbol`.** Every `(out,in)` entry is a
  single per-term scalar symbol: Coriolis `u↔v` = interpolation
  `1̂⁺1̂⁻·f`, pressure gradient `p→u,v` = `k̂⁺`, divergence `u,v→p` =
  `c²k̂⁻`. `p` is prognostic; no constraint, no inverse. `L` = matrix
  sum of the Coriolis block + the gravity (grad/div) block.
- **Nonhydro — a Schur complement, NOT pure.** The *raw* linear terms
  (Coriolis `u↔v`, buoyancy `w←b` = `δ⁻²1̂_z⁺`, stratification `b←w` =
  `−N²1̂_z⁻`) are a pure 4-block symbol. But the diagnostic pressure is
  eliminated by the Leray projector `P = I − grad·(∇²)⁻¹·div`, which
  turns those 4 blocks into the dense `A(k)` with the `1/k̂²` denominator
  (`k̂² = k̂_h² + δ⁻²k̂_z²`). That `1/k̂²` is exactly one scalar
  `Symbol.inverse` of the Laplacian — the identical inverse
  `nonhydro2/modules/pressure.py` hand-rolls.

**Consequence:** `A_nonhydro = P · L_raw` where `P` embeds a scalar
`Symbol.inverse` (the pressure Poisson solve). **Assembling the
nonhydro `BlockSymbol` depends on the scalar-`Symbol` + `SpectralSolve`
substrate (roadmap phases C→D).** SW needs only C.

## 2. The stack `from_model`-via-`BlockSymbol` sits on

In dependency order (each layer unbuilt today unless noted):

1. **Scalar `Symbol` + per-operator `eigenvalues`** (roadmap C / plan
   S1). The leaf. Lights the dormant `base.py` composition. Prerequisite
   for all below.
2. **`SpectralSolve` / `Symbol.inverse`** (roadmap D / plan S2). Needed
   only for the **nonhydro** Leray constraint (SW skips it).
3. **`BlockSymbol` type + `BlockMatrix.eigenvalues()`.** `BlockMatrix`
   already exists (a static `m×n` grid of operators with real block
   matmul/sum, `Zero` structural zeros) — it just **inherits the raising
   base `eigenvalues`**. Add: (a) a `BlockSymbol` type (`@final`,
   `jaxify` dynamic `_data` of shape `(*n_modes, m, m)`, static aux =
   the input/output coefficient-space tuples), with matrix
   `@`/`+`/`*`/`conj`; (b) a `BlockMatrix.eigenvalues()` override that
   fills the block from each entry's scalar `eigenvalues` (the scalar
   chain/sum algebra runs *inside* each entry — reused unchanged; the
   base.py composition is leaf-type-agnostic and works the moment
   `BlockSymbol` implements matrix `@`/`+`/`*`). Per-component staggering
   phases are carried by the per-entry scalar symbols' `(space,
   codomain)` tags — no new bookkeeping.
4. **Per-term block signatures — the one genuinely missing metadata
   (see §3).**
5. **Wave-7 selection surface** (`fr.terms.linear`, `model.variant`,
   `fr.linearize`, `model.tendency`) — all stubs raising "lands at wave
   7." `from_model` calls `model.variant(term_filter=fr.terms.linear)`
   to get the linear term set.
6. **`EnergyMetric` `M` + `Eigenmodes.from_operator(L, M, grid)`**
   (roadmap A + H): assemble `L` as a `BlockMatrix` over the linear
   terms → (nonhydro) compose Leray via `Symbol.inverse` → `.eigenvalues
   (grid, coeff_space)` → `BlockSymbol` `L(k)` → batched
   `eigh(iML, M)`.

## 3. The pivot: linear terms are opaque closures, not operators

This is the true cost, and it is not the `BlockSymbol` type — it is the
**term system**. Today a `TendencyTerm` is a frozen record wrapping an
opaque closure `fn(module, state, ctx) -> dict[out_name, ScalarField]`.
It declares `advances` (outputs) but **has no `reads`**, and the
`(out,in)` coupling lives inside the closure body (Coriolis `du = f·v`,
`dv = −f·u` — the antisymmetric block is invisible to metadata). The
closures compute *through* operators (`.diff()`, `.to()`) but **discard
the operator object** — they retain no symbolic map.

So the operator-algebra `L` requires **re-authoring the four linear
terms to expose their block operators** — a `{(out,in): Operator}` map
(or an operator with a declared read set + `advances` codomain), so
`operator.eigenvalues(grid, space)` yields each block's `Symbol`. This
touches the term/module authoring surface, not just the grid layer.

**Do it as single-source-of-truth:** derive the numeric `fn` *from* the
same block operators (apply them), so the running model and the symbolic
`L` cannot drift, and the change pays for itself beyond eigenmodes
(automatic IMEX partition, general linear-stability/symbol tooling).

## 4. Sequencing: one interface, two backends

Keep `Eigenmodes.from_operator(L, M, grid) → eigh(iML, M)` as the stable
seam and let the `L`-producer be swappable:

- **H0 — numeric probe bootstrap.** Produce `L(k)` by `jax.jvp` of
  `fr.linearize(model).tendency(state, constraints=True)` against
  spectral unit-basis inputs. **`constraints=True` applies the Leray
  projection numerically**, so the probe handles the nonhydro pressure
  *for free* — no `Symbol.inverse` composition, no term re-authoring.
  Depends only on A (metric) + the wave-7 `linearize`/`tendency` surface
  + `jvp`. Validates the whole `eigh(iML,M)` energy-metric path against
  the analytic Tier-0 modes **now**, for both models.
- **H1 — symbolic `BlockSymbol` (the committed target).** Build §2.1–2.4
  and assemble `L(k)` symbolically; for nonhydro compose the Leray
  `Symbol.inverse`. Swap it in behind the unchanged `from_operator`
  seam. This is the larger investment (term re-authoring + C→D for
  nonhydro) and is where the elegance lands.

The probe is **not** the rejected `linear_operator()` fork (a hand-
written matrix); it is automatic and matrix-free, and it de-risks H1 by
proving the math + API before the term-system change.

## 5. Implications for the roadmap

- **Hardened edge:** nonhydro Phase H (symbolic) now **requires C→D**
  (Symbol → SpectralSolve), because the constraint elimination is a
  `Symbol.inverse`. SW Phase H needs only C. Recorded in the roadmap.
- **New prerequisite surfaced:** a per-term `(writes, reads)`/block
  signature on `TendencyTerm` + re-authored linear terms. This is a
  **term-system change**, larger than the `BlockSymbol` type itself, and
  should be scoped as its own step feeding H1.
- **Bootstrap unblocks the design early:** H0 (probe) lets phases
  A·B·F·G·H0 deliver the full energy-metric projection + numeric
  eigenmode story without the term re-authoring, which then lands as H1
  behind the same interface.
