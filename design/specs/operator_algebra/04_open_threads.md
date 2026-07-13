---
status: normative
date: 2026-07-13
---

# Operator algebra — Open threads

Part of the operator design notes; see
[`00_overview.md`](00_overview.md) for the document map. Rules are in
[`02_algebra.md`](02_algebra.md).

---

## 5. Open threads

Four of the six threads are **closed** — by the class-design merge
([`../grid/classes/operator_algebra_merge.md`](../grid/classes/operator_algebra_merge.md))
and by landed code. Each keeps a one-line pointer to where its
decision now lives, so the section numbering (a stable identifier)
stays valid. Two remain genuinely open; both wait on their first
consumer.

### 5.1 Interning of algebraic operators — CLOSED

**Resolved: intern them.** The leaning recorded here (structural
equality first; intern only if profiling shows key-hashing cost) was
overtaken by the class-design rule that an `Operator` is *identity*-
hashed (`__eq__` is `self is other`), which is only sound if
structurally-equal operators are the same object. The algebra objects
built by the dunders (`Composite`, `SeparableComposite`,
`ScaledOperator`, `Dispatched`, `Reshard`/`Sync`) intern on their
static structure in a shared weak table, and the `@interned` class
decorator (D6) extends the same guarantee to the concrete leaf stencil
operators through an explicit `_intern_key()`. The interning primitive
is shared with spaces. Owner:
[`../grid/classes/operators_base.md`](../grid/classes/operators_base.md)
(D6).

### 5.2 Asymmetric binary composition ergonomics — CLOSED

**Resolved as T2** of the merge note: the positional tuple form stays
(`P @ (B1, B2)`, with `P @ B` as sugar for `P @ (B, B)`); it
generalizes to the n-ary elementwise operators, so no named-operand
form (`P.compose(left=, right=)`) is added. Post-composition into a
tuple — `(A1, A2) @ P` — **is an error**: confirmed, and mechanically
already one. Owner: section 3.8.

### 5.3 Mid-chain sync insertion — CLOSED

**Resolved as T3** in principle (a sync is a copy, not arithmetic: it
changes no result and reorders no floating-point operation, so it sits
*below* the algebra and does not violate "what you wrote is what
runs"), and then **settled by landed code**: the consumption-side sync
strategy syncs at the consumption site iff the operand's trace-time
halo validity is below the application's requirement, and negotiation
*caps* traced halo widths for shardability — so a chain that exhausts
its capped width re-syncs mid-chain. Correctness is width-independent
above the per-application floor; the negotiated width only tunes the
exchange count. Owner: the halo/storage contract in
[`../grid/classes/decomposition.md`](../grid/classes/decomposition.md).

### 5.4 `Symbol` as a chain factor vs `Hadamard`-only application — CLOSED

**Resolved as T4**: a `Symbol` is *not* an `Operator` (it carries a
dynamic leaf; operators are static structure), so it can never be a
factor in an operator `@` chain. What appears in a chain is a static
diagonal *operator* (`SpectralDerivative`, `PhaseShift`, `SincShift`,
a spectral filter) that derives its symbol from `grid.wavenumbers` at
trace time and applies it as a Hadamard multiply inside `_apply`; the
raw `Symbol` stays solver-facing (`op.eigenvalues(...)`, `1 / lap`,
the spectral solve). The question therefore dissolves: the two `@`s
(operator composition, `Symbol` diagonal composition) never mix, and
there is no normalization rule to write. Section 3.7's wording is
amended accordingly. Owner: section 3.7.

### 5.5 Tensor fields: flattening vs indexed blocks — OPEN

Section 3.1 flattens tensor multi-indices into the ordered tuple and
keeps the index structure as metadata. Higher-rank consumers (strain
-> stress closures) may want index-aware block construction (`Block`
addressed by multi-index rather than position). Still open and still
without a consumer: `TensorField` is unbuilt, and the shipped block
machinery (`Block`, `BlockMatrix`) is positional. Decide when the
first tensor consumer is ported. Owner: sections 3.1/3.5.

### 5.6 Where does `VectorField.map` end and the algebra begin? — OPEN

A scalar operator applied componentwise is `VectorField.map(op)`
(section 3.1), a diagonal block operator is `Block(diag(op, op, ...))`
— the two coincide for same-space components but differ in typing when
component spaces differ (map re-dispatches per component; a block is
fixed). Both spellings ship, and the ported models use both, so the
ergonomic ruling — *one* blessed idiom for module code — is still
owed. Owner: section 3.1 with grid-redesign section 2.4.
