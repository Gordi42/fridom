# Operator algebra — Open threads

Part of the operator design notes; see
[`00_overview.md`](00_overview.md) for the document map. Rules are in
[`02_algebra.md`](02_algebra.md).

---

## 5. Open threads

Deliberately unresolved questions; each names the section that will
own the decision once made.

### 5.1 Interning of algebraic operators

Spaces are interned so equality is identity (grid-redesign
section 2.2). Should composites/sums/blocks be interned too, or is
structural equality (for jit-cache keys, section 2.2) enough?
Interning gives cheap identity checks but needs a canonical hashable
structure including bound axes and coefficient *placement* (not
values). Leaning: structural equality first; intern only if profiling
shows key-hashing cost. Owner: section 2.2.

### 5.2 Asymmetric binary composition ergonomics

Section 3.8 fixes the tuple form `P @ (B1, B2)`. Is that enough, or do
named-operand forms (`P.compose(left=..., right=...)`) earn their
keep once real flux modules are ported? Also: does post-composition
distribute into tuples on the left (`(A1, A2) @ P` has no meaning —
confirm it stays an error)? Owner: section 3.8.

### 5.3 Mid-chain sync insertion

A long un-synced chain accumulates halo linearly (section 3.6); at
some width an intermediate halo exchange is cheaper than the wider
ghost layer. May the grid insert syncs *inside* a registered chain
(it knows the accumulated depths from the trace), or are sync points
exclusively the caller's/tendency author's decision? Auto-insertion
interacts with reproducibility of "what you wrote is what runs"
(section 3.11). Owner: section 3.6 with grid-redesign section 5.

### 5.4 `Symbol` as a chain factor vs `Hadamard`-only application

Section 3.7 admits a `Symbol` as an ordinary factor in chains between
coefficient spaces. Does that blur the grid-redesign section 3.11
line that applying a symbol *is* a `Hadamard` multiply — i.e. should
`S @ A` normalize to something, or is a Symbol-in-chain just an
operator whose application happens to be Hadamard? Leaning: the
latter, no special rule. Owner: section 3.7.

### 5.5 Tensor fields: flattening vs indexed blocks

Section 3.1 flattens tensor multi-indices into the ordered tuple and
keeps the index structure as metadata. Higher-rank consumers (strain
-> stress closures) may want index-aware block construction
(`Block` addressed by multi-index rather than position). Decide when
the first tensor consumer is ported. Owner: sections 3.1/3.5.

### 5.6 Where does `VectorField.map` end and the algebra begin?

A scalar operator applied componentwise is `VectorField.map(op)`
(section 3.1), a diagonal block operator is
`Block(diag(op, op, ...))` — the two coincide for same-space
components but differ in typing when component spaces differ (map
re-dispatches per component; a block is fixed). Pick one idiom for
module code and document it. Owner: section 3.1 with grid-redesign
section 2.4.
