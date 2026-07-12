---
status: draft
date: 2026-07-12
---

# Krylov scan plan — O(1) trace for the mapped pressure solve

ROADMAP 3.6. The mapped pressure CG
([`spatial/operators/krylov.py`](../../../src/fridom/spatial/operators/krylov.py))
runs a **fixed, fully unrolled** iteration loop, so tracing and XLA
compilation are **O(iterations)**. The owner has declared the current
cost unacceptable to ship (2026-07-12).

## 1. The measured problem

| iterations | trace | compile | warm run | HLO lines |
|---|---|---|---|---|
| 12 | 1.4 s | 1.5 s | 1.6 ms | 10 497 |
| 30 (the model default) | 3.4 s | 3.6 s | 3.6 ms | 25 167 |
| 300 (the staircase reference) | 31 s | 48 s | 33 ms | 245 217 |

Everything scales linearly with the iteration count and is essentially
independent of `n` (n = 16/32/64 at 12 iterations: trace 1.40/1.44/1.43
s). **Warm execution is ~0.05% of the cost** — this is a
trace/compile problem, not a runtime one.

Two consequences beyond slow tests:

- a **jitted forced-4 solve** at 12 iterations did not finish compiling
  in 10 minutes (XLA printed its "Very slow compile?" alarm), so the
  multi-device gates run the solve eagerly, which is itself 74-87x
  slower per call than single-device;
- the model default `pressure_iterations = 30`
  ([`nonhydro2/modules/core.py`](../../../src/fridom/nonhydro2/modules/core.py))
  puts ~7-14 s of compile into **every mapped model build**.

The *other* half of the cost — metric coefficients re-derived on every
CG iteration — is already fixed (merge `479e2965`): halo exchanges
110 -> 65, HLO 10 494 -> 6 758 at 12 iterations, bit-identical. What
remains is purely the unrolled loop.

## 2. Why it was unrolled (the real obstacle)

`ScalarField` is jaxified as
`@partial(jaxify, dynamic=("_data",), annotation=("_metadata",))`
([`fields/scalar_field.py:80`](../../../src/fridom/spatial/fields/scalar_field.py)).
`_halo_valid` is therefore **neither dynamic nor annotation** — it is
static aux, and it **participates in treedef equality**.

A CG iterate carried through `lax.scan` is a `ScalarField`. Its
`_halo_valid` changes as operators consume and reset halo validity, so
the carry's output treedef can differ from its input treedef ->
`carry input/output pytree structure differ`. This is the same class of
failure as grid follow-up item 12
([`phase2_grid_followups.md`](phase2_grid_followups.md)).

The exemption route (adding `_halo_valid` to the `annotation=` category)
is **rejected**, for the reason recorded there: unlike metadata,
`halo_valid` drives sync **placement**, so excluding it from the
jit-cache key could reuse a body compiled for a different halo state.

## 3. The plan: carry arrays, not fields

**Carry raw `jax.Array`s plus a statically-closed-over space; rebuild
the `ScalarField`s inside the scan body with a canonical halo-validity
spec.** The space, the operator, the preconditioner and the metric memo
are all static or closure-captured, so the carry is a flat tuple of
arrays whose treedef is trivially stable.

Sketch (not final code):

```python
def _body(carry, _):
    x_d, r_d, p_d, z_d, rz = carry
    x, r, p, z = (self._field(d) for d in (x_d, r_d, p_d, z_d))
    ap = self._operator(p)
    alpha = _guarded_ratio(rz, self._dot(p, ap))
    x = x + alpha * p
    r = r - alpha * ap
    z = self._project(self._precondition(r))
    rz_new = self._dot(r, z)
    beta = _guarded_ratio(rz_new, rz)
    p = z + beta * p
    return (x.data, r.data, p.data, z.data, rz_new), None
```

`self._field(d)` rebuilds `(space, data)` with **one canonical
`halo_valid`** — the zero/empty spec — every iteration. That is the
whole trick: the fields entering the body are always in the same
halo state, so the consumption-side sync machinery makes the same
placement decisions every iteration (which is exactly what makes one
compiled body correct for all of them), and the treedef is stable by
construction because the carry contains no fields at all.

**`lax.scan`, not `lax.fori_loop`.** `jax.grad` must keep flowing
through the solve (`test_grad_through_the_solve_is_finite` is the gate;
reverse-mode through `fori_loop` is not supported, through `scan` it
is). `length=iterations`, `xs=None`.

### Correctness questions to settle during implementation

1. **Is the canonical halo state actually invariant?** Verify that
   every field entering the carry is in the same halo state at the end
   of the body as at the start (the steady-state claim). If the first
   iteration genuinely differs (`p = z` fresh from the preconditioner),
   either peel it before the scan or canonicalize it — measure, do not
   assume.
2. **Does the sync placement inside the body match the unrolled one?**
   The exchange-count gates
   ([`tests/spatial/test_exchange_counts.py`](../../../tests/spatial/test_exchange_counts.py))
   are the instrument: the per-iteration exchange count must not grow.
   A canonical (empty) validity claim per iteration could *add* a sync
   per iteration relative to the unrolled form, where a claim can carry
   across iterations. **If it does, that is a real trade** (one extra
   exchange per iteration against an O(1) trace) and must be measured
   and reported, not hidden.
3. **Bit-identity.** The scanned solve should be bitwise equal to the
   unrolled one at the same iteration count. If XLA's scan lowering
   reassociates anything, quantify it; the mapped-flat identity gate
   (6e-15 today) is the outer check.

## 4. Gates

- **Bitwise** equality with the unrolled solve at 12 and 30 iterations
  (or a quantified, justified deviation).
- Trace/compile **O(1) in iterations**: measure trace time and HLO line
  count at 12 / 30 / 300 iterations; the 300-iteration case must stop
  being pathological (today: 245k lines, 48 s).
- **`jax.grad` through the solve still finite** (the reason for `scan`
  over `fori_loop`).
- **Exchange counts per iteration unchanged** (or the regression named
  and justified, per question 2 above).
- A **jitted forced-4** mapped solve compiles and runs — today it does
  not finish. If it now does, the multi-device tests should stop
  solving eagerly (that is where the 74-87x blowup lives).
- Full `tests/nonhydro2`, `tests/validation`, `tests/spatial/operators`
  green; the C4 moving-geometry gates in particular (the metric memo
  must keep being rebuilt per solve, per `479e2965`).

## 5. Follow-on (not this plan)

Once the trace is O(1), reconsider:

- restoring the library default `pressure_iterations` (the test budgets
  were trimmed under the unrolled cost);
- whether the tolerance-break variant becomes worth having (a
  `while_loop` is not differentiable, but a scan with an early-exit
  mask is — only worth it if iteration budgets get large);
- the `custom_vjp`-via-implicit-function-theorem route for adjoints
  through the solve (CS-D2 records it as deferred), which becomes more
  attractive once the forward solve is a single scanned body.
