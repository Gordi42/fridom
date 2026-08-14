---
status: decided + implemented
date: 2026-08-14
---

# Collection keywords take a bare element: list/tuple is many, else one

**Status: decision + implementation record, 2026-08-14.** Owner
request (Silvano, in chat): the public entry points that take a
collection accept a bare element wherever a one-element collection
would do — `run(outputs=writer)`, `modules_extra=wave_maker`,
`fields="temp"` — and accept a `list` as readily as a `tuple`. One
structural rule covers every such parameter:

> **a `list` or a `tuple` is many; anything else is one.**

Implemented on `feat/scalar-or-sequence`; the rule lives in
`fridom/_sequences.py` (`as_tuple`), called once at each public
boundary.

## 1. Why

Passing exactly one is the dominant case, not the exception. Every
`outputs=` call site in `examples/` passes one stream (13 of them),
and every `modules_extra=` site but two passes one module — one of
those two writes `modules_extra=tuple(makers)`, converting a list
that would have worked untouched. The one-element tuple is pure
ceremony there, and the trailing comma is easy to drop.

The stronger reason is the string parameters, where the old
`tuple(...)` spelling was a **silent mis-parse**, not a mere
inconvenience. `Writer(fields="temp")` became `("t", "e", "m", "p")`
and failed only at bind, with a message about four unknown fields;
`Writer(fields="u")` worked by accident, which is what kept the trap
hidden until someone named a longer field. `propagator(wrt="mixing.
kappa")` was shredded the same way and surfaced as a
`MissingParameterError` about a parameter named `'m'`.

Two spellings for the same argument is a real cost, and the house
instinct is one supported spelling per thing. It survives here
because this is *one* rule stated once for the whole public surface
rather than N per-parameter exceptions, and because the convention
already existed in two places and just was not uniform:
`fr.ops.Session(models=...)` has always taken a bare model or a
sequence, and `ClosureBase` has always read `fields="u"` as one name.

## 2. The rule

- **Structural, never duck-typed.** `as_tuple` whitelists `list` and
  `tuple`. It does *not* ask "is this iterable", because a `str` is,
  and because fridom already carries classes that iterate over
  *names* (`BindParameterView`, `fr.io.Series`) — an element type
  that later grows an `__iter__` would otherwise be spliced
  silently. A jax array is one element too, which is what keeps a
  single-target `propagator` differentiable: `theta=jnp.asarray(2.0)`
  splices whole, never entry by entry.
- **The cost is exotic containers.** A `set`, a generator or a
  `dict.keys()` view counts as a single element and fails loudly
  downstream. Wrap those in `list()`. This is deliberate: the loud
  failure is cheaper than the silent splice the permissive rule
  costs.
- **`None` is not special-cased.** A `None` default usually carries
  its own meaning (`Writer(fields=None)` selects the lifecycle
  default, which is not "select nothing"), so call sites guard it.
- **Coerce once, at the boundary.** Everything below the public
  entry point keeps its strict tuple contract.

## 3. Where it applies, and where it does not

Applied: `fr.model.Model(modules=, io=, allow_unadvanced=)`,
`Model.run(outputs=)`, `Model.variant(extra_modules=)`,
`Model.propagator(wrt=)` and the returned `run(theta=)`,
`fr.ops.Session(models=, outputs=)`, `fr.io.Writer(fields=)`, and
`modules_extra=` on the `nonhydro2` / `shallowwater2` / `hydrostatic`
presets.

**Not applied where the element type is itself a tuple.** In
`fr.io.streams` (`Sequence[tuple[OutputStream, Path]]`), the
composer's `terms` / `stages`, and `sel_specs` in the graded
operators, `(a, b)` is genuinely ambiguous between one pair and two
elements. Those parameters are internal and stay strict — which is
the general shape of the boundary: the rule is a public-surface
convenience, not an internal one.

**`fr.io.at(times=)` is widened, not converted.** It used to refuse a
scalar with a taught error; it now reads a bare time as one time.
But it keeps accepting any iterable rather than switching to the
list/tuple whitelist, because `at(np.arange(0, 100, 10))` is a normal
spelling for output times and the whitelist would demote that ndarray
to a single element. Its element type (a float or a `timedelta64`) is
never a container, so the permissive form carries no ambiguity here.
The documented guarantee is uniform everywhere; two parameters
(`at(times=)` and `ClosureBase(fields=)`) simply tolerate more than
it promises.
