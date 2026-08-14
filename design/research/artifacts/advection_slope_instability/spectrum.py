"""Leading modes of the one-step map on the mapped grid (A0 diagnostic).

Linearizes the one-step propagator with ``jax.jvp`` about the projected
uniform state and runs ARPACK on the resulting linear operator.  Reports
the leading eigenvalues, the growth rate ``log|lambda| / dt`` and the
vertical / horizontal structure of the fastest-growing mode.
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

from clean_repro import make_flat_model, make_mapped_model

COMPONENTS = ("u", "v", "w", "b")


def step_operator(model, u0):
    """Return ``(matvec, base_state, unflatten, ndof)`` for one step."""
    model.set_fields(u=u0)
    prop = model.propagator(wrt=(), steps=1)

    def f(state):
        return prop((), state)

    base = model.carry.state
    leaves, tree = jax.tree_util.tree_flatten(base)
    sizes = [int(l.size) for l in leaves]
    shapes = [l.shape for l in leaves]
    ndof = sum(sizes)

    def unflatten(vec):
        parts, off = [], 0
        for s, sz in zip(shapes, sizes, strict=True):
            parts.append(jnp.asarray(vec[off:off + sz]).reshape(s))
            off += sz
        return jax.tree_util.tree_unflatten(tree, parts)

    @jax.jit
    def jvp(vec):
        tangent = unflatten(vec)
        _, out = jax.jvp(f, (base,), (tangent,))
        return jnp.concatenate(
            [l.reshape(-1) for l in jax.tree_util.tree_leaves(out.state)])

    return (lambda v: np.asarray(jvp(jnp.asarray(v)))), base, unflatten, ndof


def describe_mode(model, base, unflatten, vec, tag):
    state = unflatten(np.asarray(vec).real)
    print(f"    mode structure ({tag}):")
    for c in COMPONENTS:
        blk = np.asarray(state[c].data)
        prof = np.abs(blk).max(axis=(0, 1))
        peak = prof.max()
        if peak <= 0:
            continue
        iz = int(np.argmax(prof))
        row = blk[:, 0, iz]
        sp = np.abs(np.fft.fft(row))
        print(f"      {c}: peak={peak:.3e} iz={iz} "
              f"zprofile={np.array2string(prof / peak, precision=2, max_line_width=250)}")
        print(f"          x-spec={np.array2string(sp / max(sp.max(), 1e-300), precision=3, max_line_width=250)}")


def report(tag, model, dt, u0, k=6):
    matvec, base, unflatten, ndof = step_operator(model, u0)
    op = LinearOperator((ndof, ndof), matvec=matvec, dtype=np.float64)
    vals, vecs = eigs(op, k=k, which="LM", tol=1e-8,
                      v0=np.random.default_rng(0).standard_normal(ndof))
    order = np.argsort(-np.abs(vals))
    print(f"\n=== {tag} (dt={dt}, ndof={ndof}) ===")
    for i in order:
        lam = vals[i]
        print(f"  lambda = {lam.real:+.6f}{lam.imag:+.6f}j  "
              f"|lambda| = {abs(lam):.6f}  "
              f"log|lambda|/dt = {np.log(abs(lam)) / dt:+.4f}")
    describe_mode(model, base, unflatten, vecs[:, order[0]], "leading")
    return np.log(abs(vals[order[0]])) / dt


def main():
    U = 0.4
    ny = 4
    rates = {}
    for n in (8, 16):
        for dt in (0.02, 0.01):
            rates[(n, dt)] = report(
                f"mapped n={n}", make_mapped_model(n, dt, ny=ny), dt, U)
    report("flat n=8", make_flat_model(8, 0.02, ny=ny), 0.02, U)
    for amp in (0.02,):
        report(f"mapped n=8 amp={amp}",
               make_mapped_model(8, 0.02, ny=ny, amp=amp), 0.02, U)
    report("mapped n=8 U=0.2", make_mapped_model(8, 0.02, ny=ny), 0.02, 0.2)
    print("\nsummary rates:", rates)


if __name__ == "__main__":
    sys.exit(main())
