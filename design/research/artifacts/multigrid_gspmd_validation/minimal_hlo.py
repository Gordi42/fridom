"""Leg 1(a).1 MINIMAL: standalone HLO of the smoother's tridiagonal solve
on arrays sharded 4-way along the x batch axis (mirrors the real model:
P('devices',None,None), solve axis = z = 2)."""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

sys.path.insert(0, "/tmp/claude-100698/-work-uo0780-u301533-fridom-fridom-dev/4f23fac6-d4e9-4088-a855-5ccf5ca6a06b/scratchpad")
from fridom.spatial.operators.banded import tridiagonal_solve_along_axis  # noqa: E402

OUT = "/tmp/claude-100698/-work-uo0780-u301533-fridom-fridom-dev/4f23fac6-d4e9-4088-a855-5ccf5ca6a06b/scratchpad"

n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
print("devices:", jax.device_count())
mesh = Mesh(np.array(jax.devices()), ("devices",))
shard = NamedSharding(mesh, P("devices", None, None))  # x sharded, y,z local
AXIS = 2  # solve along z (unsharded), batch over x (sharded) + y (local)

key = jax.random.PRNGKey(0)
shape = (n, n, n)
rl = dtype = jnp.float64
diag = jax.device_put(
    -2.0 - jax.random.uniform(key, shape, dtype=rl), shard)
lower = jax.device_put(jnp.ones(shape, rl), shard)
upper = jax.device_put(jnp.ones(shape, rl), shard)
rhs = jax.device_put(jax.random.normal(key, shape, dtype=rl), shard)


def make(method):
    def f(lo, di, up, b):
        return tridiagonal_solve_along_axis(lo, di, up, b, AXIS, method=method)
    return f


for method in ("cusparse", "pcr"):
    f = make(method)
    lowered = jax.jit(
        f, in_shardings=(shard, shard, shard, shard)).lower(
        lower, diag, upper, rhs)
    compiled = lowered.compile()
    text = compiled.as_text()
    path = f"{OUT}/minimal_hlo_{method}_n{n}.txt"
    with open(path, "w") as fh:
        fh.write(text)
    print(f"\n==== method={method} : {path} ({text.count(chr(10))} lines) ====")
    # custom-call + collective lines
    for line in text.splitlines():
        low = line.lower()
        if ("custom-call" in low or "gtsv" in low or "tridiag" in low
                or "all-gather" in low or "all-reduce" in low
                or "collective-permute" in low or "all-to-all" in low
                or "reduce-scatter" in low or "dynamic-slice" in low
                and "custom" in low):
            print("  ", line.strip()[:200])
    # verify correctness vs replicated
    out = f(lower, diag, upper, rhs)
    rep = tridiagonal_solve_along_axis(
        np.asarray(lower), np.asarray(diag), np.asarray(upper),
        np.asarray(rhs), AXIS, method="pcr")
    err = float(jnp.abs(out - rep).max())
    print(f"   max|sharded {method} - replicated pcr| = {err:.2e}")
