"""Roofline micros for the RTX 3060 Laptop (f64/f32 triad, FMA, divide).

Each micro jits a kernel, warms it, then times CALLS invocations with
block_until_ready and reports min/median.  The compute micros use a
python-unrolled inner chain inside a lax.fori_loop so arithmetic
intensity is high (register-resident recurrence, ~one HBM read+write per
fori iteration) -> compute-bound.  The triad uses a rotating 3-array
fori carry -> 3*N^3 HBM traffic per iteration -> bandwidth-bound.

Bounded recurrences (no overflow):
  fma:  x = x*a + b   with a~0.5, b~0.25   (fixed point 0.5)
  div:  x = a/x + b    with a~1.5, b~0.3   (continued fraction, bounded)

Usage: JAX_PLATFORMS=cuda python roofline.py
"""
from __future__ import annotations

import json
import os
import statistics
from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)  # f64 arrays must not truncate

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "roofline")
os.makedirs(OUT, exist_ok=True)

N = int(os.environ.get("RL_N", "136"))
NELEM = N ** 3
UNROLL = int(os.environ.get("RL_UNROLL", "256"))
CALLS = int(os.environ.get("RL_CALLS", "12"))
RNG = np.random.default_rng(0)


def _host(lo, hi, dtype):
    a = RNG.uniform(lo, hi, size=(N, N, N))
    return np.asarray(a, dtype=dtype)


def _time(f, args, calls=CALLS):
    out = f(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(calls):
        t0 = perf_counter()
        out = f(*args)
        jax.block_until_ready(out)
        ts.append(perf_counter() - t0)
    return min(ts), statistics.median(ts)


def triad(dtype, T=32):
    itemsize = np.dtype(dtype).itemsize
    a = jnp.asarray(_host(0, 1, dtype))
    b = jnp.asarray(_host(0, 1, dtype))
    c = jnp.asarray(_host(0, 1, dtype))
    q = dtype(0.42)

    @jax.jit
    def f(a, b, c):
        def body(i, carry):
            a, b, c = carry
            return (b, c, b + q * c)
        a, b, c = lax.fori_loop(0, T, body, (a, b, c))
        return a + b + c

    tmin, tmed = _time(f, (a, b, c))
    bytes_moved = 3 * NELEM * itemsize * T
    return {
        "kind": "triad", "dtype": str(np.dtype(dtype)), "T": T,
        "t_min_s": tmin, "t_med_s": tmed,
        "GB_s_min": bytes_moved / tmin / 1e9,
        "GB_s_med": bytes_moved / tmed / 1e9,
    }


def fma(dtype, T=24):
    a = jnp.asarray(_host(0.4, 0.6, dtype))
    b = jnp.asarray(_host(0.2, 0.3, dtype))
    x = jnp.asarray(_host(0.4, 0.6, dtype))

    @jax.jit
    def f(x, a, b):
        def body(i, x):
            for _ in range(UNROLL):
                x = x * a + b
            return x
        return lax.fori_loop(0, T, body, x)

    tmin, tmed = _time(f, (x, a, b))
    flops = 2 * NELEM * T * UNROLL
    return {
        "kind": "fma", "dtype": str(np.dtype(dtype)), "T": T,
        "unroll": UNROLL, "t_min_s": tmin, "t_med_s": tmed,
        "GFLOP_s_min": flops / tmin / 1e9,
        "GFLOP_s_med": flops / tmed / 1e9,
        "GFMA_s_min": (flops / 2) / tmin / 1e9,
    }


def divide(dtype, T=8):
    a = jnp.asarray(_host(1.2, 1.8, dtype))
    b = jnp.asarray(_host(0.2, 0.4, dtype))
    x = jnp.asarray(_host(0.8, 1.2, dtype))

    @jax.jit
    def f(x, a, b):
        def body(i, x):
            for _ in range(UNROLL):
                x = a / x + b
            return x
        return lax.fori_loop(0, T, body, x)

    tmin, tmed = _time(f, (x, a, b))
    divs = NELEM * T * UNROLL
    return {
        "kind": "divide", "dtype": str(np.dtype(dtype)), "T": T,
        "unroll": UNROLL, "t_min_s": tmin, "t_med_s": tmed,
        "Gdiv_s_min": divs / tmin / 1e9,
        "Gdiv_s_med": divs / tmed / 1e9,
    }


def main():
    if (jax.default_backend() != "gpu"
            and os.environ.get("RL_ALLOW_CPU") != "1"):
        raise RuntimeError(f"not on gpu: {jax.default_backend()}")
    dev = jax.devices()[0]
    results = {"device": str(dev), "N": N, "nelem": NELEM,
               "unroll": UNROLL, "calls": CALLS, "micros": []}
    plan = [
        ("triad_f64", lambda: triad(np.float64)),
        ("triad_f32", lambda: triad(np.float32)),
        ("fma_f64", lambda: fma(np.float64)),
        ("fma_f32", lambda: fma(np.float32)),
        ("div_f64", lambda: divide(np.float64)),
        ("div_f32", lambda: divide(np.float32)),
    ]
    for name, fn in plan:
        r = fn()
        r["name"] = name
        results["micros"].append(r)
        key = ("GB_s_min" if r["kind"] == "triad"
               else "GFLOP_s_min" if r["kind"] == "fma"
               else "Gdiv_s_min")
        print(f"[{name}] t_min={r['t_min_s']*1e3:.2f}ms  "
              f"{key}={r[key]:.1f}")
    with open(os.path.join(OUT, "roofline.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote", os.path.join(OUT, "roofline.json"))


if __name__ == "__main__":
    main()
