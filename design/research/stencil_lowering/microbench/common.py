"""Shared infrastructure for the stencil-lowering micro-benchmarks.

Pure jax/lax only; never imports fridom. f64 enabled by caller before
arrays are created. All timing follows the project methodology:
block_until_ready, >=3 warmup, median of >=30 timed calls, plus a
dispatch-floor probe and a fori_loop per-iteration probe.
"""
from __future__ import annotations

import re
import statistics
from time import perf_counter

import jax
import jax.numpy as jnp

# ================================================================
#  Device sanity
# ================================================================


def check_device() -> None:
    """Abort if the GPU allocation vanished (jax silently falls to cpu)."""
    backend = jax.default_backend()
    count = jax.device_count()
    if backend != "gpu" or count != 1:
        msg = f"expected single gpu, got backend={backend} count={count}"
        raise RuntimeError(msg)


# ================================================================
#  Slicing helper (axis-generic)
# ================================================================


def sl(arr, start, stop, axis):
    idx = [slice(None)] * arr.ndim
    idx[axis] = slice(start, stop)
    return arr[tuple(idx)]


# ================================================================
#  Timing
# ================================================================


def time_median(fn, *args, n_warmup: int = 5, n_timed: int = 40) -> float:
    """Median wall seconds of a single blocking call (dispatch+kernel)."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(n_timed):
        t0 = perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(perf_counter() - t0)
    return statistics.median(times)


def dispatch_floor() -> float:
    """Per-call python->XLA dispatch latency (seconds), tiny kernel."""
    x = jnp.ones((1,), dtype=jnp.float64)
    f = jax.jit(lambda a: a + 1.0)
    f(x).block_until_ready()
    return time_median(f, x, n_warmup=10, n_timed=200)


# ================================================================
#  Roofline: copy / identity and triad
# ================================================================


def measure_roofline(n: int, dtype) -> dict:
    shape = (n, n, n)
    x = jnp.ones(shape, dtype=dtype)
    b = jnp.ones(shape, dtype=dtype)
    c = jnp.full(shape, 2.0, dtype=dtype)
    nbytes = x.size * x.dtype.itemsize

    ident = jax.jit(lambda a: a + 0.0)
    triad = jax.jit(lambda p, q: p + 0.5 * q)
    ident(x).block_until_ready()
    triad(b, c).block_until_ready()

    t_id = time_median(ident, x)
    t_tr = time_median(triad, b, c)
    # identity: read x, write y -> 2 arrays. triad: read b,c, write a -> 3.
    return {
        "n": n,
        "dtype": str(dtype.__name__ if hasattr(dtype, "__name__") else dtype),
        "field_bytes": nbytes,
        "identity_ms": t_id * 1e3,
        "identity_gbs": 2 * nbytes / t_id / 1e9,
        "triad_ms": t_tr * 1e3,
        "triad_gbs": 3 * nbytes / t_tr / 1e9,
    }


# ================================================================
#  HLO op counting
# ================================================================

OPCODES = [
    "fusion", "slice", "concatenate", "dynamic-slice",
    "dynamic-update-slice", "pad", "transpose", "reduce",
    "reduce-window", "convolution", "custom-call", "copy",
    "bitcast", "broadcast", "add", "multiply", "select",
]


def count_ops(hlo: str) -> dict:
    counts = {}
    for op in OPCODES:
        pat = re.compile(r"(?<![\w.\-])" + re.escape(op) + r"\(")
        counts[op] = len(pat.findall(hlo))
    return counts


def hlo_instr_count(hlo: str) -> int:
    """Total number of HLO instruction definitions ('%name = ...')."""
    return len(re.findall(r"^\s*%[\w.\-]+ = ", hlo, flags=re.MULTILINE))


# ================================================================
#  Analysis of one compiled case
# ================================================================


def analyze(fn, args, ideal_from_memstats: bool = True) -> dict:
    """Compile fn(*args) fresh, collect memory/cost/HLO metrics.

    Returns a metrics dict WITHOUT timing (timing done separately so the
    same compiled object is reused).  Also returns the compiled object.
    """
    jitted = jax.jit(fn)
    t0 = perf_counter()
    lowered = jitted.lower(*args)
    compiled = lowered.compile()
    compile_s = perf_counter() - t0

    mem = compiled.memory_analysis()
    try:
        cost = compiled.cost_analysis()
    except Exception:  # noqa: BLE001
        cost = {}
    flops = float(cost.get("flops", 0.0)) if cost else 0.0
    bytes_acc = float(cost.get("bytes accessed", 0.0)) if cost else 0.0

    hlo = compiled.as_text()
    ops = count_ops(hlo)

    jaxpr = jax.make_jaxpr(fn)(*args)
    jaxpr_lines = len(str(jaxpr).splitlines())

    arg_b = mem.argument_size_in_bytes
    out_b = mem.output_size_in_bytes
    metrics = {
        "compile_s": compile_s,
        "temp_bytes": mem.temp_size_in_bytes,
        "arg_bytes": arg_b,
        "out_bytes": out_b,
        "peak_bytes": mem.peak_memory_in_bytes,
        "gen_code_bytes": mem.generated_code_size_in_bytes,
        "flops": flops,
        "bytes_accessed": bytes_acc,
        "jaxpr_lines": jaxpr_lines,
        "hlo_chars": len(hlo),
        "hlo_instrs": hlo_instr_count(hlo),
        "ideal_bytes": arg_b + out_b,
        "ops": ops,
    }
    return compiled, metrics, hlo


def run_case(name: str, fn, args, hlo_dir=None,
             n_timed: int = 40) -> dict:
    """Full single-call case: analyze + time.  Optionally save HLO."""
    check_device()
    compiled, metrics, hlo = analyze(fn, args)
    t = time_median(compiled, *args, n_timed=n_timed)
    metrics["ms_single"] = t * 1e3
    metrics["gbs_single"] = metrics["ideal_bytes"] / t / 1e9
    metrics["name"] = name
    if hlo_dir is not None:
        path = f"{hlo_dir}/{name}.txt"
        with open(path, "w") as fh:  # noqa: PTH123
            fh.write(hlo)
        metrics["hlo_path"] = path
    # keep a truncated fusion-body sample for the report
    metrics["hlo_head"] = hlo[:0]  # not stored inline; full text saved
    return metrics


# ================================================================
#  fori_loop per-iteration probe (shape-preserving)
# ================================================================


def loop_periter(apply_once, x0, T: int = 50, n_timed: int = 30) -> dict:
    """Time T back-to-back shape-preserving applications in one jit.

    Returns per-iteration seconds (loop_time / T) and the single loop
    wall.  apply_once must map array->array of identical shape.
    """
    def looped(x):
        return jax.lax.fori_loop(0, T, lambda i, s: apply_once(s), x)

    jitted = jax.jit(looped)
    compiled = jitted.lower(x0).compile()
    t = time_median(compiled, x0, n_timed=n_timed)
    return {"loop_ms": t * 1e3, "periter_ms": t * 1e3 / T, "T": T}
