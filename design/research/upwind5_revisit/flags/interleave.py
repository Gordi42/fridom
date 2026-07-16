"""Thermally-controlled interleave: baseline brackets every candidate.

Runs a fresh python per (label) via the p3b harness, records median/min
ms/step and GPU temp before/after, into interleave.jsonl.  Every
candidate is bracketed by an empty-flags baseline immediately before and
after, so its thermal-corrected delta is (median - mean(prev, next)).
"""
from __future__ import annotations

import json
import subprocess
import sys

SP = ("/tmp/claude-1000/-home-silvano-Projects-fridom/"
      "84ef5624-971a-4abb-a5b0-6cbecbfad680/scratchpad")
PY = f"{SP}/venv-gpu/bin/python"
HARNESS = f"{SP}/fridom_bench/p3b_model.py"
WT = f"{SP}/wt-flags/src"
RES = f"{SP}/fridom_bench/results"
OUT = f"{SP}/flag_sweep/interleave.jsonl"

CANDS = [
    ("f1_dblbuf", "--xla_gpu_enable_while_loop_double_buffering=true"),
    ("f2_nomof", "--xla_disable_hlo_passes=multi_output_fusion"),
    ("f3_lhs", "--xla_gpu_enable_latency_hiding_scheduler=true"),
    ("f4_slop95", "--xla_gpu_memory_limit_slop_factor=95"),
    ("f5_autotune4", "--xla_gpu_autotune_level=4"),
    ("f6_notriton", "--xla_gpu_enable_triton_gemm=false"),
    ("x1_nocmdbuf", "--xla_gpu_enable_command_buffer="),
    ("x2_aliasscope", "--xla_llvm_enable_alias_scope_metadata=true"),
]


def smi():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip()
    clk, temp, pwr = (x.strip() for x in out.split(","))
    return {"clk": float(clk), "temp": float(temp), "pwr": float(pwr)}


def run(variant, label, flags):
    env = {
        "PATH": "/usr/bin:/bin", "PYTHONPATH": WT, "JAX_PLATFORMS": "cuda",
        "JAX_ENABLE_X64": "true", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_FLAGS": flags,
    }
    before = smi()
    p = subprocess.run(
        [PY, HARNESS, "time", "--variant", variant, "--n", "160",
         "--chunks", "4"],
        capture_output=True, text=True, env=env, timeout=900)
    after = smi()
    jf = f"{RES}/p3b_time_{variant}_n160.json"
    d = json.load(open(jf))
    rec = {
        "variant": variant, "label": label, "flags": flags,
        "median": d["ms_per_step_median"], "min": d["ms_per_step_min"],
        "all": d["ms_per_step_all"],
        "temp_bytes": d["chunk"].get("temp_bytes"),
        "fridom_file": d["fridom_file"],
        "t_before": before["temp"], "t_after": after["temp"],
        "clk_before": before["clk"], "clk_after": after["clk"],
        "rc": p.returncode,
    }
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"{variant:12s} {label:14s} med={rec['median']:.3f} "
          f"min={rec['min']:.3f} T={before['temp']:.0f}->{after['temp']:.0f} "
          f"clk={after['clk']:.0f}", flush=True)
    return rec


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "u5_baseline"
    open(OUT, "w").close() if variant == "u5_baseline" else None
    # warmup to reach thermal steady state (discarded)
    print("# warmup", flush=True)
    for _ in range(2):
        run(variant, "warmup", "")
    # fully-bracketed interleave
    run(variant, "base", "")
    for label, flags in CANDS:
        run(variant, label, flags)
        run(variant, "base", "")


if __name__ == "__main__":
    main()
