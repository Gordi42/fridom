"""Compile-only driver for the buffer-assignment attribution.

Builds the matched-config nonhydro2 model, applies the p3b patch (if any),
and runs advance(50) once so XLA compiles the 50-step chunk. XLA_FLAGS
(set by the caller) dumps HLO + buffer-assignment to disk. No timing.

Verifies fridom resolves inside wt-attrib and aborts loudly otherwise.
"""
from __future__ import annotations

import os
import sys

WT = os.environ["WT_SRC"]
sys.path.insert(0, WT)  # belt-and-suspenders over PYTHONPATH

import fridom as fr  # noqa: E402

_ff = os.path.abspath(fr.__file__)
if not _ff.startswith(os.path.abspath(WT)):
    raise SystemExit(f"ABORT: fridom resolved to {_ff}, not under {WT}")
print(f"OK fridom.__file__ = {_ff}", flush=True)

# import the harness (same dir); reuse its build_model + patch machinery
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import p3b_model as P  # noqa: E402


def main() -> None:
    variant = sys.argv[1]
    n = int(sys.argv[2])
    scheme, kind = P._VARIANTS[variant]
    if kind is not None:
        P.apply_patch(kind)
    model = P.build_model(n, P._advection(scheme))
    P._block(model)
    model.advance(50)          # compile + run the 50-step chunk
    P._block(model)
    exe = P._chunk_exe(50)
    mem = exe.memory_analysis()
    print(f"[{variant} n{n}] temp={mem.temp_size_in_bytes/1e6:.1f}MB "
          f"peak={mem.peak_memory_in_bytes/1e6:.1f}MB "
          f"trace_calls={P._PATCH_CALLS['n']}", flush=True)


if __name__ == "__main__":
    main()
