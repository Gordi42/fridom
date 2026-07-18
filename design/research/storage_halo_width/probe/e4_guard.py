"""E4: forcing width 2 for upwind5 must trip the stencil-reach guard.

The biased order-5 face reconstruction reaches 3 cells/side; the
staggering/reconstruct guard raises when the per-side reach exceeds the
negotiated storage width. Force width 2 and capture the error verbatim.
"""
from __future__ import annotations

import sys
import traceback

import _common as C
from _force_halo import force_halo


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    print(f"=== E4 guard, upwind5 forced width 2, n={n}^3 ===\n")
    try:
        with force_halo(2):
            model = C.make_model(n, C.upwind5())
            C.seed_state(model)
            print(f"  widths after negotiate: {C.widths(model)}")
            model.run(steps=1, progress=False)
        print("  NO ERROR RAISED (unexpected)")
    except Exception as exc:  # noqa: BLE001
        print(f"  RAISED {type(exc).__name__}:")
        print(f"  {exc}")
        print("\n  --- traceback tail ---")
        tb = traceback.format_exc().strip().splitlines()
        for line in tb[-12:]:
            print(f"  {line}")
