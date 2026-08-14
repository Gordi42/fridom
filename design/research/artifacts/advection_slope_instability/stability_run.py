"""Forward-integration stability check on the mapped grid (A0)."""
from __future__ import annotations

import sys

from clean_repro import make_flat_model, make_mapped_model, run


def main():
    for n, dt in ((32, 0.02), (64, 0.01)):
        print(f"mapped WALLED n={n} dt={dt}", flush=True)
        try:
            run(make_mapped_model(n, dt), 0.4, 600, 150)
        except Exception as e:  # noqa: BLE001
            print("  ", type(e).__name__, str(e)[:110], flush=True)
    print("mapped PERIODIC column n=32 dt=0.02", flush=True)
    try:
        run(make_mapped_model(32, 0.02, periodic_column=True),
            0.4, 600, 150)
    except Exception as e:  # noqa: BLE001
        print("  ", type(e).__name__, str(e)[:110], flush=True)
    print("flat n=32 dt=0.02", flush=True)
    run(make_flat_model(32, 0.02), 0.4, 600, 300)
    print("mapped WALLED n=32 amp=0.4 dt=0.02", flush=True)
    try:
        run(make_mapped_model(32, 0.02, amp=0.4), 0.4, 600, 150)
    except Exception as e:  # noqa: BLE001
        print("  ", type(e).__name__, str(e)[:110], flush=True)


if __name__ == "__main__":
    sys.exit(main())
