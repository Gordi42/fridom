"""Residual steep-slope behaviour after the A0 fix.

The 20% bump is stable indefinitely; a 40% bump still fails, much
later. This probe separates a CFL failure (dt-dependent) from a
remaining spatial instability (dt-independent), by comparing growth at
matched PHYSICAL times.
"""
from __future__ import annotations

import sys

import numpy as np

from clean_repro import make_mapped_model


def run(n, dt, amp, t_end, n_report=8):
    m = make_mapped_model(n, dt, amp=amp)
    m.set_fields(u=0.4)
    steps = int(round(t_end / dt))
    chunk = max(1, steps // n_report)
    print(f"n={n} dt={dt} amp={amp}  ({steps} steps to t={t_end})",
          flush=True)
    done = 0
    while done < steps:
        take = min(chunk, steps - done)
        try:
            m.advance(steps=take)
        except Exception as e:  # noqa: BLE001
            print(f"  t={done * dt:6.2f}  {type(e).__name__}: "
                  f"{str(e)[:80]}", flush=True)
            return
        done += take
        u = np.abs(np.asarray(m.state["u"].data))
        print(f"  t={done * dt:6.2f}  |u|max = {u.max():.6e}", flush=True)
        if not np.isfinite(u.max()):
            return


def main():
    run(32, 0.02, 0.4, 8.0)
    run(32, 0.005, 0.4, 8.0)
    run(64, 0.01, 0.4, 8.0)
    run(32, 0.02, 0.3, 8.0)


if __name__ == "__main__":
    sys.exit(main())
