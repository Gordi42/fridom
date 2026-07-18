"""E1: forced-narrow-halo parity of the final model state.

Baseline (natural negotiated width) vs a forced-narrower storage width,
periodic nonhydro2, family="nodal", n^3, N_STEPS steps. Reports per-field
max abs diff of the final state. Identical physics => 0.0 (or ~1e-16).
"""
from __future__ import annotations

import sys

import _common as C
from _force_halo import force_halo


def run_variant(n, advection_factory, cap, seed=1, steps=10):
    if cap is None:
        model = C.make_model(n, advection_factory())
        C.seed_state(model, seed)
        w = C.widths(model)
        model.run(steps=steps, progress=False)
    else:
        with force_halo(cap):
            model = C.make_model(n, advection_factory())
            C.seed_state(model, seed)
            w = C.widths(model)
            model.run(steps=steps, progress=False)
    final = {c: model.state[c] for c in C.COMPONENTS}
    return w, final


def compare(n, name, factory, base_cap, forced_cap, steps=10):
    print(f"=== {name}: n={n}^3, {steps} steps ===")
    wb, base = run_variant(n, factory, base_cap, steps=steps)
    wf, forced = run_variant(n, factory, forced_cap, steps=steps)
    print(f"  baseline widths (natural): {wb}")
    print(f"  forced   widths (cap={forced_cap}): {wf}")
    for c in C.COMPONENTS:
        d, mag = C.field_absdiff(base, forced, c)
        rel = d / mag if mag > 0 else 0.0
        print(f"  {c}: max|diff|={d:.3e}  max|base|={mag:.3e}  "
              f"rel={rel:.3e}")
    print()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    # upwind5: natural width 4 -> forced 3
    compare(n, "UpwindAdvection(5)", C.upwind5,
            base_cap=None, forced_cap=3, steps=steps)
    # centered: natural width 2 -> forced 1
    compare(n, "CenteredAdvection", C.centered,
            base_cap=None, forced_cap=1, steps=steps)
