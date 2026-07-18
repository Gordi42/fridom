"""Task 5 probe: price the CG no-op trips. mg-cusparse 128^3 4-GPU at
budget=100 vs budget=15 (both converge at 10 iters). Same GB-2 protocol
(median of 6x20, compile excluded, block_until_ready)."""
from __future__ import annotations
import sys
sys.path.insert(0, "/work/uo0780/u301533/fridom/census_scratch")
import jax  # noqa: E402
import gb2_common as G  # noqa: E402

n = 128
print("devices:", jax.device_count(), flush=True)
for budget in (100, 15):
    model = G.build_mapped_model(
        n, preconditioner="multigrid", method="cusparse", budget=budget)
    G.set_mapped_ic(model)
    res = G.time_step_ms(model, reps=6, steps=20)
    print(f"budget={budget:3d}  median={res['median']:.2f} ms/step  "
          f"min={res['min']:.2f} max={res['max']:.2f}  "
          f"compile={res['compile_s']:.1f}s  all={[round(x,2) for x in res['all']]}",
          flush=True)
