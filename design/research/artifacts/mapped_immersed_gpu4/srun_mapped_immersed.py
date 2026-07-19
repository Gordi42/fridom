r"""Mapped + immersed composition: REAL multi-process (srun -n 4) leg.

The multi-process sibling of ``smoke_mapped_immersed.py``: N OS
processes, one GPU each, every array sharded across processes. The two
model legs are built by the SAME builders as the smoke script — this
module imports ``build_model_N`` / ``build_model_H`` / ``theta_mass``
(and the leg component tuples and step count) VERBATIM from
``smoke_mapped_immersed`` rather than duplicating them, so the sharded
run steps a byte-identical model to the 1-GPU reference run.

``jax.distributed.initialize()`` is called BEFORE importing fridom (or
the smoke module, which imports fridom at its top): fridom touches the
backend at import. Bare SLURM auto-detect segfaults on these nodes (IPv6
``[::]`` bind), so we initialize explicitly per AGENTS.md, with a
deterministic coordinator port derived from the job id (all ranks agree,
no file race).

Final global fields are gathered with
``process_allgather(arr, tiled=True)`` — never ``np.asarray`` a global
sharded array. Rank 0 loads the 1-GPU ``.npy`` references written by
``smoke_mapped_immersed.py --write-ref`` and prints per-component
max-abs diffs, the theta-mass drift under sharding, and one final
``PASS`` / ``FAIL`` line (per-component threshold 1e-8).

Launch (from the main checkout, inside the allocation):
  XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion JAX_PLATFORMS=cuda \
    timeout 1200 srun -n 4 --gpu-bind=none .venv/bin/python <this>
"""
import os
import sys

import jax

# --- explicit distributed init BEFORE importing fridom ----------------
ntasks = int(os.environ["SLURM_NTASKS"])
procid = int(os.environ["SLURM_PROCID"])
localid = int(os.environ["SLURM_LOCALID"])

# deterministic coordinator port from the job id (all ranks agree, no
# file race). SLURM_JOB_ID is identical across tasks of one step.
jobid = int(os.environ["SLURM_JOB_ID"])
stepid = int(os.environ.get("SLURM_STEPID", os.environ.get("SLURM_STEP_ID", 0)))
port = 29500 + ((jobid + stepid * 7919) % 3000)
ART = os.path.dirname(os.path.abspath(__file__))

jax.distributed.initialize(
    coordinator_address=f"localhost:{port}",
    num_processes=ntasks,
    process_id=procid,
    local_device_ids=[localid],
)

# --- now import fridom (via the shared smoke builders) ----------------
import numpy as np  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402

# path-insensitive import of the smoke module (same directory as this
# file); it imports fridom at its top, which is why it is imported only
# AFTER jax.distributed.initialize() above.
sys.path.insert(0, ART)
import smoke_mapped_immersed as smoke  # noqa: E402

THRESHOLD = 1e-8


def _mark(pidx, msg):
    """Rank-0 progress marker so a hang is localizable in the log."""
    if pidx == 0:
        print(f"[rank0] {msg}", flush=True)


def _run_leg(pidx, name, model, comps, tag, steps):
    """Advance one leg; gather finals; rank 0 compares vs the ref.

    Returns the per-leg (max_abs_diff, mass_drift) on rank 0, or
    (None, mass_drift) on the other ranks.
    """
    _mark(pidx, f"leg {name}: theta-mass before")
    mass_before = smoke.theta_mass(model)
    _mark(pidx, f"leg {name}: advance {steps} steps")
    model.advance(steps)
    mass_after = smoke.theta_mass(model)
    drift = mass_after - mass_before
    if pidx == 0:
        print(f"[{name}] panicked: {model.panicked}")
        print(f"[{name}] theta-mass drift under sharding: {drift:.3e} "
              f"(before={mass_before:.12e} after={mass_after:.12e})")

    _mark(pidx, f"leg {name}: gather finals")
    maxdiff = 0.0
    for c in comps:
        local = model.state[c].data
        glob = multihost_utils.process_allgather(local, tiled=True)
        if pidx == 0:
            ref = np.load(os.path.join(ART, f"{tag}_{c}.npy"))
            g = np.asarray(glob)
            if g.shape != ref.shape:
                print(f"[{name}]   {c}: SHAPE MISMATCH gathered={g.shape} "
                      f"ref={ref.shape}")
                maxdiff = float("inf")
                continue
            d = float(np.abs(g - ref).max())
            print(f"[{name}]   max_abs_diff {c}: {d:.3e} (shape {g.shape})")
            maxdiff = max(maxdiff, d)
    if pidx == 0:
        print(f"[{name}] OVERALL srun max_abs_diff vs 1-GPU ref: "
              f"{maxdiff:.3e}")
        return maxdiff, drift
    return None, drift


def main():
    pidx = jax.process_index()
    if pidx == 0:
        print(f"process_count={jax.process_count()} "
              f"device_count={jax.device_count()} "
              f"local_devices={jax.local_devices()}", flush=True)
        print(f"config: steps={smoke.STEPS} threshold={THRESHOLD}",
              flush=True)

    # ---- Leg N: nonhydro2 composed --------------------------------
    _mark(pidx, "build model N")
    model_n = smoke.build_model_N()
    diff_n, _drift_n = _run_leg(
        pidx, "N", model_n, smoke.COMPS_N, "refN", smoke.STEPS)

    # ---- Leg H: hydrostatic M5 ------------------------------------
    _mark(pidx, "build model H")
    model_h = smoke.build_model_H()
    diff_h, _drift_h = _run_leg(
        pidx, "H", model_h, smoke.COMPS_H, "refH", smoke.STEPS)

    if pidx == 0:
        worst = max(diff_n, diff_h)
        verdict = "PASS" if worst <= THRESHOLD else "FAIL"
        print(f"OVERALL worst max_abs_diff (both legs): {worst:.3e}")
        print(verdict)


if __name__ == "__main__":
    main()
