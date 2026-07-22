"""The thickness DIAGNOSE stage is device-count invariant (forced-4).

Prefix-mirrored shard of ``sw/modules/core.py``: one 2-rank nonlinear
step-parity gate exercising the per-substage thickness exchange (the
DIAGNOSE-stage field is consumed at reach 2 by the Sadourny corner
chain, so its halo must be provisioned and exchanged under a sharded
decomposition exactly like the old inline compute).
"""
import jax
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw

N = 16
DT = 5e-3
NAMES = ("u", "v", "p")


@pytest.mark.multi_device
def test_nonlinear_thickness_run_is_device_count_invariant(
        forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices

    def build(device_ids):
        mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), name="x")
        my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                            periodic=False, name="y")
        return sw.Model(
            grid=fr.spatial.Grid((mx, my), device_ids=device_ids),
            core=sw.Core(froude_number=0.25),
            scaling=fr.scaling.GravityWave(),
            coriolis=sw.modules.FPlaneCoriolis(rossby_number=0.5),
            advection=True,
            time_stepper=fr.model.time_steppers.AdamBashforth(
                DT, order=3))

    rng = np.random.default_rng(4)
    fields = {"u": 0.1 * rng.standard_normal((N, N)),
              "v": 0.1 * rng.standard_normal((N, N - 1)),
              "p": 0.03 * rng.standard_normal((N, N))}
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        model = build(device_ids)
        model.set_fields(**fields)
        model.advance(4)
        results[tag] = {c: np.asarray(model.state[c].data)
                        for c in NAMES}
        if tag == "many":
            assert (model.state["u"]._data.sharding.spec[0]
                    == "devices")
    assert max(
        float(np.abs(results["many"][c] - results["one"][c]).max())
        for c in NAMES) < 1e-12
