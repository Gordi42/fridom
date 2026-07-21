"""Golden-file parity against the pre-refactor dev tip (a1b668be).

The nondimensionalization-refactor acceptance gate: the today-parity
spelling on the scaling surface reproduces pre-refactor golden runs
(4 geometries x 2 parameter sets x +/- CoriolisEnergyCorrection, 10
AB3 steps) BITWISE at the dyadic pins (ratified policy: bitwise at
self-normalizing/dyadic pins, <= 2 ulp generic, <= 1 ulp chart).

The golden .npz files are pinned pre-refactor artifacts, not
repository data: point ``FRIDOM_SW_GOLDEN_DIR`` at the capture
directory (scripts + README beside the files) to run this gate; the
tests skip when the variable is unset.
"""
import os
import pathlib

import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw

GOLDEN_DIR = os.environ.get("FRIDOM_SW_GOLDEN_DIR", "")

pytestmark = pytest.mark.skipif(
    not GOLDEN_DIR, reason="FRIDOM_SW_GOLDEN_DIR not set (the golden "
    ".npz files are pre-refactor capture artifacts)")

N = 16
DT = 5e-3
STEPS = 10
F0 = 0.5
SEED = 42

#: parameter sets: A = the dyadic pin, B = the generic set
SETS = {
    "A": {"csqr": 1.0, "rossby_number": 0.25},
    "B": {"csqr": 0.5, "rossby_number": 1.0},
}

#: ratified per-configuration ulp budgets
BUDGET = {"flat": 0, "walled": 0, "immersed": 2, "spherical": 2}


def flat_grid(periodic_y=True):
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), name="x")
    my = fr.spatial.meshes.IntervalMesh(
        N, (0.0, 1.0), periodic=periodic_y, name="y")
    return fr.spatial.Grid((mx, my), device_ids=(0,))


def immersed_grid():
    box = lambda x, y: (  # noqa: E731
        (x > 2) & (x < 10) & (y > 2) & (y < 10)).astype(float)
    im = fr.spatial.meshes.IntervalMesh
    return fr.spatial.Grid(
        (im(12, (0.0, 12.0), periodic=True, name="x"),
         im(12, (0.0, 12.0), periodic=True, name="y")),
        immersed=fr.spatial.ImmersedDomain(box), device_ids=(0,))


def sphere_grid():
    lat_max = float(np.deg2rad(80.0))
    return fr.spatial.spherical.Grid(
        (16, 8), radius=1.0, lat_extent=(-lat_max, lat_max),
        device_ids=(0,))


def build(geom, pset, corr):
    """Build the today-parity spelling on the scaling surface."""
    csqr = SETS[pset]["csqr"]
    ro = SETS[pset]["rossby_number"]
    stepper = fr.model.time_steppers.AdamBashforth(DT, order=3)
    if geom == "spherical":
        return sw.Model(
            grid=sphere_grid(),
            core=sw.Core(froude_number=ro, depth=csqr,
                         coords=("lon", "lat")),
            scaling=fr.scaling.GravityWave(),
            coriolis=sw.modules.RotationCoriolis(
                omega=(0.0, 0.0, F0), coords=("lon", "lat"),
                metric_weight="csqr"),
            advection=True,
            modules_extra=((sw.modules.CoriolisEnergyCorrection(
                coords=("lon", "lat")),) if corr else ()),
            time_stepper=stepper)
    grid = {"flat": lambda: flat_grid(periodic_y=True),
            "walled": lambda: flat_grid(periodic_y=False),
            "immersed": immersed_grid}[geom]()
    return sw.Model(
        grid=grid,
        core=sw.Core(froude_number=ro, depth=csqr),
        scaling=fr.scaling.GravityWave(),
        coriolis=sw.modules.FPlaneCoriolis(rossby_number=ro / F0),
        advection=True,
        modules_extra=((sw.modules.CoriolisEnergyCorrection(),)
                       if corr else ()),
        time_stepper=stepper)


def set_ic(model, geom):
    rng = np.random.default_rng(SEED)
    fields = {
        "u": 0.1 * rng.standard_normal(model.state["u"].shape),
        "v": 0.1 * rng.standard_normal(model.state["v"].shape),
        "p": 0.03 * rng.standard_normal(model.state["p"].shape)}
    if geom == "immersed":
        mask = np.asarray(model.grid.immersed.mask(
            model.state["p"].function_space).data)
        fields["p"] = fields["p"] * mask
    model.set_fields(**fields)


def ulp_distance(a, b):
    """Max units-in-last-place distance between two float64 arrays."""
    if np.array_equal(a, b):
        return 0
    ai = a.view(np.int64)
    bi = b.view(np.int64)
    ai = np.where(ai < 0, np.int64(-(2**63) + 1) - ai - 1, ai)
    bi = np.where(bi < 0, np.int64(-(2**63) + 1) - bi - 1, bi)
    return int(np.max(np.abs(ai - bi)))


@pytest.mark.parametrize("corr", [False, True],
                         ids=["nocorr", "corr"])
@pytest.mark.parametrize("pset", ["A", "B"])
@pytest.mark.parametrize(
    "geom", ["flat", "walled", "immersed", "spherical"])
def test_golden_parity(geom, pset, corr):
    tag = "corr" if corr else "nocorr"
    path = (pathlib.Path(GOLDEN_DIR)
            / f"sw_{geom}_{pset}_{tag}.npz")
    if not path.exists():
        pytest.skip(f"golden file {path.name} not captured")
    ref = np.load(path)
    model = build(geom, pset, corr)
    set_ic(model, geom)
    worst = 0
    for name in ("u", "v", "p"):
        worst = max(worst, ulp_distance(
            np.asarray(model.state[name].data),
            ref[f"init_{name}"]))
    for step in range(1, STEPS + 1):
        model.advance(1)
        for name in ("u", "v", "p"):
            worst = max(worst, ulp_distance(
                np.asarray(model.state[name].data),
                ref[f"s{step:02d}_{name}"]))
    assert worst <= BUDGET[geom], (
        f"sw_{geom}_{pset}_{tag}: {worst} ulp exceeds the "
        f"{BUDGET[geom]}-ulp budget")
