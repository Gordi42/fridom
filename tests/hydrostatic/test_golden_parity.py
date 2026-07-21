"""Golden-file parity against the pre-refactor dev tip (a1b668be).

The Branch-2 nondimensionalization acceptance gate for hydrostatic:
the today-parity spellings on the gravity-first scaling surface
reproduce pre-refactor golden runs (8x8x4 flat dyadic configs, unit
depth, explicit free surface, 10 AB3 steps). Ratified policy: bitwise
at the dyadic pins; the ONE exception is the H7 surface-closure
boundary row of the default advection (its traced-scalar multiply
folded away in the dimensional variant — a <= 1 ulp closure-row seed,
accepted), so the closure-ON dimensional config carries a small
roundoff budget while the legacy-closure config is bitwise.

Configs (all golden runs used csqr = c^2 = g * H with H = 1, so
``gravity`` equals the captured ``csqr``):

- ``hy_flat_dyadic_dim_nosf.npz`` (csqr=1, ro=1, f0=0.5, n2=4,
  surface_advective_flux=False): DIMENSIONAL, bitwise.
- ``hy_flat_dyadic_dim.npz`` (same, default closure): DIMENSIONAL,
  the H7 closure-row roundoff budget.
- ``hy_flat_dyadic.npz`` (csqr=1, ro=0.25, f0=0.5): ExternalWave()
  — eps = Fr_ext = 0.25 ((eps/Fr_ext)^2 = 1 = csqr, self-normalized),
  Coriolis Ro = eps/f0 = 0.5, Fr_int = eps/sqrt(n2) = 0.125; bitwise.
- ``hy_flat_dyadic_rot.npz`` (csqr=0.25, ro=0.25, f0=1.0):
  Rotational() — eps = Ro = 0.25, Fr_ext = eps/sqrt(csqr) = 0.5
  ((eps/Fr_ext)^2 = 0.25 exact), Fr_int = 0.125; bitwise.

Point ``FRIDOM_HY_GOLDEN_DIR`` at the capture directory (scripts +
README beside the files); the tests skip when the variable is unset.
"""
import os
import pathlib

import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy

GOLDEN_DIR = os.environ.get("FRIDOM_HY_GOLDEN_DIR", "")

pytestmark = pytest.mark.skipif(
    not GOLDEN_DIR, reason="FRIDOM_HY_GOLDEN_DIR not set (the golden "
    ".npz files are pre-refactor capture artifacts)")

DT = 2.0 ** -7
STEPS = 10
SEED = 9
COMPONENTS = ("u", "v", "w", "p_hyd", "b", "ps")

#: per-config ulp budgets: bitwise everywhere except the accepted H7
#: closure-row seed of the closure-ON dimensional config (measured:
#: <= 64 ulp after 10 AB3 steps at ~3e-17 absolute)
BUDGET = {"dim_nosf": 0, "dim": 128, "ext": 0, "rot": 0}


def make_grid():
    im = fr.spatial.meshes.IntervalMesh
    return fr.spatial.Grid((
        im(8, (0.0, 1.0), periodic=True, name="x"),
        im(8, (0.0, 1.0), periodic=True, name="y"),
        im(4, (0.0, 1.0), periodic=False, name="z")),
        device_ids=(0,))


def build(config):
    stepper = fr.model.time_steppers.AdamBashforth(DT, order=3)
    if config in ("dim", "dim_nosf"):
        return hy.Model(
            grid=make_grid(), core=hy.Core(gravity=1.0),
            coriolis=hy.FPlaneCoriolis(f0=0.5),
            stratification=hy.ConstantStratification(n2=4.0),
            free_surface=hy.ExplicitFreeSurface(),
            advection=True,
            surface_advective_flux=(False if config == "dim_nosf"
                                    else None),
            time_stepper=stepper)
    if config == "ext":
        return hy.Model(
            grid=make_grid(), core=hy.Core(),
            scaling=fr.scaling.ExternalWave(),
            coriolis=hy.FPlaneCoriolis(rossby_number=0.5),
            stratification=hy.ConstantStratification(
                froude_number=0.125),
            free_surface=hy.ExplicitFreeSurface(froude_number=0.25),
            advection=True, time_stepper=stepper)
    return hy.Model(
        grid=make_grid(), core=hy.Core(),
        scaling=fr.scaling.Rotational(),
        coriolis=hy.FPlaneCoriolis(rossby_number=0.25),
        stratification=hy.ConstantStratification(froude_number=0.125),
        free_surface=hy.ExplicitFreeSurface(froude_number=0.5),
        advection=True, time_stepper=stepper)


def ulp_distance(a, b):
    """Max units-in-last-place distance between two float64 arrays."""
    if np.array_equal(a, b):
        return 0
    ai = a.view(np.int64)
    bi = b.view(np.int64)
    ai = np.where(ai < 0, np.int64(-(2**63) + 1) - ai - 1, ai)
    bi = np.where(bi < 0, np.int64(-(2**63) + 1) - bi - 1, bi)
    return int(np.max(np.abs(ai - bi)))


_FILES = {
    "dim_nosf": "hy_flat_dyadic_dim_nosf.npz",
    "dim": "hy_flat_dyadic_dim.npz",
    "ext": "hy_flat_dyadic.npz",
    "rot": "hy_flat_dyadic_rot.npz",
}


@pytest.mark.parametrize("config", list(_FILES))
def test_golden_parity(config):
    path = pathlib.Path(GOLDEN_DIR) / _FILES[config]
    if not path.exists():
        pytest.skip(f"golden file {path.name} not captured")
    ref = np.load(path)
    model = build(config)
    rng = np.random.default_rng(SEED)
    model.set_fields(**{
        name: 0.02 * rng.standard_normal(model.state[name].shape)
        for name in ("u", "v", "b")})
    worst = 0
    for step in range(1, STEPS + 1):
        model.advance(1)
        for name in COMPONENTS:
            key = f"s{step:02d}_{name}"
            if key not in ref:
                continue
            worst = max(worst, ulp_distance(
                np.asarray(model.state[name].data), ref[key]))
    assert worst <= BUDGET[config], (
        f"hy_{config}: {worst} ulp exceeds the "
        f"{BUDGET[config]}-ulp budget")
