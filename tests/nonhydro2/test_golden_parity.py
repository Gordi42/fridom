"""Golden-file parity against the pre-refactor dev tip (a1b668be).

The Branch-2 nondimensionalization acceptance gate for nonhydro2: the
today-parity spellings on the scaling surface reproduce pre-refactor
golden runs (8^3 triply periodic flat dyadic configs, 10 AB3 steps)
BITWISE (ratified policy: bitwise at the dyadic pins; the flat nh
model has no H7 surface closure, so no closure-row budget applies).

Configs:

- ``nh_flat_dyadic_dim.npz`` (dsqr=0.25, rossby_number=1.0, f0=0.5,
  n2=4.0): the DIMENSIONAL assembly — ``nh.Core(aspect_ratio=0.5)``
  (delta^2 == dsqr exact) + ``FPlaneCoriolis(f0=0.5)`` +
  ``ConstantStratification(n2=4.0)`` — today's rossby_number=1
  (the de-scaled advection carries no factor).
- ``nh_flat_dyadic_rot.npz`` (dsqr=0.25, rossby_number=0.25, f0=1.0,
  n2=4.0): ``Rotational()`` + ``FPlaneCoriolis(rossby_number=0.25)``
  (eps/Ro = 1.0 exact — the rotational frame's f0=1) +
  ``ConstantStratification(froude_number=0.125)``
  ((eps/Fr)^2 = 4.0 exact, dyadic).

The golden .npz files are pinned pre-refactor artifacts, not
repository data: point ``FRIDOM_NH_GOLDEN_DIR`` at the capture
directory (scripts + README beside the files) to run this gate; the
tests skip when the variable is unset.
"""
import os
import pathlib

import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh

GOLDEN_DIR = os.environ.get("FRIDOM_NH_GOLDEN_DIR", "")

pytestmark = pytest.mark.skipif(
    not GOLDEN_DIR, reason="FRIDOM_NH_GOLDEN_DIR not set (the golden "
    ".npz files are pre-refactor capture artifacts)")

N = 8
DT = 2.0 ** -6
STEPS = 10
SEED = 7
COMPONENTS = ("u", "v", "w", "p", "b")


def make_grid():
    im = fr.spatial.meshes.IntervalMesh
    return fr.spatial.Grid(tuple(
        im(N, (0.0, 2.0 * np.pi), periodic=True, name=name)
        for name in ("x", "y", "z")), device_ids=(0,))


def build(config):
    stepper = fr.model.time_steppers.AdamBashforth(DT, order=3)
    if config == "dim":
        return nh.Model(
            grid=make_grid(), core=nh.Core(aspect_ratio=0.5),
            coriolis=nh.FPlaneCoriolis(f0=0.5),
            stratification=nh.ConstantStratification(n2=4.0),
            advection=True, time_stepper=stepper)
    return nh.Model(
        grid=make_grid(), core=nh.Core(aspect_ratio=0.5),
        scaling=fr.scaling.Rotational(),
        coriolis=nh.FPlaneCoriolis(rossby_number=0.25),
        stratification=nh.ConstantStratification(froude_number=0.125),
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


@pytest.mark.parametrize("config", ["dim", "rot"])
def test_golden_parity(config):
    path = (pathlib.Path(GOLDEN_DIR)
            / f"nh_flat_dyadic_{config}.npz")
    if not path.exists():
        pytest.skip(f"golden file {path.name} not captured")
    ref = np.load(path)
    model = build(config)
    rng = np.random.default_rng(SEED)
    model.set_fields(**{
        name: 0.05 * rng.standard_normal(model.state[name].shape)
        for name in ("u", "v", "w", "b")})
    worst = 0
    for name in COMPONENTS:
        worst = max(worst, ulp_distance(
            np.asarray(model.state[name].data), ref[f"init_{name}"]))
    for step in range(1, STEPS + 1):
        model.advance(1)
        for name in COMPONENTS:
            worst = max(worst, ulp_distance(
                np.asarray(model.state[name].data),
                ref[f"s{step:02d}_{name}"]))
    assert worst == 0, (
        f"nh_{config}: {worst} ulp exceeds the bitwise budget")
