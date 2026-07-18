"""Shared builders for the storage-halo width probe (CPU, periodic)."""
from __future__ import annotations

import numpy as np

import fridom as fr  # noqa: F401 — ensures fridom import side effects
import fridom.nonhydro2 as nh
from fridom.model.modules.advection import (
    CenteredAdvection,
    UpwindAdvection,
)
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

L = 2 * np.pi
COMPONENTS = ("u", "v", "w", "b")


def make_grid(n, length=L):
    """Triperiodic n^3 cube."""
    return Grid(tuple(
        IntervalMesh(n, (0.0, length), periodic=True, name=name)
        for name in ("x", "y", "z")))


def make_model(n, advection, *, dt=0.001):
    """Periodic nonhydro2 model, nodal family, given advection module."""
    return nh.Model(
        grid=make_grid(n), dt=dt, advection=advection,
        family="nodal")


def upwind5():
    return UpwindAdvection(5)


def centered():
    return CenteredAdvection()


def seed_state(model, seed=1):
    """Fill (u, v, w, b) with reproducible random data (true shape)."""
    rng = np.random.default_rng(seed)
    shape = np.asarray(model.state["u"].data).shape
    model.set_fields(**{
        c: rng.standard_normal(shape) for c in COMPONENTS})


def field_absdiff(state_a, state_b, component):
    a = np.asarray(state_a[component].data)
    b = np.asarray(state_b[component].data)
    return float(np.abs(a - b).max()), float(np.abs(a).max())


def widths(model):
    d = model.state["u"].grid.decomposition
    return {name: d.halo[name] for name in ("x", "y", "z")}


def storage_shapes(model):
    return {c: tuple(model.state[c].storage.shape)
            for c in COMPONENTS}
