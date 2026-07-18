"""Derived ``extra_halo`` of the shallowwater2 dynamical core.

The chart / immersed gravity term resolves rows the halo trace cannot
follow, so the core declares its own ghost width -- now DERIVED at bind
from the order-2 ``diff`` rows the term applies (width 1), not a
hardcoded 2. On a plain flat grid the term is traced and no substitute
is declared (``None``). See ``pressure_solver_halo.md``.
"""
import numpy as np

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh


def _mesh(n, name, *, periodic=True):
    return IntervalMesh(n, (0.0, 1.0), periodic=periodic, name=name)


def _flat_model(**kwargs):
    grid = Grid((_mesh(8, "x"), _mesh(8, "y")))
    return sw.Model(grid=grid, coords=("x", "y"), csqr=0.7,
                    rossby_number=0.2, coriolis=None, advection=False,
                    **kwargs)


# ================================================================
#  Flat grid: fully traced, no substitute declared
# ================================================================
def test_flat_core_declares_no_extra_halo():
    core = _flat_model().module(sw.modules.DynamicalCore)
    assert core.extra_halo is None


# ================================================================
#  Orthogonal vs non-orthogonal chart: both derive width 1
# ================================================================
def test_non_orthogonal_chart_derives_width_one():
    # a sheared (non-orthogonal) chart: raise_index composes a cross-
    # interpolation, but its window is opposite-biased to the gradient
    # difference and telescopes two-sided back to reach 1 -- so the
    # derived extra_halo is 1, not the naive diff + interp sum of 2
    # (pressure_solver_halo.md; bitwise-verified in that record).
    grid = Grid(
        (_mesh(8, "x"), _mesh(8, "y")),
        mapping=fr.spatial.CoordinateMapping(
            chart={"X": lambda x, y: (x + 0.4 * y, y, 0.0 * x)}))
    model = sw.Model(grid=grid, coords=("x", "y"), csqr=0.7,
                     rossby_number=0.2, coriolis=None, advection=False)
    # the cross-interp is genuinely present (a non-diagonal metric)
    assert not grid.mapping.orthogonal
    core = model.module(sw.modules.DynamicalCore)
    assert dict(core.extra_halo.widths) == {"x": 1, "y": 1}
    assert model.grid.decomposition.halo["x"] == 1
    assert model.grid.decomposition.halo["y"] == 1


# ================================================================
#  Immersed grid: masked continuity is exempt, derives width 1
# ================================================================
def test_immersed_core_derives_width_one():
    grid = Grid((_mesh(8, "x"), _mesh(8, "y")),
                immersed=ImmersedDomain(lambda x, y: x * 0.0 + 1.0))  # noqa: ARG005
    model = sw.Model(grid=grid, coords=("x", "y"), csqr=0.7,
                     rossby_number=0.2, coriolis=None, advection=False)
    core = model.module(sw.modules.DynamicalCore)
    assert core.extra_halo is not None
    assert dict(core.extra_halo.widths) == {"x": 1, "y": 1}


# ================================================================
#  Parity: derived width 1 reproduces the old width 2 (bitwise)
# ================================================================
def test_non_orthogonal_chart_parity_with_forced_width_two():
    def build():
        grid = Grid(
            (_mesh(8, "x"), _mesh(8, "y")),
            mapping=fr.spatial.CoordinateMapping(
                chart={"X": lambda x, y: (x + 0.4 * y, y, 0.0 * x)}))
        return sw.Model(grid=grid, coords=("x", "y"), csqr=0.7,
                        rossby_number=0.2, coriolis=None, advection=False)

    def run():
        m = build()
        rng = np.random.default_rng(0)
        m.set_fields(**{c: 0.05 * rng.standard_normal(m.state[c].shape)
                        for c in ("u", "v", "p")})
        m.run(steps=8, progress=False)
        return {c: np.asarray(m.state[c].data) for c in ("u", "v", "p")}

    derived = run()
    cls = sw.modules.DynamicalCore
    orig = cls.__dict__.get("extra_halo")
    try:
        cls.extra_halo = property(
            lambda self: (None if self._extra_halo is None
                          else HaloSpec(dict.fromkeys(self._coords, 2))))
        forced = run()
    finally:
        cls.extra_halo = orig
    md = max(float(np.max(np.abs(derived[c] - forced[c])))
             for c in ("u", "v", "p"))
    assert md == 0.0  # storage-frame bitwise identical
