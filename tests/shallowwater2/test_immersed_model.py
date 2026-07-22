"""Gates for the immersed (cut-cell) shallowwater2 model (stage I4).

The IP-D10 end-to-end gates: the factory installs the shared
``MaskState`` (gate g), a nonlinear masked run conserves ``sum theta V
p`` to machine zero (gate a), a balanced state stays steady away from
the mask (gate b), an all-wet immersed model reproduces the unimmersed
run (gate c), a staircase channel matches the walled model (gate d),
genuine lateral partial cells run stably and conserve mass (gate e),
dry DOFs stay exactly zero (gate f), and the eigenmode / background
taught errors are pinned (IP-D8).
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

TWO_PI = 2.0 * np.pi


def _mesh(n, a, b, *, periodic, name):
    return IntervalMesh(n, (a, b), periodic=periodic, name=name)


def _model(grid, *, csqr=0.8, rossby=0.3, f0=1.0, dt=0.01, order=3,
           advection=True, **kwargs):
    return sw.Model(
        grid=grid,
        core=sw.Core(froude_number=rossby, depth=csqr),
        scaling=fr.scaling.GravityWave(),
        coriolis=sw.modules.FPlaneCoriolis(rossby_number=rossby / f0),
        advection=advection,
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=order),
        **kwargs)


def _theta_mass(model):
    """Return ``sum_c theta_c V_c p_c`` (the cut-cell mass)."""
    p = model.state["p"]
    theta = model.grid.immersed.fraction(p.function_space)
    return float(jnp.sum((theta * p).integrate().data))


# ================================================================
#  Factory wiring (IP-D5) / gate g
# ================================================================
def test_immersed_factory_installs_maskstate():
    grid = Grid((_mesh(8, 0.0, 8.0, periodic=True, name="x"),
                 _mesh(8, 0.0, 8.0, periodic=True, name="y")),
                immersed=ImmersedDomain(lambda x, y: x * 0.0 + 1.0))  # noqa: ARG005
    model = _model(grid, advection=True)
    names = [type(m).__name__ for m in model.modules]
    assert "MaskState" in names
    # MaskState is appended last (runs after the physics)
    assert names[-1] == "MaskState"


def test_unimmersed_factory_has_no_maskstate():
    grid = Grid((_mesh(8, 0.0, 8.0, periodic=True, name="x"),
                 _mesh(8, 0.0, 8.0, periodic=True, name="y")))
    model = _model(grid)
    assert not any(type(m).__name__ == "MaskState" for m in model.modules)


# ================================================================
#  Gate a: theta-weighted mass conserved to machine zero (nonlinear)
# ================================================================
@pytest.mark.parametrize("advection", [False, True])
def test_masked_mass_is_conserved_to_machine_zero(advection):
    # dry outside 2 < x < 10, 2 < y < 10 on a 12x12 periodic grid
    box = lambda x, y: (  # noqa: E731
        (x > 2) & (x < 10) & (y > 2) & (y < 10)).astype(float)
    grid = Grid((_mesh(12, 0.0, 12.0, periodic=True, name="x"),
                 _mesh(12, 0.0, 12.0, periodic=True, name="y")),
                immersed=ImmersedDomain(box))
    model = _model(grid, advection=advection)
    rng = np.random.default_rng(0)
    mask = np.asarray(
        grid.immersed.mask(model.state["p"].function_space).data)
    model.set_fields(
        p=0.1 * rng.standard_normal(model.state["p"].data.shape) * mask,
        u=0.1 * rng.standard_normal(model.state["u"].data.shape),
        v=0.1 * rng.standard_normal(model.state["v"].data.shape))
    before = _theta_mass(model)
    model.advance(30)
    assert not model.panicked
    after = _theta_mass(model)
    assert abs(after - before) <= 1e-13 * max(abs(before), 1.0)


# ================================================================
#  Gate f: dry DOFs stay exactly zero over a multi-step run
# ================================================================
@pytest.mark.parametrize("advection", [False, True])
def test_dry_dofs_stay_exactly_zero(advection):
    box = lambda x, y: (  # noqa: E731
        (x > 2) & (x < 10) & (y > 2) & (y < 10)).astype(float)
    grid = Grid((_mesh(12, 0.0, 12.0, periodic=True, name="x"),
                 _mesh(12, 0.0, 12.0, periodic=True, name="y")),
                immersed=ImmersedDomain(box))
    model = _model(grid, advection=advection)
    rng = np.random.default_rng(1)
    model.set_fields(**{
        k: 0.2 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "p")})
    model.advance(12)
    assert not model.panicked
    immersed = grid.immersed
    for name in ("u", "v", "p"):
        field = model.state[name]
        mask = immersed.mask(field.function_space)
        dry = np.asarray(field.data) * (1.0 - np.asarray(mask.data))
        assert np.abs(dry).max() == 0.0


# ================================================================
#  Gate c: all-wet immersed reproduces the unimmersed run
# ================================================================
@pytest.mark.parametrize("advection", [False, True])
def test_all_wet_immersed_matches_unimmersed(advection):
    def meshes():
        return (_mesh(10, 0.0, TWO_PI, periodic=True, name="x"),
                _mesh(10, 0.0, TWO_PI, periodic=False, name="y"))

    allwet = ImmersedDomain(lambda x, y: x * 0.0 + 1.0)  # noqa: ARG005
    im = _model(Grid(meshes(), immersed=allwet), advection=advection)
    un = _model(Grid(meshes()), advection=advection)
    rng = np.random.default_rng(7)
    ic = {k: 0.2 * rng.standard_normal(im.state[k].data.shape)
          for k in ("u", "v", "p")}
    im.set_fields(**ic)
    un.set_fields(**ic)
    im.advance(12)
    un.advance(12)
    for k in ("u", "v", "p"):
        diff = np.abs(np.asarray(im.state[k].data)
                      - np.asarray(un.state[k].data)).max()
        # no elliptic solve: near-bitwise (measured ~6e-17)
        assert diff < 1e-13, (k, diff)


# ================================================================
#  Gate d: staircase channel vs the walled shallow-water model
# ================================================================
@pytest.mark.parametrize("advection", [False, True])
def test_staircase_channel_matches_the_walled_model(advection):
    # walled channel: x periodic (12), y walled 6 cells on [3, 9].
    # immersed: x periodic (12), y periodic (12) on [0, 12] with the
    # mask carving the wet band 3 < y < 9 — the same wet region.
    npg, lo, hi = 12, 3, 9
    box = lambda x, y: ((y > lo) & (y < hi)).astype(float)  # noqa: E731, ARG005
    imm = _model(
        Grid((_mesh(npg, 0.0, 12.0, periodic=True, name="x"),
              _mesh(npg, 0.0, 12.0, periodic=True, name="y")),
             immersed=ImmersedDomain(box)),
        dt=0.02, advection=advection)
    wal = _model(
        Grid((_mesh(npg, 0.0, 12.0, periodic=True, name="x"),
              _mesh(hi - lo, float(lo), float(hi), periodic=False, name="y"))),
        dt=0.02, advection=advection)
    rng = np.random.default_rng(11)
    shp = {k: wal.state[k].data.shape for k in ("u", "v", "p")}
    ic = {k: 0.1 * rng.standard_normal(shp[k]) for k in ("u", "v", "p")}
    wal.set_fields(**ic)
    full = {}
    for k in ("u", "v", "p"):
        arr = np.zeros(imm.state[k].data.shape)
        arr[:, lo:lo + shp[k][1]] = ic[k]
        full[k] = arr
    imm.set_fields(**full)
    imm.advance(12)
    wal.advance(12)
    for k in ("u", "v", "p"):
        s = wal.state[k].data.shape
        sub = np.asarray(imm.state[k].data)[:, lo:lo + s[1]]
        diff = np.abs(sub - np.asarray(wal.state[k].data)).max()
        # bit-comparable to the genuine walls (measured ~3e-17)
        assert diff < 1e-11, (k, diff)


# ================================================================
#  Gate b: geostrophic steady state preserved away from the mask
# ================================================================
def test_geostrophic_steady_state_preserved_away_from_mask():
    # a balanced Gaussian geostrophic vortex centred at (pi, pi); the
    # mask is a 1x1 dry corner block far from the vortex. The immersed
    # wet region must track the unimmersed run to far below the model's
    # own steady-state drift (the far mask must not spoil the balance).
    f0, amp, sigma = 1.0, 0.2, 0.6

    def p_bump(x, y):
        return amp * np.exp(-(((x - np.pi) ** 2 + (y - np.pi) ** 2)
                              / (2 * sigma ** 2)))

    def u_geo(x, y):  # u = -(1/f) dp/dy
        return (amp / f0) * ((y - np.pi) / sigma ** 2) * np.exp(
            -(((x - np.pi) ** 2 + (y - np.pi) ** 2) / (2 * sigma ** 2)))

    def v_geo(x, y):  # v = +(1/f) dp/dx
        return -(amp / f0) * ((x - np.pi) / sigma ** 2) * np.exp(
            -(((x - np.pi) ** 2 + (y - np.pi) ** 2) / (2 * sigma ** 2)))

    def meshes():
        return (_mesh(24, 0.0, TWO_PI, periodic=True, name="x"),
                _mesh(24, 0.0, TWO_PI, periodic=True, name="y"))

    un = _model(Grid(meshes()), f0=f0, dt=0.005)
    box = lambda x, y: (~((x < 1.0) & (y < 1.0))).astype(float)  # noqa: E731
    im = _model(Grid(meshes(), immersed=ImmersedDomain(box)),
                f0=f0, dt=0.005)
    for model in (un, im):
        model.set_fields(p=p_bump, u=u_geo, v=v_geo)
    p0 = np.asarray(un.state["p"].data).copy()
    un.advance(12)
    im.advance(12)
    # the balanced state is (approximately) steady on the unimmersed
    # model — its own drift is the tolerance the immersed run must meet
    drift = np.abs(np.asarray(un.state["p"].data) - p0).max()
    assert drift < 1e-2
    for k in ("u", "v", "p"):
        mask = np.asarray(im.grid.immersed.mask(
            im.state[k].function_space).data)
        diff = np.abs((np.asarray(im.state[k].data)
                       - np.asarray(un.state[k].data)) * mask).max()
        # the far mask perturbs the wet region far below the natural
        # drift (measured ~1e-7, i.e. ~3e-4 of the ~3e-4 drift): the
        # balance is preserved to the unimmersed model's own tolerance
        assert diff < 1e-2 * drift, (k, diff, drift)


# ================================================================
#  Gate e: genuine lateral partial cells — stable, mass machine zero
# ================================================================
def test_genuine_partial_cells_conserve_mass():
    # a smooth sloping side boundary: a linear plan-area ramp gives
    # genuine partial cells (order-2 Gauss-Legendre is exact for the
    # linear fraction, so no slow-convergence issue — I0 note 2);
    # min_fraction=0 keeps the raw partials
    def slope(x, y):  # noqa: ARG001
        return jnp.clip(1.4 - 0.35 * x, 0.0, 1.0)

    grid = Grid((_mesh(12, 0.0, 6.0, periodic=False, name="x"),
                 _mesh(12, 0.0, 6.0, periodic=True, name="y")),
                immersed=ImmersedDomain(slope, order=2, min_fraction=0.0))
    model = _model(grid, advection=True)
    theta = np.asarray(grid.immersed.fraction(
        model.state["p"].function_space).data)
    # genuine partials exist (strictly between 0 and 1)
    assert ((theta > 1e-6) & (theta < 1 - 1e-6)).sum() > 0
    rng = np.random.default_rng(2)
    mask = np.asarray(
        grid.immersed.mask(model.state["p"].function_space).data)
    model.set_fields(
        p=0.1 * rng.standard_normal(model.state["p"].data.shape) * mask,
        u=0.05 * rng.standard_normal(model.state["u"].data.shape),
        v=0.05 * rng.standard_normal(model.state["v"].data.shape))
    before = _theta_mass(model)
    model.advance(40)
    assert not model.panicked
    assert np.isfinite(np.asarray(model.state["p"].data)).all()
    after = _theta_mass(model)
    assert abs(after - before) <= 1e-13 * max(abs(before), 1.0)


# ================================================================
#  Taught gates (IP-D8)
# ================================================================
def _allwet_model(**kwargs):
    grid = Grid((_mesh(8, 0.0, 8.0, periodic=True, name="x"),
                 _mesh(8, 0.0, 8.0, periodic=True, name="y")),
                immersed=ImmersedDomain(lambda x, y: x * 0.0 + 1.0))  # noqa: ARG005
    return _model(grid, **kwargs)


def test_background_on_immersed_is_a_taught_error():
    grid = Grid((_mesh(8, 0.0, 8.0, periodic=True, name="x"),
                 _mesh(8, 0.0, 8.0, periodic=True, name="y")),
                immersed=ImmersedDomain(lambda x, y: x * 0.0 + 1.0))  # noqa: ARG005
    with pytest.raises(NotImplementedError, match="immersed"):
        sw.Model(
            grid=grid,
            core=sw.Core(gravity=1.0, depth=1.0),
            coriolis=sw.modules.FPlaneCoriolis(f0=1.0),
            advection=False,
            modules_extra=(sw.modules.SadournyAdvection(
                background={"u": 0.0}),),
            time_stepper=fr.model.time_steppers.AdamBashforth(0.01))


def test_eigenmodes_from_model_rejects_immersed():
    model = _allwet_model(advection=False)
    with pytest.raises(NotImplementedError, match="immersed"):
        sw.eigenmodes.from_model(model)


def test_eigenbasis_rejects_immersed():
    grid = Grid((_mesh(8, 0.0, 8.0, periodic=True, name="x"),
                 _mesh(8, 0.0, 8.0, periodic=False, name="y")),
                immersed=ImmersedDomain(lambda x, y: x * 0.0 + 1.0))  # noqa: ARG005
    model = _model(grid, advection=False)
    with pytest.raises(NotImplementedError, match="immersed"):
        sw.eigenmodes.eigenbasis(model)
