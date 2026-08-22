r"""wave_package(quadrature=True): the complex analytic-signal packet.

Prefix-mirrored shard of test_initial_conditions.py (AGENTS oversized
-module exception): the ``quadrature=`` surface of ``wave_package``
plus the polarized wave-maker parity that the deleted polarized wave
maker used to pin (its forcing ``A sin(omega t) z_W`` is now the
``fr.model.modules.Source`` of the real packet).

The defining property, for the returned carrier frequency ``omega``:

    Re[Q e^{-i omega t}] == wave_package(..., phase=phase + omega t)

for every ``t`` — the real packet phase-advanced by ``omega t`` (the
single mode's own linear time evolution). It is exact for every tier
and path (periodic, standing walled, and traveling), because the
family projection is a real operator and the mode is
``Re[W(x) e^{-i phase}]``; so no wave_package configuration is refused
for ``quadrature=True`` (the walled standing packet DOES admit a
single complex Q — checked below, not assumed).
"""
import inspect

import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.modules.source import Source
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.constant import ConstantSpace

N = 8
F0, N2, DSQR = 1.5, 3.0, 2.0
DT = 1e-3
COMPONENTS = ("u", "v", "w", "b")


# ================================================================
#  Self-contained fixtures (duplicated per the shard contract)
# ================================================================
def make_grid(*, periodic_z=True):
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                        periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                        periodic=True, name="y")
    mz = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                        periodic=periodic_z, name="z")
    return fr.spatial.Grid((mx, my, mz), device_ids=(0,))


def make_model(*makers, grid=None, periodic_z=True):
    return nh.Model(
        grid=make_grid(periodic_z=periodic_z) if grid is None else grid,
        core=nh.Core(aspect_ratio=DSQR ** 0.5),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=F0),
        buoyancy=nh.ConstantStratification(n2=N2),
        advection=None,
        modules_extra=makers)


@pytest.fixture(scope="module")
def periodic():
    grid = make_grid()
    return grid, nh.eigenmodes.from_model(make_model(grid=grid))


@pytest.fixture(scope="module")
def walled():
    return nh.eigenmodes.from_model(make_model(periodic_z=False))


DRIFT_K = {"x": 4, "y": 0, "z": 5}
DRIFT_ENV = {"pos": {"x": 500.0, "z": 500.0},
             "width": {"x": 220.0, "z": 220.0}}


@pytest.fixture(scope="module")
def drift():
    mx = fr.spatial.meshes.IntervalMesh(48, (0.0, 2000.0),
                                        periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(1, (0.0, 1.0),
                                        periodic=True, name="y")
    mz = fr.spatial.meshes.IntervalMesh(32, (0.0, 1000.0),
                                        periodic=False, name="z")
    grid = fr.spatial.Grid((mx, my, mz), device_ids=(0,))
    return nh.eigenmodes.from_model(nh.Model(
        grid=grid, coriolis=nh.FPlaneCoriolis(f0=1e-4),
        buoyancy=nh.ConstantStratification(n2=2.5e-5), advection=None,
        time_stepper=AdamBashforth(60.0, order=3)))


def _check_defining_property(em, k, family, envelope, phase=0.4,
                             traveling=None):
    """Re[Q e^{-i omega t}] == the phase-advanced real packet."""
    omega, q = nh.wave_package(em, k, family, envelope=envelope,
                               phase=phase, traveling=traveling,
                               quadrature=True)
    assert all(q[c].function_space.scalars is Scalars.COMPLEX
               for c in COMPONENTS)
    # Q is a genuine analytic signal: its imaginary part is nonzero
    assert max(float(np.abs(np.asarray(q[c].imag.data)).max())
               for c in COMPONENTS) > 0.0
    for t in (0.0, 0.7, 2.3, -1.4):
        _, ptrue = nh.wave_package(
            em, k, family, envelope=envelope,
            phase=phase + omega * t, traveling=traveling)
        for c in COMPONENTS:
            lhs = np.real(np.asarray(q[c].data) * np.exp(-1j * omega * t))
            np.testing.assert_allclose(
                lhs, np.asarray(ptrue[c].data), rtol=1e-11, atol=1e-12)
    return omega, q


# ================================================================
#  The defining property across every tier and path
# ================================================================
def test_quadrature_defining_property_periodic(periodic):
    _, em = periodic
    _check_defining_property(
        em, {"x": 2, "y": 0, "z": 1}, "wave+",
        nh.gaussian(pos={"x": np.pi}, width={"x": 1.5}))


def test_quadrature_defining_property_walled_standing(walled):
    # the walled STANDING packet DOES admit a single complex Q (the
    # plan's tentative "cannot" hypothesis is false): the mode is
    # Re[W e^{-i phase}] and the projection is real, so the quadrature
    # construction is exact here too — no refusal is needed.
    _check_defining_property(
        walled, {"x": 2, "y": 0, "z": 2}, "wave+",
        nh.gaussian(pos={"x": np.pi, "z": np.pi},
                    width={"x": 1.5, "z": 1.5}))


@pytest.mark.parametrize("direction", [-1, 1])
def test_quadrature_defining_property_walled_traveling(drift, direction):
    _check_defining_property(
        drift, DRIFT_K, "wave+", nh.gaussian(**DRIFT_ENV),
        traveling={"z": direction})


def test_quadrature_real_part_is_the_real_packet(periodic):
    # Re[Q] (t = 0) is bit-for-bit the quadrature=False packet, and the
    # frequency is identical — quadrature=False stays unchanged.
    _, em = periodic
    env = nh.gaussian(pos={"x": np.pi}, width={"x": 1.5})
    omega, real = nh.wave_package(em, {"x": 2, "y": 0, "z": 1}, "wave+",
                                  envelope=env, phase=0.4)
    omega_q, q = nh.wave_package(em, {"x": 2, "y": 0, "z": 1}, "wave+",
                                 envelope=env, phase=0.4, quadrature=True)
    assert omega_q == omega
    for c in COMPONENTS:
        np.testing.assert_array_equal(
            np.asarray(q[c].real.data), np.asarray(real[c].data))


def test_quadrature_still_validates(periodic):
    # quadrature=True does not bypass the wave_package taught errors
    _, em = periodic
    with pytest.raises(ValueError, match="does not have"):
        nh.wave_package(
            em, {"x": 2, "y": 0, "z": 1},
            envelope=nh.gaussian(pos={"q": 1.0}, width={"q": 1.0}),
            quadrature=True)


def test_quadrature_teaches_on_a_channel():
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                        periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                        periodic=False, name="y")
    mz = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                        periodic=True, name="z")
    grid = fr.spatial.Grid((mx, my, mz), device_ids=(0,))
    channel = nh.eigenbasis(nh.Model(
        grid=grid, core=nh.Core(aspect_ratio=DSQR ** 0.5),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=F0),
        buoyancy=nh.ConstantStratification(n2=N2), advection=None))
    with pytest.raises(ValueError, match="walled channel"):
        nh.wave_package(
            channel, {"x": 2, "y": 0, "z": 1},
            envelope=nh.gaussian(pos={"x": np.pi}, width={"x": 1.5}),
            quadrature=True)


# ================================================================
#  Polarized wave-maker parity (the deleted polarized wave maker)
# ================================================================
K = {"x": 2, "y": 0, "z": 1}
POS, WIDTH = {"x": np.pi}, {"x": 1.0}
AMP = 0.5


def _sample_gaussian(grid, space, pos, width):
    """Return the deleted maker's own-node Gaussian mask."""
    names = tuple(
        coord for factor in space.factors
        if not isinstance(factor, ConstantSpace)
        for coord in factor.names)

    def init(**coords):
        mask = np.asarray(1.0)
        for axis, p in pos.items():
            mask = mask * np.exp(-((coords[axis] - p) ** 2)
                                 / width[axis] ** 2)
        return mask

    init.__signature__ = inspect.Signature(
        [inspect.Parameter(c, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for c in names])
    return grid.create_field(space, init=init)


def _reconstruct_zW(em, branch=1):
    """z_W as the deleted polarized wave maker built it (factor 2)."""
    _, wave = em.mode("wave", K, branch=branch)
    masked = nh.State({
        c: em.kit.forward(c)(
            wave[c] * _sample_gaussian(em.grid, wave[c].function_space,
                                       POS, WIDTH))
        for c in COMPONENTS})
    packet = em.projector(branch)(masked)
    return {c: 2.0 * np.asarray(em.kit.backward(c)(packet[c]).real.data)
            for c in COMPONENTS}


def test_polarized_parity_factor_bookkeeping(periodic):
    # the deleted maker doubled its packet; the factor-free
    # wave_package packet is therefore exactly z_W / 2.
    _, em = periodic
    z_W = _reconstruct_zW(em)
    _, packet = nh.wave_package(
        em, K, "wave+", envelope=nh.gaussian(pos=POS, width=WIDTH))
    for c in COMPONENTS:
        np.testing.assert_allclose(
            np.asarray(packet[c].data), z_W[c] / 2.0, atol=1e-14)


def test_polarized_parity_source_reproduces_the_forcing(periodic):
    # The polarized wave maker (amplitude A) forcing A sin(omega t) z_W is
    # reproduced by Source(pattern=real_packet, Harmonic(2 A, omega/2pi,
    # phase=-pi/2)): the +sin comes from phase=-pi/2, the factor 2
    # absorbs the maker's doubling (the plan keeps factors in
    # wave_package; the engine is factor-free).
    grid, em = periodic
    z_W = _reconstruct_zW(em)
    omega, packet = nh.wave_package(
        em, K, "wave+", envelope=nh.gaussian(pos=POS, width=WIDTH))
    src = Source(
        "polarized", pattern=packet,
        law=fr.model.Harmonic(2.0 * AMP, omega / (2.0 * np.pi),
                              phase=-np.pi / 2))
    model = make_model(src, grid=grid)
    # t = 0: the forcing all but vanishes (dry-run smoke). The engine
    # evaluates cos(2 pi f t + phase); at phase = -pi/2 that is
    # cos(-pi/2) ~ 6e-17 (not the maker's exact sin(0) = 0), so the
    # residual is machine epsilon, not identically zero.
    tend0 = model.tendency(model.state, t=0.0, constraints=False)
    for c in COMPONENTS:
        assert np.abs(np.asarray(tend0[c].data)).max() < 1e-15
    for t in (0.0, 0.13, 0.29, 0.5 * np.pi / omega):
        tend = model.tendency(model.state, t=t, constraints=False)
        for c in COMPONENTS:
            np.testing.assert_allclose(
                np.asarray(tend[c].data),
                AMP * np.sin(omega * t) * z_W[c], atol=1e-13)


def test_polarized_parity_frequency_matches_the_eigenmode(periodic):
    _, em = periodic
    omega, _ = nh.wave_package(
        em, K, "wave+", envelope=nh.gaussian(pos=POS, width=WIDTH))
    omega_mode, _ = em.mode("wave+", K)
    assert omega == pytest.approx(omega_mode, rel=1e-12)


def test_polarized_source_amplitude_is_sweepable(periodic):
    grid, em = periodic
    omega, packet = nh.wave_package(
        em, K, "wave+", envelope=nh.gaussian(pos=POS, width=WIDTH))
    src = Source(
        "polarized", pattern=packet,
        law=fr.model.Harmonic(2.0 * AMP, omega / (2.0 * np.pi),
                              phase=-np.pi / 2))
    model = make_model(src, grid=grid)
    t = 0.5 * np.pi / omega
    before = np.asarray(model.tendency(
        model.state, t=t, constraints=False)["u"].data)
    model.update_parameters({"source.polarized.amplitude": 4.0 * AMP})
    after = np.asarray(model.tendency(
        model.state, t=t, constraints=False)["u"].data)
    np.testing.assert_allclose(after, 2.0 * before, atol=1e-13)


def test_polarized_source_energy_enters_a_running_model(periodic):
    grid, em = periodic
    omega, packet = nh.wave_package(
        em, K, "wave+", envelope=nh.gaussian(pos=POS, width=WIDTH))
    src = Source(
        "polarized", pattern=packet,
        law=fr.model.Harmonic(2.0 * AMP, omega / (2.0 * np.pi),
                              phase=-np.pi / 2))
    model = make_model(src, grid=grid)
    model.advance(50)
    u = np.asarray(model.state["u"].data)
    assert np.isfinite(u).all()
    assert np.abs(u).max() > 0.0
