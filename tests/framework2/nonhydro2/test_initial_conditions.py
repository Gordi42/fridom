"""Random-phase eigenmode initial conditions (nh.initial_conditions).

Validates the prescribed-spectra random states on the nonhydro
tiers: family purity through the projections (periodic, walled
vertical and the walled-y channel), bitwise seed determinism, the
normalization convention, and the taught errors (Kelvin without
walls, the nonphysical constraint selection).
"""
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.nonhydro2 as nh

N = 8
F0, N2, DSQR = 1.5, 3.0, 2.0
COMPONENTS = ("u", "v", "w", "b")


def make_model(*, periodic_y=True, periodic_z=True):
    """Build a small linear nonhydro model (walls as requested)."""
    mx = fr.grid.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                     periodic=True, name="x")
    my = fr.grid.meshes.IntervalMesh(N, (0.0, 1.0),
                                     periodic=periodic_y, name="y")
    mz = fr.grid.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                     periodic=periodic_z, name="z")
    return nh.Model(
        grid=fr.grid.Grid((mx, my, mz)), advection=False,
        dsqr=DSQR, coriolis=nh.FPlaneCoriolis(f0=F0),
        stratification=nh.ConstantStratification(n2=N2),
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))


@pytest.fixture(scope="module")
def periodic():
    """One fully periodic model + analytic eigenmodes (shared)."""
    model = make_model()
    return model, nh.eigenmodes.from_model(model)


@pytest.fixture(scope="module")
def walled():
    """One rigid-lid (walled z) analytic eigenmode set (shared)."""
    return nh.eigenmodes.from_model(make_model(periodic_z=False))


@pytest.fixture(scope="module")
def channel():
    """One walled-y channel eigenbasis (shared)."""
    return nh.eigenbasis(make_model(periodic_y=False))


def _energy(state):
    weights = {"u": 1.0, "v": 1.0, "w": DSQR, "b": 1.0 / N2}
    return sum(
        float((np.asarray(state[c].data) ** 2).sum()) * weights[c]
        for c in COMPONENTS)


def _same(a, b):
    return all(
        np.array_equal(np.asarray(a[c].data), np.asarray(b[c].data))
        for c in COMPONENTS)


# ================================================================
#  Family purity through the projections
# ================================================================
def test_periodic_random_vortical_is_purely_vortical(periodic):
    model, em = periodic
    state = nh.random_vortical(em, seed=11)
    total = _energy(state)
    vort = _energy(nh.transforms.VorticalProjection(em)(state))
    wave = _energy(nh.transforms.WaveProjection(em)(state))
    assert vort / total > 1.0 - 1e-10
    assert wave / total < 1e-10
    # the factory accepts the model source as well
    assert _same(state, nh.random_vortical(model, seed=11))


def test_walled_random_waves_are_purely_wave(walled):
    state = nh.random_waves(walled, seed=11)
    total = _energy(state)
    wave = _energy(nh.transforms.WaveProjection(walled)(state))
    vort = _energy(nh.transforms.VorticalProjection(walled)(state))
    assert wave / total > 1.0 - 1e-10
    assert vort / total < 1e-10


def test_channel_random_states_stay_in_their_families(channel):
    eb = channel
    vortical = nh.random_vortical(eb, seed=3)
    assert (_energy(eb.projector("vortical")(vortical))
            / _energy(vortical) > 1.0 - 1e-10)
    waves = nh.random_waves(eb, seed=3)
    covered = (_energy(eb.projector("wave")(waves))
               + _energy(eb.projector("kelvin")(waves)))
    assert covered / _energy(waves) > 1.0 - 1e-10


# ================================================================
#  Determinism, normalization, realness
# ================================================================
def test_same_seed_is_bitwise_identical(walled):
    assert _same(nh.random_vortical(walled, seed=7),
                 nh.random_vortical(walled, seed=7))
    assert not _same(nh.random_vortical(walled, seed=7),
                     nh.random_vortical(walled, seed=8))


def test_normalization_and_realness(periodic):
    _, em = periodic
    state = nh.random_waves(em, seed=9)
    peak = max(float(np.abs(np.asarray(state[c].data)).max())
               for c in ("u", "v"))
    assert peak == pytest.approx(1.0, abs=1e-12)
    for c in COMPONENTS:
        assert not np.iscomplexobj(np.asarray(state[c].data))


def test_custom_spectrum_reweights_the_state(periodic):
    _, em = periodic
    default = nh.random_vortical(em, seed=2)
    steep = nh.random_vortical(
        em, seed=2,
        spectral_energy_density=lambda kx, ky, *_kz:
        nh.initial_conditions.geostrophic_energy_spectrum(
            kx, ky, d=9.0))
    assert not _same(default, steep)


# ================================================================
#  Taught errors
# ================================================================
def test_kelvin_needs_walls(periodic):
    _, em = periodic
    with pytest.raises(ValueError, match="no walls, no Kelvin"):
        nh.random_state(em, "kelvin", seed=1)


def test_unknown_family_is_taught(periodic):
    _, em = periodic
    with pytest.raises(ValueError, match="unknown mode family"):
        nh.random_state(em, "rossby", seed=1)


def test_constraint_selection_is_nonphysical(channel):
    with pytest.raises(ValueError, match="unknown or nonphysical"):
        nh.random_state(channel, "constraint", seed=1)
