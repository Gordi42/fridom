r"""wave_package(quadrature=True) on the shallow-water analytic tier.

Prefix-mirrored shard of test_initial_conditions.py: the complex
analytic-signal packet and its ``fr.model.modules.Source`` use. For
the returned carrier frequency ``omega`` the packet satisfies, at
every ``t``,

    Re[Q e^{-i omega t}] == wave_package(..., phase=phase + omega t)

the real packet phase-advanced by ``omega t`` (the single mode's own
linear time evolution). Both analytic axes are periodic here, so only
the standing path exists — traveling always teaches.
"""
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model.modules.source import Source
from fridom.spatial.scalars import Scalars

from .conftest import make_grid, make_model

COMPONENTS = ("u", "v", "p")
CSQR, F0 = 2.0, 1.5
K = {"x": 3, "y": 0}


@pytest.fixture(scope="module")
def periodic():
    grid = make_grid()
    model = make_model(grid, csqr=CSQR, f0=F0, advection=None)
    return grid, sw.eigenmodes.from_model(model)


def _env():
    return sw.gaussian(pos={"x": 0.5}, width={"x": 0.15})


def test_quadrature_defining_property(periodic):
    _, em = periodic
    env, phase = _env(), 0.3
    omega, q = sw.wave_package(em, K, "wave+", envelope=env, phase=phase,
                               quadrature=True)
    assert all(q[c].function_space.scalars is Scalars.COMPLEX
               for c in COMPONENTS)
    assert max(float(np.abs(np.asarray(q[c].imag.data)).max())
               for c in COMPONENTS) > 0.0
    for t in (0.0, 0.4, 1.6, -0.9):
        _, ptrue = sw.wave_package(em, K, "wave+", envelope=env,
                                   phase=phase + omega * t)
        for c in COMPONENTS:
            lhs = np.real(np.asarray(q[c].data) * np.exp(-1j * omega * t))
            np.testing.assert_allclose(
                lhs, np.asarray(ptrue[c].data), rtol=1e-11, atol=1e-12)


def test_quadrature_real_part_is_the_real_packet(periodic):
    _, em = periodic
    env = _env()
    omega, real = sw.wave_package(em, K, "wave+", envelope=env, phase=0.3)
    omega_q, q = sw.wave_package(em, K, "wave+", envelope=env, phase=0.3,
                                 quadrature=True)
    assert omega_q == omega
    for c in COMPONENTS:
        np.testing.assert_array_equal(
            np.asarray(q[c].real.data), np.asarray(real[c].data))


def test_quadrature_drives_a_harmonic_source(periodic):
    # the complex packet feeds a factor-free Source directly: the
    # tendency is A Re[Q e^{-i omega t}] = A * (phase-advanced packet).
    grid, em = periodic
    env, amp = _env(), 0.4
    omega, q = sw.wave_package(em, K, "wave+", envelope=env, phase=0.3,
                               quadrature=True)
    src = Source("packet", pattern=q,
                 law=fr.model.Harmonic(amp, omega / (2.0 * np.pi)))
    model = make_model(grid, csqr=CSQR, f0=F0, advection=None,
                       modules_extra=(src,))
    for t in (0.11, 0.23):
        tend = model.tendency(model.state, t=t, constraints=False)
        _, ptrue = sw.wave_package(em, K, "wave+", envelope=env,
                                   phase=0.3 + omega * t)
        for c in COMPONENTS:
            np.testing.assert_allclose(
                np.asarray(tend[c].data),
                amp * np.asarray(ptrue[c].data), atol=1e-13)


def test_quadrature_teaches_on_a_channel():
    grid = make_grid(periodic_y=False)
    model = make_model(grid, csqr=CSQR, f0=F0, advection=None)
    channel = sw.eigenbasis(model)
    with pytest.raises(ValueError, match="walled channel"):
        sw.wave_package(channel, {"x": 2, "y": 1}, envelope=_env(),
                        quadrature=True)
