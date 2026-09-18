r"""The seawater equations of state (``fridom.hydrostatic.eos``).

The gates:

- **published check values** — the 55-term TEOS-10 polynomial
  reproduces the check values of Roquet et al. (2015a), appendix:
  at ``(Theta, S_A, p) = (10 degC, 30 g/kg, 1000 dbar)``
  ``r0 = 4.59763035``, ``r' = 1022.85377``, ``rho = 1027.45140`` and
  the sensitivities ``-d rho/d Theta = 0.179646281``,
  ``d rho/d S_A = 0.765555368``;
- the simplified Roquet et al. (2015b) family against its own
  published coefficients (Table 3), against the NEMO "S-EOS" spelling
  of the same fit (``a0``/``b0``), and against the TEOS-10 polynomial
  over the oceanographic range (its stated purpose);
- cabbeling and thermobaricity sanity (sign and size);
- the dynamic anomaly removes every pure-depth contribution;
- jit / autodiff safety, a dry cell's ``T = S = 0`` included.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.hydrostatic as hy
from fridom.hydrostatic import eos as eos_module
from fridom.hydrostatic.eos import ROQUET_COEFFICIENTS, EquationOfState

NONLINEAR = [
    pytest.param(hy.RoquetEOS, id="roquet"),
    pytest.param(hy.TEOS10EOS, id="teos10"),
]
EVERY = [pytest.param(hy.LinearEOS, id="linear"), *NONLINEAR]


# ================================================================
#  TEOS-10 polynomial: the published check values
# ================================================================
def test_teos10_reproduces_the_published_density_check_value():
    eos = hy.TEOS10EOS()
    # Roquet et al. (2015a): published to the digits asserted here
    assert float(eos.reference_profile(1000.0)) == pytest.approx(
        4.59763035, abs=5e-9)
    assert float(eos.fit(10.0, 30.0, 1000.0)) == pytest.approx(
        1022.85377, abs=5e-6)
    assert float(eos.density(10.0, 30.0, 1000.0)) == pytest.approx(
        1027.45140, abs=5e-6)


def test_teos10_reproduces_the_published_sensitivities():
    eos = hy.TEOS10EOS()
    alpha = float(eos.thermal_expansion(10.0, 30.0, 1000.0))
    beta = float(eos.haline_contraction(10.0, 30.0, 1000.0))
    assert alpha * eos.rho0 == pytest.approx(0.179646281, abs=5e-10)
    assert beta * eos.rho0 == pytest.approx(0.765555368, abs=5e-9)


def test_teos10_carries_the_55_terms_of_the_fit():
    # 52 anomaly coefficients + the 6-term reference profile, of which
    # the paper counts 55 independent (three r0/r' pairs are tied)
    assert len(eos_module._TEOS10_R) == 52
    assert len(eos_module._TEOS10_R0) == 6
    nested = sum(len(row) for rows in eos_module._TEOS10_TABLE
                 for row in rows)
    assert nested >= 52


def test_teos10_surface_density_of_standard_seawater():
    # standard seawater (S_A = 35.16504 g/kg, Theta = 0 degC, p = 0):
    # TEOS-10 gives rho = 1028.1 kg/m^3 (gsw_rho: 1028.10633)
    rho = float(hy.TEOS10EOS().density(0.0, 35.16504, 0.0))
    assert rho == pytest.approx(1028.106, abs=5e-3)


# ================================================================
#  The simplified Roquet family
# ================================================================
def test_roquet_second_order_matches_table_3_by_hand():
    r = ROQUET_COEFFICIENTS["second_order"]
    eos = hy.RoquetEOS("second_order")
    t, s, d = 4.0, 34.5, 2000.0

    def poly(t, s, d):
        z = -d
        return (r["R100"] * s + r["R010"] * t + r["R020"] * t * t
                - r["R011"] * t * z + r["R200"] * s * s
                - r["R101"] * s * z + r["R110"] * s * t)

    want = eos.rho0 + poly(t, s, d) - poly(10.0, 35.0, 0.0)
    assert float(eos.density(t, s, d)) == pytest.approx(want, rel=1e-14)
    assert float(eos.density(10.0, 35.0, 0.0)) == pytest.approx(eos.rho0)


def test_roquet_leading_coefficients_match_the_nemo_seos_spelling():
    # NEMO's S-EOS is the same Roquet et al. (2015b) fit expanded about
    # (10 degC, 35 g/kg): a0 = 1.6550e-1, b0 = 7.6554e-1 kg/m^3 per unit
    eos = hy.RoquetEOS("second_order")
    a0 = float(eos.thermal_expansion(10.0, 35.0, 0.0)) * eos.rho0
    b0 = float(eos.haline_contraction(10.0, 35.0, 0.0)) * eos.rho0
    assert a0 == pytest.approx(1.6550e-1, rel=5e-3)
    assert b0 == pytest.approx(7.6554e-1, rel=1.5e-2)
    # the thermobaric coefficient mu1 = 1.4970e-4 1/m: d(a)/d(depth)/a0
    a1 = float(eos.thermal_expansion(10.0, 35.0, 1.0)) * eos.rho0
    assert (a1 - a0) / a0 == pytest.approx(1.4970e-4, rel=2e-2)


def test_roquet_tracks_teos10_over_the_ocean_range():
    # the simplified EOS is "realistic": fitted to the ocean's actual
    # water masses, its dynamic anomaly follows the full TEOS-10
    # polynomial inside the oceanographic funnel (warm water is
    # shallow: T <= 2 + 26 exp(-d / 800 m)). Measured: max 0.069, rms
    # 0.022 kg/m^3 over a 6 kg/m^3 span — against 1.63 / 0.43 for the
    # linear EOS on the same points. (Outside the funnel — 28 degC at
    # 5 km — the simplified fit is off by up to 0.9 kg/m^3.)
    simple, full, linear = hy.RoquetEOS(), hy.TEOS10EOS(), hy.LinearEOS()
    t, s, d = jnp.meshgrid(jnp.linspace(-1.0, 28.0, 30),
                           jnp.linspace(33.0, 37.0, 9),
                           jnp.linspace(0.0, 5000.0, 11), indexing="ij")
    funnel = t <= 2.0 + 26.0 * jnp.exp(-d / 800.0)
    truth = full.density_anomaly(t, s, d)

    def errors(eos):
        err = jnp.where(
            funnel, jnp.abs(eos.density_anomaly(t, s, d) - truth), 0.0)
        return (float(err.max()),
                float(jnp.sqrt((err ** 2).sum() / funnel.sum())))

    worst, rms = errors(simple)
    assert worst < 0.1
    assert rms < 0.03
    linear_worst, linear_rms = errors(linear)
    assert linear_worst > 10.0 * worst
    assert linear_rms > 10.0 * rms


@pytest.mark.parametrize("name", sorted(ROQUET_COEFFICIENTS))
def test_roquet_sets_resolve_by_name(name):
    eos = hy.RoquetEOS(name)
    given = ROQUET_COEFFICIENTS[name]
    assert {k: v for k, v in eos.coefficients.items()
            if v != 0.0} == given
    assert eos.uses_depth == ("R011" in given or "R101" in given)
    assert name in repr(eos)


def test_roquet_takes_a_custom_coefficient_mapping():
    eos = hy.RoquetEOS({"R010": -0.2, "R100": 0.8})
    assert not eos.uses_depth
    assert float(eos.density_anomaly(11.0, 36.0)) == pytest.approx(0.6)
    assert "R010" in repr(eos)


def test_roquet_refuses_unknown_names():
    with pytest.raises(ValueError, match="unknown RoquetEOS coefficient "
                                         "set"):
        hy.RoquetEOS("third_order")
    with pytest.raises(ValueError, match=r"unknown RoquetEOS "
                                         r"coefficient\(s\)"):
        hy.RoquetEOS({"R300": 1.0})


# ================================================================
#  The linear EOS
# ================================================================
def test_linear_density_and_buoyancy_are_the_textbook_forms():
    eos = hy.LinearEOS(alpha=2e-4, beta=8e-4, temperature0=5.0,
                       salinity0=34.0, rho0=1000.0)
    assert not eos.uses_depth
    assert eos.tunable == {"alpha": 2e-4, "beta": 8e-4}
    assert float(eos.density(7.0, 35.0)) == pytest.approx(
        1000.0 * (1 - 2e-4 * 2.0 + 8e-4 * 1.0))
    assert float(eos.buoyancy(7.0, 35.0, gravity=10.0)) == pytest.approx(
        10.0 * (2e-4 * 2.0 - 8e-4 * 1.0))
    # depth independent
    assert float(eos.density(7.0, 35.0, 4000.0)) == float(
        eos.density(7.0, 35.0))


def test_linear_live_coefficients_override_the_constructor_values():
    eos = hy.LinearEOS(alpha=2e-4, beta=8e-4, rho0=1000.0)
    live = {"alpha": 4e-4, "beta": 0.0}
    got = float(eos.density_anomaly(11.0, 36.0, 0.0, live))
    assert got == pytest.approx(-1000.0 * 4e-4)


def test_linear_expansion_coefficients_are_the_constructor_values():
    eos = hy.LinearEOS(alpha=2e-4, beta=8e-4)
    assert float(eos.thermal_expansion(3.0, 33.0)) == pytest.approx(2e-4)
    assert float(eos.haline_contraction(3.0, 33.0)) == pytest.approx(8e-4)


# ================================================================
#  Physics sanity: cabbeling, thermobaricity, the dynamic anomaly
# ================================================================
@pytest.mark.parametrize("make", NONLINEAR)
def test_cabbeling_mixing_two_equal_density_parcels_densifies(make):
    eos = make()
    # a warm/salty and a cold/fresh parcel of EQUAL surface density
    t_warm, t_cold, s_cold = 20.0, 2.0, 34.0
    target = float(eos.density(t_cold, s_cold))
    lo, hi = 34.0, 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(eos.density(t_warm, mid)) < target:
            lo = mid
        else:
            hi = mid
    s_warm = 0.5 * (lo + hi)
    assert float(eos.density(t_warm, s_warm)) == pytest.approx(
        target, abs=1e-9)
    mixed = float(eos.density(0.5 * (t_warm + t_cold),
                              0.5 * (s_warm + s_cold)))
    # the mixture is denser than either parent: cabbeling, O(0.1-1)
    assert 0.1 < mixed - target < 1.5


def test_linear_eos_does_not_cabbel():
    eos = hy.LinearEOS()
    a = float(eos.density_anomaly(20.0, 36.0))
    b = float(eos.density_anomaly(2.0, 34.0))
    mixed = float(eos.density_anomaly(11.0, 35.0))
    assert mixed == pytest.approx(0.5 * (a + b), abs=1e-12)


@pytest.mark.parametrize("make", NONLINEAR)
def test_thermobaricity_expansion_grows_with_depth(make):
    eos = make()
    shallow = float(eos.thermal_expansion(2.0, 35.0, 0.0))
    deep = float(eos.thermal_expansion(2.0, 35.0, 4000.0))
    assert shallow > 0.0
    # cold water: alpha roughly doubles to triples over 4 km
    assert 1.5 < deep / shallow < 4.0


@pytest.mark.parametrize("make", EVERY)
def test_dynamic_anomaly_vanishes_on_the_reference_parcel(make):
    eos = make()
    t_ref, s_ref = eos.reference
    depth = jnp.linspace(0.0, 6000.0, 13)
    anomaly = eos.density_anomaly(t_ref, s_ref, depth)
    assert float(jnp.abs(jnp.asarray(anomaly)).max()) == 0.0


def test_dynamic_anomaly_drops_the_bulk_compressibility():
    eos = hy.TEOS10EOS()
    # in situ the density rises ~ 4.5 kg/m^3 per km; the dynamic
    # anomaly of one parcel moves by the thermobaric residual only
    # (0.54 kg/m^3 over 4 km for a parcel 6 K colder than the reference)
    rise = float(eos.density(4.0, 35.0, 4000.0)
                 - eos.density(4.0, 35.0, 0.0))
    residual = float(eos.density_anomaly(4.0, 35.0, 4000.0)
                     - eos.density_anomaly(4.0, 35.0, 0.0))
    assert rise > 15.0
    assert abs(residual) < 0.05 * rise
    # while differences at a fixed depth are untouched
    full = float(eos.density(4.0, 35.0, 3000.0)
                 - eos.density(6.0, 34.0, 3000.0))
    dyn = float(eos.density_anomaly(4.0, 35.0, 3000.0)
                - eos.density_anomaly(6.0, 34.0, 3000.0))
    assert dyn == pytest.approx(full, abs=1e-10)


@pytest.mark.parametrize("make", EVERY)
def test_buoyancy_is_minus_g_anomaly_over_rho0(make):
    eos = make()
    b = float(eos.buoyancy(15.0, 34.0, 500.0, gravity=9.81))
    want = -9.81 * float(eos.density_anomaly(15.0, 34.0, 500.0)) / eos.rho0
    assert b == pytest.approx(want, rel=1e-14)
    assert b > 0.0   # warmer and fresher than the reference: light


# ================================================================
#  jax safety
# ================================================================
@pytest.mark.parametrize("make", EVERY)
def test_eos_broadcasts_and_jits(make):
    eos = make()
    t = jnp.linspace(0.0, 20.0, 5)[:, None]
    s = jnp.linspace(33.0, 36.0, 4)[None, :]
    eager = eos.density_anomaly(t, s, 1000.0)
    jitted = jax.jit(eos.density_anomaly)(t, s, 1000.0)
    assert eager.shape == (5, 4)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager),
                               rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize("make", EVERY)
def test_gradient_is_finite_on_a_dry_cell(make):
    # a masked (dry) cell holds T = S = 0: the reverse-mode derivative
    # must stay finite there (the TEOS-10 root is of S + 32, not of S)
    eos = make()

    def loss(t, s):
        return jnp.sum(eos.density_anomaly(t, s, 100.0) ** 2)

    zeros = jnp.zeros(3)
    grad_t, grad_s = jax.grad(loss, argnums=(0, 1))(zeros, zeros)
    assert bool(jnp.all(jnp.isfinite(grad_t)))
    assert bool(jnp.all(jnp.isfinite(grad_s)))
    assert float(jnp.abs(grad_s).min()) > 0.0


# ================================================================
#  The interface
# ================================================================
def test_equal_configurations_compare_and_hash_equal():
    assert hy.TEOS10EOS() == hy.TEOS10EOS()
    assert hash(hy.RoquetEOS()) == hash(hy.RoquetEOS())
    assert hy.LinearEOS(alpha=1e-4) != hy.LinearEOS(alpha=2e-4)
    assert hy.RoquetEOS("linear") != hy.RoquetEOS("freezing")
    assert hy.TEOS10EOS() != hy.TEOS10EOS(rho0=1026.0)
    assert hy.TEOS10EOS() != "TEOS10EOS"
    assert "rho0=1020.0" in repr(hy.TEOS10EOS())
    assert "alpha=0.0001" in repr(hy.LinearEOS(alpha=1e-4))


def test_reference_density_must_be_positive():
    with pytest.raises(ValueError, match="rho0 must be positive"):
        hy.LinearEOS(rho0=0.0)


def test_the_interface_is_abstract_and_untunable_by_default():
    with pytest.raises(TypeError, match="abstract"):
        EquationOfState(rho0=1000.0, reference=(0.0, 35.0))
    assert hy.TEOS10EOS().tunable == {}
    assert hy.TEOS10EOS().uses_depth
    assert hy.TEOS10EOS().reference == (10.0, 35.0)
