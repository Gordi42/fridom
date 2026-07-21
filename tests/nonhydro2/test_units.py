"""The nonhydrostatic unit-factor table (``model.units`` rows).

Rotational pins (eps aliases Ro; the vertical scale delta*L off the
core's aspect ratio: w = delta*U, b = U^2/(eps*delta*L), N_dim
consistency n2_eff = Bu from the primitives), the Dimensional
identity/constant semantics, the MeridionalStratification
unresolvable mark, report smoke, and the real-model writer
round-trip through xarray.
"""
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.io.triggers import every
from fridom.model.io.writer import Writer

N = 4
DT = 2.0 ** -6

L_REF = 2.0e3
U_REF = 0.5
DELTA = 0.5
RO = 0.25          # eps under Rotational()
FR_INT = 0.125
F0 = 1.0e-4
N2 = 4.0e-4


def make_grid():
    im = fr.spatial.meshes.IntervalMesh
    return fr.spatial.Grid(tuple(
        im(N, (0.0, 2.0 * np.pi), periodic=True, name=name)
        for name in ("x", "y", "z")), device_ids=(0,))


def rot_model():
    return nh.Model(
        grid=make_grid(),
        core=nh.Core(aspect_ratio=DELTA),
        scaling=fr.scaling.Rotational(L=L_REF, U=U_REF),
        coriolis=nh.FPlaneCoriolis(rossby_number=RO),
        stratification=nh.ConstantStratification(froude_number=FR_INT),
        advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(DT, order=2))


def dim_model(*, stratification=None):
    return nh.Model(
        grid=make_grid(),
        core=nh.Core(aspect_ratio=DELTA),
        coriolis=nh.FPlaneCoriolis(f0=F0),
        stratification=stratification
        or nh.ConstantStratification(n2=N2),
        advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(DT, order=2))


# ================================================================
#  Rotational pins (eps aliases Ro; vertical scale delta*L)
# ================================================================
def test_rotational_amplitude_pins():
    units = rot_model().units
    assert units.factor("u") == pytest.approx(U_REF)
    assert units.factor("v") == pytest.approx(U_REF)
    assert units.factor("w") == pytest.approx(DELTA * U_REF)
    assert units.factor("p") == pytest.approx(U_REF ** 2 / RO)
    assert units.factor("b") == pytest.approx(
        U_REF ** 2 / (RO * DELTA * L_REF))


def test_rotational_coordinates_and_time():
    units = rot_model().units
    assert units.factor("x") == pytest.approx(L_REF)
    assert units.factor("y") == pytest.approx(L_REF)
    assert units.factor("z") == pytest.approx(DELTA * L_REF)
    assert units.factor("t") == pytest.approx(RO * L_REF / U_REF)


def test_rotational_n_dim_is_consistent_with_the_primitives():
    units = rot_model().units
    n_dim = units.factor("N_dim")
    assert n_dim == pytest.approx(U_REF / (FR_INT * DELTA * L_REF))
    # Bu = (N_dim H / (f_dim L))^2 reproduces the model-units
    # effective stratification n2_eff = (eps/Fr_int)^2
    height = units.factor("z")
    burger = (n_dim * height
              / (units.factor("f_dim") * units.factor("x"))) ** 2
    assert burger == pytest.approx((RO / FR_INT) ** 2)


# ================================================================
#  Dimensional semantics (identity + constants keep their meaning)
# ================================================================
def test_dimensional_raw_factors_are_identity_with_units():
    units = dim_model().units
    for name, unit in (("u", "m/s"), ("w", "m/s"), ("b", "m/s^2"),
                       ("p", "m^2/s^2"), ("z", "m"), ("t", "s")):
        entry = units.factors[name]
        assert entry.value == 1.0
        assert entry.unit == unit


def test_dimensional_constants_report_bound_values():
    units = dim_model().units
    assert units.factor("N_dim") == pytest.approx(N2 ** 0.5)
    assert units.factor("f_dim") == pytest.approx(F0)


def test_meridional_stratification_marks_n_dim():
    # the profile binds no constant n2 -> the row is marked, never a
    # false constant (the beta-plane f_dim precedent)
    model = dim_model(stratification=nh.MeridionalStratification(
        lambda y: 1.0 + 0.0 * y))
    entry = model.units.factors["N_dim"]
    assert entry.value is None
    assert entry.missing == ("stratification.n2",)
    with pytest.raises(ValueError, match=r"stratification\.n2"):
        model.units.factor("N_dim")


# ================================================================
#  Report
# ================================================================
def test_report_smoke_both_variants():
    nondim = rot_model().units.report()
    assert "Rotational, nondimensional" in nondim
    assert "N_dim" in nondim
    assert "[delta*L]" in nondim
    dim = dim_model().units.report()
    assert "Dimensional, dimensional" in dim
    assert "[sqrt(n2)]" in dim


# ================================================================
#  Writer round-trip (the real-model metadata stamp)
# ================================================================
def test_writer_stamps_a_nondimensional_model(tmp_path):
    xr = pytest.importorskip("xarray")
    model = rot_model()
    path = tmp_path / "nondim.zarr"
    writer = Writer(path, trigger=every(steps=1))
    writer.bind(model)
    writer.write(model.carry)
    writer.close()
    ds = xr.open_zarr(path, consolidated=False)
    assert ds.attrs["fridom_scaling"] == "Rotational"
    assert ds.attrs["fridom_scaling_epsilon"] == pytest.approx(RO)
    params = ds.attrs["fridom_scaling_parameters"]
    assert params["nonhydro.aspect_ratio"] == pytest.approx(DELTA)
    assert params["stratification.froude"] == pytest.approx(FR_INT)
    # per-variable factors (the vertical family rides delta)
    assert ds["u"].attrs["dimensional_factor"] == pytest.approx(U_REF)
    assert ds["w"].attrs["dimensional_factor"] == pytest.approx(
        DELTA * U_REF)
    assert ds["b"].attrs["dimensional_factor"] == pytest.approx(
        U_REF ** 2 / (RO * DELTA * L_REF))
    # coordinate factors (the stagger suffix strips to the row)
    assert ds["x"].attrs["dimensional_factor"] == pytest.approx(L_REF)
    assert ds["z_right"].attrs["dimensional_factor"] == pytest.approx(
        DELTA * L_REF)
    assert ds["time"].attrs["units"] == "1"
