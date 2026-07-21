"""The hydrostatic unit-factor table (``model.units`` rows).

Rotational pins (eps aliases Ro; the flat-only vertical convention
H = the vertical mesh extent: w = U*H/L, b = U^2/(eps*H), N_dim
consistency n2_eff = Bu from the primitives, c_dim = U/Fr_ext), the
Dimensional identity/constant semantics (c_dim = sqrt(g*H_ref)),
report smoke, and the real-model writer round-trip through xarray.
"""
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.io.triggers import every
from fridom.model.io.writer import Writer

DT = 2.0 ** -7

L_REF = 2.0e3
U_REF = 0.5
HEIGHT = 0.5       # the vertical mesh extent H
RO = 0.25          # eps under Rotational()
FR_INT = 0.125
FR_EXT = 0.25
F0 = 1.0e-4
N2 = 4.0e-4
G = 9.81
H_DIM = 100.0


def make_grid(depth=HEIGHT):
    im = fr.spatial.meshes.IntervalMesh
    return fr.spatial.Grid((
        im(4, (0.0, 1.0), periodic=True, name="x"),
        im(4, (0.0, 1.0), periodic=True, name="y"),
        im(4, (0.0, depth), periodic=False, name="z")),
        device_ids=(0,))


def rot_model(*, free_surface=None):
    return hy.Model(
        grid=make_grid(),
        core=hy.Core(),
        scaling=fr.scaling.Rotational(L=L_REF, U=U_REF),
        coriolis=hy.FPlaneCoriolis(rossby_number=RO),
        stratification=hy.ConstantStratification(froude_number=FR_INT),
        free_surface=free_surface
        or hy.ExplicitFreeSurface(froude_number=FR_EXT),
        advection=False, surface_advective_flux=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(DT, order=3))


def dim_model():
    return hy.Model(
        grid=make_grid(depth=H_DIM),
        core=hy.Core(gravity=G),
        coriolis=hy.FPlaneCoriolis(f0=F0),
        stratification=hy.ConstantStratification(n2=N2),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False, surface_advective_flux=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(DT, order=3))


# ================================================================
#  Rotational pins (eps aliases Ro; vertical scale H = mesh extent)
# ================================================================
def test_rotational_amplitude_pins():
    units = rot_model().units
    assert units.factor("u") == pytest.approx(U_REF)
    assert units.factor("v") == pytest.approx(U_REF)
    assert units.factor("w") == pytest.approx(U_REF * HEIGHT / L_REF)
    assert units.factor("b") == pytest.approx(
        U_REF ** 2 / (RO * HEIGHT))
    assert units.factor("ps") == pytest.approx(U_REF ** 2 / RO)
    assert units.factor("p_hyd") == pytest.approx(U_REF ** 2 / RO)


def test_rotational_coordinates_and_time():
    units = rot_model().units
    assert units.factor("x") == pytest.approx(L_REF)
    assert units.factor("y") == pytest.approx(L_REF)
    assert units.factor("z") == pytest.approx(HEIGHT)
    assert units.factor("t") == pytest.approx(RO * L_REF / U_REF)


def test_rotational_constants_are_consistent_with_the_primitives():
    units = rot_model().units
    assert units.factor("c_dim") == pytest.approx(U_REF / FR_EXT)
    n_dim = units.factor("N_dim")
    assert n_dim == pytest.approx(U_REF / (FR_INT * HEIGHT))
    # Bu = (N_dim H / (f_dim L))^2 reproduces the model-units
    # effective stratification n2_eff = (eps/Fr_int)^2
    burger = (n_dim * units.factor("z")
              / (units.factor("f_dim") * units.factor("x"))) ** 2
    assert burger == pytest.approx((RO / FR_INT) ** 2)


def test_implicit_free_surface_carries_the_rows():
    units = rot_model(free_surface=hy.ImplicitFreeSurface(
        froude_number=FR_EXT)).units
    assert units.factor("ps") == pytest.approx(U_REF ** 2 / RO)
    assert units.factor("c_dim") == pytest.approx(U_REF / FR_EXT)


# ================================================================
#  Dimensional semantics (identity + constants keep their meaning)
# ================================================================
def test_dimensional_raw_factors_are_identity_with_units():
    units = dim_model().units
    for name, unit in (("u", "m/s"), ("w", "m/s"), ("b", "m/s^2"),
                       ("ps", "m^2/s^2"), ("z", "m"), ("t", "s")):
        entry = units.factors[name]
        assert entry.value == 1.0
        assert entry.unit == unit


def test_dimensional_constants_report_bound_values():
    units = dim_model().units
    assert units.factor("c_dim") == pytest.approx((G * H_DIM) ** 0.5)
    assert units.factor("N_dim") == pytest.approx(N2 ** 0.5)
    assert units.factor("f_dim") == pytest.approx(F0)


# ================================================================
#  Report
# ================================================================
def test_report_smoke_both_variants():
    nondim = rot_model().units.report()
    assert "Rotational, nondimensional" in nondim
    assert "[U/Fr_ext]" in nondim
    assert "[U^2/(eps*H)]" in nondim
    dim = dim_model().units.report()
    assert "Dimensional, dimensional" in dim
    assert "[sqrt(g*H_ref)]" in dim


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
    assert params["hydrostatic.froude"] == pytest.approx(FR_EXT)
    assert params["stratification.froude"] == pytest.approx(FR_INT)
    # per-variable factors (the vertical family rides H)
    assert ds["u"].attrs["dimensional_factor"] == pytest.approx(U_REF)
    assert ds["ps"].attrs["dimensional_factor"] == pytest.approx(
        U_REF ** 2 / RO)
    assert ds["b"].attrs["dimensional_factor"] == pytest.approx(
        U_REF ** 2 / (RO * HEIGHT))
    # coordinate factors (x: L; z: the mesh-extent convention)
    assert ds["x"].attrs["dimensional_factor"] == pytest.approx(L_REF)
    assert ds["z"].attrs["dimensional_factor"] == pytest.approx(HEIGHT)
    assert ds["time"].attrs["units"] == "1"
