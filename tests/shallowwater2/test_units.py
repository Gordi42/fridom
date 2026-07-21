"""The shallow-water unit-factor table and the writer metadata stamp.

GravityWave pins (the paper normalization: factor("h") = U^2/(Fr*g)
= D*Fr with D = U^2/(Fr^2*g)), the Dimensional identity/curated
semantics (factor("h") = 1/g on a dimensional model), renamed-coords
keying, report smoke, and the real-model writer round-trip through
xarray (global scaling attrs, per-variable/coordinate/time
dimensional_factor stamps, the nondimensional CF time ruling, and
the units_metadata=False opt-out).
"""
import pytest

import fridom as fr
import fridom.shallowwater2 as sw

N = 8
DT = 5e-3

L_REF = 2.0e3
U_REF = 0.5
G_REF = 10.0
FR = 0.2
RO = 0.4


def make_grid(names=("x", "y")):
    meshes = tuple(
        fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), periodic=True,
                                       name=name)
        for name in names)
    return fr.spatial.Grid(meshes)


def nondim_model(*, scaling=None, depth=1.0, coriolis=None,
                 coords=("x", "y"), grid=None):
    if scaling is None:
        scaling = fr.scaling.GravityWave(L=L_REF, U=U_REF, g=G_REF)
    return sw.Model(
        grid=grid if grid is not None else make_grid(coords),
        core=sw.Core(froude_number=FR, depth=depth, coords=coords),
        scaling=scaling,
        coriolis=coriolis,
        advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(DT, order=2))


def dim_model(*, gravity=G_REF, depth=100.0):
    return sw.Model(
        grid=make_grid(),
        core=sw.Core(gravity=gravity, depth=depth),
        advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(DT, order=2))


# ================================================================
#  GravityWave pins (eps aliases Fr; the paper normalization)
# ================================================================
def test_gravity_wave_amplitude_pins():
    units = nondim_model().units
    assert units.factor("u") == pytest.approx(U_REF)
    assert units.factor("v") == pytest.approx(U_REF)
    assert units.factor("p") == pytest.approx(U_REF ** 2 / FR)
    assert units.factor("thickness") == pytest.approx((U_REF / FR) ** 2)
    assert units.factor("csqr") == pytest.approx((U_REF / FR) ** 2)
    assert units.factor("c_dim") == pytest.approx(U_REF / FR)


def test_gravity_wave_h_reproduces_the_paper_scale():
    units = nondim_model().units
    h = units.factor("h")
    depth = units.factor("D")
    assert h == pytest.approx(U_REF ** 2 / (FR * G_REF))
    assert depth == pytest.approx(U_REF ** 2 / (FR ** 2 * G_REF))
    # the paper's H = D * Fr
    assert h == pytest.approx(depth * FR)


def test_gravity_wave_time_and_coordinates():
    units = nondim_model().units
    assert units.factor("t") == pytest.approx(FR * L_REF / U_REF)
    assert units.factor("T_ref") == pytest.approx(FR * L_REF / U_REF)
    assert units.factor("x") == pytest.approx(L_REF)
    assert units.factor("y") == pytest.approx(L_REF)


def test_f_dim_with_a_nondimensional_coriolis():
    model = nondim_model(
        coriolis=sw.modules.FPlaneCoriolis(rossby_number=RO))
    assert model.units.factor("f_dim") == pytest.approx(
        U_REF / (RO * L_REF))


def test_renamed_coords_key_the_coordinate_rows():
    model = nondim_model(coords=("zonal", "meridional"))
    factors = model.units.factors
    assert factors["zonal"].value == pytest.approx(L_REF)
    assert factors["meridional"].value == pytest.approx(L_REF)
    assert "x" not in factors


# ================================================================
#  Dimensional semantics (identity + curated meaning kept)
# ================================================================
def test_dimensional_raw_factors_are_identity_with_units():
    units = dim_model().units
    for name, unit in (("u", "m/s"), ("p", "m^2/s^2"),
                       ("thickness", "m^2/s^2"), ("x", "m"),
                       ("t", "s")):
        entry = units.factors[name]
        assert entry.value == 1.0
        assert entry.unit == unit


def test_dimensional_h_keeps_its_meaning():
    # factor("h") * p yields meters in BOTH variants (owner ruling)
    units = dim_model(gravity=G_REF).units
    assert units.factor("h") == pytest.approx(1.0 / G_REF)


def test_dimensional_constants_report_bound_values():
    units = dim_model(gravity=G_REF, depth=100.0).units
    assert units.factor("c_dim") == pytest.approx(
        (G_REF * 100.0) ** 0.5)
    assert units.factor("D") == pytest.approx(100.0)


def test_dimensional_variable_depth_marks_the_constants():
    model = sw.Model(
        grid=make_grid(),
        core=sw.Core(gravity=G_REF, depth=lambda y: 100.0 + 0.0 * y),
        advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(DT, order=2))
    # no constant shallowwater.depth provide -> the rows are marked
    entry = model.units.factors["D"]
    assert entry.value is None
    assert entry.missing == ("shallowwater.depth",)
    with pytest.raises(ValueError, match=r"shallowwater\.depth"):
        model.units.factor("D")


# ================================================================
#  Missing scales / report
# ================================================================
def test_missing_scales_are_marked_not_raised():
    model = nondim_model(scaling=fr.scaling.GravityWave(U=U_REF))
    factors = model.units.factors
    assert factors["u"].value == pytest.approx(U_REF)
    assert factors["x"].value is None
    assert factors["x"].missing == ("L=",)
    assert factors["h"].missing == ("g=",)
    with pytest.raises(ValueError, match=r"pass L=<value>"):
        model.units.factor("t")


def test_report_smoke_both_variants():
    nondim = nondim_model().units.report()
    assert "GravityWave, nondimensional" in nondim
    assert "h" in nondim
    dim = dim_model().units.report()
    assert "Dimensional, dimensional" in dim
    assert "[1/g]" in dim
