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
from fridom.io.triggers import every
from fridom.io.writer import Writer

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
        advection=None,
        time_stepper=fr.model.time_steppers.AdamBashforth(DT, order=2))


def dim_model(*, gravity=G_REF, depth=100.0):
    return sw.Model(
        grid=make_grid(),
        core=sw.Core(gravity=gravity, depth=depth),
        advection=None,
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
        assert entry.target_unit == unit


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
        advection=None,
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


# ================================================================
#  Writer round-trip (the real-model SD metadata stamp)
# ================================================================
def _write_store(model, path, **writer_kwargs):
    writer = Writer(path, trigger=every(steps=1), **writer_kwargs)
    writer.bind(model)
    writer.write(model.carry)
    writer.close()


def test_writer_stamps_a_nondimensional_model(tmp_path):
    xr = pytest.importorskip("xarray")
    model = nondim_model(
        coriolis=sw.modules.FPlaneCoriolis(rossby_number=RO))
    path = tmp_path / "nondim.zarr"
    _write_store(model, path)
    ds = xr.open_zarr(path, consolidated=False)
    # global scaling attrs
    assert ds.attrs["fridom_scaling"] == "GravityWave"
    assert ds.attrs["fridom_scaling_nondimensional"] is True
    assert ds.attrs["fridom_scaling_L"] == L_REF
    assert ds.attrs["fridom_scaling_U"] == U_REF
    assert ds.attrs["fridom_scaling_g"] == G_REF
    t_ref = FR * L_REF / U_REF
    assert ds.attrs["fridom_scaling_T_ref"] == pytest.approx(t_ref)
    assert ds.attrs["fridom_scaling_epsilon"] == pytest.approx(FR)
    params = ds.attrs["fridom_scaling_parameters"]
    assert params["scaling.nonlinearity"] == pytest.approx(FR)
    assert params["shallowwater.froude"] == pytest.approx(FR)
    assert params["coriolis.rossby"] == pytest.approx(RO)
    # per-variable factors (u/v/p PROGNOSTIC + thickness DIAGNOSTIC)
    assert ds["u"].attrs["dimensional_factor"] == pytest.approx(U_REF)
    assert ds["u"].attrs["dimensional_units"] == "m/s"
    assert ds["u"].attrs["dimensional_factor_expr"] == "U"
    assert ds["p"].attrs["dimensional_factor"] == pytest.approx(
        U_REF ** 2 / FR)
    assert ds["thickness"].attrs["dimensional_factor"] == (
        pytest.approx((U_REF / FR) ** 2))
    # coordinate factors (the stagger suffix strips to the row)
    assert ds["x_right"].attrs["dimensional_factor"] == (
        pytest.approx(L_REF))
    assert ds["y"].attrs["dimensional_factor"] == pytest.approx(L_REF)
    # CF option (b): dimensionless time, no calendar anchor,
    # dimensional_factor = T_ref alongside
    time_attrs = ds["time"].attrs
    assert time_attrs["units"] == "1"
    assert "calendar" not in time_attrs
    assert time_attrs["dimensional_factor"] == pytest.approx(t_ref)


def test_writer_keeps_cf_time_on_a_dimensional_model(tmp_path):
    xr = pytest.importorskip("xarray")
    model = dim_model()
    path = tmp_path / "dim.zarr"
    _write_store(model, path)
    ds = xr.open_zarr(path, consolidated=False, decode_times=False)
    assert ds.attrs["fridom_scaling"] == "Dimensional"
    assert ds.attrs["fridom_scaling_nondimensional"] is False
    # identity factors with the physical units; CF attrs unchanged
    assert ds["p"].attrs["dimensional_factor"] == 1.0
    assert ds["time"].attrs["units"] == "seconds"
    assert ds["time"].attrs["dimensional_factor"] == 1.0


def test_writer_units_metadata_false_stamps_nothing(tmp_path):
    xr = pytest.importorskip("xarray")
    model = nondim_model()
    path = tmp_path / "off.zarr"
    _write_store(model, path, units_metadata=False)
    ds = xr.open_zarr(path, consolidated=False)
    assert not any(key.startswith("fridom_scaling")
                   for key in ds.attrs)
    assert "dimensional_factor" not in ds["u"].attrs
    assert "dimensional_factor" not in ds["time"].attrs


# ================================================================
#  Scaling-rendered CF units (variables, coordinates, the sphere)
# ================================================================
def test_nondimensional_store_claims_no_physical_unit(tmp_path):
    xr = pytest.importorskip("xarray")
    model = nondim_model()
    path = tmp_path / "rendered.zarr"
    _write_store(model, path)
    ds = xr.open_zarr(path, consolidated=False)
    # every variable, every coordinate and the time axis agree
    for name in ("u", "v", "p", "thickness"):
        assert ds[name].attrs["units"] == "1"
    for name in ("x", "y", "x_right", "time"):
        assert ds[name].attrs["units"] == "1"
    # the physical unit survives as the conversion target
    assert ds["u"].attrs["dimensional_units"] == "m/s"


def test_dimensional_store_keeps_the_physical_units(tmp_path):
    xr = pytest.importorskip("xarray")
    model = dim_model()
    path = tmp_path / "physical.zarr"
    _write_store(model, path)
    ds = xr.open_zarr(path, consolidated=False, decode_times=False)
    assert ds["u"].attrs["units"] == "m/s"
    assert ds["p"].attrs["units"] == "m^2/s^2"
    assert ds["x"].attrs["units"] == "m"


def test_state_fields_report_the_rendered_units():
    # the in-memory surface agrees with the store (the .xr path)
    nondim = nondim_model().state
    assert nondim["u"].metadata.units == "1"
    assert nondim["u"].metadata.physical_units == "m/s"
    assert dim_model().state["u"].metadata.units == "m/s"


def test_spherical_coordinates_are_radians_not_metres(tmp_path):
    # the row's unit is the unit of factor*value (metres of arc);
    # the stored values are angles, and the chart says so
    xr = pytest.importorskip("xarray")
    grid = fr.spatial.spherical.Grid((16, 8), radius=6.371e6,
                                     lat_extent=(-1.0, 1.0))
    model = sw.Model(
        grid=grid,
        core=sw.Core(gravity=G_REF, depth=100.0,
                     coords=("lon", "lat")),
        advection=None,
        time_stepper=fr.model.time_steppers.AdamBashforth(
            DT, order=2))
    path = tmp_path / "sphere.zarr"
    _write_store(model, path)
    ds = xr.open_zarr(path, consolidated=False, decode_times=False)
    for name in ("lon", "lat"):
        assert ds[name].attrs["units"] == "rad"
        # the metres-of-arc conversion is still stamped alongside
        assert ds[name].attrs["dimensional_units"] == "m"
    assert float(ds["lat"].max()) <= 1.0


# ================================================================
#  Drift lint: the one place the two unit sources overlap
# ================================================================
def test_component_rows_agree_with_the_field_annotations():
    """The five-ish strings kept in two places must not drift.

    ``FieldMetadata.physical_units`` is the unit of the STORED
    value; ``UnitFactor.target_unit`` is the unit of
    ``factor * value``. They coincide only for **component** rows,
    whose dimensional factor is the identity — which is exactly the
    overlap this pins. Deliberately not compared:

    - ``coordinate`` rows, where the two legitimately differ (the
      lat-lon rows store radians and target metres of arc; reading
      the row as a CF claim is what once labelled radians "m");
    - ``curated`` rows (the shallow-water ``h`` folds ``1/g``);
    - ``constant`` / ``time`` rows, which annotate no field at all.

    Derived quantities carry metadata but no row, so they cannot
    drift — they are simply not convertible back (roadmap 2c).
    """
    model = dim_model()
    state = model.state
    names = set(state.component_names)
    overlap = {
        name: (state[name].metadata.physical_units, entry.target_unit)
        for name, entry in model.units.factors.items()
        if entry.kind == "component" and name in names}
    assert overlap, "no component row annotates a state field"
    drifted = {name: pair for name, pair in overlap.items()
               if pair[0] != pair[1]}
    assert drifted == {}


# ================================================================
#  Derived-quantity rows (5.6: converting a diagnostic back)
# ================================================================
def test_derived_rows_resolve_to_the_declared_amplitudes():
    units = nondim_model().units
    for name in ("rel_vort", "divergence"):
        assert units.factor(name) == pytest.approx(U_REF / L_REF), name
    for name in ("ekin", "epot"):
        assert units.factor(name) == pytest.approx(U_REF ** 2), name
    assert units.factor("epot_full") == pytest.approx(
        (U_REF ** 2 / FR) ** 2)


def test_the_thickness_weighted_family_is_deliberately_unstamped():
    """Absent, not forgotten: their factors turn on an open call.

    ``ekin_full`` / ``etot_full`` / ``pot_vort`` weight by the
    geopotential thickness, whose row is ``(U/Fr)^2`` while ``p`` is
    ``U^2/eps``; reconciling those is an owner call
    (``units_metadata_investigation.md`` 5.6). A quantity with no
    row is honestly unconvertible — a wrong row would repeat the
    lat-lon failure of reading a row as something it is not.
    """
    rows = dict(nondim_model().units.factors)
    for name in ("ekin_full", "etot_full", "pot_vort"):
        assert name not in rows, name


def test_derived_row_units_match_the_declared_annotations():
    rows = dict(dim_model().units.factors)
    for name in ("rel_vort", "divergence"):
        assert rows[name].target_unit == "1/s", name
    for name in ("ekin", "epot"):
        assert rows[name].target_unit == "m^2/s^2", name
    assert rows["epot_full"].target_unit == "m^4/s^4"
