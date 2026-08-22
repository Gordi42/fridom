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
from fridom.io.triggers import every
from fridom.io.writer import Writer

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
        buoyancy=nh.ConstantStratification(froude_number=FR_INT),
        advection=None,
        time_stepper=fr.model.time_steppers.AdamBashforth(DT, order=2))


def dim_model(*, buoyancy=None):
    return nh.Model(
        grid=make_grid(),
        core=nh.Core(aspect_ratio=DELTA),
        coriolis=nh.FPlaneCoriolis(f0=F0),
        buoyancy=buoyancy
        or nh.ConstantStratification(n2=N2),
        advection=None,
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
        assert entry.target_unit == unit


def test_dimensional_constants_report_bound_values():
    units = dim_model().units
    assert units.factor("N_dim") == pytest.approx(N2 ** 0.5)
    assert units.factor("f_dim") == pytest.approx(F0)


def test_meridional_stratification_marks_n_dim():
    # the profile binds no constant n2 -> the row is marked, never a
    # false constant (the beta-plane f_dim precedent)
    model = dim_model(buoyancy=nh.MeridionalStratification(
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


# ================================================================
#  The scaling-purity invariant (units are correct RELIABLY)
# ================================================================
def test_a_nondimensional_model_claims_no_physical_unit():
    # the standing gate: a physical unit string anywhere on a
    # nondimensional model's state is a bug, whatever declared it
    state = rot_model().state
    offenders = {
        name: state[name].metadata.units
        for name in state.component_names
        if state[name].metadata.units not in {"1", "unknown"}}
    assert offenders == {}


def test_a_dimensional_model_keeps_its_physical_units():
    state = dim_model().state
    units = {name: state[name].metadata.units
             for name in state.component_names}
    assert units["u"] == "m/s"
    assert units["b"] == "m/s^2"


def test_the_physical_unit_survives_nondimensionalization():
    # what the dimensional_factor converts BACK to
    field = rot_model().state["u"]
    assert field.metadata.units == "1"
    assert field.metadata.physical_units == "m/s"


def test_auxiliary_fields_share_the_rendering():
    # csqr/f_coriolis and friends are AUXILIARY: they take the
    # declared annotation through the re-materialization table, and
    # must not report a physical unit a PROGNOSTIC field would not
    dim = dim_model().state
    assert dim["f_coriolis"].metadata.units == "1/s"
    rot = rot_model().state
    assert rot["f_coriolis"].metadata.units == "1"
    assert rot["f_coriolis"].metadata.physical_units == "1/s"


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
    units = rot_model().units
    assert units.factor("rel_vort_z") == pytest.approx(U_REF / L_REF)
    for name in ("ekin", "epot", "etot"):
        assert units.factor(name) == pytest.approx(U_REF ** 2), name
    # NOT U/L despite the shared 1/s unit: the nondimensional
    # definition multiplies through by eps
    assert units.factor("linear_pot_vort") == pytest.approx(
        U_REF / (RO * L_REF))
    # the total buoyancy converts like b itself
    assert units.factor("b_total") == pytest.approx(
        U_REF ** 2 / (RO * DELTA * L_REF))


def test_derived_rows_are_identity_on_a_dimensional_model():
    units = dim_model().units
    rows = dict(units.factors)
    for name in ("rel_vort_z", "ekin", "epot", "etot",
                 "linear_pot_vort", "b_total"):
        assert units.factor(name) == 1.0, name
        assert rows[name].kind == "derived"


def test_derived_row_units_match_the_declared_annotations():
    rows = dict(dim_model().units.factors)
    assert rows["rel_vort_z"].target_unit == "1/s"
    assert rows["linear_pot_vort"].target_unit == "1/s"
    assert rows["b_total"].target_unit == "m/s^2"
    for name in ("ekin", "epot", "etot"):
        assert rows[name].target_unit == "m^2/s^2", name


# ================================================================
#  The store round-trip: a diagnostic is convertible again
# ================================================================
def test_a_user_named_derived_variable_gets_the_canonical_row(
        tmp_path):
    """The output key is the user's; the row is keyed canonically."""
    xr = pytest.importorskip("xarray")
    model = rot_model()
    path = tmp_path / "derived.zarr"
    writer = Writer(
        path, fields=["u"],
        derived={"vort": lambda ms: ms.state.rel_vort_z},
        trigger=every(steps=1))
    writer.bind(model)
    writer.write(model.carry)
    writer.close()
    ds = xr.open_zarr(path, consolidated=False)
    # nondimensional store: both dimensionless, both convertible
    assert ds["u"].attrs["units"] == "1"
    assert ds["vort"].attrs["units"] == "1"
    assert ds["u"].attrs["dimensional_factor"] == pytest.approx(U_REF)
    assert ds["vort"].attrs["dimensional_factor"] == pytest.approx(
        U_REF / L_REF)
    assert ds["vort"].attrs["dimensional_units"] == "1/s"


def test_an_ad_hoc_quantity_can_declare_its_own_factor(tmp_path):
    """No package table can know a user's own diagnostic."""
    xr = pytest.importorskip("xarray")
    model = rot_model()

    def mine(model_state):
        zeta = model_state.state.rel_vort_z
        return zeta.new_quantity(
            zeta.data * 2.0, name="mine", long_name="My thing",
            units="m/s^2")

    row = fr.model.UnitFactor(
        target_unit="m/s^2", expr="U^2/L", kind="derived",
        scales=("L", "U"), fn=lambda values: (
            values["U"] ** 2 / values["L"]))
    path = tmp_path / "adhoc.zarr"
    writer = Writer(path, fields=[], derived={"mine": mine},
                    unit_factors={"mine": row},
                    trigger=every(steps=1))
    writer.bind(model)
    writer.write(model.carry)
    writer.close()
    ds = xr.open_zarr(path, consolidated=False)
    attrs = ds["mine"].attrs
    assert attrs["dimensional_factor"] == pytest.approx(
        U_REF ** 2 / L_REF)
    assert attrs["dimensional_units"] == "m/s^2"
    assert attrs["dimensional_factor_expr"] == "U^2/L"
