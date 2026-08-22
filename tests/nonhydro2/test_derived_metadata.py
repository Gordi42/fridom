"""The nonhydro2 derived-quantity metadata gate.

Every derived quantity — the parameter-free ``nh.State`` properties and
the bound ``model.diagnostics.*`` functions — must carry its OWN
annotation rather than a neighbour's or a blank one
(``design/research/units_metadata_investigation.md`` sections 3 and 9).

The check that works is **name identity**: ``metadata.name`` equals the
canonical key. A naive "is it still the default?" test passes happily
on a quantity that borrowed pressure's record (``ekin`` used to ship as
``name="p"`` / ``long_name="Pressure"``), which is the majority of the
defect; name identity catches the borrowed and the blank alike.

The last test pins the scaling half: on a nondimensional model the
reported ``units`` is ``"1"`` while ``physical_units`` still knows what
the quantity would be dimensionally.
"""
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.spatial.fields.metadata import (
    DIMENSIONLESS_UNITS,
    UNKNOWN_UNITS,
    FieldMetadata,
)

N = 8
DT = 2.0 ** -6
DEFAULT = FieldMetadata()

#: canonical key -> declared PHYSICAL unit
DERIVED = {
    "rel_vort_z": "1/s",
    "ekin": "m^2/s^2",
    "epot": "m^2/s^2",
    "etot": "m^2/s^2",
    "linear_pot_vort": "1/s",
}
KEYS = sorted(DERIVED)

#: the keys served by ``nh.State`` rather than ``model.diagnostics``
STATE_PROPERTIES = ("rel_vort_z",)


def make_grid():
    """Return a tiny triply-periodic 8^3 grid."""
    im = fr.spatial.meshes.IntervalMesh
    return fr.spatial.Grid(tuple(
        im(N, (0.0, 2.0 * np.pi), periodic=True, name=name)
        for name in ("x", "y", "z")), device_ids=(0,))


def make_model(*, nondimensional):
    """Assemble a minimal model; nothing here time-steps."""
    stepper = fr.model.time_steppers.AdamBashforth(DT, order=2)
    if nondimensional:
        return nh.Model(
            grid=make_grid(),
            core=nh.Core(aspect_ratio=0.5),
            scaling=fr.scaling.Rotational(L=2.0e3, U=0.5),
            coriolis=nh.modules.FPlaneCoriolis(rossby_number=0.25),
            buoyancy=nh.ConstantStratification(froude_number=0.125),
            advection=None, time_stepper=stepper)
    return nh.Model(
        grid=make_grid(),
        core=nh.Core(aspect_ratio=0.5),
        coriolis=nh.modules.FPlaneCoriolis(f0=1.0e-4),
        buoyancy=nh.ConstantStratification(n2=4.0e-4),
        advection=None, time_stepper=stepper)


@pytest.fixture(scope="module")
def dimensional():
    """Return a dimensional model, assembled once for the file."""
    return make_model(nondimensional=False)


@pytest.fixture(scope="module")
def nondimensional():
    """Return a nondimensional (Rotational) model, assembled once."""
    return make_model(nondimensional=True)


def derived(model, key):
    """Return the derived quantity registered under ``key``."""
    if key in STATE_PROPERTIES:
        return getattr(model.state, key)
    return getattr(model.diagnostics, key)()


# ================================================================
#  The gate: name identity, a declared long name, a declared unit
# ================================================================
@pytest.mark.parametrize("key", KEYS, ids=KEYS)
def test_derived_quantity_owns_its_name(dimensional, key):
    assert derived(dimensional, key).metadata.name == key


@pytest.mark.parametrize("key", KEYS, ids=KEYS)
def test_derived_quantity_declares_a_long_name(dimensional, key):
    long_name = derived(dimensional, key).metadata.long_name
    assert long_name != DEFAULT.long_name


@pytest.mark.parametrize("key", KEYS, ids=KEYS)
def test_derived_quantity_declares_its_physical_unit(dimensional, key):
    metadata = derived(dimensional, key).metadata
    assert metadata.physical_units != UNKNOWN_UNITS
    assert metadata.physical_units == DERIVED[key]
    assert metadata.units == DERIVED[key]


@pytest.mark.parametrize("key", KEYS, ids=KEYS)
def test_derived_quantity_exports_under_its_own_name(dimensional, key):
    # the reported symptom: diagnostics.ekin().xr was named "p"
    assert derived(dimensional, key).xr.name == key


def test_every_bound_diagnostic_is_gated():
    # a new diagnostic must be added to DERIVED, or this fails
    assert set(nh.diagnostics.DIAGNOSTICS) <= set(DERIVED)


# ================================================================
#  The scaling half: nondimensional renders "1", physical survives
# ================================================================
@pytest.mark.parametrize("key", KEYS, ids=KEYS)
def test_nondimensional_model_reports_dimensionless(
        nondimensional, key):
    metadata = derived(nondimensional, key).metadata
    assert metadata.units == DIMENSIONLESS_UNITS
    assert metadata.physical_units == DERIVED[key]


def test_a_declared_row_agrees_with_the_declared_annotation():
    """The second place the two unit sources overlap (5.5/5.6).

    A derived quantity's conversion row is identity on a dimensional
    model, so its ``target_unit`` must equal the physical unit the
    diagnostic declares. A quantity with no row is skipped: absent
    rows are deliberate where a factor is not derived.
    """
    model = make_model(nondimensional=False)
    rows = dict(model.units.factors)
    checked = {}
    for key, unit in DERIVED.items():
        row = rows.get(key)
        if row is None:
            continue
        checked[key] = (unit, row.target_unit)
    assert checked, "no derived quantity carries a conversion row"
    drifted = {k: v for k, v in checked.items() if v[0] != v[1]}
    assert drifted == {}
