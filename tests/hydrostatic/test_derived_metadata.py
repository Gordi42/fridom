"""The hydrostatic derived-quantity metadata gate.

Every derived quantity — the parameter-free ``hy.State`` properties and
the bound ``model.diagnostics.*`` functions — must carry its OWN
annotation rather than a neighbour's or a blank one
(``design/research/units_metadata_investigation.md`` sections 3 and 9).

The check that works is **name identity**: ``metadata.name`` equals the
canonical key. A naive "is it still the default?" test passes happily
on a quantity that borrowed a neighbour's record, which is the
majority of the defect elsewhere in the stack; name identity catches
the borrowed and the blank alike. This package was already the house
style, so the gate is here to keep it that way.

The last test pins the scaling half: on a nondimensional model the
reported ``units`` is ``"1"`` while ``physical_units`` still knows what
the quantity would be dimensionally.
"""
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.spatial.fields.metadata import (
    DIMENSIONLESS_UNITS,
    UNKNOWN_UNITS,
    FieldMetadata,
)

N = 8
NZ = 4
DT = 2.0 ** -7
DEFAULT = FieldMetadata()

#: canonical key -> declared PHYSICAL unit
DERIVED = {
    "rel_vort_z": "1/s",
    "hor_divergence": "1/s",
    "ekin": "m^2/s^2",
    "epot": "m^2/s^2",
    "eta": "m",
}
KEYS = sorted(DERIVED)

#: defined on the dimensional variant only: ``eta = ps / g`` reads the
#: dimensional ``hydrostatic.gravity`` provide, which the
#: nondimensional core does not carry (read ``ps`` directly there)
DIMENSIONAL_ONLY = ("eta",)
SCALED_KEYS = [key for key in KEYS if key not in DIMENSIONAL_ONLY]

#: the keys served by ``hy.State`` rather than ``model.diagnostics``
STATE_PROPERTIES = ("rel_vort_z", "hor_divergence")


def make_grid(depth):
    """Return a horizontally periodic, vertically bounded grid."""
    im = fr.spatial.meshes.IntervalMesh
    return fr.spatial.Grid((
        im(N, (0.0, 1.0), periodic=True, name="x"),
        im(N, (0.0, 1.0), periodic=True, name="y"),
        im(NZ, (0.0, depth), periodic=False, name="z")))


def make_model(*, nondimensional):
    """Assemble a minimal model; nothing here time-steps."""
    stepper = fr.model.time_steppers.AdamBashforth(DT, order=3)
    if nondimensional:
        return hy.Model(
            grid=make_grid(0.5),
            core=hy.Core(),
            scaling=fr.scaling.Rotational(L=2.0e3, U=0.5),
            coriolis=hy.FPlaneCoriolis(rossby_number=0.25),
            buoyancy=hy.ConstantStratification(froude_number=0.125),
            free_surface=hy.ExplicitFreeSurface(froude_number=0.25),
            advection=None,
            time_stepper=stepper)
    return hy.Model(
        grid=make_grid(100.0),
        core=hy.Core(gravity=9.81),
        coriolis=hy.FPlaneCoriolis(f0=1.0e-4),
        buoyancy=hy.ConstantStratification(n2=4.0e-4),
        free_surface=hy.ExplicitFreeSurface(),
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
    # the reported symptom elsewhere: diagnostics.ekin().xr named "p"
    assert derived(dimensional, key).xr.name == key


def test_every_bound_diagnostic_is_gated():
    # a new diagnostic must be added to DERIVED, or this fails
    assert set(hy.diagnostics.DIAGNOSTICS) <= set(DERIVED)


# ================================================================
#  The scaling half: nondimensional renders "1", physical survives
# ================================================================
@pytest.mark.parametrize("key", SCALED_KEYS, ids=SCALED_KEYS)
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
