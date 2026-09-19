r"""``hy.TemperatureSalinity``: prognostic T / S and an equation of state.

The gates of the formulation:

- **buoyancy-tracer equivalence** — a linear EOS with constant salinity
  reproduces the ``hy.BuoyancyTracer`` trajectory to rounding;
- the diagnosed ``b`` is the EOS buoyancy of the current ``T``/``S`` at
  the cell's geopotential depth (flat and terrain columns);
- **tracer content conservation** under advection + vertical mixing;
- **rest state**: horizontally uniform ``T(z)``, ``S(z)`` stays exactly
  at rest on a flat grid (nonlinear, depth-dependent EOS included);
- the taught errors, the reductions, metadata, the immersed mask.

The terrain pressure-gradient-error, autodiff and device-invariance
gates live in the ``test_temperature_salinity_*`` shards.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.hydrostatic.eos import EquationOfState
from fridom.hydrostatic.modules.temperature_salinity import _column
from fridom.hydrostatic.params import EOS_ALPHA, EOS_BETA
from fridom.model.errors import AssemblyError
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.spaces.average import CellAvg

IM = fr.spatial.meshes.IntervalMesh
G = 9.81
LENGTH = 1.0e5
DEPTH = 1000.0
DT = 50.0
CenteredAdvection = fr.model.modules.CenteredAdvection


# ================================================================
#  Builders
# ================================================================
def make_grid(n=8, nz=6, **kwargs):
    """Return a 100 km doubly-periodic, 1 km deep bounded-z grid."""
    return fr.spatial.Grid((
        IM(n, (0.0, LENGTH), periodic=True, name="x"),
        IM(n, (0.0, LENGTH), periodic=True, name="y"),
        IM(nz, (-DEPTH, 0.0), periodic=False, name="z")), **kwargs)


def make_model(buoyancy, *, grid=None, advection=CenteredAdvection,
               extra=(), core=None, imex=False):
    """Assemble a dimensional hydrostatic model on the buoyancy module.

    ``imex=True`` swaps in the CNAB2 stepper and the implicit free
    surface, the pair the implicit ``VerticalMixing`` needs.
    """
    return hy.Model(
        grid=make_grid() if grid is None else grid,
        core=hy.Core(gravity=G) if core is None else core,
        time_stepper=(fr.model.time_steppers.CNAB2(DT) if imex
                      else AdamBashforth(DT, order=3)),
        coriolis=hy.FPlaneCoriolis(f0=1.0e-4),
        buoyancy=buoyancy,
        free_surface=(hy.ImplicitFreeSurface() if imex
                      else hy.ExplicitFreeSurface()),
        advection=(advection() if isinstance(advection, type)
                   else advection),
        modules_extra=extra)


def temperature(x, y, z):
    """Return a thermocline with a horizontal warm anomaly."""
    return (4.0 + 14.0 * np.exp(z / 300.0)
            + 0.5 * np.sin(2 * np.pi * x / LENGTH)
            * np.cos(2 * np.pi * y / LENGTH))


def salinity(x, y, z):
    """Return a halocline with a horizontal fresh anomaly."""
    return (35.0 - 0.8 * np.exp(z / 200.0)
            + 0.1 * np.cos(2 * np.pi * x / LENGTH) + 0.0 * y)


def nodes(model, name):
    """Return the broadcastable node coordinates of the ``T`` cells."""
    space = model.state["b"].function_space
    return np.asarray(model.grid.evaluation_nodes(space.bare, name).data)


def field(model, name):
    """Return one state field as a host array."""
    return np.asarray(model.state[name].data)


EOS = [
    pytest.param(hy.LinearEOS, id="linear"),
    pytest.param(hy.RoquetEOS, id="roquet"),
    pytest.param(hy.TEOS10EOS, id="teos10"),
]


# ================================================================
#  Declarations and metadata
# ================================================================
def test_declares_the_tracers_and_a_diagnostic_buoyancy():
    model = make_model(hy.TemperatureSalinity())
    table = model._artifacts.field_table
    assert {"T", "S"} <= set(table.prognostic)
    assert "b" not in table.prognostic
    assert table["b"].lifecycle is fr.model.Lifecycle.DIAGNOSTIC
    # T and S are the mixable tracers; the diagnosed b is not one
    tracers = set(table.select(fr.model.roles.TRACER))
    assert tracers == {"T", "S"}


def test_field_metadata():
    model = make_model(hy.TemperatureSalinity())
    t_attrs = model.state["T"].xr.attrs
    s_attrs = model.state["S"].xr.attrs
    assert t_attrs["units"] == "degC"
    assert t_attrs["long_name"] == "Conservative temperature"
    assert t_attrs["standard_name"] == (
        "sea_water_conservative_temperature")
    assert s_attrs["units"] == "g/kg"
    assert s_attrs["standard_name"] == "sea_water_absolute_salinity"
    assert model.state["b"].xr.attrs["units"] == "m/s^2"


def test_the_reference_parcel_has_zero_buoyancy():
    eos = hy.TEOS10EOS(reference=(8.0, 34.5))
    model = make_model(hy.TemperatureSalinity(eos))
    # prognostic fields start at zero: both tracers must be set
    assert np.all(field(model, "T") == 0.0)
    assert float(model.diagnostics.b_total().data.min()) > 0.1
    model.set_fields(T=lambda x, y, z: 8.0 + 0.0 * (x + y + z),
                     S=lambda x, y, z: 34.5 + 0.0 * (x + y + z))
    assert float(jnp.abs(model.diagnostics.b_total().data).max()) == 0.0
    model.advance(2)
    # inside the jit-compiled step the polynomial is contracted (FMA)
    # differently from its constant-folded reference evaluation: the
    # anomaly is zero to one or two ulp of the density, ulp(1022) g /
    # rho0 = 1.1e-15 m/s^2 — and bitwise uniform, so nothing is forced
    stored = field(model, "b")
    assert np.abs(stored).max() < 5e-15
    assert np.abs(stored - stored[0, 0, 0]).max() == 0.0
    assert float(jnp.abs(model.state["u"].data).max()) < 1e-25


def test_default_eos_is_linear_and_publishes_its_coefficients():
    module = hy.TemperatureSalinity()
    assert isinstance(module.eos, hy.LinearEOS)
    assert module.tracers == ("T", "S")
    model = make_model(module)
    assert float(model.parameters[EOS_ALPHA]) == pytest.approx(1.6131e-4)
    assert float(model.parameters[EOS_BETA]) == pytest.approx(7.4614e-4)


def test_nonlinear_eos_publishes_no_coefficient():
    model = make_model(hy.TemperatureSalinity(hy.TEOS10EOS()))
    assert EOS_ALPHA not in model.parameters
    assert EOS_BETA not in model.parameters


def test_updating_alpha_changes_the_diagnosed_buoyancy():
    model = make_model(hy.TemperatureSalinity(hy.LinearEOS()))
    model.set_fields(T=temperature, S=salinity)
    before = np.asarray(model.diagnostics.b_total().data)
    model.update_parameters({EOS_ALPHA: 3.0e-4, EOS_BETA: 0.0})
    after = np.asarray(model.diagnostics.b_total().data)
    want = G * 3.0e-4 * (field(model, "T") - 10.0)
    assert not np.allclose(before, after)
    np.testing.assert_allclose(after, want, rtol=1e-13, atol=1e-18)


# ================================================================
#  The diagnosed buoyancy is the EOS buoyancy at the cell depth
# ================================================================
@pytest.mark.parametrize("make_eos", EOS)
def test_stage_writes_the_eos_buoyancy_at_the_cell_depth(make_eos):
    eos = make_eos()
    model = make_model(hy.TemperatureSalinity(eos))
    model.set_fields(T=temperature, S=salinity)
    depth = -nodes(model, "z")
    assert depth.min() > 0.0
    want = eos.buoyancy(field(model, "T"), field(model, "S"), depth,
                        gravity=G)
    # the polynomial EOS form the anomaly as a difference of two
    # ~1000 kg/m^3 densities, so two evaluation orders agree to
    # ulp(1000) / |anomaly| ~ 1e-13 / 1e-3: rtol 1e-10 is that floor
    np.testing.assert_allclose(
        np.asarray(model.diagnostics.b_total().data), np.asarray(want),
        rtol=1e-10, atol=1e-17)
    # the stored b is the value the last substage diagnosed: after one
    # step it is the EOS buoyancy of the state that entered the step
    entered = np.asarray(want)
    model.advance(1)
    np.testing.assert_allclose(field(model, "b"), entered,
                               rtol=1e-10, atol=1e-17)
    assert float(jnp.abs(model.state["p_hyd"].data).max()) > 0.0


def test_depth_is_measured_from_the_top_of_the_vertical_axis():
    # a column on z in [0, H] (surface at z = H) sees the same depths
    # as one on [-H, 0]; surface= overrides the reference height
    eos = hy.TEOS10EOS()

    def b_total(bounds, **kwargs):
        grid = fr.spatial.Grid((
            IM(4, (0.0, LENGTH), periodic=True, name="x"),
            IM(4, (0.0, LENGTH), periodic=True, name="y"),
            IM(6, bounds, periodic=False, name="z")))
        model = make_model(hy.TemperatureSalinity(eos, **kwargs),
                           grid=grid)
        model.set_fields(T=lambda x, y, z: 3.0 + 0.0 * (x + y + z),
                         S=lambda x, y, z: 35.0 + 0.0 * (x + y + z))
        return np.asarray(model.diagnostics.b_total().data)

    below = b_total((-DEPTH, 0.0))
    above = b_total((0.0, DEPTH))
    shifted = b_total((0.0, DEPTH), surface=DEPTH + 500.0)
    np.testing.assert_allclose(above, below, rtol=1e-12)
    assert not np.allclose(shifted, below, rtol=1e-3)
    # thermobaricity: the cold parcel's anomaly grows with depth
    assert abs(below[0, 0, 0]) > abs(below[0, 0, -1])


def test_terrain_column_reads_the_physical_depth():
    def bottom(x, y):
        return DEPTH * (1.0 + 0.2 * jnp.sin(2 * jnp.pi * x / LENGTH)
                        * jnp.cos(2 * jnp.pi * y / LENGTH))

    grid = fr.spatial.Grid(
        (IM(8, (0.0, LENGTH), periodic=True, name="x"),
         IM(8, (0.0, LENGTH), periodic=True, name="y"),
         IM(6, (-1.0, 0.0), periodic=False, name="z")),
        mapping=CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                  params={"H": bottom}))
    eos = hy.TEOS10EOS()
    model = make_model(hy.TemperatureSalinity(eos), grid=grid)
    model.set_fields(T=lambda x, y, z: 3.0 + 0.0 * (x + y + z),
                     S=lambda x, y, z: 35.0 + 0.0 * (x + y + z))
    depth = -nodes(model, "z") * np.asarray(
        bottom(nodes(model, "x"), nodes(model, "y")))
    assert depth.max() > 1.05 * DEPTH   # deeper than the flat column
    want = eos.buoyancy(3.0, 35.0, depth, gravity=G)
    np.testing.assert_allclose(
        np.asarray(model.diagnostics.b_total().data), np.asarray(want),
        rtol=1e-12)
    model.advance(2)
    assert not model.panicked


def test_horizontal_chart_reads_the_depth_from_the_vertical_nodes():
    # a chart that couples only the horizontal coordinates carries no
    # vertical column: the depth comes from the plain z nodes (and the
    # raising terrain.discover_column is never consulted)
    chart = fr.spatial.Grid(
        (IM(4, (0.0, 1.0), periodic=True, name="x"),
         IM(4, (0.0, 1.0), periodic=True, name="y"),
         IM(4, (-DEPTH, 0.0), periodic=False, name="z")),
        mapping=CoordinateMapping(
            chart={"X": lambda x, y: (x + 0.4 * y, y, 0.0 * x)}))
    assert _column(chart, "z") is None
    assert _column(make_grid(), "z") is None
    terrain = fr.spatial.Grid(
        (IM(4, (0.0, 1.0), periodic=True, name="x"),
         IM(4, (0.0, 1.0), periodic=True, name="y"),
         IM(4, (-1.0, 0.0), periodic=False, name="z")),
        mapping=CoordinateMapping(
            maps={"zp": lambda z, H: z * H},
            params={"H": lambda x, y: 1.0 + 0.0 * (x + y)}))
    assert _column(terrain, "z") == ("zp", "z")
    assert _column(terrain, "x") is None


# ================================================================
#  Gate: the hy.BuoyancyTracer trajectory, to rounding
# ================================================================
def _tracer_reference(steps):
    alpha = hy.LinearEOS().tunable["alpha"]
    model = make_model(hy.BuoyancyTracer())
    model.set_fields(
        b=lambda x, y, z: G * alpha * (temperature(x, y, z) - 10.0))
    model.advance(steps)
    return model


@pytest.mark.parametrize(
    "make", [
        pytest.param(lambda: hy.TemperatureSalinity(hy.LinearEOS()),
                     id="uniform-S"),
        pytest.param(lambda: hy.TemperatureSalinity(
            hy.LinearEOS(), constant_salinity=35.0), id="constant-S"),
    ])
def test_linear_eos_constant_salinity_is_the_buoyancy_tracer(make):
    steps = 20
    reference = _tracer_reference(steps)
    model = make_model(make())
    model.set_fields(T=temperature)
    if "S" in model.state:
        model.set_fields(S=lambda x, y, z: 35.0 + 0.0 * (x + y + z))
    model.advance(steps)
    alpha = hy.LinearEOS().tunable["alpha"]
    for name in ("u", "v", "w", "ps", "p_hyd"):
        want = field(reference, name)
        scale = np.abs(want).max()
        assert scale > 0.0
        np.testing.assert_allclose(
            field(model, name), want, rtol=0.0, atol=1e-11 * scale,
            err_msg=name)
    # the transported temperature IS the transported buoyancy
    b_of_t = G * alpha * (field(model, "T") - 10.0)
    want = field(reference, "b")
    np.testing.assert_allclose(b_of_t, want, rtol=0.0,
                               atol=1e-11 * np.abs(want).max())


# ================================================================
#  Gate: tracer content conservation
# ================================================================
@pytest.mark.parametrize("make_eos", EOS)
def test_heat_and_salt_content_are_conserved(make_eos):
    # flux-form advection through a closed surface + no-flux vertical
    # mixing: the heat and salt content are conserved to rounding. (The
    # hydrostatic default surface closure trades this for constancy
    # preservation — next test.)
    mixing = hy.VerticalMixing(kv=1.0e-3, kb=1.0e-3)
    model = make_model(hy.TemperatureSalinity(make_eos()),
                       advection=CenteredAdvection(surface_flux=False),
                       extra=(mixing,), imex=True)
    model.set_fields(T=temperature, S=salinity)
    before = {n: field(model, n).sum() for n in ("T", "S")}
    start = {n: field(model, n).copy() for n in ("T", "S")}
    model.advance(20)
    for name in ("T", "S"):
        assert not np.allclose(field(model, name), start[name],
                               rtol=0.0, atol=1e-6)
        # uniform cells: the content is the plain sum
        assert field(model, name).sum() == pytest.approx(
            before[name], rel=1e-13)


def test_default_surface_closure_keeps_a_uniform_salinity_uniform():
    # the constancy-preserving surface flux (the hydrostatic default):
    # a uniform S = 35 stays 35 while T drives a flow with w(0) != 0,
    # so no spurious buoyancy is made out of the salinity offset
    model = make_model(hy.TemperatureSalinity(hy.TEOS10EOS()))
    model.set_fields(T=temperature,
                     S=lambda x, y, z: 35.0 + 0.0 * (x + y + z))
    model.advance(20)
    assert float(jnp.abs(model.state["w"].data[:, :, -1]).max()) > 0.0
    assert np.abs(field(model, "S") - 35.0).max() < 1e-11


# ================================================================
#  Gate: a horizontally uniform column stays at rest (flat grid)
# ================================================================
@pytest.mark.parametrize("make_eos", EOS)
def test_flat_rest_state_stays_at_rest(make_eos):
    # the module's part is EXACT: the diagnosed b (and with it p_hyd)
    # is bitwise uniform along x and y, so the pressure gradient it
    # feeds is an exact zero. The velocities then sit on the rounding
    # floor of the core itself — hy.BuoyancyTracer at rest measures
    # the same 1e-17 m/s after 20 steps.
    model = make_model(hy.TemperatureSalinity(make_eos()))
    model.set_fields(
        T=lambda x, y, z: 4.0 + 14.0 * np.exp(z / 300.0) + 0.0 * (x + y),
        S=lambda x, y, z: 35.0 - 0.8 * np.exp(z / 200.0) + 0.0 * (x + y))
    model.advance(20)
    assert float(jnp.abs(model.state["b"].data).max()) > 1e-3
    for name in ("b", "p_hyd", "T", "S"):
        got = field(model, name)
        assert np.abs(got - got[:1, :1, :]).max() == 0.0, name
    for name in ("u", "v", "w"):
        assert float(jnp.abs(model.state[name].data).max()) < 1e-15, name


# ================================================================
#  Mixing, restoring, reductions
# ================================================================
def test_vertical_mixing_targets_the_tracers_not_the_buoyancy():
    mixing = hy.VerticalMixing(kb=1.0e-2)
    model = make_model(hy.TemperatureSalinity(hy.TEOS10EOS()),
                       extra=(mixing,), advection=None, imex=True)
    assert set(mixing._tracer_targets) == {"T", "S"}
    model.set_fields(
        T=lambda x, y, z: 4.0 + 14.0 * np.exp(z / 300.0) + 0.0 * (x + y),
        S=lambda x, y, z: 35.0 + 0.0 * (x + y + z))
    top = field(model, "T")[0, 0, -1]
    model.advance(10)
    assert field(model, "T")[0, 0, -1] < top   # the warm top cell cools


def test_explicit_vertical_mixing_runs_under_adams_bashforth():
    # the chart-grid route (no IMEX free surface there): an EXPLICIT
    # VerticalMixing under AB3, stable for kb dt / dz^2 << 1/2
    mixing = hy.VerticalMixing(kb=1.0e-2, treatment=fr.model.EXPLICIT)
    model = make_model(hy.TemperatureSalinity(hy.RoquetEOS()),
                       advection=CenteredAdvection(surface_flux=False),
                       extra=(mixing,))
    model.set_fields(T=temperature, S=salinity)
    before = field(model, "S").sum()
    model.advance(10)
    assert not model.panicked
    assert field(model, "S").sum() == pytest.approx(before, rel=1e-13)


def test_surface_restoring_by_a_masked_relaxation():
    # the documented SST-restoring spelling: a Relaxation masked to the
    # top cell, with a piston velocity / top-cell thickness rate
    dz = DEPTH / 6
    restoring = fr.model.modules.Relaxation(
        "T", rate=1.0 / 3600.0,
        target=lambda y: 20.0 + 5.0 * np.cos(2 * np.pi * y / LENGTH),
        mask=lambda z: (z > -dz).astype(float))
    model = make_model(
        hy.TemperatureSalinity(constant_salinity=35.0),
        advection=None, extra=(restoring,))
    assert "S" not in model.state
    model.set_fields(T=lambda x, y, z: 10.0 + 0.0 * (x + y + z))
    model.advance(20)
    got = field(model, "T")
    assert np.all(got[:, :, :-1] == 10.0)       # below: untouched
    assert np.all(got[:, :, -1] > 10.5)         # top cell: warmed
    assert got[:, 0, -1].mean() > got[:, 4, -1].mean()


def test_constant_temperature_reduction_carries_salinity_only():
    module = hy.TemperatureSalinity(hy.TEOS10EOS(),
                                    constant_temperature=4.0)
    assert module.tracers == ("S",)
    model = make_model(module)
    assert "T" not in model.state
    model.set_fields(S=salinity)
    want = hy.TEOS10EOS().buoyancy(4.0, field(model, "S"),
                                   -nodes(model, "z"), gravity=G)
    np.testing.assert_allclose(
        np.asarray(model.diagnostics.b_total().data), np.asarray(want),
        rtol=1e-10, atol=1e-17)
    model.advance(3)
    assert not model.panicked


def test_finite_volume_family_carries_cell_mean_tracers():
    model = make_model(hy.TemperatureSalinity(hy.RoquetEOS()),
                       advection=CenteredAdvection(surface_flux=False),
                       core=hy.Core(gravity=G, family="fv"))
    for name in ("T", "S", "b"):
        factors = model.state[name].function_space.bare.factors
        assert all(isinstance(f, CellAvg) for f in factors), name
    model.set_fields(T=temperature, S=salinity)
    before = field(model, "T").sum()
    model.advance(5)
    assert not model.panicked
    assert field(model, "T").sum() == pytest.approx(before, rel=1e-13)


# ================================================================
#  Immersed grid: the dry cells carry no buoyancy
# ================================================================
def test_immersed_dry_cells_carry_zero_buoyancy():
    def wet(x, y, z):  # noqa: ARG001
        return (z > -0.5 * DEPTH).astype(float)

    grid = make_grid(nz=8, immersed=ImmersedDomain(wet))
    model = make_model(hy.TemperatureSalinity(hy.TEOS10EOS()), grid=grid)
    model.set_fields(T=temperature, S=salinity)
    model.advance(3)
    assert not model.panicked
    dry = nodes(model, "z")[0, 0, :] < -0.5 * DEPTH
    assert dry.sum() == 4
    for name in ("T", "S", "b"):
        assert np.all(field(model, name)[:, :, dry] == 0.0), name
    assert np.abs(field(model, "b")[:, :, ~dry]).max() > 1e-3
    density = np.asarray(model.diagnostics.density().data)
    assert np.all(density[:, :, dry] == 0.0)
    assert density[:, :, ~dry].min() > 1020.0


# ================================================================
#  Bound diagnostics
# ================================================================
DERIVED = {
    "b_total": ("m/s^2", "Total buoyancy"),
    "density": ("kg/m^3", "In-situ density"),
    "potential_density": ("kg/m^3",
                          "Potential density (surface referenced)"),
}


@pytest.mark.parametrize("key", sorted(DERIVED))
def test_diagnostic_owns_its_annotation(key):
    model = make_model(hy.TemperatureSalinity(hy.TEOS10EOS()))
    quantity = getattr(model.diagnostics, key)()
    units, long_name = DERIVED[key]
    assert quantity.metadata.name == key
    assert quantity.xr.name == key
    assert quantity.metadata.long_name == long_name
    assert quantity.metadata.units == units
    assert quantity.metadata.physical_units == units


def test_density_diagnostics_are_the_eos_densities():
    eos = hy.TEOS10EOS()
    model = make_model(hy.TemperatureSalinity(eos))
    model.set_fields(T=temperature, S=salinity)
    t, s = field(model, "T"), field(model, "S")
    in_situ = np.asarray(model.diagnostics.density().data)
    potential = np.asarray(model.diagnostics.potential_density().data)
    np.testing.assert_allclose(
        in_situ, np.asarray(eos.density(t, s, -nodes(model, "z"))),
        rtol=1e-14)
    np.testing.assert_allclose(
        potential, np.asarray(eos.density(t, s, 0.0)), rtol=1e-14)
    # compression: in situ exceeds potential density below the surface
    assert np.all(in_situ > potential)
    assert set(hy.TemperatureSalinity().diagnostics) == set(DERIVED)


# ================================================================
#  Taught errors
# ================================================================
def test_eos_must_be_an_equation_of_state():
    with pytest.raises(TypeError, match="equation-of-state object"):
        hy.TemperatureSalinity(eos="linear")


def test_both_constants_leave_no_tracer():
    with pytest.raises(TypeError, match="no tracer would be left"):
        hy.TemperatureSalinity(constant_salinity=35.0,
                               constant_temperature=10.0)


class _DeepTunable(hy.LinearEOS):

    """A (deliberately invalid) depth-dependent tunable EOS."""

    @property
    def uses_depth(self):
        """Claim a depth dependence."""
        return True


class _Exotic(hy.LinearEOS):

    """An EOS whose tunable name is outside the bound vocabulary."""

    @property
    def tunable(self):
        """Name a coefficient the module cannot bind."""
        return {"gamma": 1.0}


@pytest.mark.parametrize("eos", [_DeepTunable(), _Exotic()],
                         ids=["depth-dependent", "unknown-name"])
def test_tunable_vocabulary_is_closed(eos):
    assert isinstance(eos, EquationOfState)
    with pytest.raises(TypeError, match="only a depth-independent"):
        hy.TemperatureSalinity(eos)


def test_unknown_vertical_is_a_taught_error():
    with pytest.raises(ValueError, match="not a grid coordinate"):
        make_model(hy.TemperatureSalinity(vertical="depth"))


def test_linear_assembly_without_advection_is_rejected():
    # like hy.BuoyancyTracer: nothing advances T / S
    with pytest.raises(AssemblyError, match="coverage lint"):
        make_model(hy.TemperatureSalinity(), advection=None)


def test_nondimensional_assembly_is_refused():
    grid = fr.spatial.Grid((
        IM(8, (0.0, 1.0), periodic=True, name="x"),
        IM(8, (0.0, 1.0), periodic=True, name="y"),
        IM(4, (0.0, 0.5), periodic=False, name="z")))
    with pytest.raises(AssemblyError, match="MIXED scaling variants"):
        hy.Model(
            grid=grid, core=hy.Core(),
            scaling=fr.scaling.Rotational(L=2.0e3, U=0.5),
            coriolis=hy.FPlaneCoriolis(rossby_number=0.25),
            buoyancy=hy.TemperatureSalinity(),
            free_surface=hy.ExplicitFreeSurface(froude_number=0.25),
            advection=CenteredAdvection(),
            time_stepper=AdamBashforth(2.0 ** -7, order=3))


def test_surface_buoyancy_flux_refuses_the_diagnosed_buoyancy():
    # b is no longer prognostic: force T / S (BoundaryFlux, Relaxation)
    with pytest.raises((AssemblyError, ValueError), match="b"):
        make_model(hy.TemperatureSalinity(),
                   extra=(hy.SurfaceBuoyancyFlux(1e-8),))
