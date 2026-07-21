"""The polarized wave maker: dispersion, packet forcing, errors.

On a zero state with advection off, the (unconstrained) tendency IS
the wave-maker source, so the term is checked exactly against the
materialized packet fields; the frequency is checked against the
analytic eigenmodes of the same model.
"""
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.errors import (
    MissingParameterError,
    TimeDependentParameterError,
)
from fridom.model.params import CORIOLIS_F0, STRATIFICATION_N2
from fridom.model.scheduled_field import ProfileFunction
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.polarized_wave_maker import (
    _COMPONENTS,
    PolarizedWaveMaker,
)
from fridom.nonhydro2.params import ASPECT_RATIO
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 8
LENGTH = 2 * np.pi
DT = 1e-3
K = {"x": 2, "y": 0, "z": 1}
AMP = 0.5


def make_grid(walled=()):
    return Grid(tuple(
        IntervalMesh(N, (0.0, LENGTH), periodic=name not in walled,
                     name=name)
        for name in ("x", "y", "z")), device_ids=(0,))


def make_maker(**kwargs):
    defaults = {"k": K, "position": {"x": np.pi},
                "width": {"x": 1.0}, "amplitude": AMP}
    defaults.update(kwargs)
    return PolarizedWaveMaker(**defaults)


def make_model(maker, walled=()):
    # explicit f-plane: the wave-maker's packet polarization needs the
    # constant coriolis.f0 provide, which no longer comes for free
    return nh.Model(
        grid=make_grid(walled),
        core=nh.Core(),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=1.0),
        advection=False,
        modules_extra=(maker,))


# ================================================================
#  Construction-time validation
# ================================================================
def test_geostrophic_branch_is_rejected():
    with pytest.raises(ValueError, match="must be \\+1 or -1"):
        make_maker(branch=0)


def test_position_and_width_must_share_keys():
    with pytest.raises(ValueError, match="same"):
        make_maker(position={"x": 1.0}, width={"z": 1.0})


def test_frequency_is_none_before_bind():
    assert make_maker().frequency is None


# ================================================================
#  Assembly-time validation (taught errors)
# ================================================================
def test_walled_grids_are_rejected_at_bind():
    with pytest.raises(ValueError, match="fully periodic"):
        make_model(make_maker(), walled=("z",))


def test_carrier_index_must_key_every_coordinate():
    with pytest.raises(ValueError, match="one integer mode index"):
        make_model(make_maker(k={"x": 2}))


def test_unknown_envelope_coordinate_is_rejected_at_bind():
    maker = make_maker(position={"q": 1.0}, width={"q": 1.0})
    with pytest.raises(ValueError, match="grid does not have"):
        make_model(maker)


# ================================================================
#  TDF-D6: the frozen packet refuses time-dependent inputs
# ================================================================
class _FakeRecord:

    """A duck-typed field record carrying only the TDF-D3 marker."""

    def __init__(self, *, time_dependent: bool) -> None:
        self.time_dependent = time_dependent


class _FakeTable:

    """A duck-typed bind table (the seam PolarizedWaveMaker.bind reads).

    The ``_BindTable`` contract docstring sanctions tests duck-typing
    the table; the refusal path only touches ``grid``, ``parameters``
    and the ``time_dependent`` marker of referenced field records.
    """

    def __init__(self, grid, parameters, td_fields=()):
        self.grid = grid
        self.parameters = parameters
        self._td = set(td_fields)

    def __getitem__(self, name):
        return _FakeRecord(time_dependent=name in self._td)


@pytest.mark.parametrize(
    "pname",
    [CORIOLIS_F0, STRATIFICATION_N2, ASPECT_RATIO],
    ids=["f0", "n2", "aspect_ratio"])
def test_profile_function_parameter_is_refused_at_bind(pname):
    maker = make_maker()
    params = {CORIOLIS_F0: 1.0, STRATIFICATION_N2: 1.0, ASPECT_RATIO: 1.0}
    params[pname] = ProfileFunction(lambda *args: args[0])
    table = _FakeTable(make_grid(), params)
    with pytest.raises(TimeDependentParameterError,
                       match="ProfileFunction"):
        maker.bind(table)


def test_time_dependent_dependency_field_is_refused_at_bind():
    maker = make_maker()
    params = {CORIOLIS_F0: 1.0, STRATIFICATION_N2: 1.0, ASPECT_RATIO: 1.0}
    # mark one packet-dependency field (u) as time_dependent
    table = _FakeTable(make_grid(), params, td_fields=("u",))
    with pytest.raises(TimeDependentParameterError,
                       match=r"'u'.*time_dependent"):
        maker.bind(table)


def test_meridional_law_n2_model_is_refused_before_bind():
    # A time_dependent n2 FIELD can only come from
    # MeridionalStratification (TDF-D7), which never provides the SCALAR
    # stratification.n2 the packet needs (provides-implies-constancy). So
    # the combination is refused by the missing-provide gate BEFORE bind
    # -- the packet's time_dependent-field refusal (checked over u/v/w/b,
    # its own dependencies) never sees the n2 field, and a scalar
    # ProfileFunction n2 parameter is covered above. The frozen packet
    # stays honest either way: it cannot follow a varying N^2.
    law = ProfileFunction(
        lambda y, t, a: 1.0 + a * t, params=(0.1,))  # noqa: ARG005
    maker = make_maker()
    with pytest.raises(
            MissingParameterError, match=r"stratification\.n2"):
        nh.Model(
            grid=make_grid(),
            core=nh.Core(),
            time_stepper=AdamBashforth(DT, order=3),
            coriolis=nh.FPlaneCoriolis(f0=1.0),
            stratification=nh.MeridionalStratification(n2=law),
            advection=False,
            modules_extra=(maker,))


# ================================================================
#  Dispersion and the packet
# ================================================================
def test_frequency_matches_the_analytic_eigenmodes():
    maker = make_maker()
    model = make_model(maker)
    modes = nh.eigenmodes.from_model(model)
    omega, _ = modes.mode("wave+", K)
    assert maker.frequency == pytest.approx(omega, rel=1e-12)


def test_forcing_vanishes_at_time_zero():
    model = make_model(make_maker())
    tendency = model.tendency(model.state, t=0.0, constraints=False)
    for name in _COMPONENTS:
        assert np.abs(np.asarray(tendency[name].data)).max() == 0.0


def test_forcing_is_the_packet_scaled_by_the_sinusoid():
    maker = make_maker()
    model = make_model(maker)
    omega = maker.frequency
    quarter = 0.25 * 2.0 * np.pi / omega  # sin(omega t) = 1
    tendency = model.tendency(model.state, t=quarter,
                              constraints=False)
    for name in _COMPONENTS:
        source = np.asarray(
            model.state[f"wavemaker_{name}"].data)
        np.testing.assert_allclose(
            np.asarray(tendency[name].data), AMP * source,
            atol=1e-14)
    # every component of the packet is excited
    assert all(
        np.abs(np.asarray(
            model.state[f"wavemaker_{name}"].data)).max() > 0.0
        for name in _COMPONENTS)


def test_forcing_oscillates_at_the_packet_frequency():
    maker = make_maker()
    model = make_model(maker)
    omega = maker.frequency
    t1, t2 = 0.2, 0.5
    f1 = np.asarray(model.tendency(
        model.state, t=t1, constraints=False)["u"].data)
    f2 = np.asarray(model.tendency(
        model.state, t=t2, constraints=False)["u"].data)
    ratio = np.sin(omega * t2) / np.sin(omega * t1)
    np.testing.assert_allclose(f2, ratio * f1, atol=1e-13)


def test_packet_is_localized_at_the_envelope_position():
    # the branch re-projection delocalizes the Gaussian a little
    # (it removes the vortical residue per wavevector), so the
    # check is a peak-position + amplitude-contrast statement
    maker = make_maker(position={"x": np.pi}, width={"x": 1.0})
    model = make_model(maker)
    space = model.state["u"].function_space
    x = np.asarray(model.grid.evaluation_nodes(space, "x").data)
    source = np.asarray(model.state["wavemaker_u"].data)
    envelope = np.abs(source).max(axis=(1, 2))
    peak = x.ravel()[np.argmax(envelope)]
    assert abs(peak - np.pi) < LENGTH / N
    inside = np.abs(source[np.abs(
        np.broadcast_to(x, source.shape) - np.pi) < 1.2]).max()
    outside = np.abs(source[np.abs(
        np.broadcast_to(x, source.shape) - np.pi) > 2.4]).max()
    assert inside > 4.0 * outside


# ================================================================
#  The provided amplitude (update without re-assembly)
# ================================================================
def test_update_parameters_scales_the_amplitude():
    maker = make_maker()
    model = make_model(maker)
    t = 0.25 * 2.0 * np.pi / maker.frequency
    before = np.asarray(model.tendency(
        model.state, t=t, constraints=False)["u"].data)
    model.update_parameters(
        {"wavemaker.polarized.amplitude": 2.0 * AMP})
    after = np.asarray(model.tendency(
        model.state, t=t, constraints=False)["u"].data)
    np.testing.assert_allclose(after, 2.0 * before, atol=1e-14)


def test_energy_enters_a_running_model():
    model = make_model(make_maker())
    model.advance(50)
    u = np.asarray(model.state["u"].data)
    assert np.isfinite(u).all()
    assert np.abs(u).max() > 0.0
