"""The shared Relaxation module: declarations, term, decay, errors.

The module is framework-level; the tests exercise it on nonhydro2
models (the test_coriolis precedent of borrowing a concrete port):
with n2 = 0 the buoyancy tendency reduces to the relaxation term
alone, so both the pointwise tendency and the integrated exponential
decay are exact checks.
"""
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.errors import MissingFieldError
from fridom.model.modules.relaxation import Relaxation
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 8
LENGTH = 2 * np.pi
DT = 1e-3


def make_grid():
    return Grid(tuple(
        IntervalMesh(N, (0.0, LENGTH), periodic=True, name=name)
        for name in ("x", "y", "z")))


def make_model(*modules, n2=0.0):
    """Linear nh model; f0 = n2 = 0 isolates the relaxation term."""
    return nh.Model(
        grid=make_grid(),
        core=nh.Core(),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=0.0),
        stratification=nh.ConstantStratification(n2=n2),
        advection=False,
        modules_extra=modules)


def cell_centers():
    ax = (np.arange(N) + 0.5) * (LENGTH / N)
    return np.meshgrid(ax, ax, ax, indexing="ij")


# ================================================================
#  Construction-time validation
# ================================================================
def test_field_names_normalize_and_validate():
    assert Relaxation("b").fields == ("b",)
    assert Relaxation(("u", "v")).fields == ("u", "v")
    with pytest.raises(ValueError, match="at least one field"):
        Relaxation(())
    with pytest.raises(TypeError, match="non-empty strings"):
        Relaxation(("u", 3))
    with pytest.raises(ValueError, match="unique"):
        Relaxation(("u", "u"))


def test_target_mapping_keys_every_field():
    with pytest.raises(ValueError, match="keys every relaxed field"):
        Relaxation(("u", "v"), target={"u": 0.0})
    with pytest.raises(TypeError, match="number or a profile"):
        Relaxation("b", target="warm")
    with pytest.raises(TypeError, match="mask= must be a profile"):
        Relaxation("b", mask=1.0)


def test_rate_parameter_is_named_by_the_relaxed_fields():
    assert Relaxation("b").rate_parameter == "relaxation.b.rate"
    assert (Relaxation(("u", "v")).rate_parameter
            == "relaxation.u_v.rate")


def test_declarations_cover_targets_and_mask():
    module = Relaxation(
        ("u", "b"), target={"u": 0.5, "b": np.cos},
        mask=lambda x: x)
    names = tuple(d.name for d in module.field_declarations)
    assert names == ("u_relax_target", "b_relax_target",
                     "u_b_relax_mask")
    refs = tuple(ref.name for ref in module.field_references)
    assert refs == ("u", "b")


# ================================================================
#  Assembly-time validation (taught errors)
# ================================================================
def test_unknown_field_name_is_a_taught_assembly_error():
    with pytest.raises(MissingFieldError,
                       match="Relaxation field-name spelling"):
        make_model(Relaxation("bouyancy"))


def test_non_prognostic_field_is_rejected_at_bind():
    with pytest.raises(ValueError, match="only PROGNOSTIC fields"):
        make_model(Relaxation("p"))


def test_unknown_profile_coordinate_is_rejected_at_bind():
    with pytest.raises(ValueError, match="target of 'b'"):
        make_model(Relaxation("b", target=lambda q: q))
    with pytest.raises(ValueError, match="Relaxation mask"):
        make_model(Relaxation("b", mask=lambda q: q))


# ================================================================
#  The term: pointwise tendency, targets sampled on profiles
# ================================================================
def test_tendency_is_rate_times_target_minus_field():
    model = make_model(
        Relaxation("b", rate=0.5, target=lambda z: 0.1 * np.cos(z)))
    _, _, z = cell_centers()
    model.set_fields(b=0.3 * np.sin(z))
    # the target profile lives on fr.Profile("z"): one z column,
    # sampled at the (collocated) z nodes
    target = np.asarray(model.state["b_relax_target"].data)
    z_axis = (np.arange(N) + 0.5) * (LENGTH / N)
    np.testing.assert_allclose(
        target, (0.1 * np.cos(z_axis)).reshape(target.shape),
        atol=1e-15)
    tendency = model.tendency(model.state)
    expected = 0.5 * (target - np.asarray(model.state["b"].data))
    np.testing.assert_allclose(
        np.asarray(tendency["b"].data), expected, atol=1e-15)


def test_mask_gates_where_the_relaxation_acts():
    # 0/1 mask (the v1 domain_function): zero tendency outside
    model = make_model(
        Relaxation("b", rate=2.0, target=0.4,
                   mask=lambda x: np.where(x < np.pi, 1.0, 0.0)))
    x, _, z = cell_centers()
    model.set_fields(b=0.3 * np.sin(z))
    tendency = np.asarray(
        model.tendency(model.state)["b"].data)
    b = np.asarray(model.state["b"].data)
    expected = np.where(x < np.pi, 2.0 * (0.4 - b), 0.0)
    np.testing.assert_allclose(tendency, expected, atol=1e-15)


def test_multi_field_relaxation_with_mapped_targets():
    module = Relaxation(("u", "v"), rate=3.0,
                        target={"u": 0.2, "v": 0.0})
    model = make_model(module)
    rng = np.random.default_rng(7)
    fields = {name: rng.standard_normal(
        np.asarray(model.state[name].data).shape)
        for name in ("u", "v")}
    model.set_fields(**fields)
    # constraints=False: the projection would remix u, v, w
    tendency = model.tendency(model.state, constraints=False)
    for name, target in (("u", 0.2), ("v", 0.0)):
        np.testing.assert_allclose(
            np.asarray(tendency[name].data),
            3.0 * (target - fields[name]), atol=1e-14)


# ================================================================
#  Exponential decay at the discrete rate
# ================================================================
def test_field_decays_exponentially_toward_the_target():
    rate, target = 5.0, 0.2
    model = make_model(Relaxation("b", rate=rate, target=target))
    _, _, z = cell_centers()
    b0 = 0.3 * np.sin(z)
    model.set_fields(b=b0)
    steps = 200
    model.advance(steps)
    t = steps * DT
    expected = target + (b0 - target) * np.exp(-rate * t)
    np.testing.assert_allclose(
        np.asarray(model.state["b"].data), expected, atol=2e-5)


# ================================================================
#  The provided rate parameter (update without re-assembly)
# ================================================================
def test_update_parameters_changes_the_rate():
    module = Relaxation("b", rate=0.5, target=0.0)
    model = make_model(module)
    _, _, z = cell_centers()
    model.set_fields(b=0.3 * np.sin(z))
    before = np.asarray(model.tendency(model.state)["b"].data)
    model.update_parameters({module.rate_parameter: 2.0})
    after = np.asarray(model.tendency(model.state)["b"].data)
    np.testing.assert_allclose(after, 4.0 * before, atol=1e-14)


def test_two_instances_coexist_with_distinct_rate_names():
    model = make_model(
        Relaxation("b", rate=1.0, target=0.1),
        Relaxation("u", rate=2.0))
    params = model.parameters
    assert "relaxation.b.rate" in params
    assert "relaxation.u.rate" in params
