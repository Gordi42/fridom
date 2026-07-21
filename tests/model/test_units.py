"""Tests for the model.units factor machinery (model/units.py).

Covers the duck-typed ``unit_factors`` collection (collision check),
the framework t/T_ref rows, the centralized variant switch (identity
raw rows and curated/constant dimensional resolution), missing-scale
and unbound-parameter marks with the taught ``factor()`` errors, the
Ramp/at= path (D2.4), the Coriolis ``f_dim`` row on real assemblies,
the report, and the liveness of every read under
``update_parameters``.
"""
from functools import partial
from typing import ClassVar

import pytest

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.model.declarations import FieldDeclaration
from fridom.model.errors import AssemblyError
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.modules.coriolis import (
    BetaPlaneCoriolis,
    FPlaneCoriolis,
)
from fridom.model.parameters import ParameterDeclaration, leaf
from fridom.model.params import (
    CORIOLIS_F0,
    CORIOLIS_ROSSBY,
    ParamName,
)
from fridom.model.terms import term
from fridom.model.time_dependent import Ramp
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.model.units import FactorEntry, UnitFactor, UnitsView
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated, Staggered

N = 8
DT = 1e-3

AMP = ParamName("toy.amp", units="1",
                hint="provided by Provider(value=...)")
GHOST = ParamName("toy.ghost", units="1",
                  hint="provided by a GhostProvider module")


# ================================================================
#  Toy modules
# ================================================================
@jaxify
class Table(Module):

    """One field plus a raw component/coordinate factor table."""

    field_declarations = (
        FieldDeclaration("u", space=Collocated()),)

    unit_factors: ClassVar = {
        "u": UnitFactor(unit="m/s", expr="U", kind="component",
                        scales=("U",), fn=lambda v: v["U"]),
        "x": UnitFactor(unit="m", expr="L", kind="coordinate",
                        scales=("L",), fn=lambda v: v["L"]),
    }

    @term(advances=("u",))
    def decay(self, state, _ctx):
        return {"u": state["u"] * 0.0}


@jaxify
class CollidingTable(Module):

    """A second contributor of the ``u`` row (collision)."""

    field_declarations = ()
    unit_factors: ClassVar = {
        "u": UnitFactor(unit="m/s", expr="U", kind="component",
                        scales=("U",), fn=lambda v: v["U"]),
    }


@jaxify
class FrameworkCollider(Module):

    """A contributor colliding with the framework ``t`` row."""

    field_declarations = ()
    unit_factors: ClassVar = {
        "t": UnitFactor(unit="s", expr="L/U", kind="time",
                        scales=("L", "U"),
                        fn=lambda v: v["L"] / v["U"]),
    }


@partial(jaxify, dynamic=("value",))
class Provider(Module):

    """A parameter provider whose row references its own leaf."""

    field_declarations = ()
    parameter_declarations = (
        ParameterDeclaration(AMP, attr="value", units="1"),)

    unit_factors: ClassVar = {
        "amp": UnitFactor(
            unit="m", expr="a*L", kind="constant", scales=("L",),
            params={"a": AMP}, fn=lambda v: v["a"] * v["L"],
            dim_expr="a", dim_params={"a": AMP},
            dim_fn=lambda v: v["a"]),
    }

    def __init__(self, value=2.0):
        self.value = leaf(value)


@jaxify
class GhostRef(Module):

    """A row referencing a parameter nothing provides."""

    field_declarations = ()
    unit_factors: ClassVar = {
        "ghost": UnitFactor(
            unit="m", expr="q", kind="constant",
            params={"q": GHOST}, fn=lambda v: v["q"],
            dim_expr="q", dim_params={"q": GHOST},
            dim_fn=lambda v: v["q"]),
    }


@jaxify
class BareConstant(Module):

    """A constant row without a dimensional resolver (dim_fn None)."""

    field_declarations = ()
    unit_factors: ClassVar = {
        "bare": UnitFactor(unit="m", expr="L", kind="constant",
                           scales=("L",), fn=lambda v: v["L"]),
    }


@jaxify
class Velocities(Module):

    """The u/v vocabulary the Coriolis family references."""

    field_declarations = (
        FieldDeclaration("u", space=Staggered("x")),
        FieldDeclaration("v", space=Staggered("y")),
    )

    @term(advances=("u", "v"))
    def decay(self, state, _ctx):
        return {"u": state["u"] * 0.0, "v": state["v"] * 0.0}


# ================================================================
#  Fixtures and helpers
# ================================================================
def make_grid(dims=1):
    meshes = tuple(
        IntervalMesh(N, (0.0, 1.0), periodic=True, name=name)
        for name in ("x", "y")[:dims])
    return Grid(meshes)


def make_model(modules, scaling=None, dims=1):
    return Model(grid=make_grid(dims), modules=modules,
                 time_stepper=AdamBashforth(DT, order=2),
                 scaling=scaling)


def coriolis_model(coriolis, scaling):
    return make_model((Velocities(), coriolis), scaling=scaling,
                      dims=2)


# ================================================================
#  Collection
# ================================================================
def test_units_property_returns_a_view():
    model = make_model((Table(),))
    assert isinstance(model.units, UnitsView)


def test_framework_rows_are_always_present():
    model = make_model((Table(),))
    factors = model.units.factors
    assert "t" in factors
    assert "T_ref" in factors
    assert all(isinstance(entry, FactorEntry)
               for entry in factors.values())


def test_module_rows_are_collected():
    model = make_model((Table(),))
    factors = model.units.factors
    assert {"u", "x"} <= set(factors)


def test_row_collision_raises_assembly_error():
    model = make_model((Table(), CollidingTable()))
    with pytest.raises(AssemblyError, match="unit factor 'u'"):
        _ = model.units.factors


def test_collision_with_a_framework_row_raises():
    model = make_model((FrameworkCollider(),))
    with pytest.raises(AssemblyError, match="the framework"):
        _ = model.units.factors


def test_unknown_kind_is_refused_at_declaration():
    with pytest.raises(ValueError, match="unknown unit-factor kind"):
        UnitFactor(unit="m", expr="L", kind="banana")


# ================================================================
#  The nondimensional resolution (t, T_ref, module rows)
# ================================================================
def test_t_and_t_ref_follow_the_one_rule():
    # Advective: eps is the constant 1.0 row; T_ref = eps*L/U
    model = make_model((Table(),),
                       scaling=fr.scaling.Advective(L=2.0, U=0.5))
    assert model.units.factor("t") == pytest.approx(4.0)
    assert model.units.factor("T_ref") == pytest.approx(4.0)
    entry = model.units.factors["t"]
    assert entry.unit == "s"
    assert entry.kind == "time"
    assert entry.expr == "eps*L/U"


def test_component_and_coordinate_rows_resolve():
    model = make_model((Table(),),
                       scaling=fr.scaling.Advective(L=2.0, U=0.5))
    assert model.units.factor("u") == pytest.approx(0.5)
    assert model.units.factor("x") == pytest.approx(2.0)


# ================================================================
#  Identity mode (Dimensional / scaling=None)
# ================================================================
@pytest.mark.parametrize(
    "scaling", [None, fr.scaling.Dimensional()],
    ids=["none", "dimensional"])
def test_raw_rows_are_identity_on_dimensional_models(scaling):
    model = make_model((Table(),), scaling=scaling)
    for name, unit in (("u", "m/s"), ("x", "m"), ("t", "s")):
        entry = model.units.factors[name]
        assert entry.value == 1.0
        assert entry.unit == unit
        assert entry.expr == "1"
        assert model.units.factor(name) == 1.0


def test_t_ref_is_identity_on_a_dimensional_model():
    model = make_model((Table(),), scaling=fr.scaling.Dimensional())
    assert model.units.factor("T_ref") == 1.0


def test_dimensional_constant_reports_the_bound_value():
    model = make_model((Provider(value=3.5),),
                       scaling=fr.scaling.Dimensional())
    assert model.units.factor("amp") == pytest.approx(3.5)
    assert model.units.factors["amp"].expr == "a"


def test_constant_without_dim_fn_falls_back_to_identity():
    model = make_model((BareConstant(),),
                       scaling=fr.scaling.Dimensional())
    entry = model.units.factors["bare"]
    assert entry.value == 1.0
    assert entry.expr == "1"


# ================================================================
#  Missing scales / unbound parameters (only factor() raises)
# ================================================================
def test_missing_scales_mark_the_entry():
    model = make_model((Table(),), scaling=fr.scaling.Advective())
    entry = model.units.factors["t"]
    assert entry.value is None
    assert entry.missing == ("L=", "U=")
    assert not entry.time_dependent


def test_factor_teaches_the_missing_scale():
    model = make_model((Table(),),
                       scaling=fr.scaling.Advective(L=2.0))
    with pytest.raises(ValueError, match=r"pass U=<value>"):
        model.units.factor("u")


def test_unbound_parameter_marks_and_teaches():
    model = make_model((GhostRef(),),
                       scaling=fr.scaling.Advective(L=1.0, U=1.0))
    entry = model.units.factors["ghost"]
    assert entry.value is None
    assert entry.missing == ("toy.ghost",)
    with pytest.raises(ValueError, match="GhostProvider"):
        model.units.factor("ghost")


def test_unknown_name_lists_the_available_rows():
    model = make_model((Table(),))
    with pytest.raises(ValueError, match=r"available factors:.*T_ref"):
        model.units.factor("nope")


# ================================================================
#  Ramp-valued inputs (D2.4: explicit at=)
# ================================================================
def test_ramp_marks_time_dependent_and_teaches_at():
    ramp = Ramp(1.0, 3.0, period=10.0)
    model = make_model(
        (Provider(value=ramp),),
        scaling=fr.scaling.Advective(L=2.0, U=1.0))
    entry = model.units.factors["amp"]
    assert entry.value is None
    assert entry.time_dependent
    with pytest.raises(ValueError, match="at="):
        model.units.factor("amp")


def test_ramp_resolves_with_at():
    ramp = Ramp(1.0, 3.0, period=10.0)
    model = make_model(
        (Provider(value=ramp),),
        scaling=fr.scaling.Advective(L=2.0, U=1.0))
    assert model.units.factor("amp", at=0.0) == pytest.approx(2.0)
    assert model.units.factor("amp", at=10.0) == pytest.approx(6.0)


# ================================================================
#  The Coriolis f_dim row (real assemblies, both variants)
# ================================================================
def test_f_dim_on_a_nondimensional_f_plane():
    model = coriolis_model(
        FPlaneCoriolis(rossby_number=0.5),
        fr.scaling.Rotational(L=2.0, U=1.0))
    assert model.units.factor("f_dim") == pytest.approx(1.0)
    # Rotational aliases eps onto the Rossby leaf: t = Ro*L/U
    assert model.units.factor("t") == pytest.approx(1.0)


def test_f_dim_on_a_dimensional_f_plane_is_the_bound_f0():
    model = coriolis_model(FPlaneCoriolis(f0=1e-4),
                           fr.scaling.Dimensional())
    assert model.units.factor("f_dim") == pytest.approx(1e-4)
    assert model.units.factors["f_dim"].expr == "f0"


def test_f_dim_on_a_dimensional_beta_plane_is_unresolvable():
    # the beta-plane provides no coriolis.f0 (f is the f(y) field)
    model = coriolis_model(BetaPlaneCoriolis(f0=1e-4, beta=1e-11),
                           fr.scaling.Dimensional())
    entry = model.units.factors["f_dim"]
    assert entry.value is None
    assert entry.missing == (str(CORIOLIS_F0),)


# ================================================================
#  Liveness (sweeps/ramps never go stale)
# ================================================================
def test_factors_read_live_after_update_parameters():
    model = coriolis_model(
        FPlaneCoriolis(rossby_number=0.5),
        fr.scaling.Rotational(L=2.0, U=1.0))
    assert model.units.factor("f_dim") == pytest.approx(1.0)
    model.update_parameters({CORIOLIS_ROSSBY: 0.25})
    assert model.units.factor("f_dim") == pytest.approx(2.0)


# ================================================================
#  bound_constants (the writer's parameter stamp source)
# ================================================================
def test_bound_constants_nondimensional():
    model = coriolis_model(
        FPlaneCoriolis(rossby_number=0.5),
        fr.scaling.Rotational(L=2.0, U=1.0))
    constants = model.units.bound_constants()
    assert constants["scaling.nonlinearity"] == pytest.approx(0.5)
    assert constants["coriolis.rossby"] == pytest.approx(0.5)


def test_bound_constants_dimensional():
    model = coriolis_model(FPlaneCoriolis(f0=1e-4),
                           fr.scaling.Dimensional())
    assert model.units.bound_constants() == {
        "coriolis.f0": pytest.approx(1e-4)}


def test_bound_constants_skips_ramps_and_unbound():
    ramp = Ramp(1.0, 3.0, period=10.0)
    model = make_model(
        (Provider(value=ramp), GhostRef()),
        scaling=fr.scaling.Advective(L=2.0, U=1.0))
    constants = model.units.bound_constants()
    assert "toy.amp" not in constants
    assert "toy.ghost" not in constants
    assert constants["scaling.nonlinearity"] == pytest.approx(1.0)


# ================================================================
#  report / repr (never raise)
# ================================================================
def test_report_smoke_resolved():
    model = make_model((Table(),),
                       scaling=fr.scaling.Advective(L=2.0, U=0.5))
    report = model.units.report()
    assert "unit factors (Advective, nondimensional)" in report
    assert "L = 2" in report
    assert "g = unset" in report
    assert "[eps*L/U]" in report


def test_report_marks_missing_scales():
    model = make_model((Table(),),
                       scaling=fr.scaling.Advective(U=1.0))
    assert "-- needs L=" in model.units.report()


def test_report_marks_time_dependent_and_takes_at():
    ramp = Ramp(1.0, 3.0, period=10.0)
    model = make_model((Table(), Provider(value=ramp)),
                       scaling=fr.scaling.Advective(L=2.0, U=1.0))
    assert "-- time-dependent (pass at=)" in model.units.report()
    # with at= the ramp row resolves in the same report
    assert "time-dependent" not in model.units.report(at=0.0)


def test_report_smoke_dimensional_and_none():
    dim = make_model((Table(),), scaling=fr.scaling.Dimensional())
    assert "Dimensional, dimensional" in dim.units.report()
    none = make_model((Table(),))
    report = none.units.report()
    assert "no scaling" in report
    assert "scales:" not in report


def test_repr_lists_the_rows():
    model = make_model((Table(),))
    assert repr(model.units) == "<units: T_ref, t, u, x>"
