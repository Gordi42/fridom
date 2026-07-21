"""The fr.scaling assembly machinery: alias row, validation, guards."""
from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.framework.utils import dtype_real, jaxify
from fridom.model.assembly import ParameterBindingTable
from fridom.model.declarations import FieldDeclaration
from fridom.model.errors import (
    AssemblyError,
    ParameterCollisionError,
    TimeDependentLinearOperatorError,
)
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration, leaf
from fridom.model.params import SCALING_NONLINEARITY
from fridom.model.terms import term
from fridom.model.time_dependent import Ramp
from fridom.model.time_steppers.runge_kutta import (
    ExplicitRungeKutta,
    tableaus,
)
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated

N = 8
DT = 1e-3


# ================================================================
#  Synthetic scaled module family (the sw.Core shape, miniature)
# ================================================================
@partial(jaxify, dynamic=("froude_number", "gravity"))
class ToyCore(Module):

    """A dual-variant toy core owning the gravity_wave mechanism."""

    scaling_mechanism = "gravity_wave"
    nonlinearity_attr = "froude_number"

    field_declarations = (
        FieldDeclaration("u", space=Collocated()),)

    def __init__(self, *, gravity=None, froude_number=None):
        if (gravity is None) == (froude_number is None):
            raise TypeError("exactly one kwarg set")
        self.gravity = (None if gravity is None
                        else jnp.asarray(gravity, dtype=dtype_real()))
        self.froude_number = (
            None if froude_number is None else leaf(froude_number))

    @property
    def scaling_variant(self):
        return ("dimensional" if self.froude_number is None
                else "nondimensional")

    @property
    def parameter_declarations(self):
        if self.froude_number is None:
            return (ParameterDeclaration(
                "toy.gravity", attr="gravity", units="m/s^2"),)
        return (ParameterDeclaration(
            "toy.froude", attr="froude_number", units="1"),)

    @term(name="drift", advances=("u",))
    def drift(self, state, ctx):
        if self.froude_number is None:
            return {"u": state["u"] * 0.0}
        eps = ctx.params[SCALING_NONLINEARITY]
        return {"u": state["u"] * eps}


@partial(jaxify, dynamic=("rossby_number", "f0"))
class ToyRotation(Module):

    """A second scaled family (rotation), field-free."""

    scaling_mechanism = "rotation"
    nonlinearity_attr = "rossby_number"

    field_declarations = ()

    def __init__(self, *, f0=None, rossby_number=None):
        if (f0 is None) == (rossby_number is None):
            raise TypeError("exactly one kwarg set")
        self.f0 = (None if f0 is None
                   else jnp.asarray(f0, dtype=dtype_real()))
        self.rossby_number = (
            None if rossby_number is None
            else jnp.asarray(rossby_number, dtype=dtype_real()))

    @property
    def scaling_variant(self):
        return ("dimensional" if self.rossby_number is None
                else "nondimensional")


@partial(jaxify, dynamic=("value",))
class CanonicalProvider(Module):

    """A module (wrongly) providing the canonical name itself."""

    field_declarations = ()
    parameter_declarations = (
        ParameterDeclaration(SCALING_NONLINEARITY, attr="value",
                             units="1"),)

    def __init__(self, value=0.5):
        self.value = jnp.asarray(value, dtype=dtype_real())


@jaxify
class EpsilonInL(Module):

    """A module whose linear term declares epsilon in linear_params."""

    field_declarations = ()

    @term(name="lin", advances=("u",), linear=True,
          linear_params=(SCALING_NONLINEARITY,))
    def lin(self, state, ctx):  # noqa: ARG002
        return {"u": state["u"] * 0.0}


class FrozenLStepper:

    """Duck stepper that freezes L (the refusal fires at step 2)."""

    freezes_linear_operator = True

    def __init__(self, dt=DT):
        self.dt = dt

    def init(self, _template):
        return ()


def make_grid():
    return Grid((IntervalMesh(N, (0.0, 1.0), periodic=True,
                              name="x"),), device_ids=(0,))


def make_model(modules, scaling=None):
    return Model(
        grid=make_grid(), modules=modules,
        time_stepper=ExplicitRungeKutta(DT, tableau=tableaus.RK4),
        scaling=scaling)


# ================================================================
#  The alias row (two names, one leaf)
# ================================================================
def test_mechanism_scaling_injects_the_alias_row():
    model = make_model((ToyCore(froude_number=0.2),),
                       scaling=fr.scaling.GravityWave())
    assert SCALING_NONLINEARITY in model.parameters
    assert float(model.parameters[SCALING_NONLINEARITY]) == 0.2
    entry = model._binding_table[SCALING_NONLINEARITY]
    assert entry.slot == 0
    assert entry.attr == "froude_number"
    assert (str(SCALING_NONLINEARITY)
            in model._binding_table.alias_names)
    # the module's own name binds the SAME leaf
    assert float(model.parameters["toy.froude"]) == 0.2


def test_update_through_either_name_hits_the_one_leaf():
    model = make_model((ToyCore(froude_number=0.2),),
                       scaling=fr.scaling.GravityWave())
    model.update_parameters({SCALING_NONLINEARITY: 0.3})
    assert float(model.parameters["toy.froude"]) == 0.3
    model.update_parameters({"toy.froude": 0.4})
    assert float(model.parameters[SCALING_NONLINEARITY]) == 0.4


def test_advective_scaling_binds_a_constant_one():
    model = make_model((ToyCore(froude_number=0.2),),
                       scaling=fr.scaling.Advective())
    assert float(model.parameters[SCALING_NONLINEARITY]) == 1.0
    entry = model._binding_table[SCALING_NONLINEARITY]
    assert entry.slot is None
    # constant rows own no leaf: post-assembly writes are refused
    with pytest.raises(AssemblyError, match="identity-defaulted"):
        model.update_parameters({SCALING_NONLINEARITY: 0.5})


@pytest.mark.parametrize("scaling", [None, fr.scaling.Dimensional()],
                         ids=["none", "dimensional"])
def test_dimensional_and_none_bind_no_row(scaling):
    model = make_model((ToyCore(gravity=9.81),), scaling=scaling)
    assert SCALING_NONLINEARITY not in model.parameters
    assert model._binding_table.alias_names == frozenset()
    assert model.scaling is scaling


def test_the_epsilon_row_reaches_the_traced_step():
    # the term reads ctx.params[SCALING_NONLINEARITY]; a run works and
    # the value is the froude leaf (u' = eps u over one RK4 step)
    model = make_model((ToyCore(froude_number=0.5),),
                       scaling=fr.scaling.GravityWave())
    model.set_fields(u=np.ones(N))
    model.advance(1)
    got = float(np.asarray(model.state["u"].data).ravel()[0])
    eps = 0.5 * DT
    expected = 1 + eps + eps**2 / 2 + eps**3 / 6 + eps**4 / 24
    assert got == pytest.approx(expected, rel=1e-12)


# ================================================================
#  Validation (taught errors)
# ================================================================
def test_mixed_variants_never_assemble():
    with pytest.raises(AssemblyError, match="MIXED scaling variants"):
        make_model((ToyCore(froude_number=0.2), ToyRotation(f0=1.0)),
                   scaling=fr.scaling.GravityWave())


@pytest.mark.parametrize("scaling", [None, fr.scaling.Dimensional()],
                         ids=["none", "dimensional"])
def test_nondim_module_needs_a_nondim_policy(scaling):
    with pytest.raises(AssemblyError,
                       match="need a nondimensional scaling policy"):
        make_model((ToyCore(froude_number=0.2),), scaling=scaling)


def test_nondim_policy_over_dimensional_modules_refused():
    with pytest.raises(AssemblyError,
                       match="dimensional kwarg set"):
        make_model((ToyCore(gravity=9.81),),
                   scaling=fr.scaling.GravityWave())


def test_mechanism_without_an_owner_is_a_structural_refusal():
    with pytest.raises(AssemblyError, match="no assembled module"):
        make_model((ToyCore(froude_number=0.2),),
                   scaling=fr.scaling.Rotational())


def test_two_mechanism_owners_collide():
    class SecondCore(ToyCore):
        field_declarations = ()

    with pytest.raises(AssemblyError, match="two owners"):
        make_model((ToyCore(froude_number=0.2),
                    SecondCore(froude_number=0.3)),
                   scaling=fr.scaling.GravityWave())


def test_missing_nonlinearity_attr_trait_is_taught():
    class NoAttrCore(ToyCore):
        field_declarations = ()
        nonlinearity_attr = None

    with pytest.raises(AssemblyError, match="nonlinearity_attr"):
        make_model((NoAttrCore(froude_number=0.2),),
                   scaling=fr.scaling.GravityWave())


def test_module_providing_the_canonical_name_collides():
    with pytest.raises(ParameterCollisionError,
                       match="scaling alias row"):
        make_model((ToyCore(froude_number=0.2), CanonicalProvider()),
                   scaling=fr.scaling.GravityWave())


# ================================================================
#  variant / propagator carry the policy
# ================================================================
def test_variant_carries_the_scaling_policy():
    scaling = fr.scaling.GravityWave()
    model = make_model((ToyCore(froude_number=0.2),), scaling=scaling)
    child = model.variant(updates={"toy.froude": 0.3})
    assert child.scaling is scaling
    assert float(child.parameters[SCALING_NONLINEARITY]) == 0.3
    assert (str(SCALING_NONLINEARITY)
            in child._binding_table.alias_names)


def test_propagator_runs_on_the_aliased_row():
    model = make_model((ToyCore(froude_number=0.5),),
                       scaling=fr.scaling.GravityWave())
    model.set_fields(u=np.ones(N))
    run = model.propagator(wrt=("toy.froude",), steps=2)
    out = run((jnp.asarray(0.5),))
    assert np.isfinite(np.asarray(out.state["u"].data)).all()


# ================================================================
#  The table-resolved frozen-L refusal
# ================================================================
def test_frozen_l_refuses_a_ramped_leaf_behind_the_alias_row():
    # EpsilonInL declares epsilon in a linear term's linear_params;
    # the leaf lives on ToyCore (cross-module, via the alias row), so
    # only the table-resolved pass can see the Ramp
    with pytest.raises(TimeDependentLinearOperatorError):
        Model(
            grid=make_grid(),
            modules=(ToyCore(
                froude_number=Ramp(0.0, 0.5, period=1.0)),
                EpsilonInL()),
            time_stepper=FrozenLStepper(),
            scaling=fr.scaling.GravityWave())


def test_frozen_l_skips_unbound_linear_param_names():
    # dimensional assembly: no epsilon row, so the declared name is
    # unbound and the frozen-L stepper is accepted
    model = Model(
        grid=make_grid(),
        modules=(ToyCore(gravity=9.81), EpsilonInL()),
        time_stepper=FrozenLStepper(),
        scaling=fr.scaling.Dimensional())
    assert SCALING_NONLINEARITY not in model.parameters


# ================================================================
#  build(extra_rows=) low-level surface
# ================================================================
def test_build_without_extra_rows_records_no_aliases():
    table = ParameterBindingTable.build(
        (ToyCore(froude_number=0.2),), FrozenLStepper())
    assert table.alias_names == frozenset()
    token = table.fingerprint_token()
    assert all(row[0] != "<scaling aliases>" for row in token)


def test_alias_names_enter_the_fingerprint_token():
    model = make_model((ToyCore(froude_number=0.2),),
                       scaling=fr.scaling.GravityWave())
    token = model._binding_table.fingerprint_token()
    assert ("<scaling aliases>", "scaling.nonlinearity", "", "") \
        in token
