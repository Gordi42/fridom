"""Tests for the field-declaration surface (model/declarations.py)."""
import pytest

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.fields.metadata import FieldMetadata
from fridom.framework2.model.declarations import (
    FieldDeclaration,
    FieldReference,
    Lifecycle,
)
from fridom.framework2.model.roles import ADVECTED, TRACER, Role, Velocity
from fridom.framework2.model.space_patterns import (
    Collocated,
    Profile,
    SpaceRule,
    Staggered,
)


# ================================================================
#  Lifecycle
# ================================================================
def test_lifecycle_is_closed():
    assert list(Lifecycle) == [Lifecycle.PROGNOSTIC,
                               Lifecycle.AUXILIARY,
                               Lifecycle.DIAGNOSTIC]


# ================================================================
#  FieldDeclaration construction
# ================================================================
def test_basic_construction_and_attributes():
    decl = FieldDeclaration(
        "b", space=Collocated(), roles=(TRACER, ADVECTED),
        default=1.5, long_name="Buoyancy", units="m/s²",
        nc_attrs={"standard_name": "buoyancy"})
    assert decl.name == "b"
    assert decl.space == Collocated()
    assert decl.lifecycle is Lifecycle.PROGNOSTIC
    assert decl.roles == frozenset({TRACER, ADVECTED})
    assert decl.default == 1.5
    assert decl.host_writable is False
    assert decl.long_name == "Buoyancy"
    assert decl.units == "m/s²"
    assert decl.nc_attrs == (("standard_name", "buoyancy"),)


def test_names_are_dot_free():
    with pytest.raises(ValueError, match="dot"):
        FieldDeclaration("mybgc.no3", space=Collocated())
    with pytest.raises(TypeError, match="non-empty"):
        FieldDeclaration("", space=Collocated())


def test_space_slot_accepts_pattern_or_rule():
    rule = SpaceRule(lambda grid: grid.factors[0].center)
    assert FieldDeclaration("q", space=rule).space is rule
    with pytest.raises(TypeError, match="SpacePattern"):
        FieldDeclaration("q", space="collocated")
    with pytest.raises(TypeError):
        FieldDeclaration("q")  # space is mandatory


def test_lifecycle_and_roles_validation():
    with pytest.raises(TypeError, match="Lifecycle"):
        FieldDeclaration("q", space=Collocated(),
                         lifecycle="prognostic")
    with pytest.raises(TypeError, match="Role"):
        FieldDeclaration("q", space=Collocated(),
                         roles=("advected",))
    # AUXILIARY fields carry no roles at all
    with pytest.raises(ValueError, match="AUXILIARY"):
        FieldDeclaration("n2", space=Profile("z"),
                         lifecycle=Lifecycle.AUXILIARY,
                         roles=(ADVECTED,))
    # DIAGNOSTIC admits only Velocity (V-H2)
    with pytest.raises(ValueError, match="Velocity"):
        FieldDeclaration("p", space=Collocated(),
                         lifecycle=Lifecycle.DIAGNOSTIC,
                         roles=(TRACER,))


def test_diagnosed_velocity_is_legal():
    # the hydrostatic diagnosed w (V-H2)
    decl = FieldDeclaration(
        "w", space=Staggered("z"),
        lifecycle=Lifecycle.DIAGNOSTIC, roles=(Velocity("z"),))
    assert decl.roles == frozenset({Velocity("z")})


def test_open_roles_stay_open():
    decl = FieldDeclaration("no3", space=Collocated(),
                            roles=(Role("mybgc.nutrient"),))
    assert Role("mybgc.nutrient") in decl.roles


def test_tracer_without_advected_warns():
    with pytest.warns(UserWarning, match="TRACER without ADVECTED"):
        FieldDeclaration("s", space=Collocated(), roles=(TRACER,))


def test_host_writable_gating():
    with pytest.raises(ValueError, match="host_writable"):
        FieldDeclaration("u", space=Collocated(),
                         host_writable=True)
    aux = FieldDeclaration("flux", space=Collocated(),
                           lifecycle=Lifecycle.AUXILIARY,
                           host_writable=True)
    assert aux.host_writable is True
    diag = FieldDeclaration("mean_b", space=Collocated(),
                            lifecycle=Lifecycle.DIAGNOSTIC,
                            host_writable=True)
    assert diag.host_writable is True


# ================================================================
#  default= forms and the self-disambiguation
# ================================================================
def test_default_forms():
    make = lambda default: FieldDeclaration(  # noqa: E731
        "n2", space=Profile("z"),
        lifecycle=Lifecycle.AUXILIARY, default=default)
    assert make(None).default_form == "zeros"
    assert make(2.5e-5).default_form == "constant"
    assert make(lambda z: z**2).default_form == "coordinate"
    owner = lambda self, grid, space: None  # noqa: E731, ARG005
    assert make(owner).default_form == "owner_method"


def test_default_rejects_bound_methods():
    class Owner:
        def make(self, grid, space):
            pass

    with pytest.raises(TypeError, match="UNBOUND"):
        FieldDeclaration("n2", space=Profile(),
                         lifecycle=Lifecycle.AUXILIARY,
                         default=Owner().make)
    # the unbound class attribute is the sanctioned spelling
    decl = FieldDeclaration("n2", space=Profile(),
                            lifecycle=Lifecycle.AUXILIARY,
                            default=Owner.make)
    assert decl.default_form == "owner_method"


def test_default_rejects_other_types():
    with pytest.raises(TypeError, match="default"):
        FieldDeclaration("q", space=Collocated(), default="zeros")


# ================================================================
#  Templates
# ================================================================
def test_tracer_template():
    decl = FieldDeclaration.tracer("dye", units="1")
    assert decl.lifecycle is Lifecycle.PROGNOSTIC
    assert decl.roles == frozenset({TRACER, ADVECTED})
    assert decl.space == Collocated()
    assert decl.units == "1"


def test_tracer_template_takes_a_space():
    space = Collocated(bc={"z": BC.DIRICHLET})
    assert FieldDeclaration.tracer("b", space=space).space == space


def test_velocity_template():
    decl = FieldDeclaration.velocity("u", "x", space=Staggered("x"))
    assert decl.lifecycle is Lifecycle.PROGNOSTIC
    assert decl.roles == frozenset({Velocity("x"), ADVECTED})
    assert decl.space == Staggered("x")
    with pytest.raises(TypeError):
        FieldDeclaration.velocity("u", "x")  # space is mandatory


# ================================================================
#  replace / field_metadata / repr
# ================================================================
def test_replace_is_functional():
    decl = FieldDeclaration.tracer("b", units="m/s²")
    tweaked = decl.replace(units="1", default=1.0)
    assert tweaked.units == "1"
    assert tweaked.default == 1.0
    assert tweaked.roles == decl.roles
    assert decl.units == "m/s²"  # original unchanged


def test_replace_revalidates():
    decl = FieldDeclaration.tracer("b")
    with pytest.raises(ValueError, match="dot"):
        decl.replace(name="my.b")
    with pytest.raises(TypeError, match="unexpected"):
        decl.replace(topo=(True, True, False))


def test_field_metadata_folding():
    decl = FieldDeclaration(
        "b", space=Collocated(), long_name="Buoyancy",
        units="m/s²", nc_attrs={"positive": "up"})
    assert decl.field_metadata() == FieldMetadata(
        name="b", long_name="Buoyancy", units="m/s²",
        nc_attrs=(("positive", "up"),))


def test_repr_names_the_essentials():
    decl = FieldDeclaration.tracer("dye")
    rendered = repr(decl)
    assert "'dye'" in rendered
    assert "Lifecycle.PROGNOSTIC" in rendered
    assert "fridom.tracer" in rendered


# ================================================================
#  FieldReference
# ================================================================
def test_field_reference_shape():
    ref = FieldReference("w", hint="declared by a dynamical core")
    assert ref.name == "w"
    assert ref.hint == "declared by a dynamical core"
    name, hint = ref  # NamedTuple unpacking
    assert (name, hint) == ("w", ref.hint)
    assert FieldReference("w").hint == ""
    assert FieldReference("w") == FieldReference("w")
