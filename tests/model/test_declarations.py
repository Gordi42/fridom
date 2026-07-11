"""Tests for the field-declaration surface (model/declarations.py)."""
import pytest

from fridom.model.declarations import (
    FieldDeclaration,
    FieldReference,
    Lifecycle,
)
from fridom.model.roles import ADVECTED, TRACER, Role, Velocity
from fridom.spatial.bc import BC
from fridom.spatial.fields.metadata import FieldMetadata
from fridom.spatial.space_patterns import (
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


def test_default_accepts_callables_and_defers_bound_check():
    class Owner:
        def make(self, grid, space):
            pass

    # a bound method is accepted at construction (accept-and-defer):
    # the owner-identity check runs at assembly, where the owning
    # module is known via from_declaration's owner_instance
    decl = FieldDeclaration("n2", space=Profile(),
                            lifecycle=Lifecycle.AUXILIARY,
                            default=Owner().make)
    assert callable(decl.default)
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
    # the given pattern plus the derived wall entry (C8)
    assert decl.space == Staggered("x",
                                   wall_bc={"x": BC.DIRICHLET})
    with pytest.raises(TypeError):
        FieldDeclaration.velocity("u", "x")  # space is mandatory


# ================================================================
#  The velocity wall derivation (C8, topology-driven walls)
# ================================================================
def test_velocity_derives_the_wall_dirichlet():
    decl = FieldDeclaration.velocity("w", "z", space=Staggered("z"))
    assert dict(decl.space.wall_bc) == {"z": BC.DIRICHLET}
    assert decl.space.bc == ()  # unconditional BCs untouched
    assert decl.space.tags == Staggered("z").tags


def test_velocity_wall_derivation_names_the_component_only():
    # the component axis gets the wall entry, other axes never do
    decl = FieldDeclaration.velocity(
        "u", "x", space=Staggered("x", "z"))
    assert dict(decl.space.wall_bc) == {"x": BC.DIRICHLET}


def test_velocity_respects_an_explicit_bc_pin():
    space = Staggered("z", bc={"z": BC.NEUMANN})
    decl = FieldDeclaration.velocity("w", "z", space=space)
    assert decl.space == space  # the declarer's choice wins


def test_velocity_respects_an_explicit_wall_pin():
    space = Staggered("z", wall_bc={"z": BC.NEUMANN})
    decl = FieldDeclaration.velocity("w", "z", space=space)
    assert decl.space == space


def test_velocity_space_rule_passes_through():
    rule = SpaceRule(lambda grid: grid.factors[0].center)
    decl = FieldDeclaration.velocity("w", "z", space=rule)
    assert decl.space is rule  # the escape hatch is untouched


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
