"""ClosureBase: role-target resolution, V-H2, per-field options."""
import pytest

import fridom as fr
from fridom.model.closures.base import ClosureBase
from fridom.model.declarations import (
    FieldDeclaration,
    Lifecycle,
)
from fridom.model.errors import (
    AssemblyError,
    ImmutableParameterError,
    MissingFieldError,
)
from fridom.model.field_table import (
    FieldRecord,
    FieldTable,
)
from fridom.model.roles import TRACER, Velocity
from fridom.model.terms import TendencyTerm
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import (
    Collocated,
    Staggered,
)


# ================================================================
#  A resolved table: two tracers, a velocity, a diagnosed velocity
# ================================================================
def make_table(*, immersed=None):
    grid = Grid((IntervalMesh(8, (0.0, 1.0), periodic=True,
                              name="x"),),
                immersed=immersed)
    declarations = (
        FieldDeclaration.velocity("u", "x", space=Staggered("x")),
        FieldDeclaration.tracer("b"),
        FieldDeclaration.tracer("c"),
        FieldDeclaration("p", space=Collocated(),
                         lifecycle=Lifecycle.DIAGNOSTIC),
        # V-H2: Velocity may sit on a DIAGNOSTIC field (the
        # hydrostatic diagnosed w) — never a closure write target
        FieldDeclaration("w", space=Collocated(),
                         lifecycle=Lifecycle.DIAGNOSTIC,
                         roles=(Velocity("z"),)),
    )
    records = [
        FieldRecord.from_declaration(decl, owner=0,
                                     owner_type="Core", grid=grid)
        for decl in declarations]
    return FieldTable(records, grid)


class Mixing(ClosureBase):

    """A mixing closure: TRACER-role default targets."""

    default_targets = TRACER


class Friction(ClosureBase):

    """A friction closure: Velocity-family default targets."""

    default_targets = Velocity


class NoDefault(ClosureBase):

    """A closure that forgot to declare default_targets."""


class WithOptions(Mixing):

    """A mixing closure exposing a per-field coefficient option."""

    def __init__(self, kappa, **kwargs):
        super().__init__(**kwargs)
        self._kappa = kappa

    @property
    def per_field_options(self):
        return {"kappa": self._kappa}


class Cooperative(Mixing):

    """Subclass bind reading the resolved targets cooperatively."""

    def bind(self, table):
        super().bind(table)
        self._frozen_targets = self.targets


# ================================================================
#  Role-driven defaults and the PROGNOSTIC intersection (V-H2)
# ================================================================
def test_tracer_default_targets_in_declaration_order():
    closure = Mixing()
    closure.bind(make_table())
    assert closure.targets == ("b", "c")


def test_velocity_family_targets_drop_diagnostic_members():
    closure = Friction()
    closure.bind(make_table())
    # the diagnosed "w" is a family member but not a write target
    assert closure.targets == ("u",)


def test_targets_are_empty_before_bind():
    assert Mixing().targets == ()


# ================================================================
#  fields= / exclude= overrides
# ================================================================
def test_fields_role_instance_override():
    closure = Mixing(fields=Velocity("x"))
    closure.bind(make_table())
    assert closure.targets == ("u",)


def test_fields_role_family_override():
    closure = Mixing(fields=Velocity)
    closure.bind(make_table())
    assert closure.targets == ("u",)


def test_fields_explicit_names_reordered_to_declaration_order():
    closure = Friction(fields=("c", "b"))
    closure.bind(make_table())
    assert closure.targets == ("b", "c")


def test_fields_single_string_is_one_name():
    closure = Mixing(fields="c")
    closure.bind(make_table())
    assert closure.targets == ("c",)


def test_exclude_removes_targets():
    closure = Mixing(exclude="b")
    closure.bind(make_table())
    assert closure.targets == ("c",)


def test_exclude_of_untargeted_declared_field_is_a_noop():
    closure = Mixing(exclude=("p",))
    closure.bind(make_table())
    assert closure.targets == ("b", "c")


# ================================================================
#  Taught assembly errors
# ================================================================
def test_unknown_name_in_fields_is_taught():
    closure = Mixing(fields=("b", "q"))
    with pytest.raises(MissingFieldError,
                       match=r"unknown field 'q' in fields="):
        closure.bind(make_table())


def test_unknown_name_in_exclude_is_taught():
    closure = Mixing(exclude=("q",))
    with pytest.raises(MissingFieldError,
                       match=r"unknown field 'q' in exclude="):
        closure.bind(make_table())


def test_explicit_diagnostic_name_is_rejected():
    closure = Mixing(fields=("p",))
    with pytest.raises(AssemblyError,
                       match="PROGNOSTIC fields only"):
        closure.bind(make_table())


def test_zero_resolved_targets_is_an_assembly_error():
    closure = Mixing(exclude=("b", "c"))
    with pytest.raises(AssemblyError, match="zero target"):
        closure.bind(make_table())


def test_bind_on_an_immersed_grid_is_taught():
    # per-closure capability (CL-D1): a closure that does not opt into
    # the fraction-weighted immersed spelling (_supports_immersed left
    # at the base default False) rejects an immersed grid at bind with a
    # taught error naming the plan and its §5 deferral list
    immersed = ImmersedDomain(lambda x: x * 0.0 + 1.0)
    closure = Mixing()
    assert closure._supports_immersed is False
    with pytest.raises(
            NotImplementedError,
            match="immersed_closures_sadourny_plan"):
        closure.bind(make_table(immersed=immersed))


def test_supports_immersed_closure_binds_on_an_immersed_grid():
    # a closure that opts in (CL-D1) passes the base capability gate and
    # resolves its targets normally on an immersed grid
    class ImmersedMixing(Mixing):
        _supports_immersed = True

    immersed = ImmersedDomain(lambda x: x * 0.0 + 1.0)
    closure = ImmersedMixing()
    closure.bind(make_table(immersed=immersed))
    assert closure.targets == ("b", "c")


def test_missing_default_targets_is_taught():
    closure = NoDefault()
    with pytest.raises(AssemblyError,
                       match="declares no default_targets"):
        closure.bind(make_table())


def test_allow_empty_targets_opt_out():
    class FrictionOnly(Mixing):
        _allow_empty_targets = True

    closure = FrictionOnly(exclude=("b", "c"))
    closure.bind(make_table())
    assert closure.targets == ()


# ================================================================
#  Constructor validation
# ================================================================
def test_empty_fields_tuple_selects_nothing():
    with pytest.raises(ValueError, match=r"fields=\(\) selects"):
        Mixing(fields=())


def test_fields_non_role_class_is_a_type_error():
    with pytest.raises(TypeError, match="Role family class"):
        Mixing(fields=int)


def test_fields_non_string_entries_are_a_type_error():
    with pytest.raises(TypeError, match="non-empty field names"):
        Mixing(fields=(1, 2))


def test_fields_non_iterable_is_a_type_error():
    with pytest.raises(TypeError, match="takes field names"):
        Mixing(fields=42)


def test_exclude_non_string_entries_are_a_type_error():
    with pytest.raises(TypeError, match="non-empty field names"):
        Mixing(exclude=(1,))


# ================================================================
#  Per-field option validation
# ================================================================
def test_per_field_mapping_covering_all_targets_binds():
    closure = WithOptions({"b": 1.0, "c": 2.0})
    closure.bind(make_table())
    assert closure.targets == ("b", "c")


def test_unknown_target_in_option_mapping_is_taught():
    closure = WithOptions({"b": 1.0, "x": 2.0})
    with pytest.raises(AssemblyError,
                       match=r"unknown target 'x' in kappa="):
        closure.bind(make_table())


def test_uncovered_target_in_option_mapping_is_taught():
    closure = WithOptions({"b": 1.0})
    with pytest.raises(AssemblyError,
                       match=r"no coefficient for target 'c'"):
        closure.bind(make_table())


def test_scalar_option_values_pass_unchecked():
    closure = WithOptions(1.0)
    closure.bind(make_table())
    assert closure.targets == ("b", "c")


def test_base_per_field_options_default_is_empty():
    assert Mixing().per_field_options == {}


# ================================================================
#  Bind-guard interplay (once/freeze; cooperative super())
# ================================================================
def test_bind_runs_once():
    closure = Mixing()
    closure.bind(make_table())
    with pytest.raises(ImmutableParameterError, match="already bound"):
        closure.bind(make_table())


def test_cooperative_subclass_bind_freezes_once():
    closure = Cooperative()
    closure.bind(make_table())
    assert closure._frozen_targets == ("b", "c")
    with pytest.raises(ImmutableParameterError):
        closure.something = 1


def test_failed_bind_leaves_the_closure_unbound():
    grid = Grid((IntervalMesh(8, (0.0, 1.0), periodic=True,
                              name="x"),))
    no_tracers = FieldTable((FieldRecord.from_declaration(
        FieldDeclaration.velocity("u", "x", space=Staggered("x")),
        owner=0, owner_type="Core", grid=grid),), grid)
    closure = Mixing()
    with pytest.raises(AssemblyError, match="zero target"):
        closure.bind(no_tracers)
    closure.bind(make_table())  # retry on a satisfying table works
    assert closure.targets == ("b", "c")


# ================================================================
#  The owned_by predicate follows free
# ================================================================
def test_owned_by_closurebase_matches_any_closure_subclass():
    predicate = fr.model.term_predicates.owned_by(
        fr.model.closures.ClosureBase)
    term = TendencyTerm(name="t", fn=lambda *_args: {})
    assert predicate("Mixing/t", term, Mixing()) is True
    assert predicate("Other/t", term, object()) is False


def test_fr_closures_namespace_exposes_the_base():
    assert fr.model.closures.ClosureBase is ClosureBase
