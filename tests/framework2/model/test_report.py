"""Tests for the printable assembly report (model/report.py).

Covers the eight-section structure, rendering, the targeted section
read, the run-start addendum seam, and the section content produced
by ``assemble()`` (fields table, parameters, dispatch, schedule,
halo/layout, host-writable listing, fingerprint digest).
"""
from functools import partial
from typing import ClassVar

import jax.numpy as jnp
import pytest

from fridom.framework.utils import jaxify
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.model.assembly import assemble
from fridom.model.declarations import (
    FieldDeclaration,
    Lifecycle,
)
from fridom.model.module import Module
from fridom.model.parameters import (
    ParameterDeclaration,
    ParameterReference,
)
from fridom.model.report import (
    RUN_START_PLACEHOLDER,
    SECTION_ORDER,
    AssemblyReport,
)
from fridom.model.space_patterns import (
    Collocated,
    Profile,
    Staggered,
)
from fridom.model.terms import Treatment, term


# ================================================================
#  Toy modules
# ================================================================
class Stepper:

    supported_treatments = frozenset({Treatment.EXPLICIT})

    def __init__(self, dt=1.0, order=2):
        self.dt = dt
        self.order = order


@partial(jaxify, dynamic=("nu",))
class Core(Module):

    def __init__(self, nu=1e-3):
        self.nu = jnp.asarray(nu)

    field_declarations = (
        FieldDeclaration("u", space=Staggered("x")),
        FieldDeclaration("b", space=Collocated()),
        FieldDeclaration("flux", space=Profile("x"),
                         lifecycle=Lifecycle.AUXILIARY,
                         host_writable=True),
    )
    parameter_declarations = (
        ParameterDeclaration("core.nu", attr="nu", units="1"),)
    parameter_references = (
        ParameterReference("scaling.rossby", default=1.0),)

    @term(advances=("u",))
    def pressure_force(self, state, _ctx):
        return {"u": state["b"].diff("x")}

    @term(advances=("b",))
    def restoring(self, state, _ctx):
        return {"b": state["u"].to(state["b"].function_space)
                * (-1.0)}

    def self_update(self, state, _ctx):
        return {"flux": state["flux"] * 0.5}


class Override(Module):

    dispatch: ClassVar = {
        ("diff", Collocated()): FiniteDifference(order=4)}


# ================================================================
#  Fixtures
# ================================================================
def make_grid():
    return Grid((IntervalMesh(8, (0.0, 1.0), periodic=True,
                              name="x"),))


@pytest.fixture
def artifacts():
    return assemble(grid=make_grid(), modules=(Core(),),
                    time_stepper=Stepper(), name="reported")


@pytest.fixture
def report(artifacts):
    return artifacts.report


# ================================================================
#  Structure
# ================================================================
def test_all_eight_sections_render_non_empty(report):
    assert len(SECTION_ORDER) == 8
    for name in SECTION_ORDER:
        assert report.section(name).strip()


def test_str_contains_every_section(report):
    text = str(report)
    for fragment in ("Assembly", "Fields", "Parameters", "Dispatch",
                     "Schedule", "Halo / layout", "Lint",
                     "Run start"):
        assert fragment in text


def test_unknown_section_raises_with_the_valid_names(report):
    with pytest.raises(KeyError, match="header"):
        report.section("nope")


def test_constructor_validates_the_section_keys():
    with pytest.raises(ValueError, match="missing"):
        AssemblyReport({"header": "x"})
    complete = dict.fromkeys(SECTION_ORDER, "x")
    with pytest.raises(ValueError, match="unknown"):
        AssemblyReport({**complete, "extra": "y"})


# ================================================================
#  Content
# ================================================================
def test_header_names_grid_stepper_modules_and_digest(
        artifacts, report):
    header = report.header
    assert "'reported'" in header
    assert "IntervalMesh" in header
    assert "Stepper" in header
    assert "Core" in header
    assert artifacts.fingerprint.digest in header


def test_fields_section_rows_name_pattern_space_lifecycle(report):
    fields = report.section("fields")
    assert "u:" in fields
    assert "SpacePattern" in fields          # the declared pattern
    assert "Right(x)" in fields              # the resolved space
    assert "PROGNOSTIC" in fields
    assert "AUXILIARY" in fields
    assert "Core" in fields                  # the owner


def test_fields_section_lists_host_writable_components(report):
    assert "host-writable: flux" in report.section("fields")


def test_parameters_section_lists_bindings_and_identity_defaults(
        report):
    parameters = report.section("parameters")
    assert "core.nu" in parameters
    assert "stepper.dt" in parameters
    assert "the time stepper" in parameters
    # the untouched Param-style reference binds an identity default
    assert "identity defaults in effect: scaling.rossby" \
        in parameters


def test_dispatch_section_without_overrides(report):
    assert "no module dispatch overrides" \
        in report.section("dispatch")


def test_dispatch_section_lists_merged_overrides():
    arts = assemble(grid=make_grid(),
                    modules=(Core(), Override()),
                    time_stepper=Stepper())
    dispatch = arts.report.section("dispatch")
    assert "Override" in dispatch
    assert "diff" in dispatch


def test_schedule_section_is_kind_ordered(report):
    schedule = report.section("schedule")
    assert "Core/pressure_force" in schedule
    assert "Core/self_update" in schedule
    assert schedule.index("SELF_UPDATE") < schedule.index("TERM")


def test_halo_section_reports_widths_and_layout(report):
    halo = report.section("halo")
    assert "'x'" in halo
    assert "layout" in halo


def test_lint_section_defaults_to_none(report):
    assert report.section("lint") == "none"


# ================================================================
#  The run-start addendum seam (wave 4.2)
# ================================================================
def test_run_start_placeholder_then_addendum(report):
    assert report.section("run_start") == RUN_START_PLACEHOLDER
    report.append_run_start("u: user-initialized")
    assert report.section("run_start") == "u: user-initialized"
    report.append_run_start("b: defaults")
    assert report.section("run_start") == (
        "u: user-initialized\nb: defaults")
