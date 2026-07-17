"""Tests for the implicit families (model/implicit.py)."""
import dataclasses

import numpy as np
import pytest

from fridom.model.implicit import (
    ImplicitOperator,
    VerticalDiffusion,
    reject_unsupported_solve_column,
)
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh


class FullDuck:

    """A structural (non-subclassing) protocol implementer."""

    def __init__(self):
        self.fields = ("u", "v")

    def apply(self, _module, _state, _ctx):
        return {}

    def solve(self, _module, rhs, _dt_gamma, _ctx):
        return rhs

    def merge_key(self):
        return None

    def merged_with(self, _other):
        raise NotImplementedError


class MissingSolve:

    """Two-capability surface broken: no solve."""

    fields = ("u",)

    def apply(self, _module, _state, _ctx):
        return {}

    def merge_key(self):
        return None

    def merged_with(self, _other):
        raise NotImplementedError


class MissingFields:

    """No advanced-subset declaration."""

    def apply(self, _module, _state, _ctx):
        return {}

    def solve(self, _module, rhs, _dt_gamma, _ctx):
        return rhs

    def merge_key(self):
        return None

    def merged_with(self, _other):
        raise NotImplementedError


def kappa_2(_module, _state, _ctx, _field_name):
    return 2.0


def kappa_3(_module, _state, _ctx, _field_name):
    return 3.5


# ================================================================
#  ImplicitOperator protocol
# ================================================================
def test_isinstance_positive_for_structural_implementer():
    assert isinstance(FullDuck(), ImplicitOperator)


def test_isinstance_negative_without_solve():
    assert not isinstance(MissingSolve(), ImplicitOperator)


def test_isinstance_negative_without_fields():
    assert not isinstance(MissingFields(), ImplicitOperator)


def test_vertical_diffusion_satisfies_the_protocol():
    operator = VerticalDiffusion("z", ("u",), kappa_2)
    assert isinstance(operator, ImplicitOperator)


# ================================================================
#  VerticalDiffusion construction
# ================================================================
def test_records_attributes():
    operator = VerticalDiffusion(
        axis="z", fields=("u", "v"), kappa=kappa_2)
    assert operator.axis == "z"
    assert operator.fields == ("u", "v")
    assert operator.kappa is kappa_2


def test_normalizes_fields_to_tuple():
    operator = VerticalDiffusion("z", ["u", "v"], kappa_2)
    assert operator.fields == ("u", "v")


def test_is_frozen():
    operator = VerticalDiffusion("z", ("u",), kappa_2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        operator.axis = "y"


def test_kappa_is_stored_unbound():
    class Closure:
        def kappa(self, _state, _ctx, _field_name):
            return 1.0

    operator = VerticalDiffusion("z", ("b",), Closure.kappa)
    # the plain class-body function, no captured instance
    assert operator.kappa is Closure.__dict__["kappa"]
    assert not hasattr(operator.kappa, "__self__")


def test_rejects_empty_fields():
    with pytest.raises(ValueError, match="at least one field"):
        VerticalDiffusion("z", (), kappa_2)


def test_rejects_empty_axis():
    with pytest.raises(TypeError, match="non-empty coordinate name"):
        VerticalDiffusion("", ("u",), kappa_2)


def test_rejects_non_callable_kappa():
    with pytest.raises(TypeError, match="kappa must be callable"):
        VerticalDiffusion("z", ("u",), 1e-4)


# ================================================================
#  Boundary-condition record (per-side column rows)
# ================================================================
def test_defaults_to_neumann_neumann_rows():
    operator = VerticalDiffusion("z", ("u",), kappa_2)
    assert operator.bc == ("neumann", "neumann")


def test_records_and_normalizes_bc():
    operator = VerticalDiffusion(
        "z", ("u",), kappa_2, bc=["dirichlet", "neumann"])
    assert operator.bc == ("dirichlet", "neumann")


def test_rejects_a_bad_bc():
    with pytest.raises(ValueError, match="low, high"):
        VerticalDiffusion("z", ("u",), kappa_2, bc=("robin", "robin"))


# ================================================================
#  Kernels
# ================================================================
# The apply/solve tridiagonal kernels landed at wave 5 (ROADMAP 2.5);
# their numerical oracles (exact 1D decay, the stiff-kappa column,
# apply/solve against a dense numpy reference) live in
# ``tests/model/test_implicit_kernel.py``.


# ================================================================
#  Merge hooks
# ================================================================
def test_merge_key_groups_same_axis_instances():
    op1 = VerticalDiffusion("z", ("u",), kappa_2)
    op2 = VerticalDiffusion("z", ("b",), kappa_3)
    assert op1.merge_key() == op2.merge_key()
    assert hash(op1.merge_key()) == hash(op2.merge_key())


def test_merge_key_separates_axes():
    op_z = VerticalDiffusion("z", ("u",), kappa_2)
    op_y = VerticalDiffusion("y", ("u",), kappa_2)
    assert op_z.merge_key() != op_y.merge_key()


def test_merged_operator_sums_kappa_on_shared_fields():
    op1 = VerticalDiffusion("z", ("u", "v"), kappa_2)
    op2 = VerticalDiffusion("z", ("v", "b"), kappa_3)
    merged = op1.merged_with(op2)
    assert isinstance(merged, VerticalDiffusion)
    assert isinstance(merged, ImplicitOperator)
    assert merged.axis == "z"
    assert merged.fields == ("u", "v", "b")
    # exact coefficient math, testable without the kernel:
    assert merged.kappa(None, None, None, "v") == 5.5   # both
    assert merged.kappa(None, None, None, "u") == 2.0   # op1 only
    assert merged.kappa(None, None, None, "b") == 3.5   # op2 only


def test_merged_kappa_rejects_uncovered_fields():
    op1 = VerticalDiffusion("z", ("u",), kappa_2)
    op2 = VerticalDiffusion("z", ("v",), kappa_3)
    merged = op1.merged_with(op2)
    with pytest.raises(ValueError, match="not covered"):
        merged.kappa(None, None, None, "q")


def test_merging_is_associative_in_value():
    op1 = VerticalDiffusion("z", ("u",), kappa_2)
    op2 = VerticalDiffusion("z", ("u",), kappa_3)
    op3 = VerticalDiffusion("z", ("u",), kappa_2)
    merged = op1.merged_with(op2).merged_with(op3)
    assert merged.kappa(None, None, None, "u") == 7.5


def test_merged_with_rejects_axis_mismatch():
    op_z = VerticalDiffusion("z", ("u",), kappa_2)
    op_y = VerticalDiffusion("y", ("u",), kappa_3)
    with pytest.raises(ValueError, match="different merge keys"):
        op_z.merged_with(op_y)


def test_merged_with_rejects_family_mismatch():
    operator = VerticalDiffusion("z", ("u",), kappa_2)
    with pytest.raises((ValueError, AttributeError)):
        operator.merged_with(FullDuck())


# ================================================================
#  The merge key carries the BC structure (unlike-BC legs never merge)
# ================================================================
def test_merge_key_separates_boundary_conditions():
    # a no-slip (Dirichlet-row) velocity leg and a Neumann-row leg on
    # the SAME axis must NOT merge — kappa-summing them would silently
    # combine a -3 Dirichlet corner with a -1 Neumann corner
    no_slip = VerticalDiffusion(
        "z", ("u", "v"), kappa_2, bc=("dirichlet", "neumann"))
    neumann = VerticalDiffusion("z", ("b",), kappa_3)
    assert no_slip.merge_key() != neumann.merge_key()
    with pytest.raises(ValueError, match="different merge keys"):
        no_slip.merged_with(neumann)


def test_merge_key_groups_same_boundary_conditions():
    # two same-BC legs on the same axis still merge exactly
    op1 = VerticalDiffusion(
        "z", ("u",), kappa_2, bc=("dirichlet", "neumann"))
    op2 = VerticalDiffusion(
        "z", ("v",), kappa_3, bc=("dirichlet", "neumann"))
    assert op1.merge_key() == op2.merge_key()
    merged = op1.merged_with(op2)
    assert merged.fields == ("u", "v")
    assert merged.bc == ("dirichlet", "neumann")


# ================================================================
#  The solve-column geometry gate (stretched / terrain rejection)
# ================================================================
def test_gate_accepts_a_uniform_column():
    grid = Grid((IntervalMesh(4, (0.0, 1.0), name="z"),))
    # a plain uniform IntervalMesh column raises nothing
    reject_unsupported_solve_column(grid, "z")


def test_gate_rejects_a_stretched_column():
    grid = Grid((
        MappedIntervalMesh(4, (0.0, 1.0), lambda s: s ** 2,
                           name="z"),))
    with pytest.raises(NotImplementedError, match="uniform spacing"):
        reject_unsupported_solve_column(grid, "z")


def test_gate_rejects_a_terrain_coupled_column():
    mx = IntervalMesh(4, (0.0, 2 * np.pi), periodic=True, name="x")
    ms = IntervalMesh(4, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, height: sigma * height},
        params={"height": lambda x: 1.0 + 0.2 * np.sin(x)})
    grid = Grid((mx, ms), mapping=mapping)
    with pytest.raises(NotImplementedError, match="terrain"):
        reject_unsupported_solve_column(grid, "sigma")


def test_gate_ignores_a_missing_grid_seam():
    # anything without factors / mapping is treated as unmapped
    reject_unsupported_solve_column(object(), "z")
