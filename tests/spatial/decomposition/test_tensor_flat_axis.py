r"""The halo fill on a one-cell axis, and the two-spelling twin.

A prefix-mirrored shard of ``test_tensor.py`` (AGENTS.md
oversized-module exception -- the changed-file mapping still resolves
by ``test_tensor*.py``). Two groups of claims:

- **The one-cell landscape.** ``mesh.n_cells == 1`` does not imply one
  DOF, and a one-DOF bounded factor does not imply a zero derivative.
  These exist to make one specific mistake impossible: keying a "flat
  axis needs no halo / derivatives along it are zero" rule on
  ``mesh.n_cells == 1`` or on ``factor.shape[0] == 1``. Every assertion
  is on a *number an operator produced*, not on an internal flag, so it
  fails whatever spelling such an elision takes.
- **The twin invariant.** ``_write_axis`` is documented as
  ``_fill_axis``'s ``materialize=True`` twin producing "the same
  values", so the two spellings must agree on every input -- including
  a factor the walled one-cell axis empties. They did not: the map
  honours the Dirichlet ``_VACANT`` ``sign == 0`` rule and returns
  exact zeros, while ``_ghost_values`` called ``dof(rank)`` purely for
  a shape and refused a zero-DOF factor. The fill now builds that slot
  from the storage frame instead.

Analysis: ``design/research/thin_axis_halo_investigation.md`` Sec.5
(the "zero-DOF spaces already flow through the fill undefended"
hazard).

Self-contained per the shard convention: builders are duplicated
rather than imported across test files.
"""
import itertools

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.spatial.decomposition.tensor as tensor_mod
from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.decomposition.tensor import (
    TensorDecomposition,
    _axis_map,
    _flat_elided_width,
    _flat_ghosts_are_copies,
)
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.finite_difference import FiniteDifference
from fridom.spatial.operators.flux_diff import FluxDifference
from fridom.spatial.spaces.average import CellAvg, FaceAvg
from fridom.spatial.spaces.nodal import NodeSet


def _mesh(n_cells, *, periodic):
    return IntervalMesh(n_cells, (0.0, 1.0), periodic=periodic, name="z")


def _grid(n_cells, *, periodic):
    grid = Grid((_mesh(n_cells, periodic=periodic),), device_ids=(0,))
    return grid, grid.factors[0]


def _decomp(mesh, width):
    return TensorDecomposition(
        meshes=(mesh,), names=("z",), halo=HaloSpec({"z": width}),
        layouts=(Layout({}),), device_ids=(0,))


def _space(mesh, kind, bc):
    if isinstance(kind, NodeSet):
        return mesh.nodal(kind, bc=bc)
    return mesh.average(kind, bc=bc)


# ================================================================
#  The DOF table: n_cells == 1 does NOT mean one DOF
# ================================================================
# measured at HEAD (2026-08-12); every entry is the *realized*
# ``space.shape[0]`` of a ONE-CELL axis
_DOF_AT_ONE_CELL = [
    # bounded, BC-free
    pytest.param(False, NodeSet.CENTER, BC.NONE, 1, id="center-free"),
    pytest.param(False, NodeSet.LEFT, BC.NONE, 1, id="left-free"),
    pytest.param(False, NodeSet.RIGHT, BC.NONE, 1, id="right-free"),
    pytest.param(False, NodeSet.OUTER, BC.NONE, 2, id="outer-free"),
    pytest.param(False, NodeSet.INNER, BC.NONE, 0, id="inner-free"),
    pytest.param(False, CellAvg, BC.NONE, 1, id="cellavg-free"),
    pytest.param(False, FaceAvg, BC.NONE, 0, id="faceavg-free"),
    # bounded, Dirichlet-tagged: a member boundary DOF is dropped
    pytest.param(False, NodeSet.CENTER, BC.DIRICHLET, 1, id="center-dir"),
    pytest.param(False, NodeSet.LEFT, BC.DIRICHLET, 0, id="left-dir"),
    pytest.param(False, NodeSet.RIGHT, BC.DIRICHLET, 0, id="right-dir"),
    pytest.param(False, NodeSet.OUTER, BC.DIRICHLET, 0, id="outer-dir"),
    pytest.param(False, NodeSet.INNER, BC.DIRICHLET, 0, id="inner-dir"),
    pytest.param(False, CellAvg, BC.DIRICHLET, 1, id="cellavg-dir"),
    # bounded, Neumann-tagged: nothing is dropped
    pytest.param(False, NodeSet.CENTER, BC.NEUMANN, 1, id="center-neu"),
    pytest.param(False, NodeSet.OUTER, BC.NEUMANN, 2, id="outer-neu"),
    pytest.param(False, NodeSet.INNER, BC.NEUMANN, 0, id="inner-neu"),
    pytest.param(False, CellAvg, BC.NEUMANN, 1, id="cellavg-neu"),
    # periodic: every family carries exactly one
    pytest.param(True, NodeSet.CENTER, BC.NONE, 1, id="periodic-center"),
    pytest.param(True, NodeSet.LEFT, BC.NONE, 1, id="periodic-left"),
    pytest.param(True, NodeSet.RIGHT, BC.NONE, 1, id="periodic-right"),
    pytest.param(True, CellAvg, BC.NONE, 1, id="periodic-cellavg"),
    pytest.param(True, FaceAvg, BC.NONE, 1, id="periodic-faceavg"),
]


@pytest.mark.parametrize(("periodic", "kind", "bc", "dofs"),
                         _DOF_AT_ONE_CELL)
def test_one_cell_axis_dof_count_per_space_and_bc(periodic, kind, bc,
                                                 dofs):
    """``mesh.n_cells == 1`` implies one DOF only on a periodic axis."""
    space = _space(_mesh(1, periodic=periodic), kind, bc)
    assert space.shape == (dofs,)


def test_outer_keeps_two_dofs_on_a_one_cell_bounded_axis():
    """The headline counterexample: 1 cell, 2 DOFs, real derivative."""
    grid, mesh = _grid(1, periodic=False)
    outer = mesh.nodal(NodeSet.OUTER)
    assert outer.shape == (2,)
    # a genuine vertical flux divergence in a single-cell column:
    # 1 cell, dz = 1, faces [1, 3] -> d/dz = 2, NOT 0
    field = grid.create_field(outer, data=jnp.asarray([1.0, 3.0]))
    out = FiniteDifference()["z"](field)
    assert out.function_space.shape == (1,)
    assert float(np.asarray(out.data)[0]) == pytest.approx(2.0 / mesh.dx)


# ================================================================
#  The shape[0] == 1 trap: a ONE-DOF bounded factor with a real d/dz
# ================================================================
@pytest.mark.parametrize("n_cells", [2, 3, 4])
def test_a_one_dof_bounded_factor_still_has_a_nonzero_derivative(n_cells):
    r"""``Inner(DIRICHLET)`` at ``n_cells = 2`` has ``shape == (1,)``.

    Its ``d/dz`` of a *constant* column is :math:`\pm 1/dz` (the two
    wall values are zero), so the "constant column => zero
    derivative" reasoning that holds on a periodic flat axis is false
    here. This is the exact factor nonhydro2's walled ``w`` lives on.
    """
    grid, mesh = _grid(n_cells, periodic=False)
    inner = mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    assert inner.shape == (n_cells - 1,)
    field = grid.create_field(inner, data=jnp.ones(inner.shape))
    for op in (FiniteDifference(), FluxDifference()):
        got = np.asarray(op["z"](field).data)
        assert got[0] == pytest.approx(1.0 / mesh.dx)
        assert got[-1] == pytest.approx(-1.0 / mesh.dx)
        assert np.max(np.abs(got)) > 0.0


# ================================================================
#  The soundness predicate, read off the fill map
# ================================================================
def _constant_column(factor, width):
    r"""Whether the filled storage column is the single DOF, repeated.

    The **only** sound gate for a flat-axis rule. True iff the factor
    has exactly one true DOF *and* the ghost fill copies it into every
    ghost slot with no sign flip and no zeroed slot -- which is what
    makes every consistent stencil reduce to the identity (weights sum
    to 1) or to exactly zero (weights sum to 0). Derived from
    ``_axis_map``, never declared per space, so a new space family
    cannot mis-declare it.

    This calls the **shipped** audit that guards the elision gate
    (``_flat_ghosts_are_copies``) rather than a copy of it, so the
    enumeration below cannot drift away from what the gate checks.
    """
    return _flat_ghosts_are_copies("z", factor, width)


_ONE_CELL_CASES = [
    (periodic, kind, bc)
    for periodic, kind, bc in itertools.product(
        (True, False),
        (NodeSet.CENTER, NodeSet.LEFT, NodeSet.RIGHT, NodeSet.OUTER,
         NodeSet.INNER, CellAvg, FaceAvg),
        (BC.NONE, BC.DIRICHLET, BC.NEUMANN))
]

# measured at HEAD: the complete set of (space, bc) that pass
# ``_constant_column`` on a ONE-CELL axis at width 1. Every periodic
# family passes; on a bounded axis ONLY the two center-like families
# with Neumann on both components do.
_CONSTANT_COLUMN_AT_ONE_CELL = {
    (True, "CENTER", "NONE"), (True, "LEFT", "NONE"),
    (True, "RIGHT", "NONE"), (True, "CellAvg", "NONE"),
    (True, "FaceAvg", "NONE"),
    (False, "CENTER", "NEUMANN"), (False, "CellAvg", "NEUMANN"),
}


def test_constant_column_holds_exactly_where_it_is_claimed():
    """Enumerate: nothing else on a 1-cell axis has a constant column.

    Loosening a flat-axis predicate to ``n_cells == 1`` or to
    ``shape[0] == 1`` makes this set larger, and this test names every
    member.
    """
    got = set()
    for periodic, kind, bc in _ONE_CELL_CASES:
        name = kind.name if isinstance(kind, NodeSet) else kind.__name__
        try:
            space = _space(_mesh(1, periodic=periodic), kind, bc)
        except ValueError:
            continue  # not a constructible (mesh, space, bc) triple
        try:
            if _constant_column(space, 1):
                got.add((periodic, name, bc.name))
        except NotImplementedError:
            continue  # the fill itself is ungrounded; never elidable
    assert got == _CONSTANT_COLUMN_AT_ONE_CELL


def test_the_shipped_gate_is_a_strict_subset_of_the_sound_predicate():
    """``factor.is_flat`` never over-claims.

    The shipped gate is ``mesh.periodic and n_cells == 1``, exposed as
    ``factor.is_flat`` and consumed by ``_flat_elided_width``
    (Sec.5/Sec.9). Every factor it admits really does have a constant
    column, at every negotiated width. If someone widens the gate --
    to ``n_cells == 1`` or to ``shape[0] == 1`` -- this fails.
    """
    for periodic, kind, bc in _ONE_CELL_CASES:
        try:
            space = _space(_mesh(1, periodic=periodic), kind, bc)
        except ValueError:
            continue
        assert space.is_flat is (space.mesh.periodic
                                and space.mesh.n_cells == 1)
        if not space.is_flat:
            continue
        for width in (1, 2, 3, 5):
            assert _constant_column(space, width), (space, width)
            # ... and the gate then really does drop the ghosts
            assert _flat_elided_width("z", space, width) == 0


@pytest.mark.parametrize("bc", [BC.NONE, BC.DIRICHLET, BC.NEUMANN])
def test_a_bounded_center_column_is_not_a_wrap(bc):
    """The bounded 1-cell fill is BC-structured, not a copy.

    ``NONE`` leaves the ghosts untouched (R1: exterior undefined),
    ``DIRICHLET`` odd-reflects (ghost ``= -u``, so a two-point
    derivative across the cell is ``2u/dz``), only ``NEUMANN`` mirrors.
    """
    space = _space(_mesh(1, periodic=False), NodeSet.CENTER, bc)
    src, neg, zero = _axis_map(3, 1, 1, space)
    if bc is BC.NONE:
        assert list(src) == [0, 1, 2]  # identity: nothing is filled
        assert not neg.any()
        assert not zero.any()
    elif bc is BC.DIRICHLET:
        assert list(src) == [1, 1, 1]
        assert list(neg) == [True, False, True]  # ghost = -u
    else:
        assert list(src) == [1, 1, 1]
        assert not neg.any()


def test_the_dirichlet_flat_column_really_carries_2u_over_dz():
    """Sec.5's arithmetic claim, on the synced storage itself."""
    mesh = _mesh(1, periodic=False)
    space = mesh.average(CellAvg, bc=BC.DIRICHLET)
    decomp = _decomp(mesh, 1)
    column = np.asarray(decomp.sync(
        decomp.pad(jnp.asarray([1.0]), space), space))
    assert list(column) == [-1.0, 1.0, -1.0]
    # the two-point derivative across the left face of the single cell
    left_face = (column[1] - column[0]) / mesh.dx
    assert left_face == pytest.approx(2.0 / mesh.dx)


# ================================================================
#  The two fill spellings must agree (the zero-DOF hazard)
# ================================================================
_ZERO_DOF_CASES = [
    pytest.param(NodeSet.INNER, BC.NONE, id="inner-free"),
    pytest.param(FaceAvg, BC.NONE, id="faceavg-free"),
    pytest.param(NodeSet.INNER, BC.DIRICHLET, id="inner-dirichlet"),
    pytest.param(NodeSet.OUTER, BC.DIRICHLET, id="outer-dirichlet"),
    pytest.param(NodeSet.LEFT, BC.DIRICHLET, id="left-dirichlet"),
]


@pytest.mark.parametrize(("kind", "bc"), _ZERO_DOF_CASES)
def test_both_fill_spellings_agree_on_a_zero_dof_factor(kind, bc):
    """``_write_axis`` is documented as ``_fill_axis``'s twin.

    A walled 1-cell axis empties ``Inner``/``FaceAvg`` and every
    Dirichlet-tagged member space; both spellings must still produce
    the same column. This was the reported failure: the map honours the
    ``sign == 0`` vacant-slot rule while the write path called
    ``dof(1)`` for a shape and refused.

    The column is width-1 storage with no true DOF between the ghosts,
    and it is *all* exact zeros -- the honest answer for a wall-normal
    quantity with no interior face.
    """
    mesh = _mesh(1, periodic=False)
    space = _space(mesh, kind, bc)
    assert space.shape == (0,)
    decomp = _decomp(mesh, 1)
    padded = decomp.pad(jnp.zeros((0,)), space)
    mapped = np.asarray(decomp.sync(padded, space, materialize=False))
    written = np.asarray(decomp.sync(padded, space, materialize=True))
    assert np.array_equal(mapped, written)
    assert list(mapped) == [0.0, 0.0]
    # +0.0, not -0.0: the vacant slot is a select, never a multiply
    assert not np.signbit(mapped).any()
    assert not np.signbit(written).any()


# The whole bounded matrix, not just the case that broke: the twin
# promise is an invariant over inputs, so it is asserted over every
# buildable (space, bc) on 1..4 cells at widths 1 and 2. Refusals count
# too -- a spelling that refuses where its twin fills is the same bug.
_ALL_SPACES = (NodeSet.CENTER, NodeSet.LEFT, NodeSet.RIGHT,
               NodeSet.OUTER, NodeSet.INNER, CellAvg, FaceAvg)

_FILL_CASES = [
    pytest.param(
        kind, bc,
        id=f"{kind.name if isinstance(kind, NodeSet) else kind.__name__}"
           f"-{bc.name}".lower())
    for kind, bc in itertools.product(
        _ALL_SPACES, (BC.NONE, BC.DIRICHLET, BC.NEUMANN))
]


def _outcome(decomp, space, materialize):
    """Return the synced column, or the refusal's message text."""
    values = jnp.arange(1.0, space.shape[0] + 1.0)
    try:
        return np.asarray(decomp.sync(
            decomp.pad(values, space), space, materialize=materialize))
    except NotImplementedError as exc:
        return f"NotImplementedError: {exc}"


def _agrees(mapped, written):
    if isinstance(mapped, np.ndarray) != isinstance(written, np.ndarray):
        return False  # one spelling filled, the other refused
    if isinstance(mapped, np.ndarray):
        return np.array_equal(mapped, written)
    return mapped == written


@pytest.mark.parametrize(("kind", "bc"), _FILL_CASES)
def test_both_fill_spellings_agree_on_every_bounded_case(kind, bc):
    """The headline invariant: the twins never disagree."""
    for n_cells, width in itertools.product((1, 2, 3, 4), (1, 2)):
        mesh = _mesh(n_cells, periodic=False)
        try:
            space = _space(mesh, kind, bc)
        except ValueError:
            continue  # not a constructible (space, bc) pair
        decomp = _decomp(mesh, width)
        mapped = _outcome(decomp, space, materialize=False)
        written = _outcome(decomp, space, materialize=True)
        assert _agrees(mapped, written), (n_cells, width, mapped, written)


# ================================================================
#  Working configurations keep their exact columns
# ================================================================
# Measured on 4 cells at width 1. The zero-slot rewrite must not shift
# a single value of a configuration that already worked -- including the
# ones whose fill *does* go through the ``sign == 0`` branch (the two
# member-lattice Dirichlet rows, whose end slots are the wall DOF).
_UNCHANGED_COLUMNS = [
    pytest.param(NodeSet.CENTER, BC.DIRICHLET,
                 [-1.0, 1.0, 2.0, 3.0, 4.0, -4.0], id="center-dirichlet"),
    pytest.param(NodeSet.CENTER, BC.NEUMANN,
                 [1.0, 1.0, 2.0, 3.0, 4.0, 4.0], id="center-neumann"),
    pytest.param(NodeSet.INNER, BC.DIRICHLET,
                 [0.0, 1.0, 2.0, 3.0, 0.0], id="inner-dirichlet"),
    pytest.param(NodeSet.OUTER, BC.DIRICHLET,
                 [0.0, 1.0, 2.0, 3.0, 0.0], id="outer-dirichlet"),
]


@pytest.mark.parametrize("materialize", [False, True])
@pytest.mark.parametrize(("kind", "bc", "column"), _UNCHANGED_COLUMNS)
def test_a_populated_axis_keeps_its_exact_column(kind, bc, column,
                                                 materialize):
    mesh = _mesh(4, periodic=False)
    space = _space(mesh, kind, bc)
    decomp = _decomp(mesh, 1)
    out = np.asarray(decomp.sync(
        decomp.pad(jnp.arange(1.0, space.shape[0] + 1.0), space),
        space, materialize=materialize))
    assert list(out) == column
    # a vacant Dirichlet slot stays +0.0 in both spellings
    assert not np.signbit(out[out == 0.0]).any()


def test_a_one_dof_dirichlet_axis_keeps_its_exact_column():
    """One cell, one DOF, both spellings: the odd extension."""
    mesh = _mesh(1, periodic=False)
    space = _space(mesh, NodeSet.CENTER, BC.DIRICHLET)
    decomp = _decomp(mesh, 1)
    padded = decomp.pad(jnp.asarray([1.0]), space)
    for materialize in (False, True):
        out = np.asarray(decomp.sync(padded, space,
                                     materialize=materialize))
        assert list(out) == [-1.0, 1.0, -1.0]


# ================================================================
#  Flat-axis halo elision: the shipped gate
# ================================================================
def test_flat_axis_stores_no_ghost_slots():
    # the whole point: a periodic one-cell axis carries one storage
    # slot however wide the negotiated halo is
    mesh = _mesh(1, periodic=True)
    for width in (1, 2, 3):
        decomp = _decomp(mesh, width)
        assert decomp.storage_shape(mesh.center) == (1,)
        # ... while the negotiated halo itself is left untouched: the
        # stencil reach guards and the solver halo demand read it
        assert decomp.halo["z"] == width


def test_flat_axis_pad_and_sync_are_identities():
    mesh = _mesh(1, periodic=True)
    decomp = _decomp(mesh, 3)
    padded = decomp.pad(jnp.asarray([7.0]), mesh.center)
    assert jnp.array_equal(padded, jnp.asarray([7.0]))
    assert jnp.array_equal(decomp.sync(padded, mesh.center), padded)
    assert jnp.array_equal(
        decomp.sync(padded, mesh.center, materialize=True), padded)
    assert jnp.array_equal(decomp.unpad(padded, mesh.center),
                           jnp.asarray([7.0]))


@pytest.mark.parametrize(
    ("n_cells", "periodic", "stored"),
    [
        pytest.param(1, True, 1, id="flat-elided"),
        pytest.param(2, True, 8, id="periodic-2-not-elided"),
        pytest.param(1, False, 7, id="bounded-1-not-elided"),
        pytest.param(2, False, 8, id="bounded-2-not-elided"),
    ])
def test_only_a_periodic_one_cell_axis_is_elided(n_cells, periodic,
                                                stored):
    # the nz = 2 control and the bounded controls: nothing but a
    # periodic single-cell axis loses its ghosts
    mesh = _mesh(n_cells, periodic=periodic)
    decomp = _decomp(mesh, 3)
    assert decomp.storage_shape(mesh.center) == (stored,)


def test_a_walled_two_cell_inner_factor_keeps_its_halo():
    # the trap the predicate must not fall into: Inner on a walled
    # nz = 2 mesh has shape (1,) yet a nonzero +-1/dz divergence, so
    # nothing may key on shape[0] == 1 (report section 5)
    mesh = _mesh(2, periodic=False)
    inner = mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    assert inner.shape == (1,)
    assert inner.is_flat is False
    decomp = _decomp(mesh, 2)
    assert decomp.storage_shape(inner) == (5,)


def test_the_audit_is_a_backstop_not_the_predicate():
    # a bounded one-cell NEUMANN Center mirrors its single DOF, so its
    # width-1 fill IS a pure copy and the audit alone would let it
    # through -- what refuses it is `is_flat`, which demands a
    # *periodic* topology. The audit catches a widened predicate, it
    # does not define the rule (report section 5)
    mesh = _mesh(1, periodic=False)
    space = mesh.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    assert _flat_ghosts_are_copies("z", space, 1) is True
    assert space.is_flat is False
    assert _flat_elided_width("z", space, 1) == 1


def test_flat_ghosts_are_not_copies_when_the_axis_has_two_dofs():
    # a periodic 2-cell axis wraps to [b, a, b, a, ...], never to a
    # constant -- the audit rejects it on the DOF count alone
    mesh = _mesh(2, periodic=True)
    assert not _flat_ghosts_are_copies("z", mesh.center, 1)


def test_elided_width_passes_non_flat_and_zero_width_through():
    flat = _mesh(1, periodic=True)
    deep = _mesh(8, periodic=True)
    assert _flat_elided_width("z", deep.center, 3) == 3
    assert _flat_elided_width("z", flat.center, 0) == 0
    assert _flat_elided_width("z", flat.center, 3) == 0


def test_elided_width_refuses_a_flat_axis_whose_fill_is_not_a_copy(
        monkeypatch):
    # the loud guard against a mis-keyed predicate: if the fill ever
    # stops being a pure copy on an axis reported flat, the gate
    # raises instead of dropping ghosts that carried information.
    # A raise, not an assert, so `python -O` cannot strip it
    mesh = _mesh(1, periodic=True)

    def wrong_map(size, n, width, factor):
        src, neg, zero = _axis_map(size, n, width, factor)
        neg[0] = True  # a sign flip: no longer a plain copy
        return src, neg, zero

    monkeypatch.setattr(tensor_mod, "_axis_map", wrong_map)
    with pytest.raises(AssertionError, match="section 5"):
        _flat_elided_width("z", mesh.center, 3)
