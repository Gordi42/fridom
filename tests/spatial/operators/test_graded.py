"""Tests for the shared graded near-wall closure machinery.

``operators.graded`` owns the ladder arithmetic, the wall-window
builder and the ``patch_physical_ends`` seam that both graded families
drive: the average-family ``Fallback`` (``shift = 0``) and the nodal
C-grid biased kernels of ``model.modules.advection``
(``shift = 0`` on ``Center -> Inner``, ``shift = 1`` on the dual
``Inner -> Center``).

The arithmetic is pinned here against the legality rule it encodes
(every rung window must stay inside ``[0, n_cells - 1]``, and every
face whose full-width window leaves the operand's TRUE DOF range
``[shift, n_cells - 1 - shift]`` must be a reduced face); the array
paths are exercised by the two consumers' suites
(``test_fallback*.py``, ``tests/nonhydro2/test_advection.py``).
"""
import pytest

from fridom.spatial.operators.graded import (
    WALL_RUNGS,
    Rung,
    RungSpec,
    _wall_cells,
    apply_graded_walls,
    biased_ladder,
    biased_offset,
    biased_rows,
    biased_specs,
    centered_ladder,
    centered_offset,
    centered_rows,
    min_cells,
    spec_offset,
)

BIASES = ["left", "right"]
ORDERS = [3, 5]
SIZES = [2, 4]
SHIFTS = [0, 1]
WALLS = list(WALL_RUNGS)


# ================================================================
#  Window offsets (the cell frame)
# ================================================================
@pytest.mark.parametrize("order", ORDERS)
def test_biased_offset_breaks_the_odd_stencil_tie(order):
    # left bias reaches one cell further left than right bias: the
    # window of order p at face F is [F-1-offset, F-2-offset+p]
    assert biased_offset(order, "left") == order // 2
    assert biased_offset(order, "right") == order // 2 - 1
    assert (biased_offset(order, "left")
            - biased_offset(order, "right")) == 1


def test_biased_offset_at_order_one_is_the_upwind_cell():
    # order 1: cell F-1 (left) / cell F (right) — the unit Shu row
    assert biased_offset(1, "left") == 0
    assert biased_offset(1, "right") == -1


@pytest.mark.parametrize("size", SIZES)
def test_centered_offset_straddles_the_face(size):
    # the size-s centered window at face F is [F-s/2, F+s/2-1]
    assert centered_offset(size) == size // 2 - 1


# ================================================================
#  K: exactly the faces whose full window leaves the TRUE DOFs
# ================================================================
def _illegal_faces(size, offset, shift, n_cells, side):
    """Faces whose window leaves the true-DOF cells [shift, n-1-shift]."""
    lo, hi = shift, n_cells - 1 - shift
    faces = range(1, n_cells)  # F = 1 .. n_faces
    bad = [f for f in faces
           if f - 1 - offset < lo or f - 2 - offset + size > hi]
    return [f for f in bad if (f <= n_cells // 2) == (side == 0)]


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("shift", SHIFTS)
def test_biased_rows_covers_every_exterior_reaching_face(order, shift):
    # K must be >= the count of faces per side whose full-width window
    # leaves the operand's true DOFs, for BOTH biases (one K serves the
    # upwind pair) — otherwise the interior pass' exterior read would
    # survive into the output
    n_cells = 40
    k = biased_rows(order, shift)
    assert k == order // 2 + shift
    for bias in BIASES:
        offset = biased_offset(order, bias)
        for side in (0, 1):
            need = len(_illegal_faces(order, offset, shift, n_cells,
                                      side))
            assert k >= need


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("shift", SHIFTS)
def test_centered_rows_covers_every_exterior_reaching_face(size, shift):
    n_cells = 40
    k = centered_rows(size, shift)
    assert k == size // 2 - 1 + shift
    offset = centered_offset(size)
    for side in (0, 1):
        need = len(_illegal_faces(size, offset, shift, n_cells, side))
        assert k >= need


def test_two_point_mean_on_a_bc_free_operand_needs_no_closure():
    # the interior-only corner: a size-2 centered window at face F is
    # cells [F-1, F], already inside a BC-free operand's true DOFs
    assert centered_rows(2, 0) == 0
    assert centered_ladder(2, 0) == ()


# ================================================================
#  The ladders (widest first, wall-adjacent last) are legal
# ================================================================
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("shift", SHIFTS)
def test_biased_ladder_is_widest_first_and_bottoms_out_at_upwind_one(
        order, shift):
    ladder = biased_ladder(order, shift)
    assert len(ladder) == biased_rows(order, shift)
    assert list(ladder) == sorted(ladder, reverse=True)
    assert ladder[-1] == 1               # the 1st-order upwind rung
    assert all(p % 2 == 1 for p in ladder)
    # at shift=1 the wall cells are exact Dirichlet zeros, so the
    # outermost rung keeps the interior width; at shift=0 (a BC-free
    # bounded operand) it genuinely narrows
    assert ladder[0] == (order if shift else order - 2)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("shift", SHIFTS)
@pytest.mark.parametrize("bias", BIASES)
def test_biased_ladder_windows_stay_inside_the_lattice(
        order, shift, bias):
    # the legality rule: the rung at distance d, applied at face d
    # (left wall) or n_faces-d+1 (right wall), reads only cells
    # 0 .. n_cells-1 — no exterior cell, ever
    ladder = biased_ladder(order, shift)
    k = len(ladder)
    n_cells = 40
    for d in range(1, k + 1):
        p = ladder[k - d]
        offset = biased_offset(p, bias)
        for face in (d, (n_cells - 1) - d + 1):
            start = face - 1 - offset
            end = start + p - 1
            assert start >= 0
            assert end <= n_cells - 1


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("shift", SHIFTS)
def test_centered_ladder_windows_stay_inside_the_lattice(size, shift):
    ladder = centered_ladder(size, shift)
    assert len(ladder) == centered_rows(size, shift)
    assert list(ladder) == sorted(ladder, reverse=True)
    assert all(s % 2 == 0 for s in ladder)
    k = len(ladder)
    n_cells = 40
    for d in range(1, k + 1):
        s = ladder[k - d]
        offset = centered_offset(s)
        for face in (d, (n_cells - 1) - d + 1):
            start = face - 1 - offset
            assert start >= 0
            assert start + s - 1 <= n_cells - 1
    if k:
        assert ladder[-1] == 2           # the two-point mean rung


# ================================================================
#  The bottom rung of a biased ladder (the ``wall=`` knob)
# ================================================================
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("shift", SHIFTS)
def test_biased_specs_default_to_the_upwind_one_bottom(order, shift):
    # the default reproduces biased_ladder EXACTLY (every rung an odd
    # biased row, bottoming out at the 1st-order upwind cell)
    specs = biased_specs(order, shift)
    assert specs == biased_specs(order, shift, "upwind1")
    assert tuple(spec.width for spec in specs) == biased_ladder(
        order, shift)
    assert all(spec.family == "biased" for spec in specs)
    assert specs[-1] == RungSpec("biased", 1)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("shift", SHIFTS)
def test_centered2_replaces_only_the_wall_adjacent_rung(order, shift):
    # the trade the option offers: the wall-adjacent (last) rung
    # becomes the two-point centered mean; EVERY other rung — and the
    # ladder's length K — is untouched, so the interior and the
    # reduced faces at distance d > 1 are bitwise the default's
    default = biased_specs(order, shift, "upwind1")
    centered = biased_specs(order, shift, "centered2")
    assert len(centered) == len(default) == biased_rows(order, shift)
    assert centered[:-1] == default[:-1]
    assert centered[-1] == RungSpec("centered", 2)


def test_unknown_wall_rung_is_taught():
    with pytest.raises(ValueError, match=r"wall must be one of"):
        biased_specs(3, 0, "quick")


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("shift", SHIFTS)
@pytest.mark.parametrize("bias", BIASES)
@pytest.mark.parametrize("wall", WALLS)
def test_biased_spec_windows_stay_inside_the_lattice(
        order, shift, bias, wall):
    # the legality rule (R1) holds for BOTH bottom rungs: the
    # centered2 window at the wall-adjacent face reads the two cells
    # straddling it, which is exactly where the upwind1 window sits —
    # no exterior cell, ever
    specs = biased_specs(order, shift, wall)
    k = len(specs)
    n_cells = 40
    for d in range(1, k + 1):
        spec = specs[k - d]
        offset = spec_offset(spec, bias)
        for face in (d, (n_cells - 1) - d + 1):
            start = face - 1 - offset
            assert start >= 0
            assert start + spec.width - 1 <= n_cells - 1


@pytest.mark.parametrize("bias", BIASES)
def test_spec_offset_follows_the_rung_family(bias):
    # a centered rung has no bias: both members of an upwind pair read
    # the same window (and hence return the same face value)
    assert spec_offset(RungSpec("centered", 2), bias) == (
        centered_offset(2))
    assert spec_offset(RungSpec("biased", 3), bias) == (
        biased_offset(3, bias))


@pytest.mark.parametrize("shift", SHIFTS)
def test_the_centered2_bottom_rung_reaches_the_wall_cell_alike(shift):
    # the wall-cell synthesis (shift = 1) is driven by the window, not
    # by the family: the two-point rung at distance 1 straddles the
    # wall-adjacent face, so it reads the wall cell on ONE side of each
    # wall — exactly as the upwind1 rung does on its biased side
    spec = biased_specs(3, shift, "centered2")[-1]
    rung = Rung(spec.width, spec_offset(spec, "left"),
                lambda a, _x: a)
    assert _wall_cells(rung, 0, 1, shift) == (shift, 0)
    assert _wall_cells(rung, 1, 1, shift) == (0, shift)


# ================================================================
#  Wall-cell synthesis: static, and only where a wall is reached
# ================================================================
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("shift", SHIFTS)
@pytest.mark.parametrize("bias", BIASES)
def test_wall_cells_are_synthesized_exactly_where_the_window_reaches(
        order, shift, bias):
    ladder = biased_ladder(order, shift)
    k = len(ladder)
    for d in range(1, k + 1):
        p = ladder[k - d]
        rung = Rung(p, biased_offset(p, bias), lambda a, _x: a)
        lead, trail = _wall_cells(rung, 0, d, shift)
        assert trail == 0
        # cell 0 is a wall cell only on the shift=1 frame
        assert lead == (shift if d - 1 - rung.offset == 0 else 0)
        lead, trail = _wall_cells(rung, 1, d, shift)
        assert lead == 0
        assert trail == (shift
                         if p - 1 - rung.offset - d == 0 else 0)
        # a rung NEVER reaches past the wall cell
        assert p - 1 - rung.offset - d <= 0


def test_bc_free_operand_synthesizes_no_wall_cells():
    # shift=0: the lattice cells ARE the true DOFs, so no window ever
    # needs a synthesized value (the Fallback / Center -> Inner path)
    for order in ORDERS:
        ladder = biased_ladder(order, 0)
        k = len(ladder)
        for bias in BIASES:
            for d in range(1, k + 1):
                p = ladder[k - d]
                rung = Rung(p, biased_offset(p, bias),
                            lambda a, _x: a)
                assert _wall_cells(rung, 0, d, 0) == (0, 0)
                assert _wall_cells(rung, 1, d, 0) == (0, 0)


# ================================================================
#  Minimum extent, and the empty-ladder short circuit
# ================================================================
@pytest.mark.parametrize("order", ORDERS)
def test_min_cells_admits_the_widest_rung_and_separates_the_walls(
        order):
    need = min_cells(order)
    assert need == order + 1
    for shift in SHIFTS:
        n_cells = need + shift            # Center count -> lattice
        assert order <= n_cells           # the widest window fits
        assert 2 * biased_rows(order, shift) <= n_cells - 1


def test_empty_ladder_returns_the_interior_pass_untouched():
    # K = 0 (the two-point mean on a BC-free operand): no patch runs
    # and the field object is passed through by identity
    sentinel = object()
    assert apply_graded_walls(
        None, "y", sentinel, (), 0) is sentinel
