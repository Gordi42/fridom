"""Tests for fridom.model.modules.immersed (MaskState, IP-D5)."""
import types

import numpy as np
import pytest

from fridom.model.modules.immersed import MaskState
from fridom.model.stages import StageKind
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh


def _indicator(x, y):
    """Wet region x < 0.5 (periodic) and y < 1.0 (bounded)."""
    return (x < 0.5) & (y < 1.0)


def _immersed_grid():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    my = IntervalMesh(6, (0.0, 2.0), periodic=False, name="y")
    return Grid((mx, my), immersed=ImmersedDomain(_indicator))


def _table(grid, prognostic):
    """Return a duck-typed field table exposing what bind() reads."""
    return types.SimpleNamespace(grid=grid, prognostic=prognostic)


# ================================================================
#  bind
# ================================================================
def test_bind_freezes_the_prognostic_names():
    grid = _immersed_grid()
    mod = MaskState()
    mod.bind(_table(grid, ("u", "v", "p")))
    assert mod._names == ("u", "v", "p")


def test_bind_without_immersed_domain_raises():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    my = IntervalMesh(6, (0.0, 2.0), periodic=False, name="y")
    plain = Grid((mx, my))
    mod = MaskState()
    with pytest.raises(ValueError, match="needs an immersed grid"):
        mod.bind(_table(plain, ("u",)))


def test_bind_without_grid_raises():
    mod = MaskState()
    with pytest.raises(ValueError, match="needs an immersed grid"):
        mod.bind(_table(None, ("u",)))


# ================================================================
#  Stage declaration (module protocol)
# ================================================================
def test_stage_is_a_constraint_stage():
    mod = MaskState()
    (stage,) = mod.stages
    assert stage.kind is StageKind.CONSTRAINT
    assert stage.advances == ()  # a pure correction, claims nothing


def test_collected_stages_resolves_the_masking_method():
    grid = _immersed_grid()
    mod = MaskState()
    mod.bind(_table(grid, ("u",)))
    (stage,) = mod.collected_stages()
    assert stage.kind is StageKind.CONSTRAINT
    assert stage.name == "mask_state"
    assert stage.fn is MaskState._mask_state


# ================================================================
#  The masking stage
# ================================================================
def test_mask_state_zeros_dry_cell_dofs():
    grid = _immersed_grid()
    space = grid.factors[0].center * grid.factors[1].center
    field = grid.create_field(space, init=lambda x, y: x + y + 1.0)
    mask = grid.immersed.mask(space)
    mod = MaskState()
    mod.bind(_table(grid, ("phi",)))

    out = mod._mask_state({"phi": field}, None)

    masked = np.asarray(out["phi"].data)
    original = np.asarray(field.data)
    keep = np.asarray(mask.data)
    # wet DOFs untouched, dry DOFs zeroed exactly
    assert np.array_equal(masked[keep], original[keep])
    assert np.all(masked[~keep] == 0.0)
    # the geometry genuinely has dry cells to zero
    assert not keep.all()


def test_mask_state_uses_the_face_slip_rule():
    # a velocity-role (right-face) field is masked by the staggered
    # (no-slip AND) mask, not the cell mask — the wet-face count is
    # smaller than the wet-cell count on this geometry.
    grid = _immersed_grid()
    space = grid.factors[0].right * grid.factors[1].center
    field = grid.create_field(space, init=lambda x, y: 1.0)  # noqa: ARG005
    face_mask = grid.immersed.mask(space)
    mod = MaskState()
    mod.bind(_table(grid, ("u",)))

    out = mod._mask_state({"u": field}, None)
    masked = np.asarray(out["u"].data)
    assert np.array_equal(masked, np.asarray(face_mask.data).astype(
        masked.dtype))


def test_mask_state_masks_every_prognostic():
    grid = _immersed_grid()
    cell = grid.factors[0].center * grid.factors[1].center
    face = grid.factors[0].right * grid.factors[1].center
    state = {
        "p": grid.create_field(cell, init=lambda x, y: 2.0),  # noqa: ARG005
        "u": grid.create_field(face, init=lambda x, y: 3.0),  # noqa: ARG005
    }
    mod = MaskState()
    mod.bind(_table(grid, ("p", "u")))
    out = mod._mask_state(state, None)
    for name, sp in (("p", cell), ("u", face)):
        keep = np.asarray(grid.immersed.mask(sp).data)
        assert np.all(np.asarray(out[name].data)[~keep] == 0.0)


def test_mask_state_no_prognostics_is_a_noop():
    grid = _immersed_grid()
    mod = MaskState()
    mod.bind(_table(grid, ()))
    assert mod._mask_state({}, None) == {}
