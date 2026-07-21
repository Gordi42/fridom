"""Tests for fridom.model.modules.immersed (MaskState, IP-D5)."""
import types

import numpy as np
import pytest

import fridom.hydrostatic as hy
from fridom.model.modules.immersed import _MASK_ORDER, MaskState
from fridom.model.stages import StageKind
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.decomposition.halo import HaloSpec
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


def test_bind_captures_the_immersed_descriptor_and_coords():
    # the masking reads the immersed descriptor and coordinate names
    # from bind (the real grid), never from the field's (tracer) grid
    # at trace time — so it captures them here
    grid = _immersed_grid()
    mod = MaskState()
    mod.bind(_table(grid, ("u",)))
    assert mod._immersed is grid.immersed
    assert mod._coords == grid.names


def test_extra_halo_is_zero_per_coordinate():
    # the masking is a local (zero-stencil) concrete-field multiply the
    # halo tracer cannot follow, so the stage is halo-trace exempt with
    # a zero declared halo (IP-D5)
    grid = _immersed_grid()
    mod = MaskState()
    mod.bind(_table(grid, ("u",)))
    halo = mod.extra_halo
    assert isinstance(halo, HaloSpec)
    assert halo == HaloSpec(dict.fromkeys(grid.names, 0))


def test_masking_stage_sorts_after_the_projection():
    # the CONSTRAINT-stage order sentinel puts the mask last (after the
    # pressure projection at order 0) — the composer's overlap lint
    # requires the explicit order
    (stage,) = MaskState().stages
    assert stage.order == _MASK_ORDER
    assert _MASK_ORDER > 0


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


# ================================================================
#  ConstantSpace prognostics (the 2D barotropic ps) are skipped
# ================================================================
def test_maskstate_skips_constant_space_prognostics():
    # the barotropic ps lives on Profile(x, y) -- a ConstantSpace along
    # z -- which immersed.mask() cannot resolve (it rejects a Constant
    # factor).  MaskState must SKIP such prognostics (the free-surface
    # module owns and masks the 2D barotropic state), not crash on them.
    mx = IntervalMesh(8, (0.0, 1.0), periodic=True, name="x")
    my = IntervalMesh(8, (0.0, 1.0), periodic=True, name="y")
    mz = IntervalMesh(4, (0.0, 1.0), periodic=False, name="z")
    grid = Grid((mx, my, mz), immersed=ImmersedDomain(
        lambda x, y, z: (z > 0.5).astype(float)))  # noqa: ARG005
    model = hy.Model(
        grid=grid,
        core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(0.01, order=3),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False)
    # ps is a PROGNOSTIC ConstantSpace(z) field the MaskState sees
    assert "ps" in set(model._artifacts.field_table.prognostic)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        k: 0.1 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "b")})
    # advancing steps the shared MaskState over ps without raising
    model.advance(3)
    assert not model.panicked
