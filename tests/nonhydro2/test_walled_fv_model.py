"""The walled finite-volume nonhydro model (plan stage F4, FV-D4).

Explicit ``family="fv"`` on a walled (bounded, unmapped, unimmersed)
grid is now served: scalars on ``CellAvg``, the wall-normal velocity on
the Dirichlet-tagged interior faces, the pressure DCT-II on the
Neumann ``CellAvg`` origin. The acceptance gate is **bitwise parity**
with the walled *nodal* model — the 2nd-order FV and nodal stencils are
the same numbers (scoping study §1). The parity is *exact* eagerly (the
per-step operators are bit-identical); under jit the walled mixed
``Fourier ⊗ Cosine`` pressure solve fuses with a different XLA
float-op ordering than the nodal one, so a jitted multi-step trajectory
agrees only to machine precision (~1e-14, an XLA-fusion artifact — the
purely periodic all-Fourier solve stays exactly bit-identical). The
stratified mixed model (CellAvg buoyancy coupled to the tagged-face
velocity) assembles, ``w.to(b)`` resolves through the new
``("average", Inner(DIRICHLET))`` row, and the FV flux-form advection
conserves total buoyancy.
"""
import jax
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.nonhydro2.modules.advection import CenteredAdvection
from fridom.spatial.bc import BC
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.composed import Divergence
from fridom.spatial.operators.interp import LinearInterp
from fridom.spatial.operators.reconstruct import LinearReconstruction
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

N = 8
DT = 0.02
COMPONENTS = ("u", "v", "w", "b", "p")


def _walled_grid(walled):
    return Grid(tuple(
        IntervalMesh(N, (0.0, 1.0 if name == walled else 2 * np.pi),
                     periodic=(name != walled), name=name)
        for name in ("x", "y", "z")))


def _build(family, walled, advection):
    return nh.Model(
        coriolis=nh.FPlaneCoriolis(f0=1.5), grid=_walled_grid(walled),
        dt=DT, dsqr=2.0,
        stratification=nh.ConstantStratification(n2=3.0),
        advection=advection, family=family)


def _seed_pair(fv, nodal, seed=7):
    """Write identical random data onto both models (shapes match)."""
    rng = np.random.default_rng(seed)
    data = {c: rng.standard_normal(fv.state[c].data.shape)
            for c in ("u", "v", "w", "b")}
    fv.set_fields(**data)
    nodal.set_fields(**data)


WALLED = ["z", "y"]
ADVECTION = [pytest.param(False, id="linear"),
             pytest.param(True, id="nonlinear")]


# ================================================================
#  The FV model is finite-volume on a walled grid (FV-D2 option A)
# ================================================================
def test_walled_fv_state_is_finite_volume():
    model = _build("fv", "z", advection=True)
    # scalars on CellAvg^3
    for c in ("p", "b"):
        assert all(isinstance(f, CellAvg)
                   for f in model.state[c].function_space.bare.factors)
    # w: the wall-normal velocity on the Dirichlet interior faces,
    # transverse cell averages
    w = model.state["w"].function_space.bare
    wz = w.factor("z")
    assert isinstance(wz, NodalSpace)
    assert wz.node_set is NodeSet.INNER
    assert all(c is BC.DIRICHLET for c in wz.bc.components)
    assert isinstance(w.factor("x"), CellAvg)


# ================================================================
#  Gate 4 (a): the per-step operators are EXACTLY bitwise nodal
# ================================================================
@pytest.mark.parametrize("walled", WALLED)
@pytest.mark.parametrize("advection", ADVECTION)
def test_walled_fv_step_operators_are_bitwise_nodal_eager(
        walled, advection):
    # eagerly (no XLA fusion) the FV and nodal walled models share every
    # number: the unconstrained tendency (advection / Coriolis /
    # stratification, family-agnostic on the shared stencils) and the
    # pressure projection are bit-identical
    fv = _build("fv", walled, advection)
    nodal = _build("nodal", walled, advection)
    _seed_pair(fv, nodal)
    with jax.disable_jit():
        tend_fv = fv.tendency(fv.state, constraints=False)
        tend_nd = nodal.tendency(nodal.state, constraints=False)
        con_fv = fv.constrain(fv.state)
        con_nd = nodal.constrain(nodal.state)
    for c in ("u", "v", "w", "b"):
        np.testing.assert_array_equal(
            np.asarray(tend_fv[c].data), np.asarray(tend_nd[c].data),
            err_msg=f"unconstrained tendency {c!r} differs")
        np.testing.assert_array_equal(
            np.asarray(con_fv[c].data), np.asarray(con_nd[c].data),
            err_msg=f"projected state {c!r} differs")


# ================================================================
#  Gate 4 (b): a jitted 12-step trajectory matches to machine
#  precision (the walled mixed-transform solve fuses differently
#  under XLA; the purely periodic all-Fourier solve stays exact)
# ================================================================
@pytest.mark.parametrize("walled", WALLED)
@pytest.mark.parametrize("advection", ADVECTION)
def test_walled_fv_matches_nodal_over_12_steps(walled, advection):
    steps = 12
    fv = _build("fv", walled, advection)
    nodal = _build("nodal", walled, advection)
    _seed_pair(fv, nodal)
    fv.advance(steps)
    nodal.advance(steps)
    for c in COMPONENTS:
        # exact eagerly (asserted above); jitted, the mixed
        # Fourier x Cosine walled solve carries XLA fusion round-off
        assert float(np.abs(np.asarray(fv.state[c].data)
                            - np.asarray(nodal.state[c].data)).max()) < 1e-12


# ================================================================
#  Gate 5: the stratified mixed model (CellAvg b + tagged-face w)
# ================================================================
@pytest.mark.parametrize("walled", WALLED)
def test_walled_fv_stratified_assembles_and_runs(walled):
    # the F2-found blocker is closed: w.to(b) from Inner(DIRICHLET)
    # onto CellAvg resolves through the ("average", Inner) row, so the
    # mixed stratified model assembles and steps divergence-clean
    model = _build("fv", walled, advection=True)
    rng = np.random.default_rng(3)
    model.set_fields(**{c: rng.standard_normal(model.state[c].data.shape)
                        for c in ("u", "v", "w", "b")})
    model.advance(3)
    assert not model.panicked
    for c in COMPONENTS:
        assert np.isfinite(np.asarray(model.state[c].data)).all()
    div = Divergence()(VectorField(
        {c: model.state[c] for c in ("u", "v", "w")}))
    assert float(np.abs(np.asarray(div.data)).max()) < 1e-13


@pytest.mark.parametrize("walled", WALLED)
def test_walled_fv_conserves_total_buoyancy(walled):
    # FV flux form with an exact-zero wall flux: the advective b
    # tendency sums to machine zero over the uniform cells
    model = _build("fv", walled, advection=True)
    rng = np.random.default_rng(12)
    model.set_fields(**{c: rng.standard_normal(model.state[c].data.shape)
                        for c in ("u", "v", "w", "b")})
    state = model.constrain(model.state)
    tau = model.tendency(
        state, constraints=False,
        filter=fr.model.term_predicates.owned_by(CenteredAdvection))
    db = np.asarray(tau["b"].data)
    assert abs(float(np.sum(db))) < 1e-12 * float(np.sum(np.abs(db)))


def test_average_row_interior_bitwise_and_matches_nodal_interp():
    # the new ("average", Inner(DIRICHLET)) row (w.to(b)): interior
    # cells are bitwise the BC-free two-point mean and bitwise the
    # nodal LinearInterp; the wall cells use the claimed 0
    mz = IntervalMesh(N, (0.0, 1.0), periodic=False, name="z")
    grid = Grid((mz,))
    inner_dir = mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    w = grid.random.normal(inner_dir, seed=3)
    faces = np.asarray(w.data)
    recon = np.asarray(LinearReconstruction()["z"]._apply_factor(w, "z").data)
    # interior cells 1..n-2: plain mean of the two bracketing faces
    # (cell k = (face_{k-1} + face_k) / 2)
    interior = 0.5 * (faces[:-1] + faces[1:])
    np.testing.assert_array_equal(recon[1:-1], interior)
    # wall cells: (0 + face)/2 (the homogeneous Dirichlet claim)
    np.testing.assert_array_equal(recon[0], 0.5 * faces[0])
    np.testing.assert_array_equal(recon[-1], 0.5 * faces[-1])
    # and bitwise the nodal interp Inner(DIR) -> Center
    interp = np.asarray(LinearInterp()["z"]._apply_factor(w, "z").data)
    np.testing.assert_array_equal(recon, interp)
