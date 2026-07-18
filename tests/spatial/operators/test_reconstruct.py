"""Tests for fridom.spatial.operators.reconstruct."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import (
    MappedIntervalMesh,
)
from fridom.spatial.operators.base import EigenbasisError
from fridom.spatial.operators.interp import LinearInterp
from fridom.spatial.operators.reconstruct import (
    LinearDeconvolution,
    LinearReconstruction,
    fv_node_offset,
    wall_slots_addressable,
)
from fridom.spatial.operators.registry import OperatorRegistry
from fridom.spatial.operators.spectral import fourier_wavenumbers
from fridom.spatial.spaces.nodal import NodeSet


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def recon():
    return LinearReconstruction()


# ================================================================
#  Construction and static surface
# ================================================================
def test_dispatch_kind(recon):
    assert recon.dispatch_kind == "reconstruct"


def test_target_knob(recon):
    assert recon.target is None
    assert LinearReconstruction(
        target=NodeSet.OUTER).target is NodeSet.OUTER
    with pytest.raises(TypeError, match="NodeSet member"):
        LinearReconstruction(target="outer")


def test_requirements(recon, mx):
    assert recon.requirements(mx.cell_avg).halo == 1
    assert recon.requirements(mx.cell_avg).layout == "any"


# ================================================================
#  Fourier symbols (eigenvalues) — scoping study gap G1
# ================================================================
def _symbol_matches_operator(op, mesh, dom_space, seed=5):
    """Symbol in coeff space == coeff image of the physical output."""
    grid = Grid((mesh,), device_ids=(0,))
    axis = mesh.names[0]
    f = grid.random.normal(dom_space, seed=seed)
    t_in = grid.dispatch.resolve("transform", dom_space)
    t_out = grid.dispatch.resolve("transform", op.codomain(dom_space))
    fhat = t_in.forward(f)
    sym = op[axis].eigenvalues(grid, fhat.function_space.bare)
    got = sym(fhat).data
    want = t_out.forward(op[axis](f)).data
    return float(jnp.abs(got - want).max())


@pytest.mark.parametrize("n", [8, 9, 16, 17])
@pytest.mark.parametrize("dom", ["cell_avg", "right", "center",
                                 "face_avg"])
def test_symbol_matches_operator(recon, n, dom):
    # every second-order two-point conversion is a cos(k dx/2) diagonal
    mesh = IntervalMesh(n, (0.0, 1.3), name="x")
    assert _symbol_matches_operator(
        recon, mesh, getattr(mesh, dom)) < 1e-12


def test_symbol_is_the_one_hat_retagging_symbol(recon, mx):
    grid = Grid((mx,), device_ids=(0,))
    sym = recon["x"].eigenvalues(grid, mx.fourier(origin=mx.cell_avg))
    # retags Fourier(CellAvg) -> Fourier(Right)
    assert sym.space.origin is mx.cell_avg
    assert sym.codomain.origin is mx.right
    k = fourier_wavenumbers(mx.fourier(origin=mx.cell_avg))
    dx = mx.dx
    # cos(k dx/2) with the half-cell phase; NO sinc correction at O(2)
    expected = jnp.cos(k * dx / 2.0) * jnp.exp(1j * k * 0.5 * dx)
    assert jnp.allclose(sym.data, expected)
    # the Nyquist leaf is a structural zero (cos(π/2) = 0 exactly)
    assert sym.data.ravel()[-1] == 0.0


def test_symbol_matches_the_nodal_interp_numbers(recon, mx):
    # scoping study §1: at 2nd order the FV reconstruction is bitwise
    # the nodal LinearInterp two-point mean
    grid = Grid((mx,), device_ids=(0,))
    fv = recon["x"].eigenvalues(grid, mx.fourier(origin=mx.cell_avg))
    nodal = LinearInterp()["x"].eigenvalues(
        grid, mx.fourier(origin=mx.center))
    assert jnp.array_equal(fv.data, nodal.data)
    assert fv.codomain.origin is mx.right


def test_eigenvalues_thread_bare_or_fourier_factor(recon, mx):
    grid = Grid((mx,), device_ids=(0,))
    bare = recon["x"].eigenvalues(grid, mx.cell_avg)
    coeff = recon["x"].eigenvalues(grid, mx.fourier(origin=mx.cell_avg))
    assert coeff.space is bare.space
    assert coeff.codomain is bare.codomain
    assert jnp.array_equal(coeff.data, bare.data)


def test_codomain_retags_a_fourier_factor(recon, mx):
    assert recon.codomain(mx.fourier(origin=mx.cell_avg)) is (
        mx.fourier(origin=mx.right))
    assert recon.codomain(mx.fourier(origin=mx.center)) is (
        mx.fourier(origin=mx.face_avg))


def test_eigenvalues_raise_on_the_target_variant(mx):
    outer = LinearReconstruction(target=NodeSet.OUTER)
    with pytest.raises(EigenbasisError, match="target="):
        outer["x"].eigenvalues(Grid((mx,), device_ids=(0,)), mx.cell_avg)


def test_eigenvalues_raise_on_bounded_and_mapped(recon, my):
    with pytest.raises(EigenbasisError, match="periodic"):
        recon["y"].eigenvalues(Grid((my,), device_ids=(0,)), my.cell_avg)
    mapped = MappedIntervalMesh(
        8, (0.0, 1.0),
        lambda s: s + 0.1 * jnp.sin(2 * jnp.pi * s) / (2 * jnp.pi),
        periodic=True, name="w")
    with pytest.raises(EigenbasisError, match="periodic"):
        recon["w"].eigenvalues(
            Grid((mapped,), device_ids=(0,)), mapped.cell_avg)


# ================================================================
#  FV node offsets (window-alignment calculus)
# ================================================================
def test_fv_node_offsets(mx, my):
    assert fv_node_offset(mx.cell_avg) == 0.5
    assert fv_node_offset(mx.face_avg) == 1.0
    assert fv_node_offset(my.face_avg) == 1.0
    assert fv_node_offset(mx.center) == 0.5  # nodal delegation


# ================================================================
#  Per-factor signatures (codomain table)
# ================================================================
def test_codomain_periodic(recon, mx):
    assert recon.codomain(mx.cell_avg) is mx.right
    assert recon.codomain(mx.right) is mx.cell_avg
    assert recon.codomain(mx.center) is mx.face_avg
    assert recon.codomain(mx.face_avg) is mx.center


def test_codomain_bounded(recon, my):
    assert recon.codomain(my.cell_avg) is my.inner
    assert recon.codomain(my.outer) is my.cell_avg
    assert recon.codomain(my.inner) is my.cell_avg
    assert recon.codomain(my.center) is my.face_avg
    assert recon.codomain(my.face_avg) is my.center


def test_codomain_walled_dirichlet_inner(recon, my):
    # F4 claim-consuming tagged-face reconstruction: a Dirichlet
    # interior-face domain lands on the bare BC-free CellAvg
    inner_dir = my.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    assert recon.codomain(inner_dir) is my.cell_avg


def test_codomain_rejects_neumann_inner(recon, my):
    # a Neumann tag claims no wall value, so the face -> CellAvg average
    # cannot close at the walls
    inner_neu = my.nodal(NodeSet.INNER, bc=BC.NEUMANN)
    with pytest.raises(SpaceMismatchError, match="Neumann"):
        recon.codomain(inner_neu)


def test_apply_walled_dirichlet_inner_consumes_the_claim(recon, my):
    # the interior cells are bitwise the two-point mean; the wall cells
    # use the homogeneous Dirichlet zero
    grid = Grid((my,), device_ids=(0,))
    inner_dir = my.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    f = grid.random.normal(inner_dir, seed=8)
    out = recon["y"]._apply_factor(f, "y")
    assert out.function_space.bare is my.cell_avg
    faces = np.asarray(f.data)
    got = np.asarray(out.data)
    np.testing.assert_array_equal(got[1:-1], 0.5 * (faces[:-1] + faces[1:]))
    np.testing.assert_array_equal(got[0], 0.5 * faces[0])
    np.testing.assert_array_equal(got[-1], 0.5 * faces[-1])


def test_codomain_outer_variant(my, mx):
    outer = LinearReconstruction(target=NodeSet.OUTER)
    # the Outer wall faces need exterior values, which the (always
    # BC-free) CellAvg does not define (R1, boundary_plan.md); a
    # one-sided reconstruction variant is designed-for
    with pytest.raises(SpaceMismatchError, match="one-sided"):
        outer.codomain(my.cell_avg)
    with pytest.raises(SpaceMismatchError, match="target="):
        outer.codomain(mx.cell_avg)  # periodic mesh
    with pytest.raises(SpaceMismatchError, match="target="):
        outer.codomain(my.center)  # not CellAvg


def test_codomain_preserves_scalars(recon, mx):
    assert recon.codomain(mx.cell_avg.as_complex()) is (
        mx.right.as_complex())


def test_codomain_rejects_unlisted_node_sets(recon, mx):
    with pytest.raises(SpaceMismatchError,
                       match="no reconstruct signature"):
        recon.codomain(mx.left)


def test_codomain_rejects_bc_structured_spaces(recon, my):
    space = my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    with pytest.raises(SpaceMismatchError, match="BC-free"):
        recon.codomain(space)


# ================================================================
#  Application (two-point means over halo-extended storage)
# ================================================================
def test_periodic_cell_avg_to_right_converges(recon):
    errors = []
    for n in (16, 32):
        mesh = IntervalMesh(n, (0.0, 1.0), name="x")
        grid = Grid((mesh,), device_ids=(0,))
        f = grid.create_field(
            mesh.cell_avg, init=lambda x: jnp.sin(2 * jnp.pi * x))
        g = recon["x"](f)
        assert g.function_space.bare is mesh.right
        x = grid.evaluation_nodes(mesh.right).data
        errors.append(
            jnp.abs(g.data - jnp.sin(2 * jnp.pi * x)).max())
    assert errors[0] / errors[1] > 3.0  # second order


def test_right_to_cell_avg_is_exact_on_linears(recon, mx):
    # the cell average of a linear equals the mean of its face
    # values; the first cell consumes the periodic wrap ghost
    grid = Grid((mx,), device_ids=(0,))
    f = grid.create_field(mx.right, data=jnp.arange(8.0))
    g = recon["x"](f)
    assert g.function_space.bare is mx.cell_avg
    expected = jnp.concatenate(
        [jnp.array([(7.0 + 0.0) / 2]),
         (jnp.arange(7.0) + jnp.arange(1.0, 8.0)) / 2])
    assert jnp.allclose(g.data, expected)


def test_bounded_outer_to_cell_avg_is_exact_on_linears(recon, my):
    grid = Grid((my,), device_ids=(0,))
    f = grid.create_field(my.outer, init=lambda y: 3.0 * y - 1.0)
    g = recon["y"](f)
    assert g.function_space.bare is my.cell_avg
    y_c = grid.evaluation_nodes(my.cell_avg).data
    assert jnp.allclose(g.data, 3.0 * y_c - 1.0)


def test_bounded_center_to_face_avg_is_exact_on_linears(recon, my):
    grid = Grid((my,), device_ids=(0,))
    f = grid.create_field(my.center, init=lambda y: 2.0 * y + 1.0)
    g = recon["y"](f)
    assert g.function_space.bare is my.face_avg
    y_f = grid.evaluation_nodes(my.face_avg).data
    assert jnp.allclose(g.data, 2.0 * y_f + 1.0)


def test_outer_variant_is_ungrounded_under_r1(my):
    # the wall faces are undefined on the (always BC-free) CellAvg
    # (R1, boundary_plan.md): the application raises loudly; a
    # one-sided reconstruction variant is designed-for
    grid = Grid((my,), device_ids=(0,))
    outer = LinearReconstruction(target=NodeSet.OUTER)
    f = grid.create_field(my.cell_avg, init=lambda y: 4.0 * y)
    with pytest.raises(SpaceMismatchError, match="one-sided"):
        outer["y"](f)


def test_metadata_is_kept(recon, mx):
    grid = Grid((mx,), device_ids=(0,))
    f = grid.create_field(mx.cell_avg, name="q", units="kg")
    assert recon["x"](f).name == "q"  # same quantity


def test_reconstruct_needs_the_negotiated_halo(recon, mx):
    # the reach check is frame-independent (per-block stencil reach,
    # never storage bounds), so it raises identically on the
    # single-shard frame and on blocked multi-device frames
    bare = Grid((mx,), dispatch=OperatorRegistry({}),
                device_ids=(0,))  # halo 0
    f = bare.create_field(mx.cell_avg)
    with pytest.raises(ValueError, match="halo width 0"):
        recon["x"](f)


# ================================================================
#  Registry rows and the .to sugar
# ================================================================
def test_to_reconstructs_average_sources(mx):
    grid = Grid((mx,), device_ids=(0,))
    f = grid.create_field(mx.cell_avg, data=jnp.arange(8.0))
    g = f.to(mx.right)
    assert g.function_space.bare is mx.right


def test_to_averages_nodal_sources(mx):
    # nodal -> average resolves the seeded ("average", ...) rows
    grid = Grid((mx,), device_ids=(0,))
    f = grid.create_field(mx.right, data=jnp.arange(8.0))
    assert f.to(mx.cell_avg).function_space.bare is mx.cell_avg
    h = grid.create_field(mx.center, data=jnp.arange(8.0))
    assert h.to(mx.face_avg).function_space.bare is mx.face_avg


def test_to_rejects_dual_family_transfer(mx):
    # CellAvg -> FaceAvg needs the designed-for dual transfer; the
    # registered reconstruct codomain (Right) does not match
    grid = Grid((mx,), device_ids=(0,))
    f = grid.create_field(mx.cell_avg)
    with pytest.raises(SpaceMismatchError, match="lands on"):
        f.to(mx.face_avg)


def test_average_under_interpolate_kind(mx, my):
    # G4: the reconstruct instance is seeded under the interpolate kind
    # for the average family so composed._interp_onto can hop a
    # CellAvg/FaceAvg component; codomain is the staggering face
    grid = Grid((mx, my), device_ids=(0,))
    op = grid.dispatch.resolve("interpolate", mx.cell_avg)
    assert isinstance(op, LinearReconstruction)
    assert op.codomain(mx.cell_avg) is mx.right  # periodic
    assert op.codomain(my.cell_avg) is my.inner  # bounded interior
    assert op.codomain(mx.face_avg) is mx.center  # FaceAvg free


def test_average_interpolate_kind_converges():
    # G4 accuracy: the CellAvg -> face staggering hop is 2nd order on
    # periodic and bounded axes alike (the family's standing order).
    # A periodic seam needs a periodic field; the bounded interior
    # faces (Inner) take any smooth field (no wall stencil).
    def make_field(periodic):
        def field(x):
            base = jnp.sin(2 * jnp.pi * x)
            return base if periodic else base + 0.5 * x
        return field

    for periodic in (True, False):
        field = make_field(periodic)
        errors = []
        for n in (16, 32, 64):
            mesh = IntervalMesh(n, (0.0, 1.0), periodic=periodic,
                                name="x")
            grid = Grid((mesh,), device_ids=(0,))
            f = grid.create_field(mesh.cell_avg, init=field)
            op = grid.dispatch.resolve("interpolate", mesh.cell_avg)
            g = op["x"](f)
            face = mesh.right if periodic else mesh.inner
            assert g.function_space.bare is face
            xf = grid.evaluation_nodes(face).data
            errors.append(float(jnp.abs(g.data - field(xf)).max()))
        rates = [np.log2(errors[i] / errors[i + 1])
                 for i in range(len(errors) - 1)]
        assert min(rates) > 1.9  # second order


# ================================================================
#  LinearDeconvolution — co-located CellAvg <-> Center (G3)
# ================================================================
@pytest.fixture
def deconv():
    return LinearDeconvolution()


def test_deconvolution_dispatch_kind(deconv):
    assert deconv.dispatch_kind == "deconvolve"
    assert deconv is LinearDeconvolution()  # interned singleton


def test_deconvolution_requirements(deconv, mx):
    # a pass-through identity needs no halo
    assert deconv.requirements(mx.cell_avg).halo == 0
    assert deconv.requirements(mx.cell_avg).layout == "any"


def test_deconvolution_eigenvalues_designed_for(deconv, mx):
    with pytest.raises(EigenbasisError):
        deconv.eigenvalues(None, mx.cell_avg)


def test_deconvolution_codomain(deconv, mx, my):
    # the co-located primal pair, periodic and bounded (no shift, so no
    # exterior values — grounded on bounded axes too)
    assert deconv.codomain(mx.cell_avg) is mx.center
    assert deconv.codomain(mx.center) is mx.cell_avg
    assert deconv.codomain(my.cell_avg) is my.center
    assert deconv.codomain(my.center) is my.cell_avg


def test_deconvolution_preserves_scalars(deconv, mx):
    assert deconv.codomain(mx.cell_avg.as_complex()) is (
        mx.center.as_complex())


def test_deconvolution_rejects_shifted_and_dual(deconv, mx, my):
    # FaceAvg and the shifted nodal faces are the reconstruct kind's
    # job (or designed-for); the deconvolution grounds CellAvg<->Center
    for space in (mx.right, mx.face_avg, mx.left, my.inner, my.outer):
        with pytest.raises(SpaceMismatchError, match="CellAvg <-> "):
            deconv.codomain(space)


def test_deconvolution_rejects_bc_structured_center(deconv, my):
    space = my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    with pytest.raises(SpaceMismatchError, match="CellAvg <-> "):
        deconv.codomain(space)


def test_deconvolution_is_identity_retag(deconv, mx):
    # the data is untouched (a 2nd-order identity) on both directions
    grid = Grid((mx,), device_ids=(0,))
    f = grid.create_field(mx.cell_avg, data=jnp.arange(8.0), name="q")
    g = deconv["x"](f)
    assert g.function_space.bare is mx.center
    assert jnp.array_equal(g.data, f.data)
    assert g.name == "q"  # same quantity
    back = deconv["x"](g)
    assert back.function_space.bare is mx.cell_avg
    assert jnp.array_equal(back.data, f.data)


def test_deconvolution_works_on_bounded_axis(deconv, my):
    # co-located conversion needs no exterior values (R1 does not bite)
    grid = Grid((my,), device_ids=(0,))
    f = grid.create_field(my.cell_avg, data=jnp.arange(8.0))
    g = deconv["y"](f)
    assert g.function_space.bare is my.center
    assert jnp.array_equal(g.data, f.data)


def test_deconvolution_is_second_order_vs_midpoint():
    # the honest FV claim: a *true* cell average deconvolves to the
    # midpoint value at O(dx^2). Feed analytic cell averages (not the
    # collocation shortcut) and compare against the analytic midpoint.
    deconv = LinearDeconvolution()
    errors = []
    for n in (16, 32, 64, 128):
        mesh = IntervalMesh(n, (0.0, 1.0), name="x")
        grid = Grid((mesh,), device_ids=(0,))
        dx = 1.0 / n
        xl = np.arange(n) * dx
        # (1/dx) int_cell sin(2 pi x) dx
        cell_avg = -(np.cos(2 * np.pi * (xl + dx))
                     - np.cos(2 * np.pi * xl)) / (2 * np.pi) / dx
        f = grid.create_field(mesh.cell_avg,
                              data=jnp.asarray(cell_avg))
        g = deconv["x"](f)
        xm = grid.evaluation_nodes(mesh.center).data
        errors.append(float(jnp.abs(
            g.data - jnp.sin(2 * jnp.pi * xm)).max()))
    rates = [np.log2(errors[i] / errors[i + 1])
             for i in range(len(errors) - 1)]
    assert min(rates) > 1.9  # second order


def test_to_colocated_deconvolution_round_trips(mx):
    # .to reads the "deconvolve" kind for a co-located average<->nodal
    # pair (G3); the round trip is an exact identity
    grid = Grid((mx,), device_ids=(0,))
    p = grid.create_field(mx.cell_avg, data=jnp.arange(8.0))
    center = p.to(mx.center)
    assert center.function_space.bare is mx.center
    assert jnp.array_equal(center.data, p.data)
    assert jnp.array_equal(center.to(mx.cell_avg).data, p.data)


# ================================================================
#  One-sided CellAvg -> Outer reconstruction (G9, R2)
# ================================================================
@pytest.fixture
def outer():
    return LinearReconstruction(target=NodeSet.OUTER,
                                boundary="one_sided")


@pytest.fixture
def mw():
    # a smoothly stretched (non-uniform) bounded axis: the cell widths
    # genuinely differ, so the wall weights must be geometry-derived
    return MappedIntervalMesh(
        16, (0.0, 1.0),
        lambda s: s + 0.15 * jnp.sin(2 * jnp.pi * s) / (2 * jnp.pi),
        periodic=False, name="y")


def test_one_sided_boundary_knob(recon):
    assert recon.boundary == "closed"
    outer = LinearReconstruction(target=NodeSet.OUTER,
                                 boundary="one_sided")
    assert outer.boundary == "one_sided"
    # interned on (target, boundary): distinct knobs, distinct object
    assert outer is LinearReconstruction(
        target=NodeSet.OUTER, boundary="one_sided")
    assert outer is not LinearReconstruction(target=NodeSet.OUTER)
    with pytest.raises(ValueError, match="one_sided"):
        LinearReconstruction(boundary="extrapolate")


def test_one_sided_codomain_grounds_outer(outer, my, mw):
    # the opt-in reopens the BC-free bounded CellAvg -> Outer signature
    # on uniform and stretched axes alike (R2, boundary_plan.md 2d)
    assert outer.codomain(my.cell_avg) is my.outer
    assert outer.codomain(mw.cell_avg) is mw.outer
    assert outer.codomain(my.cell_avg.as_complex()) is (
        my.outer.as_complex())


def test_one_sided_codomain_still_rejects_off_target(outer, mx, my):
    # only bounded CellAvg -> Outer is grounded; periodic and non-CellAvg
    # domains raise the target= message even under the opt-in
    with pytest.raises(SpaceMismatchError, match="target="):
        outer.codomain(mx.cell_avg)  # periodic
    with pytest.raises(SpaceMismatchError, match="target="):
        outer.codomain(my.center)  # not CellAvg


def test_closed_outer_variant_hint_names_the_opt_in(my):
    # the default (closed) variant stays ungrounded under R1 but now
    # points the user at the one-sided opt-in
    closed = LinearReconstruction(target=NodeSet.OUTER)
    with pytest.raises(SpaceMismatchError, match="one_sided"):
        closed.codomain(my.cell_avg)


def test_one_sided_requirements_demand_a_local_axis(outer, recon, my):
    # the wall patches write static physical-edge indices
    req = outer.requirements(my.cell_avg)
    assert req.layout == "local"
    assert req.halo == 1
    assert recon.requirements(my.cell_avg).layout == "any"


def test_one_sided_outer_is_exact_on_constants_and_linears(outer, my):
    # every face (interior AND both walls) reproduces a linear exactly
    # at machine precision on a uniform bounded axis
    grid = Grid((my,), device_ids=(0,))
    for poly in (lambda y: 3.0 + 0.0 * y, lambda y: 2.0 * y - 1.0):
        f = grid.create_field(my.cell_avg, init=poly)
        g = outer["y"](f)
        assert g.function_space.bare is my.outer
        assert g.data.shape[0] == my.cell_avg.shape[0] + 1  # n + 1
        y_o = grid.evaluation_nodes(my.outer).data
        assert float(jnp.abs(g.data - poly(y_o)).max()) < 1e-13


def test_one_sided_wall_faces_are_linear_extrapolation(outer, my):
    # the two wall faces are the one-sided (3 c0 - c1) / 2 closure of
    # the two nearest interior cell averages (uniform axis)
    grid = Grid((my,), device_ids=(0,))
    f = grid.create_field(my.cell_avg,
                          init=lambda y: jnp.sin(1.3 * y) + 0.2 * y)
    g = outer["y"](f)
    c = f.data
    assert jnp.allclose(g.data[0], (3.0 * c[0] - c[1]) / 2.0)
    assert jnp.allclose(g.data[-1], (3.0 * c[-1] - c[-2]) / 2.0)


def test_one_sided_interior_matches_inner_bitwise(outer, recon, my):
    # the interior faces of the Outer result are the standard symmetric
    # CellAvg -> Inner reconstruction, bit for bit (consistency gate)
    grid = Grid((my,), device_ids=(0,))
    f = grid.create_field(my.cell_avg,
                          init=lambda y: jnp.sin(1.3 * y) + 0.2 * y)
    g = outer["y"](f)
    inner = recon["y"](f)
    assert inner.function_space.bare is my.inner
    assert jnp.array_equal(g.data[1:-1], inner.data)


def test_one_sided_outer_wall_converges(outer):
    # wall-face error converges at the design order (2) for a smooth
    # non-polynomial function on a uniform bounded axis
    errors = []
    for n in (16, 32, 64, 128):
        mesh = IntervalMesh(n, (0.0, 1.0), periodic=False, name="y")
        grid = Grid((mesh,), device_ids=(0,))
        f = grid.create_field(mesh.cell_avg,
                              init=lambda y: jnp.exp(jnp.sin(3.0 * y)))
        g = outer["y"](f)
        y_o = grid.evaluation_nodes(mesh.outer).data
        exact = jnp.exp(jnp.sin(3.0 * y_o))
        errors.append(max(float(jnp.abs(g.data[0] - exact[0])),
                          float(jnp.abs(g.data[-1] - exact[-1]))))
    rates = [np.log2(errors[i] / errors[i + 1])
             for i in range(len(errors) - 1)]
    assert min(rates) > 1.9


def test_one_sided_outer_on_a_stretched_axis(outer, recon, mw):
    # non-uniform geometry: the wall weights come from the cell widths,
    # so the walls stay exact on linears (constants exact everywhere);
    # the interior keeps the symmetric mean (bitwise vs CellAvg -> Inner)
    grid = Grid((mw,), device_ids=(0,))
    y_o = grid.evaluation_nodes(mw.outer).data
    const = grid.create_field(mw.cell_avg, init=lambda y: 5.0 + 0.0 * y)
    g_c = outer["y"](const)
    assert g_c.function_space.bare is mw.outer
    assert float(jnp.abs(g_c.data - 5.0).max()) < 1e-13  # exact all faces
    lin = grid.create_field(mw.cell_avg, init=lambda y: 2.0 * y - 1.0)
    g_l = outer["y"](lin)
    exact = 2.0 * y_o - 1.0
    # geometry-derived wall weights reproduce the linear at both walls
    assert float(jnp.abs(g_l.data[0] - exact[0])) < 1e-12
    assert float(jnp.abs(g_l.data[-1] - exact[-1])) < 1e-12
    # interior faces are the standard symmetric mean, bit for bit
    inner = recon["y"](lin)
    assert jnp.array_equal(g_l.data[1:-1], inner.data)


def test_one_sided_outer_stretched_wall_converges(outer):
    # design-order convergence at the walls on a stretched axis
    def mapping(s):
        return s + 0.15 * jnp.sin(2 * jnp.pi * s) / (2 * jnp.pi)
    errors = []
    for n in (16, 32, 64, 128):
        mesh = MappedIntervalMesh(n, (0.0, 1.0), mapping,
                                  periodic=False, name="y")
        grid = Grid((mesh,), device_ids=(0,))
        f = grid.create_field(mesh.cell_avg,
                              init=lambda y: jnp.exp(jnp.sin(3.0 * y)))
        g = outer["y"](f)
        y_o = grid.evaluation_nodes(mesh.outer).data
        exact = jnp.exp(jnp.sin(3.0 * y_o))
        errors.append(max(float(jnp.abs(g.data[0] - exact[0])),
                          float(jnp.abs(g.data[-1] - exact[-1]))))
    rates = [np.log2(errors[i] / errors[i + 1])
             for i in range(len(errors) - 1)]
    assert min(rates) > 1.9


def test_one_sided_outer_needs_enough_cells(outer):
    # the two-point wall stencil needs at least two cells
    mesh = IntervalMesh(1, (0.0, 1.0), periodic=False, name="y")
    grid = Grid((mesh,), device_ids=(0,))
    f = grid.create_field(mesh.cell_avg, init=lambda y: y)
    with pytest.raises(NotImplementedError, match="needs 2 cells"):
        outer["y"](f)


def test_one_sided_metadata_is_kept(outer, my):
    grid = Grid((my,), device_ids=(0,))
    f = grid.create_field(my.cell_avg, name="q", units="kg")
    assert outer["y"](f).name == "q"  # same quantity


def test_outer_variant_is_not_a_default_to_row(my):
    # the Outer codomain is reached per-instance (opt-in), never a
    # seeded row: .to(outer) resolves the default reconstruct (-> Inner)
    # and mismatches, exactly like the nodal target=OUTER interp
    grid = Grid((my,), device_ids=(0,))
    f = grid.create_field(my.cell_avg, init=lambda y: y)
    with pytest.raises(SpaceMismatchError, match="lands on"):
        f.to(my.outer)


# ================================================================
#  Storage-frame windowed walled-face reconstruction (step-gap fix)
# ================================================================
# The claim-consuming Inner(DIRICHLET) -> CellAvg reconstruction (the
# w.to(b) seam) has two byte-for-byte equivalent spellings
# (design/research/fv_nodal_step_gap.md): the storage-frame windowed
# fast path (impose the zero wall value in the ghost slots, run the
# ordinary window) and the true-frame fallback (unpad, pad the zero
# walls, interpolate, store). The fast path keeps the operand's
# periodic-axis halo claims, which the true-frame store() drops.
@pytest.fixture
def inner_dir(my):
    return my.nodal(NodeSet.INNER, bc=BC.DIRICHLET)


def test_walled_face_windowed_equals_true_frame(recon, my, inner_dir):
    # the load-bearing invariant: the two spellings agree bit for bit
    grid = Grid((my,), device_ids=(0,))
    f = grid.random.normal(inner_dir, seed=4)
    fast = recon["y"]._reconstruct_walled_face_windowed(f, "y")
    slow = recon["y"]._reconstruct_walled_face_true_frame(f, "y")
    assert fast.function_space.bare is slow.function_space.bare
    assert fast.function_space.bare is my.cell_avg
    # bitwise by construction; forced-CPU FP reassociation is the only
    # reason a tight allclose would be needed (backend gotcha)
    assert jnp.array_equal(fast.data, slow.data)


def test_walled_face_windowed_keeps_periodic_halo_claim(recon, mx, my):
    # the mechanism: the windowed path keeps the periodic-x claim, the
    # true-frame store() drops it; the bounded axis is consumed by both
    grid = Grid((mx, my), device_ids=(0,))
    space = mx.cell_avg * my.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    f0 = grid.random.normal(space, seed=2)
    f = type(f0)(f0.grid, f0.function_space, f0._data, f0.metadata,
                 halo_valid=HaloSpec({"x": 1, "y": 0}))
    fast = recon["y"]._reconstruct_walled_face_windowed(f, "y")
    slow = recon["y"]._reconstruct_walled_face_true_frame(f, "y")
    assert fast.halo_valid["x"] == 1
    assert slow.halo_valid["x"] == 0
    assert fast.halo_valid["y"] == 0


def test_walled_face_falls_back_when_walls_unaddressable(recon, my):
    # the empty-registry grid negotiates no halo: _apply_factor routes
    # the walled-face reconstruction to the true-frame fallback and
    # still lands the exact claim-consuming mean
    bare = Grid((my,), dispatch=OperatorRegistry({}), device_ids=(0,))
    inner = my.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    f = bare.random.normal(inner, seed=7)
    assert wall_slots_addressable(f, "y") is False
    out = recon["y"]._apply_factor(f, "y")
    direct = recon["y"]._reconstruct_walled_face_true_frame(f, "y")
    assert out.function_space.bare is my.cell_avg
    assert jnp.array_equal(out.data, direct.data)
    faces = np.asarray(f.data)
    got = np.asarray(out.data)
    np.testing.assert_array_equal(got[0], 0.5 * faces[0])
    np.testing.assert_array_equal(got[-1], 0.5 * faces[-1])


def test_walled_face_windowed_grad_is_finite_and_matches_fd(
        recon, my, inner_dir):
    # reverse-mode gate (AGENTS.md diff policy): the windowed walled-face
    # reconstruction is a linear two-point mean over the wall-zeroed
    # storage (no divide/sqrt), so jax.grad is finite and matches a
    # central FD -- the storage-frame respelling keeps it so.
    def loss(c):
        grid = Grid((my,), device_ids=(0,))
        f = grid.random.normal(inner_dir, seed=8) * c
        return jnp.sum(recon["y"](f).data ** 2)

    c0 = 1.3
    grad = float(jax.grad(loss)(c0))
    assert bool(jnp.isfinite(grad))
    assert grad != 0.0
    h = 1e-4
    fd = float((loss(c0 + h) - loss(c0 - h)) / (2.0 * h))
    assert abs(grad - fd) <= 1e-4 * abs(fd)
