"""Parity of WENO selected-input reconstruction vs both-then-select.

The load-bearing check of nonhydro2's one-pass WENO upwind face value
(`_SelectedFaceReconstruction`, advection.py): a SINGLE left-biased
reconstruction of the sign-selected union window reproduces the
"reconstruct BOTH biases, then ``Where``-select" face value to
reversed-summation ulps (bitwise where the face velocity is positive),
across BOTH reconstruction families and every C-grid flux direction —
the nodal tracer ``Center -> Right``, the dual velocity self-advection
``Right -> Center`` (the ``_wall_shift`` case), and the average-family
``CellAvg -> Right | Inner`` FV tracer (the primal cell frame, shift 0,
no dual direction) — both grounded orders, and periodic and z-walled
grids. The ``v == 0`` tie takes the right-biased side (old-stack
parity). The FV union alignment is the nodal primal one, validated at
machine precision here and in the CPU oracle
(``design/research/stencil_lowering/microbench/phase3_prep/``).

Self-contained per the oversized-module test convention (the
``test_advection*`` shards share nothing): the small grid/field
builders below are duplicated, not imported.
"""

import jax
import numpy as np
import pytest
from jax.extend.core import ClosedJaxpr, Jaxpr

import fridom as fr
from fridom.model.model import Model as FrModel
from fridom.model.modules.advection import (
    UpwindAdvection,
    WENOAdvection,
    _BiasedFaceReconstruction,
    _FVBiasedReconstruction,
    _SelectedFaceReconstruction,
)
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
)
from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.select import Where
from fridom.spatial.spaces.average import AverageSpace
from fridom.spatial.spaces.nodal import NodeSet
from fridom.spatial.spaces.tensor_product import TensorProductSpace

L = 2 * np.pi
NR = 16   # cells along the reconstruction axis
NT = 8    # tangential cells


# ================================================================
#  Cases: the two C-grid directions on periodic and z-walled grids
# ================================================================
def _periodic_tracer(order):
    """Center -> Right along x (shift 0), periodic."""
    mx = IntervalMesh(NR, (0.0, L), name="x")
    my = IntervalMesh(NT, (0.0, L), name="y")
    mz = IntervalMesh(NT, (0.0, L), name="z")
    grid = Grid((mx, my, mz), device_ids=(0,))
    grid.negotiate(halo=HaloSpec({"x": order // 2 + 1}))
    domain = TensorProductSpace.of(mx.center, my.center, mz.center)
    q = grid.create_field(
        domain,
        init=lambda x, y, z: np.sin(x) + 0.3 * np.cos(2 * x)
        + 0.1 * np.sin(y) + 0.1 * np.cos(z))
    return grid, q, "x", "none"


def _periodic_velocity(order):
    """Right -> Center along x (shift 1, self-advection), periodic."""
    mx = IntervalMesh(NR, (0.0, L), name="x")
    my = IntervalMesh(NT, (0.0, L), name="y")
    mz = IntervalMesh(NT, (0.0, L), name="z")
    grid = Grid((mx, my, mz), device_ids=(0,))
    grid.negotiate(halo=HaloSpec({"x": order // 2 + 1}))
    domain = TensorProductSpace.of(mx.right, my.center, mz.center)
    q = grid.create_field(
        domain,
        init=lambda x, y, z: np.sin(x) + 0.3 * np.cos(2 * x)
        + 0.1 * np.sin(y) + 0.1 * np.cos(z))
    return grid, q, "x", "none"


def _walled_tracer(order):
    """Center -> Inner along z (shift 0), z-walled (graded)."""
    mx = IntervalMesh(NT, (0.0, L), name="x")
    my = IntervalMesh(NT, (0.0, L), name="y")
    mz = IntervalMesh(NR, (0.0, 1.0), periodic=False, name="z")
    grid = Grid((mx, my, mz), device_ids=(0,))
    grid.negotiate(halo=HaloSpec({"z": order // 2 + 1}))
    domain = TensorProductSpace.of(mx.center, my.center, mz.center)
    q = grid.create_field(
        domain,
        init=lambda x, y, z: np.cos(np.pi * z) + 0.2 * np.sin(2 * z)
        + 0.1 * np.sin(x) + 0.1 * np.cos(y))
    return grid, q, "z", "graded"


def _walled_velocity(order):
    """Inner -> Center along z (shift 1, self-advection), z-walled."""
    mx = IntervalMesh(NT, (0.0, L), name="x")
    my = IntervalMesh(NT, (0.0, L), name="y")
    mz = IntervalMesh(NR, (0.0, 1.0), periodic=False, name="z")
    grid = Grid((mx, my, mz), device_ids=(0,))
    grid.negotiate(halo=HaloSpec({"z": order // 2 + 1}))
    domain = TensorProductSpace.of(
        mx.center, my.center,
        mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET))
    # the Inner-Dirichlet operand must vanish at both walls: the z
    # factor does, so any tangential modulation keeps it wall-vanishing
    q = grid.create_field(
        domain,
        init=lambda x, y, z: np.sin(np.pi * z)
        * (1.0 + 0.4 * np.cos(3 * np.pi * z))
        * (1.0 + 0.1 * np.sin(x) + 0.1 * np.cos(y)))
    return grid, q, "z", "graded"


def _periodic_fv_tracer(order):
    """CellAvg -> Right along x (shift 0, FV tracer), periodic."""
    mx = IntervalMesh(NR, (0.0, L), name="x")
    my = IntervalMesh(NT, (0.0, L), name="y")
    mz = IntervalMesh(NT, (0.0, L), name="z")
    grid = Grid((mx, my, mz), device_ids=(0,))
    grid.negotiate(halo=HaloSpec({"x": order // 2 + 1}))
    domain = TensorProductSpace.of(mx.cell_avg, my.center, mz.center)
    q = grid.create_field(
        domain,
        init=lambda x, y, z: np.sin(x) + 0.3 * np.cos(2 * x)
        + 0.1 * np.sin(y) + 0.1 * np.cos(z))
    return grid, q, "x", "none"


def _walled_fv_tracer(order):
    """CellAvg -> Inner along z (shift 0, FV tracer), z-walled."""
    mx = IntervalMesh(NT, (0.0, L), name="x")
    my = IntervalMesh(NT, (0.0, L), name="y")
    mz = IntervalMesh(NR, (0.0, 1.0), periodic=False, name="z")
    grid = Grid((mx, my, mz), device_ids=(0,))
    grid.negotiate(halo=HaloSpec({"z": order // 2 + 1}))
    domain = TensorProductSpace.of(mx.center, my.center, mz.cell_avg)
    q = grid.create_field(
        domain,
        init=lambda x, y, z: np.cos(np.pi * z) + 0.2 * np.sin(2 * z)
        + 0.1 * np.sin(x) + 0.1 * np.cos(y))
    return grid, q, "z", "graded"


CASES = [
    pytest.param(_periodic_tracer, id="periodic-tracer"),
    pytest.param(_periodic_velocity, id="periodic-velocity"),
    pytest.param(_walled_tracer, id="walled-tracer"),
    pytest.param(_walled_velocity, id="walled-velocity"),
    pytest.param(_periodic_fv_tracer, id="periodic-fv-tracer"),
    pytest.param(_walled_fv_tracer, id="walled-fv-tracer"),
]


# ================================================================
#  Helpers
# ================================================================
def _broadcast(line, shape, axis_index):
    """Broadcast a 1-D recon-axis profile over the full face shape."""
    index = [None] * len(shape)
    index[axis_index] = slice(None)
    return line[tuple(index)] * np.ones(shape)


def _apply(build, order, sign):
    """Run both spellings; return their data plus the biased pair's.

    ``sign`` is a per-face recon-axis profile of the face velocity.
    Returns ``(new, ref, left, right, axis_index)`` as host arrays.
    The reconstruction family follows ``q``'s space along ``axis``: an
    average (``CellAvg``) factor selects the FV pair and the FV variant
    of the selected kernel, a nodal factor the nodal pair.
    """
    grid, q, axis, boundary = build(order)
    average = isinstance(q.function_space.bare.factor(axis),
                         AverageSpace)
    recon_cls = (_FVBiasedReconstruction if average
                 else _BiasedFaceReconstruction)
    left = recon_cls(order, "left", "weno", boundary)
    right = recon_cls(order, "right", "weno", boundary)
    selected = _SelectedFaceReconstruction(
        order, boundary, family=("fv" if average else "nodal"))
    codomain = left[axis](q).function_space
    axis_index = codomain.bare.names.index(axis)
    v = grid.create_field(
        codomain,
        data=_broadcast(sign(codomain.shape[axis_index]),
                        codomain.shape, axis_index))
    positive = v + abs(v)
    ref = Where()(positive,
                  left[axis](q).retag(codomain),
                  right[axis](q).retag(codomain))
    new = selected(positive, q, axis, codomain)
    return (np.asarray(new.data), np.asarray(ref.data),
            np.asarray(left[axis](q).retag(codomain).data),
            np.asarray(right[axis](q).retag(codomain).data),
            axis_index)


def _mixed_sign(m):
    # a smooth sign-changing profile with no exact zero (offset 0.3)
    return np.cos(2 * np.pi * (np.arange(m) + 0.3) / m)


# ================================================================
#  Parity: mixed-sign velocity to machine precision
# ================================================================
@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("build", CASES)
def test_selected_matches_both_then_select_mixed_sign(build, order):
    # the whole point: one reconstruction of the sign-selected union
    # window == both biased recons then Where-select, to reversed-
    # summation ulps (exact algebra modulo FP reassociation)
    new, ref, _left, _right, _ai = _apply(build, order, _mixed_sign)
    assert np.isfinite(new).all()
    assert np.abs(new - ref).max() <= 1e-13


# ================================================================
#  Parity: a uniformly positive velocity is bitwise the left recon
# ================================================================
@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("build", CASES)
def test_selected_is_bitwise_where_velocity_is_positive(build, order):
    # where v > 0 the taps are the left window untouched, so the
    # selected value is the plain left reconstruction bit-for-bit —
    # and so is Where(positive, left, right)
    new, ref, left, _right, _ai = _apply(
        build, order, lambda m: np.full(m, 1.7))
    assert np.array_equal(new, ref)
    assert np.array_equal(new, left)


# ================================================================
#  The v == 0 tie takes the right-biased side (old-stack parity)
# ================================================================
@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("build", CASES)
def test_zero_velocity_tie_takes_the_right_biased_side(build, order):
    # a single recon-axis face with v == 0 exactly: positive =
    # v + |v| = 0 there, so the face reads the RIGHT-biased value
    tie = NR // 2

    def sign(m):
        line = _mixed_sign(m)
        line[tie] = 0.0
        return line

    new, _ref, left, right, axis_index = _apply(build, order, sign)
    index = [slice(None)] * new.ndim
    index[axis_index] = tie
    face = tuple(index)
    # the tie reads the right recon (to reversed-summation ulps) ...
    assert np.abs(new[face] - right[face]).max() <= 1e-13
    # ... and is genuinely NOT the left recon (they differ on this
    # non-symmetric window, so the tie-break direction is pinned)
    assert np.abs(new[face] - left[face]).max() > 1e-9


# ================================================================
#  The operator's structure (interning, delegated signature)
# ================================================================
def test_selected_operator_properties_and_interning():
    op = _SelectedFaceReconstruction(5, "graded", "centered2")
    assert (op.order, op.boundary, op.wall) == (5, "graded",
                                                "centered2")
    assert op._family == "nodal"
    # D6 interning: structurally-equal requests are the same object
    assert op is _SelectedFaceReconstruction(5, "graded", "centered2")
    assert op is not _SelectedFaceReconstruction(3, "graded",
                                                 "centered2")
    assert op is not _SelectedFaceReconstruction(5, "none",
                                                 "centered2")
    # the average-family (FV) twin interns distinctly on the key
    fv = _SelectedFaceReconstruction(5, "graded", "centered2",
                                     family="fv")
    assert fv._family == "fv"
    assert fv is not op
    assert fv is _SelectedFaceReconstruction(5, "graded", "centered2",
                                             family="fv")


def test_selected_operator_delegates_signature_to_left_recon():
    # the union window's frame is the left reconstruction's, so the
    # codomain and the halo demand are exactly that kernel's — per
    # family (nodal Center/Right, and the FV CellAvg twin)
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    for order, halo in ((3, 2), (5, 3)):
        op = _SelectedFaceReconstruction(order)
        ref = _BiasedFaceReconstruction(order, "left", "weno")
        assert op.codomain(mx.center) is ref.codomain(mx.center)
        assert op.codomain(mx.right) is ref.codomain(mx.right)
        assert op.requirements(mx.center).halo == halo
        fv = _SelectedFaceReconstruction(order, family="fv")
        fvref = _FVBiasedReconstruction(order, "left", "weno")
        assert fv.codomain(mx.cell_avg) is fvref.codomain(mx.cell_avg)
        assert fv.requirements(mx.cell_avg).halo == halo


# ================================================================
#  WENOAdvection wiring (the trace branch + the module overrides)
# ================================================================
def _weno_model(order, *, walled=False):
    """Build a minimal nh WENO model (periodic or z-walled grid)."""
    if walled:
        meshes = (IntervalMesh(NT, (0.0, L), name="x"),
                  IntervalMesh(NT, (0.0, L), name="y"),
                  IntervalMesh(NR, (0.0, 1.0), periodic=False,
                               name="z"))
    else:
        meshes = (IntervalMesh(NR, (0.0, L), name="x"),
                  IntervalMesh(NT, (0.0, L), name="y"),
                  IntervalMesh(NT, (0.0, L), name="z"))
    advection = WENOAdvection(order)
    model = FrModel(
        grid=Grid(meshes),
        modules=(DynamicalCore(), ConstantStratification(n2=1.0),
                 advection),
        time_stepper=AdamBashforth(0.01, order=3))
    return model, advection


def _advection_tendency(model):
    return model.tendency(
        model.state, constraints=False,
        filter=fr.model.term_predicates.owned_by(WENOAdvection))


@pytest.mark.parametrize("order", [3, 5])
def test_weno_model_installs_the_selected_kernel_periodic(order):
    # building the model runs the halo trace (the tracer-delegation
    # branch of the operator) and installs the periodic selected kernel
    model, advection = _weno_model(order)
    assert advection._selected.order == order
    assert advection._selected.boundary == "none"
    # the average-family twin is installed alongside, same structure
    assert advection._fv_selected.order == order
    assert advection._fv_selected.boundary == "none"
    assert advection._fv_selected._family == "fv"
    # constancy preservation exercises the real _face_value path
    ones = np.ones((NR, NT, NT))
    model.set_fields(u=1.5 * ones, v=-0.5 * ones, w=0.25 * ones,
                     b=3.3 * ones)
    tau = _advection_tendency(model)
    for name in ("u", "v", "w", "b"):
        assert np.abs(np.asarray(tau[name].data)).max() < 1e-13


def test_weno_model_installs_the_graded_selected_kernel_walled():
    # a z-walled grid swaps the selected kernel for its graded variant
    # at bind (order / wall matched to the biased pair it supersedes)
    model, advection = _weno_model(5, walled=True)
    assert advection._selected.boundary == "graded"
    assert advection._selected.wall == advection.wall
    assert advection._fv_selected.boundary == "graded"
    assert advection._fv_selected.wall == advection.wall
    assert advection._walled == ("z",)
    ones = np.ones((NT, NT, NR))
    zc = (np.arange(NR) + 0.5) / NR
    # w is Inner-Dirichlet on the walled z (fewer DOFs); leave it zero
    model.set_fields(u=0.3 * ones,
                     b=np.sin(np.pi * zc)[None, None, :] * ones)
    tau = _advection_tendency(model)
    for name in ("u", "v", "w", "b"):
        assert np.isfinite(np.asarray(tau[name].data)).all()


def test_face_value_dispatches_selected_kernel_by_family():
    # `_face_value` mirrors the base `_biased_pair` routing: a CellAvg
    # tracer runs through the FV selected kernel, a nodal field through
    # the nodal one — pinned bitwise against the kernels themselves
    _, advection = _weno_model(3)
    assert advection._fv_selected is not advection._selected
    cases = ((_periodic_tracer, _BiasedFaceReconstruction,
              advection._selected),
             (_periodic_fv_tracer, _FVBiasedReconstruction,
              advection._fv_selected))
    for build, recon_cls, kernel in cases:
        grid, q, axis, _ = build(3)
        left = recon_cls(3, "left", "weno")
        flux_space = left[axis](q).function_space
        axis_index = flux_space.bare.names.index(axis)
        v = grid.create_field(
            flux_space,
            data=_broadcast(_mixed_sign(flux_space.shape[axis_index]),
                            flux_space.shape, axis_index))
        got = advection._face_value(q, v, axis, flux_space)
        want = kernel(v + abs(v), q, axis, flux_space)
        assert np.array_equal(np.asarray(got.data),
                              np.asarray(want.data))


# ================================================================
#  The selected-input performance guard (jaxpr divide count)
# ================================================================
def _count_div_primitives(jaxpr):
    """Count lax ``div`` primitives in a jaxpr.

    Descends into every nested sub-jaxpr (pjit / cond / scan closures
    carry their bodies as equation params), so a divide hidden behind
    a jitted operator is still counted.
    """
    if isinstance(jaxpr, ClosedJaxpr):
        jaxpr = jaxpr.jaxpr
    total = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "div":
            total += 1
        for value in eqn.params.values():
            items = value if isinstance(value, (tuple, list)) else (value,)
            for item in items:
                if isinstance(item, (Jaxpr, ClosedJaxpr)):
                    total += _count_div_primitives(item)
    return total


def _face_value_div_counts(build, recon_cls, order):
    """``div`` counts for one isolated WENO face value, both ways.

    Traces the face-value computation with ``jax.make_jaxpr`` over the
    tracer field data (``with_data``), once via the WENO
    selected-input override as production dispatches it, and once via
    the inherited ``UpwindAdvection._face_value`` both-then-select base
    invoked explicitly on the same operands. Returns
    ``(selected, both)``.
    """
    _, advection = _weno_model(order)            # a periodic WENO model
    grid, q, axis, _ = build(order)
    left = recon_cls(order, "left", "weno")
    flux_space = left[axis](q).function_space
    axis_index = flux_space.bare.names.index(axis)
    v = grid.create_field(
        flux_space,
        data=_broadcast(_mixed_sign(flux_space.shape[axis_index]),
                        flux_space.shape, axis_index))

    def selected(qd):
        return WENOAdvection._face_value(
            advection, q.with_data(qd), v, axis, flux_space).data

    def both(qd):
        return UpwindAdvection._face_value(
            advection, q.with_data(qd), v, axis, flux_space).data

    return (_count_div_primitives(jax.make_jaxpr(selected)(q.data)),
            _count_div_primitives(jax.make_jaxpr(both)(q.data)))


@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize(
    ("build", "recon_cls"),
    [pytest.param(_periodic_tracer, _BiasedFaceReconstruction,
                  id="nodal"),
     pytest.param(_periodic_fv_tracer, _FVBiasedReconstruction,
                  id="fv")])
def test_selected_input_halves_the_divide_count(build, recon_cls,
                                                order):
    # The load-bearing PERFORMANCE guard (perf_guard_plan.md gap C):
    # the selected-input override reconstructs ONE sign-selected window,
    # running the WENO nonlinear weights once, so its jaxpr carries
    # ~half the ``div`` primitives of the inherited both-then-select
    # base (which reconstructs both biases, then ``Where``-selects).
    # The two spellings are bitwise-identical where v > 0, so the parity
    # tests above CANNOT see a silent revert to both-then-select -- the
    # divide count can. Both sides are measured here (never the historic
    # 253/493 hardcoded), so only the RATIO is pinned and the guard
    # survives unrelated arithmetic changes. On these periodic (interior
    # only) rows the halving is clean; a revert makes selected == both
    # and trips ``selected < 0.6 * both`` with margin on either side.
    selected, both = _face_value_div_counts(build, recon_cls, order)
    assert both > 0
    assert selected <= both / 2 + 4
    assert selected < 0.6 * both
