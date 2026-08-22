r"""Biased/WENO advection on stretched (``MappedIntervalMesh``) axes.

Route (ii) of the high-order mapped plan (design record
``research/nonuniform_weno_survey.md``): the biased face
reconstructions are built from the factor's own cell widths
(``weno.cell_widths`` -> the ``co_operands`` / ``co_storages`` seams),
so a stretched axis is no longer refused at bind. The load-bearing
checks here are

- **polynomial exactness** of every face kernel on a stretched
  lattice, in all four C-grid directions and both biases: it pins the
  width FRAME (primal vs dual cells, and the two wall half cells the
  graded ladder reads through the ``co_storages`` seam) far more
  sharply than a convergence rate does;
- the FV family's restored **design order** (5 / 3) on a stretched
  periodic axis, and the nodal family's honest **2nd order** there
  (the Shu-Osher FD identity needs a uniform lattice; see
  `test_stretched_nodal_transport_stays_second_order`);
- **constancy** and **conservation**, which route (ii) must leave
  exactly where they were;
- the one-pass selected WENO kernel's parity with both-then-select
  under the width tap-select;
- **uniform-grid parity**: a uniform factor takes the static float
  tables, bitwise.

Self-contained per the oversized-module shard convention (the
``test_advection*`` shards share nothing): the small builders below
are duplicated, not imported.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.modules.advection import (
    CenteredAdvection,
    UpwindAdvection,
    WENOAdvection,
    _BiasedFaceReconstruction,
    _CenteredFaceInterpolation,
    _face_widths,
    _FVBiasedReconstruction,
    _SelectedFaceReconstruction,
)
from fridom.model.modules.tracer import Tracer
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import (
    MappedIntervalMesh,
)
from fridom.spatial.operators.select import Where
from fridom.spatial.spaces.average import AverageSpace
from fridom.spatial.spaces.nodal import NodeSet
from fridom.spatial.spaces.tensor_product import TensorProductSpace

DT = 1e-3
NT = 4          # tangential cells of the 3-D fixtures
STRETCH = 1.5   # tanh column steepness (the coastal-upwelling shape)


# ================================================================
#  Maps and meshes
# ================================================================
def wavy(s):
    """Smooth wavy stretching of the unit computational interval."""
    return s + 0.1 * jnp.sin(2.0 * jnp.pi * s) / (2.0 * jnp.pi)


def wavy_np(s):
    """Host twin of `wavy` (the test's own geometry)."""
    return s + 0.1 * np.sin(2.0 * np.pi * s) / (2.0 * np.pi)


def tanh_map(s):
    """Two-sided tanh column (the coastal-upwelling map shape)."""
    return (jnp.tanh(STRETCH * (2.0 * s - 1.0)) / jnp.tanh(STRETCH)
            + 1.0) / 2.0


def tanh_np(s):
    """Host twin of `tanh_map`."""
    return (np.tanh(STRETCH * (2.0 * s - 1.0)) / np.tanh(STRETCH)
            + 1.0) / 2.0


def stretched_mesh(n, *, periodic=True, name="z"):
    """Build a stretched 1-D mesh (wavy periodic, tanh walled)."""
    return MappedIntervalMesh(
        n, (0.0, 1.0), wavy if periodic else tanh_map,
        periodic=periodic, name=name)


def face_positions(n, *, periodic=True):
    """Host face positions of `stretched_mesh`."""
    smap = wavy_np if periodic else tanh_np
    return smap(np.arange(n + 1) / n)


# ================================================================
#  The smooth transport profile (design-order studies)
# ================================================================
TWO_PI = 2.0 * np.pi


def profile(x):
    """Sample the smooth transport profile."""
    return np.sin(TWO_PI * x) + 0.3 * np.cos(2 * TWO_PI * x + 1.0)


def profile_primitive(x):
    """Return an antiderivative of `profile`."""
    return (-np.cos(TWO_PI * x) / TWO_PI
            + 0.3 * np.sin(2 * TWO_PI * x + 1.0) / (2 * TWO_PI))


def profile_prime(x):
    """Return the derivative of `profile`."""
    return (TWO_PI * np.cos(TWO_PI * x)
            - 0.6 * TWO_PI * np.sin(2 * TWO_PI * x + 1.0))


# ================================================================
#  Polynomial helpers (the exactness oracle)
# ================================================================
def poly(coeffs):
    """Return the polynomial with these ascending coefficients."""
    return lambda x: sum(c * x ** i for i, c in enumerate(coeffs))


def antideriv(coeffs):
    """Its antiderivative (vanishing at 0)."""
    return lambda x: sum(c * x ** (i + 1) / (i + 1)
                         for i, c in enumerate(coeffs))


def cell_averages(coeffs, edges):
    """Exact averages of ``poly(coeffs)`` over the ``edges`` cells."""
    prim = antideriv(coeffs)
    return (prim(edges[1:]) - prim(edges[:-1])) / np.diff(edges)


def wall_free_quadratic(lo, first, last, hi):
    """Return a quadratic whose wall half-cell averages vanish.

    Description
    -----------
    The ``Inner -> Center`` operand's wall DOFs are exact Dirichlet
    zeros, so an exactness oracle for that direction must be a
    polynomial whose averages over the two wall half cells
    ``[lo, first]`` and ``[last, hi]`` are zero. Two linear
    constraints on ``a + b x + x**2`` fix ``(a, b)``.
    """
    matrix = np.array([
        [first - lo, (first ** 2 - lo ** 2) / 2],
        [hi - last, (hi ** 2 - last ** 2) / 2]])
    rhs = -np.array([(first ** 3 - lo ** 3) / 3,
                     (hi ** 3 - last ** 3) / 3])
    a, b = np.linalg.solve(matrix, rhs)
    return (a, b, 1.0)


# ================================================================
#  A toy 3-D C-grid core (no pressure solve: the spectral projection
#  of nh.Model refuses a MappedIntervalMesh, C2)
# ================================================================
class Core(Module):

    """Toy core: the three C-grid velocities and a tracer."""

    field_declarations = (
        fr.model.FieldDeclaration.velocity(
            "u", "x", space=fr.spatial.Staggered("x")),
        fr.model.FieldDeclaration.velocity(
            "v", "y", space=fr.spatial.Staggered("y")),
        fr.model.FieldDeclaration.velocity(
            "w", "z", space=fr.spatial.Staggered("z")),
        fr.model.FieldDeclaration.tracer("b"),
    )

    @fr.model.term(advances=("u", "v", "w", "b"), linear=True,
                   transports=("u", "v", "w", "b"))
    def zero(self, state, _ctx):
        """Return a zero tendency (advection carries the physics)."""
        return {name: 0.0 * state[name]
                for name in ("u", "v", "w", "b")}


def stretched_grid(n=8, *, periodic=True, immersed=False):
    """Build a 3-D grid whose ``z`` factor is stretched."""
    kwargs = {}
    if immersed:
        kwargs["immersed"] = ImmersedDomain(
            lambda x, y, z: ((x > 0.2) & (x < 0.8)
                             & (y > 0.2) & (y < 0.8)
                             & (z > 0.2) & (z < 0.8)).astype(float))
    return Grid((
        IntervalMesh(NT, (0.0, 1.0), name="x"),
        IntervalMesh(NT, (0.0, 1.0), name="y"),
        stretched_mesh(n, periodic=periodic),
    ), **kwargs)


def toy_model(advection, *, n=8, periodic=True, extra=()):
    """Assemble the toy core plus ``advection`` on a stretched grid."""
    return Model(grid=stretched_grid(n, periodic=periodic),
                 modules=(Core(), *extra, advection),
                 time_stepper=AdamBashforth(DT, order=3))


def advection_tendency(model, cls):
    """Return the advection module's own tendency contribution."""
    return model.tendency(
        model.state, constraints=False,
        filter=fr.model.term_predicates.owned_by(cls))


def random_state(model, seed, names=("u", "v", "w", "b")):
    """Seed every prognostic field with smooth-scale noise."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        name: rng.standard_normal(model.state[name].data.shape)
        for name in names})


# ================================================================
#  1. Polynomial exactness: the width FRAME of every direction
# ================================================================
#: (direction id, bias) cases of `test_face_kernels_are_exact_on_a
#: _polynomial`; ``shift = 1`` only on the dual bounded direction
EXACT_N = 12


def _away_from_the_seam(n, reach=3):
    """Mark the faces whose window does not cross the periodic wrap.

    Description
    -----------
    The polynomial oracle is not periodic, so a window reading the
    wrap fill sees a different function; those faces carry no claim.
    """
    index = np.arange(n)
    return np.where((index >= reach) & (index < n - reach), 9, 0)


def _exactness_case(direction, bias, order=5):
    """Run one direction and return ``(got, exact, distance)``."""
    periodic = direction.startswith("periodic")
    mesh = stretched_mesh(EXACT_N, periodic=periodic)
    grid = Grid((mesh,), device_ids=(0,))
    grid.negotiate(halo=HaloSpec({"z": order // 2 + 1}))
    faces = face_positions(EXACT_N, periodic=periodic)
    smap = wavy_np if periodic else tanh_np
    centers = smap((np.arange(EXACT_N) + 0.5) / EXACT_N)
    boundary = "none" if periodic else "graded"

    if direction == "periodic-primal":
        # CellAvg -> Right, degree 4: the full 5-cell row is exact.
        # The windows near the seam read the periodic wrap of a
        # non-periodic polynomial, so only the interior faces count.
        coeffs = (0.2, -0.7, 1.3, -0.9, 0.5)
        values = cell_averages(coeffs, faces)
        space = mesh.cell_avg
        op = _FVBiasedReconstruction(order, bias, "linear", boundary)
        targets = faces[1:]
        distance = _away_from_the_seam(EXACT_N)
    elif direction == "periodic-dual":
        # Right -> Center: the lattice cells are the dual cells
        # around the faces, the lattice faces the mesh centers.
        coeffs = (0.2, -0.7, 1.3, -0.9, 0.5)
        edges = np.concatenate([centers, [1.0 + centers[0]]])
        values = cell_averages(coeffs, edges)
        space = mesh.right
        op = _BiasedFaceReconstruction(order, bias, "linear", boundary)
        targets = edges[:-1]
        distance = _away_from_the_seam(EXACT_N)
    elif direction in ("bounded-primal-fv", "bounded-primal-nodal"):
        # CellAvg | Center -> Inner, degree 2: the graded rungs at
        # distance d >= 2 carry order >= 3 and are exact on it.
        coeffs = (0.3, -1.1, 2.0)
        values = cell_averages(coeffs, faces)
        space = (mesh.cell_avg if direction.endswith("fv")
                 else mesh.center)
        cls = (_FVBiasedReconstruction if direction.endswith("fv")
               else _BiasedFaceReconstruction)
        op = cls(order, bias, "linear", boundary)
        targets = faces[1:-1]
        idx = np.arange(1, EXACT_N)
        distance = np.minimum(idx, EXACT_N - idx)
    else:  # bounded-dual
        # Inner -> Center: the wall lattice cells are the half cells
        # and their DOFs are exact Dirichlet zeros, so the oracle is
        # a quadratic with vanishing wall half-cell averages.
        coeffs = wall_free_quadratic(faces[0], centers[0],
                                     centers[-1], faces[-1])
        edges = np.concatenate([[faces[0]], centers, [faces[-1]]])
        values = cell_averages(coeffs, edges)[1:-1]
        space = mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
        op = _BiasedFaceReconstruction(order, bias, "linear", boundary)
        targets = centers
        idx = np.arange(1, EXACT_N + 1)
        distance = np.minimum(idx, EXACT_N + 1 - idx)

    field = grid.create_field(space, data=jnp.asarray(values))
    got = np.asarray(op["z"](field).data)
    return got, poly(coeffs)(targets), distance


DIRECTIONS = [
    pytest.param("periodic-primal", id="periodic-CellAvg-Right"),
    pytest.param("periodic-dual", id="periodic-Right-Center"),
    pytest.param("bounded-primal-fv", id="bounded-CellAvg-Inner"),
    pytest.param("bounded-primal-nodal", id="bounded-Center-Inner"),
    pytest.param("bounded-dual", id="bounded-Inner-Center"),
]


@pytest.mark.parametrize("bias", ["left", "right"])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_face_kernels_are_exact_on_a_polynomial(direction, bias):
    # THE frame test. A width co-operand sliced one slot off, or a
    # wall rung reading a synthesized zero instead of the wall half
    # cell, breaks polynomial exactness immediately; a convergence
    # rate would still look fine. The bounded cases keep the faces at
    # distance >= 2 from each wall (the bottom rung is 1st order, so
    # it reproduces constants only).
    got, exact, distance = _exactness_case(direction, bias)
    keep = distance >= 2
    assert keep.any()
    np.testing.assert_allclose(got[keep], exact[keep],
                               rtol=0, atol=1e-11)


# ================================================================
#  2. Design order on a stretched periodic axis (FV family)
# ================================================================
def _transport_error(n, order, weighting, family, *, stretched=True):
    """Max error of the semi-discrete transport of `profile`."""
    mesh = (stretched_mesh(n) if stretched
            else IntervalMesh(n, (0.0, 1.0), name="z"))
    grid = Grid((mesh,), device_ids=(0,))
    grid.negotiate(halo=HaloSpec({"z": order // 2 + 1}))
    faces = (face_positions(n) if stretched
             else np.arange(n + 1) / n)
    centers = (wavy_np((np.arange(n) + 0.5) / n) if stretched
               else (np.arange(n) + 0.5) / n)
    widths = np.diff(faces)

    if family == "fv":
        values = (profile_primitive(faces[1:])
                  - profile_primitive(faces[:-1])) / widths
        space = mesh.cell_avg
        op = _FVBiasedReconstruction(order, "left", weighting)
        exact = (profile(faces[1:]) - profile(faces[:-1])) / widths
        divergence = grid.dispatch.resolve("flux_diff", mesh.right)["z"]
    else:
        values = profile(centers)
        space = mesh.center
        op = _BiasedFaceReconstruction(order, "left", weighting)
        exact = profile_prime(centers)

        def divergence(flux):
            return flux.diff("z")

    field = grid.create_field(space, data=jnp.asarray(values))
    got = np.asarray(divergence(op["z"](field)).data)
    err = np.abs(got - exact)
    if weighting == "weno":
        # WENO-JS degrades at the critical points of the profile
        # (old-stack parity); mask them, as the uniform-mesh order
        # tests of test_advection.py do
        err = err[np.abs(exact) > 0.3 * np.abs(exact).max()]
    return float(err.max())


def _rates(errors):
    """Pairwise convergence rates of an error sequence."""
    return [np.log2(errors[i] / errors[i + 1])
            for i in range(len(errors) - 1)]


SIZES = (24, 48, 96)


@pytest.mark.parametrize(
    ("order", "weighting", "low"),
    [pytest.param(3, "linear", 2.7, id="upwind3"),
     pytest.param(5, "linear", 4.6, id="upwind5"),
     pytest.param(5, "weno", 4.3, id="weno5")])
def test_stretched_fv_transport_reaches_design_order(
        order, weighting, low):
    # route (ii)'s headline claim: on the average family the
    # width-aware rows restore the design order on a stretched mesh,
    # where the uniform-offset rows measured 2 (survey section 4)
    errors = [_transport_error(n, order, weighting, "fv")
              for n in SIZES]
    rates = _rates(errors)
    assert min(rates) > low
    assert max(rates) < order + 0.6


def test_stretched_nodal_transport_stays_second_order():
    # The nodal C-grid family is 2nd order on a stretched axis, and
    # route (ii) does NOT change that -- for a reason that has
    # nothing to do with the face value: its DOFs are point values,
    # and the flux difference divided by the physical cell width is a
    # high-order derivative ONLY on a uniform lattice (the Shu-Osher
    # FD identity; survey section 4, "Nodal family"). Restoring the
    # design order there needs route (i)'s same-row Jacobian divisor,
    # which costs exact 3-D constancy (survey section 5) and is out
    # of scope. Pinned so the expectation is on record.
    rates = _rates([_transport_error(n, 5, "linear", "nodal")
                    for n in SIZES])
    assert min(rates) > 1.7
    assert max(rates) < 2.5


@pytest.mark.parametrize("family", ["fv", "nodal"])
def test_uniform_mesh_keeps_the_design_order_of_both_families(family):
    # the negative control of the two tests above: on a UNIFORM mesh
    # both families reach order 5 (the nodal one through Shu-Osher),
    # so the 2nd order above is the stretching, not the plumbing
    rates = _rates([
        _transport_error(n, 5, "linear", family, stretched=False)
        for n in SIZES])
    assert min(rates) > 4.6


# ================================================================
#  3. Constancy and conservation (untouched by route (ii))
# ================================================================
@pytest.mark.parametrize(
    "factory", [pytest.param(lambda: UpwindAdvection(5), id="upwind5"),
                pytest.param(lambda: WENOAdvection(5), id="weno5")])
def test_constant_tracer_has_zero_tendency_on_a_stretched_axis(
        factory):
    # a constant advecting velocity is discretely divergence-free on
    # any mesh, and the width-aware rows sum to one, so A(q = const)
    # is exactly zero -- the property route (i) would have broken
    # (survey section 5).
    scheme = factory()
    model = toy_model(scheme)
    shapes = {c: model.state[c].data.shape for c in ("u", "v", "w", "b")}
    model.set_fields(u=0.7 * np.ones(shapes["u"]),
                     v=-0.4 * np.ones(shapes["v"]),
                     w=1.3 * np.ones(shapes["w"]),
                     b=np.ones(shapes["b"]))
    tau = advection_tendency(model, type(scheme))
    assert float(np.abs(np.asarray(tau["b"].data)).max()) < 1e-13


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize(
    "factory", [pytest.param(lambda: UpwindAdvection(5), id="upwind5"),
                pytest.param(lambda: WENOAdvection(5), id="weno5")])
def test_constant_tracer_matches_the_centered_divergence(
        factory, periodic):
    # with an arbitrary velocity A(q = 1) is -div(v) whatever the
    # scheme, because every row reproduces a constant. Comparing
    # against CenteredAdvection tests that on the bounded tanh column
    # too, where the graded ladder's wall rungs (half cells included)
    # have to reproduce the constant as well.
    scheme = factory()
    biased = toy_model(scheme, periodic=periodic)
    centered = toy_model(CenteredAdvection(), periodic=periodic)
    rng = np.random.default_rng(3)
    fields = {c: rng.standard_normal(biased.state[c].data.shape)
              for c in ("u", "v", "w")}
    fields["b"] = np.ones(biased.state["b"].data.shape)
    biased.set_fields(**fields)
    centered.set_fields(**fields)
    got = np.asarray(advection_tendency(biased, type(scheme))["b"].data)
    ref = np.asarray(
        advection_tendency(centered, CenteredAdvection)["b"].data)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def test_fv_tracer_content_is_conserved_on_a_stretched_axis():
    # flux_diff telescopes whatever the face value, so the FV tracer
    # content tendency is machine zero on the stretched wrap too
    scheme = WENOAdvection(5)
    model = toy_model(scheme, extra=(Tracer("c", family="fv"),))
    random_state(model, 11, names=("u", "v", "w", "b", "c"))
    tau = advection_tendency(model, WENOAdvection)
    total = float(np.asarray(tau["c"].integrate().data).ravel()[0])
    scale = float(np.abs(np.asarray(tau["c"].data)).sum()) + 1.0
    assert abs(total) < 1e-13 * scale


# ================================================================
#  4. The one-pass selected WENO kernel under the width tap-select
# ================================================================
SELECTED = [
    pytest.param("periodic-tracer", id="periodic-tracer"),
    pytest.param("periodic-velocity", id="periodic-velocity"),
    pytest.param("periodic-fv", id="periodic-fv"),
    pytest.param("bounded-tracer", id="bounded-tracer"),
    pytest.param("bounded-velocity", id="bounded-velocity"),
    pytest.param("bounded-fv", id="bounded-fv"),
]


@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("case", SELECTED)
def test_selected_kernel_matches_the_pair_on_a_stretched_axis(
        case, order):
    # the mirror identity survives the non-uniform lattice only
    # because the SAME predicate selects the width taps: the left
    # kernel on the selected (data, width) pair is the right-biased
    # non-uniform reconstruction where the flux is negative
    periodic = case.startswith("periodic")
    n_recon = 16
    mz = stretched_mesh(n_recon, periodic=periodic)
    mx = IntervalMesh(NT, (0.0, 1.0), name="x")
    my = IntervalMesh(NT, (0.0, 1.0), name="y")
    grid = Grid((mx, my, mz), device_ids=(0,))
    grid.negotiate(halo=HaloSpec({"z": order // 2 + 1}))
    boundary = "none" if periodic else "graded"
    if case.endswith("tracer"):
        factor = mz.center
    elif case.endswith("fv"):
        factor = mz.cell_avg
    elif periodic:
        factor = mz.right
    else:
        factor = mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    domain = TensorProductSpace.of(mx.center, my.center, factor)
    q = grid.create_field(
        domain,
        init=lambda x, y, z: np.sin(2 * np.pi * z)
        + 0.3 * np.cos(4 * np.pi * z) + 0.1 * np.sin(2 * np.pi * x)
        + 0.1 * np.cos(2 * np.pi * y))
    average = isinstance(factor, AverageSpace)
    cls = (_FVBiasedReconstruction if average
           else _BiasedFaceReconstruction)
    left = cls(order, "left", "weno", boundary)
    right = cls(order, "right", "weno", boundary)
    selected = _SelectedFaceReconstruction(
        order, boundary, family=("fv" if average else "nodal"))
    codomain = left["z"](q).function_space
    axis_index = codomain.bare.names.index("z")
    length = codomain.shape[axis_index]
    line = np.cos(2 * np.pi * (np.arange(length) + 0.3) / length)
    index = [None] * len(codomain.shape)
    index[axis_index] = slice(None)
    v = grid.create_field(
        codomain, data=line[tuple(index)] * np.ones(codomain.shape))
    positive = v + abs(v)
    ref = Where()(positive, left["z"](q).retag(codomain),
                  right["z"](q).retag(codomain))
    new = selected(positive, q, "z", codomain)
    np.testing.assert_allclose(
        np.asarray(new.data), np.asarray(ref.data),
        rtol=0, atol=1e-14)


# ================================================================
#  5. ENO on a stretched axis
# ================================================================
@pytest.mark.parametrize("bias", ["left", "right"])
def test_weno_step_reconstruction_gains_no_overshoot(bias):
    # the ENO property is what the biased schemes are FOR; the
    # non-uniform candidate rows must not manufacture a new extremum
    # at a step on a stretched lattice
    n = 32
    mesh = stretched_mesh(n)
    grid = Grid((mesh,), device_ids=(0,))
    grid.negotiate(halo=HaloSpec({"z": 3}))
    faces = face_positions(n)
    centers = 0.5 * (faces[1:] + faces[:-1])
    step = np.where((centers > 0.25) & (centers < 0.75), 1.0, 0.0)
    field = grid.create_field(mesh.cell_avg, data=jnp.asarray(step))
    got = np.asarray(
        _FVBiasedReconstruction(5, bias, "weno")["z"](field).data)
    assert got.max() < 1.0 + 1e-12
    assert got.min() > -1e-12
    # the linear row DOES ring (the control that makes the above
    # meaningful)
    linear = np.asarray(
        _FVBiasedReconstruction(5, bias, "linear")["z"](field).data)
    assert linear.max() > 1.0 + 1e-3


# ================================================================
#  6. Uniform-grid parity: the static float tables, bitwise
# ================================================================
def test_uniform_factors_take_the_static_path():
    mesh = IntervalMesh(8, (0.0, 1.0), name="z")
    grid = Grid((mesh,), device_ids=(0,))
    field = grid.create_field(mesh.center)
    assert _face_widths(field, "z") is None
    stretched = stretched_mesh(8)
    sgrid = Grid((stretched,), device_ids=(0,))
    sgrid.negotiate(halo=HaloSpec({"z": 3}))
    assert _face_widths(
        sgrid.create_field(stretched.center), "z") is not None


@pytest.mark.parametrize(
    "factory", [pytest.param(lambda: UpwindAdvection(5), id="upwind5"),
                pytest.param(lambda: WENOAdvection(5), id="weno5")])
def test_uniform_grid_tendency_is_bitwise_unchanged(factory, monkeypatch):
    # a uniform factor must reach the static float tables: forcing
    # cell_widths to None (the pre-route-(ii) path) may not move a
    # single bit of a uniform-grid tendency
    def build():
        scheme = factory()
        model = Model(
            grid=Grid(tuple(
                IntervalMesh(8, (0.0, 1.0), name=name)
                for name in ("x", "y", "z"))),
            modules=(Core(), scheme),
            time_stepper=AdamBashforth(DT, order=3))
        random_state(model, 5)
        return model, type(scheme)

    model, cls = build()
    reference = {name: np.asarray(field.data) for name, field
                 in zip(("u", "v", "w", "b"),
                        (advection_tendency(model, cls)[c]
                         for c in ("u", "v", "w", "b")), strict=True)}
    monkeypatch.setattr(
        "fridom.model.modules.advection.cell_widths",
        lambda _f, _axis: None)
    model, cls = build()
    tau = advection_tendency(model, cls)
    for name in ("u", "v", "w", "b"):
        np.testing.assert_array_equal(
            np.asarray(tau[name].data), reference[name])


@pytest.mark.parametrize(
    "factory", [pytest.param(lambda: UpwindAdvection(5), id="upwind5"),
                pytest.param(lambda: WENOAdvection(5), id="weno5")])
def test_stretched_grid_tendency_uses_the_widths(factory, monkeypatch):
    # the twin of the parity test: on a STRETCHED factor the width
    # path must actually change the numbers (otherwise the parity
    # test above would pass vacuously)
    def build():
        scheme = factory()
        model = toy_model(scheme)
        random_state(model, 5)
        return model, type(scheme)

    model, cls = build()
    with_widths = np.asarray(advection_tendency(model, cls)["b"].data)
    monkeypatch.setattr(
        "fridom.model.modules.advection.cell_widths",
        lambda _f, _axis: None)
    model, cls = build()
    without = np.asarray(advection_tendency(model, cls)["b"].data)
    assert np.abs(with_widths - without).max() > 1e-6


# ================================================================
#  7. Differentiability (AGENTS.md policy)
# ================================================================
def test_propagator_grad_through_a_stretched_weno_run_matches_fd():
    # the width tables are static geometry, but the new width-aware
    # arithmetic (ratios of summed widths, the ideal-weight divides)
    # is inside the VJP: a masked singularity there would show as a
    # NaN gradient
    model = toy_model(WENOAdvection(5), n=8)
    random_state(model, 2)
    run = model.propagator(wrt=("b",), steps=6)
    b0 = model._carry.state["b"].storage

    def loss(field):
        return sum(jnp.sum(f.data ** 2) for f in run((field,)).state)

    grad = np.asarray(jax.grad(loss)(b0))
    assert bool(np.all(np.isfinite(grad)))
    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(b0.shape),
                            dtype=b0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-5
    finite = (float(loss(b0 + eps * direction))
              - float(loss(b0 - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(finite, rel=1e-4)


# ================================================================
#  8. The residual refusal: stretched + immersed
# ================================================================
@pytest.mark.parametrize("cls", [UpwindAdvection, WENOAdvection])
def test_stretched_immersed_is_a_taught_error(cls):
    # the mask-keyed ladder carries no width co-operand, so the
    # biased schemes keep refusing this ONE combination
    module = cls(5)
    grid = stretched_grid(8, immersed=True)
    with pytest.raises(
            NotImplementedError,
            match=r"stretched \(mapped\) mesh on an immersed"):
        module._bind_mapping(grid)
    # the centered scheme is grounded on the combination
    CenteredAdvection()._bind_mapping(grid)  # no raise


def test_stretched_immersed_face_kernel_is_a_taught_error():
    # the operator-level twin (a direct application)
    grid = stretched_grid(8, immersed=True)
    grid.negotiate(halo=HaloSpec({"z": 3}))
    space = TensorProductSpace.of(
        grid.factors[0].center, grid.factors[1].center,
        grid.factors[2].center)
    field = grid.create_field(space)
    with pytest.raises(NotImplementedError,
                       match="immersed"):
        _BiasedFaceReconstruction(5, "left", "weno")["z"](field)
    with pytest.raises(NotImplementedError, match="immersed"):
        _CenteredFaceInterpolation(4)["z"](field)
