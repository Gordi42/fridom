"""FV (CellAvg) tracer slice of the flux-form advection modules.

Prefix-mirrored shard of ``advection.py`` (stage F2): the
``_FVBiasedReconstruction`` operator signature, the per-component
family switch (`_is_average_space`), and the three acceptance gates
of the FV tracer slice — conservation to machine zero, bitwise (or
machine-eps) parity with the nodal path, and a mixed nodal-velocity
plus FV-tracer model. Self-contained (import-mode=importlib): the
small builders are duplicated from ``test_advection.py``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.model import Model as FrModel
from fridom.model.modules.advection import (
    CenteredAdvection,
    UpwindAdvection,
    WENOAdvection,
    _FVBiasedReconstruction,
    _is_average_space,
)
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.nonhydro2.modules.core import Core
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
)
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import (
    MappedIntervalMesh,
)
from fridom.spatial.operators.composed import Divergence
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.nodal import NodalSpace
from fridom.spatial.spaces.tensor_product import TensorProductSpace

L = 2 * np.pi
DT = 0.004


# ================================================================
#  Helpers
# ================================================================
class _PassiveTracer(fr.model.Module):

    """A passive CellAvg (or nodal) tracer ``b``, advected only."""

    def __init__(self, family=None):
        self._family = family

    @property
    def field_declarations(self):
        return (fr.model.FieldDeclaration.tracer(
            "b", space=fr.spatial.Collocated(family=self._family),
            long_name="tracer", units="1"),)


def make_grid(walled=(), n=12):
    return Grid(tuple(
        IntervalMesh(n, (0.0, 1.0 if nm in walled else L),
                     periodic=(nm not in walled), name=nm)
        for nm in ("x", "y", "z")))


def make_model(walled, adv, family="fv", n=12):
    return FrModel(grid=make_grid(walled, n),
        modules=(Core(), FPlaneCoriolis(f0=0.5),
                 _PassiveTracer(family=family), adv),
        time_stepper=AdamBashforth(DT, order=3))


def advection_tendency(model, cls):
    return model.tendency(model.state, constraints=False,
                          filter=fr.model.term_predicates.owned_by(cls))


def set_random_state(model, seed=3):
    rng = np.random.default_rng(seed)
    model.set_fields(**{c: rng.standard_normal(model.state[c].data.shape)
                        for c in ("u", "v", "w", "b")})


def _wavy(s):
    """Smooth wavy stretching of the unit computational interval."""
    return s + 0.1 * jnp.sin(2.0 * jnp.pi * s) / (2.0 * jnp.pi)


def _depth(x):
    """Smooth periodic water depth H(x) (20% slope)."""
    return 1.0 + 0.2 * jnp.sin(x)


def make_mapped_grid(n=8, ny=4, *, periodic_column=True):
    """Terrain-following grid ``zp = z * H(x)`` (a mapped column)."""
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H}, params={"H": _depth})
    return Grid((
        IntervalMesh(n, (0.0, L), name="x"),
        IntervalMesh(ny, (0.0, L), name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=periodic_column,
                     name="z"),
    ), mapping=mapping)


# ================================================================
#  _FVBiasedReconstruction: the C-grid FV face signature
# ================================================================
def test_fv_reconstruction_codomain_signature():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    mb = IntervalMesh(8, (0.0, 1.0), periodic=False, name="b")
    # periodic CellAvg -> Right, whatever the bias / weighting
    assert (_FVBiasedReconstruction(3, "left", "weno")
            .codomain(mx.cell_avg) is mx.right)
    assert (_FVBiasedReconstruction(5, "right", "linear")
            .codomain(mx.cell_avg) is mx.right)
    # the bounded signature is grounded under boundary="graded"
    assert (_FVBiasedReconstruction(3, "left", "weno",
                                    boundary="graded")
            .codomain(mb.cell_avg) is mb.inner)
    # boundary="none" is periodic-only on a bounded mesh
    with pytest.raises(SpaceMismatchError, match="periodic-only"):
        _FVBiasedReconstruction(3, "left", "weno").codomain(
            mb.cell_avg)


def test_fv_reconstruction_rejects_nodal_and_complex():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    op = _FVBiasedReconstruction(3, "left", "weno")
    # a nodal factor is not a CellAvg tracer
    with pytest.raises(SpaceMismatchError, match="CellAvg"):
        op.codomain(mx.center)
    # nor is a complex CellAvg
    with pytest.raises(SpaceMismatchError, match="CellAvg"):
        op.codomain(mx.cell_avg.as_complex())


def test_fv_reconstruction_rejects_a_stretched_factor():
    # the operator-level uniform-mesh refusal (the biased FV rows are
    # uniform-offset weights): a stretched (mapped) mesh raises
    mesh = MappedIntervalMesh(8, (0.0, 1.0), _wavy,
                              periodic=True, name="z")
    with pytest.raises(SpaceMismatchError, match="uniform-mesh only"):
        _FVBiasedReconstruction(3, "left", "weno").codomain(
            mesh.cell_avg)


def test_fv_reconstruction_constructor_validation():
    with pytest.raises(ValueError, match="weighting must be"):
        _FVBiasedReconstruction(3, "left", "quadratic")
    with pytest.raises(ValueError, match="boundary must be"):
        _FVBiasedReconstruction(3, "left", "weno", "one_sided")
    with pytest.raises(ValueError, match="wall must be"):
        _FVBiasedReconstruction(3, "left", "weno", "graded", "quick")
    # a non-grounded order raises through the framework WENO tables
    with pytest.raises(ValueError):  # noqa: PT011
        _FVBiasedReconstruction(4, "left", "weno")


def test_fv_reconstruction_properties_requirements_interning():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    op = _FVBiasedReconstruction(3, "left", "weno")
    assert op.order == 3
    assert op.bias == "left"
    assert op.weighting == "weno"
    assert op.boundary == "none"
    assert op.wall == "upwind1"
    # two-sided reach: the left-biased order-3 window on CellAvg->Right
    # is [-1,+1] (m0 = biased_offset(3, "left") = 1), so halo 1 -- the
    # true kernel reach, tighter than the symmetric order//2+1 = 2
    assert op.requirements(mx.cell_avg).reach == (1, 1)
    assert op.requirements(mx.cell_avg).halo == 1
    # every constructor arg round-trips through the properties
    op5 = _FVBiasedReconstruction(5, "right", "linear", "graded",
                                  "centered2")
    assert (op5.order, op5.bias, op5.weighting, op5.boundary,
            op5.wall) == (5, "right", "linear", "graded", "centered2")
    assert op5.requirements(mx.cell_avg).halo == 5 // 2 + 1
    # interned on its structural key
    assert op is _FVBiasedReconstruction(3, "left", "weno")
    assert op is not _FVBiasedReconstruction(3, "right", "weno")


def test_is_average_space_flags_a_cellavg_product():
    mx = IntervalMesh(8, (0.0, L), name="x")
    my = IntervalMesh(8, (0.0, L), name="y")
    fv = TensorProductSpace.of(mx.cell_avg, my.center)
    nodal = TensorProductSpace.of(mx.center, my.center)
    assert _is_average_space(fv)
    assert not _is_average_space(nodal)


# ================================================================
#  Gate 1: conservation (telescoping to machine zero)
# ================================================================
CONSERVE = [
    pytest.param(CenteredAdvection, id="centered"),
    pytest.param(lambda: UpwindAdvection(3), id="upwind3"),
    pytest.param(lambda: WENOAdvection(3), id="weno3"),
    pytest.param(lambda: WENOAdvection(5), id="weno5"),
]

WALLED = [
    pytest.param((), id="periodic"),
    pytest.param(("z",), id="lid"),
    pytest.param(("x", "y", "z"), id="box"),
]


@pytest.mark.parametrize("walled", WALLED)
@pytest.mark.parametrize("factory", CONSERVE)
def test_fv_tracer_mass_is_conserved(factory, walled):
    scheme = factory()
    cls = type(scheme)
    m = make_model(walled, scheme)
    set_random_state(m)
    tau = advection_tendency(m, cls)
    total = float(np.asarray(tau["b"].integrate().data).ravel()[0])
    scale = float(np.abs(np.asarray(tau["b"].data)).sum()) + 1.0
    assert abs(total) < 1e-12 * scale
    # and the discrete mass drifts by machine zero over a short run
    mass0 = float(np.asarray(m.state["b"].integrate().data).ravel()[0])
    m.advance(4)
    mass1 = float(np.asarray(m.state["b"].integrate().data).ravel()[0])
    assert abs(mass1 - mass0) < 1e-11 * (abs(mass0) + 1.0)
    for c in ("u", "v", "w", "b"):
        assert np.isfinite(np.asarray(m.state[c].data)).all()


# ================================================================
#  Gate 2: parity of the FV path with the nodal path
# ================================================================
PARITY = [
    pytest.param(CenteredAdvection, id="centered"),
    pytest.param(lambda: UpwindAdvection(3), id="upwind3"),
    pytest.param(lambda: UpwindAdvection(5), id="upwind5"),
    pytest.param(lambda: WENOAdvection(3), id="weno3"),
    pytest.param(lambda: WENOAdvection(5), id="weno5"),
]


@pytest.mark.parametrize(
    "walled", [pytest.param((), id="periodic"),
               pytest.param(("z",), id="lid")])
@pytest.mark.parametrize("factory", PARITY)
def test_fv_tracer_tendency_matches_the_nodal_tracer(factory, walled):
    # the FV tracer (CellAvg) and the collocated nodal tracer share the
    # primal cell midpoints, the same kernels, and the same graded
    # ladder, so the b tendency agrees: bitwise on a periodic grid,
    # machine-eps on a walled one (the flux_diff vs diff+retag seam)
    cls = type(factory())
    mf = make_model(walled, factory(), family="fv")
    mn = make_model(walled, factory(), family=None)
    rng = np.random.default_rng(7)
    fields = {c: rng.standard_normal(mf.state[c].data.shape)
              for c in ("u", "v", "w", "b")}
    mf.set_fields(**fields)
    mn.set_fields(**fields)
    bf = np.asarray(advection_tendency(mf, cls)["b"].data)
    bn = np.asarray(advection_tendency(mn, cls)["b"].data)
    if walled:
        np.testing.assert_allclose(bf, bn, rtol=0, atol=1e-11)
    else:
        np.testing.assert_array_equal(bf, bn)


# ================================================================
#  Gate 3: a mixed nodal-velocity + FV-tracer model
# ================================================================
@pytest.mark.parametrize(
    "adv", [pytest.param(CenteredAdvection(), id="centered"),
            pytest.param(WENOAdvection(3), id="weno3")])
def test_mixed_fv_tracer_model_runs_and_projects(adv):
    m = FrModel(
        grid=make_grid(),
        modules=(Core(), FPlaneCoriolis(f0=0.5),
                 ConstantStratification(n2=1.0, family="fv"), adv),
        time_stepper=AdamBashforth(DT, order=3))
    # the buoyancy tracer is fully on CellAvg; the pressure and the
    # velocity stay nodal (Center / Right)
    assert all(isinstance(f, CellAvg)
               for f in m.state["b"].function_space.bare.factors)
    assert all(isinstance(f, NodalSpace)
               for f in m.state["p"].function_space.bare.factors)
    assert all(isinstance(f, NodalSpace)
               for f in m.state["u"].function_space.bare.factors)
    set_random_state(m)
    m.advance(6)
    for c in ("u", "v", "w", "p", "b"):
        assert np.isfinite(np.asarray(m.state[c].data)).all()
    vel = VectorField({"u": m.state["u"], "v": m.state["v"],
                       "w": m.state["w"]})
    assert np.abs(np.asarray(Divergence()(vel).data)).max() < 1e-12


# ================================================================
#  Background flow + FV tracer (the linear-weight FV pair branch)
# ================================================================
@pytest.mark.parametrize(
    "cls", [UpwindAdvection, WENOAdvection])
def test_fv_background_split_conserves_and_stays_finite(cls):
    # a biased scheme with a background flow contributes the linear
    # background_advection term, which routes the FV tracer through
    # the linear-weight average-family reconstruction pair; the split
    # still telescopes onto the exact-zero boundary flux
    adv = cls(3, background={"u": lambda y: 1.0 + 0.3 * np.sin(y),
                             "v": 0.5})
    m = make_model((), adv)
    set_random_state(m)
    tau = advection_tendency(m, cls)
    db = np.asarray(tau["b"].data)
    total = float(np.asarray(tau["b"].integrate().data).ravel()[0])
    assert abs(total) < 1e-11 * (np.abs(db).sum() + 1.0)
    m.advance(4)
    for c in ("u", "v", "w", "p", "b"):
        assert np.isfinite(np.asarray(m.state[c].data)).all()


# ================================================================
#  Stage F5: FV tracer on a mapped column (J-weighted conservation)
# ================================================================
@pytest.mark.parametrize("periodic_column", [True, False])
def test_fv_tracer_on_a_mapped_grid_conserves_physical_content(
        periodic_column):
    # CenteredAdvection transports a CellAvg tracer on a terrain-
    # following column in J-weighted conservative flux form (stage F5),
    # so its PHYSICAL content int(q dV) = int(J q dx) is conserved to
    # machine zero — the FV headline property on genuine terrain, which
    # the consistent nodal mapped divergence does not give. On this
    # mapped grid the ``.integrate()`` verb is physical (it supplies
    # the column Jacobian), so ``tau["b"].integrate()`` IS int(J q dx).
    grid = make_mapped_grid(periodic_column=periodic_column)
    model = FrModel(grid=grid,
                    modules=(Core(),
                             _PassiveTracer(family="fv"),
                             CenteredAdvection()),
                    time_stepper=AdamBashforth(DT, order=3))
    set_random_state(model)
    tau = advection_tendency(model, CenteredAdvection)
    physical = float(np.asarray(tau["b"].integrate().data).ravel()[0])
    db = np.asarray(tau["b"].data)
    scale = float(np.sum(np.abs(db)))
    assert abs(physical) < 1e-11 * (scale + 1.0)


def test_fv_mapped_tracer_conserves_and_nodal_does_not():
    # the same b advected on the nodal mapped model does NOT conserve
    # its physical content: the FV conservative flux form is the new
    # property, not a shared one. Both grids are static terrain, so the
    # physical ``.integrate()`` verb supplies the column Jacobian and
    # ``tau.integrate()`` measures int(J q dx) directly.
    grid = make_mapped_grid()
    fv = FrModel(grid=grid,
                 modules=(Core(),
                          _PassiveTracer(family="fv"),
                          CenteredAdvection()),
                 time_stepper=AdamBashforth(DT, order=3))
    nodal = FrModel(grid=make_mapped_grid(),
                    modules=(Core(),
                             _PassiveTracer(family="nodal"),
                             CenteredAdvection()),
                    time_stepper=AdamBashforth(DT, order=3))
    set_random_state(fv)
    set_random_state(nodal)
    tau_fv = advection_tendency(fv, CenteredAdvection)["b"]
    tau_nod = advection_tendency(nodal, CenteredAdvection)["b"]
    w_fv = abs(float(np.asarray(tau_fv.integrate().data).ravel()[0]))
    w_nod = abs(float(np.asarray(tau_nod.integrate().data).ravel()[0]))
    scale = float(np.sum(np.abs(np.asarray(tau_fv.data))))
    assert w_fv < 1e-11 * (scale + 1.0)
    assert w_nod > 1e-6 * scale  # the nodal form is not conservative


# ================================================================
#  FV velocity self-advection through the biased C-grid path
#  (the F3 default-flip blocker, side-finding upwind5_revisit.md §7)
# ================================================================
# On the FV C-grid the velocity is face-normal: u lives on
# ``Right(x) ⊗ CellAvg(y) ⊗ CellAvg(z)``, so its self-advection axis
# crosses nodal -> average (the diagnosed ``Center`` vs ``CellAvg``
# ``SpaceMismatchError`` in the biased ``_face_value`` retag). These
# tests drive the real FV C-grid (``nh.Model``, FV default) with the
# biased schemes, which the raw-model ``_PassiveTracer`` shard above
# (nodal velocities, CellAvg tracer only) never assembled.
BIASED_FV = [
    pytest.param(lambda: UpwindAdvection(3), UpwindAdvection, id="upwind3"),
    pytest.param(lambda: UpwindAdvection(5), UpwindAdvection, id="upwind5"),
    pytest.param(lambda: WENOAdvection(3), WENOAdvection, id="weno3"),
    pytest.param(lambda: WENOAdvection(5), WENOAdvection, id="weno5"),
]


def _nh_model(walled, factory, family=None, n=8):
    return nh.Model(
        grid=make_grid(walled, n=n),
        core=nh.Core(aspect_ratio=(2.0) ** 0.5, family=family),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=1.0),
        advection=factory())


def _seed_pair(fv, nodal, seed=7):
    # identical random data on both models (FV and nodal shapes match:
    # the velocity staggering is shared, CellAvg and Center both carry n
    # cells, and the wall-normal Inner faces have the same count)
    rng = np.random.default_rng(seed)
    data = {c: rng.standard_normal(fv.state[c].data.shape)
            for c in ("u", "v", "w", "b")}
    fv.set_fields(**data)
    nodal.set_fields(**data)


def _assert_fv_velocity(model):
    # the model really resolved the FV face-normal C-grid velocity, so
    # the biased path crosses nodal -> average on the self-advection axis
    u = model.state["u"].function_space.bare
    assert isinstance(u.factor("x"), NodalSpace)   # the normal face
    assert isinstance(u.factor("y"), CellAvg)       # transverse average
    assert isinstance(u.factor("z"), CellAvg)


@pytest.mark.parametrize(("factory", "cls"), BIASED_FV)
def test_fv_velocity_self_advection_matches_nodal_periodic(factory, cls):
    # gate: on a periodic FV box the biased velocity self-advection is
    # BITWISE the nodal one (the Center->CellAvg flux bridge and the
    # CellAvg->Center velocity deconvolve are the 2nd-order identity, so
    # the order-coupled tendency is preserved to the bit — the F-series
    # pattern), whatever the reconstruction order.
    fv = _nh_model((), factory)
    nodal = _nh_model((), factory, family="nodal")
    _assert_fv_velocity(fv)
    _seed_pair(fv, nodal)
    for c in ("u", "v", "w", "b"):
        tf = np.asarray(advection_tendency(fv, cls)[c].data)
        tn = np.asarray(advection_tendency(nodal, cls)[c].data)
        np.testing.assert_array_equal(
            tf, tn, err_msg=f"FV vs nodal biased tendency on {c!r}")
    # and a short trajectory stays bitwise identical
    fv.advance(6)
    nodal.advance(6)
    for c in ("u", "v", "w", "b", "p"):
        np.testing.assert_array_equal(
            np.asarray(fv.state[c].data), np.asarray(nodal.state[c].data))


@pytest.mark.parametrize("walled", [pytest.param(("z",), id="lid"),
                                    pytest.param(("x", "y", "z"), id="box")])
@pytest.mark.parametrize(("factory", "cls"), BIASED_FV)
def test_fv_velocity_self_advection_walled_assembles_and_conserves(
        factory, cls, walled):
    # gate 3: the biased schemes assemble and step on a WALLED FV grid
    # (the graded near-wall ladder on the wall-normal velocity's own
    # staggered axis), the FV tracer mass telescopes to machine zero,
    # and the run stays finite.
    model = _nh_model(walled, factory)
    _assert_fv_velocity(model)
    set_random_state(model)
    tau = advection_tendency(model, cls)
    db = np.asarray(tau["b"].data)
    total = float(np.asarray(tau["b"].integrate().data).ravel()[0])
    assert abs(total) < 1e-11 * (np.abs(db).sum() + 1.0)
    # the projected velocity tendency is divergence-free (the pressure
    # DCT-II solve on the Neumann CellAvg origin), mirroring the nodal
    # walled invariant
    state = model.constrain(model.state)
    ptau = model.tendency(state, constraints=True)
    div = Divergence()(VectorField({c: ptau[c] for c in ("u", "v", "w")}))
    tau_scale = max(float(np.abs(np.asarray(ptau[c].data)).max())
                    for c in ("u", "v", "w"))
    assert float(np.abs(np.asarray(div.data)).max()) < 1e-12 * tau_scale
    model.advance(6)
    for c in ("u", "v", "w", "b", "p"):
        assert np.isfinite(np.asarray(model.state[c].data)).all()


@pytest.mark.parametrize(("factory", "cls"), BIASED_FV)
def test_fv_velocity_self_advection_walled_matches_nodal(factory, cls):
    # gate 3: parity with the walled nodal model. The per-step operators
    # are bit-identical in pure-eager (the FV/nodal 2nd-order stencils
    # agree and the deconvolve bridge is an identity), so the advection
    # tendency agrees to machine precision under jit — the walled mixed
    # Fourier ⊗ Cosine graph fuses its float adds in a different order
    # than the nodal one (the documented F4 XLA-fusion artifact; the
    # purely periodic path above stays exactly bitwise).
    fv = _nh_model(("z",), factory)
    nodal = _nh_model(("z",), factory, family="nodal")
    _seed_pair(fv, nodal)
    for c in ("u", "v", "w", "b"):
        tf = np.asarray(advection_tendency(fv, cls)[c].data)
        tn = np.asarray(advection_tendency(nodal, cls)[c].data)
        np.testing.assert_allclose(tf, tn, rtol=0, atol=1e-12)
