"""Tests for the linear-term block signatures (T1-T4)."""
from dataclasses import replace

import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.nonhydro2 as nh
import fridom.shallowwater2 as sw
from fridom.framework2.grid.operators.base import (
    EigenbasisError,
    Operator,
)
from fridom.framework2.model.context import StepContext
from fridom.framework2.model.linear_blocks import (
    Coeff,
    Diff,
    Interp,
    LinearBlock,
    Scale,
    apply_linear_blocks,
    linear_blocks,
    linear_blocks_halo,
)
from fridom.framework2.model.params import (
    STRATIFICATION_N2,
)
from fridom.framework2.model.terms import TERM_ATTRIBUTE
from fridom.framework2.modules.coriolis import _CORIOLIS_BLOCKS
from fridom.nonhydro2.modules.stratification import (
    _BUOYANCY_BLOCKS,
    _RESTORING_BLOCKS,
)
from fridom.nonhydro2.params import DSQR
from fridom.shallowwater2 import params as sw_params
from fridom.shallowwater2.modules.core import (
    _GRAVITY_BLOCKS,
    DynamicalCore,
)

DT = 5e-3


# ================================================================
#  Fixtures — assembled models with random state
# ================================================================
def _sw_grid(n=16):
    """Return a tiny doubly-periodic square grid."""
    mx = fr.grid.meshes.IntervalMesh(n, (0.0, 1.0), periodic=True,
                                     name="x")
    my = fr.grid.meshes.IntervalMesh(n, (0.0, 1.0), periodic=True,
                                     name="y")
    return fr.grid.Grid((mx, my))


def _nh_grid(n=8, length=2 * np.pi):
    """Return a tiny triply-periodic cube grid."""
    ms = [fr.grid.meshes.IntervalMesh(n, (0.0, length), periodic=True,
                                      name=a) for a in "xyz"]
    return fr.grid.Grid(tuple(ms))


def _ctx(model):
    """Return a StepContext frozen at t=0 for the model's params."""
    return StepContext(params=model.parameters.at_time(0.0),
                       clock=0.0, dt=DT, stage_dt=DT)


@pytest.fixture
def sw_model():
    """Return a shallow-water model with random prognostic state."""
    model = sw.Model(
        grid=_sw_grid(), csqr=1.3, rossby_number=0.2,
        coriolis=sw.modules.FPlaneCoriolis(f0=0.7), advection=True,
        time_stepper=fr.time_steppers.AdamBashforth(DT, order=3))
    _randomize(model, ("u", "v", "p"), seed=1)
    return model


@pytest.fixture
def nh_model():
    """Return a nonhydrostatic model with random prognostic state."""
    model = nh.Model(grid=_nh_grid(), dt=DT)
    _randomize(model, ("u", "v", "w", "b"), seed=2)
    return model


def _randomize(model, names, *, seed):
    """Write random true-shape arrays onto the named components."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        name: rng.standard_normal(model.state[name].shape)
        for name in names})


def _same(a, b):
    """Bitwise array equality of two fields' true-shape data."""
    return bool((a.data == b.data).all())


# ================================================================
#  T4 — the numeric equivalence (block-derived == hand closures)
# ================================================================
def test_gravity_blocks_bit_identical(sw_model):
    """Block-derived gravity matches the hand flux-form bit-for-bit."""
    state, ctx = sw_model.state, _ctx(sw_model)
    u, v, p = state["u"], state["v"], state["p"]
    c = state["csqr"]
    flux_u = c.to(u.function_space) * u
    flux_v = c.to(v.function_space) * v
    hand = {"u": -p.diff("x"), "v": -p.diff("y"),
            "p": -(flux_u.diff("x") + flux_v.diff("y"))}
    derived = apply_linear_blocks(_GRAVITY_BLOCKS, state, ctx)
    assert set(derived) == set(hand)
    for name, expected in hand.items():
        assert _same(derived[name], expected)


def test_coriolis_blocks_bit_identical(sw_model):
    """Block-derived Coriolis matches the hand field arithmetic."""
    state, ctx = sw_model.state, _ctx(sw_model)
    u, v = state["u"], state["v"]
    f = state["f_coriolis"]
    hand = {
        "u": f.to(u.function_space) * v.to(u.function_space),
        "v": -(f.to(v.function_space) * u.to(v.function_space))}
    derived = apply_linear_blocks(_CORIOLIS_BLOCKS, state, ctx)
    assert set(derived) == set(hand)
    for name, expected in hand.items():
        assert _same(derived[name], expected)


def test_buoyancy_block_bit_identical(nh_model):
    """Block-derived buoyancy force matches the hand ``b/dsqr``."""
    state, ctx = nh_model.state, _ctx(nh_model)
    dsqr = ctx.params[DSQR]
    b = state["b"].to(state["w"].function_space)
    hand = b.with_data(b.data / dsqr)
    derived = apply_linear_blocks(_BUOYANCY_BLOCKS, state, ctx)
    assert _same(derived["w"], hand)


def test_restoring_block_bit_identical(nh_model):
    """Block-derived restoring matches the hand ``-N^2 w``."""
    state, ctx = nh_model.state, _ctx(nh_model)
    n2 = ctx.params[STRATIFICATION_N2]
    w = state["w"].to(state["b"].function_space)
    hand = w.with_data(-n2 * w.data)
    derived = apply_linear_blocks(_RESTORING_BLOCKS, state, ctx)
    assert _same(derived["b"], hand)


def test_existing_model_tendency_runs(sw_model):
    """The re-authored terms still advance the model (smoke)."""
    sw_model.advance(3)
    assert not sw_model.panicked


# ================================================================
#  R10 — the fr.linear_term factory
# ================================================================
def test_linear_term_stamps_and_matches(sw_model):
    """fr.linear_term stamps a term whose fn == apply_linear_blocks."""
    fn = fr.linear_term(
        "grav", advances=("u", "v", "p"), blocks=_GRAVITY_BLOCKS)
    decl = getattr(fn, TERM_ATTRIBUTE)
    assert decl.name == "grav"
    assert decl.fn is fn
    assert decl.linear is True
    assert decl.treatment is fr.EXPLICIT
    assert decl.advances == ("u", "v", "p")
    assert decl.blocks == tuple(_GRAVITY_BLOCKS)
    # the synthesized numeric fn (composer's (module, state, ctx)
    # convention) matches the hand-rolled apply_linear_blocks
    state, ctx = sw_model.state, _ctx(sw_model)
    derived = apply_linear_blocks(_GRAVITY_BLOCKS, state, ctx)
    via_fn = fn(None, state, ctx)
    assert set(via_fn) == set(derived)
    for name in derived:
        assert _same(via_fn[name], derived[name])


def test_linear_term_defaults_and_transports():
    """Optional fields default; transports is recorded verbatim."""
    fn = fr.linear_term(
        "t", blocks=_GRAVITY_BLOCKS, transports=("p",))
    decl = getattr(fn, TERM_ATTRIBUTE)
    assert decl.advances is None
    assert decl.transports == ("p",)


# ================================================================
#  T2/T3 — the linear_blocks accessor structure
# ================================================================
def test_linear_blocks_sw_structure(sw_model):
    """The SW accessor yields the expected (out,src)->(op,coeff)."""
    resolved = linear_blocks(sw_model)
    got = {(b.out, b.src): b for b in resolved}
    # every op is a retained fr.Operator
    assert all(isinstance(b.op, Operator) for b in resolved)
    # gravity: bare pressure gradients, flux-form divergences
    assert got[("u", "p")].coeff == -1
    assert got[("v", "p")].coeff == -1
    assert got[("p", "u")].coeff == -1
    assert got[("p", "v")].coeff == -1
    # Coriolis: antisymmetric, constant = +/- f0
    assert got[("u", "v")].coeff == pytest.approx(0.7)
    assert got[("v", "u")].coeff == pytest.approx(-0.7)
    assert set(got) == {("u", "p"), ("v", "p"), ("p", "u"),
                        ("p", "v"), ("u", "v"), ("v", "u")}


def test_linear_blocks_nh_structure(nh_model):
    """The nonhydro accessor yields Coriolis + stratification blocks."""
    resolved = linear_blocks(nh_model)
    got = {(b.out, b.src): b.coeff for b in resolved}
    assert got[("u", "v")] == pytest.approx(1.0)
    assert got[("v", "u")] == pytest.approx(-1.0)
    # buoyancy force: +1/dsqr (invert); restoring: -N^2
    assert got[("w", "b")] == pytest.approx(1.0)
    assert got[("b", "w")] == pytest.approx(-1.0)
    assert set(got) == {("u", "v"), ("v", "u"),
                        ("w", "b"), ("b", "w")}


def test_linear_blocks_invert_coeff(nh_model):
    """A non-unit dsqr resolves the buoyancy coeff to 1/dsqr."""
    nh_model.update_parameters({DSQR: 4.0})
    got = {(b.out, b.src): b.coeff for b in linear_blocks(nh_model)}
    assert got[("w", "b")] == pytest.approx(0.25)


def test_linear_blocks_flux_form_op_folds_csqr(sw_model):
    """The p<-u block resolves to a composite carrying c^2 (constant)."""
    resolved = {(b.out, b.src): b for b in linear_blocks(sw_model)}
    block = resolved[("p", "u")]
    # a composite: FiniteDifference @ (c^2 * Identity)
    assert hasattr(block.op, "factors")
    assert len(block.op.factors) == 2


# ================================================================
#  T3 — the constant-coefficient gate declines variable coefficients
# ================================================================
def test_beta_plane_declined():
    """A beta-plane f(y) provides no f0: the gate declines it."""
    model = nh.Model(
        grid=_nh_grid(), dt=DT, advection=False,
        coriolis=fr.modules.BetaPlaneCoriolis(f0=1.0, beta=0.5))
    with pytest.raises(EigenbasisError, match=r"coriolis\.f0"):
        linear_blocks(model)


def test_bad_src_declined(sw_model):
    """A block whose src is not a state component is declined."""
    module = sw_model.module(DynamicalCore)
    method = type(module).gravity
    decl = getattr(method, TERM_ATTRIBUTE)
    bad = LinearBlock("u", "ghost", Diff("x"), Coeff(const=-1))
    patched = replace(decl, blocks=(bad, *decl.blocks))
    setattr(method, TERM_ATTRIBUTE, patched)
    try:
        with pytest.raises(EigenbasisError, match="not a state"):
            linear_blocks(sw_model)
    finally:
        setattr(method, TERM_ATTRIBUTE, decl)


# ================================================================
#  Coeff — validation and the sign/literal branches
# ================================================================
def test_coeff_rejects_bad_sign():
    """A sign other than +/-1 is rejected."""
    with pytest.raises(ValueError, match="sign"):
        Coeff(const=1, sign=0)


def test_coeff_rejects_param_and_aux():
    """Reject param and aux as mutually exclusive sources."""
    with pytest.raises(ValueError, match="one of param"):
        Coeff(param=STRATIFICATION_N2, aux="f_coriolis")


def test_coeff_aux_needs_const():
    """An aux source needs its backing const ParamName."""
    with pytest.raises(ValueError, match="needs const"):
        Coeff(aux="f_coriolis")


def test_coeff_needs_a_source():
    """A coeff with no runtime source is rejected."""
    with pytest.raises(ValueError, match="runtime source"):
        Coeff()


def test_coeff_literal_negative_sign(sw_model):
    """The literal ``sign<0`` branch negates the scaled field."""
    state = sw_model.state
    p = state["p"]
    result = Coeff(const=2, sign=-1).apply_numeric(p, state, None)
    assert _same(result, p.with_data(-2 * p.data))


# ================================================================
#  Operator specs — composition guard
# ================================================================
def test_operator_spec_matmul_rejects_non_spec():
    """``@`` with a non-spec raises TypeError (NotImplemented path)."""
    with pytest.raises(TypeError):
        _ = Diff("x") @ 5


def test_term_blocks_out_must_advance():
    """A block whose out is not advanced is an authoring error."""
    with pytest.raises(ValueError, match="not in advances"):
        fr.term(
            name="bad", advances=("u",), linear=True,
            blocks=(LinearBlock("v", "u", Interp(), Coeff(const=1)),),
        )(lambda *_: {})


def test_term_blocks_repr():
    """The block count shows in the term repr."""
    block = LinearBlock("u", "v", Interp(), Coeff(const=1))
    stamped = fr.term(
        name="t", advances=("u",), linear=True, blocks=(block,),
    )(lambda *_: {})
    decl = getattr(stamped, TERM_ATTRIBUTE)
    assert "blocks=1" in repr(decl)


def test_scale_needs_csqr_param():
    """Scale exposes its aux name and backing const."""
    scale = Scale("csqr", sw_params.CSQR)
    assert scale.aux == "csqr"
    assert scale.const == sw_params.CSQR


# ================================================================
#  R13 — auto-derived linear-term halo (op-only trace)
# ================================================================
def _spaces_and_registry(model):
    """Return the (name -> bare space) map and the merged registry."""
    spaces = {name: model.state[name].function_space.bare
              for name in model.state.component_names}
    return spaces, model.grid.dispatch


def _width(spec, name):
    """Ghost width along `name` (0 where the spec omits the name)."""
    return dict(spec.widths).get(name, 0)


def test_linear_blocks_halo_stratification_is_z_only(nh_model):
    """Buoyancy/restoring w-b interp demands only ``{z: 1}`` (R13).

    The hand-declared ``extra_halo`` was the over-conservative
    ``{x: 1, y: 1, z: 1}``; the derived demand is tighter — w is
    Staggered("z"), b is Collocated, so only the vertical face
    interpolation carries a stencil.
    """
    spaces, reg = _spaces_and_registry(nh_model)
    for blocks in (_BUOYANCY_BLOCKS, _RESTORING_BLOCKS):
        spec = linear_blocks_halo(blocks, spaces, reg)
        assert _width(spec, "z") == 1
        assert _width(spec, "x") == 0
        assert _width(spec, "y") == 0


def test_linear_blocks_halo_coriolis_is_xy(nh_model):
    """The u-v rotation interp demands ``{x: 1, y: 1}`` (R13)."""
    spaces, reg = _spaces_and_registry(nh_model)
    spec = linear_blocks_halo(_CORIOLIS_BLOCKS, spaces, reg)
    assert _width(spec, "x") == 1
    assert _width(spec, "y") == 1
    assert _width(spec, "z") == 0


def test_linear_blocks_halo_sw_gravity_traces_scale(sw_model):
    """The ``Diff @ Scale`` gravity blocks trace op-only (no state raise).

    The flux-form ``Scale`` factor reads the constant ``csqr`` AUX
    field off the tracer state (a halo-0 broadcast), and the pressure
    gradients/divergences carry the finite-difference stencil.
    """
    spaces, reg = _spaces_and_registry(sw_model)
    spec = linear_blocks_halo(_GRAVITY_BLOCKS, spaces, reg)
    assert _width(spec, "x") == 1
    assert _width(spec, "y") == 1


def test_negotiated_halo_covers_derived_linear_demand(nh_model):
    """The negotiated halo is >= every linear term's derived demand.

    Never smaller where a real stencil exists (the R13 lower-bound
    guarantee): the tightened per-term demand still fits under the
    model-wide negotiated width (dominated by advection).
    """
    spaces, reg = _spaces_and_registry(nh_model)
    negotiated = nh_model.grid.fingerprint.halo
    for blocks in (_BUOYANCY_BLOCKS, _RESTORING_BLOCKS, _CORIOLIS_BLOCKS):
        demand = linear_blocks_halo(blocks, spaces, reg)
        for name, width in demand.widths:
            assert _width(negotiated, name) >= width


def test_linear_modules_declare_no_extra_halo():
    """R13: the linear physics modules stop hand-writing extra_halo."""
    assert getattr(
        nh.modules.ConstantStratification(), "extra_halo", None) is None
    assert getattr(
        nh.modules.CenteredAdvection(), "extra_halo", None) is None
