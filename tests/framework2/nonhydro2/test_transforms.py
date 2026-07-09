"""Nonhydrostatic eigenmode projections as StateTransforms (wave 7 C).

Validates the ``nh.transforms`` projections on the staggered physical
``(u, v, w, b)`` state (the discrete C-grid eigenvectors, round-tripped
through the grid's own per-component transforms): idempotency, the
``WaveProjection = P(+1) + P(-1)`` algebra identity, partition of
unity, the concrete staggered signature, and the Hermitian-closure
semantics of single branches on the rfft half-lattice. Unlike shallow
water, the four nonhydro components leave a genuine unbalanced
residual, so ``DivergenceProjection`` is non-trivial here.
"""
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.nonhydro2 as nh
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.transforms.errors import SignatureMismatchError
from fridom.framework2.transforms.projection import EigenProjection

DT = 0.02

COMPONENTS = ("u", "v", "w", "b")


def make_grid(n=8, length=2 * np.pi):
    return Grid(tuple(
        IntervalMesh(n, (0.0, length), periodic=True, name=name)
        for name in ("x", "y", "z")))


def _model():
    return nh.Model(grid=make_grid(), dt=DT, advection=False)


def _state(model, seed=1):
    """Build a random staggered ``(u, v, w, b)`` probe state."""
    rng = np.random.default_rng(seed)
    shape = np.asarray(model.state["u"].data).shape
    model.set_fields(**{
        c: rng.standard_normal(shape) for c in COMPONENTS})
    return nh.State({c: model.state[c] for c in COMPONENTS})


def _absmax(a, b):
    return max(
        float(np.abs(np.asarray(a[c].data) - np.asarray(b[c].data)).max())
        for c in COMPONENTS)


# ================================================================
#  Idempotency
# ================================================================
def test_all_three_projections_are_idempotent():
    model = _model()
    z = _state(model)
    for build in (nh.transforms.VorticalProjection,
                  nh.transforms.WaveProjection,
                  nh.transforms.DivergenceProjection):
        fr.transforms.assert_idempotent(build.from_model(model), z)


# ================================================================
#  The algebra
# ================================================================
def test_wave_equals_sum_of_single_mode_projections():
    model = _model()
    em = nh.eigenmodes.from_model(model)
    z = _state(model)
    wave = nh.transforms.WaveProjection(em)
    manual = (nh.transforms.mode_projection(em, 1)
              + nh.transforms.mode_projection(em, -1))
    assert _absmax(wave(z), manual(z)) < 1e-12


def test_partition_of_unity_reconstructs_the_state():
    model = _model()
    em = nh.eigenmodes.from_model(model)
    z = _state(model)
    partition = (nh.transforms.VorticalProjection(em)
                 + nh.transforms.WaveProjection(em)
                 + nh.transforms.DivergenceProjection(em))
    assert _absmax(partition(z), z) < 1e-12


def test_divergence_projection_is_non_trivial():
    # the four components are not spanned by the three mode families, so
    # the residual (divergence) projection is genuinely non-zero.
    model = _model()
    em = nh.eigenmodes.from_model(model)
    z = _state(model)
    div = nh.transforms.DivergenceProjection(em)(z)
    assert max(float(np.abs(np.asarray(div[c].data)).max())
               for c in COMPONENTS) > 1e-2


# ================================================================
#  Single branches: Hermitian closure on the rfft half-lattice
# ================================================================
def test_single_branch_is_real_and_branches_sum_to_wave():
    # a single branch on a real state applies P(s) on the stored rfft
    # half-lattice; the implicit conjugate half carries the mirrored
    # -s branch, so the output is the real Hermitian-closed field and
    # the separately applied branches still sum to the wave field.
    model = _model()
    em = nh.eigenmodes.from_model(model)
    z = _state(model)
    plus = nh.transforms.mode_projection(em, 1)(z)
    minus = nh.transforms.mode_projection(em, -1)(z)
    for c in COMPONENTS:
        assert not np.iscomplexobj(np.asarray(plus[c].data))
        assert not np.iscomplexobj(np.asarray(minus[c].data))
    wave = nh.transforms.WaveProjection(em)(z)
    assert _absmax(plus + minus, wave) < 1e-12


# ================================================================
#  The shared base: concrete staggered signature, dual constructors
# ================================================================
def test_projections_carry_the_staggered_signature():
    model = _model()
    em = nh.eigenmodes.from_model(model)
    proj = nh.transforms.VorticalProjection(em)
    sig = proj.domain
    assert sig is proj.codomain
    assert sig.grid is model.grid
    assert sig.names == COMPONENTS
    assert proj.modes == (0,)
    # each component on its OWN physical space: u/v/w face-staggered,
    # b collocated (bare spaces are interned, == is identity)
    grid = model.grid
    spaces = dict(sig.components)
    assert spaces["u"] == fr.Staggered("x").resolve(grid).bare
    assert spaces["v"] == fr.Staggered("y").resolve(grid).bare
    assert spaces["w"] == fr.Staggered("z").resolve(grid).bare
    assert spaces["b"] == fr.Collocated().resolve(grid).bare


def test_call_rejects_a_state_missing_a_mapped_component():
    model = _model()
    em = nh.eigenmodes.from_model(model)
    partial = nh.State({c: model.state[c] for c in ("u", "v", "w")})
    with pytest.raises(SignatureMismatchError, match=r"missing: \('b',\)"):
        nh.transforms.VorticalProjection(em)(partial)


def test_call_rejects_a_component_on_the_wrong_space():
    model = _model()
    em = nh.eigenmodes.from_model(model)
    grid = model.grid
    center = fr.Collocated().resolve(grid)
    collocated = nh.State({
        c: grid.create_field(center, name=c) for c in COMPONENTS})
    with pytest.raises(SignatureMismatchError,
                       match="space mismatch for 'u'"):
        nh.transforms.VorticalProjection(em)(collocated)


def test_from_model_and_explicit_agree():
    model = _model()
    em = nh.eigenmodes.from_model(model)
    z = _state(model)
    from_model = nh.transforms.WaveProjection.from_model(model)
    explicit = nh.transforms.WaveProjection(em)
    # both are the merged wave projection over the same discrete grid
    assert isinstance(from_model, EigenProjection)
    assert from_model.modes == explicit.modes == (-1, 1)
    # (different eigenmode objects, identical numerics)
    assert _absmax(from_model(z), explicit(z)) < 1e-12
