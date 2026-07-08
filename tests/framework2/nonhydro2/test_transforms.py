"""Nonhydrostatic eigenmode projections as StateTransforms (wave 7 C).

Validates the ``nh.transforms`` projections on the staggered physical
``(u, v, w, b)`` state (the discrete C-grid eigenvectors, transformed
with a plain ``fftn``): idempotency, the ``WaveProjection = P(+1) +
P(-1)`` algebra identity, and partition of unity. Unlike shallow water,
the four nonhydro components leave a genuine unbalanced residual, so
``DivergenceProjection`` is non-trivial here.
"""
import numpy as np

import fridom.framework2 as fr
import fridom.nonhydro2 as nh
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.transforms.projection import EigenProjection

DT = 0.02


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
        c: rng.standard_normal(shape) for c in ("u", "v", "w", "b")})
    return nh.State({c: model.state[c] for c in ("u", "v", "w", "b")})


def _absmax(a, b):
    return max(
        float(np.abs(np.asarray(a[c].data) - np.asarray(b[c].data)).max())
        for c in ("u", "v", "w", "b"))


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
               for c in ("u", "v", "w", "b")) > 1e-2


# ================================================================
#  The shared base: polymorphic signature, dual constructors
# ================================================================
def test_projections_are_signature_polymorphic():
    em = nh.eigenmodes.from_model(_model())
    proj = nh.transforms.VorticalProjection(em)
    assert proj.domain is None
    assert proj.codomain is None
    assert proj.modes == (0,)


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
