"""Tests for MappedIntervalMesh (spatial/meshes/mapped_interval.py)."""
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh

N = 8


def tanh_map(s):
    return jnp.tanh(2.0 * s) / jnp.tanh(2.0)


def wavy_map(s):
    return s + 0.1 * jnp.sin(2.0 * jnp.pi * s) / (2.0 * jnp.pi)


@pytest.fixture
def bounded():
    return MappedIntervalMesh(N, (0.0, 1.0), tanh_map, name="z")


@pytest.fixture
def periodic():
    return MappedIntervalMesh(N, (0.0, 1.0), wavy_map,
                              periodic=True, name="p")


# ================================================================
#  Construction and static surface
# ================================================================
def test_periodic_default_false():
    mesh = MappedIntervalMesh(N, (0.0, 1.0), tanh_map, name="z")
    assert mesh.periodic is False


def test_mapping_is_stored_by_identity(bounded):
    assert bounded.mapping is tanh_map
    assert bounded.coordinate_map is tanh_map


def test_uniform_seam_default_is_none():
    assert IntervalMesh(N, (0.0, 1.0), name="x").coordinate_map is None


def test_no_scalar_dx(bounded):
    # asking for one is the bug the metric-field design prevents
    assert not hasattr(bounded, "dx")


def test_mapping_must_be_callable():
    with pytest.raises(TypeError, match="callable"):
        MappedIntervalMesh(N, (0.0, 1.0), 0.125, name="z")


def test_mapping_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        MappedIntervalMesh(N, (0.0, 1.0),
                           lambda s: jnp.sin(2 * jnp.pi * s),
                           name="z")


def test_mapping_must_be_elementwise():
    with pytest.raises(ValueError, match="elementwise"):
        MappedIntervalMesh(N, (0.0, 1.0), jnp.sum, name="z")


def test_mapping_must_be_finite():
    with pytest.raises(ValueError, match="finite"):
        MappedIntervalMesh(N, (0.0, 1.0),
                           lambda s: 1.0 / jnp.asarray(s),
                           name="z")


def test_mapping_endpoints_must_match_extent():
    with pytest.raises(ValueError, match="endpoints"):
        MappedIntervalMesh(N, (0.0, 1.0), lambda s: 2.0 * s,
                           name="z")


# ================================================================
#  Space family (full nodal/average surface, interned per mesh)
# ================================================================
def test_nodal_and_average_families_intern(bounded):
    assert bounded.center is bounded.center
    assert bounded.outer is bounded.outer
    assert bounded.inner is bounded.inner
    assert bounded.cell_avg is bounded.cell_avg
    assert bounded.face_avg is bounded.face_avg
    assert bounded.center.shape == (N,)
    assert bounded.outer.shape == (N + 1,)


def test_periodic_family(periodic):
    assert periodic.right is periodic.right
    assert periodic.face_avg.shape == (N,)
    with pytest.raises(ValueError, match="bounded"):
        _ = periodic.outer


def test_spaces_of_two_equal_meshes_do_not_conflate():
    a = MappedIntervalMesh(N, (0.0, 1.0), tanh_map, name="z")
    b = MappedIntervalMesh(N, (0.0, 1.0), tanh_map, name="z")
    assert a is not b
    assert a.center is not b.center


# ================================================================
#  Coefficient factories are deferred (stage C2)
# ================================================================
def test_fourier_is_deferred(periodic):
    with pytest.raises(NotImplementedError, match="deferred"):
        periodic.fourier(periodic.center)


def test_sine_and_cosine_are_deferred(bounded):
    with pytest.raises(NotImplementedError, match="deferred"):
        bounded.sine(bounded.center)
    with pytest.raises(NotImplementedError, match="deferred"):
        bounded.cosine(bounded.center)


# ================================================================
#  Refinement (same mapping, scaled cell count)
# ================================================================
def test_refined_keeps_the_mapping(bounded):
    finer = bounded.refined(Fraction(3, 2))
    assert isinstance(finer, MappedIntervalMesh)
    assert finer.n_cells == 12
    assert finer.mapping is bounded.mapping
    assert finer.extent == bounded.extent
    assert finer.refined_from is bounded
    assert bounded.refined(Fraction(3, 2)) is finer


# ================================================================
#  Mapping evaluation (node placement is grid-owned; the mesh
#  stores the function only)
# ================================================================
def test_mapping_matches_analytic_samples(bounded):
    s = np.linspace(0.0, 1.0, 2 * N + 1)
    assert np.allclose(np.asarray(bounded.mapping(s)),
                       np.tanh(2.0 * s) / np.tanh(2.0))
