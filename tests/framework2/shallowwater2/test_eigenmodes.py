"""Linear shallow-water eigenmodes: dispersion, vectors, projector."""
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.shallowwater2 as sw

from .conftest import make_grid, make_model


def _eig(f0=1.0, csqr=1.0):
    model = make_model(csqr=csqr, f0=f0)
    return sw.eigenmodes.from_model(model), model


# ================================================================
#  Dispersion relation
# ================================================================
def test_geostrophic_mode_has_zero_frequency():
    em, _ = _eig()
    assert float(np.abs(em.omega(0)).max()) == 0.0


@pytest.mark.parametrize(("f0", "csqr"), [(1.0, 1.0), (0.5, 4.0)])
def test_dispersion_matches_the_analytic_relation(f0, csqr):
    em, _ = _eig(f0=f0, csqr=csqr)
    for kx, ky in [(2 * np.pi, 0.0), (0.0, 4 * np.pi),
                   (2 * np.pi, 2 * np.pi)]:
        expect = np.sqrt(f0 ** 2 + csqr * (kx ** 2 + ky ** 2))
        assert float(em.omega(1, kx, ky)) == pytest.approx(expect)
        assert float(em.omega(-1, kx, ky)) == pytest.approx(-expect)


# ================================================================
#  Eigen-/projection-vector biorthonormality
# ================================================================
def test_projection_vectors_are_biorthonormal():
    em, _ = _eig()
    for s in (0, 1, -1):
        q = em.q(s)
        p = em.p(s)
        # p^s* . q^s == 1 on every non-trivial wavenumber
        inner = (np.conj(p["u"].data) * q["u"].data
                 + np.conj(p["v"].data) * q["v"].data
                 + np.conj(p["p"].data) * q["p"].data)
        inner = np.asarray(inner)
        nonzero = np.abs(inner) > 1e-8
        np.testing.assert_allclose(inner[nonzero], 1.0, atol=1e-6)


def test_projection_vector_is_the_metric_image_of_q():
    # p is now DERIVED as p = M q / <q, q>_M, not hand-written; assert
    # it component-wise against the energy metric diag(1, 1, 1/c^2).
    csqr = 4.0
    em, _ = _eig(csqr=csqr)
    weights = {"u": 1.0, "v": 1.0, "p": 1.0 / csqr}
    for s in (0, 1, -1):
        q = {c: np.asarray(em.q(s)[c].data) for c in ("u", "v", "p")}
        p = {c: np.asarray(em.p(s)[c].data) for c in ("u", "v", "p")}
        qq_m = sum(weights[c] * np.abs(q[c]) ** 2 for c in q)
        good = qq_m > 1e-10
        for c in ("u", "v", "p"):
            expect = np.where(
                good, weights[c] * q[c] / np.where(good, qq_m, 1.0),
                0.0)
            np.testing.assert_allclose(p[c], expect, atol=1e-12)


def test_projectors_are_idempotent_and_partition_unity():
    em, _ = _eig()
    q1 = em.q(1)
    projectors = {s: em.projector(s) for s in (0, 1, -1)}
    # P_1 is idempotent on its own eigenvector
    p1 = projectors[1](q1)
    for c in ("u", "v", "p"):
        np.testing.assert_allclose(
            np.asarray(p1[c].data), np.asarray(q1[c].data),
            atol=1e-8)
    # the other modes annihilate q1
    for s in (0, -1):
        out = projectors[s](q1)
        for c in ("u", "v", "p"):
            assert float(np.abs(out[c].data).max()) < 1e-6


# ================================================================
#  from_model structural validation
# ================================================================
def test_from_model_rejects_a_beta_plane():
    grid = make_grid()
    model = sw.Model(
        grid=grid, csqr=1.0,
        coriolis=sw.modules.BetaPlaneCoriolis(f0=1.0, beta=2.0),
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))
    with pytest.raises(ValueError, match=r"coriolis\.f0"):
        sw.eigenmodes.from_model(model)
