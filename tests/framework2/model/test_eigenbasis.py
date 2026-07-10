"""The shared labeled-eigenbasis machinery (fr.model.eigenbasis).

Covers the host-side labeling helpers the package labelers share
(segment energy, degenerate-cluster recovery, slow/fast band split),
the selection-string vocabulary derivation, the wrapper base's
projector entry, and the ``fr.eigenbasis`` package dispatch. The
engine projection apply itself is exercised end to end by the
package suites (``tests/framework2/shallowwater2`` and
``tests/framework2/nonhydro2``).
"""
from types import SimpleNamespace

import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.shallowwater2 as sw
from fridom.framework2.model.eigenbasis import (
    ChannelEigenmodesBase,
    _selection_map,
    eigenbasis,
    family_projection,
    recover_crisp_column,
    segment_energy,
    split_frequency_bands,
)

N = 8
SEG = slice(2, 4)  # the "wall-normal" segment of the toy columns
D = 6


# ================================================================
#  Helper fixtures: toy planes
# ================================================================
def unit(d, idx):
    """Return a complex unit vector of length d."""
    e = np.zeros(d, dtype=complex)
    e[idx] = 1.0
    return e


def make_model():
    """Build a small walled shallow-water channel model."""
    mx = fr.grid.meshes.IntervalMesh(N, (0.0, 1.0), periodic=True,
                                     name="x")
    my = fr.grid.meshes.IntervalMesh(N, (0.0, 1.0), periodic=False,
                                     name="y")
    return sw.Model(
        grid=fr.grid.Grid((mx, my)), csqr=0.7, rossby_number=0.2,
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0), advection=False,
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))


@pytest.fixture(scope="module")
def channel():
    """One labeled sw channel eigenbasis through fr.eigenbasis."""
    model = make_model()
    return model, eigenbasis(model)


# ================================================================
#  segment_energy
# ================================================================
def test_segment_energy_reads_the_weighted_segment():
    q = np.stack([unit(D, 0), (unit(D, 2) + unit(D, 3)) / np.sqrt(2),
                  unit(D, 3)], axis=1)
    metric = np.array([1.0, 1.0, 2.0, 4.0, 1.0, 1.0])
    energy = segment_energy(q, metric, SEG)
    assert np.allclose(energy, [0.0, 3.0, 4.0])


# ================================================================
#  recover_crisp_column
# ================================================================
def test_recovery_rotates_a_mixed_degenerate_pair():
    free = unit(D, 0)          # segment-free direction
    full = unit(D, 2)          # segment-full direction
    q = np.stack([(free + full) / np.sqrt(2),
                  (free - full) / np.sqrt(2), unit(D, 4)], axis=1)
    omega = np.array([1.0, 1.0, 2.0])
    col = recover_crisp_column(
        q, omega, np.array([True, True, False]), np.ones(D), SEG,
        energy_tol=1e-20, degeneracy_tol=1e-10)
    assert col == 0
    # the first column is now the segment-free direction (unitary)
    assert abs(abs(q[:, 0] @ np.conj(free)) - 1.0) < 1e-12
    assert segment_energy(q, np.ones(D), SEG)[0] < 1e-25


def test_recovery_skips_non_degenerate_and_full_clusters():
    # non-degenerate frequencies: no cluster to rotate
    q = np.stack([unit(D, 2), unit(D, 3)], axis=1)
    assert recover_crisp_column(
        q, np.array([1.0, 2.0]), np.array([True, True]),
        np.ones(D), SEG, energy_tol=1e-20,
        degeneracy_tol=1e-10) is None
    # degenerate but segment-full 2-space: no free direction
    before = q.copy()
    assert recover_crisp_column(
        q, np.array([1.0, 1.0]), np.array([True, True]),
        np.ones(D), SEG, energy_tol=1e-20,
        degeneracy_tol=1e-10) is None
    assert np.abs(q - before).max() == 0.0


# ================================================================
#  split_frequency_bands
# ================================================================
SLOW, FAST_P, FAST_M, EMPTY = 0, 3, 4, -1


def split(omega, rest, n_fast, gap_ratio=10.0):
    """Run the band split on a toy plane; return the labels."""
    labels = np.full(len(omega), EMPTY, dtype=np.int32)
    split_frequency_bands(
        labels, np.asarray(omega, dtype=float), np.asarray(rest),
        n_fast=n_fast, gap_ratio=gap_ratio, slow_code=SLOW,
        fast_plus_code=FAST_P, fast_minus_code=FAST_M)
    return labels


def test_exact_fast_count_labels_by_sign():
    labels = split([-2.0, -1.0, 0.0, 1.0, 2.0],
                   [True, True, False, True, True], n_fast=4)
    assert (labels == [FAST_M, FAST_M, EMPTY, FAST_P, FAST_P]).all()


def test_clean_gap_splits_the_slow_band():
    labels = split([-3.0, -0.01, 0.02, 2.0],
                   [True, True, True, True], n_fast=2)
    assert (labels == [FAST_M, SLOW, SLOW, FAST_P]).all()


def test_no_gap_and_short_planes_stay_unlabeled():
    # 0.5 vs 1.0 is no factor-10 gap
    labels = split([-1.0, 0.5, 1.0], [True, True, True], n_fast=2)
    assert (labels == EMPTY).all()
    # fewer columns than the structural fast count
    labels = split([-1.0, 1.0], [True, True], n_fast=4)
    assert (labels == EMPTY).all()


# ================================================================
#  The selection vocabulary
# ================================================================
def test_selection_map_derives_unsigned_pairs():
    families = {"vortical": 0, "kelvin+": 1, "kelvin-": 2,
                "wave+": 3, "wave-": 4, "constraint": 5}
    selections = _selection_map(families)
    assert selections["vortical"] == ("vortical",)
    assert selections["constraint"] == ("constraint",)
    assert selections["kelvin"] == ("kelvin+", "kelvin-")
    assert selections["wave"] == ("wave+", "wave-")
    assert selections["wave+"] == ("wave+",)
    # a lone signed branch still gets its (single-entry) unsigned name
    assert _selection_map({"rossby+": 7})["rossby"] == ("rossby+",)


# ================================================================
#  The wrapper base surface (through the sw package subclass)
# ================================================================
def test_base_projector_entry_and_errors(channel):
    _, em = channel
    assert isinstance(em, ChannelEigenmodesBase)
    assert isinstance(em, sw.ChannelEigenmodes)
    with pytest.raises(TypeError, match="family name"):
        em.projector(3.5)
    with pytest.raises(ValueError, match="unknown family selection"):
        family_projection(em, "rossby")
    with pytest.raises(ValueError, match="boolean mask"):
        em.projector(lambda om, _labels: om)


def test_family_projection_carries_the_model_spaces(channel):
    model, em = channel
    proj = em.projector("vortical")
    assert proj.domain is proj.codomain
    assert proj.domain.grid is model.grid
    assert proj.domain.names == em.components
    spaces = dict(proj.domain.components)
    for name in em.components:
        assert spaces[name] is em.spaces[name]


# ================================================================
#  fr.eigenbasis: the package dispatch
# ================================================================
def test_dispatch_returns_the_package_wrapper(channel):
    _, em = channel
    assert isinstance(em, sw.ChannelEigenmodes)
    assert em.families == sw.ChannelEigenmodes.families


def test_dispatch_forwards_at_time(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        sw, "eigenbasis",
        lambda model, *, at_time: seen.update(
            model=model, at_time=at_time))
    model = make_model()
    eigenbasis(model, at_time=2.5)
    assert seen["model"] is model
    assert seen["at_time"] == 2.5


def test_dispatch_rejects_a_vocabulary_free_state():
    # a plain VectorField state carries no model-package vocabulary
    model = make_model()
    plain = fr.grid.VectorField(
        {"u": model.state["u"], "v": model.state["v"],
         "p": model.state["p"]})
    stub = SimpleNamespace(state=plain)
    with pytest.raises(ValueError, match="state vocabulary"):
        eigenbasis(stub)


def test_dispatch_rejects_a_package_without_the_surface(monkeypatch):
    monkeypatch.setattr(sw, "eigenbasis", None)
    with pytest.raises(ValueError, match="no 'eigenbasis' surface"):
        eigenbasis(make_model())


def test_fr_top_level_export_is_the_dispatcher():
    assert fr.eigenbasis is eigenbasis


# ================================================================
#  The abstract vocabulary contract
# ================================================================
def test_base_class_requires_a_family_vocabulary():
    with pytest.raises(TypeError, match="abstract"):
        ChannelEigenmodesBase(object())
