"""The shared labeled-eigenbasis machinery (fr.model.eigenbasis).

Covers the host-side labeling helpers the package labelers share
(segment energy, degenerate-cluster recovery, slow/fast band split),
the selection-string vocabulary derivation, the wrapper base's
projector and ``function(f, sel)`` entries, and the
``fr.model.eigenbasis`` package dispatch. The engine projection apply
itself is exercised end to end by the package suites
(``tests/framework2/shallowwater2`` and
``tests/framework2/nonhydro2``).
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model.eigenbasis import (
    ChannelEigenmodesBase,
    _node_count,
    _ordered_family_columns,
    _selection_map,
    channel_random_state,
    eigenbasis,
    family_projection,
    recover_crisp_column,
    segment_energy,
    split_frequency_bands,
)
from fridom.model.transforms.projection import EigenFunction

N = 8
SEG = slice(2, 4)  # the "wall-normal" segment of the toy columns
D = 6
COMPONENTS = ("u", "v", "p")


# ================================================================
#  Helper fixtures: toy planes
# ================================================================
def unit(d, idx):
    """Return a complex unit vector of length d."""
    e = np.zeros(d, dtype=complex)
    e[idx] = 1.0
    return e


def make_model(device_ids=None):
    """Build a small walled shallow-water channel model."""
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), periodic=True,
                                     name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), periodic=False,
                                     name="y")
    return sw.Model(
        grid=fr.spatial.Grid((mx, my), device_ids=device_ids),
        csqr=0.7, rossby_number=0.2,
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0), advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


@pytest.fixture(scope="module")
def channel():
    """One labeled sw channel eigenbasis through fr.model.eigenbasis."""
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
#  The mode-indexed accessor (eb.mode) and the random-state engine
# ================================================================
def _mode_pair(em, family, indices, branch=None, phase=0.0):
    """Return the mode and its quarter-period phase shift."""
    omega, z0 = em.mode(family, indices, branch=branch, phase=phase)
    _, z1 = em.mode(family, indices, branch=branch,
                    phase=phase + np.pi / 2)
    return omega, z0, z1


@pytest.mark.parametrize(("family", "branch", "indices"), [
    pytest.param("kelvin", 1, {"x": 2, "y": 0}, id="kelvin-plus"),
    pytest.param("wave-", None, {"x": 3, "y": 1}, id="wave-minus"),
    pytest.param("vortical", None, {"x": 1, "y": 2}, id="vortical"),
])
def test_mode_satisfies_the_strong_eigen_relation(
        channel, family, branch, indices):
    # d/dt state(phase) == omega * state(phase + pi/2) through the
    # REAL model tendency
    model, em = channel
    omega, z0, z1 = _mode_pair(em, family, indices, branch=branch,
                               phase=0.3)
    assert isinstance(z0, sw.State)
    tau = model.tendency(z0)
    residual = max(
        float(np.abs(np.asarray(tau[c].data)
                     - omega * np.asarray(z1[c].data)).max())
        for c in em.components)
    assert residual < 1e-11 * (1.0 + abs(omega))


def test_mode_normalization_and_realness(channel):
    _, em = channel
    _, z0, z1 = _mode_pair(em, "wave", {"x": 2, "y": 0}, branch=1)
    peak = max(
        float((np.asarray(z0[c].data) ** 2
               + np.asarray(z1[c].data) ** 2).max())
        for c in ("u", "v"))
    assert peak == pytest.approx(1.0, abs=1e-12)
    for c in em.components:
        assert not np.iscomplexobj(np.asarray(z0[c].data))
        assert float(np.abs(np.asarray(z0[c].data)).max()) <= 1 + 1e-12


def test_mode_projectors_confirm_the_family(channel):
    _, em = channel
    _, z = em.mode("kelvin+", {"x": 2, "y": 0})
    kept = em.projector("kelvin+")(z)
    assert max(
        float(np.abs(np.asarray(kept[c].data)
                     - np.asarray(z[c].data)).max())
        for c in em.components) < 1e-12
    for other in ("vortical", "wave"):
        killed = em.projector(other)(z)
        assert max(
            float(np.abs(np.asarray(killed[c].data)).max())
            for c in em.components) < 1e-12


def test_mode_vortical_ordinals_follow_the_node_count(channel):
    _, em = channel
    omega = np.asarray(em.omega)[2]
    labels = np.asarray(em.labels)[2]
    q = np.asarray(em.q)[2]
    cols = _ordered_family_columns(em, "vortical", labels, omega, q)
    counts = [
        _node_count(q[:, c], em.components, em.slices,
                    np.asarray(em.metric)) for c in cols]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_mode_error_paths(channel):
    _, em = channel
    with pytest.raises(ValueError, match="unknown mode family"):
        em.mode("wave", {"x": 1, "y": 0})
    with pytest.raises(ValueError, match="signed family branch"):
        em.mode("wave", {"x": 1, "y": 0}, branch=2)
    with pytest.raises(ValueError, match="no signed branches"):
        em.mode("vortical", {"x": 1, "y": 0}, branch=1)
    with pytest.raises(ValueError, match="keyed by the grid axes"):
        em.mode("vortical", {"x": 1})
    with pytest.raises(ValueError, match="half"):
        em.mode("vortical", {"x": N, "y": 0})
    with pytest.raises(ValueError, match="ordinal"):
        em.mode("vortical", {"x": 1, "y": -1})
    with pytest.raises(ValueError, match="holds"):
        em.mode("vortical", {"x": 1, "y": 99})
    with pytest.raises(ValueError, match="structurally absent"):
        em.mode("kelvin+", {"x": 0, "y": 0})


def test_mode_on_a_self_conjugate_plane_is_a_real_steady_mode(
        channel):
    # kx = 0 sits on a self-conjugate half-spectrum plane: the
    # placement closes into the conjugate pair and the synthesized
    # vortical mode is real and exactly steady
    model, em = channel
    omega, z = em.mode("vortical", {"x": 0, "y": 1})
    assert abs(omega) < 1e-8
    tau = model.tendency(z)
    assert max(
        float(np.abs(np.asarray(tau[c].data)).max())
        for c in em.components) < 1e-11
    for c in em.components:
        assert not np.iscomplexobj(np.asarray(z[c].data))


def test_node_count_of_a_zero_column_is_zero(channel):
    _, em = channel
    zero = np.zeros(np.asarray(em.q).shape[-2], dtype=complex)
    assert _node_count(zero, em.components, em.slices,
                       np.asarray(em.metric)) == 0


def test_random_state_engine_rejects_unknown_selections(channel):
    _, em = channel
    with pytest.raises(ValueError, match="unknown or nonphysical"):
        channel_random_state(
            em, "rossby", lambda *_k: 1.0, seed=1,
            horizontal=("x", "y"))


# ================================================================
#  fr.model.eigenbasis: the package dispatch
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
    plain = fr.spatial.VectorField(
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
    assert fr.model.eigenbasis is eigenbasis


# ================================================================
#  The abstract vocabulary contract
# ================================================================
def test_base_class_requires_a_family_vocabulary():
    with pytest.raises(TypeError, match="abstract"):
        ChannelEigenmodesBase(object())


# ================================================================
#  function(f, sel): scalar functions of the linear operator
# ================================================================
def random_state(model, seed):
    """Write seeded random data onto the model's components."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        name: rng.standard_normal(
            np.asarray(model.state[name].data).shape)
        for name in COMPONENTS})
    return sw.State({name: model.state[name]
                     for name in COMPONENTS})


def absmax(a, b):
    """Componentwise max absolute difference of two states."""
    return max(
        float(np.abs(np.asarray(a[c].data)
                     - np.asarray(b[c].data)).max())
        for c in COMPONENTS)


@pytest.mark.parametrize("sel", [
    pytest.param("wave", id="family"),
    pytest.param("kelvin+", id="signed-branch"),
    pytest.param(lambda om, _labels: jnp.abs(om) > 0.5,
                 id="predicate"),
])
def test_function_with_unit_f_reproduces_the_projector(channel, sel):
    # f == 1 on a selection is exactly the selection's projector
    # (identical engine path, weights 1 on the mask), bitwise
    model, em = channel
    z = random_state(model, seed=51)
    got = em.function(np.ones_like, sel)(z)
    want = em.projector(sel)(z)
    for c in COMPONENTS:
        assert np.array_equal(np.asarray(got[c].data),
                              np.asarray(want[c].data))


def test_function_inverse_l_strong_test(channel):
    # THE STRONG TEST: with invL = function(1/(i omega), wave),
    # L(invL(z)) == P_wave(z) through the real model tendency (the
    # same operator the engine probed; advection off => linear)
    model, em = channel
    z = random_state(model, seed=52)
    inv = em.function(lambda om: 1.0 / (1j * om), "wave")
    tau = model.tendency(inv(z))
    want = em.projector("wave")(z)
    scale = max(float(np.abs(np.asarray(want[c].data)).max())
                for c in COMPONENTS)
    assert absmax(tau, want) / scale < 1e-12


def test_function_output_is_real_on_real_states(channel):
    # 1/(i omega) satisfies f(-omega) == conj(f(omega)) and the wave
    # selection is conjugation-closed: real states map to real states
    model, em = channel
    z = random_state(model, seed=53)
    out = em.function(lambda om: 1.0 / (1j * om), "wave")(z)
    for c in COMPONENTS:
        assert not np.iscomplexobj(np.asarray(out[c].data))


def test_function_transform_surface(channel):
    model, em = channel
    tr = em.function(lambda om: 1j * om, "wave")
    assert isinstance(tr, EigenFunction)
    assert tr.eigenmodes is em
    assert tr.domain is tr.codomain
    assert tr.domain.grid is model.grid
    assert repr(tr) == "f[wave]"
    # f(L) is not a projector: the idempotent-gated sugar refuses
    assert not tr.idempotent
    with pytest.raises(TypeError, match="idempotent"):
        _ = tr.complement


def test_function_empty_selection_is_the_zero_map(channel):
    model, em = channel
    z = random_state(model, seed=54)
    out = em.function(np.ones_like,
                      lambda om, _labels: jnp.abs(om) > 1e30)(z)
    assert max(float(np.abs(np.asarray(out[c].data)).max())
               for c in COMPONENTS) == 0.0


def test_function_structural_zero_guard_names_the_family(channel):
    # the engine's zero frequencies are numerical (~1e-14), so the
    # guard probes f(0) on the |omega| < 1e-8 columns: a singular f
    # on the vortical family is a taught error, never a floor
    _, em = channel
    with pytest.raises(ValueError,
                       match=r"'vortical' column at plane"):
        em.function(lambda om: 1.0 / (1j * om), "vortical")
    # ... and through a predicate covering the zero columns too
    with pytest.raises(ValueError, match="non-finite"):
        em.function(lambda om: 1.0 / (1j * om),
                    lambda om, _labels: jnp.abs(om) >= 0.0)
    # a finite f on the zero columns is fine
    assert isinstance(em.function(np.ones_like, "vortical"),
                      EigenFunction)


def test_function_guards_nonfinite_values_off_the_zero_band(channel):
    # the evaluation guard also catches an f that blows up at
    # NONZERO selected frequencies (f(0) alone is not the gate)
    _, em = channel

    def blows_up(om):
        return np.where(np.abs(om) > 0.5, np.inf, 1.0)

    with pytest.raises(ValueError, match="non-finite"):
        em.function(blows_up, "wave")


def test_function_rejects_bad_selections(channel):
    _, em = channel
    with pytest.raises(TypeError, match="family name"):
        em.function(np.ones_like, 3.5)
    with pytest.raises(ValueError, match="unknown family selection"):
        em.function(np.ones_like, "rossby")
    with pytest.raises(ValueError, match="boolean mask"):
        em.function(np.ones_like, lambda om, _labels: om)


@pytest.mark.multi_device
def test_function_application_is_device_count_invariant(
        forced_devices):
    # the weighted contraction runs through the same sharded engine
    # path as the projections: the many-device application matches
    # the explicit one-device grid
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    rng = np.random.default_rng(61)
    fields = {"u": rng.standard_normal((N, N)),
              "v": rng.standard_normal((N, N - 1)),
              "p": rng.standard_normal((N, N))}
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        model = make_model(device_ids=device_ids)
        model.set_fields(**fields)
        z = sw.State({c: model.state[c] for c in COMPONENTS})
        em = eigenbasis(model)
        results[tag] = em.function(
            lambda om: 1.0 / (1j * om), "wave")(z)
        if tag == "many":
            out = results[tag]["u"]._data
            assert len(out.sharding.device_set) == jax.device_count()
    assert absmax(results["many"], results["one"]) == 0.0
