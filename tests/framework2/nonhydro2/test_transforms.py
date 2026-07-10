"""Nonhydrostatic eigenmode projections as StateTransforms (wave 7 C).

Validates the ``nh.transforms`` projections on the staggered physical
``(u, v, w, b)`` state (the discrete C-grid eigenvectors, round-tripped
through the grid's own per-component transforms): idempotency, the
``WaveProjection = P(+1) + P(-1)`` algebra identity, partition of
unity, the concrete staggered signature, and the Hermitian-closure
semantics of single branches on the rfft half-lattice. Unlike shallow
water, the four nonhydro components leave a genuine unbalanced
residual, so ``DivergenceProjection`` is non-trivial here.

The channel (engine) path: the same factories on a walled-y model
route to the labeled ``nh.ChannelEigenmodes`` families — the physical
families sum to the LERAY PROJECTOR (``model.constrain``), NOT the
identity; the divergence complement equals ``I - P`` (the labeled
``constraint`` family); idempotency, mutual annihilation, the energy
partition, beta-plane predicates, and the sharded multi-device
application.
"""
import jax
import jax.numpy as jnp
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

N = 8
F0, N2, DSQR = 1.5, 3.0, 2.0


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


# ================================================================
#  The channel (engine) path: labeled family projections
# ================================================================
def make_channel_model(*, walled="y", beta=None, device_ids=None,
                       n=N):
    """Build the linear nonhydro channel with one bounded axis."""
    meshes = tuple(
        IntervalMesh(n, (0.0, 1.0 if name == walled else 2 * np.pi),
                     periodic=(name != walled), name=name)
        for name in ("x", "y", "z"))
    coriolis = (nh.FPlaneCoriolis(f0=F0) if beta is None
                else nh.BetaPlaneCoriolis(f0=F0, beta=beta))
    return nh.Model(
        grid=Grid(meshes, device_ids=device_ids), advection=False,
        dsqr=DSQR, coriolis=coriolis,
        stratification=nh.ConstantStratification(n2=N2),
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))


@pytest.fixture(scope="module")
def channel():
    """One walled-y channel model + labeled eigenbasis (shared)."""
    model = make_channel_model()
    return model, nh.eigenbasis(model)


def _channel_state(model, seed):
    """Write random data onto the channel's staggered components."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        c: rng.standard_normal(np.asarray(model.state[c].data).shape)
        for c in COMPONENTS})
    return nh.State({c: model.state[c] for c in COMPONENTS})


def _m_energy(em, state):
    """Measure-weighted energy under ``diag(1, 1, dsqr, 1/N^2)``."""
    weights = {"u": 1.0, "v": 1.0, "w": DSQR, "b": 1.0 / N2}
    total = 0.0
    for c in COMPONENTS:
        mu = np.asarray(
            state[c].measure(em.bounded_axis).data)
        total += weights[c] * float(
            np.sum(np.asarray(state[c].data) ** 2 * mu))
    return total


def test_channel_physical_families_sum_to_the_leray_projector(
        channel):
    # THE completeness statement: vortical + wave + kelvin == P (the
    # Leray projector through model.constrain), NOT the identity —
    # the labeled constraint family carries exactly the complement
    model, eb = channel
    z = _channel_state(model, seed=4)
    v = nh.transforms.VorticalProjection(eb)(z)
    w = nh.transforms.WaveProjection(eb)(z)
    k = nh.transforms.KelvinProjection(eb)(z)
    total = nh.State({c: v[c] + w[c] + k[c] for c in COMPONENTS})
    projected = model.constrain(z)
    assert _absmax(total, projected) < 1e-12
    # ... and a random state genuinely carries divergence: NOT unity
    assert _absmax(total, z) > 1e-2


def test_channel_divergence_complement_is_i_minus_p(channel):
    model, eb = channel
    z = _channel_state(model, seed=5)
    projected = model.constrain(z)
    residual = nh.State({
        c: z[c].with_data(np.asarray(z[c].data)
                          - np.asarray(projected[c].data))
        for c in COMPONENTS})
    d = nh.transforms.DivergenceProjection(eb)(z)
    assert _absmax(d, residual) < 1e-12
    # the labeled constraint family IS that complement
    assert _absmax(eb.projector("constraint")(z), residual) < 1e-12


def test_channel_projections_are_idempotent_and_annihilating(
        channel):
    model, eb = channel
    z = _channel_state(model, seed=2)
    projections = {
        "vortical": nh.transforms.VorticalProjection(eb),
        "wave": nh.transforms.WaveProjection(eb),
        "kelvin": nh.transforms.KelvinProjection(eb),
        "constraint": eb.projector("constraint")}
    for proj in projections.values():
        fr.transforms.assert_idempotent(proj, z)
    parts = {name: proj(z) for name, proj in projections.items()}
    for a, proj in projections.items():
        for b, part in parts.items():
            if a == b:
                continue
            crossed = proj(part)
            assert max(
                float(np.abs(np.asarray(crossed[c].data)).max())
                for c in COMPONENTS) < 1e-12, (a, b)


def test_channel_projector_strings_equal_the_factories(channel):
    model, eb = channel
    z = _channel_state(model, seed=21)
    pairs = (
        ("vortical", nh.transforms.VorticalProjection),
        ("wave", nh.transforms.WaveProjection),
        ("kelvin", nh.transforms.KelvinProjection))
    for selection, factory in pairs:
        assert _absmax(eb.projector(selection)(z),
                       factory(eb)(z)) == 0.0


def test_channel_energy_partition(channel):
    # the families are M-orthogonal per plane and the x/z FFTs are
    # unitary up to constants, so the physical family energies sum
    # to the energy of the Leray-projected state
    model, eb = channel
    z = _channel_state(model, seed=6)
    parts = [factory(eb)(z) for factory in (
        nh.transforms.VorticalProjection, nh.transforms.WaveProjection,
        nh.transforms.KelvinProjection)]
    total = _m_energy(eb, model.constrain(z))
    assert abs(sum(_m_energy(eb, p) for p in parts)
               - total) < 1e-12 * total


def test_channel_projection_has_the_tagged_signature(channel):
    model, eb = channel
    proj = nh.transforms.VorticalProjection(eb)
    assert isinstance(proj, EigenProjection)
    assert proj.idempotent
    assert proj.domain is proj.codomain
    assert proj.domain.grid is model.grid
    proj.domain.validate_input(_channel_state(model, seed=1))


def test_channel_from_model_routes_to_the_engine_path(channel):
    model, eb = channel
    z = _channel_state(model, seed=3)
    proj = nh.transforms.WaveProjection.from_model(model)
    assert isinstance(proj.eigenmodes, nh.ChannelEigenmodes)
    assert _absmax(proj(z),
                   nh.transforms.WaveProjection(eb)(z)) == 0.0


def test_channel_beta_predicate_projection():
    # the varying-f use case: engine + a conjugation-closed
    # frequency-threshold predicate; the slow selection equals the
    # union of the (disjoint) labeled masks it covers
    model = make_channel_model(beta=0.5)
    eb = nh.eigenbasis(model)
    labels = np.asarray(eb.labels)
    omega = np.asarray(eb.omega)
    assert (labels != -1).all()
    threshold = 0.1
    z = _channel_state(model, seed=25)
    slow = eb.projector(lambda om, _labels: jnp.abs(om) < threshold)
    got = slow(z)
    for c in COMPONENTS:
        assert not np.iscomplexobj(np.asarray(got[c].data))
    assert _absmax(slow(got), got) < 1e-12
    # decompose by labels: slow == slow-vortical + constraint +
    # slow-wave (the kz-Nyquist inertial strata slide under the
    # threshold on the beta plane — the documented blur)
    wave_codes = (eb.families["wave+"], eb.families["wave-"])
    pieces = [
        eb.projector(lambda om, lab: (jnp.abs(om) < threshold)
                     & (lab == eb.families["vortical"]))(z),
        eb.projector(lambda om, lab: (jnp.abs(om) < threshold)
                     & (lab == eb.families["constraint"]))(z),
        eb.projector(lambda om, lab: (jnp.abs(om) < threshold)
                     & jnp.isin(lab, jnp.asarray(wave_codes)))(z),
    ]
    total = nh.State({
        c: pieces[0][c] + pieces[1][c] + pieces[2][c]
        for c in COMPONENTS})
    assert _absmax(got, total) < 1e-12
    # the wave piece is genuinely nonempty at this threshold
    assert ((np.abs(omega) < threshold)
            & np.isin(labels, wave_codes)).any()


def test_channel_walled_x_twin_labels_and_completeness():
    # the axis-generic wall-normal map: walls on x make u the
    # trapped-normal component; labels resolve and the physical
    # families still sum to the Leray projector
    model = make_channel_model(walled="x")
    eb = nh.eigenbasis(model)
    assert eb.bounded_axis == "x"
    labels = np.asarray(eb.labels)
    assert (labels != -1).all()
    kelvin = (labels == eb.families["kelvin+"]).sum(axis=-1)
    assert (kelvin[1:, 1:-1] == 1).all()
    z = _channel_state(model, seed=8)
    parts = [eb.projector(sel)(z)
             for sel in ("vortical", "wave", "kelvin")]
    total = nh.State({
        c: parts[0][c] + parts[1][c] + parts[2][c]
        for c in COMPONENTS})
    assert _absmax(total, model.constrain(z)) < 1e-12


# ================================================================
#  Topology gates and the eigenbasis surfaces
# ================================================================
def test_kelvin_projection_needs_horizontal_walls():
    model = _model()
    em = nh.eigenmodes.from_model(model)
    with pytest.raises(ValueError, match="no walls, no Kelvin"):
        nh.transforms.KelvinProjection(em)
    with pytest.raises(ValueError, match="no walls, no Kelvin"):
        nh.transforms.KelvinProjection.from_model(model)


def test_eigenbasis_topology_gates():
    with pytest.raises(ValueError, match="fully periodic"):
        nh.eigenbasis(_model())
    walled_z = make_channel_model(walled="z")
    with pytest.raises(ValueError, match="walled-vertical"):
        nh.eigenbasis(walled_z)
    # the analytic walled-vertical path is untouched by the dispatch
    em = nh.eigenmodes.from_model(walled_z)
    assert isinstance(em, nh.eigenmodes.Eigenmodes)


def test_multiwalled_grids_are_rejected():
    meshes = tuple(
        IntervalMesh(N, (0.0, 1.0), periodic=(name == "z"),
                     name=name)
        for name in ("x", "y", "z"))
    model = nh.Model(
        grid=Grid(meshes), advection=False, dsqr=DSQR,
        coriolis=nh.FPlaneCoriolis(f0=F0),
        stratification=nh.ConstantStratification(n2=N2),
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))
    with pytest.raises(ValueError, match="multi-walled"):
        nh.eigenbasis(model)
    with pytest.raises(ValueError, match="multi-walled"):
        nh.eigenmodes.from_model(model)


# ================================================================
#  The sharded multi-device application (forced-devices gate)
# ================================================================
@pytest.mark.multi_device
def test_channel_projection_is_device_count_invariant(forced_devices):
    # the projector application composes under the domain
    # decomposition: the partial-axis transforms (two periodic axes)
    # and the per-plane contraction run on the sharded state (no
    # host gather), and the result matches the explicit one-device
    # grid. N = 16: below that the negotiation collapses the tiny
    # 3-D blocks onto one device and nothing would be sharded.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices

    n = 16
    rng = np.random.default_rng(12)
    fields = {"u": rng.standard_normal((n, n, n)),
              "v": rng.standard_normal((n, n - 1, n)),
              "w": rng.standard_normal((n, n, n)),
              "b": rng.standard_normal((n, n, n))}
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        model = make_channel_model(device_ids=device_ids, n=n)
        model.set_fields(**fields)
        z = nh.State({c: model.state[c] for c in COMPONENTS})
        proj = nh.transforms.VorticalProjection(nh.eigenbasis(model))
        results[tag] = proj(z)
        if tag == "many":
            # genuinely sharded in and out (x is the blocked factor)
            assert z["u"]._data.sharding.spec[0] == "devices"
            out = results[tag]["u"]._data
            assert len(out.sharding.device_set) == jax.device_count()
            assert out.sharding.spec[0] == "devices"
            # idempotent on the sharded state
            twice = proj(results[tag])
            assert _absmax(twice, results[tag]) < 1e-12
    assert max(
        float(np.abs(np.asarray(results["many"][c].data)
                     - np.asarray(results["one"][c].data)).max())
        for c in COMPONENTS) < 1e-11
