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

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.terms import term
from fridom.model.transforms.errors import SignatureMismatchError
from fridom.model.transforms.projection import EigenProjection
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

DT = 0.02

COMPONENTS = ("u", "v", "w", "b")

N = 8
F0, N2, DSQR = 1.5, 3.0, 2.0


def make_grid(n=8, length=2 * np.pi):
    # device_ids=(0,) keeps every axis local: the analytic WaveVortex
    # projections synthesize through the naive (GSPMD) transform, a
    # Tier-1 taught error on a sharded transform axis (see transform.py).
    # The sharded fused-projection path is covered separately by
    # test_channel_projection_runs_on_a_sharded_periodic_axis.
    return Grid(tuple(
        IntervalMesh(n, (0.0, length), periodic=True, name=name)
        for name in ("x", "y", "z")), device_ids=(0,))


def _model():
    # the f0 = 1 f-plane the old implicit coriolis=None default
    # installed, now named explicitly (rotation is opt-in)
    return nh.Model(grid=make_grid(), dt=DT, advection=False,
                    coriolis=nh.FPlaneCoriolis(f0=1.0))


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
        fr.model.transforms.assert_idempotent(build.from_model(model), z)


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


def test_divergence_projection_annihilates_the_nyquist_strata():
    # SEMANTICS (Nyquist completion): the even-grid Nyquist steady
    # strata belong to the vortical family — the divergence
    # complement no longer captures them
    model = _model()
    em = nh.eigenmodes.from_model(model)
    for indices in ({"x": N // 2, "y": 1, "z": 2},
                    {"x": N // 2, "y": 2, "z": N // 2}):
        _, z = em.mode("vortical", indices)
        kept = nh.transforms.VorticalProjection(em)(z)
        div = nh.transforms.DivergenceProjection(em)(z)
        scale = max(float(np.abs(np.asarray(z[c].data)).max())
                    for c in COMPONENTS)
        assert _absmax(kept, z) < 1e-12 * scale
        assert max(float(np.abs(np.asarray(div[c].data)).max())
                   for c in COMPONENTS) < 1e-12 * scale


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
    # b collocated (bare spaces are interned, == is identity). The
    # default periodic model is finite-volume (F3): the factory flips
    # the grid default to "fv", so a bare Staggered/Collocated pattern
    # resolves to the FV C-grid spaces the eigenmodes carry
    grid = model.grid
    spaces = dict(sig.components)
    assert spaces["u"] == fr.spatial.Staggered("x").resolve(grid).bare
    assert spaces["v"] == fr.spatial.Staggered("y").resolve(grid).bare
    assert spaces["w"] == fr.spatial.Staggered("z").resolve(grid).bare
    assert spaces["b"] == fr.spatial.Collocated().resolve(grid).bare


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
    center = fr.spatial.Collocated().resolve(grid)
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
#  The rest policy on a tracer-carrying state
# ================================================================
class _PassiveTracer(fr.model.Module):

    """A module declaring one prognostic passive tracer ``c``."""

    field_declarations = (fr.model.FieldDeclaration.tracer("c"),)

    @term(name="c_hold", advances=("c",))
    def hold(self, state, _ctx):
        return {"c": state["c"] * 0.0}


def test_projection_rest_zero_completes_a_passive_tracer():
    # a state extended by a prognostic passive tracer: the vortical
    # projection (rest="zero") returns the tracer as a zero field on
    # its own space, and the residual carries it fully (§10.7.2)
    model = nh.Model(grid=make_grid(), dt=DT, advection=False,
                     coriolis=nh.FPlaneCoriolis(f0=1.0),
                     modules_extra=(_PassiveTracer(),))
    _state(model, seed=13)
    rng = np.random.default_rng(14)
    shape = np.asarray(model.state["c"].data).shape
    model.set_fields(c=rng.standard_normal(shape))
    z = nh.State({c: model.state[c] for c in (*COMPONENTS, "c")})
    proj = nh.transforms.VorticalProjection.from_model(model)
    out = proj(z)
    assert out.component_names == (*COMPONENTS, "c")
    assert np.all(np.asarray(out["c"].data) == 0.0)
    assert (out["c"].function_space.bare
            == z["c"].function_space.bare)
    residual = z - out
    assert np.allclose(np.asarray(residual["c"].data),
                       np.asarray(z["c"].data))
    # the complement transform carries the tracer through unchanged
    assert np.allclose(np.asarray(proj.complement(z)["c"].data),
                       np.asarray(z["c"].data))


# ================================================================
#  The channel (engine) path: labeled family projections
# ================================================================
def make_channel_model(*, walled="y", beta=None, device_ids=(0,),
                       n=N, family=None):
    """Build the linear nonhydro channel with one bounded axis.

    ``family=None`` follows the grid default (walled auto-flips to FV
    since the 2026-07-16 ruling); pass ``family="nodal"`` to pin the
    point-value C-grid. Since stage F5 the analytic walled-vertical
    eigenmode kit runs on both families (see
    ``test_eigenbasis_topology_gates``).

    ``device_ids=(0,)`` by default keeps every axis local: the eigenbasis
    build and the analytic channel synthesis go through the naive (GSPMD)
    transform (a Tier-1 taught error on a sharded axis; see transform.py).
    The sharded fused-contraction projection path is exercised explicitly
    by ``test_channel_projection_runs_on_a_sharded_periodic_axis``, which
    passes ``device_ids=None``.
    """
    meshes = tuple(
        IntervalMesh(n, (0.0, 1.0 if name == walled else 2 * np.pi),
                     periodic=(name != walled), name=name)
        for name in ("x", "y", "z"))
    coriolis = (nh.FPlaneCoriolis(f0=F0) if beta is None
                else nh.BetaPlaneCoriolis(f0=F0, beta=beta))
    return nh.Model(
        grid=Grid(meshes, device_ids=device_ids), advection=False,
        dsqr=DSQR, coriolis=coriolis, family=family,
        stratification=nh.ConstantStratification(n2=N2),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


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
        fr.model.transforms.assert_idempotent(proj, z)
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
    # the uniform entry point dispatches instead of rejecting: a
    # fully periodic grid and the walled-vertical rigid lid both get
    # the analytic eigenmodes (rotation about the vertical keeps the
    # trigonometric basis) — on BOTH C-grid families since stage F5
    # (the FV kit mints its own BC-tagged CellAvg analysis spaces;
    # see test_fv_default::test_walled_fv_eigenmodes_build).
    assert isinstance(nh.eigenbasis(_model()),
                      nh.eigenmodes.Eigenmodes)
    for family in ("nodal", "fv"):
        walled_z = make_channel_model(walled="z", family=family)
        em = nh.eigenbasis(walled_z)
        assert isinstance(em, nh.eigenmodes.Eigenmodes)
        assert isinstance(nh.eigenmodes.from_model(walled_z),
                          nh.eigenmodes.Eigenmodes)


def test_multiwalled_grids_are_rejected():
    meshes = tuple(
        IntervalMesh(N, (0.0, 1.0), periodic=(name == "z"),
                     name=name)
        for name in ("x", "y", "z"))
    model = nh.Model(
        grid=Grid(meshes), advection=False, dsqr=DSQR,
        coriolis=nh.FPlaneCoriolis(f0=F0),
        stratification=nh.ConstantStratification(n2=N2),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    with pytest.raises(ValueError, match="multi-walled"):
        nh.eigenbasis(model)
    with pytest.raises(ValueError, match="multi-walled"):
        nh.eigenmodes.from_model(model)


# ================================================================
#  The sharded multi-device application (forced-devices gate)
# ================================================================
@pytest.mark.multi_device
def test_channel_projection_runs_on_a_sharded_periodic_axis(forced_devices):
    # The 3-D channel shards a full periodic axis (x); the per-plane
    # Fourier contraction now runs through the fused shard_map lowering
    # (spatial.operators.distributed_contract) so every FFT axis is
    # device-local when its transform runs -- instead of the plain GSPMD
    # transform that hit the upstream XLA:GPU distributed-FFT fault
    # (complex64 twiddle constants against complex128 data; see
    # multidevice_test_faults.md). The many-device projection matches the
    # explicit one-device reference to floating point, is idempotent, and
    # lands real. N = 16: below that the negotiation collapses the tiny
    # 3-D blocks onto one device and no axis shards.
    #
    # GPU-scoped: building the n=16 channel eigenbasis runs a
    # batch-144 63x63 eigh, which heap-corrupts jaxlib's CPU LAPACK on
    # many-core hosts (the T5b upstream bug); on GPU the eigh is
    # cuSOLVER and clean. Skip on the CPU backend.
    if jax.default_backend() == "cpu":
        pytest.skip(
            "channel eigenbasis batch-144 eigh heap-corrupts jaxlib's "
            "CPU LAPACK on many-core hosts (T5b); this multi-device "
            "projection gate is GPU-scoped")
    if forced_devices is not None:
        assert jax.device_count() == forced_devices

    n = 16
    rng = np.random.default_rng(12)
    fields = {"u": rng.standard_normal((n, n, n)),
              "v": rng.standard_normal((n, n - 1, n)),
              "w": rng.standard_normal((n, n, n)),
              "b": rng.standard_normal((n, n, n))}

    # many devices: x (a full periodic axis) shards, and the fused
    # distributed contraction runs the projection on the sharded grid
    many = make_channel_model(device_ids=None, n=n)
    many.set_fields(**fields)
    z_many = nh.State({c: many.state[c] for c in COMPONENTS})
    assert z_many["u"]._data.sharding.spec[0] == "devices"
    proj_many = nh.transforms.VorticalProjection(nh.eigenbasis(many))
    out_many = proj_many(z_many)
    # the synthesis lands real (the Hermitian closure)
    assert not any(
        np.iscomplexobj(np.asarray(out_many[c].data)) for c in COMPONENTS)
    # a genuine idempotent projector on the multi-device path
    assert _absmax(proj_many(out_many), out_many) < 1e-11

    # one device (device_ids=(0,)) on the same host: the replicated
    # single-device reference the distributed path must reproduce
    one = make_channel_model(device_ids=(0,), n=n)
    one.set_fields(**fields)
    z_one = nh.State({c: one.state[c] for c in COMPONENTS})
    proj_one = nh.transforms.VorticalProjection(nh.eigenbasis(one))
    out_one = proj_one(z_one)
    assert not any(
        np.iscomplexobj(np.asarray(out_one[c].data)) for c in COMPONENTS)
    assert _absmax(out_many, out_one) < 1e-11


# ================================================================
#  The fully periodic (analytic) sharded application (forced-4)
# ================================================================
def _periodic_model(device_ids, n=16):
    """Fully periodic nonhydro model at the given device layout."""
    grid = Grid(tuple(
        IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name=name)
        for name in ("x", "y", "z")), device_ids=device_ids)
    return nh.Model(
        grid=grid, dt=DT, advection=False,
        coriolis=nh.FPlaneCoriolis(f0=F0), dsqr=DSQR,
        stratification=nh.ConstantStratification(n2=N2))


@pytest.mark.multi_device
def test_analytic_projections_run_on_a_sharded_axis(forced_devices):
    # the fully periodic analytic vortical / wave / divergence
    # projections route through the fused per-mode 4x4 matrix apply
    # (spatial.operators.distributed_transform) on a grid that shards a
    # transform axis, instead of the Tier-1 taught error. The many-device
    # result matches the replicated one-device reference to floating
    # point, lands real and stays an idempotent projector.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    n = 16
    rng = np.random.default_rng(4)
    fields = {c: rng.standard_normal((n, n, n)) for c in COMPONENTS}
    many = _periodic_model(None, n)
    many.set_fields(**fields)
    z_many = nh.State({c: many.state[c] for c in COMPONENTS})
    assert z_many["u"]._data.sharding.spec[0] == "devices"
    em_many = nh.eigenmodes.from_model(many)
    one = _periodic_model((0,), n)
    one.set_fields(**fields)
    z_one = nh.State({c: one.state[c] for c in COMPONENTS})
    em_one = nh.eigenmodes.from_model(one)
    for factory in (nh.transforms.VorticalProjection,
                    nh.transforms.WaveProjection,
                    nh.transforms.DivergenceProjection):
        out_many = factory(em_many)(z_many)
        out_one = factory(em_one)(z_one)
        assert not any(
            np.iscomplexobj(np.asarray(out_many[c].data))
            for c in COMPONENTS)
        assert _absmax(out_many, out_one) < 1e-11
        assert _absmax(factory(em_many)(out_many), out_many) < 1e-10


@pytest.mark.multi_device
def test_grad_through_analytic_projection_is_finite(forced_devices):
    # jax.grad of a quadratic loss through the fused projection is finite
    # and matches a central finite difference (the per-mode matrix is a
    # constant of the loss variable; the all_to_all VJP stays finite)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    n = 8
    model = _periodic_model(None, n)
    rng = np.random.default_rng(5)
    model.set_fields(**{
        c: rng.standard_normal((n, n, n)) for c in COMPONENTS})
    base = nh.State({c: model.state[c] for c in COMPONENTS})
    proj = nh.transforms.VorticalProjection(nh.eigenmodes.from_model(model))
    u0 = jnp.asarray(base["u"].data)

    def loss(u):
        z = nh.State({
            c: (base[c].with_data(u) if c == "u" else base[c])
            for c in COMPONENTS})
        out = proj(z)
        return sum(jnp.sum(out[c].data ** 2) for c in COMPONENTS)

    grad = jax.grad(loss)(u0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0
    eps = 1e-4
    pert = jnp.asarray(rng.standard_normal((n, n, n)))
    num = (loss(u0 + eps * pert) - loss(u0 - eps * pert)) / (2 * eps)
    ana = float(jnp.sum(grad * pert))
    assert abs(num - ana) <= 1e-4 * max(1.0, abs(ana))


# ================================================================
#  Wave B: the walled-vertical analytic tier on a sharded axis
# ================================================================
@pytest.mark.multi_device
def test_walled_analytic_projections_run_on_a_sharded_axis(
        forced_devices):
    # Wave B: the walled-vertical (Fourier x Fourier x trig) analytic
    # vortical / wave / divergence projections route through the fused
    # WalledVerticalTransform region on a grid that shards a periodic
    # transform axis, instead of the Tier-1 taught error. The many-device
    # result matches the replicated one-device reference to floating
    # point, lands real and stays an idempotent projector.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    n = 8
    rng = np.random.default_rng(14)
    many = make_channel_model(walled="z", device_ids=None, n=n)
    one = make_channel_model(walled="z", device_ids=(0,), n=n)
    fields = {c: rng.standard_normal(one.state[c].data.shape)
              for c in COMPONENTS}
    many.set_fields(**fields)
    one.set_fields(**fields)
    z_many = nh.State({c: many.state[c] for c in COMPONENTS})
    z_one = nh.State({c: one.state[c] for c in COMPONENTS})
    assert z_many["u"]._data.sharding.spec[0] == "devices"
    em_many = nh.eigenmodes.from_model(many)
    em_one = nh.eigenmodes.from_model(one)
    for factory in (nh.transforms.VorticalProjection,
                    nh.transforms.WaveProjection,
                    nh.transforms.DivergenceProjection):
        out_many = factory(em_many)(z_many)
        out_one = factory(em_one)(z_one)
        assert not any(
            np.iscomplexobj(np.asarray(out_many[c].data))
            for c in COMPONENTS)
        assert _absmax(out_many, out_one) < 1e-11
        assert _absmax(factory(em_many)(out_many), out_many) < 1e-10


@pytest.mark.multi_device
def test_grad_through_walled_analytic_projection_is_finite(
        forced_devices):
    # jax.grad of a quadratic loss through the fused walled projection is
    # finite and matches a central finite difference (the trig kernels and
    # the two-all_to_all VJP stay finite; the per-mode matrix is a
    # constant of the loss variable)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    n = 8
    model = make_channel_model(walled="z", device_ids=None, n=n)
    rng = np.random.default_rng(15)
    fields = {c: rng.standard_normal(model.state[c].data.shape)
              for c in COMPONENTS}
    model.set_fields(**fields)
    base = nh.State({c: model.state[c] for c in COMPONENTS})
    proj = nh.transforms.VorticalProjection(
        nh.eigenmodes.from_model(model))
    u0 = jnp.asarray(base["u"].data)

    def loss(u):
        z = nh.State({
            c: (base[c].with_data(u) if c == "u" else base[c])
            for c in COMPONENTS})
        out = proj(z)
        return sum(jnp.sum(out[c].data ** 2) for c in COMPONENTS)

    grad = jax.grad(loss)(u0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0
    eps = 1e-4
    pert = jnp.asarray(rng.standard_normal(u0.shape))
    num = (loss(u0 + eps * pert) - loss(u0 - eps * pert)) / (2 * eps)
    ana = float(jnp.sum(grad * pert))
    assert abs(num - ana) <= 1e-4 * max(1.0, abs(ana))
