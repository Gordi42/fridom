"""The walled (rigid-lid) eigenmode battery (Phase 2, C9).

The acceptance criterion of the projections phase: the
operator-sourced eigenmodes on a grid with a bounded vertical.
Stage A — parity-tagged kit spaces, lazy symbol families, the wave
branches (analytic rigid-lid dispersion, union-lattice
biorthonormality, the strong ``L q = -i omega q`` test through the
model's own constrain stage). Stage B — the re-referenced
geostrophic column (all ``N + 1`` steady strata: the interior
geostrophic modes, the ``m = 0`` barotropic and the ``m = N``
buoyancy-top stratum), partition of unity on the represented set,
projector invariance of an exponential (cosh) boundary vortical
mode built from the grid's own discrete operators, and the
measure-weighted energy partition (Parseval).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.analytic_distributed import resolve_route
from fridom.model.context import StepContext
from fridom.model.eigen import _rest_background
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
)
from fridom.nonhydro2.state import State
from fridom.spatial.bc import BC
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.base import EigenbasisError
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.spatial.operators.interp import LinearInterp
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.coefficient import (
    CosineSpace,
    SineSpace,
)
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet
from fridom.spatial.symbols import ModeChart, rayleigh_dual

N = 8
LZ = 1.0
DT = 0.02
F0, N2, DSQR = 1.5, 3.0, 2.0
COMPONENTS = ("u", "v", "w", "b")

#: integer branch -> uniform family-name spelling (the low-level
#: q/omega surface stays integer-indexed; mode() takes families)
FAMILY = {0: "vortical", 1: "wave+", -1: "wave-"}
WEIGHTS = {"u": 1.0, "v": 1.0, "w": DSQR, "b": 1.0 / N2}


# ================================================================
#  Shared walled setup (module-scoped: compile once per worker)
# ================================================================
@pytest.fixture(scope="module", params=["nodal", "fv"])
def walled(request):
    # The walled-vertical eigenmode battery runs on BOTH C-grid
    # families: the validated point-value ("nodal") path and the
    # finite-volume ("fv") path (stage F5). On the FV family the kit
    # mints its BC-tagged CellAvg analysis siblings itself (C8 keeps
    # the *declaration* layer BC-free), and the vertical trig symbols
    # are bitwise the nodal ones (the 2nd-order FV stencils are the
    # nodal ones), so the FV eigenbasis is bit-identical to the nodal
    # one — asserted directly by
    # test_walled_fv_eigenbasis_is_bitwise_nodal.
    # device_ids=(0,) pins to one device on any device count: the
    # eigenmode projections synthesize through the naive (GSPMD)
    # transform, a Tier-1 taught error on a sharded axis (transform.py).
    grid = Grid((
        IntervalMesh(N, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(N, (0.0, 2 * np.pi), periodic=True, name="y"),
        IntervalMesh(N, (0.0, LZ), periodic=False, name="z")),
        device_ids=(0,))
    model = nh.Model(
        grid=grid, dt=DT, advection=False, dsqr=DSQR,
        coriolis=FPlaneCoriolis(f0=F0),
        stratification=ConstantStratification(n2=N2),
        family=request.param)
    em = nh.eigenmodes.from_model(model)
    return grid, model, em


def _is_fv(em):
    """Whether an eigenmode kit resolved the FV (average) family."""
    return isinstance(em.kit.coeff("b").factor("z").origin, CellAvg)


@pytest.fixture(scope="module")
def linearized(walled):
    """Build the linear variant, rest background and constrain map."""
    _, model, _ = walled
    lin = fr.model.linearize(model)
    prog, base0 = _rest_background(lin, 0.0)
    constrain = _constrain_fn(lin, base0, prog)
    return lin, prog, base0, constrain


def _constrain_fn(lin, base0, prog):
    """Bind the model's own CONSTRAINT stage as a state-wise map."""
    schedule = lin._artifacts.schedule
    modules = lin._carry.modules
    stepper = lin._stepper
    t = jnp.asarray(0.0)
    table = schedule.binding_table
    params = (table.eval_params(modules, stepper, t)
              if table is not None else {})
    ctx = StepContext(params=params, clock=t, dt=stepper.dt,
                      stage_dt=stepper.dt)
    bound = schedule.bind(modules)

    def constrain(state):
        full = base0.replace(**{name: state[name] for name in prog})
        return bound.constrain(full, ctx)

    return constrain


def _union(em, name, data):
    """Embed component-layout data onto the union mode lattice."""
    chart = ModeChart(em.grid)
    return np.asarray(chart.embed(jnp.asarray(np.asarray(data)),
                                  em.kit.coeff(name)))


def _union_column(em, s):
    return {c: _union(em, c, em.q(s)[c].data) for c in COMPONENTS}


def _dual_data(q):
    """Rayleigh dual of a union-lattice column + the represented mask."""
    norm = sum(WEIGHTS[c] * np.abs(q[c]) ** 2 for c in COMPONENTS)
    good = norm != 0
    safe = np.where(good, norm, 1.0)
    p = {c: np.where(good, WEIGHTS[c] * q[c] / safe, 0.0)
         for c in COMPONENTS}
    return p, good


# ================================================================
#  Stage A: kit spaces and the lazy symbol families
# ================================================================
def test_walled_kit_spaces_carry_the_parity_tags(walled):
    # physics-fixed z-parities: w Sine-I (Inner, Dirichlet), b Sine-II
    # (Dirichlet), u/v/p Cosine-II (Neumann). The center-like origin
    # is nodal Center on the nodal family and CellAvg on the FV family;
    # the trig family and the BC tag are identical either way (the FV
    # eigenbasis IS the nodal one), only the origin family names the
    # discretization. w is on the Inner face on both.
    _, _, em = walled
    fv = _is_fv(em)
    #: the collocated (center-like) origin of the family
    center = CellAvg if fv else NodalSpace
    rows = (("u", CosineSpace, center, BC.NEUMANN),
            ("v", CosineSpace, center, BC.NEUMANN),
            ("p", CosineSpace, center, BC.NEUMANN),
            ("w", SineSpace, NodalSpace, BC.DIRICHLET),
            ("b", SineSpace, center, BC.DIRICHLET))
    for name, trig, origin_cls, kind in rows:
        factor = em.kit.coeff(name).factor("z")
        assert isinstance(factor, trig)
        assert isinstance(factor.origin, origin_cls)
        if isinstance(factor.origin, NodalSpace):
            # w on Inner (both families); u/v/p/b Center on nodal
            expect = NodeSet.INNER if name == "w" else NodeSet.CENTER
            assert factor.origin.node_set is expect
        assert all(c is kind for c in factor.origin.bc.components)


def test_physical_spaces_match_the_model(walled):
    # the model-facing spaces (the projection signature / retag
    # targets) are exactly what the model resolves per component
    _, model, em = walled
    for c in COMPONENTS:
        assert (em.physical_space(c)
                is model.state[c].function_space.bare)


def test_periodic_kit_spaces_are_the_phase1_spaces():
    # regression: on a fully periodic grid the wall_bc channels are
    # inert and the kit resolves the identical interned spaces
    # device_ids=(0,) pins to one device on any device count: the
    # eigenmode projections synthesize through the naive (GSPMD)
    # transform, a Tier-1 taught error on a sharded axis (transform.py).
    grid = Grid(tuple(
        IntervalMesh(N, (0.0, 2 * np.pi), periodic=True, name=name)
        for name in ("x", "y", "z")), device_ids=(0,))
    em = nh.eigenmodes.Eigenmodes(grid, f0=1.0, n2=3.0, dsqr=2.0)
    phase1 = {
        "u": fr.spatial.Staggered("x").resolve(grid),
        "v": fr.spatial.Staggered("y").resolve(grid),
        "w": fr.spatial.Staggered("z").resolve(grid),
        "b": fr.spatial.Collocated().resolve(grid),
        "p": fr.spatial.Collocated().resolve(grid),
    }
    for c, space in phase1.items():
        assert em.kit.forward(c).domain is space.bare
    for c in COMPONENTS:
        assert em.physical_space(c) is phase1[c].bare


def test_periodic_projector_path_is_bitwise_phase1():
    # the chart is identity on a periodic grid: the projector's data
    # path reproduces the Phase-1 formula bitwise
    # device_ids=(0,) pins to one device on any device count: the
    # eigenmode projections synthesize through the naive (GSPMD)
    # transform, a Tier-1 taught error on a sharded axis (transform.py).
    grid = Grid(tuple(
        IntervalMesh(N, (0.0, 2 * np.pi), periodic=True, name=name)
        for name in ("x", "y", "z")), device_ids=(0,))
    em = nh.eigenmodes.Eigenmodes(grid, f0=1.0, n2=3.0, dsqr=2.0)
    rng = np.random.default_rng(3)
    template = em.q(0)
    shape = np.asarray(template["u"].data).shape
    z = State({c: template[c].with_data(jnp.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)))
        for c in COMPONENTS})
    for s in (0, 1, -1):
        got = em.projector(s)(z)
        # the Phase-1 data plane, verbatim (the even-grid vortical
        # family iterates its internal Nyquist columns)
        total = None
        for q in em._columns(s):
            p = rayleigh_dual(q, WEIGHTS)
            amp = sum(jnp.conj(p[c].data) * z[c].data for c in p)
            part = {c: q[c].data * amp for c in q}
            total = (part if total is None
                     else {c: total[c] + part[c] for c in part})
        for c in COMPONENTS:
            want = template[c].with_data(jnp.broadcast_to(
                total[c], shape).astype(template[c].data.dtype))
            assert np.array_equal(np.asarray(got[c].data),
                                  np.asarray(want.data))


def test_lazy_symbols_defer_the_skip_signal(walled):
    _, _, em = walled
    # a[z] (interp threaded on the DCT-II pressure factor) raises the
    # eigen-layer skip signal by design — and only on access
    with pytest.raises(EigenbasisError, match="skip"):
        em.a["z"]
    # the used entries build fine and memoize
    assert em.a["x"] is em.a["x"]
    assert em.kb["z"] is em.kb["z"]
    for family in (em.k, em.kb, em.a, em.ab):
        assert set(family) == {"x", "y", "z"}
        assert len(family) == 3
    with pytest.raises(KeyError):
        em.k["t"]


# ================================================================
#  Stage A: the analytic rigid-lid dispersion
# ================================================================
def _analytic_omega():
    """Build the rigid-lid dispersion from the analytic tables."""
    dx = 2 * np.pi / N
    dz = LZ / N
    kx = np.arange(N // 2 + 1, dtype=float)
    ky = np.fft.fftfreq(N, 1.0 / N)
    kz = np.pi * np.arange(1, N) / LZ
    khx = 2 * np.sin(kx * dx / 2) / dx
    khy = 2 * np.sin(ky * dx / 2) / dx
    khz = 2 * np.sin(kz * dz / 2) / dz
    ahx = np.cos(kx * dx / 2)
    ahy = np.cos(ky * dx / 2)
    ahz = np.cos(kz * dz / 2)
    kh2 = (khx[:, None, None] ** 2 + khy[None, :, None] ** 2)
    om2 = ((F0 ** 2 * ahx[:, None, None] ** 2
            * ahy[None, :, None] ** 2 * khz[None, None, :] ** 2
            + N2 * ahz[None, None, :] ** 2 * kh2)
           / (DSQR * kh2 + khz[None, None, :] ** 2))
    return np.sqrt(om2)


def test_walled_dispersion_matches_the_analytic_tables(walled):
    # same formula as the periodic dispersion, with the trig tables
    # k_hat_z = 2 sin(pi m dz / (2 Lz)) / dz and a_hat_z =
    # cos(pi m dz / (2 Lz)) on w's DST-I lattice m = 1..N-1
    _, _, em = walled
    got = np.broadcast_to(np.real(np.asarray(em.omega(1).data)),
                          (N // 2 + 1, N, N - 1))
    assert np.abs(got - _analytic_omega()).max() < 1e-12
    # branch symmetry and the zero geostrophic branch
    assert np.allclose(np.real(np.asarray(em.omega(-1).data)), -got)
    assert np.abs(np.asarray(em.omega(0).data)).max() == 0.0


def test_walled_dispersion_continuum_limit():
    # the lowest resolved mode approaches the continuum rigid-lid
    # relation (second-order discretization error); a finer
    # eigenmodes-only grid keeps the tolerance honest
    n = 16
    # device_ids=(0,) pins to one device on any device count: the
    # eigenmode projections synthesize through the naive (GSPMD)
    # transform, a Tier-1 taught error on a sharded axis (transform.py).
    grid = Grid((
        IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="y"),
        IntervalMesh(n, (0.0, LZ), periodic=False, name="z")),
        device_ids=(0,))
    em = nh.eigenmodes.Eigenmodes(grid, f0=F0, n2=N2, dsqr=DSQR)
    got = np.broadcast_to(np.real(np.asarray(em.omega(1).data)),
                          (n // 2 + 1, n, n - 1))
    kh2, kz2 = 1.0, (np.pi / LZ) ** 2
    expect = np.sqrt((F0 ** 2 * kz2 + N2 * kh2)
                     / (DSQR * kh2 + kz2))
    assert abs(got[1, 0, 0] - expect) / expect < 0.05


# ================================================================
#  Stage A + B: biorthonormality on the union mode lattice
# ================================================================
def test_walled_biorthonormality_on_the_union_lattice(walled):
    _, _, em = walled
    q = {s: _union_column(em, s) for s in (0, 1, -1)}
    for s in (0, 1, -1):
        p, good = _dual_data(q[s])
        d = sum(np.conj(p[c]) * q[s][c] for c in COMPONENTS)
        # sum_c conj(p_c) q_c == 1 wherever the mode is represented
        assert np.abs(d[good] - 1.0).max() < 1e-12
        # ... and the degenerate modes are EXACT structural zeros
        assert good.any()
        assert (~good).any()
        for c in COMPONENTS:
            assert np.all(q[s][c][~good] == 0.0)
    for s, t in [(0, 1), (0, -1), (1, -1), (1, 0), (-1, 0)]:
        p, _ = _dual_data(q[s])
        cross = sum(np.conj(p[c]) * q[t][c] for c in COMPONENTS)
        assert np.abs(cross).max() < 1e-12


def test_walled_projector_reproduces_and_is_idempotent(walled):
    _, _, em = walled
    rng = np.random.default_rng(0)
    template = em.q(0)
    z = State({c: template[c].with_data(jnp.asarray(
        rng.standard_normal(np.asarray(template[c].data).shape)
        + 1j * rng.standard_normal(
            np.asarray(template[c].data).shape)))
        for c in COMPONENTS})
    for s in (0, 1, -1):
        proj = em.projector(s)
        q = em.q(s)
        pq = proj(q)
        once = proj(z)
        twice = proj(once)
        for c in COMPONENTS:
            assert np.abs(np.asarray(pq[c].data)
                          - np.asarray(q[c].data)).max() < 1e-12
            assert np.abs(np.asarray(twice[c].data)
                          - np.asarray(once[c].data)).max() < 1e-12


# ================================================================
#  The strong test: L q(s) = -i omega(s) q(s) through the model's
#  own linearized tendency + constrain stage (state-wise, no probe)
# ================================================================
@pytest.mark.parametrize("s", [1, -1, 0])
def test_walled_strong_eigenrelation(walled, linearized, s):
    grid, _, em = walled
    lin, prog, base0, constrain = linearized
    kit = em.kit
    chart = ModeChart(grid)
    rng = np.random.default_rng(5 + s)
    q = em.q(s)
    union_shape = np.asarray(chart.embed(
        jnp.zeros(np.asarray(q["u"].data).shape),
        kit.coeff("u"))).shape
    amp = (rng.standard_normal(union_shape)
           + 1j * rng.standard_normal(union_shape))
    if s != 0:
        # on a REAL physical field the kx = 0 / Nyquist planes of the
        # rfft layout are Hermitian-mixed with the opposite wave
        # branch; probe the wave branches on interior kx only
        amp[0] = 0.0
        amp[-1] = 0.0
    coeff = State({
        c: q[c].with_data(jnp.asarray(
            np.asarray(q[c].data)
            * np.asarray(chart.restrict(jnp.asarray(amp),
                                        kit.coeff(c)))))
        for c in COMPONENTS})
    nodal = {c: kit.backward(c)(coeff[c]) for c in COMPONENTS}
    phys = base0.replace(**{
        c: base0[c].with_data(nodal[c].data) for c in COMPONENTS})
    # the honest reference: the round-tripped coefficients
    zeta = {c: np.asarray(kit.forward(c)(nodal[c]).data)
            for c in COMPONENTS}
    tau = lin.tendency(phys, t=0.0, constraints=False)
    constrained = constrain(base0.replace(
        **{c: tau[c] for c in prog}))
    omega = np.real(np.asarray(em.omega(s).data))
    om_union = np.asarray(chart.embed(
        jnp.broadcast_to(jnp.asarray(omega),
                         np.asarray(q["w"].data).shape),
        kit.coeff("w")))
    scale = max(np.abs(zeta[c]).max() for c in COMPONENTS)
    for c in COMPONENTS:
        got = _union(em, c, kit.forward(c)(
            nodal[c].with_data(constrained[c].data)).data)
        want = -1j * om_union * _union(em, c, zeta[c])
        assert np.abs(got - want).max() / scale < 1e-11


# ================================================================
#  The mode-indexed accessor (em.mode) on the walled vertical
# ================================================================
def _mode_tendency_residual(walled, linearized, s, indices):
    """Return (omega, strong-test residual) for one single mode."""
    _, _, em = walled
    lin, prog, base0, constrain = linearized
    omega, z0 = em.mode(FAMILY[s], indices)
    _, z1 = em.mode(FAMILY[s], indices, phase=np.pi / 2)
    phys = base0.replace(**{
        c: base0[c].with_data(z0[c].data) for c in COMPONENTS})
    tau = lin.tendency(phys, t=0.0, constraints=False)
    constrained = constrain(base0.replace(
        **{c: tau[c] for c in prog}))
    residual = max(
        float(np.abs(np.asarray(constrained[c].data)
                     - omega * np.asarray(z1[c].data)).max())
        for c in COMPONENTS)
    return omega, residual / (1.0 + abs(omega))


@pytest.mark.parametrize(("s", "iz"), [
    pytest.param(1, 3, id="plus-m3"),
    pytest.param(-1, 1, id="minus-m1"),
    pytest.param(0, 2, id="vortical-m2"),
    pytest.param(0, 0, id="vortical-barotropic"),
    pytest.param(0, N, id="vortical-buoyancy-top"),
])
def test_mode_satisfies_the_strong_eigen_relation(
        walled, linearized, s, iz):
    omega, residual = _mode_tendency_residual(
        walled, linearized, s, {"x": 2, "y": 1, "z": iz})
    assert residual < 1e-12
    if s == 0:
        assert omega == 0.0
    else:
        assert omega != 0.0


@pytest.mark.parametrize(("s", "indices"), [
    pytest.param(0, {"x": N // 2, "y": 1, "z": 2},
                 id="steady-nyq-x"),
    pytest.param(0, {"x": N // 2, "y": 1, "z": 0},
                 id="steady-nyq-x-barotropic"),
    pytest.param(0, {"x": N // 2, "y": 1, "z": N},
                 id="steady-nyq-x-buoyancy-top"),
    pytest.param(0, {"x": 2, "y": N // 2, "z": 3},
                 id="steady-nyq-y"),
    pytest.param(0, {"x": N // 2, "y": N // 2, "z": 3},
                 id="steady-nyq-corner"),
    pytest.param(1, {"x": N // 2, "y": 1, "z": 3},
                 id="wave-nyq-x"),
])
def test_nyquist_strata_satisfy_the_strong_eigen_relation(
        walled, linearized, s, indices):
    # the horizontal-Nyquist strata of the even walled grid: the
    # rotation-decoupled steady divergence-free modes (all vertical
    # strata, the barotropic m = 0 and buoyancy-top m = N included)
    # join the vortical family; the rotationless gravity pair stays
    # on the wave branches
    omega, residual = _mode_tendency_residual(
        walled, linearized, s, indices)
    assert residual < 1e-12
    if s == 0:
        assert omega == 0.0
    else:
        assert omega != 0.0


def test_mode_frequency_matches_the_dispersion_table(walled):
    _, _, em = walled
    omega, z = em.mode("wave+", {"x": 2, "y": 1, "z": 3})
    table = np.broadcast_to(
        np.real(np.asarray(em.omega(1).data)),
        np.asarray(em.q(1)["w"].data).shape)
    assert omega == pytest.approx(float(table[2, 1, 2]), rel=1e-13)
    for c in COMPONENTS:
        assert not np.iscomplexobj(np.asarray(z[c].data))
    peak = max(float(np.abs(np.asarray(z[c].data)).max())
               for c in ("u", "v"))
    assert peak <= 1.0 + 1e-12


def test_mode_projection_keeps_and_annihilates(walled):
    _, _, em = walled
    _, z = em.mode("wave+", {"x": 2, "y": 1, "z": 3})
    kept = nh.transforms.mode_projection(em, 1)(z)
    assert max(
        float(np.abs(np.asarray(kept[c].data)
                     - np.asarray(z[c].data)).max())
        for c in COMPONENTS) < 1e-12
    for other in (0, -1):
        killed = nh.transforms.mode_projection(em, other)(z)
        assert max(
            float(np.abs(np.asarray(killed[c].data)).max())
            for c in COMPONENTS) < 1e-12


def test_mode_absent_strata_carry_exact_zero_components(walled):
    # the barotropic stratum m = 0 exists only on u/v/p lattices:
    # the returned w and b components are exact zeros
    _, _, em = walled
    _, z = em.mode("vortical", {"x": 2, "y": 1, "z": 0})
    assert float(np.abs(np.asarray(z["w"].data)).max()) == 0.0
    assert float(np.abs(np.asarray(z["b"].data)).max()) == 0.0
    assert float(np.abs(np.asarray(z["u"].data)).max()) > 0.0


def test_mode_structural_errors_are_taught(walled):
    _, _, em = walled
    # wave branches carry no barotropic / buoyancy-top strata
    with pytest.raises(ValueError, match="structurally"):
        em.mode("wave+", {"x": 2, "y": 1, "z": 0})
    with pytest.raises(ValueError, match="structurally"):
        em.mode("wave+", {"x": 2, "y": 1, "z": N})
    # ... and no k_h = 0 columns
    with pytest.raises(ValueError, match="structurally"):
        em.mode("wave+", {"x": 0, "y": 0, "z": 3})
    with pytest.raises(ValueError, match="union modes"):
        em.mode("vortical", {"x": 2, "y": 1, "z": N + 1})


# ================================================================
#  Stage B: partition of unity and the unrepresented strata
# ================================================================
def test_partition_of_unity_on_the_represented_set(walled,
                                                   linearized):
    _, _, em = walled
    _, _, base0, constrain = linearized
    kit = em.kit
    rng = np.random.default_rng(11)
    fields = {c: base0[c].with_data(jnp.asarray(
        rng.standard_normal(base0[c].data.shape)))
        for c in COMPONENTS}
    state = constrain(base0.replace(**fields))
    zeta = State({
        c: kit.forward(c)(state[c].retag(kit.forward(c).domain))
        for c in COMPONENTS})
    total = None
    for s in (0, 1, -1):
        contribution = em.projector(s)(zeta)
        total = (contribution if total is None
                 else total + contribution)
    scale = max(np.abs(np.asarray(zeta[c].data)).max()
                for c in COMPONENTS)
    for c in COMPONENTS:
        r = (np.asarray(total[c].data)
             - np.asarray(zeta[c].data)) / scale
        # the horizontal-Nyquist planes are now covered (the steady
        # divergence-free and buoyancy-top strata joined the
        # vortical column); the only unrepresented strata left are
        # the kh = 0 inertial / mean modes of u and v
        masked = r.copy()
        if c in ("u", "v"):
            masked[0, 0, :] = 0.0
            # ... which the projections genuinely leave alone
            assert np.abs(
                np.asarray(total[c].data)[0, 0]).max() == 0.0
            # the masking is not vacuous: the residual genuinely
            # lives on the kh = 0 strata
            assert np.abs(r).max() > 1e-3
        else:
            # w and b are fully represented
            assert np.abs(r).max() < 1e-12
        assert np.abs(masked).max() < 1e-12


def test_vortical_alive_count_is_n_plus_1(walled):
    # per horizontal wavevector the steady family holds exactly
    # N + 1 strata: N - 1 interior geostrophic + m = 0 barotropic
    # + m = N buoyancy-top; on the horizontal-Nyquist planes the
    # rotation-decoupled steady strata (the N divergence-free u/v
    # modes + the buoyancy-top mode) keep the same count; kh = 0
    # holds the N pure-b modes
    _, _, em = walled
    q0 = _union_column(em, 0)
    alive = sum(np.abs(q0[c]) ** 2 for c in COMPONENTS) > 0
    counts = alive.sum(axis=-1)
    assert counts[0, 0] == N
    everywhere = np.ones_like(counts, dtype=bool)
    everywhere[0, 0] = False
    assert (counts[everywhere] == N + 1).all()
    # ... and on the Nyquist wavevectors the strata are genuinely
    # the divergence-free u/v modes (0..N-1) plus the b-top (m = N)
    nyq_col = {c: q0[c][-1, 1] for c in COMPONENTS}
    assert np.all(np.abs(nyq_col["w"]) == 0.0)
    assert np.all(np.abs(nyq_col["b"][:N]) == 0.0)
    assert np.abs(nyq_col["b"][N]) > 0.0
    assert (np.abs(nyq_col["u"][:N]) > 0.0).all()


# ================================================================
#  Stage B: an exponential boundary vortical mode is invariant
# ================================================================
def test_exponential_boundary_mode_is_projector_invariant(
        walled, linearized):
    # a geostrophically balanced state with the exponential vertical
    # profile psi(z) = cosh(kappa (z - Lz/2)) — a cross-mode
    # combination INSIDE the degenerate eigenspace — built from the
    # grid's own discrete operators (u = -interp(diff_y psi), ...;
    # b from the discrete thermal wind), is invariant under the
    # vortical projection and invisible to the wave projection.
    grid, _, em = walled
    if _is_fv(em):
        # the balanced state is hand-built from the *nodal* discrete
        # operators (FiniteDifference / LinearInterp), which do not
        # dispatch on the FV average family. The property under test —
        # a geostrophically balanced cross-mode state is vortical-
        # projector-invariant and wave-invisible — is a bitwise
        # corollary on FV: the FV eigenbasis and the projector data
        # path are bit-identical to the nodal one (asserted by
        # test_walled_fv_eigenbasis_is_bitwise_nodal), so the nodal
        # coverage carries it.
        pytest.skip("nodal-operator construction; FV parity is bitwise")
    lin, prog, base0, constrain = linearized
    kit = em.kit
    chart = ModeChart(grid)
    kappa = 3.0
    psi = grid.create_field(
        kit.forward("p").domain,
        init=lambda x, y, z: jnp.cos(x) * jnp.cos(2 * y)
        * jnp.cosh(kappa * (z - LZ / 2)))
    fd = FiniteDifference()
    li = LinearInterp()
    u = li["x"](li["y"](fd["y"](psi)))
    v = li["y"](li["x"](fd["x"](psi)))
    # discrete thermal wind, in coefficient space: interp_z(b) ==
    # f0 |a_x|^2 |a_y|^2 diff_z(psi) at the w points, mode-wise on
    # the union lattice through the grid's own symbol tables
    psi_hat = kit.forward("p")(psi)
    kz = np.asarray(em.k["z"].data).ravel()  # Cos-II -> Sine-I
    az = np.asarray(
        kit.interp("z", on="b").data).ravel()  # S-II -> S-I
    ratio = np.zeros(N + 1)
    ratio[1:N] = kz / az
    axm2 = np.asarray((em.a["x"].magnitude ** 2).data)
    aym2 = np.asarray((em.a["y"].magnitude ** 2).data)
    b_union = (F0 * axm2 * aym2 * ratio
               * np.asarray(chart.embed(psi_hat.data,
                                        kit.coeff("p"))))
    b_template = em.q(0)["b"]
    b_hat = b_template.with_data(jnp.asarray(np.asarray(
        chart.restrict(jnp.asarray(b_union), kit.coeff("b"))))
        .astype(b_template.data.dtype))
    balanced = base0.replace(
        u=base0["u"].with_data(-u.data),
        v=base0["v"].with_data(v.data),
        b=base0["b"].with_data(kit.backward("b")(b_hat).data))
    scale = max(np.abs(np.asarray(balanced[c].data)).max()
                for c in COMPONENTS)
    # it is a genuinely steady state of the constrained operator ...
    tau = lin.tendency(balanced, t=0.0, constraints=False)
    steady = constrain(base0.replace(**{c: tau[c] for c in prog}))
    assert max(np.abs(np.asarray(steady[c].data)).max()
               for c in COMPONENTS) / scale < 1e-11
    # ... and the projector family sees it as pure vortical
    out = nh.transforms.VorticalProjection(em)(balanced)
    wave = nh.transforms.WaveProjection(em)(balanced)
    for c in COMPONENTS:
        assert np.abs(np.asarray(out[c].data)
                      - np.asarray(balanced[c].data)).max() \
            / scale < 1e-11
        assert np.abs(np.asarray(wave[c].data)).max() / scale < 1e-11


# ================================================================
#  Stage B: measure-weighted energy partition (Parseval)
# ================================================================
def test_energy_partition_across_the_three_projections(walled):
    # the physical energy of a random state splits exactly across
    # vortical + wave + divergence (cross terms vanish): the claimed
    # per-mode norm-factor cancellation between the trig families
    _, model, em = walled
    rng = np.random.default_rng(21)
    z = State({c: model.state[c].with_data(jnp.asarray(
        rng.standard_normal(model.state[c].data.shape)))
        for c in COMPONENTS})

    def energy(state):
        total = 0.0
        for c in COMPONENTS:
            f = state[c]
            vol = np.asarray(f.measure("x").data)
            vol = vol * np.asarray(f.measure("y").data)
            vol = vol * np.asarray(f.measure("z").data)
            total += WEIGHTS[c] * float(
                (np.asarray(f.data) ** 2 * vol).sum())
        return total

    parts = [
        energy(projection(em)(z))
        for projection in (nh.transforms.VorticalProjection,
                           nh.transforms.WaveProjection,
                           nh.transforms.DivergenceProjection)]
    assert all(part > 0.0 for part in parts)
    total = energy(z)
    assert abs(total - sum(parts)) / total < 1e-12


# ================================================================
#  function(f, s) on the walled vertical (the union-lattice path)
# ================================================================
def _random_walled_coeff_state(em, seed):
    rng = np.random.default_rng(seed)
    template = em.q(0)
    return State({c: template[c].with_data(jnp.asarray(
        rng.standard_normal(np.asarray(template[c].data).shape)
        + 1j * rng.standard_normal(
            np.asarray(template[c].data).shape)))
        for c in COMPONENTS})


@pytest.mark.parametrize("sel", [
    pytest.param(0, id="vortical"),
    pytest.param(1, id="plus"),
    pytest.param((1, -1), id="wave-pair"),
])
def test_walled_function_with_unit_f_reproduces_the_projectors(
        walled, sel):
    # f == 1 through the union mode lattice: the s = 0 weights must
    # cover the barotropic m = 0 and buoyancy-top m = N strata (the
    # per-component re-referenced column), and the wave weights the
    # embedded w lattice — bitwise against the summed projectors
    _, _, em = walled
    z = _random_walled_coeff_state(em, seed=41)
    branches = (sel,) if isinstance(sel, int) else sel
    want = None
    for s in branches:
        part = em.projector(s)(z)
        want = part if want is None else State(
            {c: want[c] + part[c] for c in COMPONENTS})
    got = em.function(np.ones_like, sel)(z)
    for c in COMPONENTS:
        assert np.array_equal(np.asarray(got[c].data),
                              np.asarray(want[c].data))


def test_walled_function_structural_zero_guard(walled):
    # the walled geostrophic branch (all N + 1 strata) is
    # represented with omega == 0: singular f is a taught error;
    # the wave branches carry only structural zeros and pass
    _, _, em = walled
    with pytest.raises(ValueError, match=r"s=0"):
        em.function(lambda om: 1.0 / (1j * om), 0)
    assert callable(em.function(lambda om: 1.0 / (1j * om), (1, -1)))


# ================================================================
#  Gate: the walled FV eigenbasis is bitwise the walled nodal one
# ================================================================
def _walled_from_model(family):
    """Build a fresh walled model + its from_model eigenmodes."""
    # device_ids=(0,) pins to one device on any device count: the
    # eigenmode projections synthesize through the naive (GSPMD)
    # transform, a Tier-1 taught error on a sharded axis (transform.py).
    grid = Grid((
        IntervalMesh(N, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(N, (0.0, 2 * np.pi), periodic=True, name="y"),
        IntervalMesh(N, (0.0, LZ), periodic=False, name="z")),
        device_ids=(0,))
    model = nh.Model(
        grid=grid, dt=DT, advection=False, dsqr=DSQR,
        coriolis=FPlaneCoriolis(f0=F0),
        stratification=ConstantStratification(n2=N2), family=family)
    return model, nh.eigenmodes.from_model(model)


def test_walled_fv_eigenbasis_is_bitwise_nodal():
    # the 2nd-order FV C-grid stencils are bitwise the nodal ones and
    # the trig basis diagonalizes them exactly, so the walled-vertical
    # FV eigenbasis is bit-identical to the walled nodal one on the
    # same grid geometry: frequencies AND eigenvector coefficient data
    # agree to exactly 0.0 across every (kx, ky, m) mode of the
    # half-spectrum layout.
    _, en = _walled_from_model("nodal")
    _, ef = _walled_from_model("fv")
    # sanity: the two kits really are the two different families
    assert not _is_fv(en)
    assert _is_fv(ef)
    for s in (0, 1, -1):
        assert np.array_equal(np.asarray(en.omega(s).data),
                              np.asarray(ef.omega(s).data))
        qn, qf = en.q(s), ef.q(s)
        for c in COMPONENTS:
            # the coefficient LAYOUTS coincide (DST-II CellAvg and
            # DST-II Center share mode_offset + shape), so a straight
            # array compare is well-defined and exact
            assert np.asarray(qn[c].data).shape \
                == np.asarray(qf[c].data).shape
            assert np.array_equal(np.asarray(qn[c].data),
                                  np.asarray(qf[c].data))
    # the mode-indexed physical states through the union-lattice
    # machinery — the re-referenced geostrophic edge blocks m = 0
    # (barotropic, u/v/p only) and m = N (buoyancy-top, b only), and
    # the interior wave modes — are bitwise equal too (same coeffs,
    # same DST/DCT synthesis at the cell midpoints = the centers)
    for s, iz in ((1, 3), (-1, 2), (0, 0), (0, N), (0, N // 2)):
        wn, zn = en.mode(FAMILY[s], {"x": 2, "y": 1, "z": iz})
        wf, zf = ef.mode(FAMILY[s], {"x": 2, "y": 1, "z": iz})
        assert wn == wf
        for c in COMPONENTS:
            assert np.array_equal(np.asarray(zn[c].data),
                                  np.asarray(zf[c].data))


# ================================================================
#  Gate: round-trip completeness + the z-constant buoyancy case
# ================================================================
def test_walled_fv_roundtrip_completeness():
    # an arbitrary random FV state, projected to the constrained
    # (represented) subspace, round-trips through the eigenmode
    # projectors to the identity: sum_s P^s reproduces the coefficient
    # state to machine zero on the represented set (the only strata the
    # walls leave unrepresented are the kh = 0 inertial u/v modes).
    model, em = _walled_from_model("fv")
    kit = em.kit
    lin = fr.model.linearize(model)
    prog, base0 = _rest_background(lin, 0.0)
    constrain = _constrain_fn(lin, base0, prog)
    rng = np.random.default_rng(7)
    fields = {c: base0[c].with_data(jnp.asarray(
        rng.standard_normal(base0[c].data.shape))) for c in COMPONENTS}
    state = constrain(base0.replace(**fields))
    zeta = State({c: kit.forward(c)(state[c].retag(kit.forward(c).domain))
                  for c in COMPONENTS})
    total = None
    for s in (0, 1, -1):
        part = em.projector(s)(zeta)
        total = part if total is None else State(
            {c: total[c] + part[c] for c in COMPONENTS})
    scale = max(np.abs(np.asarray(zeta[c].data)).max() for c in COMPONENTS)
    for c in COMPONENTS:
        r = (np.asarray(total[c].data) - np.asarray(zeta[c].data)) / scale
        if c in ("u", "v"):
            r[0, 0, :] = 0.0  # the unrepresented kh = 0 inertial modes
        assert np.abs(r).max() < 1e-13


def test_walled_fv_zconstant_buoyancy_is_pure_vortical():
    # a z-constant (globally uniform) buoyancy anomaly with u = v =
    # w = 0. It is NONZERO at the rigid lids, yet the Dirichlet DST-II
    # cell basis is complete on the n-cell lattice, so it round-trips
    # exactly (sines are complete); and being horizontally uniform
    # (kh = 0) it lives entirely in the omega = 0 vortical subspace —
    # every wave column is a structural zero there, so the wave
    # projection is EXACTLY zero, not merely small.
    model, em = _walled_from_model("fv")
    grid, kit = model.grid, em.kit
    _, base0 = _rest_background(fr.model.linearize(model), 0.0)
    # model.state_space(name) is the State factory's space accessor
    # (04 section 6.1) — the declared "b" space, fed to create_field.
    bf = grid.create_field(model.state_space("b"),
                           init=lambda x, y, z: (x + y + z) * 0.0 + 1.0)
    phys = base0.replace(b=base0["b"].with_data(bf.data))
    zeta = State({c: kit.forward(c)(phys[c].retag(kit.forward(c).domain))
                  for c in COMPONENTS})
    # (1) completeness: the DST-II cell transform round-trips a field
    # that does not vanish at the walls
    back = kit.backward("b")(zeta["b"])
    assert np.abs(np.asarray(back.data)
                  - np.asarray(phys["b"].data)).max() < 1e-13
    # (2) round-trip through the eigenmode projectors is exact
    parts = {s: em.projector(s)(zeta) for s in (0, 1, -1)}
    total = State({c: parts[0][c] + parts[1][c] + parts[-1][c]
                   for c in COMPONENTS})
    scale = max(np.abs(np.asarray(zeta[c].data)).max() for c in COMPONENTS)
    for c in COMPONENTS:
        assert (np.abs(np.asarray(total[c].data)
                       - np.asarray(zeta[c].data)).max() / scale < 1e-13)
    # (3) the wave projections are EXACTLY zero; the vortical one is not
    for s in (1, -1):
        assert max(float(np.abs(np.asarray(parts[s][c].data)).max())
                   for c in COMPONENTS) == 0.0
    assert max(float(np.abs(np.asarray(parts[0][c].data)).max())
               for c in COMPONENTS) > 0.0
    # (4) and nh.transforms sees it as pure vortical
    wave = nh.transforms.WaveProjection(em)(phys)
    assert max(float(np.abs(np.asarray(wave[c].data)).max())
               for c in COMPONENTS) < 1e-13


# ================================================================
#  Wave B: the walled-vertical tier under a sharded periodic axis
# ================================================================
def _walled_model(device_ids, *, family="nodal", n=N):
    """Build a walled-vertical (rigid-lid z) nonhydro model."""
    grid = Grid((
        IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="y"),
        IntervalMesh(n, (0.0, LZ), periodic=False, name="z")),
        device_ids=device_ids)
    return nh.Model(
        grid=grid, dt=DT, advection=False, dsqr=DSQR,
        coriolis=FPlaneCoriolis(f0=F0),
        stratification=ConstantStratification(n2=N2), family=family)


@pytest.mark.multi_device
def test_walled_mode_is_device_count_invariant(forced_devices):
    # em.mode synthesizes each Hermitian-closed single mode through the
    # replicated (layout-free) coefficient backward, so the walled
    # single-mode state is device-count invariant when the default layout
    # shards a periodic axis (the plain path a sharded operand would trip
    # is bypassed by the bare replicated field)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    em_many = nh.eigenmodes.from_model(_walled_model(None))
    em_one = nh.eigenmodes.from_model(_walled_model((0,)))
    for s in (0, 1, -1):
        w_many, z_many = em_many.mode(FAMILY[s], {"x": 2, "y": 1, "z": 3})
        w_one, z_one = em_one.mode(FAMILY[s], {"x": 2, "y": 1, "z": 3})
        assert abs(float(w_many) - float(w_one)) < 1e-12
        assert max(
            float(np.abs(np.asarray(z_many[c].data)
                         - np.asarray(z_one[c].data)).max())
            for c in COMPONENTS) < 1e-12


@pytest.mark.multi_device
def test_walled_random_state_is_device_count_invariant(forced_devices):
    # the prescribed-spectra random state draws its Hermitian phases on
    # the replicated single-device coefficient frame, so the walled
    # synthesis is device-count invariant on the sharded-periodic-axis
    # grid (the analytic route's internal frame re-designates the half
    # axis, so the synthesis keeps its replicated backward)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many = _walled_model(None)
    one = _walled_model((0,))
    for family in ("vortical", "wave"):
        z_many = nh.random_state(many, family, seed=17)
        z_one = nh.random_state(one, family, seed=17)
        assert max(
            float(np.abs(np.asarray(z_many[c].data)
                         - np.asarray(z_one[c].data)).max())
            for c in COMPONENTS) < 1e-12


@pytest.mark.multi_device
def test_walled_operator_matrix_is_a_projector_and_guards(forced_devices):
    # the walled union-lattice per-mode matrix (assemble_walled_operator_
    # matrix via em.operator_matrix): the vortical projector is idempotent
    # per mode, the f(omega) wave operator is finite, and a singular f
    # meeting the zero-frequency geostrophic branch is a taught error
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    em = nh.eigenmodes.from_model(_walled_model(None))
    route = resolve_route(em.grid, em._analysis, em._components)
    assert route is not None
    m0 = np.asarray(em.operator_matrix(route.coeff_of, branches=(0,)))
    assert m0.shape[-2:] == (4, 4)
    # per-mode idempotency on the union lattice (the biorthonormal dual)
    m0m0 = np.einsum("...jd,...de->...je", m0, m0)
    assert np.abs(m0m0 - m0).max() < 1e-10
    # the f(omega) wave operator (L_w^{-1}) is finite
    mf = np.asarray(em.operator_matrix(
        route.coeff_of, branches=(1, -1),
        f=lambda w: -1.0 / (1j * w)))
    assert np.all(np.isfinite(mf))
    # a singular f on the zero-frequency geostrophic branch is taught
    with pytest.raises(ValueError, match="non-finite"):
        em.operator_matrix(route.coeff_of, branches=(0,),
                           f=lambda w: 1.0 / w)
