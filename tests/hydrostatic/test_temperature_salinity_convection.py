r"""``hy.TemperatureSalinity(convective_adjustment=n)``: the pairwise scheme.

Prefix-mirrored shard of ``test_temperature_salinity.py`` (AGENTS
oversized-module rule; self-contained builders). The gates: an unstable
column is driven to static stability while its heat and salt content
are conserved to rounding; a stable column is bitwise untouched; dry
cells never pair; the two tracer reductions; autodiff through the
adjusted step; device-count invariance; the taught constructor error.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.immersed_domain import ImmersedDomain

IM = fr.spatial.meshes.IntervalMesh
G = 9.81
LENGTH = 1.0e5
DEPTH = 1200.0
DT = 50.0
CenteredAdvection = fr.model.modules.CenteredAdvection

#: bottom -> top [degC]: unstable below (colder over warmer), stable
#: above; every compared pair differs by >= 1.5 K
KINKED = np.array([14.0, 13.0, 12.0, 16.0, 18.0, 20.0])


# ================================================================
#  Builders
# ================================================================
def make_grid(n=4, nz=6, **kwargs):
    """Return a small doubly-periodic, bounded-z grid (200 m cells)."""
    return fr.spatial.Grid((
        IM(n, (0.0, LENGTH), periodic=True, name="x"),
        IM(n, (0.0, LENGTH), periodic=True, name="y"),
        IM(nz, (-DEPTH, 0.0), periodic=False, name="z")), **kwargs)


def make_model(buoyancy, *, grid=None, **kwargs):
    """Assemble a dimensional hydrostatic model on the buoyancy module."""
    return hy.Model(
        grid=make_grid() if grid is None else grid,
        core=hy.Core(gravity=G),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=hy.FPlaneCoriolis(f0=1.0e-4),
        buoyancy=buoyancy,
        free_surface=hy.ExplicitFreeSurface(),
        advection=CenteredAdvection(surface_flux=False), **kwargs)


def column(model, profile, anomaly=0.0):
    """Broadcast a bottom-to-top profile over the horizontal cells."""
    shape = model.state["b"].shape
    x = (np.arange(shape[0]) + 0.5) / shape[0]
    wave = anomaly * np.sin(2 * np.pi * x)[:, None, None]
    return np.broadcast_to(profile, shape) + wave


def field(model, name):
    """Return one state field as a host array."""
    return np.asarray(model.state[name].data)


def instability(eos, t, s, depth):
    """Return the largest density inversion of adjacent cells [kg/m^3]."""
    interface = 0.5 * (depth[..., :-1] + depth[..., 1:])
    low = eos.density_anomaly(t[..., :-1], s[..., :-1], interface)
    up = eos.density_anomaly(t[..., 1:], s[..., 1:], interface)
    return float(np.max(np.asarray(up) - np.asarray(low)))


DEPTHS = DEPTH * (1.0 - (np.arange(6) + 0.5) / 6)   # bottom -> top


# ================================================================
#  The scheme
# ================================================================
@pytest.mark.parametrize(
    "make_eos", [pytest.param(hy.LinearEOS, id="linear"),
                 pytest.param(hy.TEOS10EOS, id="teos10")])
def test_unstable_column_is_stabilized_and_content_conserved(make_eos):
    eos = make_eos()
    model = make_model(
        hy.TemperatureSalinity(eos, convective_adjustment=2))
    t0 = column(model, KINKED)
    s0 = column(model, np.linspace(35.0, 34.9, 6))
    model.set_fields(T=t0, S=s0)
    before = instability(eos, t0, s0, DEPTHS)
    assert before > 0.1
    model.advance(6)
    t1, s1 = field(model, "T"), field(model, "S")
    # the pairwise scheme converges geometrically and never exactly
    # (Rahmstorf 1993): the three-cell inversion shrinks by 1/4 per
    # pass, 4^-12 = 6e-8 after 6 steps x 2 passes
    assert instability(eos, t1, s1, DEPTHS) < 1e-6 * before
    # uniform cells: the content is the plain sum
    assert t1.sum() == pytest.approx(t0.sum(), rel=1e-14)
    assert s1.sum() == pytest.approx(s0.sum(), rel=1e-14)
    # the three unstable bottom cells homogenize toward their mean, the
    # stable thermocline above them is untouched
    np.testing.assert_allclose(t1[:, :, :3], 13.0, atol=1e-5)
    np.testing.assert_array_equal(t1[:, :, 3:], t0[:, :, 3:])


def test_stable_column_is_bitwise_untouched():
    def run(passes):
        model = make_model(hy.TemperatureSalinity(
            hy.TEOS10EOS(), convective_adjustment=passes))
        model.set_fields(
            T=column(model, np.linspace(4.0, 20.0, 6), anomaly=0.5),
            S=column(model, np.linspace(35.0, 34.0, 6)))
        model.advance(5)
        return {name: field(model, name) for name in ("T", "S", "u")}

    adjusted, plain = run(3), run(None)
    assert np.abs(plain["u"]).max() > 0.0
    for name in ("T", "S", "u"):
        np.testing.assert_array_equal(adjusted[name], plain[name])


def test_one_pass_mixes_the_even_then_the_odd_pairs():
    model = make_model(hy.TemperatureSalinity(
        hy.LinearEOS(), constant_salinity=35.0, convective_adjustment=1))
    model.set_fields(T=column(model, KINKED))
    model.advance(1)
    # (0,1) -> 13.5 each; then (1,2): 13.5 over... under 12 -> 12.75
    np.testing.assert_allclose(
        field(model, "T")[0, 0], [13.5, 12.75, 12.75, 16.0, 18.0, 20.0],
        rtol=1e-14)


def test_salt_driven_instability_with_constant_temperature():
    eos = hy.RoquetEOS()
    model = make_model(hy.TemperatureSalinity(
        eos, constant_temperature=4.0, convective_adjustment=3))
    s0 = column(model, np.linspace(34.0, 35.0, 6))   # salty on top
    model.set_fields(S=s0)
    model.advance(8)
    s1 = field(model, "S")
    assert s1.sum() == pytest.approx(s0.sum(), rel=1e-14)
    assert np.ptp(s1[0, 0]) < 0.25 * np.ptp(s0[0, 0])


def test_dry_cells_never_pair():
    def wet(x, y, z):  # noqa: ARG001
        return (z > -0.5 * DEPTH).astype(float)

    grid = make_grid(immersed=ImmersedDomain(wet))
    model = make_model(hy.TemperatureSalinity(
        hy.LinearEOS(), convective_adjustment=2), grid=grid)
    # wet cells: the top three; the lowest wet cell is the WARMEST, so
    # pairing it with the dry T = 0 cell below would (wrongly) cool it
    wet_profile = np.array([0.0, 0.0, 0.0, 20.0, 12.0, 11.0])
    model.set_fields(T=column(model, wet_profile),
                     S=column(model, np.where(wet_profile > 0, 35.0, 0.0)))
    model.advance(4)
    got = field(model, "T")
    assert np.all(got[:, :, :3] == 0.0)
    assert got.sum() == pytest.approx(16 * 43.0, rel=1e-14)
    assert np.ptp(got[0, 0, 3:]) < 2.0       # the wet column mixed


def test_passes_must_be_a_positive_int():
    for bad in (0, -1, 1.5, True, "3"):
        with pytest.raises(TypeError, match="pairwise adjustment passes"):
            hy.TemperatureSalinity(convective_adjustment=bad)


# ================================================================
#  Autodiff through the adjusted step
# ================================================================
def test_grad_through_the_adjustment_matches_fd():
    # the adjustment is a jnp.where between two finite branches: the
    # gradient is finite and, away from neutral pairs (every compared
    # pair here differs by >= 1.5 K, one step), matches a central FD
    model = make_model(hy.TemperatureSalinity(
        hy.TEOS10EOS(), convective_adjustment=1))
    model.set_fields(T=column(model, KINKED, anomaly=0.1),
                     S=column(model, np.linspace(35.0, 34.0, 6)))
    run = model.propagator(wrt=("T",), steps=1)
    leaf = model._carry.state["T"].storage

    def loss(data):
        final = run((data,))
        return (jnp.sum(final.state["T"].data ** 2)
                + jnp.sum(final.state["S"].data ** 2))

    grad = np.asarray(jax.grad(loss)(leaf))
    assert bool(np.all(np.isfinite(grad)))
    rng = np.random.default_rng(5)
    direction = jnp.asarray(rng.standard_normal(leaf.shape),
                            dtype=leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(leaf + eps * direction))
          - float(loss(leaf - eps * direction))) / (2.0 * eps)
    assert abs(directional) > 1.0
    assert directional == pytest.approx(fd, rel=1e-4)


# ================================================================
#  Device-count invariance (forced host devices)
# ================================================================
def _run(device_ids):
    grid = fr.spatial.Grid(
        (IM(4, (0.0, LENGTH), periodic=True, name="x"),
         IM(4, (0.0, LENGTH), periodic=True, name="y"),
         IM(16, (-DEPTH, 0.0), periodic=False, name="z")),
        device_ids=device_ids)
    model = make_model(hy.TemperatureSalinity(
        hy.TEOS10EOS(), convective_adjustment=2), grid=grid,
        chunk_size=4)
    profile = np.concatenate([np.linspace(14.0, 8.0, 8),
                              np.linspace(12.0, 20.0, 8)])
    model.set_fields(T=column(model, profile, anomaly=0.3),
                     S=column(model, np.linspace(35.0, 34.0, 16)))
    model.advance(4)
    sharded = [name for name, _
               in grid.decomposition.default_layout.device_axes]
    return ({k: np.asarray(model.state[k].data)
             for k in ("u", "T", "S")}, sharded)


@pytest.mark.multi_device
def test_adjustment_is_device_count_invariant(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many, sharded = _run(None)
    one, _ = _run((0,))
    assert "z" in sharded      # the pairs straddle the shard seams
    for name in ("u", "T", "S"):
        scale = np.abs(one[name]).max()
        assert np.abs(many[name] - one[name]).max() <= 1e-11 * scale, name
