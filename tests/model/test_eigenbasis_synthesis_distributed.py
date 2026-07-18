"""Multi-device channel-eigenbasis synthesis through the fused inverse.

The synthesis-only features build their coefficient columns host-side
and inverse transform them: single-mode states (``mode()`` ->
``_eigenbasis._synthesize_column``) and random-phase channel states
(``_eigenbasis.channel_random_state``). On a 3-D channel whose default
layout shards a periodic axis that inverse routes through the fused
``jax.shard_map`` synthesis entry
(``spatial.operators.distributed_contract.ContractPlan.synthesize``)
instead of the plain GSPMD transform, which hits the upstream XLA:GPU
distributed-FFT fault (see ``design/research/multidevice_test_faults.md``).

Two tiers, mirroring ``test_eigenbasis_distributed.py``:

- **stub-``em`` unit tests** drive ``_synthesize_column`` /
  ``channel_random_state`` on a *bare* sharded grid with a synthetic
  eigenvector basis (no eigensolve), so they are safe on the forced-4
  CPU leg (the batched-eigh CPU-LAPACK heap corruption on many-core
  hosts never runs). The synthesized fields on the fused many-device
  path match the replicated one-device path, land real, and keep the
  grid's own layout.
- **real-eigenbasis end-to-end tests** build a genuine nh channel
  eigenbasis and are GPU-scoped (the fixture skips on the CPU backend):
  ``mode()`` and ``random_state`` on the sharded grid match the
  explicit one-device reference.

``multi_device`` marked tests need the forced-4-device suite
(``XLA_FLAGS=--xla_force_host_platform_device_count=4
FRIDOM_TEST_FORCED_DEVICES=4``) or a genuine multi-GPU node.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model._eigenbasis import (
    _synthesize_column,
    channel_random_state,
)
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

STUB_COMPONENTS = ("u", "v")


# ================================================================
#  Stub-em unit tests (no eigensolve; forced-4 CPU safe)
# ================================================================
def make_bare_channel(nx, ny, nz, device_ids=None):
    """Return a walled-y bare channel grid (shards x, nx>=3 cells)."""
    meshes = (
        IntervalMesh(nx, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(ny, (0.0, 1.0), periodic=False, name="y"),
        IntervalMesh(nz, (0.0, 2 * np.pi), periodic=True, name="z"))
    return Grid(meshes, device_ids=device_ids)


def synthetic_basis(nx, ny, nz, seed):
    """Synthetic (q, metric, omega, labels, slices) on the plane frame.

    ``labels`` mark each column as ``wave+`` (1), ``wave-`` (2) or
    background (0), enough for ``channel_random_state`` to order and
    scale the selected family without an eigensolve.
    """
    rng = np.random.default_rng(seed)
    dim = 2 * ny
    nmodes = nz // 2 + 1
    q = jnp.asarray(rng.standard_normal((nx, nmodes, dim, dim))
                    + 1j * rng.standard_normal((nx, nmodes, dim, dim)))
    metric = jnp.asarray(rng.uniform(0.5, 1.5, dim))
    omega = jnp.asarray(rng.standard_normal((nx, nmodes, dim)))
    labels = jnp.asarray(
        rng.integers(0, 3, (nx, nmodes, dim)).astype(np.int32))
    slices = {"u": slice(0, ny), "v": slice(ny, 2 * ny)}
    return q, metric, omega, labels, slices


def stub_em(grid, basis):
    """Return a duck-typed eigenmodes object for the synthesis paths."""
    q, metric, omega, labels, slices = basis
    return SimpleNamespace(
        grid=grid, bounded_axis="y", periodic_axis="z",
        components=STUB_COMPONENTS, slices=slices, q=q, metric=metric,
        omega=omega, labels=labels,
        spaces={name: grid.create_field(name=name).function_space
                for name in STUB_COMPONENTS},
        families={"wave+": 1, "wave-": 2}, nonphysical_families=())


def absmax(a, b, components):
    """Max component-wise absolute difference of two field mappings."""
    return max(
        float(np.abs(np.asarray(a[c].data)
                     - np.asarray(b[c].data)).max())
        for c in components)


@pytest.mark.multi_device
@pytest.mark.parametrize("z_slot", [0, 1, 6],
                         ids=["self-conj-0", "generic", "self-conj-nyq"])
def test_synthesize_column_matches_one_device(z_slot, forced_devices):
    # the fused many-device _synthesize_column reproduces the replicated
    # one-device path bit-for-bit (to floating point) and lands real, on
    # the self-conjugate half-axis planes (0, nz/2 -> the conjugate-pair
    # closure) and a generic plane alike
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    nx, ny, nz = 12, 8, 12
    basis = synthetic_basis(nx, ny, nz, seed=2)
    rng = np.random.default_rng(3)
    values = jnp.asarray(rng.standard_normal(2 * ny)
                         + 1j * rng.standard_normal(2 * ny))
    slots = (2, z_slot)  # (x, z) in periodic order; z is the half axis

    g_many = make_bare_channel(nx, ny, nz)
    assert g_many.decomposition.default_layout.device_axes == (
        ("x", "devices"),)
    out_many = _synthesize_column(stub_em(g_many, basis), slots, values)
    g_one = make_bare_channel(nx, ny, nz, device_ids=(0,))
    out_one = _synthesize_column(stub_em(g_one, basis), slots, values)
    for name in STUB_COMPONENTS:
        assert not np.iscomplexobj(np.asarray(out_many[name].data))
        assert out_many[name]._data.sharding.spec[0] == "devices"
    assert absmax(out_many, out_one, STUB_COMPONENTS) <= 1e-11


@pytest.mark.multi_device
def test_channel_random_state_matches_one_device(forced_devices):
    # the random-phase channel synthesis runs through the same fused
    # inverse; same seed => the many-device state matches one device and
    # keeps the grid layout (the periodic axis x is never gathered)
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    nx, ny, nz = 12, 8, 12
    basis = synthetic_basis(nx, ny, nz, seed=5)
    kwargs = {"spectral_energy_density": lambda *_k: jnp.asarray(1.0),
              "seed": 7, "horizontal": ("x", "z")}

    g_many = make_bare_channel(nx, ny, nz)
    out_many = channel_random_state(
        stub_em(g_many, basis), "wave", **kwargs)
    g_one = make_bare_channel(nx, ny, nz, device_ids=(0,))
    out_one = channel_random_state(
        stub_em(g_one, basis), "wave", **kwargs)
    for name in STUB_COMPONENTS:
        assert not np.iscomplexobj(np.asarray(out_many[name].data))
        assert out_many[name]._data.sharding.spec[0] == "devices"
    assert absmax(out_many, out_one, STUB_COMPONENTS) <= 1e-11


@pytest.mark.multi_device
def test_channel_random_state_is_deterministic(forced_devices):
    # same seed => identical sharded state; different seed => different
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    nx, ny, nz = 12, 8, 12
    basis = synthetic_basis(nx, ny, nz, seed=5)
    grid = make_bare_channel(nx, ny, nz)

    def draw(seed):
        return channel_random_state(
            stub_em(grid, basis), "wave",
            spectral_energy_density=lambda *_k: jnp.asarray(1.0),
            seed=seed, horizontal=("x", "z"))

    assert absmax(draw(11), draw(11), STUB_COMPONENTS) == 0.0
    assert absmax(draw(11), draw(12), STUB_COMPONENTS) > 0.0


# ================================================================
#  Real-eigenbasis end-to-end tests (GPU-scoped: fixture skips CPU)
# ================================================================
NH_COMPONENTS = ("u", "v", "w", "b")


def make_nh_channel(device_ids=None):
    """Return a walled-y nh channel (x=12 shards, z=12, y=8)."""
    meshes = tuple(
        IntervalMesh(n, (0.0, 1.0 if name == "y" else 2 * np.pi),
                     periodic=(name != "y"), name=name)
        for name, n in (("x", 12), ("y", 8), ("z", 12)))
    return nh.Model(
        grid=Grid(meshes, device_ids=device_ids),
        advection=False, dsqr=2.0,
        coriolis=nh.FPlaneCoriolis(f0=1.5),
        stratification=nh.ConstantStratification(n2=3.0),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


@pytest.fixture(scope="module")
def nh_pair():
    """Many/one nh channels + eigenbases + shared fields (GPU only).

    Building the channel eigenbasis runs a batched ``eigh`` that
    heap-corrupts jaxlib's CPU LAPACK on many-core hosts (T5b) -- the
    real-eigenbasis synthesis is therefore GPU-scoped and skips on the
    CPU backend; the stub-em tier above carries the forced-4 CPU leg.
    """
    if jax.default_backend() == "cpu":
        pytest.skip(
            "real channel eigenbasis build runs a batched eigh that "
            "heap-corrupts jaxlib's CPU LAPACK on many-core hosts "
            "(T5b); the real-eigenbasis path is GPU-scoped")
    many = make_nh_channel()
    one = make_nh_channel(device_ids=(0,))
    return many, nh.eigenbasis(many), one, nh.eigenbasis(one)


@pytest.mark.multi_device
def test_real_mode_matches_one_device(nh_pair, forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    _many_m, many_eb, _one_m, one_eb = nh_pair
    indices = {"x": 1, "y": 0, "z": 1}
    om_many, st_many = many_eb.mode("wave+", indices)
    om_one, st_one = one_eb.mode("wave+", indices)
    assert st_many["u"]._data.sharding.spec[0] == "devices"
    assert om_many == pytest.approx(om_one, rel=1e-11)
    for c in NH_COMPONENTS:
        assert not np.iscomplexobj(np.asarray(st_many[c].data))
    assert absmax(st_many, st_one, NH_COMPONENTS) <= 1e-10


@pytest.mark.multi_device
def test_real_random_state_matches_one_device(nh_pair, forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many_m, _many_eb, one_m, _one_eb = nh_pair
    rm = nh.random_state(many_m, "wave", seed=17)
    ro = nh.random_state(one_m, "wave", seed=17)
    assert rm["u"]._data.sharding.spec[0] == "devices"
    for c in NH_COMPONENTS:
        assert not np.iscomplexobj(np.asarray(rm[c].data))
    assert absmax(rm, ro, NH_COMPONENTS) <= 1e-10
