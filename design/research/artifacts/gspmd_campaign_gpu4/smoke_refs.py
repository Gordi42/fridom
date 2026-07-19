r"""GSPMD illegality campaign: single-process reference smoke.

Computes the campaign's device-count-invariant multi-device paths on
whatever device set jax exposes and, with ``--write-ref``, saves each
produced field as a ``.npy`` reference for the real-multi-process
sibling ``srun_campaign.py`` to gather-and-compare against under
sharding.

The five smoke items each mirror an existing green test's setup (the
public consumer surface, which auto-routes through the fused
``shard_map`` regions when the grid shards a transform axis and stays
bit-identical eager on a single device):

* **a -- Wave B walled-vertical.** nh2 walled-vertical grid (x, y
  periodic, z walled); vortical + wave projections and an order-1
  balance expansion of a deterministic state. Routes through
  ``AnalyticDistributedRoute.apply_matrix`` (walled tier) when sharded.
  Mirrors ``tests/model/test_analytic_distributed.py`` /
  ``tests/nonhydro2/test_walled_eigenmodes.py`` /
  ``tests/model/transforms/test_balance_expansion.py``.
* **b -- no-gather synthesis.** ``random_state`` (vortical + wave) and a
  ``mode()`` state on (i) an all-periodic nh2 grid (the
  ``hermitian_reframe`` route) and (ii) the walled grid from (a) (the
  ``can_synthesize=False`` replicated fallback). Mirrors
  ``tests/model/test_eigenbasis_synthesis_distributed.py``.
* **c -- trig/mixed.** ``ComposedTransform.apply_diagonal`` running a
  Helmholtz-style spectral apply on a walled grid. Mirrors
  ``tests/spatial/operators/test_mixed.py``.
* **d -- Tier-2 escape.** ``SpectralSolve(..., allow_replicated=True)``
  on a picky grid where the slab declines. Mirrors
  ``tests/spatial/operators/test_spectral_solve.py``. (The taught-error
  guard on the UNESCAPED path is asserted multi-process only, in
  ``srun_campaign.py``.)
* **e -- ETDRK4.** A short sw2 channel ETDRK4 run (real ``eigh`` basis)
  + a ``jax.grad`` of a quadratic loss through a 3-step
  ``Model.propagator`` run w.r.t. a scalar amplitude. Mirrors
  ``tests/model/time_steppers/test_exponential.py``.

Sharding is entirely fridom's decomposition: every grid is built with
``device_ids=None`` (all visible devices). On the 1-GPU reference leg
that is one device (the eager / replicated reference); under
``srun -n 4`` it shards x across the four processes. All arithmetic is
float64.

Usage:
  1 GPU ref:  CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
                python smoke_refs.py --write-ref
  local CPU:  JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=8 \
                python smoke_refs.py --write-ref
Always with XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion on
a real 4-GPU launch (jax#39100).
"""
from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

import fridom as fr  # noqa: F401 — lazy subpackage root (fr.spatial, ...)
import fridom.nonhydro2 as nh
import fridom.shallowwater2 as sw
from fridom.model import term_predicates as terms
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.model.time_steppers.exponential import ETDRK4
from fridom.model.transforms.balance_expansion import BalanceExpansion
from fridom.spatial.bc import BC
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.base import EigenbasisError, Identity
from fridom.spatial.operators.mixed import ComposedTransform, resolve_transform
from fridom.spatial.operators.spectral import SpectralDerivative
from fridom.spatial.operators.spectral_solve import SpectralSolve
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.coefficient import FourierSpace
from fridom.spatial.spaces.nodal import NodeSet

TWO_PI = 2.0 * np.pi
ART = os.path.dirname(os.path.abspath(__file__))

NH_COMPS = ("u", "v", "w", "b")
SW_COMPS = ("u", "v", "p")

N_ETDRK4 = 16
AB3_DT = 0.2558 / N_ETDRK4


def tol_for(name: str) -> tuple[str, float]:
    """Return the ``(kind, threshold)`` device-invariance tolerance."""
    if name.startswith("e_grad"):
        return ("rel", 1e-6)
    if name.startswith("e_final"):
        return ("abs", 1e-8)
    if name.startswith("d_"):
        return ("abs", 1e-11)
    return ("abs", 1e-10)


# ================================================================
#  Grid / model builders (shared verbatim with srun_campaign.py)
# ================================================================
def build_walled_model(n: int = 8):
    """nh2 walled-vertical model: x, y periodic, z walled."""
    meshes = (
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="z"))
    return nh.Model(
        grid=Grid(meshes, device_ids=None),
        # advection ON so the order-1 balance's quadratic variant
        # (~terms.linear) has nonlinear terms to keep (item a).
        advection=True, dsqr=2.0,
        coriolis=nh.FPlaneCoriolis(f0=1.5),
        stratification=nh.ConstantStratification(n2=3.0),
        time_stepper=AdamBashforth(5e-3, order=3))


def build_periodic_model(n: int = 8):
    """nh2 fully periodic model (the hermitian_reframe route grid)."""
    meshes = tuple(
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name=nm)
        for nm in ("x", "y", "z"))
    return nh.Model(
        grid=Grid(meshes, device_ids=None),
        advection=False, dsqr=1.0,
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=3.0),
        time_stepper=AdamBashforth(5e-3, order=3))


def deterministic_state(em, seed: int = 2):
    """A deterministic nh2 physical state on ``em``'s staggered spaces.

    Built from a fixed-seed numpy array of each component's GLOBAL
    shape (``field.data.shape`` is the global shape, no host fetch), so
    it is byte-identical across device counts before any transform.
    """
    rng = np.random.default_rng(seed)
    fields = {}
    for c in NH_COMPS:
        space = em.physical_space(c)
        shape = em.grid.create_field(space).data.shape
        fields[c] = em.grid.create_field(
            space, data=rng.standard_normal(shape))
    return nh.State(fields)


# ---- item c: mixed walled grid + space -------------------------------
def walled_grid_space(shape, periodic):
    """Return ``(grid, mixed_space)`` for the trig/mixed apply."""
    names = ("x", "y", "z")[:len(shape)]
    lengths = (1.0, 2.0, 3.0)[:len(shape)]
    meshes = tuple(
        IntervalMesh(n, (0.0, ln), periodic=p, name=nm)
        for n, ln, p, nm in zip(shape, lengths, periodic, names,
                                strict=True))
    grid = Grid(meshes, device_ids=None)
    space = None
    for mesh, per in zip(meshes, periodic, strict=True):
        factor = (mesh.center if per
                  else mesh.nodal(NodeSet.CENTER, bc=BC.NEUMANN))
        space = factor if space is None else space * factor
    return grid, space


def helmholtz():
    """The ``I - 0.05 * Laplacian`` mixed spectral operator."""
    lap = (SpectralDerivative()["x"] @ SpectralDerivative()["x"]
           + SpectralDerivative()["y"] @ SpectralDerivative()["y"]
           + SpectralDerivative()["z"] @ SpectralDerivative()["z"])
    return Identity() + (-0.05) * lap


# ---- item d: the picky Tier-2 solve ----------------------------------
def laplacian_2d():
    """The 2-D spectral Laplacian."""
    return (SpectralDerivative()["x"] @ SpectralDerivative()["x"]
            + SpectralDerivative()["y"] @ SpectralDerivative()["y"])


class _HalfOnly:
    """Eigenvalues only on a codomain with a REAL (half) Fourier axis.

    Refusing the fully-complex distributed frame forces
    ``SpectralSolve.slab is None`` -- the replicated-escape /
    taught-error path the Tier-2 check exercises.
    """

    def __init__(self, op):
        self._op = op

    def eigenvalues(self, grid, space):
        if not any(isinstance(f, FourierSpace)
                   and f.scalars is Scalars.REAL
                   for f in space.factors):
            raise EigenbasisError(
                "no eigenvalues on fully complex spectra")
        return self._op.eigenvalues(grid, space)


def picky_solve(*, allow_replicated: bool):
    """Return ``(solve, rhs)`` for the Tier-2 picky spectral solve."""
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    my = IntervalMesh(16, (0.0, 2.0), name="y")
    grid = Grid((mx, my), device_ids=None)
    rhs = grid.create_field(
        init=lambda x, y: jnp.sin(4 * jnp.pi * x) * jnp.cos(jnp.pi * y))
    solve = SpectralSolve(
        _HalfOnly(laplacian_2d()), grid, rhs.function_space,
        allow_replicated=allow_replicated)
    return solve, rhs


# ---- item e: the sw2 ETDRK4 channel ----------------------------------
def channel(nx: int, ny: int):
    """A periodic-x / walled-y shallow-water channel grid."""
    mx = IntervalMesh(nx, (0.0, 1.0), periodic=True, name="x")
    my = IntervalMesh(ny, (0.0, 1.0), periodic=False, name="y")
    return Grid((mx, my), device_ids=None)


def sw_model(grid, stepper, *, filtered: bool, rossby: float = 0.2):
    """A shallow-water channel model (ETDRK4 needs ``filtered=True``)."""
    extra = {"term_filter": ~terms.linear} if filtered else {}
    return sw.Model(
        grid=grid, csqr=1.0, rossby_number=rossby,
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0), advection=True,
        time_stepper=stepper, **extra)


# ================================================================
#  Smoke items -- each returns dict[str, jnp.ndarray]
# ================================================================
def compute_wave_b():
    """Item a: walled-vertical projections + balance expansion."""
    model = build_walled_model()
    em = nh.eigenmodes.from_model(model)
    state = deterministic_state(em, seed=2)
    vort = nh.transforms.VorticalProjection(em)(state)
    wave = nh.transforms.WaveProjection(em)(state)
    bal = BalanceExpansion(model, order=1, lint=False)(state)
    out = {}
    for c in NH_COMPS:
        out[f"a_vort_{c}"] = vort[c].data
        out[f"a_wave_{c}"] = wave[c].data
        out[f"a_bal_{c}"] = bal[c].data
    return out


def compute_no_gather():
    """Item b: random_state + mode on periodic and walled grids."""
    out = {}
    # (i) all-periodic grid -> the hermitian_reframe route
    pmodel = build_periodic_model()
    pem = nh.eigenmodes.from_model(pmodel)
    rvort = nh.random_state(pmodel, "vortical", seed=101)
    rwave = nh.random_state(pmodel, "wave", seed=202)
    _, pmode = pem.mode(1, {"x": 2, "y": 1, "z": 3})
    # (ii) walled grid -> the can_synthesize=False replicated fallback
    wmodel = build_walled_model()
    wem = nh.eigenmodes.from_model(wmodel)
    wwave = nh.random_state(wmodel, "wave", seed=303)
    _, wmode = wem.mode(1, {"x": 2, "y": 1, "z": 2})
    for c in NH_COMPS:
        out[f"b_peri_vort_{c}"] = rvort[c].data
        out[f"b_peri_wave_{c}"] = rwave[c].data
        out[f"b_peri_mode_{c}"] = pmode[c].data
        out[f"b_wall_wave_{c}"] = wwave[c].data
        out[f"b_wall_mode_{c}"] = wmode[c].data
    return out


def compute_mixed():
    """Item c: ComposedTransform.apply_diagonal Helmholtz apply."""
    grid, space = walled_grid_space((16, 16, 16), (True, True, False))
    field = grid.create_field(
        init=lambda x, y, z: jnp.exp(
            -((x - 0.5) ** 2 + (y - 1.0) ** 2 + (z - 1.5) ** 2)))
    field = field.retag(space)
    op = helmholtz()
    tf = resolve_transform(grid, space)
    if not isinstance(tf, ComposedTransform):
        raise TypeError(f"expected ComposedTransform, got {type(tf)}")
    out = tf.apply_diagonal(field, lambda cb: op.eigenvalues(grid, cb))
    return {"c_helmholtz": out.data}


def compute_escape():
    """Item d: the Tier-2 allow_replicated=True escape solve."""
    solve, rhs = picky_solve(allow_replicated=True)
    out = solve(rhs)
    return {"d_escape": out.data}


def compute_etdrk4():
    """Item e: a short sw2 ETDRK4 run + a scalar-amplitude grad."""
    grid = channel(N_ETDRK4, 8)
    dt = 5 * AB3_DT
    basis = sw.eigenbasis(sw_model(
        grid, AdamBashforth(1e-3, order=3), filtered=False))

    src = sw_model(grid, AdamBashforth(1e-3, order=3), filtered=False)
    rng = np.random.default_rng(11)
    fields = {c: rng.standard_normal(src.state[c].data.shape)
              for c in SW_COMPS}

    model = sw_model(grid, ETDRK4(dt, basis), filtered=True)
    model.set_fields(**fields)
    model.advance(2)
    out = {f"e_final_{c}": model.state[c].data for c in SW_COMPS}

    gmodel = sw_model(grid, ETDRK4(dt, basis), filtered=True)
    gmodel.set_fields(**fields)
    run = gmodel.propagator(wrt=("u",), steps=3)
    u0 = gmodel._carry.state["u"].storage  # noqa: SLF001 — owning model

    def loss(alpha):
        result = run((alpha * u0,))
        return sum(jnp.sum(f.data ** 2) for f in result.state)

    grad = jax.grad(loss)(1.0)
    out["e_grad"] = jnp.asarray(grad)
    return out


SMOKE_ITEMS = (
    ("a", "Wave B walled-vertical projections + balance", compute_wave_b),
    ("b", "no-gather random_state + mode synthesis", compute_no_gather),
    ("c", "trig/mixed ComposedTransform.apply_diagonal", compute_mixed),
    ("d", "Tier-2 allow_replicated escape solve", compute_escape),
    ("e", "sw2 ETDRK4 run + scalar-amplitude grad", compute_etdrk4),
)


# ================================================================
#  Driver (single process): compute every item, optionally save refs
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-ref", action="store_true",
                    help="save produced fields as .npy references")
    ap.add_argument("--only", default=None,
                    help="run a single item by tag (a-e)")
    args = ap.parse_args()

    print(f"jax.device_count() = {jax.device_count()}; "
          f"devices = {jax.devices()}", flush=True)
    print(f"config: write_ref={args.write_ref} only={args.only}",
          flush=True)

    for tag, desc, fn in SMOKE_ITEMS:
        if args.only is not None and tag != args.only:
            continue
        print(f"=== item {tag}: {desc} ===", flush=True)
        produced = fn()
        for name, arr in produced.items():
            host = np.asarray(arr)  # single-device -> fully addressable
            print(f"[{tag}] {name}: shape={host.shape} "
                  f"max_abs={np.abs(host).max():.6e}", flush=True)
            if args.write_ref:
                np.save(os.path.join(ART, f"{name}.npy"), host)
        if args.write_ref:
            print(f"[{tag}] saved {len(produced)} reference .npy "
                  f"to {ART}", flush=True)


if __name__ == "__main__":
    main()
