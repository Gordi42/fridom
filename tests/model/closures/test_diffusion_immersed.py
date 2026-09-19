"""The diffusion / friction closures on immersed (cut-cell) grids.

Prefix-mirrored shard of ``test_diffusion.py`` (oversized-module rule):
the immersed fraction-weighted behaviour of the harmonic and biharmonic
mixing / friction closures (immersed_closures_sadourny_plan, stage A).
Both families carry the IP-D4 fraction spelling (``_supports_
immersed``; the biharmonic family iterates the weighted pass); no-slip
friction and Smagorinsky keep the per-closure taught reject
(VerticalMixing supports immersed via its wet-aware column — see
test_vertical_mixing_immersed.py). The gates, each run for both
families:

- A-G1 (keystone): immersed **staircase** ≡ **walled** at machine zero
  (diffusion and friction);
- A-G2: all-wet ≡ unimmersed (the fraction weighting is a bitwise
  no-op when alpha = theta = 1);
- A-G3: wet tracer content (diffusion) / wet **tangential** momentum
  (friction) conserved to machine zero on genuine partials;
- A-G4: taught errors on the real bind path;
- A-G5: reverse-mode autodiff (grad wrt kappa / nu vs central FD);
- A-G6: forced-4 multi-device invariance.

The builders are duplicated (self-contained shard, import-mode
importlib) rather than imported across test files.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.closures.diffusion import (
    BiharmonicDiffusion,
    BiharmonicFriction,
    HarmonicDiffusion,
    HarmonicFriction,
    _scale_divergence,
    _weight_flux,
)
from fridom.model.model import Model, _chunk_body
from fridom.model.module import Module
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.core import fv_cgrid_overrides
from fridom.nonhydro2.modules.smagorinsky_lilly import SmagorinskyLilly
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain, Slip
from fridom.spatial.meshes.interval import IntervalMesh

TWO_PI = 2.0 * np.pi
DT = 1e-3


# ================================================================
#  FV cores (velocities on the C-grid faces, cells on CellAvg) + the
#  grid-aware FV C-grid diff profile that exposes the face flux
# ================================================================
class FVCore(Module):

    """FV toy core: u/v/w on the C-grid faces, an FV cell tracer b."""

    field_declarations = (
        fr.model.FieldDeclaration.velocity(
            "u", "x", space=fr.spatial.Staggered("x", family="fv")),
        fr.model.FieldDeclaration.velocity(
            "v", "y", space=fr.spatial.Staggered("y", family="fv")),
        fr.model.FieldDeclaration.velocity(
            "w", "z", space=fr.spatial.Staggered("z", family="fv")),
        fr.model.FieldDeclaration.tracer(
            "b", space=fr.spatial.Collocated(family="fv")),
    )

    def grid_dispatch_overrides(self, grid):
        """Install the face-exposing FV C-grid diff profile."""
        return fv_cgrid_overrides(grid.factors)

    @fr.model.term(advances=("u", "v", "w", "b"), linear=True,
                   transports=("u", "v", "w", "b"))
    def zero(self, state, _ctx):
        return {k: 0.0 * state[k] for k in ("u", "v", "w", "b")}


def data(field):
    return np.asarray(field.data)


def periodic_meshes(n, length=TWO_PI, names=("x", "y", "z")):
    return tuple(
        IntervalMesh(n, (0.0, length), periodic=True, name=nm)
        for nm in names)


def make_model(grid, *closures, fields=None):
    """Build a FV core + closures model on ``grid`` (optional IC)."""
    model = Model(grid=grid, modules=(FVCore(), *closures),
                  time_stepper=AdamBashforth(DT, order=2))
    if fields is not None:
        model.set_fields(**fields)
    return model


def seeded(model, seed, fields=("u", "v", "w", "b"), scale=0.2):
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        k: scale * rng.standard_normal(model.state[k].data.shape)
        for k in fields})
    return model


# ================================================================
#  A-G1 keystone: immersed staircase ≡ walled at machine zero
# ================================================================
def _staircase_and_walled(closure_factory, fields):
    """Build the staircase and walled models with a shared random IC.

    The staircase is a face-aligned {0, 1} immersed box in a periodic
    grid; the walled model lives on the wet sub-box.
    """
    npg, lo, hi = 12, 3, 9
    box = lambda x, y, z: (  # noqa: E731
        (x > lo) & (x < hi) & (y > lo) & (y < hi)
        & (z > lo) & (z < hi)).astype(float)
    imm = make_model(
        Grid(tuple(IntervalMesh(npg, (0.0, 12.0), periodic=True, name=nm)
                   for nm in ("x", "y", "z")),
             immersed=ImmersedDomain(box)),
        closure_factory())
    wal = make_model(
        Grid(tuple(IntervalMesh(6, (3.0, 9.0), periodic=False, name=nm)
                   for nm in ("x", "y", "z"))),
        closure_factory())
    rng = np.random.default_rng(11)
    shapes = {k: wal.state[k].data.shape for k in fields}
    ic = {k: 0.1 * rng.standard_normal(shapes[k]) for k in fields}
    full = {}
    for k in fields:
        arr = np.zeros((npg, npg, npg))
        s = shapes[k]
        arr[lo:lo + s[0], lo:lo + s[1], lo:lo + s[2]] = ic[k]
        full[k] = arr
    imm.set_fields(**full)
    wal.set_fields(**ic)
    return imm, wal, (npg, lo, hi), shapes


def test_diffusion_staircase_matches_walled_at_machine_zero():
    # A-G1 keystone: a face-aligned {0, 1} box vs the walled FV model on
    # the wet sub-box — the wet-region diffusion tendency agrees to
    # machine zero (the min-rule alpha = 0 at wet/dry faces is the exact
    # no-flux wall)
    imm, wal, (_npg, lo, _hi), shapes = _staircase_and_walled(
        lambda: HarmonicDiffusion(3e-3), ("b",))
    tdi = imm.tendency(imm.state)["b"]
    tdw = wal.tendency(wal.state)["b"]
    s = shapes["b"]
    sub = data(tdi)[lo:lo + s[0], lo:lo + s[1], lo:lo + s[2]]
    assert np.abs(sub - data(tdw)).max() < 1e-14


def test_friction_staircase_matches_walled_at_machine_zero():
    # A-G1 keystone for friction: the wet-region momentum tendency of
    # every velocity component agrees with the walled free-slip model
    imm, wal, (_npg, lo, _hi), shapes = _staircase_and_walled(
        lambda: HarmonicFriction(5e-3), ("u", "v", "w"))
    tdi = imm.tendency(imm.state)
    tdw = wal.tendency(wal.state)
    for k in ("u", "v", "w"):
        s = shapes[k]
        sub = data(tdi[k])[lo:lo + s[0], lo:lo + s[1], lo:lo + s[2]]
        assert np.abs(sub - data(tdw[k])).max() < 1e-14, k


@pytest.mark.parametrize(
    ("closure_factory", "fields"),
    [pytest.param(lambda: BiharmonicDiffusion(3e-3), ("b",),
                  id="diffusion"),
     pytest.param(lambda: BiharmonicFriction(5e-3), ("u", "v", "w"),
                  id="friction")])
def test_biharmonic_staircase_matches_walled(closure_factory, fields):
    # A-G1 for the iterated operator: both passes see the closed faces
    # (alpha = 0) and the sealed inner Laplacian exactly as the walled
    # model sees its walls on both passes (Griffies & Hallberg), so the
    # wet-region tendency agrees to rounding of the two-pass chain
    imm, wal, (_npg, lo, _hi), shapes = _staircase_and_walled(
        closure_factory, fields)
    tdi = imm.tendency(imm.state)
    tdw = wal.tendency(wal.state)
    for k in fields:
        s = shapes[k]
        sub = data(tdi[k])[lo:lo + s[0], lo:lo + s[1], lo:lo + s[2]]
        scale = np.abs(data(tdw[k])).max()
        assert scale > 0.0
        assert np.abs(sub - data(tdw[k])).max() < 1e-13 * scale, k


# ================================================================
#  A-G2: all-wet ≡ unimmersed (the fraction weighting is a no-op)
# ================================================================
def test_weighting_helpers_are_bitwise_noops_when_all_wet():
    # A-G2 (closure level): on an all-wet grid alpha = theta = 1 exactly,
    # so weight_flux / scale_divergence are bitwise identities — the
    # immersed spelling reduces to the direct chain bit-for-bit
    grid = Grid(periodic_meshes(10),
                immersed=ImmersedDomain(lambda x, y, z: x * 0.0 + 1.0))  # noqa: ARG005
    model = seeded(make_model(grid, HarmonicDiffusion(3e-3)), 1)
    q = model.state["b"]
    flux = q.diff("x") * 3e-3
    weighted = _weight_flux(grid.immersed, flux)
    assert np.array_equal(data(weighted), data(flux))
    div = weighted.diff("x")
    scaled = _scale_divergence(grid.immersed, div)
    assert np.array_equal(data(scaled), data(div))


def test_helpers_are_noops_off_an_immersed_grid():
    # the parity guard: immersed=None returns the operands untouched
    sentinel = object()
    assert _weight_flux(None, sentinel) is sentinel
    assert _scale_divergence(None, sentinel) is sentinel
    assert _scale_divergence(object(), None) is None


@pytest.mark.parametrize(
    ("closure_factory", "fields"),
    [(lambda: HarmonicDiffusion(3e-3), ("b",)),
     (lambda: HarmonicFriction(5e-3), ("u", "v", "w"))])
def test_all_wet_matches_unimmersed(closure_factory, fields):
    # A-G2 (model level): an all-wet immersed model reproduces the
    # unimmersed run to machine zero (the residual is XLA fusion
    # ordering of two independently-negotiated grids, ~1 ULP)
    allwet = ImmersedDomain(lambda x, y, z: x * 0.0 + 1.0)  # noqa: ARG005
    im = seeded(make_model(Grid(periodic_meshes(10), immersed=allwet),
                           closure_factory()), 7, fields)
    un = seeded(make_model(Grid(periodic_meshes(10)),
                           closure_factory()), 7, fields)
    tdi = im.tendency(im.state)
    tdu = un.tendency(un.state)
    for k in fields:
        assert np.abs(data(tdi[k]) - data(tdu[k])).max() < 1e-15, k


@pytest.mark.parametrize(
    ("closure_factory", "fields"),
    [pytest.param(lambda: BiharmonicDiffusion(3e-3), ("b",),
                  id="diffusion"),
     pytest.param(lambda: BiharmonicFriction(5e-3), ("u", "v", "w"),
                  id="friction")])
def test_biharmonic_all_wet_matches_unimmersed(closure_factory, fields):
    # A-G2 for the iterated operator (relative: the two-pass tendency is
    # O(10), so one ULP is no longer below an absolute 1e-15)
    allwet = ImmersedDomain(lambda x, y, z: x * 0.0 + 1.0)  # noqa: ARG005
    im = seeded(make_model(Grid(periodic_meshes(10), immersed=allwet),
                           closure_factory()), 7, fields)
    un = seeded(make_model(Grid(periodic_meshes(10)),
                           closure_factory()), 7, fields)
    tdi = im.tendency(im.state)
    tdu = un.tendency(un.state)
    for k in fields:
        scale = np.abs(data(tdu[k])).max()
        assert scale > 0.0
        assert np.abs(data(tdi[k]) - data(tdu[k])).max() < 1e-14 * scale, k


# ================================================================
#  A-G3: telescoping conservation on genuine partials
# ================================================================
def _partial_channel(order=3):
    """Build a channel with genuine fractional theta in z.

    Periodic x, y with an immersed partial bottom in z (fractional
    theta via order-``order`` Gauss-Legendre quadrature).
    """
    box = lambda x, y, z: ((z > 0.18) & (z < 2.3)).astype(float)  # noqa: E731, ARG005
    return Grid(periodic_meshes(12),
                immersed=ImmersedDomain(box, order=order,
                                        slip=Slip.FREE_SLIP))


def test_partials_are_genuinely_fractional():
    # the quadrature fraction is strictly between 0 and 1 somewhere (the
    # A-G3 precondition — a {0, 1} staircase would not exercise the seal)
    grid = _partial_channel()
    model = make_model(grid, HarmonicDiffusion(3e-3))
    theta = data(grid.immersed.fraction(model.state["b"].function_space))
    assert np.any((theta > 0.0) & (theta < 1.0))


def test_diffusion_conserves_wet_tracer_content():
    # A-G3: the theta-weighted tracer content drifts machine-zero over a
    # short run on genuine partials (the wet-region flux differences
    # telescope, CL-D4)
    grid = _partial_channel()
    model = seeded(make_model(grid, HarmonicDiffusion(3e-3)), 3)
    theta = grid.immersed.fraction(model.state["b"].function_space)

    def total(state_b):
        return float(jnp.sum((theta * state_b).integrate().data))

    before = total(model.state["b"])
    final = _chunk_body(model._artifacts.record, 15,
                        model._carry, model._stepper)
    b_final = next(f for f in final.state if f.name == "b")
    assert abs(total(b_final) - before) <= 1e-13 * max(abs(before), 1.0)


def test_friction_conserves_wet_tangential_momentum():
    # A-G3: free-slip friction conserves the theta-weighted TANGENTIAL
    # momentum to machine zero (u, v are tangential to the z-cut
    # boundary — the cut-face stress is zeroed, so the flux telescopes).
    # The wall-normal w carries the physical impermeable-boundary viscous
    # force (matching walls, the keystone) and is NOT conserved — checked
    # separately below.
    grid = _partial_channel()
    model = seeded(make_model(grid, HarmonicFriction(5e-3)), 5,
                   fields=("u", "v", "w"))
    td = model.tendency(model.state)
    for k in ("u", "v"):
        theta = grid.immersed.fraction(model.state[k].function_space)
        drift = float(jnp.sum((theta * td[k]).integrate().data))
        assert abs(drift) < 1e-13, (k, drift)


def test_biharmonic_diffusion_conserves_wet_tracer_content():
    # A-G3 for the iterated operator: the outer pass is a theta-scaled
    # divergence of alpha-weighted fluxes, so the wet content telescopes
    # whatever the (sealed) inner Laplacian is
    grid = _partial_channel()
    model = seeded(make_model(grid, BiharmonicDiffusion(3e-3)), 3)
    theta = grid.immersed.fraction(model.state["b"].function_space)

    def total(state_b):
        return float(jnp.sum((theta * state_b).integrate().data))

    before = total(model.state["b"])
    final = _chunk_body(model._artifacts.record, 15,
                        model._carry, model._stepper)
    b_final = next(f for f in final.state if f.name == "b")
    assert np.isfinite(total(b_final))
    assert abs(total(b_final) - before) <= 1e-13 * max(abs(before), 1.0)


def test_biharmonic_friction_conserves_wet_tangential_momentum():
    grid = _partial_channel()
    model = seeded(make_model(grid, BiharmonicFriction(5e-3)), 5,
                   fields=("u", "v", "w"))
    td = model.tendency(model.state)
    for k in ("u", "v"):
        theta = grid.immersed.fraction(model.state[k].function_space)
        scale = float(jnp.sum(jnp.abs((theta * td[k]).integrate().data)))
        drift = float(jnp.sum((theta * td[k]).integrate().data))
        assert abs(drift) < 1e-13 * max(scale, 1.0), (k, drift)


def test_biharmonic_diffusion_is_dissipative_on_partials():
    # -L(L(q)) with L self-adjoint in the theta-weighted inner product:
    # <q, -L L q>_theta = -<L q, L q>_theta <= 0 (variance never grows)
    grid = _partial_channel()
    model = seeded(make_model(grid, BiharmonicDiffusion(3e-3)), 9)
    b = model.state["b"]
    theta = grid.immersed.fraction(b.function_space)
    td = model.tendency(model.state)["b"]
    rate = float(jnp.sum((theta * b * td).integrate().data))
    assert rate < 0.0


def test_wall_normal_friction_carries_a_physical_boundary_force():
    # the wall-normal component (w, normal to the z-cut boundary) is not
    # conserved — impermeability is a Dirichlet wall with a real normal
    # viscous force, exactly as in the walled model (this is what makes
    # A-G1 staircase ≡ walled hold; free-slip conserves only tangentials)
    grid = _partial_channel()
    model = seeded(make_model(grid, HarmonicFriction(5e-3)), 5,
                   fields=("u", "v", "w"))
    td = model.tendency(model.state)
    theta = grid.immersed.fraction(model.state["w"].function_space)
    drift = float(jnp.sum((theta * td["w"]).integrate().data))
    assert abs(drift) > 1e-6


# ================================================================
#  A-G4: taught errors on the real bind path
# ================================================================
def _nh_immersed_grid(n=10):
    box = lambda x, y, z: ((x > 1.0) & (x < 5.0)).astype(float)  # noqa: E731, ARG005
    return Grid(periodic_meshes(n), immersed=ImmersedDomain(box))


def _build_nh(*modules_extra):
    return nh.Model(
        grid=_nh_immersed_grid(),
        core=nh.Core(),
        time_stepper=AdamBashforth(0.01, order=3),
        coriolis=FPlaneCoriolis(f0=1.0),
        buoyancy=nh.ConstantStratification(n2=1.0),
        advection=None,
        modules_extra=modules_extra)


def test_smagorinsky_on_immersed_is_a_taught_error():
    with pytest.raises(NotImplementedError,
                       match="immersed_closures_sadourny_plan"):
        _build_nh(SmagorinskyLilly())


def test_no_slip_friction_on_immersed_is_a_taught_error():
    with pytest.raises(NotImplementedError, match=r"slip='no'"):
        _build_nh(HarmonicFriction(1e-2, slip="no"))


def test_no_slip_mapping_on_immersed_is_a_taught_error():
    with pytest.raises(NotImplementedError, match=r"slip='no'"):
        _build_nh(HarmonicFriction(1e-2, slip={"u": "no", "v": "free",
                                               "w": "free"}))


@pytest.mark.parametrize(
    "closure", [BiharmonicDiffusion(1e-6), BiharmonicFriction(1e-6)])
def test_biharmonic_binds_on_the_real_immersed_path(closure):
    # the former bind-time reject is lifted: the biharmonic family now
    # carries the fraction spelling and assembles on the nh.Model path
    model = _build_nh(closure)
    rng = np.random.default_rng(4)
    model.set_fields(**{
        k: 0.1 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "w", "b")})
    td = model.tendency(model.state)
    for name in ("u", "v", "w", "b"):
        assert np.all(np.isfinite(data(td[name]))), name


# ================================================================
#  Capability flags + extra_halo declaration
# ================================================================
def test_capability_flags():
    # CL-D1: the divergence-form families opt into immersed support
    # (Smagorinsky keeps the reject — see the taught-error test above)
    assert HarmonicDiffusion._supports_immersed is True
    assert HarmonicFriction._supports_immersed is True
    assert BiharmonicDiffusion._supports_immersed is True
    assert BiharmonicFriction._supports_immersed is True
    assert SmagorinskyLilly._supports_immersed is False


def test_extra_halo_is_declared_on_immersed_and_none_otherwise():
    grid = _partial_channel()
    im = make_model(grid, HarmonicDiffusion(3e-3))
    closure = next(m for m in im._carry.modules
                   if isinstance(m, HarmonicDiffusion))
    assert closure._immersed is grid.immersed
    # one-cell FD-stencil halo per coordinate (the exact harmonic reach)
    assert closure.extra_halo == HaloSpec(
        dict.fromkeys(grid.names, 1))
    # off an immersed grid the closure stays fully halo-traced (None)
    un = make_model(Grid(periodic_meshes(12)), HarmonicDiffusion(3e-3))
    closure_un = next(m for m in un._carry.modules
                      if isinstance(m, HarmonicDiffusion))
    assert closure_un._immersed is None
    assert closure_un.extra_halo is None


def test_biharmonic_extra_halo_is_two_cells_on_immersed():
    # the iterated pass reaches two cells per coordinate
    grid = _partial_channel()
    im = make_model(grid, BiharmonicDiffusion(3e-3))
    closure = next(m for m in im._carry.modules
                   if isinstance(m, BiharmonicDiffusion))
    assert closure._immersed is grid.immersed
    assert closure.extra_halo == HaloSpec(dict.fromkeys(grid.names, 2))
    un = make_model(Grid(periodic_meshes(12)), BiharmonicDiffusion(3e-3))
    closure_un = next(m for m in un._carry.modules
                      if isinstance(m, BiharmonicDiffusion))
    assert closure_un.extra_halo is None


def test_biharmonic_tendency_is_zero_on_dry_dofs():
    grid = _partial_channel()
    model = seeded(make_model(grid, BiharmonicDiffusion(3e-3),
                              BiharmonicFriction(5e-3)), 0)
    td = model.tendency(model.state)
    for name in ("u", "v", "w", "b"):
        mask = grid.immersed.mask(model.state[name].function_space)
        dry = data(td[name]) * (1.0 - data(mask))
        assert np.abs(dry).max() == 0.0, name


def test_closure_tendency_is_zero_on_dry_dofs():
    # the sealed scale_divergence zeroes the tendency on every dry DOF
    # (theta = 0), so the masked closure never drives a dry cell — the
    # dry-DOF hygiene the fraction weighting guarantees (the model-level
    # MaskState then keeps the field itself dead)
    grid = _partial_channel()
    model = seeded(make_model(grid, HarmonicDiffusion(3e-3),
                              HarmonicFriction(5e-3)), 0)
    td = model.tendency(model.state)
    for name in ("u", "v", "w", "b"):
        mask = grid.immersed.mask(model.state[name].function_space)
        dry = data(td[name]) * (1.0 - data(mask))
        assert np.abs(dry).max() == 0.0, name


# ================================================================
#  A-G5: reverse-mode autodiff (grad wrt the coefficient vs central FD)
# ================================================================
def _grad_loss(closure, attr, fields, n_steps=6):
    grid = Grid(periodic_meshes(8),
                immersed=ImmersedDomain(
                    lambda x, y, z: ((z > 0.18) & (z < 2.3)).astype(float),  # noqa: ARG005
                    order=3))
    model = seeded(make_model(grid, closure), 0, fields)
    record, carry, stepper = (
        model._artifacts.record, model._carry, model._stepper)
    leaf = getattr(next(m for m in carry.modules
                        if isinstance(m, type(closure))), attr)
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    idx = next(i for i, lf in enumerate(leaves) if lf is leaf)

    def loss(theta):
        packed = list(leaves)
        packed[idx] = theta
        c = jax.tree_util.tree_unflatten(treedef, packed)
        final = _chunk_body(record, n_steps, c, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    return loss, jnp.asarray(leaf, dtype=jnp.float64)


@pytest.mark.parametrize(
    ("closure", "attr", "fields"),
    [(HarmonicDiffusion(3e-3), "kappa", ("b",)),
     (HarmonicFriction(5e-3), "nu", ("u", "v", "w")),
     (BiharmonicDiffusion(3e-3), "kappa", ("b",)),
     (BiharmonicFriction(5e-3), "nu", ("u", "v", "w"))])
def test_grad_matches_central_fd(closure, attr, fields):
    loss, x0 = _grad_loss(closure, attr, fields)
    g = float(jax.grad(loss)(x0))
    assert np.isfinite(g)
    eps = 1e-4
    fd = float((loss(x0 * (1 + eps)) - loss(x0 * (1 - eps)))
               / (2 * x0 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-4)


# ================================================================
#  A-G6: forced-4 multi-device invariance
# ================================================================
@pytest.mark.multi_device
def test_forced_four_matches_single_device(forced_devices):
    # the depth-1 extra_halo must sync the inter-shard halo so the
    # fraction-weighted closures are device-count invariant
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    box = lambda x, y, z: (  # noqa: E731
        (x > 1.0) & (x < 5.0) & (y > 1.0) & (y < 5.0)
        & (z > 0.2) & (z < 2.5)).astype(float)
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        grid = Grid(periodic_meshes(12),
                    immersed=ImmersedDomain(box, order=3),
                    device_ids=device_ids)
        model = seeded(make_model(grid, HarmonicDiffusion(3e-3),
                                  HarmonicFriction(5e-3)), 1)
        final = _chunk_body(model._artifacts.record, 6,
                            model._carry, model._stepper)
        results[tag] = {f.name: data(f) for f in final.state}
    for name in results["one"]:
        diff = np.abs(results["many"][name] - results["one"][name]).max()
        np.testing.assert_allclose(diff, 0.0, atol=1e-13)


@pytest.mark.multi_device
def test_biharmonic_forced_four_matches_single_device(forced_devices):
    # the depth-2 extra_halo must sync both passes across the shards
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    box = lambda x, y, z: (  # noqa: E731
        (x > 1.0) & (x < 5.0) & (y > 1.0) & (y < 5.0)
        & (z > 0.2) & (z < 2.5)).astype(float)
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        grid = Grid(periodic_meshes(12),
                    immersed=ImmersedDomain(box, order=3),
                    device_ids=device_ids)
        model = seeded(make_model(grid, BiharmonicDiffusion(3e-3),
                                  BiharmonicFriction(5e-3)), 1)
        final = _chunk_body(model._artifacts.record, 6,
                            model._carry, model._stepper)
        results[tag] = {f.name: data(f) for f in final.state}
    for name in results["one"]:
        diff = np.abs(results["many"][name] - results["one"][name]).max()
        np.testing.assert_allclose(diff, 0.0, atol=1e-13)
