"""VerticalMixing on immersed (cut-cell) grids: the wet-aware column.

Prefix-mirrored shard of ``test_vertical_mixing`` (AGENTS.md
oversized-module rule): the immersed arm of the shared vertical-mixing
closure (immersed_closures_sadourny_plan §5, the implicit twin of the
CL-D2 explicit spelling). The old blanket immersed reject is lifted; the
measure-aware column now additionally respects the partial-bottom
geometry (``model/implicit.py`` ``_diffusion_bands`` / ``_face_fraction``
carry the wet weighting). The band-level oracles live in
``tests/model/test_implicit_kernel_immersed.py``; this shard exercises
the closure at the real model bind / step / autodiff surface:

- bind accepts a static immersed grid (the reject is gone);
- the buoyancy (Neumann) leg conserves the wet-width-weighted tracer
  content to machine zero over a short run on genuine partials;
- the mixing never drives a dry cell (dry values untouched);
- the step path is reverse-mode differentiable (grad wrt the
  diffusivity vs a central FD, the differentiability policy);
- forced-4 multi-device invariance (the z-solve axis stays device-local
  while x, y shard).

Self-contained: the small builders are duplicated, not imported.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.model.closures.vertical_mixing import VerticalMixing
from fridom.model.model import Model, _chunk_body
from fridom.model.module import Module
from fridom.model.time_steppers.imex import CNAB2
from fridom.nonhydro2.modules.core import fv_cgrid_overrides
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain, Slip
from fridom.spatial.meshes.interval import IntervalMesh

TWO_PI = 2.0 * np.pi
DT = 0.02


# ================================================================
#  A minimal FV tracer core + VerticalMixing model on a z-cut grid
# ================================================================
class FVTracer(Module):

    """FV toy core: one buoyancy tracer with a trivial zero tendency."""

    field_declarations = (
        fr.model.FieldDeclaration.tracer(
            "b", space=fr.spatial.Collocated(family="fv")),)

    def grid_dispatch_overrides(self, grid):
        """Install the face-exposing FV C-grid diff profile."""
        return fv_cgrid_overrides(grid.factors)

    @fr.model.term(advances=("b",), linear=True, transports=("b",))
    def zero(self, state, _ctx):
        return {"b": 0.0 * state["b"]}


def data(field):
    return np.asarray(field.data)


def immersed_channel(n=10, *, order=3, device_ids=None):
    """Return a doubly-periodic (x, y) channel, immersed partial bottom.

    The z axis is bounded and stays device-local (a tridiagonal solve is
    serial along it); x, y are periodic and carry any device sharding.
    """
    box = lambda x, y, z: ((z > 0.4) & (z < 2.3)).astype(float)  # noqa: E731, ARG005
    meshes = (
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(n, (0.0, TWO_PI), periodic=False, name="z"))
    return Grid(meshes, immersed=ImmersedDomain(
        box, order=order, slip=Slip.FREE_SLIP), device_ids=device_ids)


def mix_model(grid, *, kb=0.02, seed=1):
    model = Model(grid=grid,
                  modules=(FVTracer(), VerticalMixing(kb=kb, vertical="z")),
                  time_stepper=CNAB2(DT))
    rng = np.random.default_rng(seed)
    model.set_fields(b=0.2 * rng.standard_normal(model.state["b"].data.shape))
    return model


def wet_content(grid, state_b):
    theta = grid.immersed.fraction(state_b.function_space)
    return float(jnp.sum((theta * state_b).integrate().data))


# ================================================================
#  bind + partials
# ================================================================
def test_bind_accepts_a_static_immersed_grid():
    # the wet-aware column lifted the old reject: a static immersed grid
    # binds, assembles and steps finite
    model = mix_model(immersed_channel())
    final = _chunk_body(model._artifacts.record, 4,
                        model._carry, model._stepper)
    b = next(f for f in final.state if f.name == "b")
    assert np.all(np.isfinite(data(b)))


def test_partials_are_genuinely_fractional():
    # the run below exercises the /theta seal only if theta is strictly
    # in (0, 1) somewhere
    grid = immersed_channel()
    model = mix_model(grid)
    theta = data(grid.immersed.fraction(model.state["b"].function_space))
    assert np.any((theta > 0.0) & (theta < 1.0))


# ================================================================
#  wet-content conservation (Neumann buoyancy leg)
# ================================================================
def test_conserves_the_wet_width_weighted_tracer_content():
    # gate (iii): the buoyancy leg is Neumann (no-flux), so the
    # theta-weighted wet tracer content is invariant across steps (the
    # wet-region flux differences telescope, CL-D4)
    grid = immersed_channel()
    model = mix_model(grid)
    before = wet_content(grid, model.state["b"])
    final = _chunk_body(model._artifacts.record, 15,
                        model._carry, model._stepper)
    b = next(f for f in final.state if f.name == "b")
    assert np.all(np.isfinite(data(b)))
    assert abs(wet_content(grid, b) - before) <= 1e-13 * max(
        abs(before), 1.0)


def test_mixing_never_drives_a_dry_cell():
    # the identity dry rows + free-slip cut faces keep every dry DOF
    # exactly at its initial value across the run (no leak into solid)
    grid = immersed_channel()
    model = mix_model(grid)
    mask = data(grid.immersed.mask(model.state["b"].function_space))
    b0 = data(model.state["b"])
    final = _chunk_body(model._artifacts.record, 15,
                        model._carry, model._stepper)
    b = next(f for f in final.state if f.name == "b")
    dry_change = np.abs((data(b) - b0) * (1.0 - mask))
    assert dry_change.max() == 0.0


# ================================================================
#  differentiability policy: grad wrt the diffusivity vs central FD
# ================================================================
def test_immersed_run_is_reverse_mode_differentiable():
    # gate (v): jax.grad of a quadratic loss through a short immersed run
    # w.r.t. the vertical diffusivity is finite and matches a central FD
    # (the /theta seal is what keeps the VJP NaN-free at the cut cells)
    grid = immersed_channel(n=8)
    model = mix_model(grid)
    record, carry, stepper = (
        model._artifacts.record, model._carry, model._stepper)
    kb0 = next(m for m in carry.modules
               if isinstance(m, VerticalMixing)).kb
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    idx = next(i for i, lf in enumerate(leaves) if lf is kb0)

    def loss(x):
        packed = list(leaves)
        packed[idx] = x
        c = jax.tree_util.tree_unflatten(treedef, packed)
        final = _chunk_body(record, 6, c, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    grad = float(jax.grad(loss)(kb0))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    h = 1e-4 * float(kb0)
    fd = (float(loss(kb0 + h)) - float(loss(kb0 - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=1e-4)


# ================================================================
#  forced-4 multi-device invariance
# ================================================================
@pytest.mark.multi_device
def test_forced_four_matches_single_device(forced_devices):
    # the wet-aware z-column solve is device-count invariant: z stays
    # device-local while x, y shard across the forced devices
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        grid = immersed_channel(n=12, device_ids=device_ids)
        model = mix_model(grid, seed=1)
        final = _chunk_body(model._artifacts.record, 6,
                            model._carry, model._stepper)
        results[tag] = {f.name: data(f) for f in final.state}
    for name in results["one"]:
        diff = np.abs(results["many"][name] - results["one"][name]).max()
        np.testing.assert_allclose(diff, 0.0, atol=1e-13)
