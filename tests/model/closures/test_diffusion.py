"""The diffusion closures: discrete rates, targets, coefficients."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.model.closures.diffusion import (
    BiharmonicDiffusion,
    BiharmonicFriction,
    HarmonicDiffusion,
    HarmonicFriction,
)
from fridom.model.declarations import Lifecycle
from fridom.model.errors import AssemblyError
from fridom.model.field_table import (
    FieldRecord,
    FieldTable,
)
from fridom.model.model import Model, _chunk_body
from fridom.model.module import Module
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 16
L = 1.0
DX = L / N
DT = 1e-3


class Core(Module):

    """Toy core: velocities u/v, tracers b/c, one trivial term."""

    field_declarations = (
        fr.model.FieldDeclaration.velocity(
            "u", "x", space=fr.spatial.Staggered("x")),
        fr.model.FieldDeclaration.velocity(
            "v", "z", space=fr.spatial.Staggered("z")),
        fr.model.FieldDeclaration.tracer("b"),
        fr.model.FieldDeclaration.tracer("c"),
    )

    @fr.model.term(advances=("u", "v", "b", "c"), linear=True,
             transports=("u", "v", "b", "c"))
    def zero(self, state, _ctx):
        return {name: 0.0 * state[name]
                for name in ("u", "v", "b", "c")}


def make_grid(names=("x", "z"), periodic=True):
    return Grid(tuple(
        IntervalMesh(N, (0.0, L), periodic=periodic, name=name)
        for name in names))


def make_model(closure, grid=None):
    return Model(grid=grid or make_grid(),
                 modules=(Core(), closure),
                 time_stepper=AdamBashforth(DT, order=2))


def lam(m):
    """Exact symbol of the discrete 3-point Laplacian, mode m."""
    return (2.0 * np.sin(np.pi * m / N) / DX) ** 2


def coords():
    ax = (np.arange(N) + 0.5) * DX
    return np.meshgrid(ax, ax, indexing="ij")


def data(field):
    return np.asarray(field.data)


# ================================================================
#  The discrete damping rate (single Fourier modes, exact symbol)
# ================================================================
def test_harmonic_mode_damped_at_the_discrete_rate():
    kappa = 3e-3
    model = make_model(HarmonicDiffusion(kappa))
    x, _ = coords()
    model.set_fields(c=np.sin(2 * np.pi * x))
    td = model.tendency(model.state)
    want = -kappa * lam(1) * data(model.state["c"])
    np.testing.assert_allclose(data(td["c"]), want, atol=1e-14)


def test_biharmonic_mode_damped_at_the_squared_symbol():
    kappa = 1e-5
    model = make_model(BiharmonicDiffusion(kappa))
    x, _ = coords()
    model.set_fields(b=np.sin(4 * np.pi * x))
    td = model.tendency(model.state)
    want = -kappa * lam(2) ** 2 * data(model.state["b"])
    np.testing.assert_allclose(data(td["b"]), want, rtol=1e-12)


def test_anisotropic_vertical_coefficient():
    kh, kv = 2e-3, 7e-3
    model = make_model(
        HarmonicDiffusion(kh, kappa_v=kv, vertical="z"))
    x, z = coords()
    model.set_fields(b=np.sin(2 * np.pi * x) + np.sin(4 * np.pi * z))
    td = model.tendency(model.state)
    want = -(kh * lam(1) * np.sin(2 * np.pi * x)
             + kv * lam(2) * np.sin(4 * np.pi * z))
    np.testing.assert_allclose(data(td["b"]), want, atol=1e-13)


def test_biharmonic_anisotropic_vertical_coefficient():
    kh, kv = 1e-5, 3e-5
    model = make_model(
        BiharmonicFriction(kh, nu_v=kv, vertical="z",
                           fields=fr.model.roles.Velocity("x")))
    x, z = coords()
    model.set_fields(u=np.sin(2 * np.pi * x) + np.cos(2 * np.pi * z))
    td = model.tendency(model.state)
    # cross terms vanish for single-direction modes: the diagonal
    # damping applies kh (kv) along x (z) at the squared symbol,
    # PLUS the Griffies cross term 2 sqrt(kh kv) lam_x lam_z acting
    # on the product structure — absent here (each mode varies in
    # one direction only)
    want = -(kh * lam(1) ** 2 * np.sin(2 * np.pi * x)
             + kv * lam(1) ** 2 * np.cos(2 * np.pi * z))
    np.testing.assert_allclose(data(td["u"]), want, atol=1e-12)


def test_per_field_coefficient_mapping():
    model = make_model(HarmonicDiffusion({"b": 2e-3, "c": 5e-3}))
    x, z = coords()
    model.set_fields(b=np.sin(2 * np.pi * x), c=np.sin(4 * np.pi * z))
    td = model.tendency(model.state)
    np.testing.assert_allclose(
        data(td["b"]), -2e-3 * lam(1) * data(model.state["b"]),
        atol=1e-14)
    np.testing.assert_allclose(
        data(td["c"]), -5e-3 * lam(2) * data(model.state["c"]),
        atol=1e-14)


def test_ramped_scalar_coefficient_resolves_in_step():
    k0 = 4e-3
    ramp = fr.model.Ramp(k0, 0.0, period=1.0)
    model = make_model(HarmonicDiffusion(ramp))
    x, _ = coords()
    model.set_fields(c=np.sin(2 * np.pi * x))
    td = model.tendency(model.state)  # t = 0: the ramp start value
    want = -k0 * lam(1) * data(model.state["c"])
    np.testing.assert_allclose(data(td["c"]), want, atol=1e-14)


# ================================================================
#  Role-driven targeting: friction hits velocities, not tracers
# ================================================================
def test_friction_targets_velocities_and_not_tracers():
    nu = 1e-2
    model = make_model(HarmonicFriction(nu))
    x, _ = coords()
    model.set_fields(u=np.sin(2 * np.pi * x), b=np.sin(2 * np.pi * x))
    td = model.tendency(model.state)
    np.testing.assert_allclose(
        data(td["u"]), -nu * lam(1) * data(model.state["u"]),
        atol=1e-13)
    assert np.abs(data(td["b"])).max() == 0.0


def test_mixing_targets_tracers_and_not_velocities():
    model = make_model(HarmonicDiffusion(1e-2))
    x, _ = coords()
    model.set_fields(u=np.sin(2 * np.pi * x), b=np.sin(2 * np.pi * x))
    td = model.tendency(model.state)
    assert np.abs(data(td["u"])).max() == 0.0
    assert np.abs(data(td["b"])).max() > 0.0


def test_fields_and_exclude_narrow_the_targets():
    model = make_model(HarmonicDiffusion(1e-2, exclude=("c",)))
    closure = model._carry.modules[1]
    assert closure.targets == ("b",)
    x, _ = coords()
    model.set_fields(c=np.sin(2 * np.pi * x))
    td = model.tendency(model.state)
    assert np.abs(data(td["c"])).max() == 0.0


# ================================================================
#  Published parameters: update_parameters sweeps
# ================================================================
def test_coefficients_are_published_parameters():
    model = make_model(HarmonicDiffusion(1e-3, kappa_v=2e-3,
                                         vertical="z"))
    assert "mixing.kappa" in model.parameters
    assert "mixing.kappa_v" in model.parameters
    friction = make_model(BiharmonicFriction(1e-5))
    assert "friction.nu4" in friction.parameters
    assert "friction.nu4_v" not in friction.parameters


def test_update_parameters_sweeps_the_scalar_coefficient():
    model = make_model(HarmonicDiffusion(1e-3))
    x, _ = coords()
    model.set_fields(c=np.sin(2 * np.pi * x))
    before = data(model.tendency(model.state)["c"])
    model.update_parameters({"mixing.kappa": 2e-3})
    after = data(model.tendency(model.state)["c"])
    np.testing.assert_allclose(after, 2.0 * before, atol=1e-15)


def test_update_parameters_sweeps_a_per_field_mapping():
    model = make_model(HarmonicDiffusion({"b": 1e-3, "c": 1e-3}))
    x, _ = coords()
    model.set_fields(b=np.sin(2 * np.pi * x))
    before = data(model.tendency(model.state)["b"])
    model.update_parameters(
        {"mixing.kappa": {"b": 3e-3, "c": 1e-3}})
    after = data(model.tendency(model.state)["b"])
    np.testing.assert_allclose(after, 3.0 * before, atol=1e-15)


# ================================================================
#  Term predicates: owned_by drops closures; linear keeps them
# ================================================================
def test_variant_owned_by_closurebase_drops_all_closure_terms():
    model = make_model(HarmonicFriction(1e-2, fields=("u", "v", "b")))
    x, z = coords()
    model.set_fields(u=np.sin(2 * np.pi * x), b=np.cos(2 * np.pi * z))
    inviscid = model.variant(
        term_filter=~fr.model.term_predicates.owned_by(
            fr.model.closures.ClosureBase))
    td = inviscid.tendency(model.state)
    for name in ("u", "v", "b", "c"):
        assert np.abs(data(td[name])).max() == 0.0
    assert np.abs(data(model.tendency(model.state)["u"])).max() > 0.0


def test_diffusion_terms_are_tagged_linear():
    model = make_model(HarmonicDiffusion(1e-3))
    x, _ = coords()
    model.set_fields(b=np.sin(2 * np.pi * x))
    linear = fr.model.linearize(model)
    np.testing.assert_allclose(
        data(linear.tendency(model.state)["b"]),
        data(model.tendency(model.state)["b"]), atol=0.0)


# ================================================================
#  Construction-time coefficient validation
# ================================================================
def test_negative_biharmonic_coefficient_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        BiharmonicDiffusion(-1e-5)


def test_negative_biharmonic_mapping_entry_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        BiharmonicFriction({"u": -1e-5})


def test_negative_harmonic_coefficient_is_allowed():
    HarmonicDiffusion(-1e-3)  # antidiffusion: your foot, your aim


def test_ramp_inside_a_mapping_is_rejected():
    with pytest.raises(TypeError, match="plain numbers"):
        HarmonicDiffusion({"b": fr.model.Ramp(0.0, 1.0, period=1.0)})


def test_empty_mapping_is_rejected():
    with pytest.raises(ValueError, match="mapping is empty"):
        HarmonicDiffusion({})


def test_non_numeric_coefficient_is_rejected():
    with pytest.raises(TypeError, match="takes a number"):
        HarmonicDiffusion("large")


def test_non_string_mapping_key_is_rejected():
    with pytest.raises(TypeError, match="keys are field names"):
        HarmonicDiffusion({1: 1e-3})


# ================================================================
#  Bind-time validation (assembly errors through Model)
# ================================================================
def test_unknown_mapping_key_is_an_assembly_error():
    with pytest.raises(AssemblyError,
                       match=r"unknown target 'q' in kappa="):
        make_model(HarmonicDiffusion({"b": 1e-3, "c": 1e-3,
                                      "q": 1e-3}))


def test_uncovered_target_is_an_assembly_error():
    with pytest.raises(AssemblyError,
                       match=r"no coefficient for target 'c'"):
        make_model(HarmonicDiffusion({"b": 1e-3}))


def test_vertical_coefficient_without_vertical_coordinate():
    grid = make_grid(names=("x", "y"))
    with pytest.raises(AssemblyError, match="vertical coordinate"):
        make_model(HarmonicDiffusion(1e-3, kappa_v=2e-3), grid=grid)


def test_target_without_coordinate_axes_is_rejected():
    grid = make_grid()
    record = FieldRecord(
        name="s", owner=0, owner_type="Core",
        pattern=fr.spatial.Collocated(),
        space=SimpleNamespace(names=()),
        lifecycle=Lifecycle.PROGNOSTIC,
        roles=frozenset({fr.model.roles.TRACER}),
        host_writable=False, metadata=None)
    closure = HarmonicDiffusion(1e-3)
    with pytest.raises(AssemblyError, match="no coordinate axes"):
        closure.bind(FieldTable((record,), grid))


# ================================================================
#  Reverse-mode AD: the biharmonic sqrt-split is guarded at coeff=0
# ================================================================
# The biharmonic coefficient enters as ``sqrt(coeff)`` per Laplacian
# pass; the sqrt VJP is inf at coeff=0, so a plain ``coeff ** 0.5``
# turns ``jax.grad`` w.r.t. nu4 into NaN exactly at nu4=0 even though
# the forward value is finite. ``_biharmonic_root`` guards both sqrt
# branches: finite (pinned to the measure-zero subgradient 0) at 0,
# and the true derivative elsewhere.
def _biharmonic_grad_loss(nu4, n_steps=10):
    """Build a grad-ready loss over a toy biharmonic-friction run.

    Returns ``(loss, coeff)``: ``loss(theta)`` splices ``theta`` in
    for the friction coefficient leaf and time-steps the pure kernel
    (``_chunk_body``) for ``n_steps``, returning the sum of squares of
    the final state (a smooth scalar objective).
    """
    model = make_model(BiharmonicFriction(nu4))
    x, z = coords()
    model.set_fields(u=np.sin(2 * np.pi * x), v=np.cos(2 * np.pi * z))
    record = model._artifacts.record
    carry = model._carry
    stepper = model._stepper
    leaf = next(m for m in carry.modules
                if isinstance(m, BiharmonicFriction)).nu
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    idx = next(i for i, lf in enumerate(leaves) if lf is leaf)

    def loss(theta):
        packed = list(leaves)
        packed[idx] = theta
        c = jax.tree_util.tree_unflatten(treedef, packed)
        final = _chunk_body(record, n_steps, c, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    return loss, jnp.asarray(leaf, dtype=jnp.float64)


def test_biharmonic_grad_is_finite_zero_at_nu4_zero():
    # exactly at nu4=0 the double-where pins the (measure-zero)
    # subgradient to 0 -- finite, not the NaN of the unguarded sqrt
    loss, _ = _biharmonic_grad_loss(0.0)
    g = float(jax.grad(loss)(jnp.asarray(0.0, dtype=jnp.float64)))
    assert np.isfinite(g)
    assert g == 0.0


def test_biharmonic_grad_matches_central_fd_away_from_zero():
    nu4 = 1e-4
    loss, x0 = _biharmonic_grad_loss(nu4)
    g = float(jax.grad(loss)(x0))
    assert np.isfinite(g)
    eps = 1e-4
    fd = float((loss(x0 * (1 + eps)) - loss(x0 * (1 - eps)))
               / (2 * x0 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-4)
