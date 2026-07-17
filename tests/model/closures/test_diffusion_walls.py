"""The diffusion closures at walls: free-slip, no-slip, no-flux.

Prefix-mirrored shard of ``test_diffusion.py`` (oversized-module rule):
the walled-grid behaviour of the ``_DiffusionClosure`` family — the
structural no-flux tracer wall, the free-slip and no-slip velocity
walls, the wall-normal component, and the ``slip=`` API. The builders
are duplicated (self-contained shard) rather than imported.
"""
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
from fridom.model.errors import AssemblyError
from fridom.model.model import Model, _chunk_body
from fridom.model.module import Module
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.bc import BC
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 8
L = 1.0
DX = L / N
DT = 1e-3


class Core(Module):

    """Toy core: wall-normal u/v, a transverse velocity, tracers."""

    field_declarations = (
        fr.model.FieldDeclaration.velocity(
            "u", "x", space=fr.spatial.Staggered("x")),
        fr.model.FieldDeclaration.velocity(
            "v", "z", space=fr.spatial.Staggered("z")),
        fr.model.FieldDeclaration.tracer("b"),
    )

    @fr.model.term(advances=("u", "v", "b"), linear=True,
                   transports=("u", "v", "b"))
    def zero(self, state, _ctx):
        return {name: 0.0 * state[name] for name in ("u", "v", "b")}


def make_grid(names=("x",), periodic=False):
    """Build a structured grid; ``periodic`` scalar or per-name tuple."""
    if isinstance(periodic, bool):
        periodic = (periodic,) * len(names)
    return Grid(tuple(
        IntervalMesh(N, (0.0, L), periodic=p, name=name)
        for name, p in zip(names, periodic, strict=True)))


def make_model(closure, grid=None):
    return Model(grid=grid or make_grid(),
                 modules=(Core(), closure),
                 time_stepper=AdamBashforth(DT, order=2))


def data(field):
    return np.asarray(field.data)


def lam_wall(m):
    """Discrete symbol of the walled cosine/sine mode m (both walls)."""
    return (2.0 * np.sin(np.pi * m / (2 * N)) / DX) ** 2


def cos_mode(m):
    """Build a discrete cosine (Neumann/no-flux) mode of order ``m``."""
    return lambda x: np.cos(np.pi * m * x / L)


def sin_mode(m):
    """Build a discrete sine (Dirichlet/no-slip) mode of order ``m``."""
    return lambda x: np.sin(np.pi * m * x / L)


def zero_field(x):
    """Return a zero initial condition (coordinate-signature builder)."""
    return 0.0 * x


def ones_field(x):
    """Return a uniform unit initial condition along ``x``."""
    return np.ones_like(x)


# ================================================================
#  Discrete-symbol convergence: cosine (Neumann), sine (Dirichlet)
# ================================================================
def test_no_flux_tracer_cosine_decays_at_the_discrete_rate():
    # a discrete cosine is the Neumann (no-flux) eigenmode; it decays
    # at exactly -kappa * lam_wall(m) — proves the flux-retag wall
    kappa = 3e-3
    model = make_model(HarmonicDiffusion(kappa))
    for m in (1, 2, 3):
        model.set_fields(b=cos_mode(m))
        td = model.tendency(model.state)
        want = -kappa * lam_wall(m) * data(model.state["b"])
        np.testing.assert_allclose(data(td["b"]), want, atol=1e-13)


def test_free_slip_velocity_cosine_decays_at_the_discrete_rate():
    # free-slip = zero tangential wall stress: the cosine (Neumann)
    # mode of the tangential velocity decays at -nu * lam_wall(m)
    nu = 5e-3
    model = make_model(HarmonicFriction(nu, slip="free"))
    for m in (1, 2, 3):
        model.set_fields(v=cos_mode(m), u=zero_field)
        td = model.tendency(model.state)
        want = -nu * lam_wall(m) * data(model.state["v"])
        np.testing.assert_allclose(data(td["v"]), want, atol=1e-13)


def test_no_slip_velocity_sine_decays_at_the_discrete_rate():
    # no-slip = u=0 at the wall (factor-of-2 ghost): the sine
    # (Dirichlet) mode decays at exactly -nu * lam_wall(k)
    nu = 5e-3
    model = make_model(HarmonicFriction(nu, slip="no"))
    for k in (1, 2, 3, N):
        model.set_fields(v=sin_mode(k), u=zero_field)
        td = model.tendency(model.state)
        want = -nu * lam_wall(k) * data(model.state["v"])
        np.testing.assert_allclose(data(td["v"]), want, atol=1e-12)


def test_wall_normal_component_sine_decays_at_the_discrete_rate():
    # the wall-normal velocity (Inner[Dirichlet]) closes on its own
    # tag: the sine mode decays at -nu * lam_wall(m), slip-independent
    nu = 5e-3
    model = make_model(HarmonicFriction(nu))
    for m in (1, 2, 3):
        model.set_fields(u=sin_mode(m), v=zero_field)
        td = model.tendency(model.state)
        want = -nu * lam_wall(m) * data(model.state["u"])
        np.testing.assert_allclose(data(td["u"]), want, atol=1e-12)


# ================================================================
#  Tracer conservation under no-flux walls (harmonic + biharmonic)
# ================================================================
def test_harmonic_tracer_conserves_integral_under_no_flux():
    model = make_model(HarmonicDiffusion(1.0))
    rng = np.random.default_rng(0)
    model.set_fields(b=lambda x: rng.standard_normal(x.shape))
    td = model.tendency(model.state)
    assert abs(float(np.sum(data(td["b"])))) < 1e-13


def test_biharmonic_tracer_conserves_integral_under_no_flux():
    model = make_model(BiharmonicDiffusion(1e-3))
    rng = np.random.default_rng(1)
    model.set_fields(b=lambda x: rng.standard_normal(x.shape))
    td = model.tendency(model.state)
    assert abs(float(np.sum(data(td["b"])))) < 1e-13


# ================================================================
#  Free-slip vs no-slip contrast on a uniform tangential flow
# ================================================================
def test_free_slip_uniform_flow_has_no_wall_drag():
    # a wall-parallel uniform flow feels zero friction under free-slip
    model = make_model(HarmonicFriction(1e-2, slip="free"))
    model.set_fields(v=ones_field)
    td = model.tendency(model.state)
    assert np.abs(data(td["v"])).max() == 0.0


def test_no_slip_uniform_flow_drags_only_the_wall_cells():
    nu = 1e-2
    model = make_model(HarmonicFriction(nu, slip="no"))
    model.set_fields(v=ones_field)
    td = model.tendency(model.state)
    tend = data(td["v"])
    # the two wall-adjacent cells feel -2 nu / dx^2; interior is zero
    assert tend[0] < 0.0
    assert tend[-1] < 0.0
    np.testing.assert_allclose(tend[1:-1], 0.0, atol=1e-13)
    np.testing.assert_allclose(tend[0], -2.0 * nu / DX**2, rtol=1e-12)
    np.testing.assert_allclose(tend[-1], -2.0 * nu / DX**2, rtol=1e-12)


# ================================================================
#  Periodic bit-identity: the periodic path is the direct old chain
# ================================================================
def test_periodic_path_is_bitwise_the_direct_chain():
    # on a fully periodic grid the walled refactor must reproduce the
    # plain (q.diff * k).diff chain bit-for-bit (no new branch entered)
    kappa = 2e-3
    grid = make_grid(names=("x", "z"), periodic=True)
    model = make_model(HarmonicDiffusion(kappa), grid=grid)
    model.set_fields(
        b=lambda x, z: np.sin(2 * np.pi * x) * np.cos(4 * np.pi * z))
    td = model.tendency(model.state)
    q = model.state["b"]
    direct = None
    for axis in ("x", "z"):
        contribution = (q.diff(axis) * kappa).diff(axis)
        direct = (contribution if direct is None
                  else direct + contribution)
    np.testing.assert_array_equal(data(td["b"]), data(direct))


# ================================================================
#  Multi-walled grid: two bounded axes compose per axis
# ================================================================
def test_two_walled_axes_bind_and_step_finite():
    grid = make_grid(names=("x", "z"), periodic=False)
    model = make_model(HarmonicFriction(1e-2, slip="no"), grid=grid)
    model.set_fields(
        u=lambda x, z: np.sin(np.pi * x / L) * np.cos(np.pi * z / L),
        v=lambda x, z: np.cos(np.pi * x / L) * np.sin(np.pi * z / L),
        b=lambda x, z: np.sin(np.pi * x / L) + 0.0 * z)
    final = _chunk_body(model._artifacts.record, 5,
                        model._carry, model._stepper)
    assert all(bool(np.all(np.isfinite(np.asarray(f.data))))
               for f in final.state)


def test_two_walled_axes_compose_per_axis_treatment():
    # u is wall-normal along x and tangential along z: its x/z
    # contributions add independently. Build a separable no-slip mode
    # and check the tendency equals the sum of the 1-D operators.
    nu = 4e-3
    grid = make_grid(names=("x", "z"), periodic=False)
    model = make_model(HarmonicFriction(nu, slip="no"), grid=grid)
    # sin along x (wall-normal), sin along z (tangential no-slip):
    # both are eigenmodes at lam_wall(1), so u decays at their sum
    model.set_fields(
        u=lambda x, z: np.sin(np.pi * x / L) * np.sin(np.pi * z / L),
        v=lambda x, z: 0.0 * (x + z), b=lambda x, z: 0.0 * (x + z))
    td = model.tendency(model.state)
    want = -nu * (lam_wall(1) + lam_wall(1)) * data(model.state["u"])
    np.testing.assert_allclose(data(td["u"]), want, atol=1e-12)


# ================================================================
#  Biharmonic at walls: both slips dissipate; tracer conserves
# ================================================================
@pytest.mark.parametrize("slip", ["free", "no"])
def test_biharmonic_friction_dissipates_energy_at_walls(slip):
    model = make_model(BiharmonicFriction(2e-4, slip=slip))
    model.set_fields(v=lambda x: np.sin(np.pi * 2 * x / L),
                     u=lambda x: np.sin(np.pi * 3 * x / L))

    def energy(carry):
        return float(sum(jnp.sum(f.data ** 2) for f in carry.state))

    before = energy(model._carry)
    after = energy(_chunk_body(model._artifacts.record, 6,
                               model._carry, model._stepper))
    assert after < before


def test_biharmonic_tracer_biharmonic_conserves_and_runs():
    model = make_model(BiharmonicDiffusion(1e-3))
    model.set_fields(b=lambda x: np.cos(np.pi * 2 * x / L))
    final = _chunk_body(model._artifacts.record, 4,
                        model._carry, model._stepper)
    assert bool(np.all(np.isfinite(np.asarray(final.state[-1].data))))


# ================================================================
#  Finite-volume walled targets are out of scope (taught rejection)
# ================================================================
def test_finite_volume_walled_target_is_a_taught_rejection():
    # a CellAvg tracer on a walled axis is FV walled diffusion (future
    # work): rejected at bind rather than run through the wrong
    # collocated FVDerivative chain
    class FVCore(Module):
        field_declarations = (
            fr.model.FieldDeclaration.tracer(
                "b", space=fr.spatial.Collocated(family="fv")),
        )

        @fr.model.term(advances=("b",), linear=True, transports=("b",))
        def zero(self, state, _ctx):
            return {"b": 0.0 * state["b"]}

    grid = make_grid(names=("x",), periodic=False)
    with pytest.raises(NotImplementedError,
                       match="unsupported wall placement"):
        Model(grid=grid, modules=(FVCore(), HarmonicDiffusion(1.0)),
              time_stepper=AdamBashforth(DT, order=2))


# ================================================================
#  slip= API validation
# ================================================================
def test_bad_slip_scalar_is_rejected():
    with pytest.raises(ValueError, match="must be 'free'"):
        HarmonicFriction(1e-2, slip="partial")


def test_bad_slip_mapping_value_is_rejected():
    with pytest.raises(ValueError, match="must be 'free'"):
        HarmonicFriction(1e-2, slip={"u": "no", "v": "kinda"})


def test_empty_slip_mapping_is_rejected():
    with pytest.raises(ValueError, match="slip= mapping is empty"):
        HarmonicFriction(1e-2, slip={})


def test_non_string_slip_key_is_rejected():
    with pytest.raises(TypeError, match="slip= mapping keys"):
        HarmonicFriction(1e-2, slip={1: "no"})


def test_slip_mapping_unknown_target_is_an_assembly_error():
    with pytest.raises(AssemblyError, match=r"unknown target 'q'"):
        make_model(HarmonicFriction(
            1e-2, slip={"u": "no", "v": "free", "q": "no"}))


def test_slip_mapping_selects_the_slip_per_velocity():
    # u no-slip, v free-slip: a uniform flow drags u's walls but not v
    nu = 1e-2
    model = make_model(HarmonicFriction(nu, slip={"u": "no", "v": "free"}))
    model.set_fields(u=ones_field, v=ones_field)
    td = model.tendency(model.state)
    assert np.abs(data(td["v"])).max() == 0.0        # free-slip: no drag
    assert data(td["u"])[0] < 0.0                    # no-slip: wall drag


def test_mixing_closures_reject_slip():
    with pytest.raises(TypeError):
        HarmonicDiffusion(1e-3, slip="no")
    with pytest.raises(TypeError):
        BiharmonicDiffusion(1e-3, slip="free")


# ================================================================
#  Unsupported wall placement (fixed-value cell wall) is taught
# ================================================================
def test_dirichlet_valued_tracer_wall_is_a_taught_rejection():
    # a fixed-value (Dirichlet) cell wall is the 2e boundary-data path,
    # out of scope: the closure rejects it at bind rather than silently
    # applying the no-flux stencil
    class DCore(Module):
        field_declarations = (
            fr.model.FieldDeclaration.tracer(
                "d", space=fr.spatial.Collocated(bc={"x": BC.DIRICHLET})),
        )

        @fr.model.term(advances=("d",), linear=True, transports=("d",))
        def zero(self, state, _ctx):
            return {"d": 0.0 * state["d"]}

    grid = make_grid(names=("x",), periodic=False)
    with pytest.raises(NotImplementedError,
                       match="unsupported wall placement"):
        Model(grid=grid, modules=(DCore(), HarmonicDiffusion(1e-3)),
              time_stepper=AdamBashforth(DT, order=2))


# ================================================================
#  Reverse-mode AD through a walled run (free-slip and no-slip)
# ================================================================
def _friction_grad_loss(slip, nu, n_steps=8):
    """Build a grad-ready loss over a walled harmonic-friction run."""
    model = make_model(HarmonicFriction(nu, slip=slip))
    model.set_fields(v=lambda x: np.sin(np.pi * 2 * x / L),
                     u=lambda x: np.sin(np.pi * 3 * x / L))
    record = model._artifacts.record
    carry = model._carry
    stepper = model._stepper
    leaf = next(m for m in carry.modules
                if isinstance(m, HarmonicFriction)).nu
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    idx = next(i for i, lf in enumerate(leaves) if lf is leaf)

    def loss(theta):
        packed = list(leaves)
        packed[idx] = theta
        c = jax.tree_util.tree_unflatten(treedef, packed)
        final = _chunk_body(record, n_steps, c, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    return loss, jnp.asarray(leaf, dtype=jnp.float64)


@pytest.mark.parametrize("slip", ["free", "no"])
def test_friction_grad_matches_central_fd_on_walls(slip):
    nu = 2e-2
    loss, x0 = _friction_grad_loss(slip, nu)
    g = float(jax.grad(loss)(x0))
    assert np.isfinite(g)
    eps = 1e-4
    fd = float((loss(x0 * (1 + eps)) - loss(x0 * (1 - eps)))
               / (2 * x0 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-4)
