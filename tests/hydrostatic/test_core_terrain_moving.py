r"""The hydrostatic terrain column under a MOVING mapping parameter.

Prefix-mirrored shard of ``hy.modules.core`` (with the free-surface
and terrain seams it drives) covering stage C4 on the hydrostatic
package: every metric read of the step path threads the CURRENT
mapping-parameter fields
(:func:`~fridom.model.modules.moving_geometry.mapping_params`) through
``grid.metric(..., params=)`` and the ``with_params`` reduction seam,
so a moving column ``zp = z\,H(x, y, t)`` — the geometry a z*
free surface installs — is visible to ``w``, ``p_hyd``, the
slope-corrected pressure gradient and the barotropic transport
divergence. The gates:

1. **frozen** (``RATE = 0``): at the mapping's declared default the
   params-threaded derivation is bit-identical to the static one on
   every metric and DIAGNOSE stage, and the dynamic pipeline
   (``MovingGeometry`` state fields, ``params=`` threading, the ALE
   term) reproduces the static terrain run bitwise step by step
   (to last-bit XLA scan reassociation over a chunked advance);
2. **the DIAGNOSE stages read the moved column**: with a uniform ``b``
   the hydrostatic pressure is analytically ``p_hyd = b\,z_p`` with
   ``z_p = z\,H``, so the committed ``p_hyd`` must track the state's
   own moved ``H`` (and differ from the static-``H`` prediction by the
   motion). The scaling structure of ``zp = z H`` makes it sharper
   still: every metric is linear in ``H``, so scaling the parameter
   field scales ``p_hyd`` and the barotropic transport divergence
   ``T^*`` exactly;
3. **constancy**: a uniform ``b`` stays uniform under motion (the
   free-stream / GCL property of the ALE correction);
4. **autodiff**: ``jax.grad`` through ``Model.propagator`` on the
   moving terrain path matches a central finite difference.

Self-contained builders (AGENTS oversized-module rule).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.hydrostatic.modules.barotropic_pressure import (
    BarotropicPressureSolver,
)
from fridom.model.modules.advection import CenteredAdvection
from fridom.model.modules.moving_geometry import (
    MeshVelocityCorrection,
    MovingGeometry,
)
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.coordinate_mapping import CoordinateMapping

IM = fr.spatial.meshes.IntervalMesh
N, NZ = 8, 4
DT = 2e-3
GRAVITY = 3.0
#: relative depth growth per unit time; 10 steps of DT grow H by 20%
RATE = 10.0
STEPS = 10
#: the corrected prognostics: ``ps`` has no column factor, and
#: correcting it is a separate (geometry-branch) decision — pin the
#: selection here so this shard does not depend on that default.
CORRECTED = ("u", "v", "b")


def _depth(x, y):
    """Smooth doubly-periodic column depth ``H0(x, y)`` (20% slope)."""
    return 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def _schedule(rate):
    """Return ``H(x, y, t) = (1 + rate t) H0(x, y)`` (a pure scaling)."""
    def depth_at(x, y, t):
        return (1.0 + rate * t) * _depth(x, y)
    return depth_at


def _grid():
    """Sigma grid ``zp = z H(x, y)``, base ``z`` in ``[-1, 0]``."""
    return fr.spatial.Grid(
        (IM(N, (0.0, 1.0), periodic=True, name="x"),
         IM(N, (0.0, 1.0), periodic=True, name="y"),
         IM(NZ, (-1.0, 0.0), periodic=False, name="z")),
        mapping=CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                  params={"H": _depth}))


def _model(*, rate=None, advection=True, n2=0.0):
    """Terrain model; ``rate=None`` is the STATIC (no-geometry) run."""
    extra = ()
    if rate is not None:
        extra = (MovingGeometry({"H": _schedule(rate)}),
                 MeshVelocityCorrection(CORRECTED))
    return hy.Model(
        grid=_grid(),
        core=hy.Core(gravity=GRAVITY),
        time_stepper=AdamBashforth(DT, order=3),
        buoyancy=hy.ConstantStratification(n2=n2),
        free_surface=hy.ExplicitFreeSurface(),
        advection=CenteredAdvection() if advection else None,
        modules_extra=extra)


def _seed(model, *, uniform_b=None):
    """Set a small smooth (or uniform-``b``) initial condition."""
    rng = np.random.default_rng(7)
    fields = {k: 0.02 * rng.standard_normal(model.state[k].data.shape)
              for k in ("u", "v")}
    fields["b"] = (np.full(model.state["b"].data.shape, uniform_b)
                   if uniform_b is not None
                   else 0.02 * rng.standard_normal(
                       model.state["b"].data.shape))
    model.set_fields(**fields)
    return model


def _reproduces_static(got, want, name):
    """Frozen-motion == static reproduction, backend-aware.

    The static model and the frozen-motion model compile two different
    HLO programs (the moving pipeline carries the ``H`` / ``H_dot``
    carry leaves and threads ``params=`` through every metric
    derivation). Per **step** the two lower identically on cpu, so the
    reproduction there is BITWISE — the valuable pin. On gpu XLA
    autotuning is free to pick different kernels for the two
    physically equivalent programs, so the contract there is a
    tolerance (the ``tests/validation/test_moving_geometry.py``
    convention).
    """
    got, want = np.asarray(got), np.asarray(want)
    if jax.default_backend() == "cpu":
        assert np.array_equal(got, want), name
    else:
        assert np.allclose(got, want, rtol=0.0, atol=1e-14), name


def _nodes(model, name, space_of):
    """Return the evaluation nodes of ``name`` on ``space_of``'s space."""
    field = model.state[space_of]
    return np.asarray(field.grid.evaluation_nodes(
        field.function_space.bare, name).data)


# ================================================================
#  Gate 1: frozen motion reproduces the static terrain run bitwise
# ================================================================
def test_frozen_parameter_field_derives_the_static_metrics_bitwise():
    # the root of the frozen gate, isolated from the stepper: at the
    # frozen default the params= derivation (H read from the state
    # FIELD) and the static derivation (H read from the mapping's
    # declared CALLABLE) produce bit-identical metrics on every
    # staggering, so every threaded consumer -- the DIAGNOSE stages and
    # the slope-corrected pressure gradient -- reproduces exactly.
    model = _seed(_model(rate=0.0))
    core = model.module(hy.Core)
    state = model.state
    grid = state["u"].grid
    frozen_params = {"H": state["H"]}
    for field, metric in (("u", "dzp_dz"), ("v", "dzp_dz"),
                          ("p_hyd", "dzp_dz"), ("p_hyd", "dzp_dx"),
                          ("p_hyd", "dzp_dy"), ("w", "dzp_dx")):
        space = state[field].function_space.bare
        static = np.asarray(grid.metric(space, metric).data)
        threaded = np.asarray(
            grid.metric(space, metric, params=frozen_params).data)
        assert np.array_equal(threaded, static), f"{field}:{metric}"
    plain = {"u": state["u"], "v": state["v"], "b": state["b"]}
    moving = dict(plain, **frozen_params)
    for stage in ("_diagnose_w", "_diagnose_p_hyd"):
        out = getattr(core, stage)
        (key,) = out(plain, None)
        assert np.array_equal(
            np.asarray(out(moving, None)[key].data),
            np.asarray(out(plain, None)[key].data)), stage
    assert np.array_equal(
        np.asarray(core._slope_gradient(
            state["p_hyd"], "x", state["u"], frozen_params).data),
        np.asarray(core._slope_gradient(
            state["p_hyd"], "x", state["u"]).data))


def test_frozen_motion_reproduces_the_static_terrain_run_bitwise():
    # RATE = 0 freezes the schedule at the mapping's declared default
    # (H_dot == 0 exactly through the jvp), so the whole dynamic
    # pipeline — the MovingGeometry state fields, params= threading
    # through every hydrostatic metric read and the with_params
    # reduction seam of p_hyd, and an exact-zero ALE tendency — must
    # reproduce the static terrain assembly. Stepped one step at a
    # time (one compiled step program per model), the reproduction is
    # BITWISE on every field at every step.
    static = _seed(_model())
    frozen = _seed(_model(rate=0.0))
    for _ in range(STEPS):
        static.advance(1)
        frozen.advance(1)
        for name in ("u", "v", "w", "b", "p_hyd", "ps"):
            _reproduces_static(frozen.state[name].data,
                               static.state[name].data, name)


def test_frozen_motion_matches_the_static_chunked_run():
    # the same reproduction over a CHUNKED advance (one jitted scan of
    # STEPS steps). The step semantics are bit-identical (the test
    # above), but the two models scan different carry treedefs — the
    # moving one carries the H / H_dot leaves — so XLA is free to fuse
    # and reassociate the scan body differently. That shows up as
    # last-bit drift, not physics: measured worst 5.6e-17 absolute
    # (1.8e-16 relative) over 10 steps, against a gate with ~20x
    # headroom.
    static = _seed(_model())
    frozen = _seed(_model(rate=0.0))
    for model in (static, frozen):
        model.advance(STEPS)
    for name in ("u", "v", "w", "b", "p_hyd", "ps"):
        assert np.allclose(np.asarray(frozen.state[name].data),
                           np.asarray(static.state[name].data),
                           rtol=1e-13, atol=1e-15), name


def test_frozen_geometry_state_holds_the_static_default():
    # the frozen H state field IS the mapping's static default (this is
    # what makes the bitwise gate above meaningful), and H_dot is an
    # exact zero
    model = _seed(_model(rate=0.0))
    model.advance(3)
    h_state = np.asarray(model.state["H"].data)
    x = _nodes(model, "x", "H")
    y = _nodes(model, "y", "H")
    assert np.allclose(h_state, np.asarray(_depth(x, y)), atol=1e-14)
    assert float(np.abs(np.asarray(model.state["H_dot"].data)).max()) == 0.0


# ================================================================
#  Gate 2: the DIAGNOSE stages read the MOVED column
# ================================================================
def test_p_hyd_tracks_the_moved_column_depth():
    # uniform b: the hydrostatic pressure is analytically
    #   p_hyd(z) = -int_z^0 b J dz' = b z_p = b z H
    # exactly (the center target midpoints an exact face accumulation
    # and z_p is linear in z). SELF_UPDATE (S1) writes H before
    # DIAGNOSE (S1') reads it in the same substage, so the committed
    # p_hyd and H belong to one geometry: p_hyd must equal b z H_state.
    b0 = 0.05
    model = _seed(_model(rate=RATE, advection=True), uniform_b=b0)
    model.advance(STEPS)
    h_state = np.asarray(model.state["H"].data)
    z = _nodes(model, "z", "p_hyd")
    p_hyd = np.asarray(model.state["p_hyd"].data)
    assert np.allclose(p_hyd, b0 * z * h_state, rtol=1e-12, atol=1e-14)
    # the column really moved, and the STATIC-H prediction is wrong by
    # that motion (so this is a genuine dynamic-geometry gate)
    x, y = _nodes(model, "x", "H"), _nodes(model, "y", "H")
    h_static = np.asarray(_depth(x, y))
    assert np.abs(h_state / h_static - 1.0).max() > 0.1
    assert not np.allclose(p_hyd, b0 * z * h_static,
                           rtol=1e-3, atol=1e-6)


def test_p_hyd_scales_exactly_with_the_parameter_field():
    # every metric of zp = z H is linear in H, so scaling the parameter
    # field must scale the J-weighted running integral EXACTLY -- the
    # sharpest available witness that the CumulativeIntegral
    # with_params seam reaches grid.metric (params=None is the mapping
    # default, i.e. the static path)
    model = _seed(_model(rate=RATE))
    core = model.module(hy.Core)
    state = model.state
    scale = 1.7
    base = {"u": state["u"], "b": state["b"], "H": state["H"]}
    scaled = dict(base, H=state["H"] * scale)
    p_base = core._diagnose_p_hyd(base, None)["p_hyd"]
    p_scaled = core._diagnose_p_hyd(scaled, None)["p_hyd"]
    assert np.allclose(np.asarray(p_scaled.data),
                       scale * np.asarray(p_base.data),
                       rtol=1e-13, atol=1e-15)
    # no parameter field in the state -> the static declaration default
    p_static = core._diagnose_p_hyd(
        {"u": state["u"], "b": state["b"]}, None)["p_hyd"]
    assert np.allclose(np.asarray(p_static.data),
                       np.asarray(p_base.data), atol=1e-14)


def test_transport_divergence_scales_exactly_with_the_parameter_field():
    # the same linearity witness on the free surface's J-weighted
    # transport divergence T* (the barotropic gravity term's operand)
    model = _seed(_model(rate=RATE))
    fs = model.module(hy.ExplicitFreeSurface)
    state = model.state
    scale = 1.7
    base = {"u": state["u"], "v": state["v"], "H": state["H"]}
    d_base = np.asarray(fs._transport_div(base).data)
    d_scaled = np.asarray(
        fs._transport_div(dict(base, H=state["H"] * scale)).data)
    assert float(np.abs(d_base).max()) > 0.0
    assert np.allclose(d_scaled, scale * d_base, rtol=1e-13, atol=1e-15)
    d_static = np.asarray(fs._transport_div(
        {"u": state["u"], "v": state["v"]}).data)
    assert np.allclose(d_static, d_base, atol=1e-14)


def test_diagnosed_w_scales_exactly_with_the_parameter_field():
    # the contravariant flux Jomega and the slope terms u Z_x + v Z_y
    # are both linear in H (terrain.slope_velocity_on_w threads params
    # too), so the whole diagnosed physical w scales with the column
    model = _seed(_model(rate=RATE))
    core = model.module(hy.Core)
    state = model.state
    scale = 1.7
    base = {"u": state["u"], "v": state["v"], "H": state["H"]}
    w_base = np.asarray(core._diagnose_w(base, None)["w"].data)
    w_scaled = np.asarray(
        core._diagnose_w(dict(base, H=state["H"] * scale), None)["w"].data)
    assert float(np.abs(w_base).max()) > 0.0
    assert np.allclose(w_scaled, scale * w_base, rtol=1e-13, atol=1e-15)


# ================================================================
#  The barotropic depth weights (implicit + split-explicit operators)
# ================================================================
def test_physical_depth_reads_the_moved_column():
    # H_a = int J dz on the velocity face -- the split-explicit
    # subcycle's per-face depth and the transport commit
    model = _seed(_model(rate=RATE))
    fs = model.module(hy.ExplicitFreeSurface)
    u, h = model.state["u"], model.state["H"]
    static = np.asarray(fs._physical_depth(u).data)
    same = np.asarray(fs._physical_depth(u, {"H": h}).data)
    moved = np.asarray(fs._physical_depth(u, {"H": h * 1.7}).data)
    assert np.allclose(same, static, atol=1e-14)
    assert np.allclose(moved, 1.7 * static, rtol=1e-13)


def test_wet_depth_mean_uses_the_current_volume_element():
    # the physical depth mean int(J q dz)/int(J dz) built through the
    # with_params reduction seam. On a column map J is z-uniform, so
    # the mean is invariant under the motion -- the value the seeded
    # (static) .mean() verb also returns; the seam must reproduce it
    # rather than silently weight by a stale J.
    model = _seed(_model(rate=RATE))
    fs = model.module(hy.ExplicitFreeSurface)
    u, h = model.state["u"], model.state["H"]
    static = np.asarray(fs._wet_depth_mean(u).data)
    same = np.asarray(fs._wet_depth_mean(u, {"H": h}).data)
    moved = np.asarray(fs._wet_depth_mean(u, {"H": h * 1.7}).data)
    assert np.allclose(same, static, atol=1e-14)
    assert np.allclose(moved, static, atol=1e-14)


def test_barotropic_solver_depths_read_the_moved_column():
    # the implicit variant's variable-coefficient operator: the face
    # depths H_a and the mean-depth preconditioner coefficient
    model = _seed(_model(rate=RATE))
    state = model.state
    grid = state["ps"].grid
    space = state["ps"].function_space.bare
    common = {"epsilon": 1.0, "iterations": 4, "tolerance": None}
    static = BarotropicPressureSolver(
        grid, space, ("zp", "z"), "z", **common)
    moved = BarotropicPressureSolver(
        grid, space, ("zp", "z"), "z",
        params={"H": state["H"] * 1.7}, **common)
    assert np.allclose(np.asarray(moved._face_depth("x").data),
                       1.7 * np.asarray(static._face_depth("x").data),
                       rtol=1e-13)
    assert float(moved._mean_depth()) == pytest.approx(
        1.7 * float(static._mean_depth()), rel=1e-13)


@pytest.mark.parametrize("free_surface", [
    pytest.param(hy.ImplicitFreeSurface(pressure_iterations=6),
                 id="implicit"),
    pytest.param(hy.SplitExplicitFreeSurface(substeps=4),
                 id="split-explicit"),
])
def test_the_other_free_surface_variants_run_on_a_moving_column(
        free_surface):
    # the implicit CONSTRAINT solve and the split-explicit subcycle
    # thread the same parameter fields (the depth weights above); a
    # short run pins that they assemble, step and stay finite while
    # the column grows. Both carry a documented O(dt) GCL residual
    # (plan section 2), so nothing sharper is asserted here.
    model = hy.Model(
        grid=_grid(),
        core=hy.Core(gravity=GRAVITY),
        time_stepper=AdamBashforth(DT, order=3),
        buoyancy=hy.ConstantStratification(n2=0.0),
        free_surface=free_surface,
        advection=CenteredAdvection(),
        modules_extra=(MovingGeometry({"H": _schedule(RATE)}),
                       MeshVelocityCorrection(CORRECTED)))
    _seed(model, uniform_b=0.05)
    model.advance(4)
    for name in ("u", "v", "b", "ps"):
        assert bool(np.all(np.isfinite(
            np.asarray(model.state[name].data)))), name
    assert float(np.abs(np.asarray(
        model.state["H_dot"].data)).max()) > 0.5


# ================================================================
#  Gate 3: constancy of a uniform tracer under motion
# ================================================================
def test_a_uniform_buoyancy_stays_uniform_under_motion():
    # the free-stream / GCL property: while the column grows by 20%
    # the ALE correction and the advective tendency of a CONSTANT
    # field are both exact zeros, so b must not develop structure
    b0 = 0.05
    model = _seed(_model(rate=RATE, advection=True), uniform_b=b0)
    model.advance(STEPS)
    b = np.asarray(model.state["b"].data)
    assert float(b.max() - b.min()) < 1e-13
    assert float(np.abs(b - b0).max()) < 1e-13
    # the geometry genuinely moved during those steps
    assert float(np.abs(np.asarray(
        model.state["H_dot"].data)).max()) > 0.5


# ================================================================
#  Gate 4: autodiff through the moving terrain step path
# ================================================================
@pytest.mark.single_device
def test_grad_through_a_moving_terrain_run_matches_fd():
    # the differentiability policy on the params-threaded step path:
    # the dynamic metrics enter w, p_hyd, the slope-corrected pressure
    # gradient, the transport divergence and the ALE correction, so a
    # masked singularity anywhere there would NaN this gradient
    model = _seed(_model(rate=RATE))
    run = model.propagator(wrt=("b",), steps=8)
    b0 = model._carry.state["b"].storage

    def loss(field):
        return sum(jnp.sum(f.data ** 2) for f in run((field,)).state)

    grad = np.asarray(jax.grad(loss)(b0))
    assert bool(np.all(np.isfinite(grad)))
    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(b0.shape), dtype=b0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(b0 + eps * direction))
          - float(loss(b0 - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)
