"""
jit + scan compatibility of the field layer (task 1.7).

Fields are pytrees: the 2-D advection-diffusion tendency steps
inside ``jax.lax.scan`` over ``ScalarField`` and ``VectorField``
carries, the whole loop jits (the compile counter proves a single
trace, zero retraces on reuse), and the scanned result matches the
plain Python loop bitwise.
"""
import jax
import jax.numpy as jnp

import fridom.framework2 as fr

N = 16
N_STEPS = 5


def build():
    # note (finding): the ScalarField carry must be *unnamed* —
    # binary arithmetic resets metadata to the default ("a new
    # quantity"), and metadata is static pytree aux data, so
    # `u + dt * du` on a named field changes the treedef and
    # lax.scan rejects the carry. VectorField preserves component
    # metadata (``_keep_metadata``) exactly so its carries survive.
    mx = fr.grid.meshes.IntervalMesh(N, (0.0, 1.0), name="x")
    my = fr.grid.meshes.IntervalMesh(N, (0.0, 1.0), name="y")
    grid = fr.grid.Grid((mx, my))
    u0 = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x)
        * jnp.cos(2 * jnp.pi * y))
    q0 = grid.create_field(
        init=lambda x, y: jnp.cos(2 * jnp.pi * (x + y)))
    center = u0.function_space.bare
    cx, nu = 1.0, 0.02
    dt = 0.2 * mx.dx

    def scalar_tendency(u):
        return (-cx * u.diff("x").to(center)
                + nu * (u.diff("x").diff("x")
                        + u.diff("y").diff("y")))

    return grid, u0, q0, scalar_tendency, dt


def rk4(state, dt, tendency):
    k1 = tendency(state)
    k2 = tendency(state + (dt / 2) * k1)
    k3 = tendency(state + (dt / 2) * k2)
    k4 = tendency(state + dt * k3)
    return state + (dt / 6) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ================================================================
#  ScalarField carry
# ================================================================
def test_scan_over_a_scalar_field_matches_the_python_loop(
        compile_counter):
    _, u0, _, tendency, dt = build()

    def body(u, _):
        return rk4(u, dt, tendency), None

    @jax.jit
    def run(u):
        final, _ = jax.lax.scan(body, u, None, length=N_STEPS)
        return final

    u_scan = run(u0).block_until_ready()
    # the plain Python loop over the identically-jitted step
    step = jax.jit(lambda u: rk4(u, dt, tendency))
    u_loop = u0
    for _ in range(N_STEPS):
        u_loop = step(u_loop)
    assert jnp.array_equal(u_scan.data, u_loop.data)
    assert u_scan.function_space is u_loop.function_space
    # no retraces: re-running the jitted scan on a new carry
    # compiles nothing
    compile_counter.reset()
    u_scan2 = run(u_scan).block_until_ready()
    assert compile_counter.count == 0
    assert not jnp.array_equal(u_scan2.data, u_scan.data)


def test_fully_eager_loop_agrees_to_rounding():
    # the unjitted (op-by-op eager) loop agrees with the scanned
    # run to one ulp: XLA fuses the jitted step differently than
    # the eager per-primitive dispatch (measured max diff 1.1e-16,
    # not bitwise) — recorded as an expected finding
    _, u0, _, tendency, dt = build()

    def body(u, _):
        return rk4(u, dt, tendency), None

    @jax.jit
    def run(u):
        final, _ = jax.lax.scan(body, u, None, length=N_STEPS)
        return final

    u_scan = run(u0)
    u_eager = u0
    for _ in range(N_STEPS):
        u_eager = rk4(u_eager, dt, tendency)
    assert float(jnp.abs(u_scan.data - u_eager.data).max()) < 1e-13


# ================================================================
#  VectorField carry
# ================================================================
def test_scan_over_a_vector_field_matches_the_python_loop(
        compile_counter):
    _, u0, q0, scalar_tendency, dt = build()
    center = u0.function_space.bare
    state0 = fr.grid.VectorField({"u": u0, "q": q0})

    def tendency(state):
        du = scalar_tendency(state["u"])
        # passive tracer advected by the constant flow
        dq = -1.0 * state["q"].diff("x").to(center)
        return fr.grid.VectorField({"u": du, "q": dq})

    def body(state, _):
        return rk4(state, dt, tendency), None

    @jax.jit
    def run(state):
        final, _ = jax.lax.scan(body, state, None, length=N_STEPS)
        return final

    final = run(state0)
    step = jax.jit(lambda s: rk4(s, dt, tendency))
    state = state0
    for _ in range(N_STEPS):
        state = step(state)
    assert jnp.array_equal(final["u"].data, state["u"].data)
    assert jnp.array_equal(final["q"].data, state["q"].data)
    assert final.component_names == ("u", "q")
    # zero retraces on a fresh carry
    compile_counter.reset()
    run(final)["u"].block_until_ready()
    assert compile_counter.count == 0
