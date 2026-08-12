"""The nonhydrostatic ``State`` vocabulary: chart view, wall edges."""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.state import State
from fridom.spatial.bc import BC
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.spaces.nodal import NodeSet

IM = fr.spatial.meshes.IntervalMesh


def _depth(x, y):
    return 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def _mapping():
    return CoordinateMapping(maps={"zp": lambda z, H: z * H},
                             params={"H": _depth})


def _mapped_grid(n=8):
    return fr.spatial.Grid((
        IM(n, (0.0, 1.0), periodic=True, name="x"),
        IM(n, (0.0, 1.0), periodic=True, name="y"),
        IM(n, (-1.0, 0.0), periodic=False, name="z")), mapping=_mapping())


def _flat_grid(n=8):
    return fr.spatial.Grid((
        IM(n, (0.0, 1.0), periodic=True, name="x"),
        IM(n, (0.0, 1.0), periodic=True, name="y"),
        IM(n, (-1.0, 0.0), periodic=False, name="z")))


def _set_random(model, seed=0):
    rng = np.random.default_rng(seed)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        w=rng.standard_normal(model.state["w"].shape),
        b=rng.standard_normal(model.state["b"].shape))
    return model.state


# ================================================================
#  chart["w"] is the contravariant flux w - sum_i Z_i I(u_i)
# ================================================================
def test_chart_w_equals_hand_built_contravariant_flux():
    # on a mapped column the chart-native vertical quantity is the
    # contravariant volume flux J*omega = w - sum_i Z_i I(u_i) -- the
    # same quantity the mapped pressure solver's divergence RHS derives
    # from the stored PHYSICAL w.
    grid = _mapped_grid()
    st = _set_random(
        nh.Model(
            grid=grid,
            core=nh.Core(),
            time_stepper=AdamBashforth(1e-3, order=3),
            buoyancy=nh.ConstantStratification(n2=1.0)))
    assert isinstance(st, State)
    chart_w = st.chart["w"]

    w = st["w"]
    bare = w.function_space.bare
    expect = w
    for comp, axis in zip(("u", "v"), ("x", "y"), strict=True):
        iu = st[comp].to(w)
        zi = grid.metric(bare, f"dzp_d{axis}")
        expect = expect - zi.retag(iu) * iu
    assert np.array_equal(np.asarray(chart_w.data), np.asarray(expect.data))
    assert chart_w.function_space.bare == w.function_space.bare


# ================================================================
#  Flat identity and the read-only surface
# ================================================================
def test_chart_is_identity_on_a_flat_grid():
    st = _set_random(
        nh.Model(
            grid=_flat_grid(),
            core=nh.Core(),
            time_stepper=AdamBashforth(1e-3, order=3),
            buoyancy=nh.ConstantStratification(n2=1.0)))
    assert st.chart["w"] is st["w"]
    assert st.chart["u"] is st["u"]
    assert st.chart.w is st["w"]


def test_chart_velocities_and_read_only():
    st = _set_random(
        nh.Model(
            grid=_flat_grid(),
            core=nh.Core(),
            time_stepper=AdamBashforth(1e-3, order=3),
            buoyancy=nh.ConstantStratification(n2=1.0)))
    u, v, w = st.chart.velocities
    assert u is st["u"]
    assert v is st["v"]
    assert w is st["w"]
    with pytest.raises(TypeError, match="read-only"):
        st.chart["w"] = None
    with pytest.raises(AttributeError, match="read-only"):
        st.chart.w = None


# ================================================================
#  rel_vort_z at a wall — the free-slip edge claim
# ================================================================
def _walled_grid(n=12, *, walled_y=False):
    """Build a grid walled in x (and optionally y)."""
    return fr.spatial.Grid((
        IM(n, (0.0, 1.0), periodic=False, name="x"),
        IM(n, (0.0, 1.0), periodic=not walled_y, name="y"),
        IM(4, (-1.0, 0.0), periodic=False, name="z")))


def _walled_model(grid, family=None):
    return nh.Model(
        grid=grid,
        core=nh.Core() if family is None else nh.Core(family=family),
        buoyancy=nh.ConstantStratification(n2=1.0),
        time_stepper=AdamBashforth(1e-3, order=3))


@pytest.mark.parametrize("walled_y", [False, True],
                         ids=["channel-x", "box-xy"])
def test_walled_rel_vort_z_lands_on_the_free_slip_edge(walled_y):
    # each velocity's difference emits a BC-free face on the axis it
    # differentiated along, so the edge has to re-assert the *other*
    # velocity's wall tag: the free-slip claim zeta = 0 at the wall,
    # the same claim SmagorinskyLilly's shear strain and
    # sw.State.rel_vort make.
    grid = _walled_grid(walled_y=walled_y)
    state = _set_random(_walled_model(grid))
    edge = state.rel_vort_z.function_space.bare
    mx, my = grid.factors[0], grid.factors[1]
    assert edge.factor("x") is mx.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    expected_y = (my.nodal(NodeSet.INNER, bc=BC.DIRICHLET) if walled_y
                  else state.v.function_space.bare.factor("y"))
    assert edge.factor("y") is expected_y
    # the retag is a claim, never a value: z is untouched
    assert edge.factor("z") is state.v.function_space.bare.factor("z")


def test_walled_rel_vort_z_values_are_the_plain_centred_stencils():
    # the wall tag changes no sample: every edge value is still the
    # centred pair of differences
    grid = _walled_grid()
    state = _set_random(_walled_model(grid))
    u, v = np.asarray(state.u.data), np.asarray(state.v.data)
    dx = float(grid.factor("x").dx)
    dy = float(grid.factor("y").dx)
    expected = ((v[1:] - v[:-1]) / dx
                - (np.roll(u, -1, axis=1) - u) / dy)
    np.testing.assert_allclose(np.asarray(state.rel_vort_z.data),
                               expected, rtol=0.0, atol=1e-13)


def test_walled_rel_vort_z_at_cells_closes_on_the_wall_claim():
    # the regression this pins: on the BC-free edge the conversion to
    # cell centres averaged an *unrepaired wall ghost* into the two
    # wall columns. That ghost holds the stencil's own out-of-range
    # output, v[wall cell] / dx, so the artifact grew as 1 / dx and
    # dominated the field under refinement. With the claim the wall
    # cell is the mean of the claimed zero and the first interior edge.
    grid = _walled_grid()
    state = _set_random(_walled_model(grid))
    zeta = state.rel_vort_z
    target = state.b.function_space.bare.factor("x")  # x only
    edge = np.asarray(zeta.data)
    cells = np.asarray(zeta.to(target).data)
    np.testing.assert_allclose(cells[0], 0.5 * edge[0],
                               rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(cells[-1], 0.5 * edge[-1],
                               rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(cells[1:-1],
                               0.5 * (edge[:-1] + edge[1:]),
                               rtol=0.0, atol=1e-14)
    # so the wall columns can never out-shout the interior again
    assert (float(np.abs(cells[[0, -1]]).max())
            <= float(np.abs(edge).max()))


def test_walled_rel_vort_z_at_cells_reads_only_true_dofs():
    # the same statement as a storage invariant: a wall row that reads
    # only true DOFs cannot notice the ghost slots being rebuilt. The
    # BC-free edge failed this — it read whatever the difference
    # kernels had left outside the true region.
    grid = _walled_grid()
    state = _set_random(_walled_model(grid))
    zeta = state.rel_vort_z
    target = state.b.function_space.bare.factor("x")
    fresh = zeta.with_data(zeta.data)  # identical DOFs, ghosts rebuilt
    np.testing.assert_array_equal(np.asarray(fresh.to(target).data),
                                  np.asarray(zeta.to(target).data))


@pytest.mark.parametrize("family", ["nodal", "fv"])
def test_walled_rel_vort_z_differentiates_along_the_wall(family):
    # the half-tagged edge had no registered 'diff' row on the walled
    # axis under the nodal family (DispatchError); the claim restores
    # it, and the two families agree on where it lands.
    grid = _walled_grid()
    state = _set_random(_walled_model(grid, family=family))
    cell_x = state.b.function_space.bare.factor("x")
    assert (state.rel_vort_z.diff("x").function_space.bare.factor("x")
            is cell_x)
