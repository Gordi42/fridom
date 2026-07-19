"""The nonhydrostatic ``State`` vocabulary: the chart-native view."""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.nonhydro2.state import State
from fridom.spatial.coordinate_mapping import CoordinateMapping

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
    st = _set_random(nh.Model(grid=grid, dt=1e-3))
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
    st = _set_random(nh.Model(grid=_flat_grid(), dt=1e-3))
    assert st.chart["w"] is st["w"]
    assert st.chart["u"] is st["u"]
    assert st.chart.w is st["w"]


def test_chart_velocities_and_read_only():
    st = _set_random(nh.Model(grid=_flat_grid(), dt=1e-3))
    u, v, w = st.chart.velocities
    assert u is st["u"]
    assert v is st["v"]
    assert w is st["w"]
    with pytest.raises(TypeError, match="read-only"):
        st.chart["w"] = None
    with pytest.raises(AttributeError, match="read-only"):
        st.chart.w = None
