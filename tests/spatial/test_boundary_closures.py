"""
Boundary closures across the spatial layer (boundary_plan.md, R1-R4).

Description
-----------
The home of the staged boundary work. Stage 2a is landed here:
per-side (mixed) BC grounding and fills through ``grid.sync``, and
the ``BC.ROBIN`` structural member (kinds-only key; the
data-parameterized alpha/g fill arrives with the ``("ghost_fill",
space)`` path of stage 2e). The principle under test: the space key
carries per-side closure *structure* only, boundary *values* are
dynamic, and the storage layer never invents values.

The 2b-2d stages (BC-structured operator rows, the R1 legality
flip, one-sided opt-in rows) are NOT landed on this branch: they
collide with dev's post-fork walled-model work — the Sadourny
advection (commit 4b4bc85) builds nonlinear-product fields on the
kept Dirichlet tag for its exact-zero wall values, which the
branch's product-drop corollary and kind-flipped Center->Outer
codomain both contradict (see the rework report).
"""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.spaces.nodal import NodeSet

N = 8


@pytest.fixture
def mesh():
    return IntervalMesh(N, (0.0, 1.0), periodic=False, name="y")


@pytest.fixture
def grid(mesh):
    # one device pinned: the fill assertions index the single-shard
    # storage frame (blocked variants live in the multi-device suite)
    return Grid((mesh,), device_ids=(0,))


# ================================================================
#  Stage 2a: per-side (mixed) fills through grid.sync
# ================================================================
def test_mixed_pair_fills_per_side(grid, mesh):
    # Dirichlet left (odd about the wall), Neumann right (even):
    # one field, two different mirrors
    space = mesh.nodal(NodeSet.CENTER, bc=(BC.DIRICHLET, BC.NEUMANN))
    f = grid.sync(grid.create_field(
        space, init=lambda y: jnp.sin(jnp.pi * y / 2)))
    storage = np.asarray(f._data)
    data = np.asarray(f.data)
    w = grid.decomposition.halo["y"]
    # left side is odd — ghost slot k mirrors the k-th DOF
    assert storage[w - 1] == pytest.approx(-data[0])
    assert storage[w - 2] == pytest.approx(-data[1])
    # right side is even
    assert storage[-w] == pytest.approx(data[-1])
    assert storage[-w + 1] == pytest.approx(data[-2])


def test_neumann_dirichlet_flipped_pair(grid, mesh):
    space = mesh.nodal(NodeSet.CENTER, bc=(BC.NEUMANN, BC.DIRICHLET))
    f = grid.sync(grid.create_field(
        space, init=lambda y: jnp.cos(jnp.pi * y / 2)))
    storage = np.asarray(f._data)
    data = np.asarray(f.data)
    w = grid.decomposition.halo["y"]
    assert storage[w - 1] == pytest.approx(data[0])       # even
    assert storage[-w] == pytest.approx(-data[-1])        # odd


# ================================================================
#  Stage 2a: Robin structure — kinds only, no invented fill
# ================================================================
def test_robin_fill_points_at_the_data_path(grid, mesh):
    # Robin fills are data-parameterized and arrive with stage 2e;
    # until then a sync on a Robin space is a loud, guiding error
    space = mesh.nodal(NodeSet.CENTER, bc=BC.ROBIN)
    f = grid.create_field(space, init=lambda y: y)
    with pytest.raises(NotImplementedError, match="ghost_fill"):
        grid.sync(f)


def test_robin_field_creation_and_arithmetic_work(grid, mesh):
    # structure without fills: fields exist, pointwise work is fine
    space = mesh.nodal(NodeSet.CENTER, bc=(BC.ROBIN, BC.NEUMANN))
    f = grid.create_field(space, init=lambda y: y)
    g = 2.0 * f + f
    assert jnp.allclose(g.data, 3.0 * f.data)
