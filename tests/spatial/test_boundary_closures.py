"""
Boundary closures across the spatial layer (boundary_plan.md, R1-R4).

Description
-----------
The home of the staged boundary work, on wall-value semantics
(the owner's walled-Sadourny decisions stay: tags are wall-value
claims kept through products, the tag governs the ghost fill, and
nodal operator outputs are BC-free). Landed here:

- **2a** — per-side (mixed) BC grounding and fills through
  ``grid.sync``; the ``BC.ROBIN`` structural member (kinds-only
  key; the data-parameterized alpha/g fill arrives with the
  ``("ghost_fill", space)`` path of stage 2e).
- **2c'** — the R1 flip: BC-free bounded sides define no exterior
  values; exterior-needing signatures demand BC structure on every
  needy side and un-seed themselves otherwise.

The principle under test: the space key carries per-side closure
*structure* only, boundary *values* are dynamic, and the storage
layer never invents values.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
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


# ================================================================
#  Wall-value semantics (owner decisions, walled-Sadourny work):
#  tags are wall-value claims kept through products and linear
#  arithmetic — NOT parity statements
# ================================================================
def test_products_keep_their_bc_tag(grid, mesh):
    # a tag is a wall-value claim consumed by the staggered fills:
    # keeping Dirichlet through v * v is exactly what hands the
    # walled advection its exact-zero wall fluxes (grid.py seeding
    # note; the parity-drop alternative was owner-overruled)
    space = mesh.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    f = grid.create_field(space, init=lambda y: jnp.sin(jnp.pi * y))
    assert (f * f).function_space.bare is space
    assert (f ** 2).function_space.bare is space
    assert (f / (1.0 + f)).function_space.bare is space


def test_linear_arithmetic_preserves_bc_structure(grid, mesh):
    space = mesh.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    f = grid.create_field(space, init=lambda y: jnp.sin(jnp.pi * y))
    assert (2.0 * f + f).function_space.bare is space
    assert (-f).function_space.bare is space


# ================================================================
#  Stage 2c': the R1 flip — exterior-needing BC-free signatures
#  demand a closure (declared structure or the one-sided opt-in)
# ================================================================
def test_bc_tagged_exterior_needing_rows_stay_grounded(grid, mesh):
    # the tag grounds the mirror fill, so Inner(D) -> Center stays
    # (dev semantics: the codomain is the BC-free sibling)
    fd = FiniteDifference(order=2)
    inner_d = mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    assert fd.codomain(inner_d) is mesh.center
    assert grid.dispatch.resolve("diff", inner_d) is not None


def test_wide_stencil_windows_are_gated_per_reach(mesh):
    # order 4 reads one exterior cell even for Center -> Inner: the
    # R1 gate keys on the actual window reach, not the node sets
    fd4 = FiniteDifference(order=4)
    with pytest.raises(SpaceMismatchError, match="one_sided"):
        fd4.codomain(mesh.center)
    assert fd4.codomain(
        mesh.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)) is mesh.inner


def test_partially_tagged_domains_are_gated_per_side(mesh):
    # (DIRICHLET, NONE): the left wall is grounded, the right is
    # not — Inner -> Center still raises, naming the free side
    fd = FiniteDifference(order=2)
    space = mesh.nodal(NodeSet.INNER, bc=(BC.DIRICHLET, BC.NONE))
    with pytest.raises(SpaceMismatchError, match="right wall"):
        fd.codomain(space)
