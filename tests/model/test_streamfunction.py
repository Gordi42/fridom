"""The BC-aware spectral Laplacian inverse (fr.model.streamfunction).

Covers the four topologies the vorticity-prescribed eddy must serve
(periodic, walled x, walled x and y, each with an optional rigid lid)
on both C-grid families: the discrete manufactured round trip, the
gauge asymmetry between periodic and walled axis sets, the exact
divergence-free curl, the wall condition, the image-vortex drift, and
the taught error on a mixed trig product.
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.streamfunction import (
    invert_negative_laplacian,
    spectral_sibling,
)
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.bc import BC

TOPOLOGIES = {
    "periodic": (True, True, True),
    "channel-x": (False, True, True),
    "box-xy": (False, False, True),
    "periodic+lid": (True, True, False),
    "channel-x+lid": (False, True, False),
    "box-xy+lid": (False, False, False),
}


def make_model(periodic, family="fv", n=16, nz=None):
    """Build a tiny linear nonhydro model on the given topology."""
    sizes = (n, n, n if nz is None else nz)
    meshes = tuple(
        fr.spatial.meshes.IntervalMesh(
            size, (0.0, 1.0), periodic=flag, name=name)
        for size, flag, name in zip(sizes, periodic, "xyz", strict=True))
    grid = fr.spatial.Grid(meshes, device_ids=(0,))
    return nh.Model(
        grid=grid,
        core=nh.Core(aspect_ratio=1.0, family=family),
        time_stepper=AdamBashforth(1e-3, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        buoyancy=nh.ConstantStratification(n2=1.0),
        advection=False)


def corner_space(model):
    """Return the vorticity corner (u's x factor by v's y factor)."""
    state = model.state
    return state["u"].function_space.bare.replace(
        y=state["v"].function_space.bare.factor("y"))


def sample(grid, space, values):
    """Sample a coordinate-named callable at a space's own nodes."""
    names = tuple(
        name for factor in space.factors if not factor.is_constant
        for name in factor.names)

    def init(**coords):
        return jnp.asarray(values(**coords))

    init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for name in names])
    return grid.create_field(space, init=init, name="f")


def curl_state(model, psi):
    """Return the state of the curl (u, v) = (d_y psi, -d_x psi)."""
    grid, state = model.grid, model.state
    return nh.State({
        "u": psi.diff("y").with_metadata(name="u"),
        "v": (-psi.diff("x")).with_metadata(name="v"),
        "w": grid.create_field(state["w"].function_space.bare, name="w"),
        "b": grid.create_field(state["b"].function_space.bare, name="b")})


def gaussian_at(pos_x, pos_y, width):
    """Return a z-constant Gaussian bump callable of (x, y, z)."""
    def values(x, y, z):
        return jnp.exp(
            -((x - pos_x) ** 2 + (y - pos_y) ** 2) / width ** 2
        ) * (0.0 * z + 1.0)
    return values


# ================================================================
#  The trig sibling
# ================================================================
@pytest.mark.parametrize("label", list(TOPOLOGIES))
def test_sibling_keeps_the_dirichlet_corner(label):
    """A walled horizontal factor is already Dirichlet and stays so."""
    model = make_model(TOPOLOGIES[label])
    corner = corner_space(model)
    sibling = spectral_sibling(corner, axes=("x", "y"))
    for axis, periodic in zip("xy", TOPOLOGIES[label], strict=False):
        if periodic:
            continue
        factor = sibling.factor(axis)
        assert BC.DIRICHLET in factor.bc.components
        assert factor.shape == corner.factor(axis).shape


def test_sibling_is_identity_on_a_periodic_grid():
    """A fully periodic space is its own sibling (the no-retag path)."""
    model = make_model(TOPOLOGIES["periodic"])
    corner = corner_space(model)
    assert spectral_sibling(corner, axes=("x", "y")) is corner


def test_sibling_tags_a_passive_walled_vertical():
    """A BC-free bounded cell factor is tagged for the transform."""
    model = make_model(TOPOLOGIES["periodic+lid"])
    corner = corner_space(model)
    sibling = spectral_sibling(corner, axes=("x", "y"))
    assert sibling is not corner
    assert not sibling.factor("z").bc.is_free
    assert sibling.factor("z").shape == corner.factor("z").shape


def test_sibling_follows_one_trig_family():
    """With a Dirichlet horizontal, the lid joins the sine family."""
    model = make_model(TOPOLOGIES["channel-x+lid"])
    sibling = spectral_sibling(corner_space(model), axes=("x", "y"))
    assert BC.DIRICHLET in sibling.factor("z").bc.components


def test_sibling_gives_a_cell_scalar_the_dirichlet_wall_parity():
    """An untagged bounded factor on an inverted axis takes Dirichlet."""
    model = make_model(TOPOLOGIES["box-xy"])
    cells = model.state["b"].function_space.bare
    assert cells.factor("x").bc.is_free
    sibling = spectral_sibling(cells, axes=("x", "y"))
    assert BC.DIRICHLET in sibling.factor("x").bc.components
    assert BC.DIRICHLET in sibling.factor("y").bc.components
    assert sibling.shape == cells.shape


def test_sibling_rejects_a_mixed_trig_product():
    """Committed Dirichlet and Neumann factors are a taught error."""
    model = make_model(TOPOLOGIES["channel-x+lid"])
    corner = corner_space(model)
    mesh = next(m for m in model.grid.factors if "z" in m.names)
    mixed = corner.replace(z=mesh.average(
        type(corner.factor("z")), bc=BC.NEUMANN))
    with pytest.raises(ValueError, match="mixed Dirichlet"):
        spectral_sibling(mixed, axes=("x", "y"))


# ================================================================
#  The manufactured discrete round trip
# ================================================================
@pytest.mark.parametrize("label", list(TOPOLOGIES))
@pytest.mark.parametrize("family", ["nodal", "fv"])
@pytest.mark.parametrize("n", [16, 32])
def test_manufactured_round_trip(label, family, n):
    """The manufactured psi comes back exactly, up to the gauge."""
    periodic = TOPOLOGIES[label]
    model = make_model(periodic, family, n)
    corner = corner_space(model)
    kx = 2 * np.pi * 3 if periodic[0] else np.pi * 3
    ky = 2 * np.pi * 2 if periodic[1] else np.pi * 2
    cosine_y = periodic[1]

    def psi_values(x, y, z):
        along_y = jnp.cos(ky * y) if cosine_y else jnp.sin(ky * y)
        return jnp.sin(kx * x) * along_y * (0.0 * z + 1.0)

    psi_exact = sample(model.grid, corner, psi_values)
    zeta = curl_state(model, psi_exact).rel_vort_z.retag(corner)
    psi = invert_negative_laplacian(zeta, axes=("x", "y"))
    assert psi.function_space.bare is corner
    want = np.array(psi_exact.data)
    got = np.array(psi.data)
    want -= want.mean(axis=(0, 1), keepdims=True)
    got -= got.mean(axis=(0, 1), keepdims=True)
    assert np.abs(want - got).max() / np.abs(want).max() < 1e-12


# ================================================================
#  What the caller gets back: vorticity, divergence, walls
# ================================================================
@pytest.mark.parametrize("label", list(TOPOLOGIES))
def test_prescribed_vorticity_is_reproduced(label):
    """A walled axis set recovers zeta exactly; periodic drops the mean."""
    periodic = TOPOLOGIES[label]
    model = make_model(periodic, "fv", 32)
    corner = corner_space(model)
    want = sample(model.grid, corner, gaussian_at(0.5, 0.5, 0.12))
    psi = invert_negative_laplacian(want, axes=("x", "y"))
    got = curl_state(model, psi).rel_vort_z.retag(corner)
    a, b = np.array(want.data), np.array(got.data)
    scale = np.abs(a).max()
    if periodic[0] and periodic[1]:
        # the k = 0 gauge: only the mean-free part is reproduced
        assert np.abs(a - b).max() / scale > 1e-3
        a -= a.mean(axis=(0, 1), keepdims=True)
        b -= b.mean(axis=(0, 1), keepdims=True)
    assert np.abs(a - b).max() / scale < 1e-11


@pytest.mark.parametrize("label", list(TOPOLOGIES))
@pytest.mark.parametrize("family", ["nodal", "fv"])
def test_curl_is_divergence_free(label, family):
    """The C-grid curl of the recovered psi has machine-zero divergence."""
    model = make_model(TOPOLOGIES[label], family, 16)
    corner = corner_space(model)
    zeta = sample(model.grid, corner, gaussian_at(0.4, 0.6, 0.15))
    state = curl_state(model, invert_negative_laplacian(
        zeta, axes=("x", "y")))
    divergence = (state["u"].diff("x") + state["v"].diff("y")
                  + state["w"].diff("z"))
    scale = float(np.abs(np.asarray(state["u"].data)).max())
    assert float(np.abs(np.asarray(divergence.data)).max()) / scale < 1e-12


@pytest.mark.parametrize("label", ["channel-x", "box-xy", "box-xy+lid"])
def test_wall_normal_velocity_has_no_wall_degree_of_freedom(label):
    """Psi and u carry only interior faces, so u = 0 at the wall."""
    model = make_model(TOPOLOGIES[label], "fv", 16)
    corner = corner_space(model)
    zeta = sample(model.grid, corner, gaussian_at(0.15, 0.5, 0.1))
    state = curl_state(model, invert_negative_laplacian(
        zeta, axes=("x", "y")))
    cells = next(m for m in model.grid.factors if "x" in m.names).n_cells
    assert state["u"].function_space.shape[0] == cells - 1
    # the wall-adjacent columns close exactly on u_wall = 0
    divergence = np.asarray(
        (state["u"].diff("x") + state["v"].diff("y")).data)
    scale = float(np.abs(np.asarray(state["u"].data)).max())
    assert np.abs(divergence[0]).max() / scale < 1e-12
    assert np.abs(divergence[-1]).max() / scale < 1e-12


def test_near_wall_eddy_drifts_like_its_image():
    """A near-wall eddy self-advects at the image speed -G / (4 pi d)."""
    model = make_model(TOPOLOGIES["channel-x"], "fv", 128, nz=1)
    corner = corner_space(model)
    distance = 0.12
    zeta = sample(model.grid, corner,
                  gaussian_at(distance, 0.5, 0.03))
    state = curl_state(model, invert_negative_laplacian(
        zeta, axes=("x", "y")))
    weights = np.asarray(zeta.data)
    area = (1.0 / 128) ** 2
    circulation = float(weights.sum() * area)
    v_corner = np.asarray(state["v"].to(zeta).data)
    drift = float((v_corner * weights).sum() / weights.sum())
    predicted = -circulation / (4 * np.pi * distance)
    assert abs(drift / predicted - 1.0) < 0.05


# ================================================================
#  Differentiability
# ================================================================
@pytest.mark.parametrize("label", ["periodic", "channel-x", "box-xy+lid"])
def test_gradient_matches_finite_difference(label):
    """jax.grad through the inversion is finite and correct."""
    model = make_model(TOPOLOGIES[label], "fv", 8)
    corner = corner_space(model)
    base = sample(model.grid, corner, gaussian_at(0.5, 0.4, 0.15))

    def loss(amplitude):
        zeta = base.with_data(base.data * amplitude)
        psi = invert_negative_laplacian(zeta, axes=("x", "y"))
        return 0.5 * jnp.sum(psi.data ** 2)

    gradient = float(jax.grad(loss)(1.3))
    step = 1e-5
    reference = float((loss(1.3 + step) - loss(1.3 - step)) / (2 * step))
    assert np.isfinite(gradient)
    assert abs(gradient - reference) / abs(reference) < 1e-4


# ================================================================
#  The barotropic (z-collapsed) caller pattern
# ================================================================
@pytest.mark.parametrize("label", ["periodic", "channel-x", "box-xy+lid"])
def test_barotropic_route_needs_no_vertical_transform(label):
    """A constant vertical factor skips the vertical stage entirely."""
    model = make_model(TOPOLOGIES[label], "fv", 16, nz=8)
    corner = corner_space(model)
    mesh = next(m for m in model.grid.factors if "z" in m.names)
    flat = corner.replace(z=mesh.constant)
    assert spectral_sibling(flat, axes=("x", "y")) is flat
    zeta = sample(model.grid, flat,
                  lambda x, y: jnp.exp(
                      -((x - 0.35) ** 2 + (y - 0.5) ** 2) / 0.15 ** 2))
    psi = invert_negative_laplacian(zeta, axes=("x", "y"))
    declared = model.state
    state = nh.State({
        "u": psi.diff("y").to(
            declared["u"].function_space.bare).with_metadata(name="u"),
        "v": (-psi.diff("x")).to(
            declared["v"].function_space.bare).with_metadata(name="v"),
        "w": model.grid.create_field(
            declared["w"].function_space.bare, name="w"),
        "b": model.grid.create_field(
            declared["b"].function_space.bare, name="b")})
    got = np.asarray(state.rel_vort_z.retag(corner).data)
    want = np.broadcast_to(np.asarray(zeta.data), got.shape)
    scale = np.abs(want).max()
    if TOPOLOGIES[label][0] and TOPOLOGIES[label][1]:
        got = got - got.mean(axis=(0, 1), keepdims=True)
        want = want - want.mean(axis=(0, 1), keepdims=True)
    assert np.abs(got - want).max() / scale < 1e-11
