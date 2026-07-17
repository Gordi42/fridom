"""BoundaryFlux: construction, declarations, wall weight, taught errors.

The module is framework-level; the tests exercise it on a walled-z
nonhydro2 box (the test_coriolis/test_relaxation precedent of borrowing
a concrete port). The wall-weight builder and the bind-time taught
errors (a)-(d of BF-D2) are pinned here; the physical oracles live in
the ``test_boundary_flux_oracles`` shard.
"""
import jax
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.declarations import Lifecycle
from fridom.model.errors import MissingFieldError
from fridom.model.modules.boundary_flux import BoundaryFlux
from fridom.spatial.bc import BC
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh
from fridom.spatial.space_patterns import Collocated, Profile

N = 6
LX = 2 * np.pi
LZ = 1.0
DZ = LZ / N


def make_grid(*, walled="z", device_ids=None):
    """Return a box periodic except along ``walled`` (bounded [0, LZ])."""
    meshes = tuple(
        IntervalMesh(N, (0.0, LZ if name == walled else LX),
                     periodic=(name != walled), name=name)
        for name in ("x", "y", "z"))
    return Grid(meshes, device_ids=device_ids)


def make_model(*modules, grid=None):
    """Linear nh model; f0 = n2 = 0 isolates the boundary flux."""
    return nh.Model(
        grid=make_grid() if grid is None else grid, dt=1e-3,
        advection=False, coriolis=nh.FPlaneCoriolis(f0=0.0),
        stratification=nh.ConstantStratification(n2=0.0),
        modules_extra=modules)


# ================================================================
#  A duck-typed bind table (for the DIRICHLET / chart taught errors,
#  which need a resolved space or grid not producible by nh.Model)
# ================================================================
class _Record:
    def __init__(self, space, lifecycle):
        self.space = space
        self.lifecycle = lifecycle


class _Table:
    def __init__(self, grid, records):
        self.grid = grid
        self._records = records

    def __getitem__(self, name):
        return self._records[name]


class _ChartGrid:

    """A minimal grid stub whose chart triggers the (d) rejection."""

    chart_coords = ("lon", "lat")


# ================================================================
#  Construction-time validation
# ================================================================
def test_side_must_be_left_or_right():
    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        BoundaryFlux("b", "z", "top")


def test_field_and_coord_must_be_non_empty_names():
    with pytest.raises(TypeError, match="field must be a non-empty"):
        BoundaryFlux("", "z", "right")
    with pytest.raises(TypeError, match="coord must be a non-empty"):
        BoundaryFlux("b", "", "right")


def test_flux_must_be_a_number_or_callable():
    with pytest.raises(TypeError, match="flux must be a number"):
        BoundaryFlux("b", "z", "right", flux="strong")


def test_scale_parameter_is_named_by_the_wall():
    bf = BoundaryFlux("b", "z", "right")
    assert bf.scale_parameter == "boundary_flux.b.z_right.scale"
    left = BoundaryFlux("u", "x", "left")
    assert left.scale_parameter == "boundary_flux.u.x_left.scale"


def test_properties_expose_the_target_and_wall():
    bf = BoundaryFlux("b", "z", "left")
    assert (bf.field, bf.coord, bf.side) == ("b", "z", "left")


# ================================================================
#  Declarations
# ================================================================
def test_declares_weight_flux_and_scale():
    bf = BoundaryFlux("b", "z", "right", flux=lambda x: x)
    names = tuple(d.name for d in bf.field_declarations)
    assert names == ("bflux_b_z_right_weight", "bflux_b_z_right_flux")
    for decl in bf.field_declarations:
        assert decl.lifecycle is Lifecycle.AUXILIARY
    # the callable flux varies along x (its named coordinate)
    flux_decl = bf.field_declarations[1]
    assert flux_decl.space == Profile("x")
    refs = tuple(ref.name for ref in bf.field_references)
    assert refs == ("b",)
    params = tuple(p.name for p in bf.parameter_declarations)
    assert params == ("boundary_flux.b.z_right.scale",)


def test_constant_flux_is_a_one_dof_profile():
    bf = BoundaryFlux("b", "z", "right", flux=2.5)
    assert bf.field_declarations[1].space == Profile()


# ================================================================
#  The wall weight (1/Delta n at the wall-adjacent cell)
# ================================================================
@pytest.mark.parametrize(("side", "idx"), [("right", -1), ("left", 0)])
def test_wall_weight_places_inverse_dn_at_the_wall_cell(side, idx):
    bf = BoundaryFlux("b", "z", side, flux=1.0)
    model = make_model(bf)
    weight = np.asarray(model.state[bf._weight_name].data)
    assert weight.shape == (1, 1, N)
    flat = weight.reshape(-1)
    # exactly one nonzero row, equal to 1/Delta z at the wall cell
    assert np.count_nonzero(flat) == 1
    np.testing.assert_allclose(flat[idx], 1.0 / DZ)


@pytest.mark.parametrize(("side", "idx"), [("right", -1), ("left", 0)])
def test_wall_weight_on_a_stretched_mesh_uses_the_wall_cell_dn(side, idx):
    # nh's spectral pressure solver has no transform on a mapped mesh,
    # so the weight builder is exercised directly against the measure
    # (the task's documented fallback for the stretched-normal case)
    zmap = MappedIntervalMesh(
        N, (0.0, LZ), mapping=lambda s: s**2, periodic=False, name="z")
    grid = Grid((IntervalMesh(N, (0.0, LX), periodic=True, name="x"),
                 IntervalMesh(N, (0.0, LX), periodic=True, name="y"),
                 zmap))
    space = Profile("z").resolve(grid)
    bf = BoundaryFlux("b", "z", side, flux=1.0)
    weight = np.asarray(bf._wall_weight_default(grid, space).data).ravel()
    dn = np.asarray(grid.measure(space, "z").data).ravel()
    assert np.count_nonzero(weight) == 1
    np.testing.assert_allclose(weight[idx], 1.0 / dn[idx])


# ================================================================
#  Bind-time taught errors (a)-(d)
# ================================================================
def test_unknown_field_is_a_taught_assembly_error():
    with pytest.raises(MissingFieldError,
                       match="BoundaryFlux field-name spelling"):
        make_model(BoundaryFlux("bouyancy", "z", "right"))


def test_non_prognostic_field_is_rejected():
    with pytest.raises(ValueError, match="only PROGNOSTIC fields"):
        make_model(BoundaryFlux("p", "z", "right"))


def test_periodic_coord_is_rejected():
    with pytest.raises(ValueError, match="is periodic"):
        make_model(BoundaryFlux("b", "x", "right"))


def test_unknown_coord_is_rejected():
    with pytest.raises(ValueError, match="not a coordinate of the grid"):
        make_model(BoundaryFlux("b", "q", "right"))


def test_wall_normal_velocity_is_rejected():
    # w is staggered along z (its wall faces are not DOFs): taught (b)
    with pytest.raises(ValueError, match="staggered along 'z'"):
        make_model(BoundaryFlux("w", "z", "right"))


def test_flux_callable_naming_the_normal_coord_is_rejected():
    with pytest.raises(ValueError, match="names the normal coordinate"):
        make_model(BoundaryFlux("b", "z", "right", flux=lambda z: z))


def test_flux_callable_naming_an_unknown_coord_is_rejected():
    with pytest.raises(ValueError, match="does not have"):
        make_model(BoundaryFlux("b", "z", "right", flux=lambda q: q))


def test_dirichlet_wall_is_rejected():
    # a cell-centred field pinned DIRICHLET at the wall: the value is
    # fixed, so a flux cannot be prescribed (taught (c)). Built on a
    # duck-typed table — nh declares no such collocated-Dirichlet field
    grid = make_grid()
    space = Collocated(wall_bc={"z": BC.DIRICHLET}).resolve(grid)
    assert space.factor("z").bc.components[-1] is BC.DIRICHLET
    table = _Table(grid, {"s": _Record(space, Lifecycle.PROGNOSTIC)})
    with pytest.raises(ValueError, match="is DIRICHLET"):
        BoundaryFlux("s", "z", "right").bind(table)


def test_chart_grid_is_rejected():
    table = _Table(_ChartGrid(), {})
    with pytest.raises(ValueError, match="carries an embedding chart"):
        BoundaryFlux("b", "z", "right").bind(table)


# ================================================================
#  Parameter-name collisions and coexistence
# ================================================================
def test_two_instances_on_the_same_wall_collide():
    with pytest.raises(Exception, match=r"boundary_flux\.b\.z_right|bflux_b"):
        make_model(BoundaryFlux("b", "z", "right"),
                   BoundaryFlux("b", "z", "right"))


def test_different_sides_and_fields_coexist():
    model = make_model(
        BoundaryFlux("b", "z", "right", flux=0.3),
        BoundaryFlux("b", "z", "left", flux=0.4),
        BoundaryFlux("u", "z", "right", flux=0.5))
    params = model.parameters
    assert "boundary_flux.b.z_right.scale" in params
    assert "boundary_flux.b.z_left.scale" in params
    assert "boundary_flux.u.z_right.scale" in params


# ================================================================
#  Decomposition invariance (forced-4 vs single device)
# ================================================================
@pytest.mark.multi_device
def test_wall_weight_is_device_count_invariant(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices

    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        bf = BoundaryFlux("b", "z", "right", flux=0.6)
        model = make_model(bf, grid=make_grid(device_ids=device_ids))
        weight = np.asarray(model.state[bf._weight_name].data)
        # exactly one nonzero row globally, on every device count
        assert np.count_nonzero(weight) == 1
        model.advance(4)
        results[tag] = np.asarray(model.state["b"].data)
    np.testing.assert_allclose(results["many"], results["one"],
                               rtol=1e-11, atol=1e-13)
