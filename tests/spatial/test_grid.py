"""Tests for fridom.spatial.grid."""
import jax
import jax.numpy as jnp
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.errors import (
    GridFrozenError,
    GridMismatchError,
    SpaceMismatchError,
)
from fridom.spatial.fields.metadata import FieldMetadata
from fridom.spatial.grid import Grid, _tagged_trig_origins
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import (
    MappedIntervalMesh,
)
from fridom.spatial.meshes.point import PointMesh
from fridom.spatial.operators.base import (
    OperatorRequirements,
)
from fridom.spatial.operators.composed import (
    LowerIndex,
    MetricCurl,
    MetricDivergence,
    MetricGradient,
    MetricLaplacian,
    RaiseIndex,
)
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.spatial.operators.integrate import Integral
from fridom.spatial.operators.interp import LinearInterp
from fridom.spatial.operators.mapped import MappedDerivative
from fridom.spatial.operators.reconstruct import (
    LinearDeconvolution,
    LinearReconstruction,
)
from fridom.spatial.operators.registry import (
    DispatchError,
    OperatorRegistry,
)
from fridom.spatial.operators.restrict import Restriction
from fridom.spatial.spaces.nodal import NodeSet


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(4, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def grid(mx, my):
    return Grid((mx, my))


@pytest.fixture
def grid1d(mx):
    return Grid((mx,))


# ================================================================
#  Construction and identity
# ================================================================
def test_structure(grid, mx, my):
    assert grid.factors == (mx, my)
    assert grid.names == ("x", "y")


def test_empty_meshes_rejected():
    with pytest.raises(ValueError, match="at least one mesh"):
        Grid(())


def test_duplicate_names_rejected(mx):
    other = IntervalMesh(4, (0.0, 2.0), name="x")
    with pytest.raises(ValueError, match="duplicate coordinate"):
        Grid((mx, other))


def test_identity_equality_and_hash(mx, my):
    a = Grid((mx, my))
    b = Grid((mx, my))
    assert a == a  # noqa: PLR0124 — identity semantics under test
    assert a != b
    assert a != "grid"
    assert hash(a) == id(a)


def test_default_family_defaults_to_nodal(grid):
    assert grid.default_family == "nodal"


def test_default_family_is_settable(mx, my):
    assert Grid((mx, my), family="fv").default_family == "fv"


def test_default_family_validated(mx, my):
    with pytest.raises(ValueError,
                       match="grid-level default family must be one"):
        Grid((mx, my), family="bogus")


def test_set_default_family_flips_pre_freeze(mx, my):
    # the assembly-phase mutation (F3): a model factory adopts a family
    grid = Grid((mx, my))
    grid.set_default_family("fv")
    assert grid.default_family == "fv"


def test_set_default_family_validates(mx, my):
    with pytest.raises(ValueError,
                       match="grid-level default family must be one"):
        Grid((mx, my)).set_default_family("bogus")


def test_set_default_family_frozen_is_a_noop_when_unchanged(mx, my):
    grid = Grid((mx, my), family="fv")
    grid.freeze()
    grid.set_default_family("fv")  # unchanged: no-op on a frozen grid
    assert grid.default_family == "fv"


def test_set_default_family_frozen_rejects_a_change(mx, my):
    grid = Grid((mx, my))
    grid.freeze()
    with pytest.raises(GridFrozenError, match="the grid is frozen"):
        grid.set_default_family("fv")


def test_dispatch_defaults_to_seeded_registry_and_is_settable(mx):
    registry = Grid((mx,)).dispatch
    assert isinstance(registry, OperatorRegistry)
    assert isinstance(registry.resolve("diff", mx.center),
                      FiniteDifference)
    assert isinstance(registry.resolve("interpolate", mx.center),
                      LinearInterp)
    custom = object()
    assert Grid((mx,), dispatch=custom).dispatch is custom


def test_decomposition_is_single_device_provisional_halo(mx, my):
    # provisional negotiation: halo = per-operator max over the
    # seeded registry; the widest entry is the two-factor
    # FV-derivative chain (reconstruct [0,+1] then flux_diff [-1,0]),
    # whose two-sided composed window [-1,+1] is width 1.
    # Pinned to one device: the assertions read the single-shard
    # storage frame.
    dec = Grid((mx, my), device_ids=(0,)).decomposition
    assert dec.halo["x"] == 1
    assert dec.halo["y"] == 1
    space = mx.center * my.center
    assert dec.storage_shape(space) == (8 + 2, 4 + 2)
    assert dec.default_layout.device_axes == ()


def test_duck_registry_without_items_gets_zero_halo(mx):
    grid = Grid((mx,), dispatch=object())
    assert grid.decomposition.halo["x"] == 0


def test_seeded_registry_covers_the_default_rows(grid, mx, my):
    registry = grid.dispatch
    fd = registry.resolve("diff", mx.center)
    assert fd is registry.resolve("diff", mx.right)
    assert fd is registry.resolve("diff", my.outer)
    # BC-free bounded Inner -> Center needs the wall faces: the row
    # un-seeds itself under R1 (boundary_plan.md) — declare BC
    # structure or opt into boundary="one_sided"
    with pytest.raises(DispatchError, match="diff"):
        registry.resolve("diff", my.inner)
    multiply = registry.resolve("multiply", mx.center)
    assert multiply is registry.resolve("multiply", mx.cell_avg)
    assert multiply is registry.resolve(
        "multiply", mx.center.as_complex())
    registry.resolve("divide", my.face_avg)
    registry.resolve("power", mx.cell_avg)
    registry.resolve("abs", mx.center)
    registry.resolve("abs", mx.cell_avg)  # abs is pointwise on averages
    with pytest.raises(DispatchError, match="diff"):
        registry.resolve("diff", my.right)  # no bounded Right row


def test_seeded_registry_covers_the_restrict_row(grid, mx, my):
    # the Outer -> Inner restriction (H2b): the exact interior-face
    # selection the hydrostatic advection resolves for w-on-Outer. It
    # is seeded only on a bounded factor's Outer space (a periodic
    # factor has no Outer); no other node set grounds it, and the
    # separate Outer -> Center interpolate row is untouched.
    registry = grid.dispatch
    restrict = registry.resolve("restrict", my.outer)
    assert isinstance(restrict, Restriction)
    assert restrict.codomain(my.outer) is my.inner
    # Outer -> Center stays the (distinct) interpolate row
    assert isinstance(registry.resolve("interpolate", my.outer),
                      LinearInterp)
    # no restrict row on the other node sets or on a periodic factor
    for space in (my.center, my.inner, mx.center):
        with pytest.raises(DispatchError, match="restrict"):
            registry.resolve("restrict", space)


def test_seeded_registry_covers_the_deconvolve_rows(grid, mx, my):
    # G3: the co-located CellAvg <-> Center deconvolution is seeded on
    # both members of the primal pair, periodic and bounded, sharing
    # one LinearDeconvolution instance; nothing else grounds it
    registry = grid.dispatch
    deconv = registry.resolve("deconvolve", mx.cell_avg)
    assert isinstance(deconv, LinearDeconvolution)
    assert deconv is registry.resolve("deconvolve", mx.center)
    assert deconv is registry.resolve("deconvolve", my.cell_avg)
    assert deconv is registry.resolve("deconvolve", my.center)
    for absent in (mx.right, my.inner, mx.face_avg):
        with pytest.raises(DispatchError, match="deconvolve"):
            registry.resolve("deconvolve", absent)


def test_seeded_registry_covers_the_average_interpolate_rows(
        grid, mx, my):
    # G4: the average family under the interpolate kind reuses the
    # reconstruct instance (the staggering hop the composed metric
    # machinery needs); FaceAvg falls out of the identical path free
    registry = grid.dispatch
    recon = registry.resolve("reconstruct", mx.cell_avg)
    assert isinstance(recon, LinearReconstruction)
    assert registry.resolve("interpolate", mx.cell_avg) is recon
    assert registry.resolve("interpolate", mx.face_avg) is recon
    assert registry.resolve("interpolate", my.cell_avg) is recon
    assert registry.resolve("interpolate", my.face_avg) is recon


def test_seeded_registry_covers_the_bc_tagged_trig_origins(grid, my):
    # the four bounded trig origins (DST-II/DST-I/DCT-II/DCT-I) get
    # the same shared stencil rows as the BC-free nodal family (C3)
    registry = grid.dispatch
    fd = registry.resolve("diff", my.center)
    interp = registry.resolve("interpolate", my.center)
    for space in (my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET),
                  my.nodal(NodeSet.INNER, bc=BC.DIRICHLET),
                  my.nodal(NodeSet.CENTER, bc=BC.NEUMANN),
                  my.nodal(NodeSet.OUTER, bc=BC.NEUMANN)):
        assert registry.resolve("diff", space) is fd
        assert registry.resolve("interpolate", space) is interp
    with pytest.raises(DispatchError, match="diff"):
        # Dirichlet-Outer drops DOFs: deliberately no row
        registry.resolve("diff",
                         my.nodal(NodeSet.OUTER, bc=BC.DIRICHLET))


def test_seeded_registry_covers_tagged_elementwise_rows(grid, my):
    # walled-grid fields must interoperate pointwise: the elementwise
    # and integrate rows are seeded on the BC-tagged trig origins too
    # (e.g. the flux form ``csqr.to(v) * v`` on a Dirichlet face
    # space), sharing the one instance per kind of the BC-free family
    registry = grid.dispatch
    for kind in ("multiply", "divide", "power", "select", "abs",
                 "integrate"):
        shared = registry.resolve(kind, my.center)
        for space in (my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET),
                      my.nodal(NodeSet.INNER, bc=BC.DIRICHLET),
                      my.nodal(NodeSet.CENTER, bc=BC.NEUMANN),
                      my.nodal(NodeSet.OUTER, bc=BC.NEUMANN)):
            assert registry.resolve(kind, space) is shared
            assert registry.resolve(kind,
                                    space.as_complex()) is shared


def test_tagged_product_keeps_the_tagged_space(grid, mx, my):
    tagged = my.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    space = mx.center * tagged
    f = grid.create_field(space, init=lambda x, y: x + y)
    g = grid.create_field(space, init=lambda x, y: x * y + 1.0)
    product = f * g
    assert product.function_space.bare is space
    # the tag governs only the ghost fill: the true-shape payload is
    # byte-identical to the BC-free product of the same samples
    bare = mx.center * my.inner
    fb = grid.create_field(bare, data=jnp.asarray(f.data))
    gb = grid.create_field(bare, data=jnp.asarray(g.data))
    assert jnp.array_equal(product.data, (fb * gb).data)


def test_periodic_meshes_ground_no_tagged_rows(mx):
    # regression: the tagged elementwise seeding is a no-op on
    # periodic meshes — their registry stays the BC-free family
    assert _tagged_trig_origins(mx) == ()


def test_seeded_registry_covers_the_spectral_rows(grid, mx, my):
    registry = grid.dispatch
    spectral = registry.resolve("diff", mx.fourier(origin=mx.center))
    assert type(spectral).__name__ == "SpectralDerivative"
    assert spectral is registry.resolve(
        "diff", mx.fourier(origin=mx.cell_avg))
    dirichlet = my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    assert spectral is registry.resolve("diff", my.sine(dirichlet))
    # one-directional origin shifts to Center (wave-3B modules)
    shift = registry.resolve(
        "interpolate", mx.fourier(origin=mx.right))
    assert type(shift).__name__ == "PhaseShift"
    assert shift.to is NodeSet.CENTER
    sinc = registry.resolve(
        "interpolate", mx.fourier(origin=mx.cell_avg))
    assert type(sinc).__name__ == "SincShift"
    with pytest.raises(DispatchError, match="interpolate"):
        # center origins need no shift; no row by design
        registry.resolve("interpolate", mx.fourier(origin=mx.center))


def test_seeded_registry_covers_the_transform_rows(grid, mx, my):
    registry = grid.dispatch
    fourier = registry.resolve("transform", mx.center)
    assert type(fourier).__name__ == "Fourier"
    assert fourier.grid is grid
    assert fourier.axes == ("x",)  # periodic family axes only
    # one shared lazy instance across the family's rows
    assert fourier is registry.resolve("transform", mx.cell_avg)
    assert fourier is registry.resolve(
        "transform", mx.right.as_complex())
    dirichlet = my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    sine = registry.resolve("transform", dirichlet)
    assert type(sine).__name__ == "Sine"
    assert sine.axes == ("y",)
    neumann = my.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    assert type(registry.resolve(
        "transform", neumann)).__name__ == "Cosine"
    with pytest.raises(DispatchError, match="transform"):
        registry.resolve("transform", my.center)  # BC-free bounded


# ================================================================
#  Lifecycle (negotiate / freeze; merge_overrides is a later stub)
# ================================================================
def test_merge_overrides_is_pre_freeze_only(grid):
    grid.merge_overrides({})
    grid.freeze()
    with pytest.raises(GridFrozenError, match="frozen"):
        grid.merge_overrides({})


def test_negotiate_returns_a_resharding_report(grid):
    report = grid.negotiate()
    assert report.old == report.new
    assert report.changed is False
    assert report.new is grid.decomposition.default_layout


def test_negotiate_honors_an_explicit_halo(mx, my):
    # pin to one device so no shardability cap engages: on a sharded
    # grid the explicit halo would be capped to the shortest-shard
    # extent (established, deliberate semantics), which would mask the
    # "negotiate honors the explicit width" claim. The uncapped honoring
    # is what this test asserts, so it must run where no cap applies --
    # and stays unmarked (passes at any device count).
    grid = Grid((mx, my), device_ids=(0,))
    grid.negotiate(halo=HaloSpec({"x": 3}))
    assert grid.decomposition.halo["x"] == 3
    assert grid.decomposition.halo["y"] == 0


@pytest.mark.multi_device
def test_negotiate_caps_an_explicit_halo_on_a_sharded_axis():
    # the sharded counterpart: on a genuinely sharded axis the explicit
    # halo is capped to the shortest-shard extent (shortest shard - 1),
    # and negotiate records the capped width, not the requested one.
    # nx = 4 * device_count keeps the shortest shard 4 cells for any
    # device count >= 2, so the cap is 3 and x still shards.
    nx = 4 * jax.device_count()
    grid = Grid((
        IntervalMesh(nx, (0.0, 1.0), name="x"),
        IntervalMesh(nx, (0.0, 2.0), periodic=False, name="y")))
    assert dict(grid.decomposition.default_layout.device_axes) == {
        "x": "devices"}
    grid.negotiate(halo=HaloSpec({"x": 20}))
    assert grid.decomposition.halo["x"] == 3
    assert grid.decomposition.halo["y"] == 0


def test_negotiate_traces_a_tendency(grid):
    space = grid.create_field().function_space

    def tendency(state):
        return state.diff("x")

    grid.negotiate(state_spaces=(space,), tendency=tendency)
    assert grid.decomposition.halo["x"] == 1
    assert grid.decomposition.halo["y"] == 0


def test_negotiate_tendency_requires_state_spaces(grid):
    with pytest.raises(ValueError, match="state_spaces"):
        grid.negotiate(tendency=lambda state: state)


def test_freeze_ends_the_assembly_phase(mx, my):
    # pin to one device so the larger-demand raise is genuine at any
    # device count: on a sharded grid the shardable-cap would absorb
    # halo={x: 99} to the shard extent (width above the floor is
    # satisfiable via runtime re-sync), so the raw comparison must be
    # exercised on a non-sharding grid
    grid = Grid((mx, my), device_ids=(0,))
    grid.freeze()
    # satisfiable demands verify against the frozen record ...
    report = grid.negotiate()
    assert report.changed is False
    # ... larger demands (and mutators) raise GridFrozenError
    with pytest.raises(GridFrozenError, match="frozen"):
        grid.negotiate(halo=HaloSpec({"x": 99}))


def test_sync_is_identity_on_one_device(grid):
    f = grid.create_field(data=jnp.arange(32.0).reshape(8, 4),
                          name="f")
    synced = grid.sync(f)
    assert jnp.array_equal(synced.data, f.data)
    assert synced.function_space is f.function_space
    assert synced.metadata == f.metadata


def test_sync_boundary_data_not_implemented(grid):
    f = grid.create_field()
    with pytest.raises(NotImplementedError, match="ghost fill"):
        grid.sync(f, boundary_data={})


def test_materialized_sync_matches_the_default_spelling(grid):
    # materialize=True changes the fill's SPELLING (in-place writes
    # for a dead operand at a materialization boundary), never its
    # values or claims
    f = grid.create_field(data=jnp.arange(32.0).reshape(8, 4),
                          name="f")
    synced = grid.sync(f, materialize=True)
    assert jnp.array_equal(synced._data, grid.sync(f)._data)
    assert synced.halo_valid == grid.sync(f).halo_valid


# ================================================================
#  create_field — spaces and layout
# ================================================================
def test_default_space_is_all_center_laid_out(grid, mx, my):
    f = grid.create_field()
    assert f.function_space.bare is mx.center * my.center
    assert f.function_space.layout is grid.decomposition.default_layout
    assert jnp.array_equal(f.data, jnp.zeros((8, 4)))


def test_bare_space_gets_default_layout(grid, mx, my):
    f = grid.create_field(mx.right * my.center)
    assert f.function_space.layout is grid.decomposition.default_layout


def test_laid_out_space_is_honored(grid, mx, my):
    space = (mx.center * my.center).with_layout(
        grid.decomposition.default_layout)
    f = grid.create_field(space)
    assert f.function_space is space


def test_lone_factor_space(grid1d, mx):
    f = grid1d.create_field(mx.center)
    assert f.function_space.bare is mx.center
    assert f.shape == (8,)


def test_foreign_mesh_space_raises(grid):
    foreign = IntervalMesh(8, (0.0, 1.0), name="z")
    with pytest.raises(GridMismatchError, match="not a factor"):
        grid.create_field(foreign.center)


def test_refined_mesh_spaces_are_adopted(grid1d, mx):
    fine = mx.refined(2)
    f = grid1d.create_field(fine.center)
    assert f.shape == (16,)


def test_default_space_needs_center_family():
    points = PointMesh(((0.0,), (1.0,)), name="p")
    grid = Grid((points,))
    with pytest.raises(TypeError, match="no Center space"):
        grid.create_field()


# ================================================================
#  create_field — argument validation and metadata sugar
# ================================================================
def test_init_data_and_init_coeff_are_pairwise_exclusive(grid):
    with pytest.raises(ValueError, match="pairwise"):
        grid.create_field(init=lambda x, y: x + y,
                          data=jnp.zeros((8, 4)))
    with pytest.raises(ValueError, match="pairwise"):
        grid.create_field(init=lambda x, y: x + y,
                          init_coeff=lambda kx, ky: kx + ky)
    with pytest.raises(ValueError, match="pairwise"):
        grid.create_field(init_coeff=lambda kx, ky: kx + ky,
                          data=jnp.zeros((8, 4)))


def test_metadata_and_sugar_are_exclusive(grid):
    with pytest.raises(ValueError, match="mutually exclusive"):
        grid.create_field(metadata=FieldMetadata(), name="f")


def test_metadata_sugar(grid):
    f = grid.create_field(name="u", units="m/s")
    assert f.metadata == FieldMetadata.create(name="u", units="m/s")
    g = grid.create_field(units="K")
    assert g.metadata.units == "K"
    assert g.metadata.name == "unnamed"


def test_full_metadata_record(grid):
    md = FieldMetadata.create(name="b", nc_attrs={"axis": "Z"})
    assert grid.create_field(metadata=md).metadata == md


# ================================================================
#  create_field — data path
# ================================================================
def test_data_shape_validated(grid):
    with pytest.raises(ValueError, match="true shape"):
        grid.create_field(data=jnp.zeros((4, 8)))


def test_data_dtype_coerced_to_space_dtype(grid):
    f = grid.create_field(data=jnp.zeros((8, 4), dtype=jnp.int32))
    assert f.dtype == jnp.float64


def test_complex_data_into_real_space_raises(grid):
    with pytest.raises(ValueError, match="demoted"):
        grid.create_field(data=jnp.zeros((8, 4), dtype=complex))


def test_complex_data_into_complex_space(grid, mx, my):
    space = (mx.center * my.center).as_complex()
    f = grid.create_field(space, data=jnp.full((8, 4), 1.0 + 1.0j))
    assert f.dtype == jnp.complex128


def test_data_into_fourier_space_projects_hermitian(grid1d, mx):
    space = mx.fourier(origin=mx.center)
    data = jnp.full(space.shape, 1.0 + 1.0j)
    f = grid1d.create_field(space, data=data)
    assert f.dtype == jnp.complex128
    assert f.data[0] == 1.0 + 0.0j
    assert f.data[-1] == 1.0 + 0.0j  # Nyquist (n = 8 even)
    assert jnp.array_equal(f.data[1:-1], data[1:-1])


# ================================================================
#  create_field — init path (collocation discretization)
# ================================================================
def test_init_matches_analytic_function(grid):
    f = grid.create_field(init=lambda x, y: x**2 + 3.0 * y)
    x = (jnp.arange(8) + 0.5) * 0.125
    y = (jnp.arange(4) + 0.5) * 0.5
    expected = x[:, None] ** 2 + 3.0 * y[None, :]
    assert jnp.allclose(f.data, expected)


def test_init_matches_by_keyword_not_order(grid):
    f = grid.create_field(init=lambda y, x: x + 10.0 * y)
    x = (jnp.arange(8) + 0.5) * 0.125
    y = (jnp.arange(4) + 0.5) * 0.5
    assert jnp.allclose(f.data, x[:, None] + 10.0 * y[None, :])


def test_init_on_staggered_space_samples_staggered_nodes(
        grid, mx, my):
    f = grid.create_field(mx.right * my.center,
                          init=lambda x, y: x + 0.0 * y)
    x = (jnp.arange(8) + 1.0) * 0.125
    assert jnp.allclose(f.data, jnp.broadcast_to(x[:, None], (8, 4)))


def test_init_with_constant_factor_omits_its_name(grid, mx, my):
    f = grid.create_field(mx.constant * my.center,
                          init=lambda y: 2.0 * y)
    y = (jnp.arange(4) + 0.5) * 0.5
    assert jnp.allclose(f.data, (2.0 * y)[None, :])
    assert f.shape == (1, 4)


def test_init_signature_validated(grid):
    with pytest.raises(TypeError, match="exactly"):
        grid.create_field(init=lambda x: x)
    with pytest.raises(TypeError, match="exactly"):
        grid.create_field(init=lambda x, y, z: x + y + z)


def test_init_on_coefficient_space_composes_the_transform(mx):
    # discretize = transform o discretize-on-origin (rules 3.10)
    # pin to one device: the host-side Fourier forward needs every axis
    # local (under floor-1 sharding x=8 would shard the transform axis
    # and the change-of-representation path refuses to run); the
    # transform values are single-controller, so this is unmarked and
    # holds at any device count.
    grid1d = Grid((mx,), device_ids=(0,))
    coeff = mx.fourier(origin=mx.center)
    f = grid1d.create_field(
        coeff, init=lambda x: jnp.sin(2 * jnp.pi * x), name="f")
    assert f.function_space.bare is coeff
    assert f.name == "f"
    # index-based amplitude convention with the Center half-cell
    # inter-origin phase: c_1 = -i/2 * e^{i pi / 8}
    expected = jnp.zeros(coeff.shape, complex).at[1].set(
        -0.5j * jnp.exp(1j * jnp.pi / 8))
    assert jnp.allclose(f.data, expected, atol=1e-14)


def test_init_on_complex_full_spectrum_space(mx):
    # pin to one device: host-side Fourier forward needs a local
    # transform axis (see above); unmarked, holds at any device count.
    grid1d = Grid((mx,), device_ids=(0,))
    coeff = mx.fourier(origin=mx.center.as_complex())
    f = grid1d.create_field(
        coeff, init=lambda x: jnp.exp(2j * jnp.pi * x))
    assert f.function_space.bare is coeff
    # a single mode 1 with the Center half-cell phase e^{i pi / 8}
    expected = jnp.zeros(8, complex).at[1].set(
        jnp.exp(1j * jnp.pi / 8))
    assert jnp.allclose(f.data, expected, atol=1e-14)


def test_init_on_multi_axis_coefficient_space(mx):
    # pin to one device: host-side Fourier forward needs local
    # transform axes; unmarked, holds at any device count.
    mp = IntervalMesh(4, (0.0, 2.0), name="p")
    grid = Grid((mx, mp), device_ids=(0,))
    space = (mx.fourier(origin=mx.center)
             * mp.fourier(origin=mp.center.as_complex()))

    def init(x, p):
        return jnp.sin(2 * jnp.pi * x) * (1.0 + jnp.cos(jnp.pi * p))

    f = grid.create_field(space, init=init)
    assert f.function_space.bare is space
    t = grid.dispatch.resolve("transform", mx.center * mp.center)
    reference = t.forward(grid.create_field(
        mx.center * mp.center, init=init))
    assert jnp.allclose(f.data, reference.data)


def test_init_on_mixed_coefficient_nodal_space(mx, my):
    # only the coefficient factor's axis is transformed; the nodal
    # factor stays collocated. Pin to one device: the host-side Fourier
    # forward needs a local transform axis; unmarked, holds at any
    # device count.
    grid = Grid((mx, my), device_ids=(0,))
    space = mx.fourier(origin=mx.center) * my.center

    def init(x, y):
        return jnp.sin(2 * jnp.pi * x) * (1.0 + y)

    f = grid.create_field(space, init=init)
    assert f.function_space.bare is space
    t = grid.dispatch.resolve("transform", mx.center)
    reference = t.forward(grid.create_field(
        mx.center * my.center, init=init))
    assert jnp.allclose(f.data, reference.data)


def test_init_on_unreachable_coefficient_mix_raises(mx):
    # both axes requested as half spectra: the rfftn schedule puts
    # the half spectrum on the first-listed axis only. Pin to one
    # device so the transform runs and raises the intended
    # SpaceMismatchError (a sharded transform axis would raise the
    # taught cannot-run error first); unmarked, holds at any device
    # count.
    mp = IntervalMesh(4, (0.0, 2.0), name="p")
    grid = Grid((mx, mp), device_ids=(0,))
    both_half = (mx.fourier(origin=mx.center)
                 * mp.fourier(origin=mp.center))
    with pytest.raises(SpaceMismatchError, match="half spectrum"):
        grid.create_field(both_half,
                          init=lambda x, p: jnp.sin(x) + p)


def test_init_coeff_skips_constant_factors(grid, mx, my):
    space = mx.fourier(origin=mx.center) * my.constant
    f = grid.create_field(space, init_coeff=lambda kx: kx + 1.0)
    k = grid.wavenumbers(space, name="x")
    assert jnp.allclose(f.data, k.data + 1.0)


def test_init_coeff_assigns_at_the_wavenumbers(grid1d, mx):
    coeff = mx.fourier(origin=mx.center)
    f = grid1d.create_field(coeff,
                            init_coeff=lambda kx: jnp.exp(-kx))
    k = grid1d.wavenumbers(coeff).data
    assert jnp.allclose(f.data, jnp.exp(-k))
    with pytest.raises(TypeError, match="wavenumber names"):
        grid1d.create_field(coeff, init_coeff=lambda x: x)
    with pytest.raises(TypeError, match="coefficient space"):
        grid1d.create_field(mx.center, init_coeff=lambda kx: kx)


# ================================================================
#  create_field — init path (per-cell quadrature on averages, G8)
# ================================================================
def _cell_average(power, edges):
    """Exact per-cell average of ``x**power`` from the edge array."""
    p1 = power + 1
    return (edges[1:] ** p1 - edges[:-1] ** p1) / (
        p1 * (edges[1:] - edges[:-1]))


def test_quadrature_default_is_bitwise_midpoint_sampling(grid, mx,
                                                         my):
    # the semantics pin: the average-space default (order=None) is
    # exactly the current midpoint collocation — a plain sample of
    # init at the cell_avg evaluation nodes, bitwise
    def init(x, y):
        return jnp.exp(jnp.sin(x)) + y**2

    f = grid.create_field(mx.cell_avg * my.cell_avg, init=init)
    xnodes = (jnp.arange(8) + 0.5) * 0.125
    ynodes = (jnp.arange(4) + 0.5) * 0.5
    midpoint = init(xnodes[:, None], ynodes[None, :])
    assert jnp.array_equal(f.data, midpoint)
    # order=1 is the same one-point rule, also bitwise identical
    one = grid.create_field(mx.cell_avg * my.cell_avg, init=init,
                            order=1)
    assert jnp.array_equal(one.data, f.data)


def test_quadrature_order_is_validated(grid, mx):
    for bad in (0, -3, 2.0, True):
        with pytest.raises(ValueError, match="order="):
            grid.create_field(mx.cell_avg, init=lambda x: x, order=bad)


def test_quadrature_is_exact_for_polynomials_to_degree_2n_minus_1(
        grid1d, mx):
    # n-point Gauss is exact for per-cell averages of polynomials up
    # to degree 2n-1: order 2 -> cubic, order 3 -> quintic
    edges = jnp.arange(9) * 0.125
    cubic = grid1d.create_field(mx.cell_avg, init=lambda x: x**3,
                                order=2)
    assert jnp.allclose(cubic.data, _cell_average(3, edges),
                        atol=1e-14)
    quintic = grid1d.create_field(mx.cell_avg, init=lambda x: x**5,
                                  order=3)
    assert jnp.allclose(quintic.data, _cell_average(5, edges),
                        atol=1e-14)
    # one point short: order 2 is NOT exact for the quartic
    quartic = grid1d.create_field(mx.cell_avg, init=lambda x: x**4,
                                  order=2)
    assert not jnp.allclose(quartic.data, _cell_average(4, edges),
                            atol=1e-10)


def test_quadrature_converges_in_order_at_fixed_mesh(mx):
    # a smooth non-polynomial: the per-cell-average error against an
    # ultra-high-order reference falls by orders of magnitude per
    # added point (robust ratio assertions, not a fitted rate)
    grid = Grid((mx,))

    def init(x):
        return jnp.exp(jnp.sin(2.0 * jnp.pi * x))

    ref = grid.create_field(mx.cell_avg, init=init, order=12).data

    def err(order):
        got = grid.create_field(mx.cell_avg, init=init, order=order)
        return float(jnp.max(jnp.abs(got.data - ref)))

    e1, e2, e3 = err(1), err(2), err(3)
    assert e2 < e1 / 100.0
    assert e3 < e2 / 100.0


def test_quadrature_averages_only_the_average_axes(grid, mx, my):
    # the FV velocity pattern Right(x) (x) CellAvg(y): a point value
    # along x, a quadrature average along y
    f = grid.create_field(mx.right * my.cell_avg,
                          init=lambda x, y: x + y**3, order=3)
    xr = (jnp.arange(8) + 1.0) * 0.125
    yedges = jnp.arange(5) * 0.5
    expected = xr[:, None] + _cell_average(3, yedges)[None, :]
    assert jnp.allclose(f.data, expected, atol=1e-14)


def test_quadrature_skips_constant_factors(grid, mx, my):
    # a constant x-factor drops out of the quadrature exactly as it
    # drops out of the collocation signature
    f = grid.create_field(mx.constant * my.cell_avg,
                          init=lambda y: y**3, order=3)
    yedges = jnp.arange(5) * 0.5
    assert f.shape == (1, 4)
    assert jnp.allclose(f.data, _cell_average(3, yedges)[None, :],
                        atol=1e-14)


def test_quadrature_order_is_a_noop_without_average_factors(grid):
    # order= only bites average factors; a pure-nodal space samples
    # its nodes whatever the order (bitwise the collocation default)
    plain = grid.create_field(init=lambda x, y: x**2 + 3.0 * y)
    high = grid.create_field(init=lambda x, y: x**2 + 3.0 * y,
                             order=4)
    assert jnp.array_equal(high.data, plain.data)


def test_quadrature_on_a_stretched_bounded_axis(mzm):
    # per-cell Gauss nodes scale with each cell's own physical width
    # (from the mapped edges): order 2 is exact for the cubic average
    grid = Grid((mzm,))
    edges = tanh_map(_s_faces())
    cubic = grid.create_field(mzm.cell_avg, init=lambda z: z**3,
                              order=2)
    assert jnp.allclose(cubic.data, _cell_average(3, edges),
                        atol=1e-14)


def test_quadrature_face_avg_is_guarded(grid1d, mx):
    # FaceAvg dual-cell quadrature is designed-for, not built; the
    # midpoint default still works, only order>=2 is refused
    with pytest.raises(NotImplementedError, match="FaceAvg"):
        grid1d.create_field(mx.face_avg, init=lambda x: x**2,
                            order=2)


def test_quadrature_rides_through_to_an_average_origin(mx):
    # discretize = transform o discretize-on-origin: order= reaches
    # the CellAvg origin discretize before the Fourier forward. Pin to
    # one device: the host-side Fourier forward needs a local transform
    # axis; unmarked, holds at any device count.
    grid1d = Grid((mx,), device_ids=(0,))
    coeff = mx.fourier(origin=mx.cell_avg)

    def init(x):
        return jnp.exp(jnp.sin(2.0 * jnp.pi * x))

    f = grid1d.create_field(coeff, init=init, order=4)
    transform = grid1d.dispatch.resolve("transform", mx.cell_avg)
    reference = transform.forward(grid1d.create_field(
        mx.cell_avg, init=init, order=4))
    assert jnp.allclose(f.data, reference.data, atol=1e-14)
    # and it differs from the midpoint default (a real high-order lift)
    default = grid1d.create_field(coeff, init=init)
    assert not jnp.allclose(f.data, default.data, atol=1e-8)


# ================================================================
#  evaluation_nodes
# ================================================================
def test_nodes_center(grid1d, mx):
    nodes = grid1d.evaluation_nodes(mx.center)
    assert nodes.function_space.bare is mx.center
    assert jnp.allclose(nodes.data,
                        (jnp.arange(8) + 0.5) * 0.125)
    assert nodes.name == "x"


@pytest.mark.parametrize(("attr", "expected"), [
    pytest.param("left", jnp.arange(8) * 0.125, id="left"),
    pytest.param("right", (jnp.arange(8) + 1) * 0.125, id="right"),
    pytest.param("cell_avg", (jnp.arange(8) + 0.5) * 0.125,
                 id="cell_avg-midpoints"),
    pytest.param("face_avg", (jnp.arange(8) + 1) * 0.125,
                 id="face_avg-periodic-faces"),
])
def test_nodes_periodic_families(grid1d, mx, attr, expected):
    nodes = grid1d.evaluation_nodes(getattr(mx, attr))
    assert jnp.allclose(nodes.data, expected)


@pytest.mark.parametrize(("attr", "expected"), [
    pytest.param("outer", jnp.arange(5) * 0.5, id="outer"),
    pytest.param("inner", (jnp.arange(3) + 1) * 0.5, id="inner"),
    pytest.param("face_avg", (jnp.arange(3) + 1) * 0.5,
                 id="face_avg-bounded-inner-faces"),
])
def test_nodes_bounded_families(my, attr, expected):
    grid = Grid((my,))
    nodes = grid.evaluation_nodes(getattr(my, attr))
    assert jnp.allclose(nodes.data, expected)


def test_nodes_drop_bc_constrained_boundary_dofs(my):
    grid = Grid((my,))
    space = my.nodal(NodeSet.OUTER, bc=BC.DIRICHLET)
    nodes = grid.evaluation_nodes(space)
    assert space.shape == (3,)
    assert jnp.allclose(nodes.data, (jnp.arange(3) + 1) * 0.5)


def test_nodes_on_product_replace_other_factors_by_constant(
        grid, mx, my):
    space = mx.center * my.center
    x = grid.evaluation_nodes(space, "x")
    assert x.shape == (8, 1)
    assert x.function_space.factor("y") is my.constant
    assert x.function_space.factor("x") is mx.center
    y = grid.evaluation_nodes(space, "y")
    assert y.shape == (1, 4)
    assert y.function_space.factor("x") is mx.constant


def test_nodes_broadcast_under_the_strict_algebra(grid):
    f = grid.create_field(init=lambda x, y: x + 0.0 * y)
    x = grid.evaluation_nodes(f.function_space, "x")
    h = f - x
    assert jnp.allclose(h.data, jnp.zeros((8, 4)))


def test_nodes_name_required_when_ambiguous(grid, mx, my):
    with pytest.raises(ValueError, match="ambiguous"):
        grid.evaluation_nodes(mx.center * my.center)


def test_nodes_unambiguous_with_constant_factor(grid, mx, my):
    nodes = grid.evaluation_nodes(mx.constant * my.center)
    assert nodes.name == "y"
    assert nodes.shape == (1, 4)


def test_nodes_unknown_name_raises(grid, mx, my):
    with pytest.raises(KeyError, match="no factor"):
        grid.evaluation_nodes(mx.center * my.center, "z")


def test_nodes_constant_factor_has_none(grid1d, mx):
    with pytest.raises(ValueError, match="constant"):
        grid1d.evaluation_nodes(mx.constant, "x")


def test_nodes_coefficient_factor_raises(grid1d, mx):
    with pytest.raises(ValueError, match="wavenumbers"):
        grid1d.evaluation_nodes(mx.fourier(origin=mx.center))


def test_nodes_on_chebyshev_are_ascending_gauss_lobatto():
    cheb = ChebyshevMesh(8, (0.0, 1.0), name="s")
    grid = Grid((cheb,))
    nodes = grid.evaluation_nodes(cheb.outer)
    expected = 0.5 * (1.0 - jnp.cos(jnp.pi * jnp.arange(9) / 8))
    assert jnp.allclose(nodes.data, expected)


def test_nodes_and_measure_on_point_mesh_raise():
    pm = PointMesh(((0.0,), (1.0,)), name="q")
    grid = Grid((pm,))
    with pytest.raises(NotImplementedError, match="structured 1D"):
        grid.evaluation_nodes(pm.points)
    with pytest.raises(NotImplementedError, match="structured 1D"):
        grid.measure(pm.points)


def test_wavenumbers_fourier(grid1d, mx):
    coeff = mx.fourier(origin=mx.center)
    k = grid1d.wavenumbers(coeff)
    assert k.function_space.bare is coeff
    assert k.name == "kx"
    assert jnp.allclose(k.data.real, 2 * jnp.pi * jnp.arange(5))
    full = grid1d.wavenumbers(
        mx.fourier(origin=mx.center.as_complex()))
    assert jnp.allclose(full.data.real,
                        2 * jnp.pi * jnp.fft.fftfreq(8, 1.0 / 8))


def test_wavenumbers_trig_and_product(grid, mx, my):
    dirichlet = my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    sine = my.sine(dirichlet)
    space = mx.center * sine
    k = grid.wavenumbers(space, name="y")
    # sine modes k = 1..n on L = 2: pi k / L; x replaced by constant
    assert jnp.allclose(
        k.data, (jnp.pi / 2.0) * jnp.arange(1, 5).reshape(1, 4))
    assert k.function_space.bare.factor("y") is sine
    neumann = my.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    cos = grid.wavenumbers(my.cosine(neumann))
    assert jnp.allclose(cos.data, (jnp.pi / 2.0) * jnp.arange(4))


def test_wavenumbers_chebyshev_mode_indices():
    mz = ChebyshevMesh(8, (0.0, 1.0), name="z")
    grid = Grid((mz,))
    modes = grid.wavenumbers(mz.chebyshev(mz.lobatto))
    assert jnp.allclose(modes.data, jnp.arange(9))


def test_wavenumbers_rejects_non_coefficient_factors(grid1d, mx):
    with pytest.raises(ValueError, match="no wavenumbers"):
        grid1d.wavenumbers(mx.center)
    with pytest.raises(ValueError, match="constant"):
        grid1d.wavenumbers(mx.constant, name="x")


def test_registry_halo_derivation_skips_unusable_entries(mx):
    class ForeignSpace:
        names = ("z",)

    class WideOp:
        def requirements(self, space):  # noqa: ARG002
            return OperatorRequirements(halo=3)

    class DuckRegistry:
        def items(self):
            yield "kind_only", object()          # no space to size on
            yield ("diff", ForeignSpace()), WideOp()  # foreign name
            yield ("multiply", mx.center), object()   # no requirements

    grid = Grid((mx,), dispatch=DuckRegistry())
    assert grid.decomposition.halo["x"] == 0


# ================================================================
#  Neumann shape decision follow-ups (owner decision 2026-07-07)
# ================================================================
def test_neumann_outer_coordinates_and_measure_keep_all_nodes(my):
    grid = Grid((my,), device_ids=(0,))
    space = my.nodal(NodeSet.OUTER, bc=BC.NEUMANN)
    nodes = grid.evaluation_nodes(space)
    assert nodes.function_space.bare is space
    assert nodes.data.shape == space.shape == (5,)  # n + 1, no drop
    weights = grid.measure(space, name="y")
    assert weights.data.shape == (5,)
    # trapezoid: half cells at both walls, total = extent length
    assert float(weights.data.sum()) == pytest.approx(2.0)


def test_neumann_outer_seeds_the_dct1_transform_row(my):
    grid = Grid((my,), device_ids=(0,))
    outer_neumann = my.nodal(NodeSet.OUTER, bc=BC.NEUMANN)
    cosine = grid.dispatch.resolve("transform", outer_neumann)
    assert type(cosine).__name__ == "Cosine"
    spectral = grid.dispatch.resolve("diff", my.cosine(outer_neumann))
    assert type(spectral).__name__ == "SpectralDerivative"


# ================================================================
#  Mapped meshes: nodes and staggered measure fields (stage C0)
# ================================================================
MN = 8


def tanh_map(s):
    return jnp.tanh(2.0 * s) / jnp.tanh(2.0)


def wavy_map(s):
    return s + 0.1 * jnp.sin(2.0 * jnp.pi * s) / (2.0 * jnp.pi)


@pytest.fixture
def mzm():
    return MappedIntervalMesh(MN, (0.0, 1.0), tanh_map, name="z")


@pytest.fixture
def mpm():
    return MappedIntervalMesh(MN, (0.0, 1.0), wavy_map,
                              periodic=True, name="p")


def _s_centers():
    return (jnp.arange(MN) + 0.5) / MN


def _s_faces():
    return jnp.arange(MN + 1) / MN


@pytest.mark.parametrize(("attr", "s_nodes"), [
    pytest.param("center", _s_centers(), id="center"),
    pytest.param("outer", _s_faces(), id="outer"),
    pytest.param("inner", _s_faces()[1:-1], id="inner"),
    pytest.param("left", _s_faces()[:-1], id="left"),
    pytest.param("right", _s_faces()[1:], id="right"),
    pytest.param("cell_avg", _s_centers(), id="cell_avg"),
    pytest.param("face_avg", _s_faces()[1:-1], id="face_avg"),
])
def test_mapped_nodes_compose_the_mapping(mzm, attr, s_nodes):
    grid = Grid((mzm,))
    nodes = grid.evaluation_nodes(getattr(mzm, attr))
    assert jnp.allclose(nodes.data, tanh_map(s_nodes))


@pytest.mark.parametrize(("attr", "s_nodes"), [
    pytest.param("center", _s_centers(), id="center"),
    pytest.param("left", _s_faces()[:-1], id="left"),
    pytest.param("right", _s_faces()[1:], id="right"),
    pytest.param("face_avg", _s_faces()[1:], id="face_avg"),
])
def test_mapped_periodic_nodes_compose_the_mapping(
        mpm, attr, s_nodes):
    grid = Grid((mpm,))
    nodes = grid.evaluation_nodes(getattr(mpm, attr))
    assert jnp.allclose(nodes.data, wavy_map(s_nodes))


def test_mapped_primal_measure_is_the_cell_width(mzm):
    grid = Grid((mzm,))
    faces = tanh_map(_s_faces())
    for space in (mzm.center, mzm.cell_avg):
        w = grid.measure(space, name="z")
        assert jnp.allclose(w.data, jnp.diff(faces))
        assert float(w.data.sum()) == pytest.approx(1.0)


def test_measure_is_memoized_per_space_and_name(mx, my):
    # measures are static mesh geometry: repeated queries return the
    # SAME field object (stable identity keeps the operator-level
    # sync memo warm); distinct names key distinct entries
    grid = Grid((mx, my))
    space = mx.center * my.center
    wx = grid.measure(space, name="x")
    assert grid.measure(space, name="x") is wx
    wy = grid.measure(space, name="y")
    assert wy is not wx
    assert grid.measure(space, name="y") is wy
    assert jnp.allclose(wx.data, 1.0 / 8)


def test_measure_under_a_trace_is_not_cached(mx, my):
    # a query issued under a jax trace yields that trace's tracer;
    # caching it would leak it into every later query (the eager-
    # operator kernel jit surfaced this: integrate queries the
    # measure inside its jit, then mean's eager normalization query
    # hit the cached tracer). The traced query stays uncached; the
    # first eager query re-derives concrete weights and memoizes.
    grid = Grid((mx, my))
    space = mx.center * my.center

    @jax.jit
    def traced():
        return grid.measure(space, name="x").data.sum()

    traced()
    weight = grid.measure(space, name="x")
    assert not isinstance(weight._data, jax.core.Tracer)
    assert grid.measure(space, name="x") is weight
    assert jnp.allclose(weight.data, 1.0 / 8)


@pytest.mark.multi_device
def test_eager_mean_on_a_cold_multi_device_grid():
    # regression (2026-07-15): on a multi-device operand the eager
    # operator kernel runs under one jax.jit trace, so a cold-cache
    # integral computed its measures inside that trace; mean's
    # eager normalization query then used the leaked tracer and
    # raised UnexpectedTracerError (first seen as the mapped model
    # failing to construct on 4 devices)
    meshes = tuple(
        IntervalMesh(16, (0.0, 1.0), periodic=True, name=n)
        for n in ("x", "y", "z"))
    grid = Grid(meshes)
    field = grid.create_field(data=jnp.ones((16, 16, 16)))
    mean = field.mean()
    assert float(mean.data.ravel()[0]) == pytest.approx(1.0)


def test_mapped_dual_measures_clip_at_the_walls(mzm):
    grid = Grid((mzm,))
    centers = tanh_map(_s_centers())
    interior = jnp.diff(centers)
    outer = grid.measure(mzm.outer, name="z")
    expected = jnp.concatenate([
        centers[:1] - 0.0, interior, 1.0 - centers[-1:]])
    assert jnp.allclose(outer.data, expected)
    assert float(outer.data.sum()) == pytest.approx(1.0)
    inner = grid.measure(mzm.inner, name="z")
    assert jnp.allclose(inner.data, interior)
    assert jnp.allclose(grid.measure(mzm.face_avg, name="z").data,
                        interior)


def test_mapped_bounded_left_right_clip_one_side(mzm):
    grid = Grid((mzm,))
    centers = tanh_map(_s_centers())
    interior = jnp.diff(centers)
    left = grid.measure(mzm.left, name="z")
    assert jnp.allclose(
        left.data, jnp.concatenate([centers[:1] - 0.0, interior]))
    right = grid.measure(mzm.right, name="z")
    assert jnp.allclose(
        right.data,
        jnp.concatenate([interior, 1.0 - centers[-1:]]))


def test_mapped_measures_genuinely_differ(mzm):
    # the primal and dual dx are different fields on a stretched
    # mesh (concepts 2.7); on a uniform mesh both collapse to dx
    grid = Grid((mzm,))
    primal = grid.measure(mzm.center, name="z").data
    dual = grid.measure(mzm.inner, name="z").data
    assert not jnp.allclose(primal[1:], dual)


def test_mapped_periodic_dual_measure_wraps(mpm):
    grid = Grid((mpm,))
    centers = wavy_map(_s_centers())
    wrap = centers[:1] + 1.0 - centers[-1:]
    interior = jnp.diff(centers)
    right = grid.measure(mpm.right, name="p")
    assert jnp.allclose(right.data,
                        jnp.concatenate([interior, wrap]))
    assert float(right.data.sum()) == pytest.approx(1.0)
    left = grid.measure(mpm.left, name="p")
    assert jnp.allclose(left.data,
                        jnp.concatenate([wrap, interior]))
    assert jnp.allclose(grid.measure(mpm.face_avg, name="p").data,
                        right.data)


def test_mapped_measure_drops_bc_constrained_dofs(mzm):
    grid = Grid((mzm,))
    space = mzm.nodal(NodeSet.OUTER, bc=BC.DIRICHLET)
    w = grid.measure(space, name="z")
    assert w.data.shape == space.shape == (MN - 1,)
    centers = tanh_map(_s_centers())
    assert jnp.allclose(w.data, jnp.diff(centers))


def test_measure_on_chebyshev_awaits_clenshaw_curtis():
    cheb = ChebyshevMesh(8, (0.0, 1.0), name="s")
    grid = Grid((cheb,))
    with pytest.raises(NotImplementedError, match="Clenshaw"):
        grid.measure(cheb.outer)


def test_mapped_mesh_seeds_no_transform_rows(mpm, mzm):
    # computational-space bases are deferred (stage C2): neither
    # the periodic Fourier row nor the bounded trig rows seed
    with pytest.raises(DispatchError):
        Grid((mpm,)).dispatch.resolve("transform", mpm.center)
    tagged = mzm.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    with pytest.raises(DispatchError):
        Grid((mzm,)).dispatch.resolve("transform", tagged)


# ================================================================
#  Coordinate mapping: attachment, grid.metric, seeded kind (C1)
# ================================================================
def _depth(x):
    return 1.0 + 0.2 * jnp.sin(x)


@pytest.fixture
def mapped_grid(mx):
    ms = IntervalMesh(8, (0.0, 1.0), name="sigma")
    mapping = CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H},
        params={"H": _depth})
    return Grid((mx, ms), mapping=mapping)


def test_mapping_defaults_to_none(grid):
    assert grid.mapping is None


def test_mapping_attachment(mapped_grid):
    assert mapped_grid.mapping is not None
    assert mapped_grid.mapping.param_names == ("H",)


def test_metric_without_mapping_raises(grid, mx, my):
    with pytest.raises(ValueError, match="no coordinate mapping"):
        grid.metric(mx.center * my.center, "dz_dsigma")


def test_metric_delegates_to_the_mapping(mapped_grid):
    mx, ms = mapped_grid.factors
    space = mx.center * ms.center
    metric = mapped_grid.metric(space, "dz_dsigma")
    x = mapped_grid.evaluation_nodes(space, "x").data
    assert jnp.allclose(metric.data, _depth(x))
    assert metric.function_space.bare is space.bare


def test_metric_params_overload_threads_through(mapped_grid):
    mx, ms = mapped_grid.factors
    space = mx.center * ms.center
    h = mapped_grid.create_field(
        mx.center, init=lambda x: 2.0 + 0.0 * x)
    metric = mapped_grid.metric(space, "dz_dsigma",
                                params={"H": h})
    assert jnp.allclose(metric.data, 2.0)


def test_mapping_seeds_the_physical_diff_row(mapped_grid):
    mx, ms = mapped_grid.factors
    op = mapped_grid.dispatch.resolve("physical_diff",
                                      mx.center * ms.center)
    assert isinstance(op, MappedDerivative)
    assert op.corrections == {"sigma": ("z", "sigma"),
                              "x": ("z", "sigma")}


def test_no_mapping_seeds_no_physical_diff_row(grid, mx, my):
    with pytest.raises(DispatchError, match="physical_diff"):
        grid.dispatch.resolve("physical_diff",
                              mx.center * my.center)


def test_chart_only_mapping_seeds_no_physical_diff_row(mx, my):
    mapping = CoordinateMapping(chart={
        "X": lambda x, y: (jnp.cos(x), jnp.sin(x), y)})
    grid = Grid((mx, my), mapping=mapping)
    assert grid.mapping is mapping
    with pytest.raises(DispatchError, match="physical_diff"):
        grid.dispatch.resolve("physical_diff",
                              mx.center * my.center)


def test_bounded_mapped_column_seeds_the_one_sided_closure(mx):
    # the near-wall closure of the correction chains (stage C4):
    # a BOUNDED mapped column opens the BC-free Inner -> Center
    # interpolation hop with the explicit one-sided variant
    ms = IntervalMesh(8, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"z": lambda sigma, H: sigma * H},
        params={"H": _depth})
    grid = Grid((mx, ms), mapping=mapping)
    op = grid.dispatch.resolve("interpolate", ms.inner)["sigma"]
    assert op.boundary == "one_sided"
    # the coupled (periodic) coordinate is untouched
    assert grid.dispatch.resolve(
        "interpolate", mx.center)["x"].boundary == "closed"


def test_periodic_mapped_column_seeds_no_closure(mapped_grid):
    # a periodic column has no wall to close; nothing extra seeds
    _, ms = mapped_grid.factors
    with pytest.raises(ValueError, match="bounded"):
        ms.inner  # noqa: B018 — the property itself raises


def test_flat_bounded_column_keeps_the_closed_legality(mx, mzm):
    # without a mapping the BC-free Inner -> Center hop stays
    # closed (boundary_plan 2c): zero behavior change on flat grids
    grid = Grid((mx, mzm))
    with pytest.raises(DispatchError):
        grid.dispatch.resolve("interpolate", mzm.inner)


# ================================================================
#  Chart seeding: the metric-aware vector calculus (stage C2)
# ================================================================
def _chart_grid(mx, my):
    mapping = CoordinateMapping(chart={
        "X": lambda x, y: (jnp.cos(x), jnp.sin(x), y)})
    return Grid((mx, my), mapping=mapping)


def test_chart_mapping_seeds_the_metric_calculus_rows(mx, my):
    grid = _chart_grid(mx, my)
    space = mx.center * my.center
    registry = grid.dispatch
    assert isinstance(registry.resolve("grad", space),
                      MetricGradient)
    assert isinstance(registry.resolve("div", space),
                      MetricDivergence)
    assert isinstance(registry.resolve("curl", space), MetricCurl)
    assert isinstance(registry.resolve("laplacian", space),
                      MetricLaplacian)
    assert isinstance(registry.resolve("raise_index", space),
                      RaiseIndex)
    assert isinstance(registry.resolve("lower_index", space),
                      LowerIndex)
    for row in ("grad", "div", "curl", "laplacian",
                "raise_index", "lower_index"):
        assert registry.resolve(row, space).coords == ("x", "y")


def test_chart_index_moves_default_to_the_dense_expansion(mx, my):
    # the default keeps the full cross-term expansion (correct-or-loud
    # on a non-orthogonal chart); orthogonal= is opt-in
    registry = _chart_grid(mx, my).dispatch
    space = mx.center * my.center
    for kind in ("raise_index", "lower_index"):
        assert registry.resolve(kind, space).diagonal is False


def test_orthogonal_chart_seeds_diagonal_index_moves(mx, my):
    # orthogonal=True drops the off-diagonal contractions so an index
    # move assembles across a bounded chart axis (chart-ergonomics E2)
    mapping = CoordinateMapping(chart={
        "X": lambda x, y: (jnp.cos(x), jnp.sin(x), y)},
        orthogonal=True)
    registry = Grid((mx, my), mapping=mapping).dispatch
    space = mx.center * my.center
    for kind in ("raise_index", "lower_index"):
        assert registry.resolve(kind, space).diagonal is True


def test_chart_mapping_seeds_the_jacobian_integrate_rows(mx, my):
    grid = _chart_grid(mx, my)
    row = grid.dispatch.resolve("integrate", mx.center)
    assert isinstance(row, Integral)
    assert row.jacobian == ("x", "y")


def test_chartless_grids_keep_the_flat_builders(grid, mx, my):
    space = mx.center * my.center
    registry = grid.dispatch
    assert not isinstance(registry.resolve("grad", space),
                          MetricGradient)
    assert not isinstance(registry.resolve("laplacian", space),
                          MetricLaplacian)
    assert registry.resolve("integrate", mx.center).jacobian is None
    for kind in ("raise_index", "lower_index"):
        with pytest.raises(DispatchError, match=kind):
            registry.resolve(kind, space)


def test_map_only_mappings_seed_no_metric_calculus(mapped_grid):
    mx, ms = mapped_grid.factors
    space = mx.center * ms.center
    registry = mapped_grid.dispatch
    assert not isinstance(registry.resolve("grad", space),
                          MetricGradient)
    # a single-base maps= grid derives a volume element, so the seeded
    # reductions ARE Jacobian-weighted (the physical-integral-default
    # flip) -- the mapped physical name "z" -- even though it seeds no
    # embedding-chart vector calculus
    assert registry.resolve("integrate", mx.center).jacobian == ("z",)
    assert registry.resolve("cumint", mx.center).jacobian == ("z",)
    with pytest.raises(DispatchError, match="raise_index"):
        registry.resolve("raise_index", space)


def test_one_coordinate_charts_seed_the_jacobian_only(mx):
    # a curve chart has arc-length measure but no vector calculus
    mapping = CoordinateMapping(chart={
        "X": lambda x: (jnp.cos(x), jnp.sin(x))})
    grid = Grid((mx,), mapping=mapping)
    assert grid.dispatch.resolve(
        "integrate", mx.center).jacobian == ("x",)
    assert not isinstance(grid.dispatch.resolve("grad", mx.center),
                          MetricGradient)
    with pytest.raises(DispatchError, match="raise_index"):
        grid.dispatch.resolve("raise_index", mx.center)


def test_multi_base_maps_keep_the_computational_reductions(mx):
    # a multi-base analytic map derives Jacobian metrics but no
    # single-base column volume element, so the seeded reductions stay
    # computational (jacobian=None): _reduction_jacobian's empty branch
    ms = IntervalMesh(8, (0.0, 1.0), name="sigma")
    mapping = CoordinateMapping(
        maps={"z": lambda sigma, x: sigma * (1.0 + 0.2 * x)})
    grid = Grid((mx, ms), mapping=mapping)
    assert grid.dispatch.resolve(
        "integrate", mx.center).jacobian is None
    assert grid.dispatch.resolve(
        "cumint", mx.center).jacobian is None


# ================================================================
#  Grid.coarsened (multigrid / regrid sibling grids, A2)
# ================================================================
def test_coarsened_uniform_factor_halves_every_axis(grid):
    coarse = grid.coarsened(2)
    assert coarse.names == grid.names
    assert tuple(m.n_cells for m in coarse.factors) == (4, 2)
    # a fresh, unfrozen grid with its own decomposition (provisional
    # negotiation done, no fingerprint sealed)
    assert coarse is not grid
    assert coarse.fingerprint is None
    assert coarse.decomposition is not grid.decomposition


def test_coarsened_per_name_factors_are_semicoarsening(grid):
    coarse = grid.coarsened({"x": 2})
    # x halved, y (missing -> factor 1) at full resolution (MG-D4)
    assert tuple(m.n_cells for m in coarse.factors) == (4, 4)
    # the pass-through axis reuses the fine mesh object
    assert coarse.factors[1] is grid.factors[1]


def test_coarsened_carries_the_family(mx):
    grid = Grid((mx,), family="fv")
    assert grid.coarsened(2).default_family == "fv"


def test_coarsened_seeds_a_fresh_default_registry(grid):
    coarse = grid.coarsened(2)
    # a working registry keyed on the coarse spaces
    assert coarse.dispatch is not grid.dispatch
    field = coarse.create_field()
    assert field.diff("x").function_space is not None


def test_coarsened_inherits_the_device_set(grid):
    coarse = grid.coarsened(2)
    assert (coarse.decomposition.device_count
            == grid.decomposition.device_count)


def test_coarsened_rejects_unknown_name(grid):
    with pytest.raises(ValueError, match="unknown coordinate"):
        grid.coarsened({"q": 2})


@pytest.mark.parametrize("bad", [0, -1, 2.0, True])
def test_coarsened_rejects_bad_factor(grid, bad):
    with pytest.raises((ValueError, TypeError)):
        grid.coarsened({"x": bad})


def test_coarsened_rejects_non_mapping_non_int(grid):
    with pytest.raises(TypeError, match="uniform"):
        grid.coarsened([2])


def test_coarsened_rebinds_the_coordinate_mapping(mx):
    mapping = CoordinateMapping(
        maps={"z": lambda x, H: x * H},
        params={"H": lambda x: 1.0 + 0.0 * x})
    grid = Grid((mx,), mapping=mapping)
    coarse = grid.coarsened(2)
    assert coarse.mapping is not None
    assert coarse.mapping is not grid.mapping
    # the clone is re-bound to the coarse grid and re-derives metrics
    metric = coarse.metric(coarse.factors[0].center, "dz_dx")
    assert metric.data.shape == (4,)


def test_coarsened_rebinds_the_immersed_domain(mx, my):
    immersed = ImmersedDomain(
        lambda x, y: ((x < 0.6) & (y >= 0.0)).astype(float))
    grid = Grid((mx, my), immersed=immersed)
    coarse = grid.coarsened(2)
    assert coarse.immersed is not None
    assert coarse.immersed is not grid.immersed
    # fresh cache + re-bind: fractions re-derive on the coarse spaces
    space = coarse.factors[0].center * coarse.factors[1].center
    frac = coarse.immersed.fraction(space)
    assert frac.data.shape == (4, 2)


def test_coarsened_does_not_carry_dispatch_overrides(mx):
    grid = Grid((mx,))
    op = grid.dispatch.resolve("diff", mx.center)
    grid.merge_overrides({("diff", mx.center): op})
    coarse = grid.coarsened(2)
    # the coarse grid is seeded from a fresh default registry (model
    # overrides do not carry); it resolves defaults on its own spaces
    assert coarse.dispatch is not grid.dispatch
    assert coarse.dispatch.resolve(
        "diff", coarse.factors[0].center) is not None


def test_coarsened_memoizes_per_factors_and_devices(grid):
    # structure caching (MG-D3/D5): identical arguments return the
    # identical coarse Grid object, so a hierarchy rebuilt every solver
    # trace re-uses one stable grid identity per level
    a = grid.coarsened({"x": 2})
    b = grid.coarsened({"x": 2})
    assert a is b
    # the uniform-int and per-name forms that normalize to the SAME
    # factor map share the memo entry (x:2, y:1 both ways would differ,
    # so use a form that truly matches: {"x": 2} vs {"x": 2, "y": 1})
    assert grid.coarsened({"x": 2, "y": 1}) is a
    # a different factor map is a different object
    assert grid.coarsened(2) is not a
    # an explicit device_ids key is distinct from the inherited one
    assert grid.coarsened({"x": 2}, device_ids=(0,)) is not a


def test_override_keys_records_merged_overrides(mx):
    grid = Grid((mx,))
    assert grid.override_keys == frozenset()
    op = grid.dispatch.resolve("diff", mx.center)
    grid.merge_overrides({("diff", mx.center): op})
    assert ("diff", mx.center) in grid.override_keys
    # a coarse sibling starts with an empty record (overrides do not
    # carry, so a re-discretizing caller merges its own profile once)
    assert grid.coarsened(2).override_keys == frozenset()
