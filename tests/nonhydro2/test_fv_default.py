"""The FV C-grid default and the flip (stage F3).

The acceptance gate is **bitwise parity**: a linear and a nonlinear
periodic nonhydro run, FV default vs the same model with
``family="nodal"``, produce bit-identical prognostic trajectories
(exact array equality on every prognostic after N >= 10 steps) — safe
because the 2nd-order FV and nodal C-grid stencils are the same numbers
(scoping study §1). The rest of the file pins the model-assembly family
API: the auto flip, the taught error on walled / mapped grids, the FV
C-grid diff profile, and the frozen-grid override verification.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.assembly import _collect_dispatch_overrides
from fridom.model.errors import AssemblyError
from fridom.model.model import Model as FrModel
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.model.modules.moving_geometry import (
    MeshVelocityCorrection,
    MovingGeometry,
)
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.core import (
    DynamicalCore,
    _fv_capable,
    fv_cgrid_overrides,
    resolve_model_family,
)
from fridom.nonhydro2.modules.stratification import ConstantStratification
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.flux_diff import (
    FaceDifference,
    FluxDifference,
)
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.nodal import NodalSpace

N = 8
DT = 0.02
LENGTH = 2 * np.pi


def periodic_grid(n=N):
    return Grid(tuple(
        IntervalMesh(n, (0.0, LENGTH), periodic=True, name=name)
        for name in ("x", "y", "z")))


def walled_grid(n=N):
    return Grid((
        IntervalMesh(n, (0.0, LENGTH), periodic=True, name="x"),
        IntervalMesh(n, (0.0, LENGTH), periodic=True, name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="z"),
    ))


def mapped_grid(n=N):
    # jnp (not np) in the param callable: the FV/nodal mapped model now
    # traces the projection's metric derivation, so the depth profile is
    # a compute callable (an np.sin would raise under jit)
    mapping = CoordinateMapping(
        maps={"zp": lambda z, h: z * h},
        params={"h": lambda x: 1.0 + 0.2 * jnp.sin(x)})
    return Grid((
        IntervalMesh(n, (0.0, LENGTH), periodic=True, name="x"),
        IntervalMesh(n, (0.0, LENGTH), periodic=True, name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="z"),
    ), mapping=mapping)


def coords(n=N, length=LENGTH):
    ax = (np.arange(n) + 0.5) * (length / n)
    return np.meshgrid(ax, ax, ax, indexing="ij")


# ================================================================
#  The bitwise-parity gate (the verbatim F3 acceptance test)
# ================================================================
def _seed(model):
    x, y, z = coords()
    model.set_fields(
        u=np.sin(x) * np.cos(y), v=0.3 * np.cos(x) * np.sin(z),
        w=0.2 * np.sin(z) * np.cos(y), b=0.1 * np.cos(x) * np.cos(z))
    return model


@pytest.mark.parametrize(
    "advection", [pytest.param(False, id="linear"),
                  pytest.param(True, id="nonlinear")])
def test_fv_default_is_bitwise_identical_to_nodal(advection):
    steps = 12  # >= 10
    fv = _seed(nh.Model(coriolis=FPlaneCoriolis(f0=1.0),
                        grid=periodic_grid(), dt=DT, dsqr=2.0,
                        rossby_number=1.0, advection=advection))
    nodal = _seed(nh.Model(coriolis=FPlaneCoriolis(f0=1.0),
                           grid=periodic_grid(), dt=DT, dsqr=2.0,
                           rossby_number=1.0, advection=advection,
                           family="nodal"))
    fv.advance(steps)
    nodal.advance(steps)
    for c in ("u", "v", "w", "b", "p"):
        np.testing.assert_array_equal(
            np.asarray(fv.state[c].data), np.asarray(nodal.state[c].data),
            err_msg=f"FV vs nodal diverged on {c!r}")


BIASED = [
    pytest.param(lambda: nh.UpwindAdvection(3), id="upwind3"),
    pytest.param(lambda: nh.UpwindAdvection(5), id="upwind5"),
    pytest.param(lambda: nh.WENOAdvection(3), id="weno3"),
    pytest.param(lambda: nh.WENOAdvection(5), id="weno5"),
]


@pytest.mark.parametrize("advection", BIASED)
def test_fv_default_biased_is_bitwise_identical_to_nodal(advection):
    # the F3-default-flip blocker fix (side-finding upwind5_revisit.md
    # §7): biased (upwind / WENO) advection on the FV default is served
    # and stays BITWISE the nodal trajectory on a periodic box, exactly
    # as centered does above. Velocity self-advection is face-normal on
    # the FV C-grid (u on Right ⊗ CellAvg^2), so its own axis crosses
    # nodal -> average through the 2nd-order-identity deconvolve — the
    # numbers are preserved to the bit whatever the order.
    steps = 12  # >= 10
    fv = _seed(nh.Model(coriolis=FPlaneCoriolis(f0=1.0),
                        grid=periodic_grid(), dt=DT, dsqr=2.0,
                        rossby_number=1.0, advection=advection()))
    nodal = _seed(nh.Model(coriolis=FPlaneCoriolis(f0=1.0),
                           grid=periodic_grid(), dt=DT, dsqr=2.0,
                           rossby_number=1.0, advection=advection(),
                           family="nodal"))
    fv.advance(steps)
    nodal.advance(steps)
    for c in ("u", "v", "w", "b", "p"):
        np.testing.assert_array_equal(
            np.asarray(fv.state[c].data), np.asarray(nodal.state[c].data),
            err_msg=f"FV vs nodal biased diverged on {c!r}")


def test_fv_default_state_is_finite_volume():
    model = nh.Model(coriolis=FPlaneCoriolis(f0=1.0),
                     grid=periodic_grid(), dt=DT)
    # scalars on CellAvg^3, velocities face-normal (Right ⊗ CellAvg^2)
    for c in ("p", "b"):
        assert all(isinstance(f, CellAvg)
                   for f in model.state[c].function_space.bare.factors)
    u = model.state["u"].function_space.bare
    assert isinstance(u.factor("x"), NodalSpace)  # the normal face
    assert isinstance(u.factor("y"), CellAvg)     # transverse average
    assert isinstance(u.factor("z"), CellAvg)


def test_projection_drives_divergence_to_machine_zero_on_fv():
    model = nh.Model(coriolis=FPlaneCoriolis(f0=1.0),
                     grid=periodic_grid(), dt=DT, advection=False)
    x, y, z = coords()
    model.set_fields(u=np.sin(x) * np.cos(y), v=0.3 * np.cos(x),
                     w=0.2 * np.sin(z))
    model.advance(1)
    vel = fr.spatial.fields.vector_field.VectorField(
        {c: model.state[c] for c in ("u", "v", "w")})
    div = np.asarray(fr.spatial.operators.composed.Divergence()(vel).data)
    assert np.abs(div).max() < 1e-12


# ================================================================
#  The auto flip and the family resolution
# ================================================================
def test_auto_flips_every_unimmersed_grid_to_fv():
    # owner ruling 2026-07-17: the auto default is FV on every grid it
    # can carry — periodic, walled, mapped terrain-following (static
    # AND dynamically driven, since the ALE-on-FV closure —
    # test_dynamic_mapping_auto_flips_to_fv_with_family_aware_ale). An
    # immersed grid also auto-flips (test_immersed_grid_is_fv_capable).
    assert resolve_model_family(None, periodic_grid()) == "fv"
    assert resolve_model_family(None, walled_grid()) == "fv"
    assert resolve_model_family(None, mapped_grid()) == "fv"


def test_explicit_family_is_honored():
    assert resolve_model_family("nodal", periodic_grid()) == "nodal"
    assert resolve_model_family("fv", periodic_grid()) == "fv"


def test_grid_default_family_promotes_and_defers():
    # None follows the grid's own default; an FV grid stays FV, and a
    # nodal grid is promoted to FV when it can carry it
    fv_grid = Grid(tuple(
        IntervalMesh(N, (0.0, LENGTH), periodic=True, name=n)
        for n in ("x", "y", "z")), family="fv")
    assert resolve_model_family(None, fv_grid) == "fv"


def test_invalid_family_is_rejected():
    with pytest.raises(ValueError, match="one of"):
        resolve_model_family("spectral", periodic_grid())
    with pytest.raises(ValueError, match="one of"):
        DynamicalCore(family="bogus")


def test_fv_on_mapped_grid_is_served_explicit_and_auto():
    # F5 + the 2026-07-17 mapped auto flip: family="fv" on a mapped
    # terrain-following grid is served — the projection routes to the
    # family-aware MappedPressureSolver and a CellAvg tracer transports
    # in conservative flux form. Since the flip the AUTO default (no
    # family=) resolves to FV on a static mapped grid too, so an
    # explicitly-fv and an auto-default mapped model land b on the same
    # CellAvg^3 family.
    assert resolve_model_family("fv", mapped_grid()) == "fv"
    explicit = nh.Model(coriolis=FPlaneCoriolis(f0=1.0),
                        grid=mapped_grid(), dt=DT, advection=False,
                        family="fv")
    auto = nh.Model(coriolis=FPlaneCoriolis(f0=1.0), grid=mapped_grid(),
                    dt=DT, advection=False)  # auto -> fv
    for model in (explicit, auto):
        assert all(isinstance(f, CellAvg)
                   for f in model.state["b"].function_space.bare.factors)
    # the auto mapped-FV model runs a full projected step end to end.
    # the walled-z column staggers w onto n-1 faces, so seed via init
    # callables (evaluated on each field's own nodes) not a flat array
    auto.set_fields(
        u=lambda x, y, z: jnp.sin(x) * jnp.cos(y),  # noqa: ARG005
        v=lambda x, y, z: 0.3 * jnp.cos(x),  # noqa: ARG005
        w=lambda x, y, z: 0.2 * jnp.sin(jnp.pi * z))  # noqa: ARG005
    auto.advance(1)
    assert not auto.panicked


def test_explicit_fv_on_walled_grid_is_now_served():
    # F4: explicit family="fv" on a walled (bounded, unmapped,
    # unimmersed) grid is served -- the pressure DCT-II runs on the
    # Neumann CellAvg origin. Since the 2026-07-16 owner ruling the
    # AUTO default also resolves FV on a walled grid (tested above);
    # this pins that an explicit "fv" agrees.
    assert resolve_model_family("fv", walled_grid()) == "fv"
    model = nh.Model(coriolis=FPlaneCoriolis(f0=1.0), grid=walled_grid(),
                     dt=DT, advection=False, family="fv")
    assert all(isinstance(f, CellAvg)
               for f in model.state["b"].function_space.bare.factors)


def test_walled_fv_eigenmodes_build():
    # since stage F5 the analytic walled-vertical eigenmode kit IS
    # wired for the FV family: the kit mints its BC-tagged CellAvg
    # analysis siblings itself (C8 keeps the *declaration* layer
    # BC-free — no Collocated(wall_bc=..., family="fv") pattern), so
    # from_model on the auto-FV walled model builds. The FV eigenbasis
    # is bitwise the nodal one; the parity + round-trip battery lives
    # in test_walled_eigenmodes.py.
    from fridom.spatial.bc import BC  # noqa: PLC0415
    model = nh.Model(coriolis=FPlaneCoriolis(f0=1.0), grid=walled_grid(),
                     dt=DT, advection=False)  # auto -> fv
    assert all(isinstance(f, CellAvg)
               for f in model.state["b"].function_space.bare.factors)
    em = nh.eigenmodes.from_model(model)
    # it really resolved the FV family: b's vertical analysis origin is
    # a (Dirichlet-tagged) CellAvg cell average, not a nodal center
    origin = em.kit.coeff("b").factor("z").origin
    assert isinstance(origin, CellAvg)
    assert all(c is BC.DIRICHLET for c in origin.bc.components)


def test_fv_capable_flags():
    # since the ALE-on-FV closure (2026-07-17) the auto predicate is
    # True on EVERY grid — periodic, walled, static mapped, immersed,
    # AND a dynamically driven mapping (the ALE mesh-velocity
    # correction is now family-aware, so moving geometry runs on FV
    # wherever a static mapped one does). The one unserved composition
    # (mapped + immersed) is a taught error in _require_fv_capable.
    assert _fv_capable(periodic_grid())
    assert _fv_capable(walled_grid())
    assert _fv_capable(mapped_grid())


def test_immersed_grid_is_fv_capable_and_auto_flips():
    # IP-D7: an immersed grid IS finite-volume-capable (immersed
    # physics is FV — the fractions are volume/area weights). It
    # auto-flips to FV, serves explicit family="fv", and rejects an
    # explicit family="nodal" as a taught error (the nodal path
    # ignores the mask).
    grid = periodic_grid().with_immersed(
        ImmersedDomain(lambda x, y, z: (x + y + z) * 0.0 + 1.0))
    assert _fv_capable(grid)
    assert resolve_model_family(None, grid) == "fv"
    assert resolve_model_family("fv", grid) == "fv"
    with pytest.raises(NotImplementedError, match="finite-volume"):
        resolve_model_family("nodal", grid)


def test_dynamic_mapping_auto_flips_to_fv_with_family_aware_ale():
    # the ALE-on-FV closure (2026-07-17): a mapping driven in time (a
    # MovingGeometry module) is now FV-capable — the ALE mesh-velocity
    # correction is family-aware — so the auto default flips a
    # moving-geometry model to FV, exactly like a static mapped one.
    assert resolve_model_family(None, mapped_grid()) == "fv"

    # the nh.Model factory auto-resolves a moving-geometry model to FV:
    # b lands on CellAvg^3 (the cell average), the same family a static
    # mapped model gets
    auto = nh.Model(
        coriolis=FPlaneCoriolis(f0=1.0), grid=mapped_grid(), dt=DT,
        advection=False,
        modules_extra=(MovingGeometry(
            {"h": lambda x, t: 1.0 + 0.2 * jnp.sin(x) + 0.0 * t}),))
    assert all(
        isinstance(f, CellAvg)
        for f in auto.state["b"].function_space.bare.factors)


def test_explicit_fv_with_ale_binds_and_runs():
    # an explicit family="fv" WITH the ALE module binds (no taught
    # error any more — the family-aware routing handles the CellAvg
    # column factors) and advances a projected step end to end. The
    # walled-z column staggers w onto n-1 faces, so seed w via an init
    # callable evaluated on its own nodes.
    model = nh.Model(
        coriolis=FPlaneCoriolis(f0=1.0), grid=mapped_grid(), dt=DT,
        advection=False, family="fv",
        modules_extra=(
            MovingGeometry(
                {"h": lambda x, t: 1.0 + 0.2 * jnp.sin(x)
                 + 0.05 * t}),
            MeshVelocityCorrection()))
    assert all(
        isinstance(f, CellAvg)
        for f in model.state["b"].function_space.bare.factors)
    model.set_fields(
        b=lambda x, y, z: 0.01 * jnp.cos(jnp.pi * z),  # noqa: ARG005
        w=lambda x, y, z: 0.1 * jnp.sin(jnp.pi * z))  # noqa: ARG005
    model.advance(2)
    assert not model.panicked


def test_dispatch_hook_colliding_with_static_dispatch_is_rejected():
    # a module whose static ``dispatch`` and grid-aware
    # ``grid_dispatch_overrides`` contribute the same resolved key is a
    # taught error (the intra-module collision guard)
    grid = periodic_grid()
    key = ("diff", grid.factors[0].cell_avg)

    class _Colliding:
        dispatch = {key: FaceDifference()}  # noqa: RUF012

        def grid_dispatch_overrides(self, grid):  # noqa: ARG002
            return {key: FaceDifference()}

    with pytest.raises(AssemblyError, match="both"):
        _collect_dispatch_overrides((_Colliding(),), grid)


# ================================================================
#  The FV C-grid diff profile (FV-D3)
# ================================================================
def test_fv_cgrid_overrides_repoints_the_diff_rows():
    meshes = periodic_grid().factors
    overrides = fv_cgrid_overrides(meshes)
    for mesh in meshes:
        assert isinstance(overrides[("diff", mesh.cell_avg)],
                          FaceDifference)
        assert isinstance(overrides[("diff", mesh.right)],
                          FluxDifference)


def test_fv_cgrid_overrides_repoints_the_walled_diff_rows():
    # F4: on a walled mesh factor the profile re-points the BC-free
    # CellAvg / Inner faces AND the tagged pressure (Neumann CellAvg)
    # and velocity (Dirichlet Inner) origins
    from fridom.spatial.bc import BC  # noqa: PLC0415
    from fridom.spatial.spaces.average import CellAvg  # noqa: PLC0415
    from fridom.spatial.spaces.nodal import NodeSet  # noqa: PLC0415
    meshes = walled_grid().factors
    overrides = fv_cgrid_overrides(meshes)
    mz = meshes[2]  # the bounded z factor
    assert isinstance(overrides[("diff", mz.cell_avg)], FaceDifference)
    assert isinstance(
        overrides[("diff", mz.average(CellAvg, bc=BC.NEUMANN))],
        FaceDifference)
    assert isinstance(overrides[("diff", mz.inner)], FluxDifference)
    assert isinstance(
        overrides[("diff", mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET))],
        FluxDifference)
    # the periodic x factor keeps the Right-face divergence
    assert isinstance(overrides[("diff", meshes[0].right)], FluxDifference)


def test_fv_cgrid_overrides_skips_a_mesh_without_cell_avg():
    # a mesh with no CellAvg / Right family contributes no rows (the
    # except branch): accessing mesh.cell_avg raises, and it is skipped
    class _NoFvMesh:
        @property
        def cell_avg(self):
            raise NotImplementedError

    overrides = fv_cgrid_overrides((_NoFvMesh(),))
    assert overrides == {}


def test_fv_model_grid_carries_the_profile():
    grid = periodic_grid()
    nh.Model(coriolis=FPlaneCoriolis(f0=1.0), grid=grid, dt=DT)
    for mesh in grid.factors:
        assert isinstance(grid.dispatch.resolve("diff", mesh.cell_avg),
                          FaceDifference)
        assert isinstance(grid.dispatch.resolve("diff", mesh.right),
                          FluxDifference)


def test_nodal_model_grid_keeps_the_nodal_diff():
    grid = periodic_grid()
    nh.Model(coriolis=FPlaneCoriolis(f0=1.0), grid=grid, dt=DT,
             family="nodal")
    # the nodal Right -> Center chain survives (no FV override)
    right = grid.factors[0].right
    assert not isinstance(grid.dispatch.resolve("diff", right),
                          FluxDifference)


# ================================================================
#  Preset == explicit assembly; the grid-aware dispatch hook
# ================================================================
def test_explicit_assembly_fv_core_gets_the_profile_and_runs():
    # DynamicalCore(family="fv") threads the profile through the
    # grid_dispatch_overrides hook on the generic assembly path too
    grid = periodic_grid()
    model = FrModel(
        grid=grid,
        modules=(DynamicalCore(family="fv"),
                 FPlaneCoriolis(f0=1.0),
                 ConstantStratification(n2=1.0, family="fv")),
        time_stepper=AdamBashforth(DT, order=3))
    assert all(isinstance(f, CellAvg)
               for f in model.state["b"].function_space.bare.factors)
    assert isinstance(grid.dispatch.resolve("diff", grid.factors[0].right),
                      FluxDifference)


def test_two_nodal_models_share_a_grid():
    # a second nodal model on a frozen grid re-assembles with no
    # dispatch overrides at all (the frozen, override-free branch).
    # family="nodal" is now explicit: since the 2026-07-16 ruling a
    # walled grid auto-flips to FV, so the nodal frozen-grid branch is
    # pinned by an explicit family, not by the walled default.
    grid = walled_grid()
    nh.Model(coriolis=FPlaneCoriolis(f0=1.0), grid=grid, dt=DT,
             advection=False, family="nodal")
    second = nh.Model(coriolis=FPlaneCoriolis(f0=1.0), grid=grid, dt=DT,
                      advection=False, family="nodal")
    assert not any(
        isinstance(f, CellAvg)
        for f in second.state["b"].function_space.bare.factors)


def test_two_fv_models_share_a_grid():
    # the second FV model on the (now frozen) grid verifies the profile
    # is already present instead of re-merging
    grid = periodic_grid()
    nh.Model(coriolis=FPlaneCoriolis(f0=1.0), grid=grid, dt=DT,
             advection=False)
    second = nh.Model(coriolis=FPlaneCoriolis(f0=1.0), grid=grid, dt=DT,
                      advection=False)
    assert all(isinstance(f, CellAvg)
               for f in second.state["b"].function_space.bare.factors)


def test_fv_model_on_a_grid_frozen_nodal_is_a_taught_error():
    # a grid frozen by a nodal model cannot satisfy a later FV model's
    # profile demand — a loud error, not silently-wrong stencils
    grid = periodic_grid()
    nh.Model(coriolis=FPlaneCoriolis(f0=1.0), grid=grid, dt=DT,
             advection=False, family="nodal")
    with pytest.raises(AssemblyError, match="frozen grid"):
        FrModel(
            grid=grid,
            modules=(DynamicalCore(family="fv"),
                     FPlaneCoriolis(f0=1.0),
                     ConstantStratification(n2=1.0, family="fv")),
            time_stepper=AdamBashforth(DT, order=3))
