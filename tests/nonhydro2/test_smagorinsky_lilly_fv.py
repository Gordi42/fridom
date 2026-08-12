"""SmagorinskyLilly on the finite-volume family, walled and periodic.

Prefix-mirrored shard of ``test_smagorinsky_lilly.py`` (oversized-module
rule): the walled ``CellAvg`` grids that used to be a blanket rejection.
The decisive oracle is **parity with the walled nodal model** — the 2nd
order FV and nodal stencils are the same numbers (FV-D4), so every
walled result of ``test_smagorinsky_lilly_walls.py`` must reproduce here
— plus the same ``Cs=0`` reduction to walled ``HarmonicFriction`` and
the mirror-symmetry oracles on the FV family itself. The builders are
duplicated (self-contained shard) rather than imported across files.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.closures.diffusion import HarmonicFriction
from fridom.model.model import Model, _chunk_body
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.core import Core
from fridom.nonhydro2.modules.smagorinsky_lilly import SmagorinskyLilly
from fridom.nonhydro2.modules.stratification import ConstantStratification
from fridom.nonhydro2.params import (
    SMAG_BACKGROUND_KAPPA,
    SMAG_BACKGROUND_NU,
    SMAG_BUOYANCY_MULTIPLIER,
    SMAG_CS,
    SMAG_PRANDTL,
    STRATIFICATION_N2,
)
from fridom.spatial.bc import BC
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.spaces.average import CellAvg

N = 8
L = 1.0
DZ = L / N
DT = 1e-3


def make_grid(periodic, family="fv", length=L, nz=N, device_ids=None):
    """Build the x/y/z grid and adopt ``family`` as the grid default.

    ``set_default_family`` is exactly what the ``nh.Model`` preset does
    (FV-D3): it makes every ``family=None`` field of the model follow
    one family, so an FV model has no accidental nodal field.
    """
    specs = (("x", N, L, periodic["x"]), ("y", N, L, periodic["y"]),
             ("z", nz, length, periodic["z"]))
    grid = Grid(tuple(
        IntervalMesh(n, (0.0, ln), periodic=p, name=nm)
        for nm, n, ln, p in specs), device_ids=device_ids)
    grid.set_default_family(family)
    return grid


def smag_model(grid, family="fv", n2=0.0, **kwargs):
    return Model(
        grid=grid,
        modules=(Core(family=family), ConstantStratification(n2=n2),
                 SmagorinskyLilly(**kwargs)),
        time_stepper=AdamBashforth(DT, order=3))


def friction_model(grid, nu, family="fv", slip="free", n2=0.0):
    return Model(
        grid=grid,
        modules=(Core(family=family), ConstantStratification(n2=n2),
                 HarmonicFriction(nu, slip=slip)),
        time_stepper=AdamBashforth(DT, order=3))


def data(field):
    return np.asarray(field.data)


def bound_closure(model):
    return next(m for m in model._carry.modules
                if isinstance(m, SmagorinskyLilly))


def ctx_for(cs=0.16, n2=0.0, bg_nu=1e-3, bg_kappa=1e-3):
    return SimpleNamespace(params={
        SMAG_CS: cs, SMAG_BUOYANCY_MULTIPLIER: 1.0, STRATIFICATION_N2: n2,
        SMAG_BACKGROUND_NU: bg_nu, SMAG_PRANDTL: 1.0,
        SMAG_BACKGROUND_KAPPA: bg_kappa})


def cos_z(m):
    """Return a wall-parallel cosine (free-slip) z-profile of order m."""
    return lambda x, y, z: np.cos(np.pi * m * z / L) + 0.0 * (x + y)


def sin_z(m):
    """Return a wall-normal sine (Dirichlet) z-profile of order m."""
    return lambda x, y, z: np.sin(np.pi * m * z / L) + 0.0 * (x + y)


# ================================================================
#  The FV state really is the average family
# ================================================================
def test_the_fv_model_carries_cell_average_factors():
    grid = make_grid({"x": True, "y": True, "z": False})
    model = smag_model(grid)
    space = model.state["b"].function_space.bare
    assert all(isinstance(space.factor(a), CellAvg)
               for a in ("x", "y", "z"))
    # the velocity's own staggered factor stays nodal (the C-grid face)
    u_space = model.state["u"].function_space.bare
    assert isinstance(u_space.factor("y"), CellAvg)
    assert not isinstance(u_space.factor("x"), CellAvg)


def test_the_default_walled_nonhydro_model_carries_the_closure():
    # nh.Model's family=None auto-promotes to FV on every grid, so the
    # DEFAULT walled model is finite-volume: before the lift, the plain
    # spelling below was refused and walled Smagorinsky was reachable
    # only through an explicit core=nh.Core(family="nodal").
    grid = Grid(tuple(
        IntervalMesh(N, (0.0, L), periodic=(nm != "z"), name=nm)
        for nm in ("x", "y", "z")), device_ids=(0,))
    model = nh.Model(
        grid=grid, core=nh.Core(),
        time_stepper=AdamBashforth(DT, order=3),
        buoyancy=nh.ConstantStratification(n2=1.0),
        advection=nh.CenteredAdvection(),
        modules_extra=(nh.SmagorinskyLilly(background_viscosity=1e-3),))
    assert isinstance(
        model.state["b"].function_space.bare.factor("z"), CellAvg)
    model.set_fields(u=cos_z(1))
    assert np.abs(data(model.tendency(model.state)["u"])).max() > 0.0


# ================================================================
#  Decisive: the walled FV closure terms equal the walled nodal ones
# ================================================================
@pytest.mark.parametrize("slip", ["free", "no"])
@pytest.mark.parametrize(
    "walled",
    [pytest.param(("z",), id="one-wall"),
     pytest.param(("y", "z"), id="two-walls"),
     pytest.param(("x", "y", "z"), id="three-walls")])
def test_walled_fv_closure_terms_match_the_nodal_model(walled, slip):
    # the closure hooks are compared directly (not the model tendency),
    # so the pressure projection cannot mask a difference. The 2nd-order
    # FV and nodal stencils are the same numbers, so the two families
    # must agree to floating-point noise.
    #
    # This is also the decisive guard against the R1 hole in the FV
    # staggering family (roadmap open.md 2d): the nodal family DOES
    # enforce require_grounded_bounded_sides, so a bounded BC-free
    # factor reconstructed outward on the FV side would read an
    # unrepaired wall ghost while the nodal side either grounds it or
    # raises -- and the two answers would part company in the wall
    # columns. They do not.
    kwargs = {"smagorinsky_constant": 0.16, "background_viscosity": 1e-3,
              "background_diffusivity": 1e-3, "slip": slip}
    periodic = {a: a not in walled for a in ("x", "y", "z")}
    models = {fam: smag_model(make_grid(periodic, family=fam), family=fam,
                              n2=1.5, **kwargs)
              for fam in ("nodal", "fv")}
    shapes = {nm: models["nodal"].state[nm].function_space.shape
              for nm in ("u", "v", "w", "b")}
    rng = np.random.default_rng(11)
    fields = {nm: 0.3 * rng.standard_normal(sh)
              for nm, sh in shapes.items()}
    ctx = ctx_for(n2=1.5)
    out = {}
    for fam, model in models.items():
        assert {nm: model.state[nm].function_space.shape
                for nm in shapes} == shapes
        model.set_fields(**fields)
        closure = bound_closure(model)
        terms = closure._stress(model.state, ctx)
        terms.update(closure._mixing(model.state, ctx))
        out[fam] = {nm: data(field) for nm, field in terms.items()}
    scale = max(np.abs(v).max() for v in out["nodal"].values())
    assert scale > 1.0
    for nm in ("u", "v", "w", "b"):
        np.testing.assert_allclose(out["fv"][nm], out["nodal"][nm],
                                   rtol=0.0, atol=1e-12 * scale)


def test_every_bounded_strain_factor_is_grounded_on_fv():
    # the structural half of the R1 guard above: on a bounded axis a
    # strain factor is either the anchor's own CellAvg (the .to is the
    # identity, no reconstruction) or carries the free-slip Dirichlet
    # tag (a wall-value claim, so the reconstruction grounds). Neither
    # can read the unrepaired wall ghost the FV family fails to fence.
    model = smag_model(make_grid({"x": True, "y": True, "z": False}))
    closure = bound_closure(model)
    anchor = closure._strain(model.state, 0, 0)
    cell = anchor.function_space.bare.factor("z")
    fields = [closure._strain(model.state, i, j)
              for i in range(3) for j in range(3)]
    fields.append(closure._eddy_viscosity(model.state, ctx_for()))
    for field in fields:
        factor = field.function_space.bare.factor("z")
        assert (factor is cell
                or BC.DIRICHLET in factor.bc.components), repr(factor)


# ================================================================
#  Cs=0 walled FV stress reduces to the walled FV background friction
# ================================================================
def test_cs_zero_walled_fv_tangential_stress_is_harmonic_friction():
    # tau = nu Sigma (no factor 2) makes the wall-parallel shear damp at
    # HALF the harmonic rate -> bit-for-bit HarmonicFriction(nu_bg/2),
    # on the FV family this time (the FV walled friction chain).
    nu_bg = 3e-3
    smag = smag_model(make_grid({"x": True, "y": True, "z": False}),
                      smagorinsky_constant=0.0,
                      background_viscosity=nu_bg)
    fric = friction_model(make_grid({"x": True, "y": True, "z": False}),
                          nu=0.5 * nu_bg, slip="free")
    for m in (1, 2, 3):
        smag.set_fields(u=cos_z(m))
        fric.set_fields(u=cos_z(m))
        ts = data(smag.tendency(smag.state)["u"])
        tf = data(fric.tendency(fric.state)["u"])
        np.testing.assert_array_equal(ts, tf)
        assert np.abs(ts).max() > 0.0


def test_cs_zero_no_slip_walled_fv_stress_is_harmonic_friction():
    nu_bg = 3e-3
    smag = smag_model(make_grid({"x": True, "y": True, "z": False}),
                      smagorinsky_constant=0.0,
                      background_viscosity=nu_bg, slip="no")
    fric = friction_model(make_grid({"x": True, "y": True, "z": False}),
                          nu=0.5 * nu_bg, slip="no")
    for m in (1, 2, 3, N):
        smag.set_fields(u=sin_z(m))
        fric.set_fields(u=sin_z(m))
        ts = data(smag.tendency(smag.state)["u"])
        tf = data(fric.tendency(fric.state)["u"])
        np.testing.assert_array_equal(ts, tf)
        assert np.abs(ts).max() > 0.0


# ================================================================
#  Mirror-symmetry oracles on the FV family
# ================================================================
def test_free_slip_walled_fv_matches_doubled_periodic_mirror():
    kwargs = {"smagorinsky_constant": 0.2, "background_viscosity": 1e-3}
    walled = smag_model(make_grid({"x": True, "y": True, "z": False}),
                        **kwargs)
    periodic = smag_model(
        make_grid({"x": True, "y": True, "z": True},
                  length=2 * L, nz=2 * N), **kwargs)
    rng = np.random.default_rng(0)
    r = rng.standard_normal(N)
    rp = np.concatenate([r, r[::-1]])  # even reflection about z = L
    walled.set_fields(u=np.broadcast_to(r, (N, N, N)).copy())
    periodic.set_fields(u=np.broadcast_to(rp, (N, N, 2 * N)).copy())
    tw = data(walled.tendency(walled.state)["u"])
    tp = data(periodic.tendency(periodic.state)["u"])
    np.testing.assert_array_equal(tw, tp[:, :, :N])


def test_no_slip_walled_fv_matches_doubled_periodic_odd_mirror():
    # the decisive two-consumer oracle (the |Sigma|^2 injection and the
    # stress drag must be mutually consistent), on the FV family
    kwargs = {"smagorinsky_constant": 0.2, "background_viscosity": 1e-3}
    walled = smag_model(make_grid({"x": True, "y": True, "z": False}),
                        slip="no", **kwargs)
    periodic = smag_model(
        make_grid({"x": True, "y": True, "z": True},
                  length=2 * L, nz=2 * N), slip="no", **kwargs)
    rng = np.random.default_rng(7)
    r = rng.standard_normal(N)
    rp = np.concatenate([r, -r[::-1]])  # ODD reflection about z = L
    walled.set_fields(u=np.broadcast_to(r, (N, N, N)).copy())
    periodic.set_fields(u=np.broadcast_to(rp, (N, N, 2 * N)).copy())
    tw = data(walled.tendency(walled.state)["u"])
    tp = data(periodic.tendency(periodic.state)["u"])
    np.testing.assert_allclose(tw, tp[:, :, :N], rtol=0.0, atol=1e-14)


def test_free_slip_uniform_wall_parallel_flow_has_no_drag_on_fv():
    model = smag_model(make_grid({"x": True, "y": True, "z": False}),
                       smagorinsky_constant=0.16,
                       background_viscosity=1e-2)
    model.set_fields(u=lambda x, y, z: np.ones_like(x + y + z))
    assert np.abs(data(model.tendency(model.state)["u"])).max() == 0.0


def test_no_slip_uniform_flow_drags_only_the_wall_cells_on_fv():
    nu_bg = 1e-2
    model = smag_model(make_grid({"x": True, "y": True, "z": False}),
                       smagorinsky_constant=0.0,
                       background_viscosity=nu_bg, slip="no")
    model.set_fields(u=lambda x, y, z: np.ones_like(x + y + z))
    tend = data(model.tendency(model.state)["u"])
    np.testing.assert_allclose(tend[:, :, 1:-1], 0.0, atol=1e-13)
    np.testing.assert_allclose(tend[:, :, 0], -nu_bg / DZ**2, rtol=1e-12)
    np.testing.assert_allclose(tend[:, :, -1], -nu_bg / DZ**2, rtol=1e-12)


# ================================================================
#  A walled FV run: finite, energy-dissipating, corners compose
# ================================================================
def test_walled_fv_run_dissipates_kinetic_energy():
    model = smag_model(make_grid({"x": True, "y": True, "z": False}),
                       n2=1.0, smagorinsky_constant=0.16,
                       background_viscosity=1e-2,
                       background_diffusivity=1e-2)
    model.set_fields(u=cos_z(1), v=cos_z(2))
    energies = []
    for _ in range(5):
        model.advance(2)
        energies.append(float(np.sum(data(model.diagnostics.ekin()))))
    energies = np.asarray(energies)
    assert np.isfinite(energies).all()
    assert (np.diff(energies) < 0.0).all()


def test_two_walled_axes_compose_on_fv_and_step_finite():
    model = smag_model(make_grid({"x": False, "y": True, "z": False}),
                       smagorinsky_constant=0.16,
                       background_viscosity=1e-2,
                       background_diffusivity=1e-2)
    model.set_fields(
        u=lambda x, y, z: np.sin(np.pi * x) * np.cos(np.pi * z) + 0.0 * y,
        v=lambda x, y, z: np.cos(np.pi * y) * np.sin(np.pi * z) + 0.0 * x,
        b=lambda x, y, z: np.sin(np.pi * z) + 0.0 * (x + y))
    final = _chunk_body(model._artifacts.record, 5,
                        model._carry, model._stepper)
    assert all(bool(np.all(np.isfinite(np.asarray(f.data))))
               for f in final.state)


# ================================================================
#  Reverse-mode AD through a walled FV run matches central FD
# ================================================================
def _fv_grad_loss(slip="free", n_steps=8, cs=0.16, n2=15.0):
    """Build a grad-ready loss over a walled FV Smagorinsky run."""
    model = smag_model(make_grid({"x": True, "y": True, "z": False}),
                       n2=n2, smagorinsky_constant=cs, slip=slip)
    rng = np.random.default_rng(1)
    sh = (N, N, N)
    model.set_fields(u=0.2 * rng.standard_normal(sh),
                     v=0.2 * rng.standard_normal(sh),
                     w=0.2 * rng.standard_normal((N, N, N - 1)),
                     b=0.01 * rng.standard_normal(sh))
    closure = bound_closure(model)
    n_clipped = int(np.sum(data(closure._eddy_viscosity(
        model.state, ctx_for(cs=cs, n2=n2))) == 0.0))
    record = model._artifacts.record
    carry = model._carry
    stepper = model._stepper
    leaf = closure.smagorinsky_constant
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    idx = next(i for i, lf in enumerate(leaves) if lf is leaf)

    def loss(theta):
        packed = list(leaves)
        packed[idx] = theta
        c = jax.tree_util.tree_unflatten(treedef, packed)
        final = _chunk_body(record, n_steps, c, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    return loss, jnp.asarray(leaf, dtype=jnp.float64), n_clipped


@pytest.mark.parametrize("slip", ["free", "no"])
def test_walled_fv_grad_is_finite_and_matches_fd(slip):
    loss, x0, n_clipped = _fv_grad_loss(slip=slip)
    assert 0 < n_clipped < N ** 3  # the clip fires (some cells, not all)
    g = float(jax.grad(loss)(x0))
    assert np.isfinite(g)
    eps = 1e-4
    fd = float((loss(x0 * (1 + eps)) - loss(x0 * (1 - eps)))
               / (2 * x0 * eps))
    np.testing.assert_allclose(g, fd, rtol=1e-4)


# ================================================================
#  Taught rejection: a CellAvg target with no face-exposing diff row
# ================================================================
def test_fv_target_without_a_face_exposing_diff_row_is_a_taught_rejection():
    # a nodal model whose buoyancy tracer alone is declared family="fv"
    # (FV-D1b) leaves the grid on the nodal dispatch profile, so
    # ("diff", CellAvg) resolves to the collocated FVDerivative, whose
    # flux never surfaces as a face field and cannot close a wall. The
    # diffusion campaign's probe catches it at bind, attributed.
    grid = make_grid({"x": True, "y": True, "z": False}, family="nodal")
    with pytest.raises(NotImplementedError,
                       match="does not expose a face-located flux"):
        Model(grid=grid,
              modules=(Core(family="nodal"),
                       ConstantStratification(n2=0.0, family="fv"),
                       SmagorinskyLilly()),
              time_stepper=AdamBashforth(DT, order=3))


def test_periodic_fv_tracer_on_a_nodal_model_still_assembles():
    # the same mixed declaration on a fully periodic grid has no wall to
    # close, so it stays served (the probe is a walled-axis check only)
    grid = make_grid({"x": True, "y": True, "z": True}, family="nodal")
    model = Model(grid=grid,
                  modules=(Core(family="nodal"),
                           ConstantStratification(n2=0.0, family="fv"),
                           SmagorinskyLilly(background_diffusivity=1e-3)),
                  time_stepper=AdamBashforth(DT, order=3))
    model.set_fields(b=lambda x, y, z: np.sin(2 * np.pi * z / L)
                     + 0.0 * (x + y))
    assert np.abs(data(model.tendency(model.state)["b"])).max() > 0.0


# ================================================================
#  Sharded walled FV axis == single device
# ================================================================
@pytest.mark.multi_device
@pytest.mark.parametrize("slip", ["free", "no"])
def test_sharded_walled_fv_stress_matches_single_device(forced_devices,
                                                        slip):
    # the FV twin of the nodal shard test: an all-walled grid shards a
    # bounded axis, so the CellAvg -> Inner strain diff, the free-slip
    # retag and the no-slip wall-weight fields cross a shard boundary.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    nu_bg = 3e-3
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        grid = make_grid({"x": False, "y": False, "z": False},
                         device_ids=device_ids)
        model = smag_model(grid, smagorinsky_constant=0.16,
                           background_viscosity=nu_bg,
                           background_diffusivity=nu_bg, slip=slip)
        model.set_fields(
            u=lambda x, y, z: 0.3 * np.sin(np.pi * x) * np.cos(np.pi * y)
            + 0.0 * z,
            v=lambda x, y, z: 0.2 * np.cos(np.pi * x) * np.sin(np.pi * y)
            + 0.0 * z,
            b=lambda x, y, z: 0.1 * np.sin(np.pi * x) * np.sin(np.pi * z)
            + 0.0 * y)
        results[tag] = {nm: data(model.tendency(model.state)[nm])
                        for nm in ("u", "v", "w", "b")}
        if tag == "many" and jax.device_count() > 1:
            assert model.state["u"]._data.sharding.spec[0] == "devices"
    for nm in ("u", "v", "w", "b"):
        np.testing.assert_allclose(results["many"][nm], results["one"][nm],
                                   rtol=1e-12, atol=1e-14)
