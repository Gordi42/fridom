"""The hydrostatic scaling surface (Branch 2): variants, errors, parity.

The prefix-mirrored scaling shard of ``test_model.py``: the
gravity-first core, the dual-variant stratification / free-surface
family, the retired preset kwargs, the ExternalWave/Rotational
today-parity spellings across all three free-surface variants, the
energy re-key, and the propagator autodiff gate through the nondim
barotropic path.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.energy import EnergyMetric
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

DT = 2.0 ** -7
STEPS = 6
NAMES = ("u", "v", "b", "ps")


def make_grid(nz=4):
    im = fr.spatial.meshes.IntervalMesh
    return fr.spatial.Grid((
        im(8, (0.0, 1.0), periodic=True, name="x"),
        im(8, (0.0, 1.0), periodic=True, name="y"),
        im(nz, (0.0, 1.0), periodic=False, name="z")),
        device_ids=(0,))


def dim_model(*, free_surface=None, advection=True):
    """Build the dimensional twin (today: csqr=1, ro=1, f0=0.5, n2=4).

    ``surface_advective_flux=False`` (the legacy closure) keeps the
    dim-vs-nondim parity gates bitwise: the H7 closure row is the one
    accepted <=1-ulp folded-statics seam (see test_golden_parity).
    """
    return hy.Model(
        grid=make_grid(), core=hy.Core(gravity=1.0),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        stratification=hy.ConstantStratification(n2=4.0),
        free_surface=free_surface or hy.ExplicitFreeSurface(),
        advection=advection, surface_advective_flux=False,
        time_stepper=AdamBashforth(DT, order=3))


def ext_model(*, ro=1.0, free_surface_cls=hy.ExplicitFreeSurface,
              advection=True, **fs_kwargs):
    """ExternalWave-frame parity twin of :func:`dim_model`.

    eps = Fr_ext = ro; Coriolis Ro = ro/f0 = 2*ro (eps/Ro = 0.5);
    Fr_int = ro/2 ((eps/Fr_int)^2 = 4); (eps/Fr_ext)^2 = 1 = csqr.
    """
    return hy.Model(
        grid=make_grid(), core=hy.Core(),
        scaling=fr.scaling.ExternalWave(),
        coriolis=hy.FPlaneCoriolis(rossby_number=2.0 * ro),
        stratification=hy.ConstantStratification(froude_number=ro / 2),
        free_surface=free_surface_cls(froude_number=ro, **fs_kwargs),
        advection=advection, surface_advective_flux=False,
        time_stepper=AdamBashforth(DT, order=3))


def random_fields(model, amp=0.02, seed=9):
    rng = np.random.default_rng(seed)
    return {name: amp * rng.standard_normal(model.state[name].shape)
            for name in ("u", "v", "b")}


# ================================================================
#  Retired preset kwargs, required modules, params (taught errors)
# ================================================================
@pytest.mark.parametrize(
    ("kwarg", "match"),
    [pytest.param({"csqr": 1.0}, "csqr= is retired", id="csqr"),
     pytest.param({"rossby_number": 0.5},
                  "rossby_number= is retired", id="rossby"),
     pytest.param({"dt": 0.1}, "dt= is retired", id="dt")])
def test_preset_teaches_the_retired_kwargs(kwarg, match):
    with pytest.raises(TypeError, match=match):
        hy.Model(grid=make_grid(), core=hy.Core(gravity=1.0),
                 stratification=hy.ConstantStratification(n2=1.0),
                 free_surface=hy.ExplicitFreeSurface(),
                 time_stepper=AdamBashforth(DT, order=3), **kwarg)


def test_preset_requires_the_physics_modules():
    with pytest.raises(TypeError, match="REQUIRED"):
        hy.Model(grid=make_grid(), core=hy.Core(gravity=1.0),
                 stratification=hy.ConstantStratification(n2=1.0),
                 free_surface=None,
                 time_stepper=AdamBashforth(DT, order=3))


def test_params_teach_the_retired_csqr_name():
    with pytest.raises(AttributeError, match="GRAVITY"):
        _ = hy.params.CSQR


def test_core_gravity_variants():
    dim = hy.Core(gravity=9.81)
    assert dim.scaling_variant == "dimensional"
    assert [d.name for d in dim.parameter_declarations] == [
        "hydrostatic.gravity"]
    nondim = hy.Core()
    assert nondim.scaling_variant == "nondimensional"
    assert nondim.parameter_declarations == ()
    with pytest.raises(TypeError, match="gravity=0"):
        hy.Core(gravity=0.0)


def test_hy_stratification_takes_exactly_one_kwarg_set():
    with pytest.raises(TypeError, match="exactly one kwarg set"):
        hy.ConstantStratification()
    with pytest.raises(TypeError, match="froude_number=0"):
        hy.ConstantStratification(froude_number=0.0)


def test_free_surface_froude_variants():
    dim = hy.ExplicitFreeSurface()
    assert dim.scaling_variant == "dimensional"
    assert dim.parameter_declarations == ()
    assert [r.name for r in dim.parameter_references] == [
        "hydrostatic.gravity"]
    nondim = hy.ExplicitFreeSurface(froude_number=0.5)
    assert nondim.scaling_variant == "nondimensional"
    assert [d.name for d in nondim.parameter_declarations] == [
        "hydrostatic.froude"]
    assert nondim.parameter_references == ()
    assert nondim.scaling_mechanism == "external_wave"
    with pytest.raises(TypeError, match="froude_number=0"):
        hy.SplitExplicitFreeSurface(froude_number=0.0)


def test_mixed_variants_are_refused_at_assembly():
    # a nondim core next to dimensional stratification/free surface
    # never assembles (the fr.scaling mixed-variant taught error)
    with pytest.raises(fr.model.errors.AssemblyError,
                       match="MIXED scaling variants"):
        fr.model.Model(
            grid=make_grid(),
            modules=(hy.Core(),
                     hy.ConstantStratification(n2=1.0),
                     hy.ExplicitFreeSurface()),
            time_stepper=AdamBashforth(DT, order=3))


# ================================================================
#  Today-parity: the ExternalWave spelling equals the dim twin
# ================================================================
@pytest.mark.parametrize(
    ("fs_cls", "fs_kwargs", "budget"),
    [pytest.param(hy.ExplicitFreeSurface, {}, 0.0, id="explicit"),
     pytest.param(hy.ImplicitFreeSurface,
                  {"epsilon": 1.0}, 1e-13, id="implicit"),
     pytest.param(hy.SplitExplicitFreeSurface,
                  {"substeps": 8}, 1e-13, id="split")])
def test_external_wave_parity_across_the_variants(
        fs_cls, fs_kwargs, budget):
    # ro=1: eps = Fr_ext = 1 aliased -> the barotropic coefficient is
    # exactly 1.0 = csqr, the internal ratio exactly 4.0 = n2, the
    # rotation ratio exactly 0.5 = f0; the explicit variant is
    # bitwise, the solve-carrying variants to solver roundoff
    dim = dim_model(free_surface=fs_cls(**fs_kwargs))
    ext = ext_model(ro=1.0, free_surface_cls=fs_cls, **fs_kwargs)
    fields = random_fields(dim)
    dim.set_fields(**fields)
    ext.set_fields(**fields)
    dim.advance(3)
    ext.advance(3)
    for name in NAMES:
        a = np.asarray(dim.state[name].data)
        b = np.asarray(ext.state[name].data)
        if budget == 0.0:
            assert np.array_equal(a, b), name
        else:
            np.testing.assert_allclose(a, b, rtol=0, atol=budget)


def test_nondim_surface_closure_tracks_the_dim_twin():
    # the H7 surface closure's nondim branch (the epsilon-scaled
    # boundary row) tracks the dimensional twin to closure-row
    # roundoff (the accepted folded-statics seam; bitwise gates live
    # in test_golden_parity)
    stepper = AdamBashforth(DT, order=3)
    dim = hy.Model(
        grid=make_grid(), core=hy.Core(gravity=1.0),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        stratification=hy.ConstantStratification(n2=4.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=True, time_stepper=stepper)
    ext = hy.Model(
        grid=make_grid(), core=hy.Core(),
        scaling=fr.scaling.ExternalWave(),
        coriolis=hy.FPlaneCoriolis(rossby_number=2.0),
        stratification=hy.ConstantStratification(froude_number=0.5),
        free_surface=hy.ExplicitFreeSurface(froude_number=1.0),
        advection=True, time_stepper=AdamBashforth(DT, order=3))
    fields = random_fields(dim)
    dim.set_fields(**fields)
    ext.set_fields(**fields)
    dim.advance(3)
    ext.advance(3)
    for name in NAMES:
        np.testing.assert_allclose(
            np.asarray(dim.state[name].data),
            np.asarray(ext.state[name].data), rtol=0, atol=1e-13)


def test_energy_metric_re_keys_on_the_nondim_primitives():
    # nondim ps weight H_ref/(eps/Fr_ext)^2 == dim 1/g at parity
    dim = EnergyMetric.from_model(dim_model())
    ext = EnergyMetric.from_model(ext_model(ro=1.0))
    for name in ("u", "v", "b", "ps"):
        assert ext.weights[name] == pytest.approx(dim.weights[name])


def test_nondim_diagnostics_re_key_on_the_effective_n2():
    dim = dim_model()
    ext = ext_model(ro=1.0)
    fields = random_fields(dim)
    dim.set_fields(**fields)
    ext.set_fields(**fields)
    for name in ("ekin", "epot"):
        a = np.asarray(getattr(dim.diagnostics, name)().data)
        b = np.asarray(getattr(ext.diagnostics, name)().data)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_nondim_eigenbasis_matches_the_dim_twin():
    # the numeric probe engine sees identical operators at parity
    dim = hy.eigenbasis(dim_model(advection=False))
    ext = hy.eigenbasis(ext_model(ro=1.0, advection=False))
    np.testing.assert_allclose(
        np.sort(np.abs(np.asarray(dim.basis.omega).ravel())),
        np.sort(np.abs(np.asarray(ext.basis.omega).ravel())),
        rtol=1e-10, atol=1e-10)


# ================================================================
#  Autodiff through the nondim barotropic path (propagator gate)
# ================================================================
def test_ic_grad_through_the_nondim_barotropic_path_matches_fd():
    model = ext_model(ro=0.5)
    model.set_fields(**random_fields(model))
    run = model.propagator(wrt=("u",), steps=STEPS)
    u0 = model._carry.state["u"].storage

    def loss(field):
        final = run((field,))
        return sum(jnp.sum(final.state[c].data ** 2) for c in NAMES)

    grad = np.asarray(jax.grad(loss)(u0))
    assert bool(np.all(np.isfinite(grad)))
    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(u0.shape),
                            dtype=u0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(u0 + eps * direction))
          - float(loss(u0 - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)
