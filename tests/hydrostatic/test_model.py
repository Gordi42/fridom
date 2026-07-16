"""The hydrostatic preset factory: assembly, defaults, taught errors."""
import jax
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.model import chunk_cache_size

IM = fr.spatial.meshes.IntervalMesh


def make_grid(nx=8, nz=4):
    """Return a doubly-periodic horizontal, bounded-vertical grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, 1.0), periodic=False, name="z")))


def make_model(grid=None, *, dt=1e-3, **kwargs):
    """Return a minimal linear hydrostatic model through the factory."""
    if grid is None:
        grid = make_grid()
    kwargs.setdefault("csqr", 1.0)
    kwargs.setdefault("coriolis", hy.FPlaneCoriolis(f0=1.0))
    return hy.Model(
        grid=grid, dt=dt, advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3),
        **kwargs)


# ================================================================
#  D4: the preset is a thin factory (identical carry treedef)
# ================================================================
def test_preset_equals_explicit_assembly_treedef():
    # the canonical D4 check (shallowwater precedent): ONE shared grid,
    # preset vs explicit assembly, identical full-carry treedef
    grid = make_grid()
    stepper = fr.model.time_steppers.AdamBashforth(1e-3, order=3)
    preset = hy.Model(
        grid=grid, csqr=3.0, rossby_number=0.2,
        coriolis=hy.FPlaneCoriolis(f0=1.3),
        stratification=hy.ConstantStratification(n2=2.0),
        free_surface=hy.ExplicitFreeSurface(),
        time_stepper=stepper)
    explicit = fr.model.Model(
        grid=grid,
        modules=(
            hy.HydrostaticCore(csqr=3.0, rossby_number=0.2),
            hy.FPlaneCoriolis(f0=1.3),
            hy.ConstantStratification(n2=2.0),
            hy.ExplicitFreeSurface(),
            # the factory default is advection=True -> CenteredAdvection,
            # appended after the free surface (H2b)
            fr.model.modules.CenteredAdvection()),
        time_stepper=fr.model.time_steppers.AdamBashforth(1e-3, order=3))
    assert (jax.tree_util.tree_structure(preset._carry)
            == jax.tree_util.tree_structure(explicit._carry))
    # same module order (core, coriolis, stratification, free surface)
    assert ([type(m) for m in preset._carry.modules]
            == [type(m) for m in explicit._carry.modules])
    # same state vocabulary and component order
    assert type(preset.state) is type(explicit.state)
    assert preset.state.component_names == explicit.state.component_names


def test_preset_is_a_plain_model_not_a_subclass():
    assert type(make_model()) is fr.model.Model


# ================================================================
#  Defaults
# ================================================================
def test_default_modules_and_stepper():
    # coriolis omitted -> no rotation at all
    model = hy.Model(grid=make_grid(), csqr=1.0)
    modules = model._carry.modules
    assert any(isinstance(m, hy.HydrostaticCore) for m in modules)
    assert any(isinstance(m, hy.ExplicitFreeSurface) for m in modules)
    strat = [m for m in modules
             if isinstance(m, hy.ConstantStratification)]
    assert len(strat) == 1
    assert float(model.parameters[hy.params.STRATIFICATION_N2]) == 1.0
    assert isinstance(model._stepper,
                      fr.model.time_steppers.AdamBashforth)


def test_default_is_no_rotation_at_all():
    model = hy.Model(grid=make_grid(), csqr=1.0)
    assert "f_coriolis" not in model.state
    assert fr.model.params.CORIOLIS_F0 not in model.parameters
    with pytest.raises(LookupError, match="no live module matches"):
        model.module(hy.FPlaneCoriolis)


def test_a_named_coriolis_is_installed():
    model = hy.Model(grid=make_grid(), csqr=1.0,
                     coriolis=hy.FPlaneCoriolis(f0=1.5))
    assert "f_coriolis" in model.state
    assert float(model.parameters[fr.model.params.CORIOLIS_F0]) == 1.5


# ================================================================
#  Advection is installed (stage H2b); False keeps the linear model
# ================================================================
@pytest.mark.parametrize(
    "advection",
    [
        pytest.param(True, id="bool-true"),
        pytest.param(fr.model.modules.CenteredAdvection(), id="centered"),
        pytest.param(fr.model.modules.UpwindAdvection(order=3),
                     id="upwind"),
    ],
)
def test_advection_is_installed(advection):
    model = hy.Model(grid=make_grid(), csqr=1.0, advection=advection)
    assert any("Advection" in type(m).__name__
               for m in model._carry.modules)


def test_advection_false_installs_no_advection_module():
    model = hy.Model(grid=make_grid(), csqr=1.0, advection=False)
    assert not any("Advection" in type(m).__name__
                   for m in model._carry.modules)


# ================================================================
#  modules_extra appends after the core stack
# ================================================================
def test_modules_extra_appends():
    # a bare model (coriolis omitted) carries no rotation ...
    bare = hy.Model(grid=make_grid(), csqr=1.0)
    assert not any(isinstance(m, fr.model.modules.FPlaneCoriolis)
                   for m in bare._carry.modules)
    # ... and an extra module is appended and assembled (its rotation
    # term and parameter take effect)
    extra = fr.model.modules.FPlaneCoriolis(f0=0.9)
    model = hy.Model(grid=make_grid(), csqr=1.0, coriolis=None,
                     modules_extra=(extra,))
    assert any(isinstance(m, fr.model.modules.FPlaneCoriolis)
               for m in model._carry.modules)
    assert "f_coriolis" in model.state
    assert float(model.parameters[fr.model.params.CORIOLIS_F0]) == 0.9


# ================================================================
#  Smoke run: finite, treedef stable, single compile
# ================================================================
def _seed(model):
    rng = np.random.default_rng(7)
    model.set_fields(
        u=0.01 * rng.standard_normal(model.state["u"].shape),
        v=0.01 * rng.standard_normal(model.state["v"].shape),
        b=0.01 * rng.standard_normal(model.state["b"].shape),
        ps=0.01 * rng.standard_normal(model.state["ps"].shape))


def test_run_stays_finite_and_treedef_stable():
    model = make_model()
    _seed(model)
    before = jax.tree_util.tree_structure(model._carry)
    model.advance(5)
    after = jax.tree_util.tree_structure(model._carry)
    assert before == after
    for name in ("u", "v", "b", "ps"):
        assert np.isfinite(np.asarray(model.state[name].data)).all()


def test_repeated_advance_compiles_nothing(compile_counter):
    model = make_model()
    _seed(model)
    model.advance(5)                       # warm every path
    compile_counter.reset()
    reference = chunk_cache_size()
    model.advance(5)
    assert compile_counter.count == 0
    assert chunk_cache_size() == reference
