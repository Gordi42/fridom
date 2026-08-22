"""The hydrostatic bare buoyancy tracer (no background stratification)."""
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.errors import AssemblyError
from fridom.model.params import STRATIFICATION_N2
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

IM = fr.spatial.meshes.IntervalMesh


def make_grid(nx=8, nz=4):
    """Return a doubly-periodic horizontal, bounded-vertical grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, 1.0), periodic=False, name="z")))


# the class as the default: each model gets a fresh instance (a module
# binds to one model only)
def make_model(*, advection=fr.model.modules.CenteredAdvection):
    """Return a hydrostatic model whose buoyancy is a bare tracer."""
    return hy.Model(
        grid=make_grid(),
        core=hy.Core(gravity=2.0),
        time_stepper=AdamBashforth(1e-2, order=3),
        coriolis=hy.FPlaneCoriolis(f0=1.0),
        buoyancy=hy.BuoyancyTracer(),
        free_surface=hy.ExplicitFreeSurface(),
        advection=(advection() if isinstance(advection, type)
                   else advection))


def test_declares_the_buoyancy_tracer():
    model = make_model()
    assert "b" in model.state
    assert model.state["b"].xr.attrs["units"] == "m/s^2"


def test_provides_no_background_stratification():
    # a bare tracer publishes no stratification.n2 / .froude, so the
    # 1/N^2 consumers refuse the model through the missing provide
    model = make_model()
    assert STRATIFICATION_N2 not in model.parameters


def test_advection_advances_the_tracer():
    model = make_model()
    model.set_fields(
        b=lambda x, y, z: np.sin(2 * np.pi * x) + 0.0 * (y + z),
        u=lambda x, y, z: 0.2 + 0.0 * (x + y + z))
    before = np.asarray(model.state["b"].data).copy()
    model.run(steps=5)
    after = np.asarray(model.state["b"].data)
    assert np.isfinite(after).all()
    # the advecting flow moves the tracer, so b changes
    assert not np.allclose(before, after)


def test_linear_assembly_is_rejected():
    # with no advection the bare tracer leaves b advanced by no term at
    # all (the restoring is absent, not zero), so the coverage lint
    # rejects it — unlike ConstantStratification(n2=0.0)
    with pytest.raises(AssemblyError, match="coverage lint"):
        make_model(advection=None)


def test_constant_stratification_zero_is_accepted_where_the_tracer_is_not():
    # the contrast: n2=0.0 keeps the -N^2 w term (times zero), so b IS
    # advanced and a linear assembly stands
    model = hy.Model(
        grid=make_grid(),
        core=hy.Core(gravity=2.0),
        time_stepper=AdamBashforth(1e-2, order=3),
        coriolis=hy.FPlaneCoriolis(f0=1.0),
        buoyancy=hy.ConstantStratification(n2=0.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=None)
    assert "b" in model.state
