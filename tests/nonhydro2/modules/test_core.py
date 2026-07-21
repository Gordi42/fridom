"""Core: the cross-module ramped-aspect-ratio guard (TDF-D4).

The aspect ratio enters the frozen linear operator ``L`` through the
pressure projection, which is a CONSTRAINT stage rather than a
``linear=True`` term, so the structural term sweep cannot see it.
``Core`` reports a ramped ``aspect_ratio`` from its own leaf, closing
the hole a model assembled without stratification would otherwise slip
through. A re-reading stepper (``AdamBashforth``) is unaffected: the
ramped model assembles and advances to finite values.
"""
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model import term_predicates as terms
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.params import ASPECT_RATIO

N = 8
F0, N2 = 1.5, 3.0
DT = 1e-3


def _grid(*, periodic_y=True):
    """Build a small grid; walled in y admits a channel eigenbasis."""
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                        periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                        periodic=periodic_y, name="y")
    mz = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                        periodic=True, name="z")
    return fr.spatial.Grid((mx, my, mz), device_ids=(0,))


def _model(aspect_ratio, stepper, *, grid=None):
    """Build a small linear nonhydro model (aspect ratio + stepper)."""
    return nh.Model(
        grid=grid if grid is not None else _grid(),
        core=nh.Core(aspect_ratio=aspect_ratio),
        time_stepper=stepper,
        coriolis=nh.FPlaneCoriolis(f0=F0),
        stratification=nh.ConstantStratification(n2=N2),
        advection=False)


def test_core_reports_a_ramped_aspect_ratio():
    """The owner-side report: () for a float, (name,) for a Ramp."""
    assert (nh.Core(aspect_ratio=2.0)
            .time_dependent_linear_parameters() == ())
    ramp = fr.model.Ramp(1.0, 2.0, period=1.0)
    assert (nh.Core(aspect_ratio=ramp)
            .time_dependent_linear_parameters()
            == (str(ASPECT_RATIO),))


def test_core_refuses_zero_aspect_ratio():
    """aspect_ratio=0 is refused at construction (taught error)."""
    with pytest.raises(TypeError, match="aspect_ratio=0"):
        nh.Core(aspect_ratio=0.0)


def test_etdrk4_refuses_a_ramped_aspect_ratio():
    """A frozen-L (ETDRK4) stepper refuses a ramped aspect ratio.

    The aspect ratio scales the pressure projection that builds ``L``,
    so a ramped leaf makes ``L(t)`` time-dependent; ``exp(L dt)`` from
    the frozen eigenbasis would silently integrate a stale operator.
    The guard fires at ASSEMBLY of the ETDRK4 model.
    """
    grid = _grid(periodic_y=False)
    static = _model(
        2.0, AdamBashforth(DT, order=3),
        grid=grid)
    basis = nh.eigenbasis(static)
    ramp = fr.model.Ramp(1.0, 2.0, period=1.0, curve="exp")
    with pytest.raises(
            fr.model.errors.TimeDependentLinearOperatorError,
            match=r"nonhydro\.aspect_ratio \(Core\)") as ex:
        nh.Model(
            grid=_grid(periodic_y=False),
            core=nh.Core(aspect_ratio=ramp),
            time_stepper=fr.model.time_steppers.ETDRK4(DT, basis),
            coriolis=nh.FPlaneCoriolis(f0=F0),
            stratification=nh.ConstantStratification(n2=N2),
            advection=False,
            term_filter=~terms.linear)
    # the taught error points at the AB fallback and the design record
    assert "AdamBashforth" in str(ex.value)
    assert "exponential_stepper.md" in str(ex.value)


def test_core_stores_and_validates_multigrid_agglomerate():
    """Thread and validate the MG-D10 agglomeration knob on the core."""
    assert nh.Core(
        aspect_ratio=2.0,
        multigrid_agglomerate=4)._multigrid_agglomerate == 4
    assert nh.Core(aspect_ratio=2.0)._multigrid_agglomerate is None
    with pytest.raises(ValueError, match="positive integer"):
        nh.Core(aspect_ratio=2.0, multigrid_agglomerate=0)


def test_ramped_aspect_ratio_advances_under_adam_bashforth():
    """A re-reading stepper handles L(t): the ramped model runs."""
    ramp = fr.model.Ramp(1.0, 2.0, period=6 * DT, curve="cosine")
    model = _model(
        ramp, AdamBashforth(DT, order=3))
    model.advance(4)
    for comp in ("u", "v", "w", "b"):
        assert np.all(np.isfinite(np.asarray(model.state[comp].data)))
