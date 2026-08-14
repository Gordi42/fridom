"""Advection on immersed (cut-cell) grids: fraction weighting (IP-D4).

The shared ``_FluxFormAdvection`` weights every face flux by the
open-area fraction and divides the flux divergence by the cell volume
fraction when the grid is immersed; the centered scheme is the
supported family, the biased (upwind/WENO) schemes reject an immersed
grid at bind (their wide windows reach across dry cells — designed-for,
IP-D8). The end-to-end conservation and the model taught error live in
``tests/nonhydro2/test_immersed_model.py``; these are the module-level
contract tests.
"""
import numpy as np
import pytest

from fridom.model.model import Model as FrModel
from fridom.model.modules.advection import (
    CenteredAdvection,
    UpwindAdvection,
    WENOAdvection,
)
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.core import Core
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
)
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

TWO_PI = 2.0 * np.pi
DT = 0.01


def _immersed_grid(n=10):
    box = lambda x, y, z: (  # noqa: E731
        (x > 1.0) & (x < 5.0) & (y > 1.0) & (y < 5.0)
        & (z > 0.2) & (z < 0.8)).astype(float)
    return Grid(tuple(
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name=nm)
        for nm in ("x", "y", "z")), immersed=ImmersedDomain(box))


def _fv_modules(advection):
    """Core(fv) + b + advection, an assemblable module set."""
    return (
        Core(family="fv"),
        ConstantStratification(n2=0.0, family="fv"),
        FPlaneCoriolis(f0=1.0),
        advection,
    )


# ================================================================
#  The capability flags (IP-D4 / GA-D6): every scheme supports immersed
# ================================================================
def test_every_scheme_supports_immersed():
    # the biased schemes gained the mask-keyed graded closure (GA-D6),
    # so all three flux-form schemes now bind on an immersed grid
    assert CenteredAdvection._supports_immersed is True
    assert UpwindAdvection._supports_immersed is True
    assert WENOAdvection._supports_immersed is True


# ================================================================
#  Biased schemes now bind on an immersed grid (GA-D4): the mask-keyed
#  graded kernels install and the module captures the descriptor
# ================================================================
@pytest.mark.parametrize("cls", [UpwindAdvection, WENOAdvection])
@pytest.mark.parametrize("order", [3, 5])
def test_biased_schemes_bind_on_immersed(cls, order):
    grid = _immersed_grid()
    model = FrModel(grid=grid, modules=_fv_modules(cls(order)),
                    time_stepper=AdamBashforth(DT, order=3))
    (advection,) = [m for m in model.modules if isinstance(m, cls)]
    # the descriptor is captured and the biased kernels declare the
    # order//2+1 mask-keyed halo (wider than the base order-2 fraction
    # stencil for order 5), routed through the explicit extra_halo (GA-D3)
    assert advection._immersed is grid.immersed
    assert advection.extra_halo == HaloSpec(
        dict.fromkeys(grid.names, order // 2 + 1))


# ================================================================
#  Centered binds on an immersed grid and declares the halo exemption
# ================================================================
def test_centered_binds_on_immersed_and_declares_extra_halo():
    grid = _immersed_grid()
    model = FrModel(grid=grid, modules=_fv_modules(CenteredAdvection()),
                    time_stepper=AdamBashforth(DT, order=3))
    (advection,) = [m for m in model.modules
                    if isinstance(m, CenteredAdvection)]
    # the bound module captured the immersed descriptor and declares a
    # (halo-2, order-2 centered) FD-stencil halo — the fraction multiply
    # is a concrete field the halo tracer cannot follow
    assert advection._immersed is grid.immersed
    assert advection.extra_halo == HaloSpec(
        dict.fromkeys(grid.names, 2))


# ================================================================
#  The weighting helpers are structural no-ops off an immersed grid
# ================================================================
def test_immersed_helpers_are_noops_off_an_immersed_grid():
    # the parity guard: an unimmersed module never touches the fraction
    # branches (bit-for-bit the pre-I2 path)
    module = CenteredAdvection()
    assert module._immersed is None
    sentinel = object()
    assert module._immersed_flux(sentinel, None) is sentinel
    assert module._immersed_scale(sentinel, None) is sentinel
    assert module._immersed_scale(None, sentinel) is None
    # the immersed-background guard is a no-op off an immersed grid and
    # off a background, in both orders
    assert module._reject_immersed_background(None) is None
    assert CenteredAdvection(
        background={"u": 1.0})._reject_immersed_background(None) is None


# ================================================================
#  background= on an immersed grid is a taught error (defect A3b)
# ================================================================
@pytest.mark.parametrize(
    "cls", [CenteredAdvection, UpwindAdvection, WENOAdvection])
def test_background_on_an_immersed_grid_is_a_taught_error(cls):
    # the Doppler split needs the inhomogeneous body condition
    # u'.n = -U.n, but the immersed projection enforces u'.n = 0 and
    # MaskState zeroes u' in dry cells -- so u' = 0 is an exact fixed
    # point of the step and the body is perfectly transparent to the
    # background flow. That failure is SILENT (a clean run, a flat
    # field), so bind refuses rather than letting it happen. Measured
    # before the guard landed: a disc on a 16^3 grid with
    # background={"u": 1.0} and a zero perturbation start held
    # max|u'| = max|v'| = max|w'| = 0.0 exactly after 20 steps.
    grid = _immersed_grid()
    with pytest.raises(NotImplementedError, match="exact fixed point"):
        FrModel(grid=grid,
                modules=_fv_modules(cls(background={"u": 1.0})),
                time_stepper=AdamBashforth(DT, order=3))


def test_background_binds_on_an_unimmersed_grid():
    # the guard is narrow: the same background on a plain (unimmersed)
    # grid still assembles. Nodal Core -- the background samples resolve
    # on the nh C-grid staggering, so background= and family="fv" do not
    # compose today (an unrelated limitation of _bind_background)
    grid = Grid(tuple(
        IntervalMesh(10, (0.0, TWO_PI), periodic=True, name=nm)
        for nm in ("x", "y", "z")))
    advection = CenteredAdvection(background={"u": 1.0})
    model = FrModel(
        grid=grid,
        modules=(Core(), ConstantStratification(n2=0.0),
                 FPlaneCoriolis(f0=1.0), advection),
        time_stepper=AdamBashforth(DT, order=3))
    (bound,) = [
        m for m in model.modules if isinstance(m, CenteredAdvection)]
    assert bound._immersed is None
    assert bound._background == {"u": 1.0}
