"""Core: the cross-module ramped-aspect-ratio guard (TDF-D4).

The aspect ratio enters the frozen linear operator ``L`` through the
pressure projection, which is a CONSTRAINT stage rather than a
``linear=True`` term, so the structural term sweep cannot see it.
``Core`` reports a ramped ``aspect_ratio`` from its own leaf, closing
the hole a model assembled without stratification would otherwise slip
through. A re-reading stepper (``AdamBashforth``) is unaffected: the
ramped model assembles and advances to finite values.

Also the cheap (assembly-only) projection-routing guards: the auto
preconditioner per route, the stretched-column routing predicate, and
the taught error replacing the bare ``StopIteration`` a ``coords=``
with no staggered velocity face used to raise. The model-level
stretched runs live in the ``test_core_stretched`` shard.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model import term_predicates as terms
from fridom.model.model import _chunk_body
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.core import _stretched_column
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


def _stretched_grid():
    """Build the same grid with a stretched (mapped-mesh) vertical."""
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                        periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                        periodic=True, name="y")
    mz = fr.spatial.meshes.MappedIntervalMesh(
        N, (0.0, 1.0),
        lambda s: s + 0.15 * jnp.sin(2 * np.pi * s) / (2 * np.pi),
        name="z")
    return fr.spatial.Grid((mx, my, mz), device_ids=(0,))


def _renamed_vertical_grid():
    """Build a grid whose vertical mesh is named ``"s"``, not ``"z"``."""
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                        periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                        periodic=True, name="y")
    ms = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), name="s")
    return fr.spatial.Grid((mx, my, ms), device_ids=(0,))


def _model(aspect_ratio, stepper, *, grid=None):
    """Build a small linear nonhydro model (aspect ratio + stepper)."""
    return nh.Model(
        grid=grid if grid is not None else _grid(),
        core=nh.Core(aspect_ratio=aspect_ratio),
        time_stepper=stepper,
        coriolis=nh.FPlaneCoriolis(f0=F0),
        buoyancy=nh.ConstantStratification(n2=N2),
        advection=None)


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
            buoyancy=nh.ConstantStratification(n2=N2),
            advection=None,
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


# ================================================================
#  Projection routing (S4): the auto preconditioner and the
#  stretched-column predicate
# ================================================================
def test_auto_preconditioner_resolves_per_route():
    """None = auto: multigrid on a composed *or* stretched grid.

    A stretched (``MappedIntervalMesh``) column carries no spectral
    basis, so the separable spectral inverse rejects it at construction
    (N1) — auto must not hand it one. An explicit string is honoured on
    every route.
    """
    core = nh.Core()
    assert core._resolved_preconditioner(composed=False) == "spectral"
    assert core._resolved_preconditioner(composed=True) == "multigrid"
    assert core._resolved_preconditioner(
        composed=False, stretched=True) == "multigrid"
    pinned = nh.Core(pressure_preconditioner="none")
    assert pinned._resolved_preconditioner(
        composed=True, stretched=True) == "none"


def test_stretched_column_predicate_detects_a_mapped_mesh():
    """The routing predicate keys on the mesh, not on the mapping."""
    assert _stretched_column(_grid()) is False
    assert _stretched_column(_stretched_grid()) is True


def test_core_refuses_a_coordinate_with_no_staggered_face():
    """A coords= naming a coordinate u/v/w do not stagger on.

    The velocity trio is declared on ``x``, ``y``, ``z``, so a grid
    whose vertical mesh is named ``"s"`` leaves ``"s"`` collocated in
    every component and the projection has no divergence leg there.
    Before the taught error this was a bare ``StopIteration`` out of
    ``next()`` in :meth:`Core.bind`.
    """
    with pytest.raises(ValueError, match="no velocity component is "
                       "staggered along 's'") as ex:
        nh.Model(
            grid=_renamed_vertical_grid(),
            core=nh.Core(vertical="s", coords=("x", "y", "s")),
            time_stepper=AdamBashforth(DT, order=3),
            coriolis=nh.FPlaneCoriolis(f0=F0),
            buoyancy=nh.ConstantStratification(n2=N2),
            advection=None)
    assert "coords=('x', 'y', 's')" in str(ex.value)


# ================================================================
#  Geometry refusal: nonhydro is Cartesian-only (no chart grids)
# ================================================================
def _chart_grid():
    """Build a lat-lon sphere chart grid with a Cartesian vertical."""
    mlon = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                          periodic=True, name="lon")
    mlat = fr.spatial.meshes.IntervalMesh(N, (-0.6, 0.6), name="lat")
    mz = fr.spatial.meshes.IntervalMesh(N // 2, (0.0, 1.0), name="z")
    return fr.spatial.Grid(
        (mlon, mlat, mz),
        mapping=fr.spatial.charts.lonlat_sphere(radius=6.4e6),
        device_ids=(0,))


def _chart_model(coords):
    return nh.Model(
        grid=_chart_grid(),
        core=nh.Core(coords=coords),
        time_stepper=AdamBashforth(DT, order=3),
        buoyancy=nh.ConstantStratification(n2=N2),
        advection=None)


@pytest.mark.parametrize(
    "coords",
    [pytest.param(("x", "y", "z"), id="default-coords"),
     pytest.param(("lon", "lat", "z"), id="chart-coords")],
)
def test_core_refuses_a_chart_grid(coords):
    """Both chart spellings land on ONE taught refusal.

    Before the fence the default ``coords=`` raised a bare ``KeyError``
    out of the pressure space's ``factor("x")`` lookup, and naming the
    chart coordinates reached ``_no_staggered_face``, whose advice
    (rename the vertical mesh to ``"z"``) is wrong here — the vertical
    already *is* ``"z"``; the charted pair is horizontal.
    """
    with pytest.raises(NotImplementedError,
                       match="carries an embedding chart") as ex:
        _chart_model(coords)
    message = str(ex.value)
    assert "('lon', 'lat')" in message
    # the refusal teaches what IS supported, and where a chart model
    # lives instead
    assert "fr.spatial.CoordinateMapping(maps=...)" in message
    assert "sw.Model" in message
    # and it does not fall through to the vertical-naming message
    assert "no velocity component is staggered" not in message


def test_chart_refusal_does_not_catch_a_plain_mapped_column():
    """A per-coordinate map is not a chart: the model still binds.

    ``chart_coords`` is None on a mapping that declares only
    ``maps=`` (a terrain-following / stretched column couples no
    coordinates), so the fence must not reach it.
    """
    grid = _stretched_grid()
    assert grid.chart_coords is None
    model = _model(1.0, AdamBashforth(DT, order=3), grid=grid)
    assert model.state["p"] is not None


# ================================================================
#  A3: the pressure-solver knobs are validated on the core, and the
#  convergence report makes the achieved PCG count observable
# ================================================================
def test_core_validates_the_pressure_solver_knobs():
    """A bad budget / tolerance is a construction error.

    ``ConjugateGradient`` validates the same pair, but it is built at
    stage time inside the jit trace — so a bad value used to surface
    as a ``TermEvaluationError`` wrapping the real message, and on a
    flat (spectral) grid, which iterates nothing, never surfaced at
    all.
    """
    with pytest.raises(TypeError, match="pressure_iterations must be"):
        nh.Core(pressure_iterations=12.0)
    with pytest.raises(ValueError, match="pressure_iterations must be"):
        nh.Core(pressure_iterations=0)
    with pytest.raises(TypeError, match="pressure_tolerance must be"):
        nh.Core(pressure_tolerance="1e-8")
    with pytest.raises(ValueError, match="pressure_tolerance must be"):
        nh.Core(pressure_tolerance=0.0)
    # the accepted pair, including the fixed-count opt-out
    assert nh.Core(pressure_iterations=12)._pressure_iterations == 12
    assert nh.Core(pressure_tolerance=None)._pressure_tolerance is None
    assert nh.Core()._pressure_report is False


def _immersed_model(*, report, iterations=12):
    """Build a tiny immersed (PCG-routed) model with a random state."""
    meshes = tuple(
        fr.spatial.meshes.IntervalMesh(
            N, (0.0, 2 * np.pi), periodic=True, name=nm)
        for nm in ("x", "y", "z"))
    sphere = lambda x, y, z: (  # noqa: E731
        (x - np.pi) ** 2 + (y - np.pi) ** 2
        + (z - np.pi) ** 2 < 1.4 ** 2).astype(float)
    grid = fr.spatial.Grid(
        meshes, immersed=fr.spatial.ImmersedDomain(sphere),
        device_ids=(0,))
    model = nh.Model(
        grid=grid,
        core=nh.Core(pressure_iterations=iterations,
                     pressure_report=report),
        time_stepper=AdamBashforth(1e-2, order=3),
        coriolis=nh.FPlaneCoriolis(f0=F0),
        buoyancy=nh.ConstantStratification(n2=N2),
        advection=None)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        name: 0.2 * rng.standard_normal(model.state[name].data.shape)
        for name in ("u", "v", "w", "b")})
    return model


def _advanced_op_by_op(model, steps):
    """Advance ``steps`` with the pure step kernel evaluated op by op.

    Bitwise claims hold only between identically compiled paths (model
    spec, ``run()``): two DIFFERENT assemblies are two different XLA
    programs, whose fusions contract different mul/add pairs into FMAs,
    so their compiled runs agree to the last bit on some CPUs and
    differ by one ulp on others (CI flake 2026-09-19; reproduced with
    ``XLA_FLAGS=--xla_cpu_use_fusion_emitters=false``). A parity claim
    between two assemblies is a claim about ARITHMETIC, so it is
    checked where there is no fusion to differ.
    """
    with jax.disable_jit():
        return _chunk_body(model._artifacts.record, steps,
                           model._carry, model._stepper).state


def test_pressure_report_prints_the_achieved_iteration_count(capfd):
    """The A3 evidence seam: ``k`` against the budget, per solve.

    The default ``pressure_tolerance`` break fires well inside the
    budget on an immersed route, which is exactly what a user sizing
    ``pressure_iterations`` needs to see — and what nothing exposed
    before. Reporting is a side effect only: the reported run is
    **bitwise** the silent one.
    """
    loud = _immersed_model(report=True)
    loud.advance(2)
    jax.effects_barrier()
    out = capfd.readouterr().out
    lines = [ln for ln in out.splitlines() if "PCG" in ln]
    assert len(lines) == 2, out
    for line in lines:
        assert "ImmersedPressureSolver PCG: k=" in line
        assert "/12" in line
        assert "|r|/|b|=" in line
        assert "tolerance 1e-08" in line
        # the break fires: the achieved count is inside the budget
        achieved = int(line.split("k=")[1].split("/")[0])
        assert 1 <= achieved < 12

    quiet = _immersed_model(report=False)
    quiet.advance(2)
    jax.effects_barrier()
    assert "PCG" not in capfd.readouterr().out
    # the reporting solve and the silent one are two different programs
    # (the callback sits inside the PCG loop), so "a side effect only"
    # is checked op by op (see _advanced_op_by_op)
    loud_state = _advanced_op_by_op(_immersed_model(report=True), 2)
    quiet_state = _advanced_op_by_op(_immersed_model(report=False), 2)
    jax.effects_barrier()
    for name in ("u", "v", "w", "b"):
        assert np.array_equal(np.asarray(loud_state[name].data),
                              np.asarray(quiet_state[name].data))
    assert np.abs(np.asarray(quiet_state["u"].data)).max() > 0.0


def test_pressure_report_keeps_the_run_differentiable():
    """The differentiability policy on the changed step path.

    ``jax.debug.print`` produces no value, so it adds no data path;
    ``jax.grad`` of a quadratic loss through a reporting run must still
    match a central finite difference.
    """
    model = _immersed_model(report=True)
    run = model.propagator(wrt=(str(ASPECT_RATIO),), steps=2)

    def loss(theta):
        state = run(theta).state
        return sum(jnp.sum(state[name].data ** 2)
                   for name in ("u", "v", "w"))

    grad = jax.grad(loss)((1.0,))[0]
    eps = 1e-5
    fd = (loss((1.0 + eps,)) - loss((1.0 - eps,))) / (2 * eps)
    assert np.isfinite(float(grad))
    assert float(grad) == pytest.approx(float(fd), rel=1e-4)
