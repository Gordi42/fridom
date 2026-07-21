"""Stratification: ConstantStratification report + MeridionalStratification.

A time-dependent ``n2`` (a spun-up stratification, ``fr.Ramp``) is a
scalar parameter that advances correctly under AdamBashforth (R1's
scalar path — it is NOT field-materialized, unlike the sw2 ``csqr`` or
the beta-plane ``f(y)``, so it never bare-crashes). But it feeds the
module's ``linear=True`` restoring term, so a frozen-``L`` (exponential)
stepper must refuse it (AR-D7), exactly as R1 does for ``coriolis.f0``.

The ``MeridionalStratification`` law path (TDF-D7) is the field twin: a
``fr.model.ProfileFunction`` law ``n2(y, t, *params)`` drives a
``time_dependent`` ``n2`` field rewritten every substage by a
SELF_UPDATE stage, so a frozen-``L`` (``ETDRK4``) stepper refuses it
automatically while ``AdamBashforth`` runs it. The static-callable
profile path is untouched (no stage, no marker, no ``extra_halo``).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model import term_predicates as terms
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
    MeridionalStratification,
)


def test_static_n2_reports_no_time_dependent_linear_parameter():
    assert ConstantStratification(n2=1.0).time_dependent_linear_parameters() \
        == ()


def test_ramped_n2_reports_the_stratification_parameter():
    ramp = fr.model.Ramp(0.0, 1.0, period=1.0)
    assert ConstantStratification(
        n2=ramp).time_dependent_linear_parameters() == (
        str(fr.model.params.STRATIFICATION_N2),)


def test_ramped_n2_assembles_and_advances_under_adam_bashforth():
    # the scalar Ramp path (spun-up stratification) is untouched: it
    # assembles cleanly and is NOT rejected at construction.
    grid = fr.spatial.Grid((
        fr.spatial.meshes.IntervalMesh(6, (0.0, 1.0), periodic=True,
                                       name="x"),
        fr.spatial.meshes.IntervalMesh(6, (0.0, 1.0), periodic=True,
                                       name="y"),
        fr.spatial.meshes.IntervalMesh(6, (0.0, 1.0), periodic=False,
                                       name="z")))
    model = nh.Model(
        grid=grid,
        core=nh.Core(aspect_ratio=1.0),
        time_stepper=AdamBashforth(1e-3, order=1),
        stratification=ConstantStratification(
            n2=fr.model.Ramp(0.0, 1.0, period=1.0)))
    assert isinstance(
        model.module(ConstantStratification).n2, fr.model.Ramp)
    model.advance(2)


# ================================================================
#  MeridionalStratification: the law-valued n2(y, t) path (TDF-D7)
# ================================================================
NM = 6


def _walled_y_grid(n=NM, device_ids=(0,)):
    """Return a walled-y channel (x/z periodic) the engine serves."""
    return fr.spatial.Grid((
        fr.spatial.meshes.IntervalMesh(n, (0.0, 2 * np.pi),
                                       periodic=True, name="x"),
        fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0),
                                       periodic=False, name="y"),
        fr.spatial.meshes.IntervalMesh(n, (0.0, 2 * np.pi),
                                       periodic=True, name="z")),
        device_ids=device_ids)


def _affine_n2_law(n0=1.0, s=0.5):
    """N^2(y, t) = n0 + s*t + 0.1*y — a strictly positive law."""
    return fr.model.ProfileFunction(
        lambda y, t, n0, s: n0 + s * t + 0.1 * y, params=(n0, s))


def _law_model(grid, law, *, order=1, dt=0.02):
    """Return a linear (advection-off) law-n2 channel; f-plane."""
    return nh.Model(
        grid=grid,
        core=nh.Core(aspect_ratio=1.0),
        time_stepper=AdamBashforth(dt, order=order),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=MeridionalStratification(n2=law),
        advection=False)


def test_meridional_static_carries_no_marker_stage_or_halo():
    # the static-callable profile path is untouched: no time_dependent
    # marker, no SELF_UPDATE stage, no extra_halo substitute, and the
    # frozen-L hook reports nothing (bit-identical assembly fingerprint)
    module = MeridionalStratification(n2=lambda y: 1.0 + y * y)
    assert not module._profile_active
    assert module.extra_halo is None
    assert module.stages == ()
    assert module.time_dependent_linear_parameters() == ()
    decls = {d.name: d for d in module.field_declarations}
    assert not decls["n2"].time_dependent


def test_meridional_rejects_a_constant_teaching_all_spellings():
    # a bare number teaches the static-callable, ProfileFunction and
    # ConstantStratification spellings
    with pytest.raises(TypeError, match="ConstantStratification"):
        MeridionalStratification(n2=2.0)
    with pytest.raises(TypeError, match="ProfileFunction"):
        MeridionalStratification(n2=2.0)


def test_meridional_law_declares_time_dependent_n2_and_a_stage():
    module = MeridionalStratification(n2=_affine_n2_law())
    assert module._profile_active
    assert module.time_dependent_linear_parameters() == ("n2",)
    decls = {d.name: d for d in module.field_declarations}
    assert decls["n2"].time_dependent
    (stage,) = module.stages
    assert stage.kind is fr.model.StageKind.SELF_UPDATE
    assert stage.reads == ("n2",)
    assert stage.writes == ("n2",)


def test_meridional_law_materializes_at_t0_and_tracks_stage_time():
    dt = 0.02
    model = _law_model(_walled_y_grid(), _affine_n2_law(n0=1.0, s=0.5),
                       order=1, dt=dt)
    y = (np.arange(NM) + 0.5) / NM
    # t = 0 snapshot: N^2(y, 0) = 1 + 0.1 y
    np.testing.assert_allclose(
        np.asarray(model.state["n2"].data).ravel(), 1.0 + 0.1 * y,
        atol=1e-13)
    # after a run the SELF_UPDATE stage tracks the stage clock
    model.advance(3)
    t_last = 2 * dt
    np.testing.assert_allclose(
        np.asarray(model.state["n2"].data).ravel(),
        1.0 + 0.5 * t_last + 0.1 * y, atol=1e-12)


def test_meridional_law_etdrk4_refuses_while_adam_bashforth_runs():
    grid = _walled_y_grid()
    # a static-n2 channel supplies the frozen eigenbasis L
    static = nh.Model(
        grid=grid,
        core=nh.Core(aspect_ratio=1.0),
        time_stepper=AdamBashforth(5e-3, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=MeridionalStratification(
            n2=lambda y: 1.0 + 2.0 * y * y),
        advection=False)
    basis = nh.eigenbasis(static)
    law = _affine_n2_law()
    with pytest.raises(
            fr.model.errors.TimeDependentLinearOperatorError, match="n2"):
        nh.Model(
            grid=grid,
            core=nh.Core(aspect_ratio=1.0),
            time_stepper=fr.model.time_steppers.ETDRK4(5e-3, basis),
            coriolis=nh.FPlaneCoriolis(f0=1.0),
            stratification=MeridionalStratification(n2=law),
            advection=False,
            term_filter=~terms.linear)
    # AdamBashforth runs the same law-n2 model to a finite state
    model = _law_model(grid, law, order=3, dt=5e-3)
    rng = np.random.default_rng(1)
    model.set_fields(**{
        c: rng.standard_normal(np.asarray(model.state[c].data).shape)
        for c in ("u", "v", "w", "b")})
    model.advance(4)
    assert all(np.all(np.isfinite(np.asarray(model.state[c].data)))
               for c in ("u", "v", "w", "b"))


def test_meridional_law_actually_changes_the_answer():
    # a time-dependent law run differs from its t=0 freeze
    grid = _walled_y_grid()
    comps = ("u", "v", "w", "b")
    scheduled = _law_model(grid, _affine_n2_law(n0=1.0, s=2.0),
                           order=3, dt=5e-3)
    rng = np.random.default_rng(2)
    init = {c: rng.standard_normal(np.asarray(scheduled.state[c].data).shape)
            for c in comps}
    scheduled.set_fields(**init)
    scheduled.advance(6)
    out = {c: np.asarray(scheduled.state[c].data) for c in comps}

    frozen = _law_model(grid, _affine_n2_law(n0=1.0, s=0.0),
                        order=3, dt=5e-3)
    frozen.set_fields(**init)
    frozen.advance(6)
    assert any(not np.allclose(out[c], np.asarray(frozen.state[c].data))
               for c in comps)


def test_meridional_law_grad_wrt_ic_is_finite_and_matches_fd():
    # TDF-D8: grad through a short law-n2 run w.r.t. the buoyancy IC is
    # finite and FD-matched (the SELF_UPDATE sample is plain — no masked
    # singularity), through the public Model.propagator surface
    grid = _walled_y_grid()
    model = _law_model(grid, _affine_n2_law(n0=1.0, s=0.5), order=1, dt=0.01)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        c: rng.standard_normal(np.asarray(model.state[c].data).shape)
        for c in ("u", "v", "w", "b")})
    run = model.propagator(wrt=("b",), steps=5)
    b0 = model._carry.state["b"].storage

    def loss(field):
        out = run((field,))
        return sum(jnp.sum(f.data ** 2) for f in out.state)

    grad = np.asarray(jax.grad(loss)(b0))
    assert bool(np.all(np.isfinite(grad)))

    direction = jnp.asarray(rng.standard_normal(b0.shape), dtype=b0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(b0 + eps * direction))
          - float(loss(b0 - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


@pytest.mark.multi_device
def test_meridional_law_is_device_count_invariant(forced_devices):
    """Gate (d): the law N^2(y, t) recompute is halo-correct (forced-4)."""
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    law = _affine_n2_law(n0=1.0, s=0.7)
    ref = _law_model(_walled_y_grid(device_ids=(0,)), law, order=3, dt=5e-3)
    rng = np.random.default_rng(4)
    fields = {c: rng.standard_normal(np.asarray(ref.state[c].data).shape)
              for c in ("u", "v", "w", "b")}

    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        model = _law_model(_walled_y_grid(device_ids=device_ids), law,
                           order=3, dt=5e-3)
        model.set_fields(**fields)
        model.advance(5)
        results[tag] = {c: np.asarray(model.state[c].data)
                        for c in ("u", "v", "w", "b")}
    assert max(
        float(np.abs(results["many"][c] - results["one"][c]).max())
        for c in ("u", "v", "w", "b")) < 1e-11
