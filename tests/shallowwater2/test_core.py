"""The shallow-water core (``sw.Core``): constructor and extra_halo.

The dual-variant constructor surface (dimensional ``gravity= + depth=``
XOR nondimensional ``froude_number=``), the derived ``extra_halo`` of
the chart / immersed gravity term (width 1, not a hardcoded 2; on a
plain flat grid the term is traced and no substitute is declared), and
the law-valued ``csqr(y,t)`` ProfileFunction path (TDF-D7). See
``pressure_solver_halo.md``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model import term_predicates as terms
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

N = 8


def _mesh(n, name, *, periodic=True):
    return IntervalMesh(n, (0.0, 1.0), periodic=periodic, name=name)


def _stepper(dt=5e-3, order=3):
    return fr.model.time_steppers.AdamBashforth(dt, order=order)


def _flat_model(**kwargs):
    kwargs.setdefault("time_stepper", _stepper())
    grid = Grid((_mesh(8, "x"), _mesh(8, "y")))
    return sw.Model(grid=grid,
                    core=sw.Core(gravity=1.0, depth=0.7,
                                 coords=("x", "y")),
                    coriolis=None, advection=False, **kwargs)


# ================================================================
#  The dual-variant constructor (fr.scaling seam)
# ================================================================
def test_core_takes_exactly_one_kwarg_set():
    with pytest.raises(TypeError, match="exactly one kwarg set"):
        sw.Core()
    with pytest.raises(TypeError, match="exactly one kwarg set"):
        sw.Core(gravity=1.0, depth=1.0, froude_number=0.2)
    with pytest.raises(TypeError, match="BOTH gravity= and depth="):
        sw.Core(gravity=1.0)


def test_core_refuses_exactly_zero_parameters():
    # the live scaling ratios and analytic consumers divide by them
    with pytest.raises(TypeError, match="gravity=0 is refused"):
        sw.Core(gravity=0.0, depth=1.0)
    with pytest.raises(TypeError, match="depth=0 is refused"):
        sw.Core(gravity=1.0, depth=0.0)
    with pytest.raises(TypeError, match="froude_number=0 is refused"):
        sw.Core(froude_number=0.0)


def test_core_rejects_a_callable_gravity():
    with pytest.raises(TypeError, match="spatial variation"):
        sw.Core(gravity=lambda y: 1.0 + 0.1 * y, depth=1.0)


def test_scaling_variant_and_variable_depth_properties():
    dim = sw.Core(gravity=1.0, depth=0.7)
    assert dim.scaling_variant == "dimensional"
    assert not dim.variable_depth
    assert dim.coords == ("x", "y")
    nondim = sw.Core(froude_number=0.2)
    assert nondim.scaling_variant == "nondimensional"
    varying = sw.Core(froude_number=0.2,
                      depth=lambda y: 1.0 + 0.1 * y)
    assert varying.variable_depth


def test_provides_follow_the_variant():
    # dimensional: GRAVITY always, DEPTH when the depth is a scalar
    dim = {d.name for d in sw.Core(
        gravity=9.81, depth=100.0).parameter_declarations}
    assert dim == {sw.params.GRAVITY, sw.params.DEPTH}
    varying = {d.name for d in sw.Core(
        gravity=1.0,
        depth=lambda y: 1.0 + 0.1 * y).parameter_declarations}
    assert varying == {sw.params.GRAVITY}
    # nondimensional: FROUDE always, DEPTH (the ratio) when constant
    nondim = {d.name for d in sw.Core(
        froude_number=0.2).parameter_declarations}
    assert nondim == {sw.params.FROUDE, sw.params.DEPTH}


def test_thickness_is_a_diagnostic_field_declaration():
    core = sw.Core(gravity=1.0, depth=1.0)
    decls = {d.name: d for d in core.field_declarations}
    assert decls["thickness"].lifecycle is fr.model.Lifecycle.DIAGNOSTIC
    assert tuple(decls) == ("u", "v", "p", "csqr", "thickness")


def test_ramp_gravity_constructs_and_marks_csqr_time_dependent():
    # the old csqr=Ramp TypeError is retired: a Ramp-valued gravity
    # (or depth) is supported — csqr is marked time-dependent and a
    # SELF_UPDATE stage rewrites it with the stage-time g(t) D
    ramp = fr.model.Ramp(1.0, 2.0, period=1.0)
    core = sw.Core(gravity=ramp, depth=0.7)
    decls = {d.name: d for d in core.field_declarations}
    assert decls["csqr"].time_dependent
    assert any(s.kind is fr.model.StageKind.SELF_UPDATE
               for s in core.stages)
    depth_ramp = sw.Core(gravity=1.0, depth=ramp)
    assert any(s.kind is fr.model.StageKind.SELF_UPDATE
               for s in depth_ramp.stages)


# ================================================================
#  Flat grid: fully traced, no substitute declared
# ================================================================
def test_flat_core_declares_no_extra_halo():
    core = _flat_model().module(sw.Core)
    assert core.extra_halo is None


# ================================================================
#  Orthogonal vs non-orthogonal chart: both derive width 1
# ================================================================
def test_non_orthogonal_chart_derives_width_one():
    # a sheared (non-orthogonal) chart: raise_index composes a cross-
    # interpolation, but its window is opposite-biased to the gradient
    # difference and telescopes two-sided back to reach 1 -- so the
    # derived extra_halo is 1, not the naive diff + interp sum of 2
    # (pressure_solver_halo.md; bitwise-verified in that record).
    grid = Grid(
        (_mesh(8, "x"), _mesh(8, "y")),
        mapping=fr.spatial.CoordinateMapping(
            chart={"X": lambda x, y: (x + 0.4 * y, y, 0.0 * x)}))
    model = sw.Model(grid=grid,
                     core=sw.Core(gravity=1.0, depth=0.7,
                                  coords=("x", "y")),
                     coriolis=None, advection=False,
                     time_stepper=_stepper())
    # the cross-interp is genuinely present (a non-diagonal metric)
    assert not grid.mapping.orthogonal
    core = model.module(sw.Core)
    assert dict(core.extra_halo.widths) == {"x": 1, "y": 1}
    assert model.grid.decomposition.halo["x"] == 1
    assert model.grid.decomposition.halo["y"] == 1


# ================================================================
#  Immersed grid: masked continuity is exempt, derives width 1
# ================================================================
def test_immersed_core_derives_width_one():
    grid = Grid((_mesh(8, "x"), _mesh(8, "y")),
                immersed=ImmersedDomain(lambda x, y: x * 0.0 + 1.0))  # noqa: ARG005
    model = sw.Model(grid=grid,
                     core=sw.Core(gravity=1.0, depth=0.7,
                                  coords=("x", "y")),
                     coriolis=None, advection=False,
                     time_stepper=_stepper())
    core = model.module(sw.Core)
    assert core.extra_halo is not None
    assert dict(core.extra_halo.widths) == {"x": 1, "y": 1}


# ================================================================
#  Taught error: chart + immersed is unsupported (silent wrong physics)
# ================================================================
def test_linear_chart_plus_immersed_is_a_taught_error():
    # a grid carrying BOTH a chart and an immersed domain must be
    # refused at bind even for a *linear* model (advection=False, so no
    # SadournyAdvection guard runs): the chart gravity/continuity path
    # is unmasked, so it would silently ignore the immersed mask and let
    # the geopotential flux cross the wet-region boundary. sw2
    # mapped+immersed is a recorded follow-up of the mapped+immersed
    # composition plan. Mirrors the SadournyAdvection.bind guard.
    grid = Grid(
        (_mesh(8, "x"), _mesh(8, "y")),
        mapping=fr.spatial.CoordinateMapping(
            chart={"X": lambda x, y: (x + 0.4 * y, y, 0.0 * x)}),
        immersed=ImmersedDomain(lambda x, y: x * 0.0 + 1.0))  # noqa: ARG005
    with pytest.raises(NotImplementedError,
                       match="BOTH an embedding chart"):
        sw.Model(grid=grid,
                 core=sw.Core(gravity=1.0, depth=0.7,
                              coords=("x", "y")),
                 coriolis=None, advection=False,
                 time_stepper=_stepper())


# ================================================================
#  Parity: derived width 1 reproduces the old width 2 (bitwise)
# ================================================================
def test_non_orthogonal_chart_parity_with_forced_width_two():
    def build():
        grid = Grid(
            (_mesh(8, "x"), _mesh(8, "y")),
            mapping=fr.spatial.CoordinateMapping(
                chart={"X": lambda x, y: (x + 0.4 * y, y, 0.0 * x)}))
        return sw.Model(grid=grid,
                        core=sw.Core(gravity=1.0, depth=0.7,
                                     coords=("x", "y")),
                        coriolis=None, advection=False,
                        time_stepper=_stepper())

    def run():
        m = build()
        rng = np.random.default_rng(0)
        m.set_fields(**{c: 0.05 * rng.standard_normal(m.state[c].shape)
                        for c in ("u", "v", "p")})
        m.run(steps=8, progress=False)
        return {c: np.asarray(m.state[c].data) for c in ("u", "v", "p")}

    derived = run()
    cls = sw.Core
    orig = cls.__dict__.get("extra_halo")
    try:
        cls.extra_halo = property(
            lambda self: (None if self._extra_halo is None
                          else HaloSpec(dict.fromkeys(self._coords, 2))))
        forced = run()
    finally:
        cls.extra_halo = orig
    md = max(float(np.max(np.abs(derived[c] - forced[c])))
             for c in ("u", "v", "p"))
    assert md == 0.0  # storage-frame bitwise identical


# ================================================================
#  Law-valued csqr(y,t): the ProfileFunction path (TDF-D7)
# ================================================================
# A ProfileFunction D(y,t) makes csqr a time_dependent AUXILIARY field
# rewritten each substage by a SELF_UPDATE stage; the gravity divergence
# (and every other csqr consumer) reads the fresh stage-time field. There
# is no constant shallowwater.depth provide, and a frozen-L (ETDRK4)
# stepper refuses the marked field automatically.
def _walled_grid(device_ids=None):
    """Periodic-x / walled-y channel (a periodic axis to shard)."""
    return Grid((_mesh(N, "x"), _mesh(N, "y", periodic=False)),
                device_ids=device_ids)


def _affine_csqr_law(c0=1.0, s=0.5):
    """Return c^2(y,t) = c0 + s*t + 0.1*y (a static profile per t)."""
    return fr.model.ProfileFunction(
        lambda y, t, c0, s: c0 + s * t + 0.1 * y, params=(c0, s))


def _csqr_law_model(law, *, order=3, grid=None, dt=5e-3):
    """Return a linear sw channel with a law-valued csqr (no rotation)."""
    return sw.Model(
        grid=_walled_grid() if grid is None else grid,
        core=sw.Core(gravity=1.0, depth=law, coords=("x", "y")),
        coriolis=None, advection=False,
        time_stepper=_stepper(dt, order))


def _y_coords():
    return (np.arange(N) + 0.5) * (1.0 / N)


def test_csqr_law_declares_a_time_dependent_field_and_no_provide():
    model = _csqr_law_model(_affine_csqr_law())
    core = model.module(sw.Core)
    assert core._depth_law is not None
    decls = {d.name: d for d in core.field_declarations}
    assert decls["csqr"].time_dependent
    provided = {str(p) for p in model.parameters}
    assert "shallowwater.depth" not in provided


def test_csqr_law_declares_the_gravity_halo_on_a_flat_grid():
    # the SELF_UPDATE rewrite exempts the module, so it declares the
    # gravity term's reach itself (one ghost per axis) even flat
    core = _csqr_law_model(_affine_csqr_law()).module(sw.Core)
    assert core.extra_halo is not None
    assert dict(core.extra_halo.widths) == {"x": 1, "y": 1}


def test_csqr_law_materializes_the_law_at_t0():
    model = _csqr_law_model(_affine_csqr_law(c0=1.0, s=0.5))
    field = np.asarray(model.state["csqr"].data).ravel()
    np.testing.assert_allclose(field, 1.0 + 0.1 * _y_coords(), atol=1e-13)


def test_csqr_law_field_equals_the_law_at_stage_time():
    model = _csqr_law_model(_affine_csqr_law(c0=1.0, s=0.5))
    model.advance(3)
    t_last = 2 * 5e-3
    field = np.asarray(model.state["csqr"].data).ravel()
    np.testing.assert_allclose(
        field, 1.0 + 0.5 * t_last + 0.1 * _y_coords(), atol=1e-12)


@pytest.mark.parametrize("t", [0.0, 0.017, 0.05, 0.2])
def test_csqr_law_tendency_matches_static_profile_at_stage_time(t):
    """law.tendency(z, t) == static-profile model, for an affine law.

    c^2(y,t) = c0 + s*t + 0.1*y equals a static profile
    c^2(y) = (c0 + s*t) + 0.1*y at stage time t, so the SELF_UPDATE-fed
    gravity divergence matches the frozen-profile static model.
    """
    grid = _walled_grid()
    c0, s = 1.0, 0.5
    law = _csqr_law_model(_affine_csqr_law(c0, s), order=1, grid=grid)
    rng = np.random.default_rng(5)
    fields = {c: rng.standard_normal(np.asarray(law.state[c].data).shape)
              for c in ("u", "v", "p")}
    law.set_fields(**fields)
    z = sw.State({c: law.state[c] for c in ("u", "v", "p")})
    got = law.tendency(z, t=t)

    static = sw.Model(
        grid=grid,
        core=sw.Core(gravity=1.0,
                     depth=lambda y: c0 + s * t + 0.1 * y,
                     coords=("x", "y")),
        coriolis=None, advection=False,
        time_stepper=_stepper(order=1))
    static.set_fields(**fields)
    z_static = sw.State({c: static.state[c] for c in ("u", "v", "p")})
    want = static.tendency(z_static)
    for c in ("u", "v", "p"):
        np.testing.assert_allclose(
            np.asarray(got[c].data), np.asarray(want[c].data),
            rtol=1e-12, atol=1e-13)


def test_csqr_law_time_dependent_linear_parameters():
    core = _csqr_law_model(_affine_csqr_law()).module(sw.Core)
    assert core.time_dependent_linear_parameters() == ("csqr",)


def test_csqr_law_etdrk4_refuses():
    """A frozen-L (ETDRK4) stepper refuses a scheduled csqr(y,t)."""
    grid = _walled_grid()
    static = sw.Model(
        grid=grid,
        core=sw.Core(gravity=1.0, depth=1.0, coords=("x", "y")),
        coriolis=None, advection=True,
        time_stepper=_stepper())
    basis = sw.eigenbasis(static)
    with pytest.raises(
            fr.model.errors.TimeDependentLinearOperatorError,
            match=r"csqr"):
        sw.Model(
            grid=grid,
            core=sw.Core(gravity=1.0, depth=_affine_csqr_law(),
                         coords=("x", "y")),
            coriolis=None, advection=True,
            time_stepper=fr.model.time_steppers.ETDRK4(5e-3, basis),
            term_filter=~terms.linear)


def test_csqr_law_adam_bashforth_runs_finite():
    law = fr.model.ProfileFunction(
        lambda y, t, a: 1.0 + a * (0.5 + 0.5 * jnp.sin(y + 3.0 * t)),
        params=(0.3,))
    model = _csqr_law_model(law, order=3)
    rng = np.random.default_rng(1)
    model.set_fields(**{c: rng.standard_normal(
        np.asarray(model.state[c].data).shape) for c in ("u", "v", "p")})
    model.advance(6)
    assert all(np.all(np.isfinite(np.asarray(model.state[c].data)))
               for c in ("u", "v", "p"))


def test_csqr_law_actually_changes_the_answer():
    """Sanity: a c^2(y,t) run differs from its t=0 freeze."""
    grid = _walled_grid()
    comps = ("u", "v", "p")
    law = fr.model.ProfileFunction(
        lambda y, t, a: 1.0 + a * (0.5 + 0.5 * jnp.sin(y + 4.0 * t)),
        params=(0.3,))
    scheduled = _csqr_law_model(law, order=3, grid=grid)
    rng = np.random.default_rng(2)
    init = {c: rng.standard_normal(np.asarray(scheduled.state[c].data).shape)
            for c in comps}
    scheduled.set_fields(**init)
    scheduled.advance(6)
    out = {c: np.asarray(scheduled.state[c].data) for c in comps}

    frozen_law = fr.model.ProfileFunction(
        lambda y, t, a: 1.0 + a * (0.5 + 0.5 * jnp.sin(y)),  # noqa: ARG005
        params=(0.3,))
    frozen = _csqr_law_model(frozen_law, order=3, grid=grid)
    frozen.set_fields(**init)
    frozen.advance(6)
    assert any(not np.allclose(out[c], np.asarray(frozen.state[c].data))
               for c in comps)


@pytest.mark.multi_device
def test_csqr_law_is_device_count_invariant(forced_devices):
    """Gate (d): the c^2(y,t) recompute is halo-correct (forced-4)."""
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    law = fr.model.ProfileFunction(
        lambda y, t, a: 1.0 + a * (0.5 + 0.5 * jnp.sin(y + 4.0 * t)),
        params=(0.3,))
    rng = np.random.default_rng(4)
    fields = {"u": rng.standard_normal((N, N)),
              "v": rng.standard_normal((N, N - 1)),
              "p": rng.standard_normal((N, N))}
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        model = _csqr_law_model(law, order=3,
                                grid=_walled_grid(device_ids=device_ids))
        model.set_fields(**fields)
        model.advance(5)
        results[tag] = {c: np.asarray(model.state[c].data)
                        for c in ("u", "v", "p")}
        if tag == "many":
            assert model.state["u"]._data.sharding.spec[0] == "devices"
    assert max(
        float(np.abs(results["many"][c] - results["one"][c]).max())
        for c in ("u", "v", "p")) < 1e-11
