"""BalanceExpansion: slaving balance on the model eigenmode tiers.

The P3+P4 acceptance battery of ``design/plans/active/nnmd_rewrite_plan.md``:

- order 0 is exactly the package vortical projection (analytic and
  channel tiers);
- order 1 equals an independently assembled Machenhauer state built
  from the ``function(f, sel)`` applicator and a hand polarization;
- the T2 epsilon-slope test: ``residual_series`` of order ``N``
  scales as ``Ro**(N+1)`` over a decade of Rossby numbers;
- nonhydro periodic and channel smokes (real prognostic output,
  residuals decreasing with order);
- the walled/beta first-slice requirement: the f-plane channel
  balances through the labeled families, and a beta-plane channel
  balances through a frequency-predicate slow band;
- the quadraticity lint on the ``nonlinear`` term filter;
- ``residual_fast`` finite, positive, and decreasing with order.
"""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
import fridom.shallowwater2 as sw
from fridom.framework.utils import jaxify
from fridom.model.transforms.balance_expansion import (
    BalanceExpansion,
)

from .conftest import make_model, set_wave_ic

N = 16
SW_COMPONENTS = ("u", "v", "p")
NH_COMPONENTS = ("u", "v", "w", "b")


# ================================================================
#  Model and state builders
# ================================================================
def make_sw_model(*, ro=0.1, periodic_y=True, coriolis=None, n=N):
    """Build a small shallow-water model (walled y if requested)."""
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0), periodic=True,
                                     name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0),
                                     periodic=periodic_y, name="y")
    if coriolis is None:
        coriolis = sw.modules.FPlaneCoriolis(f0=1.0)
    return sw.Model(
        grid=fr.spatial.Grid((mx, my)), csqr=1.0, rossby_number=ro,
        coriolis=coriolis, advection=True,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


def sw_state(model, *, walled=False, n=N):
    """Seed a smooth band-limited ``(u, v, p)`` state."""
    x = (np.arange(n) + 0.5) / n
    gx, gy = np.meshgrid(x, x, indexing="ij")
    if walled:
        model.set_fields(
            u=np.sin(2 * np.pi * gx) * np.sin(np.pi * gy),
            p=0.1 * np.sin(2 * np.pi * gx) * np.cos(np.pi * gy))
    else:
        model.set_fields(
            u=np.sin(2 * np.pi * gx) * np.cos(2 * np.pi * gy),
            v=0.3 * np.cos(2 * np.pi * gx)
            - 0.2 * np.sin(2 * np.pi * gy),
            p=0.1 * np.sin(2 * np.pi * (gx + gy)))
    return sw.State({c: model.state[c] for c in SW_COMPONENTS})


def make_nh_model(*, ro=0.05, walled=None, n=8, **kwargs):
    """Build a small nonhydro model (optionally walled along y)."""
    meshes = tuple(
        fr.spatial.meshes.IntervalMesh(
            n, (0.0, 2 * np.pi), periodic=(name != walled), name=name)
        for name in ("x", "y", "z"))
    return nh.Model(
        grid=fr.spatial.Grid(meshes), dt=0.02, rossby_number=ro,
        dsqr=1.0, coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=4.0), **kwargs)


def nh_state(model, *, names=NH_COMPONENTS, seed=3):
    """Seed a random state on the given nonhydro components."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        c: 0.3 * rng.standard_normal(
            np.asarray(model.state[c].data).shape) for c in names})
    return nh.State({c: model.state[c] for c in NH_COMPONENTS})


def absmax(a, b, components=SW_COMPONENTS):
    """Componentwise max absolute difference of two states."""
    return max(
        float(np.abs(np.asarray(a[c].data)
                     - np.asarray(b[c].data)).max())
        for c in components)


def is_real(state, components):
    """Whether every component carries a real dtype."""
    return all(not np.iscomplexobj(np.asarray(state[c].data))
               for c in components)


@jaxify
class ZeroQuadratic(fr.model.Module):

    """A term selected as nonlinear whose tendency is identically 0."""

    field_declarations = ()

    @fr.model.term(name="zero", advances=("u", "v", "p"))
    def zero(self, state, _ctx):
        return {c: state[c] * 0.0 for c in ("u", "v", "p")}


# ================================================================
#  Shared fixtures (module scope: compile reuse)
# ================================================================
@pytest.fixture(scope="module")
def sw_setup():
    """One periodic shallow-water model + probe state (Ro = 0.1)."""
    model = make_sw_model()
    return model, sw_state(model)


@pytest.fixture(scope="module")
def sw_channel_setup():
    """One walled-y f-plane shallow-water model + probe state."""
    model = make_sw_model(periodic_y=False)
    return model, sw_state(model, walled=True)


# ================================================================
#  1. order 0 == the package vortical projection
# ================================================================
def test_order_zero_is_the_vortical_projection(sw_setup):
    model, z = sw_setup
    bal = BalanceExpansion(model, order=0, lint=False)
    want = sw.transforms.VorticalProjection.from_model(model)(z)
    assert absmax(bal(z), want) < 1e-13
    # the analytic tier also takes the branch-value spelling
    by_branch = BalanceExpansion(model, order=0, slow=0, lint=False)
    assert absmax(by_branch(z), want) < 1e-13


def test_order_zero_on_the_channel_matches_the_labeled_family(
        sw_channel_setup):
    model, z = sw_channel_setup
    bal = BalanceExpansion(model, order=0, lint=False)
    want = sw.transforms.VorticalProjection.from_model(model)(z)
    assert absmax(bal(z), want) < 1e-13


# ================================================================
#  2. order 1 == the hand-assembled Machenhauer state
# ================================================================
def test_order_one_matches_the_hand_machenhauer_state(sw_setup):
    # z1 = v - invLw(W B(v, v)) assembled independently from the
    # function(f, sel) applicator and a direct nonlinear tendency
    model, z = sw_setup
    em = sw.eigenmodes.from_model(model)
    kit = em._kit
    v = sw.transforms.VorticalProjection(em)(z)
    quadratic = model.variant(term_filter=~fr.model.term_predicates.linear)
    b_vv = quadratic.tendency(v, t=0.0, constraints=True)
    coeff = fr.spatial.VectorField({
        c: kit.forward(c)(b_vv[c]) for c in SW_COMPONENTS})
    corr = em.function(lambda w: 1.0 / (1j * w), (1, -1))(coeff)
    want = sw.State({
        c: v[c] - kit.backward(c)(corr[c]).real
        for c in SW_COMPONENTS})
    got = BalanceExpansion(model, order=1, lint=False)(z)
    scale = max(float(np.abs(np.asarray(z[c].data)).max())
                for c in SW_COMPONENTS)
    assert absmax(got, want) / scale < 1e-13


# ================================================================
#  3. the T2 epsilon-slope test (sw periodic, orders 1 and 2)
# ================================================================
def test_epsilon_slope_over_a_rossby_decade():
    rossby = np.geomspace(0.02, 0.2, 4)
    residuals = {1: [], 2: []}
    for ro in rossby:
        model = make_sw_model(ro=float(ro))
        z = sw_state(model)
        for order, res in residuals.items():
            bal = BalanceExpansion(model, order=order, lint=False)
            res.append(bal.residual_series(z))
    for order, res in residuals.items():
        assert np.all(np.isfinite(res))
        slope = float(np.polyfit(np.log(rossby), np.log(res), 1)[0])
        assert slope >= order + 0.6, (
            f"order {order}: slope {slope:.3f}, residuals {res}")


# ================================================================
#  3b. even-grid Nyquist steady strata land in the SLOW set
# ================================================================
def test_nyquist_steady_strata_are_slow():
    # the even-grid interpolation-Nyquist steady modes joined the
    # vortical family: order-0 balance (the slow projection) keeps
    # them verbatim, the internal fast set excludes them (the
    # 1/(i omega) inverse never meets a represented zero
    # frequency), and balance runs at higher orders on a random
    # Nyquist-carrying state with a decreasing series residual
    model = make_sw_model()
    em = sw.eigenmodes.from_model(model)
    _, z = em.mode(0, {"x": N // 2, "y": 3})
    bal = BalanceExpansion(model, order=0, lint=False)
    assert absmax(bal(z), z) < 1e-12
    wave = sw.transforms.WaveProjection(em)(z)
    assert max(float(np.abs(np.asarray(wave[c].data)).max())
               for c in SW_COMPONENTS) < 1e-12
    rng = np.random.default_rng(8)
    shape = np.asarray(model.state["u"].data).shape
    model.set_fields(**{
        c: 0.1 * rng.standard_normal(shape) for c in SW_COMPONENTS})
    zr = sw.State({c: model.state[c] for c in SW_COMPONENTS})
    residuals = [
        BalanceExpansion(model, order=order,
                         lint=False).residual_series(zr)
        for order in (0, 1, 2)]
    assert np.all(np.isfinite(residuals))
    assert residuals[0] > residuals[1] > residuals[2]


# ================================================================
#  4. nonhydro periodic smoke
# ================================================================
def test_nh_periodic_orders_run_and_residual_decreases():
    model = make_nh_model(advection=True)
    z = nh_state(model)
    residuals = []
    for order in (0, 1, 2):
        bal = BalanceExpansion(model, order=order, lint=False)
        out = bal(z)
        assert isinstance(out, nh.State)
        assert out.component_names == NH_COMPONENTS
        assert is_real(out, NH_COMPONENTS)
        residuals.append(bal.residual_series(z))
    assert residuals[0] > residuals[1] > residuals[2]


# ================================================================
#  5. the walled first slice (P4)
# ================================================================
def test_sw_channel_orders_run_and_residual_decreases(
        sw_channel_setup):
    model, z = sw_channel_setup
    residuals = []
    for order in (0, 1, 2):
        bal = BalanceExpansion(model, order=order, lint=False)
        out = bal(z)
        assert is_real(out, SW_COMPONENTS)
        residuals.append(bal.residual_series(z))
    assert residuals[0] > residuals[1] > residuals[2]


def test_sw_beta_channel_balances_a_predicate_slow_band():
    # the genuinely new capability: a beta-plane channel with the
    # slow Rossby band selected by a frequency predicate under the
    # spectral gap (the eb.projector grammar)
    model = make_sw_model(
        periodic_y=False,
        coriolis=sw.modules.BetaPlaneCoriolis(f0=1.0, beta=2.0))
    z = sw_state(model, walled=True)
    eb = sw.eigenbasis(model)
    labels = np.asarray(eb.labels)
    omega = np.asarray(eb.omega)
    wave = np.isin(labels, (eb.families["wave+"],
                            eb.families["wave-"]))
    threshold = 0.5 * float(np.abs(omega[wave]).min())

    def slow_band(om, _labels):
        return jnp.abs(om) < threshold

    r0 = BalanceExpansion(model, order=0, slow=slow_band,
                          lint=False).residual_series(z)
    bal = BalanceExpansion(model, order=1, slow=slow_band,
                           lint=False)
    out = bal(z)
    assert is_real(out, SW_COMPONENTS)
    r1 = bal.residual_series(z)
    assert 0.0 < r1 < r0


def test_nh_channel_orders_run_and_residual_decreases():
    # the walled-y nonhydro channel with the REAL CenteredAdvection
    # (walled-capable: structural-zero wall fluxes) — the payoff of
    # the walled advection support
    model = make_nh_model(walled="y", advection=True)
    z = nh_state(model, seed=5)
    residuals = []
    for order in (0, 1):
        bal = BalanceExpansion(model, order=order)
        out = bal(z)
        assert isinstance(out, nh.State)
        assert is_real(out, NH_COMPONENTS)
        residuals.append(bal.residual_series(z))
    assert np.all(np.isfinite(residuals))
    assert residuals[0] > residuals[1]


# ================================================================
#  6. the quadraticity lint
# ================================================================
def test_lint_warns_when_the_filter_keeps_linear_terms(sw_setup):
    model, _ = sw_setup
    with pytest.warns(UserWarning, match="not quadratic"):
        BalanceExpansion(model, order=0, nonlinear=fr.model.term_predicates.explicit)


def test_lint_is_quiet_on_the_default_filter_and_optout(sw_setup):
    model, _ = sw_setup
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # the default advection-only selection is quadratic
        BalanceExpansion(model, order=0)
        # the opt-out skips the check even on a bad selection
        BalanceExpansion(model, order=0, nonlinear=fr.model.term_predicates.explicit,
                         lint=False)


def test_lint_skips_a_selection_with_zero_nonlinear_tendency():
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), periodic=True,
                                     name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), periodic=True,
                                     name="y")
    quiet = sw.Model(
        grid=fr.spatial.Grid((mx, my)), csqr=1.0, rossby_number=0.1,
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0), advection=True,
        modules_extra=(ZeroQuadratic(),),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BalanceExpansion(
            quiet, order=0,
            nonlinear=fr.model.term_predicates.named("ZeroQuadratic/zero"))


# ================================================================
#  7. residual_fast
# ================================================================
def test_residual_fast_is_positive_and_decreases_with_order(
        sw_setup):
    model, z = sw_setup
    residuals = []
    for order in (0, 1, 2):
        bal = BalanceExpansion(model, order=order, lint=False)
        rho = bal.residual_fast(z)
        assert np.isfinite(rho)
        assert rho > 0.0
        residuals.append(rho)
    assert residuals[0] > residuals[1] > residuals[2]


def test_residuals_vanish_on_degenerate_states(sw_setup):
    model, z = sw_setup
    bal = BalanceExpansion(model, order=1, lint=False)
    zero = sw.State({c: z[c].with_data(jnp.zeros_like(z[c].data))
                     for c in SW_COMPONENTS})
    assert bal.residual_series(zero) == 0.0
    assert bal.residual_fast(zero) == 0.0
    # a steady k = 0 mean-pressure state: F(z_b) = 0 exactly, so
    # the slow rate vanishes and the fast residual is zero
    mean = sw.State({
        c: z[c].with_data(jnp.full_like(
            z[c].data, 0.1 if c == "p" else 0.0))
        for c in SW_COMPONENTS})
    assert bal.residual_fast(mean) == 0.0


# ================================================================
#  8. v1 regression: the old-stack shallow-water NNMD
# ================================================================
def test_matches_the_v1_shallow_water_nnmd():
    # the old-stack NNMD (fr.projection.NNMD, hard-coded to the
    # shallow-water branch pairing — correct for SW) on a matched
    # periodic config, IC transferred via raw arrays; analytic
    # derivatives (use_model=False). The closures first differ at
    # order 3, so orders 0..2 must coincide to numerical precision.
    import fridom.framework as frold  # noqa: PLC0415 — old stack
    import fridom.shallowwater as swold  # noqa: PLC0415 — old stack

    ro = 0.1
    mset = swold.ModelSettings(
        swold.grid.cartesian.Grid(shape=(N, N),
                                  domain_size=(1.0, 1.0)),
        f0=1.0, csqr=1.0, rossby_number=ro,
        time_stepper=swold.time_steppers.AdamBashforth(
            dt=2 ** -6, order=3)).setup()
    model = make_sw_model(ro=ro)
    z_new = sw_state(model)

    z_old = mset.state_constructor()
    halo = (np.asarray(z_old.u.arr).shape[0] - N) // 2
    inner = slice(halo, halo + N)
    for c in SW_COMPONENTS:
        field = getattr(z_old, c)
        arr = jnp.asarray(field.arr)
        field.arr = arr.at[inner, inner].set(
            jnp.asarray(np.asarray(z_new[c].data)))
    z_old = z_old.sync()

    for order in (0, 1, 2):
        nnmd = frold.projection.NNMD(mset, order=order,
                                     use_model=False)
        old_bal = nnmd(z_old)
        new_bal = BalanceExpansion(model, order=order,
                                   lint=False)(z_new)
        diff = max(
            float(np.abs(
                np.asarray(getattr(old_bal, c).arr)[inner, inner]
                - np.asarray(new_bal[c].data)).max())
            for c in SW_COMPONENTS)
        assert diff < 1e-10, f"order {order}: {diff:.3e}"


# ================================================================
#  9. surface, structure and validation errors
# ================================================================
def test_declared_structure_and_repr(sw_setup, sw_channel_setup):
    model, _ = sw_setup
    bal = BalanceExpansion(model, order=2, lint=False)
    assert not bal.traceable
    assert bal.order == 2
    assert bal.domain is None  # analytic tier: polymorphic
    assert bal.codomain is None
    assert bal.eigenmodes is not None
    assert repr(bal) == "BalanceExpansion(order=2, slow='vortical')"
    channel_model, _ = sw_channel_setup
    channel = BalanceExpansion(channel_model, order=0, lint=False)
    assert channel.domain is channel.codomain
    assert channel.domain is not None
    assert channel.domain.grid is channel_model.grid


def test_export_is_public():
    assert fr.model.transforms.BalanceExpansion is BalanceExpansion


def test_validation_errors(sw_setup, sw_channel_setup):
    model, z = sw_setup
    with pytest.raises(ValueError, match="non-negative"):
        BalanceExpansion(model, order=-1, lint=False)
    with pytest.raises(ValueError, match="non-negative"):
        BalanceExpansion(model, order=1.5, lint=False)
    with pytest.raises(ValueError, match="vortical"):
        BalanceExpansion(model, order=0, slow="rossby", lint=False)
    with pytest.raises(TypeError, match="channel"):
        BalanceExpansion(model, order=0, lint=False,
                         slow=lambda om, _labels: jnp.abs(om) < 0.1)
    with pytest.raises(ValueError, match="fast complement"):
        BalanceExpansion(model, order=0, slow=(0, 1, -1), lint=False)
    channel_model, _ = sw_channel_setup
    with pytest.raises(TypeError, match="family name"):
        BalanceExpansion(channel_model, order=0, slow=3.5,
                         lint=False)
    bal = BalanceExpansion(model, order=0, lint=False)
    with pytest.raises(ValueError, match="missing"):
        bal(sw.State({"u": z["u"]}))


def test_dispatch_rejects_a_vocabulary_free_model():
    toy = make_model()
    set_wave_ic(toy)
    with pytest.raises(ValueError, match="state vocabulary"):
        BalanceExpansion(toy, order=0, lint=False)


def test_dispatch_rejects_a_package_without_the_surface(
        monkeypatch, sw_setup):
    model, _ = sw_setup
    monkeypatch.setattr(sw.eigenmodes, "from_model", None)
    with pytest.raises(ValueError, match="from_model"):
        BalanceExpansion(model, order=0, lint=False)
