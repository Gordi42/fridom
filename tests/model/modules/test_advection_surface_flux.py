r"""H7 surface-flux slice form: equivalence, constancy, differentiability.

Prefix-mirrored shard of ``advection.py`` (boundary_trace_plan.md §6).
Covers the constancy-preserving surface closure's 2D slice lowering of
:math:`A(\mathbf 1)`: its exact equivalence with the pre-slice full-3D
form (the load-bearing check), the constancy gate, and the step-path
autodiff regression (both the flat divide and the immersed wet-volume
seal). Self-contained (import-mode=importlib): the small builders are
duplicated from the hydrostatic advection shard.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.model import _chunk_body
from fridom.model.modules import advection as adv_mod
from fridom.model.modules.advection import (
    CenteredAdvection,
    UpwindAdvection,
    WENOAdvection,
)
from fridom.model.term_predicates import owned_by
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.immersed_domain import ImmersedDomain

IM = fr.spatial.meshes.IntervalMesh
MIM = fr.spatial.meshes.MappedIntervalMesh


# ================================================================
#  Builders
# ================================================================
def _wavy(s):
    return s + 0.12 * jnp.sin(2.0 * jnp.pi * s) / (2.0 * jnp.pi)


def _depth(x, y):
    return 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def _flat_bottom(x, y, z):  # noqa: ARG001 — x, y unused (flat bottom)
    return (z > 0.4).astype(float)


def _sidewall(x, y, z):  # noqa: ARG001 — y unused (x-partial side wall)
    return ((z > 0.3) | (x > 0.5)).astype(float)


class _FVTracer(fr.model.Module):

    """A passive CellAvg tracer ``c``, advected only (FV path)."""

    @property
    def field_declarations(self):
        return (fr.model.FieldDeclaration.tracer(
            "c", space=fr.spatial.Collocated(family="fv"),
            long_name="fv tracer", units="1"),)


def _grid(kind, nx=6, nz=5):
    gkw = {}
    zmesh = IM(nz, (0.0, 1.0), periodic=False, name="z")
    if kind == "stretched":
        zmesh = MIM(nz, (0.0, 1.0), _wavy, periodic=False, name="z")
    elif kind == "terrain":
        gkw["mapping"] = CoordinateMapping(
            maps={"zp": lambda z, H: z * H}, params={"H": _depth})
    elif kind == "immersed_flat":
        gkw["immersed"] = ImmersedDomain(_flat_bottom, min_fraction=0.1)
    elif kind == "immersed_side":
        gkw["immersed"] = ImmersedDomain(_sidewall, min_fraction=0.1)
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"), zmesh), **gkw)


def _model(kind, *, fv_tracer=True, surface_flux=True):
    extra = [_FVTracer()] if fv_tracer else []
    return hy.Model(
        grid=_grid(kind), dt=1e-3, csqr=1.0,
        stratification=hy.ConstantStratification(n2=1.0),
        advection=CenteredAdvection(surface_flux=surface_flux),
        modules_extra=extra)


def _weno_model():
    # WENO5 needs order+1 = 6 cells on the walled z axis; no FV tracer
    # (the biased family opts out of immersed/FV — nodal momentum only).
    grid = fr.spatial.Grid((
        IM(6, (0.0, 1.0), periodic=True, name="x"),
        IM(6, (0.0, 1.0), periodic=True, name="y"),
        IM(6, (0.0, 1.0), periodic=False, name="z")))
    return hy.Model(
        grid=grid, dt=1e-3, csqr=1.0,
        stratification=hy.ConstantStratification(n2=1.0),
        advection=WENOAdvection(order=5))


class _Ctx:
    params = {fr.model.params.SCALING_NONLINEARITY: 1.0}  # noqa: RUF012


def _diagnosed_state(model, seed=0):
    """Return a state with a divergence-free diagnosed ``w``."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        k: 0.1 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "b", "c") if k in model.state})
    core = model.module(hy.HydrostaticCore)
    return model.state.replace(w=core._diagnose_w(model.state, _Ctx)["w"])


def _module(model):
    return next(m for m in model.modules
                if isinstance(m, CenteredAdvection))


def _pre_slice_advect(module, state):
    """Return the pre-slice full-3D ``_advect`` (the reference)."""
    ro = _Ctx.params[fr.model.params.SCALING_NONLINEARITY]
    params = module._geometry_params(state)
    out = {}
    for qname in module._advected:
        q = state[qname]
        tend = ro * module._immersed_scale(
            module._transport(state, q, params), q)
        if module._surface_flux_on:
            corr = module._correction_full(state, q, params)
            tend = tend - q * (ro * module._immersed_scale(corr, q))
        out[qname] = tend
    return out


# ================================================================
#  Test 1: equivalence with the pre-slice full-3D form (load-bearing)
# ================================================================
@pytest.mark.parametrize("kind", [
    "uniform", "stretched", "terrain", "immersed_flat", "immersed_side"])
@pytest.mark.parametrize("lowering", ["scatter", "embed"])
def test_slice_matches_the_pre_slice_full_form(kind, lowering, monkeypatch):
    # the slice correction reproduces the pre-slice full-3D
    # ``-q * A(1)`` to rtol ~1e-13 on every cell (the slice makes the
    # interior telescoping noise exact zeros; the boundary row agrees).
    # terrain / immersed side-wall momentum take the exact full-3D
    # fallback, so they agree byte-for-byte there.
    monkeypatch.setattr(adv_mod, "_SURFACE_FLUX_LOWERING", lowering)
    model = _model(kind)
    module = _module(model)
    assert module._surface_flux_on is True
    state = _diagnosed_state(model)
    new = module._advect(state, _Ctx)
    old = _pre_slice_advect(module, state)
    worst = 0.0
    for qname in module._advected:
        a = np.asarray(new[qname].data)
        b = np.asarray(old[qname].data)
        scale = np.abs(b).max() + 1e-30
        worst = max(worst, np.abs(a - b).max() / scale)
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-13)
    assert worst < 1e-12


@pytest.mark.parametrize("kind", ["uniform", "stretched", "immersed_flat"])
def test_slice_correction_interior_is_exactly_zero(kind):
    # for a slice-form component the new correction only ever touches the
    # boundary (top) row: tend - base is exactly zero in the interior
    # (the slice makes the pre-slice ~1e-16 telescoping noise exact
    # zeros). A staggered momentum component on an immersed grid takes
    # the full-3D fallback (_slice_valid is False), so it is exempt.
    model = _model(kind)
    module = _module(model)
    state = _diagnosed_state(model)
    ro = _Ctx.params[fr.model.params.SCALING_NONLINEARITY]
    params = module._geometry_params(state)
    new = module._advect(state, _Ctx)
    checked = 0
    for qname in module._advected:
        q = state[qname]
        if not module._slice_valid(q):
            continue
        base = ro * module._immersed_scale(
            module._transport(state, q, params), q)
        contrib = np.asarray(new[qname].data) - np.asarray(base.data)
        # every row except the top (HIGH, index -1) is untouched
        assert np.abs(contrib[..., :-1]).max() == 0.0, qname
        assert np.abs(contrib[..., -1]).max() > 1e-6, qname
        checked += 1
    assert checked > 0


def _advect_under(module, state, lowering, monkeypatch):
    """Return ``module._advect`` with a forced global lowering."""
    with monkeypatch.context() as m:
        m.setattr(adv_mod, "_SURFACE_FLUX_LOWERING", lowering)
        return module._advect(state, _Ctx)


def test_weno_lowering_equivalence(monkeypatch):
    # the biased/upwind family (WENO) takes the ``embed`` lowering of the
    # surface correction (nodal momentum). Both lowerings feed the same
    # 2D boundary term through ``_apply_correction``, so scatter and embed
    # give the same tendency — and the ``None`` default (embed for WENO)
    # matches the forced-embed result. Small walled grid; the slice ->
    # full equivalence is a separate property tested (centered) above.
    model = _weno_model()
    module = next(m for m in model.modules
                  if isinstance(m, WENOAdvection))
    assert module._surface_flux_on is True
    assert module._surface_flux_lowering == "embed"
    state = _diagnosed_state(model)
    scatter = _advect_under(module, state, "scatter", monkeypatch)
    embed = _advect_under(module, state, "embed", monkeypatch)
    default = module._advect(state, _Ctx)  # None -> per-scheme embed
    for qname in module._advected:
        s = np.asarray(scatter[qname].data)
        e = np.asarray(embed[qname].data)
        d = np.asarray(default[qname].data)
        np.testing.assert_allclose(s, e, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(d, e, rtol=1e-13, atol=1e-13)


def test_lowering_default_resolves_per_scheme():
    # with the module-level override at its ``None`` default, each scheme
    # resolves its own ``_surface_flux_lowering`` ClassVar: scatter for
    # the centered flux form, embed for the biased/upwind family.
    assert adv_mod._SURFACE_FLUX_LOWERING is None
    assert CenteredAdvection()._surface_flux_lowering == "scatter"
    assert UpwindAdvection(order=3)._surface_flux_lowering == "embed"
    assert WENOAdvection(order=5)._surface_flux_lowering == "embed"


# ================================================================
#  Test 2: the constancy gate
# ================================================================
_DIVERGENT = {
    "u": lambda x, y, z: 0.3 * np.sin(2 * np.pi * x) + 0.0 * (y + z),
    "v": lambda x, y, z: 0.3 * np.sin(2 * np.pi * y) * (1 + 0.5 * z)
    + 0.0 * x,
}


@pytest.mark.parametrize("kind", [
    "uniform", "stretched", "immersed_flat", "immersed_side"])
def test_constancy_gate_slice_form(kind):
    # a constant tracer under a horizontally-divergent hydrostatic
    # velocity: the closure advects through the top face, so A(const)
    # is machine-zero in every cell (the surface cell included).
    model = _model(kind, fv_tracer=False)
    grid = model.grid
    inits = dict(_DIVERGENT)
    inits["b"] = lambda x, y, z: 2.5 + 0.0 * (x + y + z)
    data = {name: grid.create_field(
        model.state[name].function_space, init=init).data
        for name, init in inits.items()}
    model.set_fields(**data)
    tb = np.asarray(model.tendency(
        model.state, filter=owned_by(CenteredAdvection))["b"].data)
    assert np.max(np.abs(tb)) <= 1e-13


# ================================================================
#  Test 3: autodiff regression (differentiability policy)
# ================================================================
def _grad_matches_fd(model, seed_init, seed_dir):
    """Grad of a quadratic loss through _chunk_body vs central FD."""
    rng = np.random.default_rng(seed_init)
    model.set_fields(**{
        k: 0.1 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "b")})
    record = model._artifacts.record
    carry, stepper = model._carry, model._stepper
    b_leaf = carry.state["b"].storage
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is b_leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        spliced = jax.tree_util.tree_unflatten(treedef, new)
        final = _chunk_body(record, 6, spliced, stepper)
        return sum(jnp.sum(f.data ** 2) for f in final.state)

    grad = np.asarray(jax.grad(loss)(b_leaf))
    assert bool(np.all(np.isfinite(grad)))
    rng = np.random.default_rng(seed_dir)
    direction = jnp.asarray(rng.standard_normal(b_leaf.shape),
                            dtype=b_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(b_leaf + eps * direction))
          - float(loss(b_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


def test_surface_flux_slice_grad_matches_fd_stretched():
    # the slice divide (_safe_ratio on the top-cell width) stays finite
    # and FD-exact through a short stretched-grid run with the closure on.
    grid = fr.spatial.Grid((
        IM(8, (0.0, 1.0), periodic=True, name="x"),
        IM(8, (0.0, 1.0), periodic=True, name="y"),
        MIM(4, (0.0, 1.0), _wavy, periodic=False, name="z")))
    model = hy.Model(
        grid=grid, dt=2e-3, csqr=1.0,
        stratification=hy.ConstantStratification(n2=1.0),
        advection=CenteredAdvection(surface_flux=True))
    _grad_matches_fd(model, seed_init=3, seed_dir=7)


def _dry_top(x, y, z):  # noqa: ARG001 — y unused (lid depends on x, z)
    # a partial immersed lid: the top cell is dry where x > 0.5
    return jnp.where((z < 0.75) | (x < 0.5), 1.0, 0.0)


def test_surface_flux_slice_grad_matches_fd_immersed_sealed_top():
    # the immersed wet-volume seal (_immersed_scale_2d, the dry top cell
    # theta == 0 guard) must keep jax.grad finite: a partial dry top cell
    # divides by zero in the numerator-identically-zero branch.
    grid = fr.spatial.Grid(
        (IM(8, (0.0, 1.0), periodic=True, name="x"),
         IM(8, (0.0, 1.0), periodic=True, name="y"),
         IM(4, (0.0, 1.0), periodic=False, name="z")),
        immersed=ImmersedDomain(_dry_top, min_fraction=0.1))
    model = hy.Model(
        grid=grid, dt=2e-3, csqr=1.0,
        stratification=hy.ConstantStratification(n2=1.0),
        advection=CenteredAdvection(surface_flux=True))
    _grad_matches_fd(model, seed_init=5, seed_dir=11)


# ================================================================
#  Test 4: the biased family (Upwind / WENO) surface closure
# ================================================================
# The slice traces the surface ``w`` and relocates it horizontally onto
# ``q``'s flux column with the plain two-point ``.to``. A staggered
# momentum component's column is a face node set, so that relocation is a
# genuine half-cell interpolation — and the biased schemes advect the
# velocity with an ``(order - 1)``-point centered interpolation, which
# only coincides with the two-point ``.to`` at ``order == 3``. At
# ``order == 5`` the slice's top-row ``A(1)`` for u / v uses the wrong
# surface ``w`` (a real constancy break, ~15-17% of the top-row momentum
# tendency), so staggered momentum takes the exact full-3D fallback;
# a cell-collocated tracer (buoyancy) needs no relocation and keeps the
# cheap slice for every scheme.
def _biased_model(scheme):
    # WENO/Upwind order 5 needs order + 1 = 6 walled z cells; the biased
    # family is nodal-momentum only (no FV tracer / immersed).
    grid = fr.spatial.Grid((
        IM(6, (0.0, 1.0), periodic=True, name="x"),
        IM(6, (0.0, 1.0), periodic=True, name="y"),
        IM(6, (0.0, 1.0), periodic=False, name="z")))
    return hy.Model(
        grid=grid, dt=1e-3, csqr=1.0,
        stratification=hy.ConstantStratification(n2=1.0),
        advection=scheme)


def _biased_module(model, cls):
    return next(m for m in model.modules if isinstance(m, cls))


def _ones_on(q):
    return q.with_data(jnp.ones_like(q.data))


_BIASED = [
    pytest.param(lambda: UpwindAdvection(order=3, surface_flux=True),
                 UpwindAdvection, id="upwind3"),
    pytest.param(lambda: UpwindAdvection(order=5, surface_flux=True),
                 UpwindAdvection, id="upwind5"),
    pytest.param(lambda: WENOAdvection(order=3, surface_flux=True),
                 WENOAdvection, id="weno3"),
    pytest.param(lambda: WENOAdvection(order=5, surface_flux=True),
                 WENOAdvection, id="weno5"),
]


@pytest.mark.parametrize(("make", "cls"), _BIASED)
def test_biased_advect_matches_full_form(make, cls):
    # the routed ``_advect`` (slice for the collocated tracer, exact
    # full-3D for staggered momentum) reproduces the pre-slice full-3D
    # correction on every cell. On unfixed dev this FAILS for order-5
    # momentum (the slice's top row is wrong by ~1.3).
    model = _biased_model(make())
    module = _biased_module(model, cls)
    assert module._surface_flux_on is True
    state = _diagnosed_state(model)
    new = module._advect(state, _Ctx)
    old = _pre_slice_advect(module, state)
    for qname in module._advected:
        a = np.asarray(new[qname].data)
        b = np.asarray(old[qname].data)
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-13)


def test_biased_momentum_routes_to_full_fallback():
    # a staggered momentum component (u / v) under a biased scheme takes
    # the exact full-3D fallback; the cell-collocated tracer (b) keeps the
    # cheap slice. The centered scheme relocates with the same two-point
    # ``.to``, so all its components stay on the slice.
    for make, cls in (
            (lambda: UpwindAdvection(order=3, surface_flux=True),
             UpwindAdvection),
            (lambda: UpwindAdvection(order=5, surface_flux=True),
             UpwindAdvection),
            (lambda: WENOAdvection(order=5, surface_flux=True),
             WENOAdvection)):
        model = _biased_model(make())
        module = _biased_module(model, cls)
        state = _diagnosed_state(model)
        assert module._slice_valid(state["u"]) is False
        assert module._slice_valid(state["v"]) is False
        assert module._slice_valid(state["b"]) is True
    cmodel = _biased_model(CenteredAdvection(surface_flux=True))
    cmod = _biased_module(cmodel, CenteredAdvection)
    cstate = _diagnosed_state(cmodel)
    for qname in cmod._advected:
        assert cmod._slice_valid(cstate[qname]) is True


@pytest.mark.parametrize(("make", "cls"), _BIASED)
def test_biased_full_correction_preserves_constancy(make, cls):
    # the exact constancy oracle: the correction the fallback subtracts
    # (``_correction_full``) equals the scheme applied to a constant
    # field (``_transport`` of ones) on every component. That identity is
    # what makes A annihilate a constant in every cell -- the surface cell
    # included -- for the biased family, independent of the slice.
    model = _biased_model(make())
    module = _biased_module(model, cls)
    state = _diagnosed_state(model)
    params = module._geometry_params(state)
    for qname in module._advected:
        q = state[qname]
        a1_true = np.asarray(
            module._transport(state, _ones_on(q), params).data)
        corr_full = np.asarray(
            module._correction_full(state, q, params).data)
        scale = np.abs(a1_true).max() + 1e-30
        np.testing.assert_allclose(
            corr_full, a1_true, rtol=0, atol=1e-12 * scale)


def test_biased_naive_slice_would_break_momentum_constancy():
    # regression pinning the mechanism: the raw slice boundary term (the
    # value the fallback now bypasses) differs from the true A(1) at the
    # top row for order-5 biased momentum, and matches it for buoyancy and
    # for order 3. Guards the fallback against being dropped.
    for make, cls, broken in (
            (lambda: UpwindAdvection(order=3, surface_flux=True),
             UpwindAdvection, False),
            (lambda: WENOAdvection(order=5, surface_flux=True),
             WENOAdvection, True)):
        model = _biased_model(make())
        module = _biased_module(model, cls)
        state = _diagnosed_state(model)
        params = module._geometry_params(state)
        for qname, staggered in (("u", True), ("b", False)):
            q = state[qname]
            a1_true = np.asarray(
                module._transport(state, _ones_on(q), params).data)
            a1 = module._surface_boundary_term(q, state["w"], "z")
            naive = np.asarray(a1.embed("z").data)
            err = np.abs(naive - a1_true).max()
            if broken and staggered:
                assert err > 1e-2  # the shipped-dev slice was wrong here
            else:
                assert err <= 1e-10


def test_surface_flux_weno_grad_matches_fd():
    # differentiability policy: the routing change sends biased momentum
    # through the full-3D correction; jax.grad through a short WENO5 run
    # with the surface closure on stays finite and FD-exact.
    grid = fr.spatial.Grid((
        IM(6, (0.0, 1.0), periodic=True, name="x"),
        IM(6, (0.0, 1.0), periodic=True, name="y"),
        IM(6, (0.0, 1.0), periodic=False, name="z")))
    model = hy.Model(
        grid=grid, dt=1e-3, csqr=1.0,
        stratification=hy.ConstantStratification(n2=1.0),
        advection=WENOAdvection(order=5, surface_flux=True))
    _grad_matches_fd(model, seed_init=4, seed_dir=9)
