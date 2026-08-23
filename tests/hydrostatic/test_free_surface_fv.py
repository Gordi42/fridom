r"""The free-surface family on the finite-volume (cell-average) family.

Prefix-mirrored shard of ``hy.modules.free_surface`` (AGENTS
oversized-module rule) covering stage F3: all three variants —
``ExplicitFreeSurface``, ``ImplicitFreeSurface``,
``SplitExplicitFreeSurface`` — on the FV C-grid, over the flat, walled
and terrain (sigma) geometries. The gates:

- the barotropic ``ps`` and the split transports land on the FV
  C-grid spaces (``CellAvg`` cells, point-value faces — FV-D2 option
  A), and the split transport space still equals the runtime depth
  mean it is written from;
- the explicit variant's barotropic pair stays exactly skew-adjoint
  under the hydrostatic energy metric (the H2 energy gate);
- the implicit variant's 2-D Helmholtz solve builds on the ``CellAvg``
  origins (the walled solve expands on the Neumann-tagged cell
  average) and conserves ``int ps``;
- the split-explicit subcycle conserves ``int ps`` and the tracer
  mass;
- and, because the FV and nodal 2nd-order stencils are bit-identical,
  every one of those runs is **bitwise** its nodal twin.

Self-contained builders (AGENTS oversized-module rule).
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.spaces.average import AverageSpace, CellAvg

IM = fr.spatial.meshes.IntervalMesh
G, DT, N, NZ = 2.0, 2e-3, 8, 4


def _depth(x, y):
    return 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def _meshes(n=N, nz=NZ, *, walled=False):
    return (IM(n, (0.0, 1.0), periodic=not walled, name="x"),
            IM(n, (0.0, 1.0), periodic=not walled, name="y"),
            IM(nz, (-1.0, 0.0), periodic=False, name="z"))


def _grid(kind, n=N, nz=NZ):
    if kind == "terrain":
        return fr.spatial.Grid(
            _meshes(n, nz),
            mapping=CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                      params={"H": _depth}))
    return fr.spatial.Grid(_meshes(n, nz, walled=(kind == "walled")))


VARIANTS = {
    "explicit": hy.ExplicitFreeSurface,
    "implicit": hy.ImplicitFreeSurface,
    "split": lambda: hy.SplitExplicitFreeSurface(substeps=4),
}


def _model(kind, variant, family, *, n2=1.0, f0=0.0, dt=DT, n=N, nz=NZ):
    return hy.Model(
        grid=_grid(kind, n, nz),
        core=hy.Core(gravity=G, family=family),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=hy.FPlaneCoriolis(f0=f0) if f0 else None,
        buoyancy=hy.ConstantStratification(n2=n2),
        free_surface=VARIANTS[variant](),
        advection=None)


def _seed(model, seed=3):
    rng = np.random.default_rng(seed)
    model.set_fields(
        u=0.1 * rng.standard_normal(model.state["u"].shape),
        v=0.1 * rng.standard_normal(model.state["v"].shape),
        b=0.1 * rng.standard_normal(model.state["b"].shape),
        ps=0.05 * rng.standard_normal(model.state["ps"].shape))
    return model


# ================================================================
#  The FV barotropic spaces
# ================================================================
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_ps_lands_on_the_cell_average(variant):
    model = _model("flat", variant, "fv")
    ps = model.state["ps"].function_space.bare
    assert isinstance(ps.factor("x"), CellAvg)
    assert isinstance(ps.factor("y"), CellAvg)


def test_split_transports_are_option_a_faces():
    # FV-D2 option A: point value along the transport's own axis, cell
    # average transversely -- and still exactly the space the substage
    # snapshot writes (``u.mean(z)``), which the assembly checks.
    model = _model("flat", "split", "fv")
    assert str(model.state["U"].function_space.bare) \
        == "Right(x) ⊗ CellAvg(y) ⊗ Constant(z)"
    assert str(model.state["V"].function_space.bare) \
        == "CellAvg(x) ⊗ Right(y) ⊗ Constant(z)"
    for name, comp in (("U", "u"), ("V", "v")):
        assert (model.state[name].function_space
                == model.state[comp].mean("z").function_space), name
        buf = {"U": "ubar_prev", "V": "vbar_prev"}[name]
        assert (model.state[buf].function_space
                == model.state[name].function_space)


def test_walled_split_transport_keeps_the_wall_tag_on_fv():
    model = _model("walled", "split", "fv")
    assert "DIRICHLET" in repr(model.state["U"].function_space.bare)
    assert (model.state["U"].function_space
            == model.state["u"].mean("z").function_space)


# ================================================================
#  Gate 4a: the explicit variant's skew-adjoint barotropic pair
# ================================================================
@pytest.mark.parametrize("kind", ["flat", "walled"])
def test_linear_energy_is_conserved_to_roundoff_on_fv(kind):
    # flat / walled only: the terrain barotropic weight is the
    # PHYSICAL column depth int(J dz), not the flat g*H used here (the
    # nodal terrain shard carries that variant); the terrain FV run is
    # covered bitwise against its nodal twin below.
    n2 = 2.0
    model = _seed(_model(kind, "explicit", "fv", n2=n2, f0=1.3))
    state = model.state
    dX = model.tendency(state)
    p3 = state["b"].function_space
    csqr = G * 1.0  # gravity * the (unit) column depth

    def integral(field):
        return float(field.integrate().data.ravel()[0])

    terms = [
        integral(state["u"] * dX["u"]),
        integral(state["v"] * dX["v"]),
        integral((state["b"] / n2) * dX["b"]),
        integral((state["ps"].to(p3) / csqr) * dX["ps"].to(p3)),
    ]
    skew = sum(terms)
    scale = sum(abs(t) for t in terms)
    assert abs(skew) < 1e-12 * scale


# ================================================================
#  Gate 4b: the implicit solve on the FV origins
# ================================================================
@pytest.mark.parametrize("kind", ["flat", "walled", "terrain"])
def test_implicit_solve_conserves_the_surface_pressure_on_fv(kind):
    # d/dt int(ps) = -g int(T*) = 0 (the transport divergence
    # telescopes), so the measure-weighted ps integral drifts only at
    # round-off through the 2-D Helmholtz solve -- on the CellAvg
    # origins the walled solve expands on (the Neumann-tagged cell
    # average is the DCT-II sampling grid the nodal Center is).
    model = _seed(_model(kind, "implicit", "fv", n2=1.0, f0=0.5,
                         dt=1e-2))

    def total(m):
        return float(np.asarray(
            m.state["ps"].integrate("x", "y").data).ravel()[0])

    before = total(model)
    model.advance(10)
    assert not model.panicked
    assert abs(total(model) - before) < 1e-12 * max(abs(before), 1e-3)


# ================================================================
#  Gate 4c: the split-explicit subcycle on FV
# ================================================================
def test_split_subcycle_conserves_ps_mean_and_tracer_mass_on_fv():
    model = hy.Model(
        grid=_grid("flat", 16, 4),
        core=hy.Core(gravity=4.0, family="fv"),
        time_stepper=AdamBashforth(1e-2, order=3),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        buoyancy=hy.ConstantStratification(n2=1.0),
        free_surface=hy.SplitExplicitFreeSurface(substeps=16),
        advection=None)
    rng = np.random.default_rng(2)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        b=rng.standard_normal(model.state["b"].shape),
        ps=rng.standard_normal(model.state["ps"].shape))

    def ps_mean(m):
        return float(np.asarray(
            m.state["ps"].mean("x", "y").data).ravel()[0])

    def b_mass(m):
        return float(np.asarray(
            m.state["b"].integrate("x", "y", "z").data).ravel()[0])

    pm0, bm0 = ps_mean(model), b_mass(model)
    model.advance(20)
    assert not model.panicked
    assert abs(ps_mean(model) - pm0) < 1e-12
    assert abs(b_mass(model) - bm0) / max(abs(bm0), 1e-30) < 1e-12


@pytest.mark.parametrize("kind", ["flat", "terrain"])
def test_split_subcycle_runs_on_fv(kind):
    model = _seed(_model(kind, "split", "fv", dt=1e-2))
    model.advance(6)
    assert not model.panicked
    for name in ("u", "v", "b", "ps", "U", "V"):
        assert bool(np.all(np.isfinite(
            np.asarray(model.state[name].data)))), name


# ================================================================
#  Gate 2 (free-surface half): bitwise parity with the nodal family
# ================================================================
@pytest.mark.parametrize("kind", ["flat", "walled", "terrain"])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_free_surface_run_is_bitwise_the_nodal_one(kind, variant):
    # every 2nd-order stencil the barotropic path uses -- the transport
    # divergence (flux_diff), the -grad ps force (FaceDifference), the
    # Div @ Diag @ Grad solve chain and the subcycle -- carries the
    # nodal numbers on the average family
    out = {}
    for family in ("fv", "nodal"):
        model = _seed(_model(kind, variant, family, n2=1.0, f0=0.5,
                             dt=1e-2))
        model.advance(6)
        assert not model.panicked
        out[family] = {c: np.asarray(model.state[c].data)
                       for c in ("u", "v", "b", "ps")}
    for name, want in out["nodal"].items():
        np.testing.assert_array_equal(out["fv"][name], want,
                                      err_msg=f"{kind}/{variant}/{name}")


def test_split_derive_initial_fields_matches_the_nodal_transports():
    out = {}
    for family in ("fv", "nodal"):
        model = _seed(_model("terrain", "split", family, dt=1e-2))
        out[family] = {c: np.asarray(model.state[c].data)
                       for c in ("U", "V")}
        assert isinstance(
            model.state["U"].function_space.bare.factor("y"),
            AverageSpace if family == "fv" else type(
                model.state["U"].function_space.bare.factor("y")))
    for name, want in out["nodal"].items():
        np.testing.assert_array_equal(out["fv"][name], want,
                                      err_msg=name)
