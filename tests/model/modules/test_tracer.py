r"""The one-liner tracer module: declaration, assembly, transport.

The claims under test:

- ``Tracer("dye")`` declares exactly one PROGNOSTIC field with the
  roles ``{TRACER, ADVECTED}``, forwarding the ``FieldDeclaration.tracer``
  slots (``space=``/``family=``, ``default=``, ``long_name=``,
  ``units=``, ``nc_attrs=``); bad arguments are refused at the call
  site, not at assembly.
- the declared field enters the assembled model's PROGNOSTIC set and
  the role selections, so the role-driven consumers find it: flux-form
  advection **transports** it (a sine tracer under a uniform velocity
  shifts by :math:`Ut`, matching the centered-2 discrete phase speed)
  and a TRACER-targeting closure mixes it (analytic Fourier decay).
- several tracers coexist; a duplicate name is a ``FieldCollisionError``
  and a tracer no term advances (``advection=None``, no closure) is
  the documented D1.4 coverage ``AssemblyError``.

Differentiability policy (AGENTS.md): ``Tracer`` is declarations-only
— it contributes no step-path term of its own, so the policy does not
bite. The one autodiff test below is not about a term but about the
*carry*: a user-declared field is a new leaf of the state vector, and
the test pins that a gradient w.r.t. it stays finite and matches a
central FD.

Self-contained per the shard convention: the small builders are
duplicated rather than imported across test files.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.closures.diffusion import HarmonicDiffusion
from fridom.model.errors import AssemblyError, FieldCollisionError
from fridom.model.model import Model
from fridom.model.modules.tracer import Tracer
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 12
LENGTH = 1.0
DT = 1e-3
KAPPA = 1e-3

# the advection run (a periodic x-column with a thin transverse slab)
NX = 64
NT = 4
L2PI = 2 * np.pi
U = 0.5
DT_ADV = 2e-3
STEPS = 100


# ================================================================
#  Helpers
# ================================================================
def make_grid():
    """Return a tiny periodic (x, z) grid."""
    return Grid(tuple(
        IntervalMesh(N, (0.0, LENGTH), periodic=True, name=name)
        for name in ("x", "z")))


def make_model(*modules, kappa=KAPPA):
    """Return a tracer model whose only term is the mixing closure."""
    return Model(
        grid=make_grid(),
        modules=(*modules, HarmonicDiffusion(kappa)),
        time_stepper=AdamBashforth(DT, order=3))


def cell_centers(n=N, length=LENGTH):
    """Return the (x, z) cell-center meshgrid of the tiny grid."""
    ax = (np.arange(n) + 0.5) * (length / n)
    return np.meshgrid(ax, ax, indexing="ij")


def uniform_u(x, y, z):  # noqa: ARG001 — the IC names every coordinate
    """Return the uniform x-velocity U over the whole domain."""
    return 0.0 * x + U


def sine_x(x, y, z):  # noqa: ARG001 — the IC names every coordinate
    """Return the sine tracer distribution sin(x)."""
    return np.sin(x)


def state_sq(final):
    """Sum of squares of every final-state field (a smooth loss)."""
    return sum(jnp.sum(f.data ** 2) for f in final.state)


# ================================================================
#  The declaration
# ================================================================
def test_declares_one_prognostic_advected_tracer():
    module = Tracer("dye")
    assert module.name == "dye"
    assert repr(module) == "Tracer('dye')"
    (declaration,) = module.field_declarations
    assert declaration.name == "dye"
    assert declaration.lifecycle is fr.model.Lifecycle.PROGNOSTIC
    assert declaration.roles == frozenset(
        {fr.model.roles.TRACER, fr.model.roles.ADVECTED})
    # the template default: collocated everywhere, grid-default family
    assert declaration.space == fr.spatial.Collocated()
    # and nothing else: no term, no parameter, no dispatch
    assert module.tendency_terms() == ()
    assert module.stages == ()
    assert module.parameter_declarations == ()


def test_annotation_slots_are_forwarded():
    (declaration,) = Tracer(
        "dye", default=2.0, long_name="Dye", units="1",
        nc_attrs={"b": "2", "a": "1"}).field_declarations
    assert declaration.default == 2.0
    assert declaration.long_name == "Dye"
    assert declaration.units == "1"
    assert declaration.nc_attrs == (("a", "1"), ("b", "2"))


def test_declaration_defaults_match_the_wrapped_template():
    (wrapped,) = Tracer("dye").field_declarations
    template = fr.model.FieldDeclaration.tracer("dye")
    assert wrapped.long_name == template.long_name == "Unnamed"
    assert wrapped.units == template.units == "unknown"
    assert wrapped.default is template.default is None


@pytest.mark.parametrize("family", ["fv", "nodal"])
def test_family_is_the_collocated_shorthand(family):
    (declaration,) = Tracer("dye", family=family).field_declarations
    assert declaration.space == fr.spatial.Collocated(family=family)


def test_explicit_space_passes_through():
    space = fr.spatial.Staggered("x", family="fv")
    (declaration,) = Tracer("dye", space=space).field_declarations
    assert declaration.space == space


def test_the_module_is_reachable_from_the_modules_namespace():
    # the spelled surface (specs say ``fr.modules.Tracer``; the new
    # stack's module library lives under ``fr.model.modules``)
    assert fr.model.modules.Tracer is Tracer


# ================================================================
#  Construction-time validation
# ================================================================
def test_space_and_family_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        Tracer("dye", space=fr.spatial.Collocated(), family="fv")


def test_a_bad_name_is_refused_at_the_call_site():
    # eager validation: the wrapped declaration is built in __init__,
    # so the error names the caller's line, not the assembly
    with pytest.raises(ValueError, match="contains a dot"):
        Tracer("a.dye")
    with pytest.raises(TypeError, match="name"):
        Tracer(3)


def test_a_bad_default_is_refused_at_the_call_site():
    with pytest.raises(TypeError, match="default"):
        Tracer("dye", default="warm")


# ================================================================
#  Assembly: the field enters the state vector
# ================================================================
def test_the_tracer_is_prognostic_and_role_tagged():
    model = make_model(Tracer("dye", long_name="Dye", units="1"))
    assert model.field_table.prognostic == ("dye",)
    assert model.field_table.select(fr.model.roles.TRACER) == ("dye",)
    assert model.field_table.select(fr.model.roles.ADVECTED) == ("dye",)
    assert model.state["dye"].shape == (N, N)
    metadata = model.field_table["dye"].metadata
    assert (metadata.long_name, metadata.units) == ("Dye", "1")


def test_several_tracers_are_several_modules():
    model = make_model(Tracer("dye"), Tracer("age"))
    assert model.field_table.prognostic == ("dye", "age")
    assert model.field_table.select(fr.model.roles.TRACER) == (
        "dye", "age")


def test_a_duplicate_tracer_name_collides():
    with pytest.raises(FieldCollisionError, match="declared twice"):
        make_model(Tracer("dye"), Tracer("dye"))


def test_a_tracer_no_term_advances_is_refused():
    # the documented caveat: declarations-only means the D1.4
    # coverage lint needs *some* term to advance the field
    with pytest.raises(AssemblyError, match="advanced by no term"):
        Model(grid=make_grid(), modules=(Tracer("dye"),),
              time_stepper=AdamBashforth(DT, order=3))


# ================================================================
#  The role-driven consumers pick it up
# ================================================================
def test_a_tracer_targeting_closure_mixes_it():
    """The default TRACER-targeted mixing decays the tracer mode."""
    model = make_model(Tracer("dye"))
    x, _ = cell_centers()
    k = 2 * np.pi / LENGTH
    model.set_fields(dye=np.sin(k * x))
    steps = 50
    model.run(steps=steps)

    dx = LENGTH / N
    eigenvalue = (2 - 2 * np.cos(k * dx)) / dx**2  # discrete laplacian
    exact = np.exp(-KAPPA * eigenvalue * steps * DT) * np.sin(k * x)
    assert np.max(np.abs(np.asarray(model.state["dye"].data) - exact)) < 1e-8


def test_a_tracer_is_advected_by_a_uniform_velocity():
    """A sine tracer shifts by U t under a uniform flow (nh model)."""
    grid = Grid((
        IntervalMesh(NX, (0.0, L2PI), name="x"),
        IntervalMesh(NT, (0.0, L2PI), name="y"),
        IntervalMesh(NT, (0.0, L2PI), name="z"),
    ))
    model = nh.Model(
        advection=nh.CenteredAdvection(),
        grid=grid, core=nh.Core(),
        time_stepper=AdamBashforth(DT_ADV, order=3),
        modules_extra=(Tracer("dye", units="1"),))
    model.set_fields(u=uniform_u, dye=sine_x)
    model.run(steps=STEPS)

    # the uniform flow is an exact steady solution: nothing else moves
    assert np.max(np.abs(np.asarray(model.state["u"].data) - U)) == 0.0

    dye = np.asarray(model.state["dye"].data)
    ax = (np.arange(NX) + 0.5) * (L2PI / NX)
    time = STEPS * DT_ADV
    shifted = np.sin(ax - U * time)[:, None, None]
    # (a) it really moved: the shift is far larger than the error
    assert np.max(np.abs(dye - np.sin(ax)[:, None, None])) > 0.09
    # (b) it moved the analytically expected distance
    assert np.max(np.abs(dye - shifted)) < 1e-3
    # (c) the residual is the centered-2 phase-speed error, nothing else
    dx = L2PI / NX
    c_eff = np.sin(dx) / dx
    corrected = np.sin(ax - U * c_eff * time)[:, None, None]
    assert np.max(np.abs(dye - corrected)) < 1e-5


# ================================================================
#  Autodiff: a user-declared field is a differentiable carry leaf
# ================================================================
def test_grad_wrt_the_tracer_initial_condition_matches_fd():
    model = make_model(Tracer("dye"))
    x, _ = cell_centers()
    model.set_fields(dye=np.sin(2 * np.pi * x))
    run = model.propagator(wrt=("dye",), steps=5)
    dye0 = model.state["dye"].storage

    def loss(field):
        return state_sq(run((field,)))

    grad = np.asarray(jax.grad(loss)(dye0))
    assert bool(np.all(np.isfinite(grad)))

    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(dye0.shape),
                            dtype=dye0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(dye0 + eps * direction))
          - float(loss(dye0 - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)
