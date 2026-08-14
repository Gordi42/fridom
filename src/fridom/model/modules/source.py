r"""The generic separable source term: ``dz += Q(x) g(t)``.

Description
-----------
The model-agnostic volumetric forcing of the source-module plan
(SRC-D1..D5): one instance adds a single separable term

.. math::
    S(\boldsymbol{x}, t) = A \,
        \mathrm{Re}\!\left[Q(\boldsymbol{x})\,
        e^{-i(2\pi f t + \varphi)}\right]

to the tendencies of one or more prognostic variables, with a
state-valued spatial pattern :math:`Q` (a coordinate-named callable
or an already-built field, per variable) and a scalar time law
:math:`g`. That formula is the :class:`~fridom.model.Harmonic` law's:
it is the one law the module reads structurally, and it publishes
:math:`A`, :math:`f` and :math:`\varphi` as the sweepable dynamic-leaf
parameters ``source.<label>.{amplitude, frequency, phase}``. Any other
:class:`~fridom.model.TimeDependent` is accepted as the escape hatch,
and then the term is exactly

.. math::
    S(\boldsymbol{x}, t) = g(t)\, Q(\boldsymbol{x})

— **no amplitude factor of its own**. See the class docstring's
"The two law branches" for what that implies for scaling a forcing.

Each pattern component materializes onto the forced variable's own
negotiated space — a ``LikeField(variable)`` AUXILIARY field, sampled
at that component's own nodes (staggered faces for velocities, cell
centres for scalars). A complex pattern expands into a real
quadrature pair (two AUXILIARY fields per component,
``source_<label>_<var>`` and ``source_<label>_<var>_im``), and the
term evaluates :math:`A[\cos\theta\,\mathrm{Re}\,Q +
\sin\theta\,\mathrm{Im}\,Q]` with :math:`\theta = 2\pi f t + \varphi`;
a complex pattern therefore requires the :class:`Harmonic` law (the
expansion needs :math:`f` and :math:`\varphi` structurally). The term
is a pure scalar-times-field product read off the traced clock, so it
needs no ``extra_halo`` and stays jit-pure (V-N2).
"""
from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import jaxify
from fridom.model.declarations import (
    FieldDeclaration,
    FieldReference,
    Lifecycle,
    LikeField,
)
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration, leaf
from fridom.model.params import ParamName
from fridom.model.terms import TendencyTerm, Treatment
from fridom.model.time_dependent import Harmonic, TimeDependent
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.scalars import Scalars

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import jax

    from fridom.spatial.grid import Grid
    from fridom.spatial.spaces.tensor_product import TensorProductSpace

_VARIABLE_HINT = ("the forced variable must be declared by another "
                  "module (a dynamical core declares u, v, w; a "
                  "buoyancy/tracer module declares b and tracers)")


def _pattern_axes(fn: Callable) -> tuple[str, ...]:
    """Return the coordinate names a pattern callable varies along."""
    return tuple(inspect.signature(fn).parameters)


def _callable_is_complex(fn: Callable) -> bool:
    """Probe a pattern callable's Körper at the coordinate origin.

    Evaluated host-side at construction with a zero for each named
    coordinate: a complex-valued sample declares the quadrature pair.
    A callable that cannot be probed (or names no coordinate) is
    treated as real — bind re-validates its coordinate names.
    """
    axes = _pattern_axes(fn)
    if not axes:
        return False
    try:
        sample = fn(**{axis: jnp.asarray(0.0) for axis in axes})
    except Exception:  # noqa: BLE001 — unprobeable ⇒ real; bind re-checks
        return False
    return bool(jnp.iscomplexobj(sample))


def _pattern_is_complex(pattern: object) -> bool:
    """Whether one pattern component samples to complex values."""
    if isinstance(pattern, ScalarField):
        return pattern.function_space.scalars is Scalars.COMPLEX
    return _callable_is_complex(pattern)


def _pattern_default(field: str) -> Callable:
    """Build the AUXILIARY owner-method default of one pattern field.

    The returned closure leads with ``self`` (the owner-method form):
    it wraps the bind-precomputed data array of ``field`` onto the
    forced variable's own negotiated ``space`` at materialization,
    with the *live* module.
    """
    def default(self, grid, space):  # noqa: ANN001, ANN202
        return grid.create_field(
            space, data=self._source_data[field], name=field)

    default.__name__ = f"_{field}_default"
    return default


@partial(jaxify,
         dynamic=("amplitude", "frequency", "phase", "law",
                  "_source_data"))
class Source(Module):

    r"""
    Add one separable source term ``Q(x) g(t)`` to a tendency.

    Description
    -----------
    Contributes a separable ``Q(x) g(t)`` term to the tendency of every
    variable the ``pattern`` names: the spatial pattern :math:`Q`
    freezes into AUXILIARY fields on the forced variables' own spaces
    at assembly, and the scalar law :math:`g` rides the traced clock.

    The two law branches
    ~~~~~~~~~~~~~~~~~~~~
    **What the term evaluates to depends on which law is passed**, and
    the two branches do not have the same free parameters.

    ``law=fr.model.Harmonic(amplitude=A, frequency=f, phase=phi)`` is
    the blessed law: the module reads :math:`A`, :math:`f` and
    :math:`\varphi` structurally and evaluates

    .. math::
        S = A\,\mathrm{Re}\!\left[Q\,e^{-i\theta}\right],
        \qquad \theta = 2\pi f t + \varphi

    — that is :math:`A\cos\theta\,Q` for a real pattern and the
    quadrature pair :math:`A[\cos\theta\,\mathrm{Re}\,Q +
    \sin\theta\,\mathrm{Im}\,Q]` for a complex one. All three publish
    as the dynamic-leaf parameters
    ``source.<label>.{amplitude, frequency, phase}``, so
    ``model.update_parameters`` sweeps them without re-assembly.

    **Any other** :class:`~fridom.model.TimeDependent` is the escape
    hatch, and there the term is exactly

    .. math::
        S = g(t)\, Q(\boldsymbol{x})

    with **no amplitude, frequency or phase of the module's own** —
    the module never inspects a generic law, it only calls it. Three
    consequences worth stating outright:

    - **The forcing's magnitude has to live in** :math:`Q` **or in**
      :math:`g`, because the module contributes no factor between
      them. Scaling by hand is the whole contract: either fold it into
      the pattern (``{"u": lambda x: tau / (rho * h) * shape(x)}``) or
      let the law carry it (``fr.model.TimeFunction(lambda t, a:
      a * jnp.tanh(t / T), params=(a0,))``).
    - **Which of the two you pick decides whether it is sweepable.**
      A pattern is sampled once at bind and frozen into an AUXILIARY
      field, so a magnitude folded into :math:`Q` only changes on
      re-assembly. A ``TimeFunction``'s ``params`` are dynamic leaves,
      so a magnitude carried there is swept without recompiling (and
      ``jax.grad`` flows through it) — it is just not addressable by
      name through ``update_parameters``, since a generic law
      publishes nothing.
    - **A complex pattern is refused** on this branch: the quadrature
      expansion needs :math:`f` and :math:`\varphi` structurally, and
      a generic law exposes neither. Pass a :class:`Harmonic`, or a
      real pattern.

    Parameters
    ----------
    label : str
        The instance label (mandatory, first positional): names the
        tendency term, the AUXILIARY fields (``source_<label>_<var>``,
        plus ``_im`` for complex quadratures), and the published
        parameters (``source.<label>.*``). Unique per model — a
        duplicate collides in the field/parameter tables.
    pattern : Mapping[str, Callable | ScalarField]
        Forced variable -> spatial pattern: a coordinate-named
        callable (its signature names the coordinates it varies
        along, sampled at the variable's own nodes) or an already-built
        ScalarField on the variable's own space. A ``State`` works
        directly (it is such a mapping).
    law : Harmonic | TimeDependent
        The scalar time law: a :class:`~fridom.model.Harmonic` (the
        blessed, parameter-publishing law, evaluated as :math:`A\,
        \mathrm{Re}[Q e^{-i\theta}]`) or any other
        :class:`~fridom.model.TimeDependent` (the escape hatch, e.g. a
        :class:`~fridom.model.TimeFunction` chirp, evaluated as the
        bare product :math:`g(t)\,Q`). The two branches differ in more
        than parameter publishing — see "The two law branches" above.
    """

    def __init__(
        self,
        label: str,
        pattern: Mapping[str, Callable | ScalarField],
        law: Harmonic | TimeDependent,
    ) -> None:
        """Classify the patterns; hoist the law's leaves; see class doc."""
        if not isinstance(label, str) or not label:
            raise TypeError(
                f"Source label must be a non-empty string, got "
                f"{label!r}")
        if "." in label:
            raise ValueError(
                f"Source label {label!r} contains a dot; the label "
                "keys the flat AUXILIARY field names and the dotted "
                "source.<label>.* parameter namespace (D2.1)")
        # a State / VectorField is a component-named vocabulary object
        # (its own iteration yields the fields, not the names), so read
        # its ``.components`` mapping; a plain Mapping is taken directly.
        components = getattr(pattern, "components", None)
        pattern = dict(pattern if components is None else components)
        if not pattern:
            raise ValueError(
                "Source needs at least one forced variable in "
                "pattern=; got an empty mapping")
        for variable, value in pattern.items():
            if not isinstance(variable, str) or not variable:
                raise TypeError(
                    f"Source pattern keys are prognostic variable "
                    f"names, got {variable!r}")
            if not (callable(value) or isinstance(value, ScalarField)):
                raise TypeError(
                    f"the Source pattern of {variable!r} must be a "
                    f"coordinate-named callable or a ScalarField, got "
                    f"{value!r}")
        if not isinstance(law, TimeDependent):
            raise TypeError(
                f"Source law= must be a TimeDependent — a "
                f"fr.model.Harmonic, or any other law such as a "
                f"fr.model.TimeFunction; got {law!r}")

        self._label: str = label
        self._patterns: dict[str, Callable | ScalarField] = pattern
        self._variables: tuple[str, ...] = tuple(pattern)
        self._complex: dict[str, bool] = {
            variable: _pattern_is_complex(value)
            for variable, value in pattern.items()}
        self._real_field: dict[str, str] = {
            variable: f"source_{label}_{variable}"
            for variable in self._variables}
        self._imag_field: dict[str, str] = {
            variable: f"source_{label}_{variable}_im"
            for variable in self._variables if self._complex[variable]}

        self._is_harmonic: bool = isinstance(law, Harmonic)
        if any(self._complex.values()) and not self._is_harmonic:
            offending = tuple(sorted(
                variable for variable in self._variables
                if self._complex[variable]))
            raise TypeError(
                f"the Source pattern of {offending} is complex, but "
                f"law={type(law).__name__} is not a Harmonic: the "
                "quadrature expansion A[cos(theta) Re Q + sin(theta) "
                "Im Q] needs the frequency and phase structurally. "
                "Pass a fr.model.Harmonic law, or a real pattern")

        # bind precomputes the source data arrays (host-side, once)
        self._source_data: dict[str, jax.Array] | None = None
        if self._is_harmonic:
            self.amplitude: object = leaf(law.amplitude)
            self.frequency: object = leaf(law.frequency)
            self.phase: object = leaf(law.phase)
            self.law: TimeDependent | None = None
            self._amplitude_name: ParamName = ParamName(
                f"source.{label}.amplitude", units="n/a",
                hint=f"provided by the fr.model.modules.Source {label!r}")
            self._frequency_name: ParamName = ParamName(
                f"source.{label}.frequency", units="1/s",
                hint=f"provided by the fr.model.modules.Source {label!r}")
            self._phase_name: ParamName = ParamName(
                f"source.{label}.phase", units="rad",
                hint=f"provided by the fr.model.modules.Source {label!r}")
        else:
            self.amplitude = None
            self.frequency = None
            self.phase = None
            self.law = law

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def label(self) -> str:
        """The instance label (names fields, term, and parameters)."""
        return self._label

    @property
    def variables(self) -> tuple[str, ...]:
        """The forced prognostic variable names."""
        return self._variables

    # ================================================================
    #  Declarations
    # ================================================================
    @property
    def field_references(self) -> tuple[FieldReference, ...]:
        """One checked claim per forced variable."""
        return tuple(
            FieldReference(variable, hint=_VARIABLE_HINT)
            for variable in self._variables)

    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """The pattern AUXILIARY fields on the variables' own spaces."""
        declarations: list[FieldDeclaration] = []
        for variable in self._variables:
            real = self._real_field[variable]
            declarations.append(FieldDeclaration(
                real, space=LikeField(variable),
                lifecycle=Lifecycle.AUXILIARY,
                default=_pattern_default(real),
                long_name=f"Source {self._label!r} pattern on "
                          f"{variable}",
                units="1"))
            if self._complex[variable]:
                imag = self._imag_field[variable]
                declarations.append(FieldDeclaration(
                    imag, space=LikeField(variable),
                    lifecycle=Lifecycle.AUXILIARY,
                    default=_pattern_default(imag),
                    long_name=f"Source {self._label!r} pattern "
                              f"(imag) on {variable}",
                    units="1"))
        return tuple(declarations)

    @property
    def parameter_declarations(
        self,
    ) -> tuple[ParameterDeclaration, ...]:
        """Publish the Harmonic leaves (nothing for a generic law)."""
        if not self._is_harmonic:
            return ()
        return (
            ParameterDeclaration(
                self._amplitude_name, attr="amplitude", units="n/a",
                doc=f"source {self._label!r} amplitude"),
            ParameterDeclaration(
                self._frequency_name, attr="frequency", units="1/s",
                doc=f"source {self._label!r} frequency"),
            ParameterDeclaration(
                self._phase_name, attr="phase", units="rad",
                doc=f"source {self._label!r} phase"),
        )

    # ================================================================
    #  Bind: sample the patterns onto the variables' own spaces
    # ================================================================
    def bind(self, table: object) -> None:
        """Validate the targets and precompute the source arrays.

        Raises
        ------
        ValueError
            If a forced variable is not PROGNOSTIC, if a callable
            pattern names a coordinate the grid does not have, or if a
            field-valued pattern does not live on the forced
            variable's own space.
        """
        # deferred: keep the module import light (the lazypimp spirit)
        from fridom.model.eigenstates import (  # noqa: PLC0415
            pattern_axes,
            sample_pattern,
        )

        grid = table.grid
        source_data: dict[str, jax.Array] = {}
        for variable in self._variables:
            record = table[variable]
            if record.lifecycle is not Lifecycle.PROGNOSTIC:
                raise ValueError(
                    f"Source {self._label!r} forces {variable!r}, "
                    f"which is {record.lifecycle.name}: only "
                    "PROGNOSTIC fields are advanced from tendencies")
            space = record.space
            pattern = self._patterns[variable]
            if isinstance(pattern, ScalarField):
                self._bind_field(variable, pattern, space, source_data)
            else:
                axes = pattern_axes(
                    pattern, grid.names,
                    f"Source {self._label!r}")
                self._bind_callable(
                    variable, pattern, space, axes, grid,
                    sample_pattern, source_data)
        self._source_data = source_data

    def _bind_field(
        self,
        variable: str,
        field: ScalarField,
        space: TensorProductSpace,
        source_data: dict,
    ) -> None:
        """Validate and store a field-valued pattern on its own space."""
        if field.real.function_space.bare is not space:
            raise ValueError(
                f"the Source {self._label!r} pattern of {variable!r} "
                f"is a field on {field.function_space.bare!r}, but "
                f"{variable!r} lives on {space!r}: a field-valued "
                "pattern must live on the forced variable's own space "
                "(build it on that variable's function space, or pass "
                "a coordinate-named callable to be sampled there)")
        source_data[self._real_field[variable]] = field.real.data
        if self._complex[variable]:
            source_data[self._imag_field[variable]] = field.imag.data

    def _bind_callable(
        self,
        variable: str,
        pattern: Callable,
        space: TensorProductSpace,
        axes: tuple[str, ...],
        grid: Grid,
        sample_pattern: Callable,
        source_data: dict,
    ) -> None:
        """Sample a callable pattern at the variable's own nodes."""
        if self._complex[variable]:
            sampled = sample_pattern(
                grid, space.as_complex(), pattern, axes)
            source_data[self._real_field[variable]] = sampled.real.data
            source_data[self._imag_field[variable]] = sampled.imag.data
        else:
            source_data[self._real_field[variable]] = sample_pattern(
                grid, space, pattern, axes).data

    # ================================================================
    #  The forcing term
    # ================================================================
    #: The forcing is pointwise (zero halo), but the traced-clock law
    #: is a jax scalar the halo tracer cannot follow (V-N2) — declare
    #: the substitute instead of being traced.
    extra_halo = HaloSpec({})

    def tendency_terms(self) -> tuple[TendencyTerm, ...]:
        """One term forcing every variable the pattern names."""
        return (
            TendencyTerm(
                name=self._label, fn=self._force,
                treatment=Treatment.EXPLICIT,
                advances=self._variables),
        )

    def _force(self, state, ctx) -> dict:  # noqa: ANN001
        r"""Add the branch's product off the traced clock.

        ``dz/dt += A Re[Q e^{-i(2 pi f t + phi)}]`` under a Harmonic
        law; ``dz/dt += g(t) Q`` (no amplitude) under any other.
        """
        # ctx.clock is the Clock in-run, a bare stage-time scalar in
        # dry-run/tendency contexts (the schedule.context idiom)
        time = getattr(ctx.clock, "time", ctx.clock)
        if not self._is_harmonic:
            g = self.law(time)
            return {variable: g * state[self._real_field[variable]]
                    for variable in self._variables}
        amp = ctx.params[self._amplitude_name]
        freq = ctx.params[self._frequency_name]
        phase = ctx.params[self._phase_name]
        theta = 2.0 * jnp.pi * freq * time + phase
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)
        out = {}
        for variable in self._variables:
            real = state[self._real_field[variable]]
            if variable in self._imag_field:
                out[variable] = amp * (
                    cos_t * real
                    + sin_t * state[self._imag_field[variable]])
            else:
                out[variable] = (amp * cos_t) * real
        return out
