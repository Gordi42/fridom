r"""A Gaussian wave maker: time-periodic, spatially localized forcing.

Description
-----------
The port of the v1 ``nh.modules.forcings.GaussianWaveMaker``:
a source term with a Gaussian envelope in space and a sinusoid in time,

.. math::
    S(\boldsymbol{x}, t) = A \sin(2\pi f t)
        \prod_{i} \exp\left(-\frac{(x_i - p_i)^2}{w_i^2}\right),

added to the tendency of one prognostic variable (``u`` by default).
The envelope is carried as an AUXILIARY mask field declared on the
forced variable's own space pattern — the Gaussian is sampled at that
component's own node positions (staggered faces for ``u``/``v``/``w``,
cell centers for ``b`` and tracers) when the field is materialized.
Position and width are name-keyed mappings; axes absent from both are
constant (the v1 ``None`` entries). The term is a pure
scalar-times-field product read off the traced clock
(``ctx.clock.time``), so it needs no ``extra_halo`` and is jit-pure.

Amplitude and frequency are provided dynamic-leaf parameters
(``wavemaker.<variable>.amplitude`` / ``.frequency``), so
``model.update_parameters`` sweeps them without re-assembly.
"""
from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.model.params import ParamName
from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.spaces.constant import ConstantSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import jax

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.space_patterns import SpacePattern

#: The velocity components' staggering axes (the nh C-grid).
_VELOCITY_AXES = {"u": "x", "v": "y", "w": "z"}

_VARIABLE_HINT = ("the forced variable must be declared by another "
                  "module (nh.DynamicalCore declares u, v, w; a "
                  "stratification module declares b)")


def sample_gaussian_mask(
    grid,  # noqa: ANN001
    space,  # noqa: ANN001
    position: Mapping[str, float],
    width: Mapping[str, float],
    name: str | None = None,
) -> ScalarField:
    r"""
    Sample a Gaussian envelope at a space's own node positions.

    Description
    -----------
    Materializes
    :math:`\prod_i \exp(-(x_i - p_i)^2 / w_i^2)` over the
    coordinates named in ``position`` (constant along the others),
    evaluated at ``space``'s nodes: the ``init`` signature is
    stamped with the space's non-constant coordinate names (the
    MeridionalStratification precedent). Shared by the Gaussian and
    polarized wave makers.

    Parameters
    ----------
    grid : fr.spatial.Grid
        The grid to materialize on.
    space : SpaceLike
        The target function space (the forced component's space).
    position : Mapping[str, float]
        Envelope centers, keyed by coordinate name.
    width : Mapping[str, float]
        Envelope widths; same keys as ``position``.
    name : str | None, optional
        Metadata name of the mask field (default: None).

    Returns
    -------
    ScalarField
        The sampled envelope (ones when ``position`` is empty).
    """
    names = tuple(
        coordinate for factor in space.factors
        if not isinstance(factor, ConstantSpace)
        for coordinate in factor.names)

    def init(**coords: jax.Array) -> jax.Array:
        mask = jnp.asarray(1.0)
        for axis, pos in position.items():
            mask = mask * jnp.exp(
                -((coords[axis] - pos) ** 2) / width[axis] ** 2)
        return mask

    init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(
            coordinate, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for coordinate in names])
    return grid.create_field(space, init=init, name=name)


def _variable_pattern(variable: str) -> SpacePattern:
    """Return the forced variable's space pattern (nh vocabulary).

    Velocity components live on their staggered faces and carry the
    topology-conditional wall Dirichlet of the velocity template
    (inert on periodic grids); everything else (``b``, tracers) is
    collocated — so the declared mask resolves to the identical
    interned space as the forced variable.
    """
    axis = _VELOCITY_AXES.get(variable)
    if axis is None:
        return fr.spatial.Collocated()
    return fr.spatial.Staggered(axis, wall_bc={axis: BC.DIRICHLET})


@partial(jaxify, dynamic=("frequency", "amplitude"))
class GaussianWaveMaker(fr.model.Module):

    r"""
    Force one variable with a Gaussian envelope and a sinusoid.

    Description
    -----------
    Adds :math:`A \sin(2\pi f t)\, M(\boldsymbol{x})` to the forced
    variable's tendency, with the stationary Gaussian mask

    .. math::
        M(\boldsymbol{x}) =
            \prod_{i} \exp\left(-\frac{(x_i - p_i)^2}{w_i^2}\right)

    over the coordinates named in ``position``/``width`` (constant
    along the others). The mask is an AUXILIARY field on the forced
    variable's own space, sampled at its node positions.

    Parameters
    ----------
    position : Mapping[str, float]
        Center :math:`p_i` of the Gaussian, keyed by coordinate name;
        unnamed axes are constant.
    width : Mapping[str, float]
        Width :math:`w_i` of the Gaussian; same keys as ``position``.
    frequency : float | fr.model.Ramp
        The forcing frequency :math:`f` (the sinusoid runs at
        :math:`2\pi f`).
    amplitude : float | fr.model.Ramp
        The forcing amplitude :math:`A`.
    variable : str, optional
        The prognostic variable to force (default: "u").
    """

    def __init__(
        self,
        position: Mapping[str, float],
        width: Mapping[str, float],
        frequency: float | fr.model.Ramp,
        amplitude: float | fr.model.Ramp,
        variable: str = "u",
    ) -> None:
        """Store the leaves; freeze the envelope and target names."""
        if not isinstance(variable, str) or not variable:
            raise TypeError(
                f"variable must be a non-empty field name, got "
                f"{variable!r}")
        position = dict(position)
        width = dict(width)
        if set(position) != set(width):
            raise ValueError(
                f"position and width must name the same "
                f"coordinates; got position keys "
                f"{tuple(sorted(position))} and width keys "
                f"{tuple(sorted(width))}")
        self._position: dict[str, float] = position
        self._width: dict[str, float] = width
        self._variable: str = variable
        self._mask_name: str = f"wavemaker_{variable}_mask"
        self.frequency = fr.model.leaf(frequency)
        self.amplitude = fr.model.leaf(amplitude)
        self._frequency_name: ParamName = ParamName(
            f"wavemaker.{variable}.frequency", units="1/s",
            hint="provided by the nh.GaussianWaveMaker forcing "
                 f"{variable!r}")
        self._amplitude_name: ParamName = ParamName(
            f"wavemaker.{variable}.amplitude", units="n/a",
            hint="provided by the nh.GaussianWaveMaker forcing "
                 f"{variable!r}")

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def variable(self) -> str:
        """The forced prognostic variable."""
        return self._variable

    # ================================================================
    #  Declarations
    # ================================================================
    @property
    def field_references(self) -> tuple[fr.model.FieldReference, ...]:
        """The checked claim on the forced variable."""
        return (fr.model.FieldReference(self._variable,
                                  hint=_VARIABLE_HINT),)

    @property
    def field_declarations(self) -> tuple[fr.model.FieldDeclaration, ...]:
        """The Gaussian mask on the forced variable's own pattern."""
        return (
            fr.model.FieldDeclaration(
                self._mask_name,
                space=_variable_pattern(self._variable),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._mask_default,
                long_name=f"Wave-maker mask on {self._variable}",
                units="1"),
        )

    @property
    def parameter_declarations(
        self,
    ) -> tuple[fr.model.ParameterDeclaration, ...]:
        """Publish the live frequency and amplitude leaves."""
        return (
            fr.model.ParameterDeclaration(
                self._frequency_name, attr="frequency",
                units="1/s",
                doc=f"wave-maker frequency on {self._variable}"),
            fr.model.ParameterDeclaration(
                self._amplitude_name, attr="amplitude",
                doc=f"wave-maker amplitude on {self._variable}"),
        )

    def _mask_default(
        self, grid, space,  # noqa: ANN001
    ) -> ScalarField:
        """Owner-method default: sample the Gaussian on ``space``.

        The envelope is evaluated at the forced component's own node
        positions (see :func:`sample_gaussian_mask`).
        """
        return sample_gaussian_mask(
            grid, space, self._position, self._width,
            name=self._mask_name)

    # ================================================================
    #  Bind-time validation
    # ================================================================
    def bind(self, table: object) -> None:
        """Validate the envelope axes and the forced variable.

        Raises
        ------
        ValueError
            If ``position`` names a coordinate the grid does not
            have, if the forced variable is not PROGNOSTIC, or if
            the mask could not be co-located with the variable's
            space (a variable outside the ``u``/``v``/``w``/``b`` +
            collocated-tracer vocabulary).
        """
        unknown = sorted(set(self._position) - set(table.grid.names))
        if unknown:
            raise ValueError(
                f"the GaussianWaveMaker envelope names the "
                f"coordinate(s) {unknown}, which the grid does not "
                f"have (coordinates: {table.grid.names})")
        record = table[self._variable]
        if record.lifecycle is not fr.model.Lifecycle.PROGNOSTIC:
            raise ValueError(
                f"GaussianWaveMaker forces {self._variable!r}, "
                f"which is {record.lifecycle.name}: only PROGNOSTIC "
                "fields are advanced from tendencies")
        if record.space is not table[self._mask_name].space:
            raise ValueError(
                f"the GaussianWaveMaker mask resolves on "
                f"{table[self._mask_name].space!r} but "
                f"{self._variable!r} lives on {record.space!r}; the "
                "wave maker samples its envelope on the forced "
                "variable's own space and supports u, v, w, b and "
                "collocated tracers")

    # ================================================================
    #  The forcing term
    # ================================================================
    #: The forcing is pointwise (zero halo), but the traced-clock
    #: sinusoid is a jax scalar the halo tracer cannot follow
    #: (V-N2) — declare the substitute instead of being traced.
    extra_halo = HaloSpec({})

    def tendency_terms(self) -> tuple[fr.model.TendencyTerm, ...]:
        """One term forcing the configured variable."""
        return (
            fr.model.TendencyTerm(
                name="wave_maker", fn=self._force,
                treatment=fr.model.Treatment.EXPLICIT,
                advances=(self._variable,)),
        )

    def _force(self, state, ctx) -> dict:  # noqa: ANN001
        r"""``d var/dt += A sin(2 pi f t) M(x)`` off the traced clock."""
        amp = ctx.params[self._amplitude_name]
        freq = ctx.params[self._frequency_name]
        # ctx.clock is the Clock in-run, a bare stage-time scalar in
        # dry-run/tendency contexts (the schedule.context idiom)
        time = getattr(ctx.clock, "time", ctx.clock)
        oscillation = amp * jnp.sin(2.0 * jnp.pi * freq * time)
        return {self._variable: oscillation * state[self._mask_name]}
