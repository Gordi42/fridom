r"""A polarized wave maker: force one internal-wave packet resonantly.

Description
-----------
The port of the v1
``nh.modules.forcings.PolarizedWaveMaker``: the source term is the
masked, re-projected single-mode wave packet of the v1 ``WavePackage``
initial condition, oscillated at the packet's own discrete frequency,

.. math::
    S(\boldsymbol{x}, t) = A \sin(\omega t)\,
        \boldsymbol{z}_W(\boldsymbol{x}),

where :math:`\omega` is read off the operator-sourced analytic
eigenmodes (:class:`~fridom.nonhydro2.eigenmodes.Eigenmodes` — the
discrete dispersion relation, spatial-discretization errors included)
and :math:`\boldsymbol{z}_W` is the packet: the polarized single mode
at wavevector index ``k``, multiplied by a Gaussian envelope and
projected back onto the wave branch.

The bind/in-step split (D2.1): ``bind`` reads the constant ``f0``,
``N^2`` and the aspect ratio through the gated bind-time view (a Ramp-valued
parameter raises the taught ``TimeDependentParameterError`` — the
packet's polarization is frozen structure), builds the eigenmodes,
and precomputes the packet as plain data arrays; the four source
components are then materialized as AUXILIARY fields at assembly
step 8 and the per-step term is the jit-pure pointwise product
``A sin(omega t) * source`` off the traced clock. The forcing needs
zero halo, but the traced-clock sinusoid is a jax scalar the halo
tracer cannot follow, so the module declares the zero-halo
``extra_halo`` substitute (V-N2).

The amplitude is the provided dynamic-leaf parameter
``wavemaker.polarized.amplitude`` (``model.update_parameters``
sweeps it); the frequency is derived structure, exposed read-only as
:attr:`PolarizedWaveMaker.frequency`.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.model.errors import TimeDependentParameterError
from fridom.model.params import (
    CORIOLIS_F0,
    STRATIFICATION_N2,
    ParamName,
)
from fridom.model.scheduled_field import ProfileFunction
from fridom.nonhydro2.params import ASPECT_RATIO
from fridom.spatial.decomposition.halo import HaloSpec

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import jax

#: The packet's prognostic components (the analytic-tier vocabulary).
_COMPONENTS = ("u", "v", "w", "b")

#: The components' space patterns (the nh C-grid declarations).
_PATTERNS = {
    "u": fr.spatial.Staggered("x"),
    "v": fr.spatial.Staggered("y"),
    "w": fr.spatial.Staggered("z"),
    "b": fr.spatial.Collocated(),
}

#: The provided amplitude parameter (one polarized maker per model).
POLARIZED_AMPLITUDE = ParamName(
    "wavemaker.polarized.amplitude", units="n/a",
    hint="provided by nh.PolarizedWaveMaker(amplitude=...)")

_COMPONENT_HINT = ("the wave packet spans u, v, w and b: the "
                   "velocities are declared by nh.Core, "
                   "the buoyancy by a stratification module")


def _source_default(component: str) -> Callable:
    """Build the AUXILIARY default of one packet component.

    The returned closure leads with ``self`` (the owner-method
    default form): it is called with the *live* module at
    materialization and wraps the bind-precomputed data array of
    ``component`` onto the negotiated space.
    """
    def default(self, grid, space):  # noqa: ANN001, ANN202
        return grid.create_field(
            space, data=self._sources[component],
            name=f"wavemaker_{component}")

    default.__name__ = f"_wavemaker_{component}_default"
    return default


@partial(jaxify, dynamic=("amplitude", "_omega", "_sources"))
class PolarizedWaveMaker(fr.model.Module):

    r"""
    Force a polarized internal-wave packet at its own frequency.

    Description
    -----------
    Adds :math:`A \sin(\omega t)\,\boldsymbol{z}_W` to the ``u``,
    ``v``, ``w``, ``b`` tendencies: :math:`\boldsymbol{z}_W` is the
    discrete single mode at index ``k`` of the inertia-gravity
    branch ``s`` (polarization and frequency from the analytic
    eigenmodes of the assembled model's constant ``f0``, ``N^2``,
    the aspect ratio), enveloped by the Gaussian

    .. math::
        M(\boldsymbol{x}) =
            \prod_{i} \exp\left(-\frac{(x_i - p_i)^2}{w_i^2}\right)

    and re-projected onto the same branch (the v1 ``WavePackage``
    construction, factor 2 included), so the forcing stays in the
    wave subspace and drives the packet resonantly.

    Parameters
    ----------
    k : Mapping[str, int]
        Axis-keyed integer mode indices of the carrier wave, one
        entry per grid coordinate (e.g. ``{"x": 2, "y": 0, "z": 1}``).
    position : Mapping[str, float]
        Center of the Gaussian envelope, keyed by coordinate name;
        unnamed axes are constant.
    width : Mapping[str, float]
        Width of the Gaussian envelope; same keys as ``position``.
    amplitude : float | fr.model.Ramp, optional
        The forcing amplitude :math:`A` (default: 1.0).
    branch : int, optional
        The inertia-gravity branch, +1 or -1 (the ``"wave+"`` /
        ``"wave-"`` families) (default: 1).
    vertical : str, optional
        The vertical coordinate name (default: "z").
    """

    def __init__(
        self,
        k: Mapping[str, int],
        position: Mapping[str, float],
        width: Mapping[str, float],
        *,
        amplitude: float | fr.model.Ramp = 1.0,
        branch: int = 1,
        vertical: str = "z",
    ) -> None:
        """Store the amplitude leaf; freeze the packet structure."""
        if branch not in (1, -1):
            raise ValueError(
                f"the wave maker oscillates an inertia-gravity "
                f"packet: branch must be +1 or -1, got {branch} "
                "(the vortical branch is steady — force it with a "
                "stationary module instead)")
        position = dict(position)
        width = dict(width)
        if set(position) != set(width):
            raise ValueError(
                f"position and width must name the same "
                f"coordinates; got position keys "
                f"{tuple(sorted(position))} and width keys "
                f"{tuple(sorted(width))}")
        self._k: dict[str, int] = dict(k)
        self._position: dict[str, float] = position
        self._width: dict[str, float] = width
        self._branch: int = branch
        self._vertical: str = vertical
        self.amplitude = fr.model.leaf(amplitude)
        # bind precomputes the frequency and the packet data arrays
        self._omega: jax.Array | None = None
        self._sources: dict[str, jax.Array] | None = None

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def frequency(self) -> float | None:
        """The packet's discrete frequency (None before bind)."""
        if self._omega is None:
            return None
        return float(self._omega)

    # ================================================================
    #  Declarations
    # ================================================================
    field_references = tuple(
        fr.model.FieldReference(name, hint=_COMPONENT_HINT)
        for name in _COMPONENTS)

    field_declarations = tuple(
        fr.model.FieldDeclaration(
            f"wavemaker_{name}", space=_PATTERNS[name],
            lifecycle=fr.model.Lifecycle.AUXILIARY,
            default=_source_default(name),
            long_name=f"Wave-maker source on {name}")
        for name in _COMPONENTS)

    parameter_declarations = (
        fr.model.ParameterDeclaration(
            POLARIZED_AMPLITUDE, attr="amplitude",
            doc="polarized wave-maker amplitude"),
    )

    parameter_references = (
        fr.model.ParameterReference(
            CORIOLIS_F0,
            hint="the packet polarization needs a constant f0 "
                 "(fr.model.modules.FPlaneCoriolis)"),
        fr.model.ParameterReference(
            STRATIFICATION_N2,
            hint="the packet polarization needs a constant N^2 "
                 "(nh.ConstantStratification)"),
        fr.model.ParameterReference(
            ASPECT_RATIO, hint="declared by nh.Core"),
    )

    # ================================================================
    #  Bind: eigenmodes -> packet data arrays (host-side, once)
    # ================================================================
    def bind(self, table: object) -> None:
        """Precompute the packet's frequency and source arrays.

        Raises
        ------
        ValueError
            On a walled grid (the packet is synthesized from the
            Fourier eigenmodes), on envelope/carrier coordinates
            the grid does not have, or on a structurally
            unrepresented carrier mode.
        TimeDependentParameterError
            On any time-dependent input the frozen packet cannot
            follow (TDF-D6): a Ramp-valued ``f0``/``N^2``/``aspect_ratio``
            (caught by the bind-time parameter gate), a
            ``ProfileFunction``-valued one, or a dependency field
            marked ``time_dependent``. The packet polarization and
            frequency are baked once at bind — ramp the amplitude
            instead.
        """
        # deferred: sibling-package imports resolved at bind keep the
        # module import light (the lazypimp spirit)
        from fridom.nonhydro2.eigenmodes import Eigenmodes  # noqa: PLC0415
        from fridom.nonhydro2.modules.gaussian_wave_maker import (  # noqa: PLC0415
            sample_gaussian_mask,
        )
        from fridom.nonhydro2.state import State  # noqa: PLC0415

        grid = table.grid
        walled = tuple(
            name for mesh in grid.factors for name in mesh.names
            if not getattr(mesh, "periodic", True))
        if walled:
            raise ValueError(
                f"PolarizedWaveMaker needs a fully periodic grid "
                f"(bounded coordinates: {walled}): the packet is "
                "synthesized from the analytic Fourier eigenmodes")
        names = set(grid.names)
        if set(self._k) != names:
            raise ValueError(
                f"the carrier index k names {tuple(sorted(self._k))}"
                f", but the grid coordinates are {grid.names}; give "
                "one integer mode index per coordinate")
        unknown = sorted(set(self._position) - names)
        if unknown:
            raise ValueError(
                f"the PolarizedWaveMaker envelope names the "
                f"coordinate(s) {unknown}, which the grid does not "
                f"have (coordinates: {grid.names})")

        # TDF-D6: the packet is precomputed once at bind, so any
        # time-dependent input would leave a stale t=0 snapshot. The
        # Ramp case is caught by the parameter gate on read below;
        # refuse the two forms the general-time-dependence plan adds —
        # a ProfileFunction-valued parameter and a field marked
        # time_dependent — the same way (name the offender, prescribe
        # ramping the amplitude).
        parameters = table.parameters
        f0 = parameters[CORIOLIS_F0]
        n2 = parameters[STRATIFICATION_N2]
        delta = parameters[ASPECT_RATIO]
        for name, value in (
            (CORIOLIS_F0, f0),
            (STRATIFICATION_N2, n2),
            (ASPECT_RATIO, delta),
        ):
            if isinstance(value, ProfileFunction):
                raise TimeDependentParameterError(
                    f"PolarizedWaveMaker freezes its wave packet from "
                    f"a constant {name!r}, but it was given a "
                    "time-dependent ProfileFunction law: the packet "
                    "polarization and frequency are baked once at "
                    f"bind and cannot follow a {name!r}(y, t) profile."
                    " Pass a constant and ramp "
                    "wavemaker.polarized.amplitude to drive the "
                    "forcing in time instead")
        for ref in self.field_references:
            if getattr(table[ref.name], "time_dependent", False):
                raise TimeDependentParameterError(
                    f"PolarizedWaveMaker depends on the field "
                    f"{ref.name!r}, which is marked time_dependent "
                    "(its values evolve every substage), but the wave "
                    "packet is precomputed once at bind from a frozen "
                    "snapshot and cannot track it. Ramp "
                    "wavemaker.polarized.amplitude to drive the "
                    "forcing in time instead")
        modes = Eigenmodes(
            grid,
            f0=float(f0),
            n2=float(n2),
            dsqr=float(delta) ** 2,
            vertical=self._vertical)
        omega, wave = modes.mode(
            "wave", self._k, branch=self._branch)

        # mask each component at its own nodes, re-project onto the
        # branch, and keep the doubled real packet (the v1
        # WavePackage construction)
        masked = State({
            name: modes.kit.forward(name)(
                wave[name] * sample_gaussian_mask(
                    grid, wave[name].function_space,
                    self._position, self._width))
            for name in _COMPONENTS})
        packet = modes.projector(self._branch)(masked)
        self._sources = {
            name: 2.0 * jnp.asarray(
                modes.kit.backward(name)(packet[name]).real.data)
            for name in _COMPONENTS}
        self._omega = jnp.asarray(omega)

    # ================================================================
    #  The forcing term
    # ================================================================
    #: The forcing is pointwise (zero halo), but the traced-clock
    #: sinusoid is a jax scalar the halo tracer cannot follow
    #: (V-N2) — declare the substitute instead of being traced.
    extra_halo = HaloSpec({})

    @fr.model.term(advances=_COMPONENTS, name="wave_maker")
    def _force(self, state, ctx) -> dict:  # noqa: ANN001
        r"""``dz/dt += A sin(omega t) z_W`` off the traced clock."""
        amp = ctx.params[POLARIZED_AMPLITUDE]
        # ctx.clock is the Clock in-run, a bare stage-time scalar in
        # dry-run/tendency contexts (the schedule.context idiom)
        time = getattr(ctx.clock, "time", ctx.clock)
        oscillation = amp * jnp.sin(self._omega * time)
        return {name: oscillation * state[f"wavemaker_{name}"]
                for name in _COMPONENTS}
