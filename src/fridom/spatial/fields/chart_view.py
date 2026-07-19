r"""A read-only chart-native view of a model state's velocity trio.

Description
-----------
``ChartView`` is the small read-only object a vocabulary ``State``
returns from ``state.chart`` (``physical_state_components.md`` ruling
(d)). The user-facing state components ``u, v, w`` denote the
**physical** velocities on every grid; ``state.chart`` exposes the
matching **chart-native** quantities — the contravariant volume flux
``J\omega`` on a mapped column, the coordinate velocities on an
embedding chart — derived on demand, never stored, never written.

The view is deliberately generic: it holds a per-package *derivation
hook* ``derive(state, name) -> ScalarField`` and the ordered velocity
component names. It performs **no** ``isinstance`` on the vocabulary
``State`` classes (CS-14); each package supplies its own hook, which
decides what "chart-native" means for that geometry (the sphere's
horizontal quantity is a coordinate velocity, the mapped vertical is a
J-weighted flux) and returns the identity for uncoupled components and
on unmapped grids.

Surface: ``chart["w"]`` and ``chart.w`` both derive the ``w``
component; ``u, v, w = chart.velocities`` destructures the trio in
grid axis order (vertical last). Any write is refused.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.fields.vector_field import VectorField


class ChartView:

    r"""Read-only view exposing a state's chart-native components.

    Description
    -----------
    Constructed by a vocabulary ``State``'s ``chart`` property with the
    state, the package's derivation hook and the ordered velocity
    component names. Every read routes through the hook; every write is
    refused. See the module docstring for the contract.

    Parameters
    ----------
    state : VectorField
        The physical-component state the view derives from.
    derive : Callable[[VectorField, str], ScalarField]
        The per-package hook mapping a component name to its
        chart-native field (the identity for uncoupled components and
        on unmapped grids).
    velocities : tuple[str, ...]
        The velocity component names in grid axis order (vertical
        last) — the destructuring order of :attr:`velocities`.
    """

    __slots__ = ("_derive", "_state", "_velocities")

    def __init__(
        self,
        state: VectorField,
        derive: Callable[[VectorField, str], ScalarField],
        velocities: tuple[str, ...],
    ) -> None:
        """Store the state, the derivation hook and the velocity names."""
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_derive", derive)
        object.__setattr__(self, "_velocities", tuple(velocities))

    # ================================================================
    #  Read surface (item, attribute, destructuring)
    # ================================================================
    def __getitem__(self, name: str) -> ScalarField:
        """Return the chart-native ``name`` component (``chart["w"]``)."""
        if not isinstance(name, str):
            raise TypeError(
                f"chart components are indexed by name, got {name!r}")
        return self._derive(self._state, name)

    def __getattr__(self, name: str) -> ScalarField:
        """Return the chart-native ``name`` component (``chart.w``).

        Description
        -----------
        Any non-underscore attribute is treated as a component-name
        read routed through the hook, so ``chart.w`` mirrors
        ``chart["w"]``. Underscore names are never components (the
        slots resolve normally; a genuine miss raises
        ``AttributeError``), which keeps ``copy`` / pickle protocols
        well-behaved.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return self._derive(self._state, name)

    @property
    def velocities(self) -> tuple[ScalarField, ...]:
        """The chart-native velocity trio (grid axis order, vertical last)."""
        return tuple(
            self._derive(self._state, name) for name in self._velocities)

    # ================================================================
    #  Read-only guards
    # ================================================================
    def __setitem__(self, name: str, value: object) -> None:
        """Refuse writes: ``state.chart`` is a read-only view."""
        raise TypeError(
            "state.chart is a read-only view of the chart-native "
            "components; write the physical component on the state "
            f"instead (state[{name!r}] = ...)")

    def __setattr__(self, name: str, value: object) -> None:
        """Refuse attribute writes: ``state.chart`` is read-only."""
        raise AttributeError(
            "state.chart is a read-only view of the chart-native "
            "components; write the physical component on the state "
            "instead")

    def __repr__(self) -> str:
        """Return a terse identification carrying the velocity names."""
        return f"ChartView(velocities={self._velocities})"
