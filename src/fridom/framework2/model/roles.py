"""
Physical roles of state components (``fr.roles``).

Description
-----------
Owning class spec: ``design/specs/model/classes/declarations.md``
("Role, ``Velocity``, and the ``fr.roles`` namespace"). Roles are
opt-in, typed, namespaced marker objects tagging what a PROGNOSTIC
field *is*, physically (model D1.4) — with the one signed exception
that ``Velocity`` may also mark DIAGNOSTIC fields (V-H2, the
hydrostatic diagnosed ``w``). They are frozen, value-hashable static
markers: equality and hash go by key, so independently imported role
constants select the same fields. Family matching is
class-vs-instance (``table.select(Velocity)`` matches any component;
``Velocity("x")`` exactly one); that query surface is the model
cluster's — no role logic runs inside the traced step.
"""
# Wave 2 A: Role, Velocity, ADVECTED, TRACER
from __future__ import annotations

from typing import Final, final


class Role:

    """
    Frozen, namespaced marker; identity/equality/hash by key.

    Description
    -----------
    Open by construction: packages mint their own roles
    (``Role("mybgc.nutrient")``) in their own namespaces. The key is
    dotted — the namespace discipline mirrors parameter names, and
    keeps user roles collision-free without a registry. Roles are
    importable, documented module-level objects, never process-global
    mutable state.

    Parameters
    ----------
    key : str
        The namespaced (dotted) role key, e.g. ``"mybgc.nutrient"``.
    """

    __slots__ = ("_key",)

    def __init__(self, key: str) -> None:
        """Create a flat role with a namespaced key."""
        if not isinstance(key, str) or not key:
            raise TypeError(
                f"role keys are non-empty strings, got {key!r}")
        if "." not in key:
            raise ValueError(
                f"role key {key!r} is not namespaced; keys are "
                "dotted by package, e.g. Role('mybgc.nutrient') "
                "(the framework's own live under 'fridom.')")
        self._key: str | tuple[str, ...] = key

    # ================================================================
    #  Value semantics (frozen, hashable by key)
    # ================================================================
    @property
    def key(self) -> str | tuple[str, ...]:
        """The namespaced key (``"fridom.advected"``); Velocity: a pair."""
        return self._key

    def __eq__(self, other: object) -> bool:
        """Equality by key (value semantics)."""
        if not isinstance(other, Role):
            return NotImplemented
        return self._key == other._key

    def __hash__(self) -> int:
        """Value hash by key, matching ``__eq__``."""
        return hash(self._key)

    def __repr__(self) -> str:
        """Round-tripping repr; roles are importable objects."""
        return f"Role({self._key!r})"


@final
class Velocity(Role):

    """
    The one parameterized role family: a velocity component label.

    Description
    -----------
    The component label is explicit, never space-derived (B-grid
    ``u``/``v`` share one interned space; A-grids carry no signal).
    It is opaque to the framework — keying and stable order only,
    never math. On tensor-product grids a label absent from
    ``grid.names`` marks a transverse (slaved) component (V-N1),
    excluded from directional consumers while remaining a full
    family member for friction, CFL, and energy. Final: family
    matching is class-vs-instance, so subclasses would silently
    change selection semantics.

    Parameters
    ----------
    component : str
        The explicit component label (normally a coordinate name).
    """

    __slots__ = ("_component",)

    def __init__(self, component: str) -> None:
        """Create the component-labelled velocity role."""
        if not isinstance(component, str) or not component:
            raise TypeError(
                f"velocity component labels are non-empty strings, "
                f"got {component!r}")
        self._component: str = component
        self._key = ("fridom.velocity", component)

    def __init_subclass__(cls) -> None:
        """Reject subclasses: the family is final by design."""
        raise TypeError(
            "Velocity is final: family selection matches the class "
            "itself, so a subclass would silently change which "
            "fields role queries select")

    @property
    def component(self) -> str:
        """Explicit component label (normally a coordinate name)."""
        return self._component

    def __repr__(self) -> str:
        """Round-tripping repr, e.g. ``Velocity('x')``."""
        return f"Velocity({self._component!r})"


# ================================================================
#  The framework's own role constants
# ================================================================
#: transported by the advection scheme(s); strictly PROGNOSTIC
ADVECTED: Final[Role] = Role("fridom.advected")

#: a mixed tracer quantity; strictly PROGNOSTIC (TRACER without
#: ADVECTED warns at declaration — suppressible)
TRACER: Final[Role] = Role("fridom.tracer")
