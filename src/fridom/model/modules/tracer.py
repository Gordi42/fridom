r"""
The one-liner declaring module for plain user tracers.

Description
-----------
``Tracer("dye")`` is the ergonomic face of
``FieldDeclaration.tracer``: a module that declares one PROGNOSTIC
field carrying the TRACER and ADVECTED roles and contributes
nothing else — no tendency term, no parameter, no dispatch. It is
the replacement for the old stack's ``mset.custom_state_fields``
(01_concepts D1.5, sketch 7.3): the extra field enters the state
vector through the same registration act every built-in tracer uses,
so the role-selecting consumers pick it up without a line of wiring —
advection transports it (it selects ``roles.ADVECTED`` at bind) and
TRACER-targeting closures mix it.

Several tracers are several modules::

    model = nh.Model(..., modules_extra=(fr.model.modules.Tracer("dye"),
                                         fr.model.modules.Tracer("age")))

The module owns no term, so an assembly that advances the tracer by
nothing at all — ``advection=False`` with no closure targeting it —
is rejected by the D1.4 coverage lint (``AssemblyError``: "PROGNOSTIC
fields (...) are advanced by no term"). That is the intended
diagnosis, not a limitation of this module: a passive tracer in a
model with no transport is a typo. Give the assembly an advection
scheme, add a closure, or write a declaring module with a term of
its own.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.model.declarations import FieldDeclaration
from fridom.model.module import Module
from fridom.spatial.space_patterns import Collocated

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from fridom.spatial.space_patterns import SpacePattern, SpaceRule


class Tracer(Module):

    r"""
    Declare one passive tracer field, advected and mixable.

    Description
    -----------
    A declarations-only module wrapping
    :meth:`~fridom.model.declarations.FieldDeclaration.tracer`: the
    named field is PROGNOSTIC with the roles ``{TRACER, ADVECTED}``,
    so the assembled advection scheme transports it and
    TRACER-selecting closures (``fr.closures.HarmonicMixing`` and
    friends) diffuse it, with no further wiring.

    .. code-block:: python

        model = nh.Model(grid=grid, ...,
            modules_extra=(fr.model.modules.Tracer("dye", units="1"),))
        model.set_fields(dye=...)

    The tracer is advanced by advection alone: an assembly in which
    no term advances it (``advection=False`` and no closure
    targeting TRACER) is rejected by the D1.4 coverage lint.

    Parameters
    ----------
    name : str
        The flat, dot-free field name.
    space : SpacePattern | SpaceRule | None, optional
        The space descriptor; None resolves to ``Collocated()``,
        optionally carrying ``family=`` (default: None).
    family : str | None, optional
        Shorthand for the discretization family of the default
        collocated space (FV-D1b): ``"fv"`` lands the tracer on the
        average family (``CellAvg``), ``"nodal"`` collocates it with
        the pressure cell, None defers to the grid-level default.
        Mutually exclusive with an explicit ``space=`` — spell the
        family inside the pattern there (default: None).
    default : float | Callable | None, optional
        The background initializer: None means zeros, a number a
        constant fill, a callable of coordinate names a sampled
        profile. Not the initial condition — that is
        ``model.set_fields`` (default: None).
    long_name : str, optional
        Descriptive nc-style name (default: "Unnamed").
    units : str, optional
        Physical units annotation (default: "n/a").
    nc_attrs : Mapping[str, str] | None, optional
        Extra netCDF attributes (default: None).

    Raises
    ------
    ValueError
        If both ``space=`` and ``family=`` are given, or if the
        wrapped declaration rejects the name or the default.
    """

    def __init__(
        self,
        name: str,
        *,
        space: SpacePattern | SpaceRule | None = None,
        family: str | None = None,
        default: float | Callable | None = None,
        long_name: str = "Unnamed",
        units: str = "n/a",
        nc_attrs: Mapping[str, str] | None = None,
    ) -> None:
        """Store the tracer's declaration arguments (validated)."""
        if space is not None and family is not None:
            raise ValueError(
                f"tracer {name!r}: space= and family= are mutually "
                "exclusive; family= is the shorthand for the default "
                "collocated space, so spell it inside the pattern "
                "instead (e.g. space=fr.spatial.Staggered('z', "
                f"family={family!r}))")
        self._name: str = name
        self._space: SpacePattern | SpaceRule | None = space
        self._family: str | None = family
        self._default: float | Callable | None = default
        self._long_name: str = long_name
        self._units: str = units
        # kept as sorted pairs: the module is a pytree whose statics
        # must stay cheaply, structurally comparable (a dict is not
        # hashable)
        self._nc_attrs: tuple[tuple[str, str], ...] | None = (
            None if nc_attrs is None
            else tuple(sorted(nc_attrs.items())))
        # build the declaration once so a bad name/default/units is
        # reported here, at the call site, instead of at assembly
        # (the declaration itself is rebuilt per access: it is host
        # data without value equality and must never become an
        # instance attribute of a pytree module)
        _ = self.field_declarations

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def name(self) -> str:
        """The declared tracer's field name."""
        return self._name

    # ================================================================
    #  Declarations
    # ================================================================
    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """The single PROGNOSTIC + {TRACER, ADVECTED} declaration."""
        space = (Collocated(family=self._family)
                 if self._space is None else self._space)
        return (
            FieldDeclaration.tracer(
                self._name, space=space, default=self._default,
                long_name=self._long_name, units=self._units,
                nc_attrs=(None if self._nc_attrs is None
                          else dict(self._nc_attrs))),
        )

    # ================================================================
    #  Representation
    # ================================================================
    def __repr__(self) -> str:
        """Show the tracer name (the module's whole content)."""
        return f"Tracer({self._name!r})"
