"""
The Module base class.

Description
-----------
``Module``: the unit of physics/numerics contributing any subset of
the capability menu to an assembled ``fr.Model`` — field
declarations and references (D1), provided/consumed parameters (D2),
dispatch overrides (D4), tendency terms and stages (D3), and the
per-substage ``self_update`` of its own dynamic state. "Computes a
tendency" is one capability, not the definition of a module. Owning
class spec: ``notes/framework2/model/classes/module.md``.

Modules ride the traced carry: subclasses are jaxify-registered
pytrees (``@partial(fr.utils.jaxify, dynamic=(...))``; registration
is automatic on subclassing, the decorator only declares leaves).
The hardened-jaxify discipline is normative: dynamic leaves are
numeric values coerced through ``jnp.asarray`` (or pytree values
such as ``fr.Ramp`` / named ``ScalarField``s — legal since metadata
became annotation-exempt from aux equality); statics must stay
cheaply structurally comparable, with identity-hashed statics (grid,
spaces) interned; host-side observers belong in
``_eq_ignored_attrs``. ``tree_unflatten`` bypasses ``__init__``, so
no constructor-established invariant may be assumed in-trace, and
declared leaf order must never change in a released module.

``bind(table)`` is the one sanctioned mutation site — host-side,
once per instance, before the registration freeze. It precomputes
operators, spaces, and name tuples ONLY (a bind-materialized field
is a bug: fields exist only from assembly step 8). On completion the
instance freezes: attribute pokes raise ``ImmutableParameterError``
(the teaching shim; leaf values change through
``model.update_parameters``, structure through re-assembly).
Bind-time reads of time-dependent parameters go through
`BindParameterView` and raise ``TimeDependentParameterError`` unless
spelled ``at_time(0.0)`` — grid factor at bind, parameter factor
in-step.
"""
# Wave 3 B: Module (declarations, references, dispatch, terms,
#           stages, bind, extra_halo, state_type)
from __future__ import annotations

import dataclasses
import functools
import weakref
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from fridom.framework.utils import jaxify
from fridom.framework2.model.errors import (
    ImmutableParameterError,
    TimeDependentParameterError,
)
from fridom.framework2.model.stages import (
    STAGE_ATTRIBUTE,
    Stage,
    StageKind,
)
from fridom.framework2.model.terms import TERM_ATTRIBUTE, TendencyTerm
from fridom.framework2.model.time_dependent import (
    TimeDependent,
    resolve_at,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator

    from fridom.framework2.grid.decomposition.halo import HaloSpec
    from fridom.framework2.model.declarations import (
        FieldDeclaration,
        FieldReference,
    )
    from fridom.framework2.model.parameters import (
        ParameterDeclaration,
        ParameterReference,
    )
    from fridom.framework2.model.space_patterns import SpacePattern


# ================================================================
#  Bind bookkeeping (host-side; never part of the pytree)
# ================================================================
# The bound/frozen state is deliberately kept OUT of the instance
# __dict__: jaxify's aux data is the whole __dict__ minus the dynamic
# leaves, and tree_unflatten rebuilds objects attribute by attribute
# through __setattr__ — an in-dict freeze flag would either poison
# the aux equality or break unflattening. An id-keyed side table with
# weakref staleness guards (the jax_utils._EQ_MEMO pattern) freezes
# the host-side instance only; in-trace unflattened copies are
# transient and stay writable (hooks are pure and never mutate self).

_BOUND_MODULES: Final[dict[int, weakref.ref]] = {}
_BINDING_IN_PROGRESS: Final[set[int]] = set()


def _is_bound(module: Module) -> bool:
    """Return whether `module` completed its bind (host-frozen)."""
    ref = _BOUND_MODULES.get(id(module))
    if ref is None:
        return False
    if ref() is module:
        return True
    # stale entry: the id was reused by a new object
    del _BOUND_MODULES[id(module)]
    return False


def _mark_bound(module: Module) -> None:
    """Freeze `module`: record it in the bound-instance table."""
    key = id(module)

    def _drop(ref: weakref.ref, key: int = key) -> None:
        """Drop the entry when the module is garbage collected."""
        if _BOUND_MODULES.get(key) is ref:
            del _BOUND_MODULES[key]

    _BOUND_MODULES[key] = weakref.ref(module, _drop)


def _install_bind_guard(cls: type) -> None:
    """
    Wrap a class-defined ``bind`` with the once/freeze guard.

    Description
    -----------
    Every ``bind`` defined in a `Module` subclass body is wrapped at
    class creation: entering an already-bound instance raises
    ``ImmutableParameterError``; on completion of the outermost call
    the instance is marked bound (attribute pokes raise from then
    on). Cooperative ``super().bind(table)`` calls are reentrant —
    only the outermost frame marks. A failed bind leaves the
    instance unbound (assembly aborts anyway; retry stays possible).
    """
    fn = cls.__dict__.get("bind")
    if fn is None or getattr(fn, "_fridom_bind_guarded", False):
        return

    @functools.wraps(fn)
    def bind(self: Module, table: Any) -> Any:
        """Run the wrapped bind under the once/freeze guard."""
        if _is_bound(self):
            raise ImmutableParameterError(
                f"bind(table) runs exactly once per module instance;"
                f" this {type(self).__name__} is already bound — "
                "assemble a fresh instance to re-bind (the script is"
                " the recipe; no pickled models)")
        key = id(self)
        outermost = key not in _BINDING_IN_PROGRESS
        if outermost:
            _BINDING_IN_PROGRESS.add(key)
        try:
            result = fn(self, table)
        finally:
            if outermost:
                _BINDING_IN_PROGRESS.discard(key)
        if outermost:
            _mark_bound(self)
        return result

    bind._fridom_bind_guarded = True  # noqa: SLF001
    cls.bind = bind


# ================================================================
#  The bind-time parameter read gate
# ================================================================
class BindParameterView(Mapping[str, object]):

    """
    Bind-time parameter reads, gated against stale coefficients.

    Description
    -----------
    A read-only mapping over the parameter values visible during
    ``bind(table)`` (the proposed seam: assembly builds the bind
    table's ``parameters`` attribute from the
    ``ParameterBindingTable`` using this class; tests duck-type the
    table). A bare read of a `TimeDependent` value raises
    ``TimeDependentParameterError`` — the bind/in-step split (D2.1):
    grid factors resolve at bind, parameter factors resolve in-step
    (this kills the ``BiharmonicClosure`` stale-coefficient bug
    class). The sanctioned spelling for a deliberate assembly-time
    value is ``table.parameters.at_time(0.0)[name]``.

    Membership tests and iteration never evaluate values; only
    ``__getitem__`` (and the ``Mapping`` mixins built on it) is
    gated.

    Parameters
    ----------
    values : Mapping[str, object]
        The bind-time parameter values by canonical dotted name;
        entries may be plain scalars or `TimeDependent` curves.
    """

    def __init__(self, values: Mapping[str, object]) -> None:
        self._values = dict(values)

    def __getitem__(self, name: str) -> object:
        """Return the value; raise on bare time-dependent reads."""
        value = self._values[name]
        if isinstance(value, TimeDependent):
            raise TimeDependentParameterError(
                f"bare bind-time read of time-dependent parameter "
                f"{name!r}: grid factors resolve at bind, parameter "
                "factors resolve in-step (ctx.params). To freeze "
                "the initial value deliberately, spell the read "
                f"table.parameters.at_time(0.0)[{name!r}]")
        return value

    def __contains__(self, name: object) -> bool:
        """Return membership without evaluating the value."""
        return name in self._values

    def __iter__(self) -> Iterator[str]:
        """Iterate over the parameter names."""
        return iter(self._values)

    def __len__(self) -> int:
        """Return the number of visible parameters."""
        return len(self._values)

    def at_time(self, t: float) -> dict[str, object]:
        """
        Evaluate every time-dependent entry explicitly at ``t``.

        Description
        -----------
        The sanctioned assembly-time spelling (mirrors
        ``ParameterView.at_time``): `TimeDependent` values are
        evaluated at ``t`` via ``resolve_at``; plain scalars pass
        through unchanged.

        Parameters
        ----------
        t : float
            The (signed) clock time to evaluate at — bind-time
            reads use ``0.0``.

        Returns
        -------
        dict[str, object]
            All visible parameters with time-dependent values
            resolved at ``t``.
        """
        return {name: resolve_at(value, t)
                for name, value in self._values.items()}

    def __repr__(self) -> str:
        """Return a names-only repr (values may be curves)."""
        return f"BindParameterView({sorted(self._values)!r})"


# ================================================================
#  Capability collection helpers
# ================================================================
def _declared_members(cls: type) -> dict[str, Any]:
    """
    Return the class members in definition order.

    Description
    -----------
    Walks the MRO base-first, so inherited members come before
    subclass additions and an override keeps its original position
    (dict insertion-order semantics) — the deterministic declaration
    order the term/stage collectors rely on.
    """
    members: dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        members.update(vars(klass))
    return members


# ================================================================
#  Module
# ================================================================
class Module:

    """
    A unit of physics/numerics contributing capabilities to a Model.

    Description
    -----------
    Concrete base, subclassable, with no abstract methods: every
    capability is optional. The base defines no constructor and no
    dynamic leaves; subclasses own their ``__init__`` and declare
    their pytree leaves with
    ``@partial(fr.utils.jaxify, dynamic=("kh", "kv"))`` (pytree
    registration itself is automatic on subclassing). Convention:
    user-facing treatment overrides live on subclass constructors
    (``VerticalMixing(kv=..., treatment=fr.IMPLICIT)``), never on
    the Model.

    jaxify discipline for authors: dynamic = everything numeric that
    may be swept or ramped (parameter leaves, owned precomputed
    arrays), coerced through ``jnp.asarray`` at construction (pytree
    values — ``fr.Ramp``, named ``ScalarField``s — ride as declared
    leaves directly); static = structure (target name tuples, solver
    choices, substep counts, curve shapes — changing them is
    different math: one recompile, correct). Statics must stay
    cheaply structurally comparable and identity-hashed statics
    (grid, spaces) must be interned objects. Never reorder declared
    leaves in a released module: the treedef is snapshot-relevant.

    Assembled (bound) modules are read-only from the host: attribute
    pokes raise ``ImmutableParameterError`` — leaf values change
    through ``model.update_parameters``, structure through
    re-assembly.
    """

    # host-side bookkeeping attribute names excluded from the
    # direct structural comparison (jaxify `_eq_ignored_attrs`): the
    # base adds no host-side instance state (bind bookkeeping lives
    # in an id-keyed side table), so the base set is empty.
    # Subclasses carrying observers (counters, writer handles)
    # REDECLARE the full set — the lookup is not merged across the
    # MRO — and must ADDITIONALLY list the same names in jaxify's
    # `annotation=` category: under the landed jaxify,
    # `_eq_ignored_attrs` governs `_object_eq` (module == module and
    # modules nested as static values) while the treedef/jit-cache
    # aux equality is governed by the annotation category alone.
    _eq_ignored_attrs: frozenset[str] = frozenset()

    # ================================================================
    #  Fields (D1) — consumed at assembly steps 1 and 4
    # ================================================================

    field_declarations: tuple[FieldDeclaration, ...] = ()
    """What the module contributes to the state vector (D1.1).
    Transient assembly data — except AUXILIARY ``default=``
    closures, retained in the re-materialization table. May be an
    instance property built from constructor arguments."""

    field_references: tuple[FieldReference, ...] = ()
    """Components consumed but not owned: ``FieldReference(name,
    hint)``, checked at assembly (``MissingFieldError`` with the
    hint). No auto-creation — a reference carrying a space is a
    declaration in disguise (D1.5)."""

    # ================================================================
    #  Parameters (D2) — consumed at assembly step 2
    # ================================================================

    parameter_declarations: tuple[ParameterDeclaration, ...] = ()
    """Published scalars: ``ParameterDeclaration(name, attr=...)``
    names where the value lives (a dynamic leaf of this module),
    never a frozen copy. Provides implies constancy (02_rules)."""

    parameter_references: tuple[ParameterReference, ...] = ()
    """Consumed scalars — the exact twin of `field_references`.
    ``fr.Param(name, default=...)``-valued constructor slots are the
    defaulted spelling and are collected into this set by assembly
    (D2 reconciliation 4)."""

    # ================================================================
    #  Dispatch and negotiation inputs (D4) — steps 3 and 5
    # ================================================================

    dispatch: Mapping[str | tuple[str, SpacePattern], Any] = (
        MappingProxyType({}))
    """Constructor-frozen registry overrides, keyed ``kind`` or
    ``(kind, SpacePattern)``; model-resolved and merged into the
    grid registry exactly once, at assembly step 3. Values may be
    lazy operator factories (the transform-row mechanism);
    ``("declared_space", ...)`` resolver entries are never
    module-mergeable (grid-level only, D1.2)."""

    extra_halo: HaloSpec | None = None
    """Declared halo substitute for this module's terms/stages that
    the halo trace cannot follow (V-N2). Only stored and exposed
    here: assembly step 5 collects it, exempts the module's terms
    from the halo trace (validating them in the zero-valued second
    dry-run mode instead), and merges it into negotiation as
    trace-spec ``merge_max`` extra_halo (step 7)."""

    state_type: type | None = None
    """The ``State`` vocabulary class supplied by the dynamical-core
    module (D1.3 commitment 4); more than one provider across the
    module list is an assembly error; ``fr.Model(state_type=...)``
    is the fallback/override. Declared as a plain class attribute:
    it stays out of the instance ``__dict__`` and hence out of the
    pytree aux data entirely (no treedef or jit-cache coupling); an
    instance-level override is permitted and enters the aux, where
    class objects compare by identity — interned by nature."""

    # ================================================================
    #  Assembly hook (host-side, runs once — step 4)
    # ================================================================

    def bind(self, table: Any) -> None:  # noqa: ARG002
        """
        Precompute static bind state from the resolved field table.

        Description
        -----------
        Assembly step 4 (module order, after the dispatch merge, so
        precomputed operators are the ones that will actually run):
        freeze role selections to static name tuples
        (``table.select(ADVECTED)``, ``table.velocity()``), declare
        name couplings (``table.require("u", "v", "p")``), and do
        grid-factor precomputes. Default: no-op.

        ``bind`` precomputes operators, spaces, and name tuples ONLY
        — it must not materialize real fields (fields exist only
        from assembly step 8, after the final negotiation). The
        bind/in-step split (D2.1): grid factor at bind, parameter
        factor in-step — bind-time reads of time-dependent
        parameters raise ``TimeDependentParameterError`` unless
        spelled ``at_time(0.0)`` (see `BindParameterView`).

        Runs exactly once per instance, host-side, before the
        registration freeze; on completion the instance is frozen
        (attribute pokes raise ``ImmutableParameterError``).
        Overrides are guarded automatically at class creation and
        may — but need not — call ``super().bind(table)``.

        Parameters
        ----------
        table : FieldTable
            The resolved field table (model.md); its ``parameters``
            attribute is the bind-time parameter view.
        """
        return

    # ================================================================
    #  Terms and stages (D3) — collected post-bind, step 5
    # ================================================================

    stages: tuple[Stage, ...] = ()
    """Owned stage declarations (constructed spelling). Schedule
    position is a pure function of each stage's declared kind —
    never of module list position (D1.3 commitment 5). May be an
    instance property; a string ``fn`` names a method of this
    module, resolved to the unbound method by `collected_stages`."""

    def tendency_terms(self) -> tuple[TendencyTerm, ...]:
        """
        Return the tendency terms contributed by this module.

        Description
        -----------
        Default implementation: scan for ``@fr.term``-stamped
        methods in definition order (inherited terms first, an
        override at its original position) — trivial modules need
        zero ceremony; modules with constructed terms override this
        method. Collection runs after ``bind(table)``, so
        ``advances`` may come from role selections; it is a
        deterministic pure function of the bound module (the same
        tuple on re-collection — ``model.variant`` relies on it).
        The stamped ``fn`` slots are the plain class-body functions,
        stored UNBOUND per the aliasing rule.

        Returns
        -------
        tuple[TendencyTerm, ...]
            The declared terms, definition order.
        """
        terms: list[TendencyTerm] = []
        for member in _declared_members(type(self)).values():
            declaration = getattr(member, TERM_ATTRIBUTE, None)
            if isinstance(declaration, TendencyTerm):
                terms.append(declaration)
        return tuple(terms)

    def collected_stages(self) -> tuple[Stage, ...]:
        """
        Return every stage declaration owned by this module.

        Description
        -----------
        The stage collection assembly step 5 consumes; merges, in
        deterministic order:

        1. ``@fr.self_update``-stamped methods (and any other
           ``Stage``-stamped method), class definition order;
        2. a bare (undecorated) ``self_update`` method, wrapped into
           a SELF_UPDATE-kind stage with ``reads=()`` — the two
           spellings are equivalent;
        3. the declared `stages` tuple, declaration order, with
           string ``fn`` entries resolved to the unbound methods and
           default names filled from the function name.

        Like `tendency_terms`, a deterministic pure function of the
        bound module: the same tuple on re-collection. Kind ordering
        and all schedule validation are assembly machinery.

        Returns
        -------
        tuple[Stage, ...]
            The owned stages with resolved unbound ``fn`` slots.

        Raises
        ------
        AttributeError
            If a declared stage names a method this module does not
            define.
        """
        collected: list[Stage] = []
        for name, member in _declared_members(type(self)).items():
            declaration = getattr(member, STAGE_ATTRIBUTE, None)
            if isinstance(declaration, Stage):
                collected.append(self._resolved_stage(declaration))
            elif name == "self_update" and callable(member):
                # the bare method spelling == reads=()
                collected.append(Stage(
                    kind=StageKind.SELF_UPDATE, fn=member,
                    name="self_update"))
        collected.extend(
            self._resolved_stage(stage) for stage in self.stages)
        return tuple(collected)

    def _resolved_stage(self, stage: Stage) -> Stage:
        """
        Resolve a stage's ``fn`` string and default its name.

        Description
        -----------
        A string ``fn`` is the method name, resolved to the unbound
        method at collection (the aliasing rule: the composer calls
        ``fn(carry.modules[slot], state, ctx)``); a ``None`` name is
        filled from the function name (the ``"Module/stage"``
        attribution key part).
        """
        fn = stage.fn
        if isinstance(fn, str):
            resolved = getattr(type(self), fn, None)
            if not callable(resolved):
                # AttributeError is the honest type: the stage names
                # a method this module does not define
                raise AttributeError(  # noqa: TRY004
                    f"stage fn {fn!r}: {type(self).__name__} "
                    "defines no method of that name")
            fn = resolved
        name = stage.name
        if name is None:
            name = getattr(fn, "__name__", None)
        if fn is stage.fn and name == stage.name:
            return stage
        return dataclasses.replace(stage, fn=fn, name=name)

    # ================================================================
    #  The self-update hook (traced; S1 of every substage)
    # ================================================================

    # deliberately NOT defined on the base: defining a
    # ``self_update(self, state, ctx) -> dict`` method (bare, or
    # decorated with @fr.self_update(reads=...)) opts the module in;
    # write gate: own AUXILIARY, applied via replace. It runs per
    # SUBSTAGE — never use it for step-frequency accumulation (the
    # S6 DIAGNOSTIC accumulation idiom is the sanctioned home).

    # ================================================================
    #  Post-bind immutability (the teaching shim)
    # ================================================================

    def __setattr__(self, name: str, value: object) -> None:
        """Set an attribute; bound instances raise (teaching shim)."""
        if _is_bound(self):
            raise ImmutableParameterError(
                f"cannot set {name!r}: this {type(self).__name__} "
                "is assembled and read-only from the host — change "
                "leaf values through model.update_parameters(...), "
                "or re-assemble for structural changes")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Delete an attribute; bound instances raise."""
        if _is_bound(self):
            raise ImmutableParameterError(
                f"cannot delete {name!r}: this "
                f"{type(self).__name__} is assembled and read-only "
                "from the host — re-assemble for structural changes")
        object.__delattr__(self, name)


# ================================================================
#  Pytree registration and the subclass hook
# ================================================================
# The base is jaxified with zero dynamic leaves, so every subclass is
# automatically pytree-registered on creation; the per-class
# @partial(fr.utils.jaxify, dynamic=(...)) decorator then only
# declares the leaves. jaxify installs its own auto-registration
# __init_subclass__; it is replaced below by a hook that additionally
# installs the bind once/freeze guard on subclass-defined binds.

jaxify(Module)


def _module_init_subclass(cls: type, **kwargs: Any) -> None:
    """Register Module subclasses and guard their ``bind``."""
    super(Module, cls).__init_subclass__(**kwargs)
    jaxify(cls)
    _install_bind_guard(cls)


Module.__init_subclass__ = classmethod(_module_init_subclass)
_install_bind_guard(Module)
