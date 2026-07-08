r"""
Eigenmode-backed projection transforms (wave 7 C).

Description
-----------
The shared Tier-1 base for the package-specific vortical / wave /
divergence projections (``nh.transforms`` / ``sw.transforms``). An
:class:`EigenProjection` wraps a coefficient-basis, per-mode spectral
projector (``em.projector(s)``) as a composable ``State -> State``
transform: the projector onto the span of a *set* of eigenmodes is the
sum of the single-mode projectors, and — the modes being
:math:`M`-orthogonal — that sum is itself an orthogonal projector
(idempotent). ``notes/framework2/projection_eigenmode_plan.md`` §4.3;
``notes/framework2/model/08_state_transforms.md`` §10.5.

Two design notes:

- **Idempotency for the algebra.** ``WaveProjection = P(+1) + P(-1)``
  and ``DivergenceProjection = (P_vortical + P_wave).complement`` need
  the sum of orthogonal projectors to *stay* an (idempotent) projector.
  The generic ``Sum`` node declares no idempotency (structural
  normalization only), so :meth:`EigenProjection.__add__` **merges the
  mode sets** of two projections on the same eigenmode set into one
  idempotent ``EigenProjection`` — the ``+`` and ``.complement``
  spellings hold, while ``.complement`` (idempotent-gated) still works.
  Heterogeneous ``+`` (e.g. against ``Identity``) falls back to the
  generic ``Sum``.
- **Coefficient basis.** The projector acts in the eigenmode's
  coefficient basis; the package ``project_fn`` owns the (optional)
  forward/inverse transform round-trip that carries a physical state in
  and out (clean for the shallow-water *collocated* modes; the nonhydro
  staggered-physical round-trip needs the deferred staggered spectral
  transforms).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.framework.utils import jaxify
from fridom.framework2.transforms.base import StateTransform
from fridom.framework2.transforms.info import TransformInfo

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.framework2.transforms.signature import StateSignature


@jaxify
class EigenProjection(StateTransform):

    r"""
    Projection onto the span of a set of eigenmodes (Tier 1).

    Description
    -----------
    Holds the eigenmode set, the mode signs it projects onto, its
    ``(name, space)`` signature, and the package ``project_fn`` that
    realizes ``sum_s P_s`` (coefficient projector + optional transform
    round-trip). Idempotent (a sum of :math:`M`-orthogonal mode
    projectors is an orthogonal projector); ``__add__`` merges mode
    sets on the same eigenmode object so ``.complement`` stays valid.

    Parameters
    ----------
    eigenmodes : object
        The package ``Eigenmodes`` object (its ``projector(s)`` is
        wrapped; compared by identity when merging).
    modes : tuple[int, ...]
        The mode signs projected onto (e.g. ``(0,)`` vortical,
        ``(-1, 1)`` wave).
    signature : StateSignature | None
        The endo domain/codomain signature; ``None`` is
        signature-polymorphic (the nonhydro staggered-spectral
        signature is deferred, so its projections run polymorphic).
    project_fn : Callable
        ``(eigenmodes, modes, state) -> state`` realizing ``sum_s
        P_s``; identity-compared when merging.
    name : str
        A short repr label (e.g. ``"VorticalProjection"``).
    """

    def __init__(
        self,
        *,
        eigenmodes: object,
        modes: tuple[int, ...],
        signature: StateSignature | None,
        project_fn: Callable,
        name: str,
    ) -> None:
        """Store the modes, signature and package projector."""
        self._eigenmodes = eigenmodes
        self._modes = tuple(sorted(set(modes)))
        self._signature = signature
        self._project_fn = project_fn
        self._name = name

    # ================================================================
    #  Declared structure
    # ================================================================
    @property
    def eigenmodes(self) -> object:
        """The wrapped eigenmode set."""
        return self._eigenmodes

    @property
    def modes(self) -> tuple[int, ...]:
        """The mode signs projected onto (sorted)."""
        return self._modes

    @property
    def domain(self) -> StateSignature | None:
        """The endo domain signature (``None`` if polymorphic)."""
        return self._signature

    @property
    def codomain(self) -> StateSignature | None:
        """The endo codomain signature (``None`` if polymorphic)."""
        return self._signature

    @property
    def idempotent(self) -> bool:
        """A sum of ``M``-orthogonal mode projectors is a projector."""
        return True

    # ================================================================
    #  Application
    # ================================================================
    def _evaluate(
        self, state: object,
    ) -> tuple[object, TransformInfo]:
        """Apply ``sum_s P_s`` via the package projector."""
        out = self._project_fn(self._eigenmodes, self._modes, state)
        return out, TransformInfo.EMPTY

    # ================================================================
    #  Mode-merging sum (keeps the composite idempotent)
    # ================================================================
    def __add__(self, other: object) -> StateTransform:
        """Merge mode sets on the same eigenmodes; else generic Sum."""
        if (isinstance(other, EigenProjection)
                and other._eigenmodes is self._eigenmodes
                and other._project_fn is self._project_fn):
            merged = tuple(sorted(set(self._modes) | set(other._modes)))
            return EigenProjection(
                eigenmodes=self._eigenmodes,
                modes=merged,
                signature=self._signature,
                project_fn=self._project_fn,
                name=f"{self._name}+{other._name}")
        return super().__add__(other)

    def __repr__(self) -> str:
        """``name(modes=...)``."""
        return f"{self._name}(modes={self._modes})"


class ProjectionFactory:

    r"""
    Dual-source builder for a named eigenmode projection.

    Description
    -----------
    The public ``VorticalProjection`` / ``WaveProjection`` /
    ``DivergenceProjection`` handles: call with an explicit
    ``Eigenmodes`` (``VorticalProjection(em)``), or build from an
    assembled model (``VorticalProjection.from_model(model, ...)``,
    which resolves the eigenmodes first). Both routes run the same
    ``build`` callable.

    Parameters
    ----------
    build : Callable[[object], StateTransform]
        Builds the projection transform from an ``Eigenmodes`` object.
    eigenmodes_from_model : Callable[..., object]
        The package ``eigenmodes.from_model`` (``model, **kwargs ->
        Eigenmodes``).
    name : str
        The projection name (its repr).
    """

    def __init__(
        self,
        build: Callable,
        eigenmodes_from_model: Callable,
        name: str,
    ) -> None:
        """Store the build and eigenmode-resolution callables."""
        self._build = build
        self._from_model = eigenmodes_from_model
        self._name = name

    def __call__(self, eigenmodes: object) -> StateTransform:
        """Build the projection from an explicit ``Eigenmodes``."""
        return self._build(eigenmodes)

    def from_model(
        self, model: object, **kwargs: object,
    ) -> StateTransform:
        """Resolve the eigenmodes from a model, then build."""
        return self._build(self._from_model(model, **kwargs))

    def __repr__(self) -> str:
        """Return the projection name."""
        return f"{self._name}"
