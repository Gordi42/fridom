r"""The term-envelope module: scale selected terms by rho(t)."""
from __future__ import annotations

from functools import partial

from fridom.framework.utils import jaxify
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration, leaf
from fridom.model.params import RAMPING_ENVELOPE
from fridom.model.term_predicates import TermPredicate


@partial(jaxify, dynamic=("envelope",))
class TendencyEnvelope(Module):

    r"""
    Multiply selected tendency terms by an envelope :math:`\rho(t)`.

    Description
    -----------
    Declares no fields, terms, or stages: the module carries exactly
    one dynamic leaf, ``envelope``, provided as
    ``fr.params.RAMPING_ENVELOPE`` (``"ramping.envelope"``), plus a
    static :class:`~fridom.model.term_predicates.TermPredicate`
    selecting the enveloped terms. The ``TendencyComposer`` detects
    the module (through its ``envelope_terms`` capability attribute)
    and wraps every matching term's hook so its contribution dict is
    scaled by the stage-time envelope value

    .. math::
        \partial_t z = L z + \rho(t)\,[\text{matched terms}].

    The envelope value is **never** closed over as a host value: the
    wrapped hook reads ``ctx.params[fr.params.RAMPING_ENVELOPE]``,
    which resolves this module's live leaf at stage time (the D2
    no-host-capture rule). A ``fr.Ramp``-valued ``envelope`` therefore
    rides the carry exactly like ``FPlaneCoriolis.f0`` — stage-time
    resolved, zero-recompile sweeps — and is how
    ``AdiabaticRamping(envelope=...)`` drives its legs.

    The composer refuses a predicate that matches an ``IMPLICIT``
    term (an enveloped implicit solve is unsound), a ``linear=True``
    term (ramping ``L`` is the parameter-deformation path's job), or
    nothing at all (error without a ``term_filter``; a warning under
    one), and at most one envelope module may join an assembly.

    Parameters
    ----------
    terms : TermPredicate
        The ``fr.terms`` predicate selecting the enveloped tendency
        terms (e.g. ``~fr.terms.linear & fr.terms.explicit``);
        keyword-only.
    envelope : float | TimeDependent, optional
        The envelope value :math:`\rho` — a constant or a time
        curve such as ``fr.Ramp(0.0, 1.0, period=...)``
        (default: 1.0).
    """

    field_declarations = ()

    parameter_declarations = (
        ParameterDeclaration(
            RAMPING_ENVELOPE, attr="envelope", units="n/a",
            doc="tendency-term envelope rho(t)"),
    )

    def __init__(
        self,
        *,
        terms: TermPredicate,
        envelope: object = 1.0,
    ) -> None:
        """Store the predicate and the envelope leaf; see class doc."""
        if not isinstance(terms, TermPredicate):
            raise TypeError(
                "TendencyEnvelope(terms=...) takes an fr.terms "
                "predicate (a TermPredicate, e.g. ~fr.terms.linear & "
                f"fr.terms.explicit); got {terms!r}. A plain callable "
                "cannot join the assembly fingerprint — compose the "
                "selection from the fr.terms leaves instead")
        self.envelope = leaf(envelope)
        self._terms = terms

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def envelope_terms(self) -> TermPredicate:
        """The enveloped-term predicate (the composer's capability)."""
        return self._terms
