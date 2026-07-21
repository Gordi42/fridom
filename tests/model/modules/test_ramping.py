"""Tests for the TendencyEnvelope term-envelope module."""
import jax.numpy as jnp
import pytest

from fridom.model import term_predicates as terms
from fridom.model.modules.ramping import TendencyEnvelope
from fridom.model.params import RAMPING_ENVELOPE
from fridom.model.time_dependent import Ramp


# ================================================================
#  Construction
# ================================================================
def test_default_envelope_is_a_unit_leaf():
    module = TendencyEnvelope(terms=~terms.linear & terms.explicit)
    assert isinstance(module.envelope, jnp.ndarray)
    assert float(module.envelope) == pytest.approx(1.0)


def test_float_envelope_coerces_to_an_array_leaf():
    module = TendencyEnvelope(terms=terms.explicit, envelope=0.25)
    assert isinstance(module.envelope, jnp.ndarray)
    assert float(module.envelope) == pytest.approx(0.25)


def test_ramp_envelope_passes_through_untouched():
    ramp = Ramp(0.0, 1.0, period=2.0, curve="exp")
    module = TendencyEnvelope(terms=terms.explicit, envelope=ramp)
    assert module.envelope is ramp


def test_envelope_terms_exposes_the_predicate():
    predicate = ~terms.linear & terms.explicit
    module = TendencyEnvelope(terms=predicate)
    assert module.envelope_terms is predicate


def test_non_predicate_terms_is_a_taught_type_error():
    with pytest.raises(TypeError, match="TermPredicate"):
        TendencyEnvelope(terms=lambda _key, _term: True)


def test_terms_is_keyword_only():
    with pytest.raises(TypeError):
        TendencyEnvelope(terms.explicit)  # positional slot refused


# ================================================================
#  The provides row (assembly reads parameter_declarations)
# ================================================================
def test_provides_the_ramping_envelope_parameter():
    module = TendencyEnvelope(terms=terms.explicit)
    declarations = module.parameter_declarations
    assert len(declarations) == 1
    assert declarations[0].name == RAMPING_ENVELOPE
    assert declarations[0].attr == "envelope"


def test_declares_no_fields():
    module = TendencyEnvelope(terms=terms.explicit)
    assert module.field_declarations == ()
