"""
Transform error types (``fr.transforms`` cluster, task 2.8).

Description
-----------
The two error entries of the state-transform algebra
(``notes/framework2/model/classes/transforms.md`` §"SignatureMismatchError,
TraceError"; design source ``notes/framework2/model/08_state_transforms.md``
§10.3 laws 2/3). Both also register in the model-layer error surface
(``fridom.framework2.model.errors``).
"""
from __future__ import annotations


class SignatureMismatchError(TypeError):

    """
    Compose- or call-time state-transform signature mismatch.

    Description
    -----------
    Raised by the algebra's eager compose-time checks and by the
    call-time ``StateSignature.validate_input`` recheck. Carries the
    composition-tree path, a componentwise diff (missing / extra /
    space-mismatched components, in order), and the grid-identity
    verdict with a hint (a same-shaped different grid object teaches
    "one grid, many models: build both on one grid").
    """


class TraceError(TypeError):

    """
    A Tier-2 transform received tracer-valued input.

    Description
    -----------
    A ``traceable=False`` transform runs a model internally and thus
    cannot appear under ``jit``/``vmap``/``grad``. The taught guidance
    is to hoist the call to the host level, or use a Tier-1 transform
    inside the traced region.
    """


class FixedPointDivergenceError(RuntimeError):

    """
    A ``FixedPoint(on_divergence="raise")`` iteration diverged.

    Description
    -----------
    Raised when the update norm grows (``err > prev``) under the
    ``"raise"`` divergence policy; the message carries the full error
    series. (Spec-proposed spelling — transforms.md open question 5
    reserves the naming for confirmation; the alternatives
    ``"stop_best"`` / ``"ignore"`` never raise.)
    """
