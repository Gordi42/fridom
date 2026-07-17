.. _adiabatic-ramping:

=================
Adiabatic Ramping
=================

This chapter explains how FRIDOM deforms a model between two operator
configurations slowly enough that a balanced flow follows the
deformation without radiating waves, and how that single idea supplies
a projector onto the slow manifold, a diagnostic for imbalance, and the
ramping half of optimal balance. After reading it you can build the
four legs of a ramp, chain them into a staggered protocol, and measure
the diabatic leakage that the protocol leaves behind.

**Prerequisites.** The state-transform algebra (composition with
``@``, Tier-1 versus Tier-2 transforms) and the eigenmode projections
(``VorticalProjection`` and its siblings). The worked protocol is the
gallery example :doc:`/auto_examples/shallowwater/adiabatic_double_ramp`,
which this chapter refers to throughout.

.. _adiabatic-ramping-deformation:

Deforming a Model Between Two Configurations
============================================

Many questions in geophysical fluid dynamics compare two versions of
the same model that differ only in a few parameters: an f-plane against
a beta-plane, a linear system against its weakly nonlinear counterpart,
a resting stratification against a perturbed one. A *deformation* makes
that comparison continuous. It is a pair of endpoint parameter
assignments on one assembly, a *reference* configuration at
:math:`\lambda = 0` and a *target* configuration at
:math:`\lambda = 1`, together with a smooth path :math:`L(\lambda)` of
linear operators joining them, with :math:`L(0) = L_\mathrm{ref}` and
:math:`L(1) = L_\mathrm{target}`. Wherever every parameter enters the
operator affinely, which covers the Coriolis parameter, the nonlinear
scaling, and the stratification, the path is the convex combination

.. math::
    L(\lambda) = (1 - \lambda)\,L_\mathrm{ref}
                 + \lambda\,L_\mathrm{target} .

Here :math:`\lambda` is the deformation parameter and :math:`L` the
model's linear operator. FRIDOM never forms this combination by
evaluating two operators; a term shared by both endpoints is computed
once, and only a term that genuinely differs is weighted, so the
deformation costs no more than the target model itself.

A *leg* turns the deformation into motion by driving
:math:`\lambda(t) = \rho\big((t - t_0)/\tau\big)` with a ramp function
:math:`\rho` that rises from zero to one over the ramp period
:math:`\tau`, and whose time derivatives vanish at both ends. The
vanishing endpoint derivatives are what make the deformation
*adiabatic*: a mode of the instantaneous operator whose frequency stays
well separated from the others is carried along the path with an error
that decreases faster than any power of :math:`\tau` as the ramp
lengthens. In simple terms, if the deformation is slow compared with
the fast waves, a balanced state stays balanced. This is the adiabatic
theorem applied to a fluid model, following the fast-slow splitting
method of Rosenau, Chouksey, Eden, Koul and Oliver (in preparation);
the default ramp curve ``"exp"`` is their Gevrey-2 form, asymptotically
the most efficient of the three shipped curves (``"linear"``,
``"cosine"``, ``"exp"``).

.. _adiabatic-ramping-legs:

The Ramp and Its Four Legs
==========================

An ``AdiabaticRamping`` is a Tier-2 transform: applying it resets
an internal model to the given state, integrates it while the ramp
drives the parameters, and returns the final state. The constructor
always describes the **up** leg, which carries the reference
configuration to the target, forward in time:

.. code-block:: python

    lin_up = fr.model.transforms.AdiabaticRamping(
        model,
        ramps={"coriolis.beta": (0.0, beta)},  # (v_ref, v_target)
        ramp_period=tau,
        curve="exp",
        term_filter=fr.model.term_predicates.linear)

Each entry of ``ramps`` names a parameter and its two endpoint values;
the tuple ``(v_ref, v_target)`` is shorthand for a ``fr.Ramp`` spanning
the ramp period. Passing an explicit ``fr.Ramp`` instead is the
staggered-window form described below. The ``term_filter`` restricts
the internal model to the terms the leg should integrate, here the
linear operator.

Two accessors derive the other three legs, each returning a new
transform. ``.down`` swaps the endpoints, so :math:`\lambda` runs from
one to zero while time still runs forward; ``.backward`` retraces the
same :math:`\lambda` path with the time step reversed. The four legs
are:

.. list-table::
   :header-rows: 1
   :widths: 20 22 12 10 36

   * - Leg
     - Expression
     - :math:`\lambda`
     - :math:`\mathrm{d}t`
     - Maps
   * - up
     - ``ramp``
     - 0 → 1
     - ``+``
     - reference → target
   * - down
     - ``ramp.down``
     - 1 → 0
     - ``+``
     - target → reference
   * - up retraced
     - ``ramp.backward``
     - 1 → 0
     - ``−``
     - target → reference
   * - down retraced
     - ``ramp.down.backward``
     - 0 → 1
     - ``−``
     - reference → target

The distinction between ``.down`` and ``.backward`` is the crux of the
method, and it is easy to get wrong.

.. warning::

   ``ramp`` and ``ramp.backward`` are mutual near-inverses: the up leg
   run forward and then retraced backward returns the state to where it
   started, up to diabatic leakage and time-stepper error. ``ramp`` and
   ``ramp.down`` are **not** inverses. Both run forward in time, so a
   ``ramp.down`` after a ``ramp`` advances the mode phases by roughly
   :math:`2\tau` of dynamics rather than undoing them. The pair is a
   diagnostic (the double ramp of the example), not a round trip.

.. _adiabatic-ramping-protocols:

Two Ways to Stagger a Protocol
==============================

A useful protocol chains several legs whose ramps switch on and off at
different times. The example switches the beta effect on, then the
nonlinearity, lets the flow evolve, and switches both off in reverse
order. FRIDOM offers two surfaces for this, and they are equivalent up
to the tolerance of a stepper restart.

The default surface is **composition of legs**. Each phase is its own
transform with its own step count, and the protocol is their product,
applied right to left:

.. code-block:: python

    double_ramp = lin_up.down @ nl_up.down @ free @ nl_up @ lin_up

Composition pins each phase-inactive term statically, reports the cost
and diagnostics of every phase separately, and re-warms the multistep
stepper from first order at each boundary. It is the surface to prefer
when you want to reason about, test, or profile the phases one at a
time, which is why the example uses it.

The alternative surface is a **single leg with staggered windows**. One
``AdiabaticRamping`` carries several parameters, each an explicit
``fr.Ramp`` with its own start time :math:`t_0` inside one ``ramps``
dict. There are no stepper restarts, so the multistep history is
continuous across the whole protocol, and the leg is one compiled
region. The price is that the phase boundaries are implicit and no term
is statically pinned. Prefer the window form for long protocols where
multistep continuity matters more than per-phase inspection.

.. _adiabatic-ramping-projection:

Projecting by Ramping
=====================

On a domain where the operator has no closed-form spectral
decomposition, the eigenmode projections do not apply, and yet a slow
manifold still exists. ``AdiabaticProjection`` reaches it by
ramping. It takes a built up leg on the *linearized* model and a
reference-end projector, and composes the cycle

.. code-block:: python

    P_adiab = leg @ reference_projection @ leg.backward

which ramps the state to the reference configuration where the
projector *does* exist, projects there, and ramps back to the target.

The orientation of the two legs is not a convention but a requirement.
The away leg runs **backward** in time and the return leg **forward**,
so that whatever phase the fast and slow modes accumulate on the way to
the reference is undone on the way back. A cycle built from two
forward legs would return the projected state advanced by roughly
:math:`2\tau` of linear dynamics, because the slow modes at the target
end are not stationary (an equatorial Rossby mode has a nonzero
frequency), and it would drift further on every reapplication. The
backward-forward cycle cancels that phase exactly, up to leakage.

Because the cancellation is only up to leakage, ``P_adiab`` is a
projection only approximately: it declares itself idempotent by
contract, but ``P_adiab(P_adiab(z))`` differs from ``P_adiab(z)`` by an
amount that is exponentially small in :math:`\tau`. Validation with
``assert_idempotent`` therefore takes a leg-dependent tolerance rather
than the default machine tolerance. Each application costs two linear
integrations, reported through the transform's ``cost`` and ``repr``.

.. _adiabatic-ramping-imbalance:

Measuring the Leakage
=====================

However a state reaches the target end, whether by a double ramp or a
free run, the question is how much of it left the slow manifold. The
relative imbalance answers it:

.. math::
    \eta(z) = \frac{\lVert (I - P)\,z\rVert}{\lVert z\rVert} ,

where :math:`P` is a slow projector and :math:`I - P` its complementary
residual. In words, :math:`\eta` is the fraction of the state's norm
that the projector does not capture, so a perfectly balanced state has
:math:`\eta = 0`. The helper takes the state and the projector:

.. code-block:: python

    eta = fr.model.transforms.relative_imbalance(z, P_slow)

The default norm is the volume-weighted :math:`\ell_2` norm, which is
dimensionless and parameter-free; pass an energy metric through the
``metric`` keyword for the energy norm of a dimensional stratified run.

The imbalance of a single leg decays exponentially in the square root
of the ramp period, :math:`\eta \sim \exp(-c\sqrt{\tau})`, the
Gevrey-class rate the smooth ramp is designed to reach. Two practical
consequences follow, both visible in the example. First, the smooth
ramp only overtakes a plain linear ramp once :math:`\tau` exceeds a few
gravity-wave periods; below that the flow is in a pre-asymptotic regime
where the ramp shape barely matters, so a balancing study picks a ramp
long enough to sit in the exponential regime. Second, the scaling is
read from a *single* leg, not from a projection cycle: a
``AdiabaticProjection`` or a double ramp floors at its own reversibility
residual and stops improving with :math:`\tau`, whereas one leg,
measured against the target-end slow projector, resolves the decay down
to roundoff.

.. _adiabatic-ramping-optimal-balance:

Optimal Balance as a Special Case
=================================

Optimal balance is the deformation whose reference is the linearization
of the target: ramping the nonlinearity off recovers a linear system
with a known balanced subspace, and ramping it back on carries a
balanced state onto the nonlinear slow manifold.
``OptimalBalance`` is therefore *built on* the ramp legs rather
than reimplementing them. It owns an up leg that ramps the nonlinear
scaling from zero to its nominal value and the leg's ``.backward``
retrace, and it adds only the base-point exchange and the fixed-point
iteration that its nudging algorithm needs. The ramp cycle
``forward @ base @ backward`` is the same shape as
``AdiabaticProjection``'s cycle; the difference is that optimal balance
iterates it to a fixed point rather than applying it once.

.. _adiabatic-ramping-field-blends:

For Module Authors: Blendable Field Parameters
==============================================

The ramps above drive scalar parameters that a module already
publishes, such as ``coriolis.beta`` or the nonlinear scaling. A
field-valued parameter, such as the Coriolis profile
:math:`f(y) = f_0 + \beta y`, needs one more piece of machinery, and it
lives at the declaration layer in
``fridom.model.field_blend``. A ``FieldBlend`` declares a field as an
affine combination of assembly-materialized endpoint profiles with
stage-time scalar weights,

.. math::
    p(t) = \sum_i w_i(t)\,P_i ,

the two-endpoint case being the ingredients
:math:`\{p_\mathrm{ref},\, p_\mathrm{target} - p_\mathrm{ref}\}` with
weights :math:`\{1,\, \lambda(t)\}`. The ingredient profiles are static
auxiliary fields whose halos are exchanged once at assembly, so the
pointwise blend adds no communication, and the weights are ordinary
scalar leaves that any ``fr.Ramp`` can drive without recompilation.
This is module-author machinery: an end user ramps the scalars a module
publishes, while a module author reaches for ``FieldBlend`` when a
field parameter must itself deform. The Coriolis family is its first
consumer.

Where to Go Next
================

- The worked protocol, with figures and runtimes, is
  :doc:`/auto_examples/shallowwater/adiabatic_double_ramp`.
- The reference-end projectors it relies on are the subject of the
  eigenmode-decomposition chapter.
