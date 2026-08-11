---
status: decided + implemented
date: 2026-08-11
---

# The time-law family is aliased at the package root

**Status: decision + implementation record, 2026-08-11.** Owner ruling
(Silvano, in chat, from the `internal_wave_maker` docs review):
`TimeDependent`, `Ramp`, `TimeFunction`, `TimeSeries` and `Harmonic`
re-export from `fridom` itself, so `fr.Ramp` and `fr.Harmonic` are the
blessed spellings. The `fr.model.*` spellings keep working unchanged —
these are aliases, not a move. Implemented on
`feat/time-law-root-aliases`.

## 1. Why

The package already documented the aliases it did not have. `fr.Ramp`
appears in roughly twenty docstrings across `fridom.model` —
`parameters.py`, `field_blend.py`, `module.py`, `errors.py`,
`closures/diffusion.py`, `modules/ramping.py`, and `Harmonic`'s own
class docstring — while the only importable spelling was
`fr.model.Ramp`. A reader who copied a documented type line
(`kappa : float | fr.Ramp | Mapping[str, float]`) got an
`AttributeError`. Aliasing at the root is the smaller correction: it
makes the documented surface real, rather than editing twenty
docstrings to a longer spelling nobody wanted to type.

The time laws are also the right shape for a root alias. They are
user-facing knobs written at the call site of every forcing, closure
and ramped parameter, they carry no model-package flavour (unlike
`nh.*` / `sw.*`), and they are leaf value types rather than machinery.

## 2. What changed

`src/fridom/__init__.py` gains its first entries in
`all_imports_by_origin` (previously empty, root exported subpackages
only), pointing at `fridom.model.time_dependent`. The root
`test_init.py` parametrizes over that mapping, so the five names are
covered by the existing re-export test with no new test needed.

## 3. Scope

Root aliases stay rare and are not a general licence to flatten the
namespace. Model machinery (`fr.model.modules.Source`,
`fr.model.time_steppers.*`), spatial machinery (`fr.spatial.*`) and the
model packages keep their qualified spellings; the criterion met here
is a value type that is both model-agnostic and written constantly at
call sites.
