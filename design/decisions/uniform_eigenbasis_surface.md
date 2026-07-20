---
status: decided
date: 2026-07-20
---

# One uniform eigenbasis surface across tiers

Owner requirement (docs review round of the barotropic-instability
example, 2026-07-20): "we need one uniform api, in the end. The user
should no longer see whether the eigenmodes are the fourier modes, or
numerically computed channel modes."

## Decision

- **One entry point per model package**: `sw.eigenbasis(model)` /
  `nh.eigenbasis(model)` / `hy.eigenbasis(model)` dispatch on the grid
  topology and return whichever engine serves it — the analytic
  operator-sourced `Eigenmodes` (fully periodic grids; nonhydro also
  the walled-vertical rigid lid) or the numeric labeled
  `ChannelEigenmodes` (exactly one bounded axis the analytic basis
  cannot serve). `eigenmodes.from_model` remains as a documented thin
  alias; its retirement rides the docs rewrite.
- **One `mode` spelling on both tiers**:
  `mode(family, indices, *, branch=None, phase=0.0)` with the labeled
  family vocabulary (`em.families`). The analytic tiers gained
  `families = {"vortical": 0, "wave+": +1, "wave-": -1}` and route
  through the shared `_resolve_mode_family` (`model/_eigenbasis.py`),
  which now also teaches a `TypeError` on integer branches. The
  contract is documented by the `ModeFamilySurface` protocol next to
  the resolver (a Protocol, not an ABC — the analytic tiers share no
  machinery with `ChannelEigenmodesBase`).
- **The integer-branch surface survives only on the analytic
  low-level accessors** `q(s)` / `omega(s)` / `projector(s)` /
  `function(f, s)` — the per-branch symbol kit consumed by the
  transforms and the distributed-matrix engine. The uniform
  projection surface for users is `sw.transforms` / `nh.transforms`.
- Downstream vocabulary migrated with it: `single_wave(...,
  family="wave+", branch=None)`, `wave_package(...)` likewise, and
  `PolarizedWaveMaker(branch=+1/-1)` (was `s=`).

## Why

The two tiers had diverged spellings for the same user intent —
`em.mode(0, k)` (analytic, integer branches) vs
`eb.mode("kelvin+", k)` (channel, labeled families) — and separate
entry points (`eigenbasis` rejecting periodic grids, pointing at
`from_model`). Selecting a mode should not require knowing which
engine the topology picked.
