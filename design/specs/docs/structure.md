---
status: draft
date: 2026-07-11
---

# Documentation structure

The page tree for the rebuilt docs: every page with its scope, its
authoring format, and its non-goals. Companion to
[`style_guide.md`](style_guide.md) (how pages are written) and
[`../../plans/active/docs_examples_plan.md`](../../plans/active/docs_examples_plan.md)
(how they are built and shipped). Scopes are binding for parallel
writers; changing a page's scope means changing this file first.

Authoring formats:

- **gallery** — a sphinx-gallery script under `examples/`; code,
  figures, and videos execute at build time.
- **doctest** — an rst/MyST page whose snippets run under
  `sphinx.ext.doctest`; for prose- and math-heavy pages with little or
  no plotting.
- **static** — no executed code (reference tables, shell commands,
  bibliography).

The rule of thumb: a page that produces a figure is a gallery script;
a page that mostly explains is a doctest page.

## Top level

```
Home                      static    orientation, hero video, annotated outline
Installation              static    escalating paths: pip → cuda → source → cluster
Getting Started           gallery   one complete small model, walked through
The FRIDOM Guide          9 chapters, read in order
Advanced Topics           self-contained chapters, extensible
Models                    per-model physics + parameters + examples
Verification              standard test cases, executed
Gallery                   auto-generated from examples/
Benchmarks                static    measured performance numbers
API Reference             autosummary (lazypimp pipeline)
References                static    full bibliography
Glossary & Notation       static    terms and symbols, single source
```

## Home, Installation, Getting Started

- **Home**: one-paragraph statement of what FRIDOM is and who it is
  for, the hero video, and an annotated outline linking every top
  section with one line each. No feature bullets.
- **Installation**: easiest path first, each alternative labeled with
  its trade-off; GPU extra, editable source install, cluster module
  notes (Levante recipe carried over). Ends by pointing to Getting
  Started.
- **Getting Started**: a complete shallow-water run in roughly twenty
  lines, then the same script walked through step by step. The reader
  leaves having run a model and knowing which guide chapter explains
  each step. Non-goal: teaching concepts; it links instead.

## The FRIDOM Guide (the storyline)

Read in order; each chapter assumes only its predecessors. 1D or
small-2D throughout; every chapter carries "going deeper" hooks into
Advanced Topics (style guide §5). Chapters and scopes:

| # | Chapter | Format | Scope (and non-goals) |
|---|---------|--------|-----------------------|
| 1 | Meshes and Grids | gallery | Creating a mesh and a `Grid`; axes, periodicity, spacing; what the Grid assembles. *Not:* tensor product spaces, decomposition. |
| 2 | Fields and Function Spaces | gallery | Fields live on function spaces; nodal values vs. what a field *is*; staggered positions; arithmetic; plotting via `.xr`. 1D examples. *Not:* the full space algebra, coefficient/Galerkin spaces. |
| 3 | Operators | gallery | Applying operators (diff, interpolate); operators map between spaces; composition with `@`. *Not:* the algebra rules, symbols, adjoints. |
| 4 | State and Initial Conditions | gallery | The State vector; prognostic vs. diagnostic; building initial conditions from fields; the IC library. |
| 5 | Assembling and Running a Model | gallery | ModelSettings, `Model`, `run()` variants (length, steps, date); inspecting results. |
| 6 | Tendencies and the Module System | gallery | What a module is; the tendency schedule; enabling/disabling; writing a small custom module (e.g. linear drag). *Not:* time-stepper internals, purity rules in depth. |
| 7 | Output and Visualization | gallery | `fr.io.Writer` → zarr; reading with xarray; the CDFViewer one-liner for animation; snapshots and triggers. |
| 8 | Multiple Devices and Clusters | gallery + static | Sharded runs: `jax.distributed.initialize`, device meshes, what changes in user code (little). Executed portion demonstrates sharding on forced host devices (`XLA_FLAGS=--xla_force_host_platform_device_count=4`), so the page cannot go stale; SLURM job scripts are static blocks (carried over from the old cluster tutorial, incl. restart pattern). *Not:* decomposition internals (Advanced). |
| 9 | One Model, Three Ways | gallery | The same shallow-water setup three times: minimal script (fewest readable lines), configured script (explicit settings, comments on every choice), and a small modular package (multiple files, custom module, run script). Bridges into Advanced Topics and Models. Built once, for shallow water; nonhydro links here. |

## Advanced Topics

Self-contained chapters, one concept each; explicitly **extensible**,
the list below is the opening set, not a cap. Conventions: each
chapter opens with prerequisites and skip guidance, explains its
mathematics from first principles (style guide §1), and closes with
related chapters. Mostly doctest format; gallery where figures teach.

Initial set (v1 candidates marked *):

- Meshes in Depth — mesh types, spacing functions, what a mesh
  guarantees to spaces built on it.
- Function Spaces and Tensor Product Spaces — the space algebra,
  1D factors, products, coefficient spaces.
- Boundary Conditions* — wall semantics, one-sided rows, Robin/mixed;
  the "skew-adjoint" class of explanations lives here.
- The Operator Algebra* — composition, sums, scaling, blocks,
  transposition; the laws and why they hold.
- Symbols and Spectral Analysis — operator symbols, dispersion
  relations, solving linear systems spectrally.
- Eigenmode Decomposition* — the three tiers (periodic, trig-wall,
  channel), degeneracy resolution, energy metric.
- Projections and Balancing — state transforms, vortical/wave
  projections, optimal balance, NNMD (as it lands).
- Time Steppers — RK/AB/IMEX families, the schedule, writing one.
- Performance and JAX* — jit, x64, backends, GPU memory, recompile
  pitfalls; benchmarking your own runs.
- Domain Decomposition — how sharding works underneath chapter 8.

## Models

One chapter per model, same skeleton: continuous equations and
assumptions; discretization summary (link to Advanced for depth);
parameter reference (table generated or checked against code where
possible); the model's examples embedded from `examples/` via
sphinx-gallery backreferences; pointers to its verification cases.

- Nonhydrostatic model
- Shallow water model

Examples remain single-source in `examples/`: the model chapter embeds
and orders them, the Gallery is generated from the same scripts, and
nobody maintains two lists.

## Verification

Standard test cases demonstrating correctness, each an executed
gallery script: setup, the expected behavior with its literature
reference, the measured result at documentation resolution, and the
convergence figure where applicable. Organized by what a case
verifies, which resolves the "model-level vs. module-level" split:

- **Component cases** — e.g. advection-scheme convergence orders,
  operator accuracy at boundaries, eigenmode orthogonality/energy
  identities. Linked from the corresponding Advanced chapters.
- **Model cases** — e.g. Taylor-Green vortex (nonhydro), geostrophic
  adjustment, equatorial wave dispersion (shallow water). Linked from
  the Models chapters.

Execution policy: cases that fit the per-page budget run on every full
build; heavier cases run on the weekly cron only (gallery
`filename_pattern`), so divergence is caught within a week without
slowing every build. Verification scripts double as long-horizon
regression tests; a failing weekly build is a real signal.

## Benchmarks

Performance numbers (wall time, scaling, GPU vs. CPU) are **not**
executed at doc build time: shared CI runners produce noise, not
measurements, and have no GPU. The Benchmarks page is static content
fed by dedicated runs of the `benchmarks/` suite, each table stamped
with machine, backend, versions, and date. Update cadence: on release,
or when a change moves a number materially.

## Gallery, API Reference, References, Glossary

- **Gallery**: sphinx-gallery index over `examples/` (thumbnails,
  videos); no hand-written content beyond section headers.
- **API Reference**: the lazypimp-aware autosummary pipeline
  (`load_modules.py`, custom Jinja filters, `_templates/autosummary/`)
  adapted to `fridom.spatial` / `fridom.model` / the model packages;
  per-class minigalleries via backreferences stay.
- **References**: the rendered `references.bib` (style guide §11).
- **Glossary & Notation**: `.. glossary::` for terms (halo, staggered,
  skew-adjoint, ...) linked via `:term:`; a notation table for symbols
  (single source, style guide §6).

## v1 scope and stub policy

v1 ships: Home, Installation, Getting Started, Guide 1–9, the four
starred Advanced chapters, both Models chapters, Verification with one
component case and one model case, Gallery, API Reference, References,
Glossary. Remaining Advanced chapters and verification cases follow
incrementally.

**No stub pages.** A chapter that is not written does not appear in
any toctree; the backlog lives in the build plan, not in the reader's
navigation. (The old docs' TODO stubs are the failure mode this rule
prevents.)
