---
status: done
date: 2026-07-11
---

# Plan: split `framework2` into `fridom.spatial` + `fridom.model`

Status: **done — merged into dev 2026-07-11 (639a935), full suite 7315 green** (names, split line, and import
idiom all signed off). Execution pending.

## Decision (settled — do not re-litigate)

`fridom.framework2` dissolves into two top-level packages:

- **`fridom.spatial`** — everything about discretizing space: meshes,
  function spaces, fields, operators, domain decomposition, and the
  `Grid` assembly root.
- **`fridom.model`** — everything about running a model through time:
  the model core (assembly, run loop, schedule), tendency modules,
  time steppers, state transforms, io, ops.

User-facing idiom is the **root alias**: `import fridom as fr`, with
lazypimp exposing the subpackages lazily (`fr.spatial.Grid`,
`fr.model.Model`, `fr.nonhydro2.Model`). No `fridom.framework` façade,
no per-package two-letter aliases. Rejected names (`core`,
`framework`, `numerics`, `calculus`, `discretization`, `domain`) are
settled; "spatial vs model" names the actual dividing line (spatial
discretization vs temporal/orchestration).

The split does **not** wait for the old-stack cutover: `spatial` and
`model` collide with nothing. `nonhydro2`/`shallowwater2` keep their
"2" suffix until the old stack is deleted (ROADMAP), then drop it.

## Package mapping

### `fridom.spatial` (from `framework2/grid/`)

| From `framework2/grid/` | To `fridom/spatial/` |
|---|---|
| `meshes/` | `meshes/` |
| `spaces/` | `spaces/` |
| `fields/` | `fields/` |
| `operators/` | `operators/` (promoted out of grid — the shared numerics library) |
| `decomposition/` | `decomposition/` |
| `grid.py`, `bc.py`, `symbols.py`, `random_fields.py`, `export.py`, `immersed_domain.py`, `interning.py`, `scalars.py`, `errors.py`, `coordinate_mapping.py`, `discretize.py` | top-level modules in `spatial/` |
| `cartesian/` | `cartesian/` |

Plus, moved **down** from `framework2/model/`:

- `space_patterns.py` → `spatial/space_patterns.py`. This kills the
  one layering inversion (`grid/grid.py` runtime-imports
  `model.space_patterns.Dof` upward today). After the move, `model`
  imports it downward — the correct direction.

### `fridom.model` (from `framework2/model/` + siblings)

| From `framework2/` | To `fridom/model/` |
|---|---|
| `model/*.py` (model, module, assembly, composer, schedule, stages, context, clock, declarations, field_table, parameters, params, roles, terms, term_predicates, time_dependent, implicit, energy, eigen*, results, report, errors) | top-level modules in `model/` |
| `model/closures/` | `closures/` |
| `model/time_steppers/` | `time_steppers/` |
| `transforms/` | `transforms/` |
| `io/` | `io/` |
| `ops/` | `ops/` |
| `modules/` | `modules/` |

### Physics moved out of the framework

`model/energy.py` currently hard-codes both concrete models' energy
weights. `EnergyMetric` (M) stays in `fridom.model`;
`nonhydro_energy_weights` moves to `nonhydro2`,
`shallowwater_energy_weights` + `shallowwater_varying_energy_weights`
move to `shallowwater2`.

### Root namespace

`src/fridom/__init__.py` gains lazy entries for `spatial` and `model`
(alongside the existing old-stack and `*2` entries). The `framework2`
name disappears in this change — every internal import rewrites from
`fridom.framework2.grid.X` → `fridom.spatial.X` and
`fridom.framework2.{model,transforms,io,ops,modules}.X` →
`fridom.model.….X`. `nonhydro2`/`shallowwater2` import the new paths.

## Test-tree alignment

Tests move to mirror `src/` exactly (fixes the AGENTS.md
mirrored-test-policy mismatch where model packages nest under
`tests/framework2/`):

- `tests/framework2/grid/**` → `tests/spatial/**` (operator tests to
  `tests/spatial/operators/`, etc.)
- `tests/framework2/{model,transforms,io,ops,modules}/**` →
  `tests/model/**`
- `tests/framework2/nonhydro2/**` → `tests/nonhydro2/`
- `tests/framework2/shallowwater2/**` → `tests/shallowwater2/`
- `tests/framework2/validation/**` → `tests/validation/` —
  cross-cutting end-to-end PDE suite; documented exception to
  mirroring (it mirrors no single source package).
- `tests/framework2/conftest.py` content folds into the relevant new
  roots; `benchmarks/framework2/` → `benchmarks/` follows the same
  rename late in the sequence.

## Execution

One branch per the AGENTS.md workflow: `refactor/spatial-model-split`
in its own worktree. The change is almost entirely mechanical
(`git mv` + scripted import rewrite + lazypimp `__init__` table
updates); the only semantic edits are the `space_patterns` move, the
energy-weights move, and the two `__init__.py` re-export tables for
the new packages.

Stages (each a commit on the branch):

1. `git mv` the source trees into `spatial/` + `model/`; rewrite all
   `fridom.framework2.*` imports (script; includes docstrings and
   design-doc code snippets only where they are import statements).
2. `space_patterns` down-move + delete the upward import in
   `grid.py`; energy weights out to the model packages.
3. New `spatial/__init__.py` + `model/__init__.py` lazypimp tables
   (surface = union of today's `framework2` + `framework2.grid`
   tables, split by package); root `fridom/__init__.py` entries.
4. Test tree moves + conftest folds.
5. AGENTS.md project-layout + testing-policy path updates;
   `design/README.md` note that specs' `framework2.grid` vocabulary
   now reads `fridom.spatial` (full spec prose rename deferred — the
   specs stay historically coherent; the map note is enough).

Merge gate: **full** `tests/` framework2-successor suite
(`uv run pytest tests/ -n 8 --dist loadfile`, old-stack dirs included
— they must stay green through the rename) + `uv run ruff check src
tests`. This change touches every file, so the full-suite exception
applies. Then `merge --no-ff`, delete branch + worktree.

Sequencing: land this split **before** the boundary-decision salvage
work (`BC.ROBIN`, one-sided rows), so that work lands once, on the
new paths. The pending `framework2-boundaries` decision is
independent — pure renames rebase trivially compared to semantic
conflicts.

## Open (not blocking)

- Internal grouping of `spatial/operators/` (28 flat modules →
  `core/stencil/fv/spectral/pointwise` subpackages). Invisible to
  users behind the lazypimp surface; can be a follow-up
  `refactor/operators-grouping` if wanted.
- Stub modules (`galerkin.py`, `sphere.py`, `combinators.py`, …) move
  along with their homes; whether to keep or drop them is a separate
  cleanup call.
