---
status: normative
date: 2026-07-11
---

# Documentation style guide

Binding for all prose, code, figures, and diagrams under `docs/` and
`examples/`. Companion to [`structure.md`](structure.md) (the page
tree) and
[`../../plans/active/docs_examples_plan.md`](../../plans/active/docs_examples_plan.md)
(the build program). Once approved, this file graduates to
`docs/STYLE.md` so every writer, human or agent, finds it next to the
content it governs.

Rules marked **[lint]** are enforced by the docs-lint script
(Phase 0.5); the rest are review criteria. A rule that fights the
Phase 2 pilot gets rewritten here, not silently ignored.

## 1. Voice

- **Scholarly "we" for exposition, "you" for instructions.** We use
  "we" when demonstrating or reasoning ("We now add friction to the
  model"), and "you" when the reader must act ("You can override the
  default with ..."). Never "I", never the imperative-only voice of a
  changelog.
- **Present tense** for how the code behaves and for established
  facts. Past tense only for prior literature ("Arakawa and Lamb
  proposed ...").
- **Every paragraph leads with a topic sentence** stating its point;
  the rest supports it. If a paragraph has two points, it is two
  paragraphs.
- **Connect sentences with explicit logical connectives** (However,
  Therefore, In contrast, Consequently, Moreover). The argument's
  skeleton must be visible from the connectives alone.
- **Define terms deliberately.** Introduce a new term with emphasis
  at first use ("the *halo region*"), link it to the glossary, and
  introduce every acronym as full name (ACRONYM). When a term is
  overloaded, disambiguate proactively ("we use *staggered* in the
  Arakawa sense, not ...").
- **Equation, then explanation.** After every display equation, define
  the symbols in "where ..." prose. After a dense equation or
  definition, add a plain-language gloss ("In simple terms, ...").
  Never assume mathematical vocabulary: the first use of a term like
  *skew-adjoint* gets one sentence saying what it means and why it
  matters here.
- **Calibrate claims.** Hedge what is uncertain ("appears to",
  "suggests"), state verified behavior plainly but bounded ("second
  order in the interior; at walls the order drops to one"), and name
  limitations honestly with their cause. Never oversell the framework;
  state what a feature does and where it stops.
- **Section openers forecast structure** when a section is long ("We
  start by ..., then ..., and finally ..."). "Let's ..." is allowed
  and in-voice (owner ruling 2026-07-11; his own tutorials use it);
  vary it with "We start by ..." rather than opening every section
  the same way.
- **Skip guidance for heavy material.** A derivation-heavy section
  opens by saying what can be skipped: "The derivation below is not
  needed to use the API; readers can skip to Section X."

## 2. Banned patterns [lint]

These are the patterns that make prose read machine-written. The lint
script rejects them outside code blocks and directives.

| Banned | Use instead |
|---|---|
| Em dashes (and spaced hyphens as dashes) | Commas, parentheses, or two sentences. En dash only in numeric ranges (2--4). |
| Emoji, exclamation marks | Plain statements. |
| "not X, but Y" / "no X, no Y" marketing contrast | State what it is. Scholarly contrast ("However, ...", "In contrast to ...") is encouraged and not affected. |
| Rule-of-three cadence ("fast, flexible, and friendly") | Say the one thing that matters, or list properly. |
| Bold-term feature bullets ("**Zero config:** it just works") | Prose sentences, or a real table. |
| Self-praising adjectives: powerful, seamless, robust, comprehensive, blazing, effortless, elegant, "clear error" | Describe the behavior; let the reader judge. |
| Vocabulary: leverage, delve, utilize, showcase, crucial, "it's worth noting", "keep in mind" | use, examine, use, show, important, (delete), (delete) |
| Minimizers "simply", "just", "easily" | Delete. If it were simple the sentence would not need the word. |

Weak/strong pairs:

- *Weak:* "FRIDOM's powerful operator algebra lets you seamlessly
  compose operators — no boilerplate required."
  *Strong:* "Operators compose with `@`. The composition is itself an
  operator and can be applied, transposed, or differentiated like any
  other."
- *Weak:* "Creating a grid is easy! Simply call `Grid(...)`."
  *Strong:* "Let's create a grid. `Grid` takes the mesh and the
  periodicity of each axis:"

## 3. Page anatomy

Every guide and advanced page follows this shape (reference pages and
gallery examples are exempt where noted in `structure.md`):

1. **Goal statement**, one or two sentences: what the page covers and
   what the reader can do afterwards.
2. **Prerequisites line** linking the pages this one builds on.
3. **Sections**, each opening with a motivation sentence before any
   code. Why does this concept exist; what problem does it solve.
4. **Code sandwich:** every code block is introduced by a sentence
   ending in a colon and interpreted by a sentence or paragraph after.
   Console output is shown verbatim (in a collapsed dropdown when it
   is long) and walked through, line by line where it is not obvious.
5. **Summary and hand-off:** a short recap plus "where to go next"
   links.

Target length is 100 to 250 lines of source per page. A page that
outgrows this is two pages.

## 4. Headings, names, labels

- Page titles and section headings in **Title Case**; question form is
  allowed for motivation sections ("Why Are Fields Staggered?").
- Files `snake_case.md`/`.py`, matching the page title.
- Cross-reference labels are stable and namespaced:
  `.. _<page>-<topic>:` (e.g. `operators-transposition`). Equation
  labels `eq:<page>-<name>`. Renaming a label requires a grep for its
  users. [lint: labels unique]

## 5. Admonitions (closed vocabulary)

Exactly these, and rarely [lint: vocabulary only]:

| Directive | For | Budget |
|---|---|---|
| `note` | Edge behavior, constraints | ~2 per page combined |
| `tip` | A genuine shortcut | (with note/warning) |
| `warning` | Footguns, silent wrong answers, data loss | always allowed |
| `dropdown` (sphinx-design) | Long verbatim output, optional derivations | as needed |
| "Going deeper" (`admonition` with class `going-deeper`) | The guide's hooks into Advanced topics: two sentences naming the deeper concept and linking its chapter | 1 to 3 per guide page |

Admonitions never carry load-bearing content; a reader who skips all
of them must still be able to follow the page.

## 6. Equations and notation

- Display math via `.. math::`; multi-line with `aligned`. No `\tag`
  inside `aligned` (renderer quirk).
- Vectors `\boldsymbol{u}`; upright `\mathrm{d}` in integrals;
  operators and named quantities upright (`\mathrm{Ro}`).
- Number an equation only if the text references it; refer to it as
  "Eq. (n)" via `:eq:`.
- **The notation page is the single source of symbols.** A chapter
  introducing a symbol that already exists on the notation page uses
  that symbol; a chapter needing a new symbol adds it there in the
  same change. Conflicting reuse (same symbol, different meaning) is a
  review blocker.

## 7. Code examples

- **All code executes at build time**: gallery scripts for pages that
  produce figures, `sphinx.ext.doctest` for inline snippets on prose
  pages. No dead `code-block` snippets in the guide; reference pages
  may use `code-block` for signatures and shell commands.
- Imports use the root alias: `import fridom as fr`, models as
  `import fridom.nonhydro2 as nh` / `import fridom.shallowwater2 as sw`
  (mechanically renamed at cutover).
- Snippets are complete and runnable as shown, seeded where random,
  and sized to the doc budget (a page's total compute ≤ ~2 min on a
  CI runner, per the build plan).
- Individual blocks stay under ~20 lines; longer listings are split
  and interleaved with prose.
- Code follows the house style (double quotes, 79 chars); comments in
  snippets are rare because the surrounding prose does that job.

Gallery-script rules (owner review of the Phase 2 pilot, 2026-07-11):

- **Config comments follow real code.** A `# sphinx_gallery_...`
  comment placed between the docstring and the first `# %%` renders
  as an empty leading cell, and one at the tail of an rst comment
  block renders as prose; put it after at least one code line (e.g.
  between import groups), where `remove_config_comments` strips it.
- **Imports go through the root alias** (`import fridom as fr`), as
  everywhere. Importing a class directly is the exception, not the
  rule: use it only where the aliased path forces awkward wrapping
  and the direct import genuinely reads better (e.g. `IntervalMesh`
  constructor lines).
- **Suppress bare object reprs** at the end of a code block
  (`_ = field.xr.plot(...)`); reprs that inform the reader (a
  `RunResult` after `model.run`) stay unsuppressed.
- **Build mechanics stay out of the prose.** The
  `FRIDOM_EXAMPLES_FAST` resolution switch appears as plain code in
  the settings block, without a paragraph explaining CI; the
  convention is documented in the build plan, not to readers.
- **Name tools in one working sentence** ("We use CDFViewer to render
  the vorticity animation from the zarr store"), not a capability
  pitch. §2's promotional-language ban applies to tools we ship too.

## 8. Figures (plots)

- Every figure is produced at build time by the executing page, styled
  by the shared style file `docs/source/fridom_docs.mplstyle` plus the
  named palette (Phase 0.5 deliverable). **Individual pages never
  restyle**; a figure that genuinely needs a deviation documents why
  in the script.
- **Plot through xarray wherever possible** (owner ruling 2026-07-11):
  `field.xr.plot(...)` with xarray's defaults, themed globally by the
  shared style file. This keeps snippets short and leaves little
  plotting code to maintain. Hand-rolled matplotlib appears only where
  xarray cannot draw the figure (e.g. quiver overlays, custom
  multi-panel layouts), and then still under the shared style.
- Palette requirements: colorblind-safe categorical set shared with
  the SVG diagrams; sequential colormap `cmocean`-style or viridis
  family; diverging colormap centered on zero (`RdBu_r` or cmocean
  `balance`) for signed fields.
- Axes are labeled with quantity and unit; titles state what is shown,
  captions how to read it (bold lead phrase, then content, per-panel
  for multi-panel figures).
- Vector output (svg) for line plots and diagrams; png only where
  rasterization is unavoidable (dense pcolormesh).

## 9. Hand-drawn diagrams (SVG)

- Sources live in `docs/source/_static/diagrams/`, editable in
  Inkscape, using the shared palette tokens and one stroke width
  scale. Text and math labels are generated via **typst → svg** and
  placed into the diagram (owner preference, 2026-07-11), rather than
  written as plain Inkscape text, so diagram typography matches the
  docs' math.
- **Agents may draft new diagrams**: propose an svg following these
  conventions; the owner fine-tunes it in Inkscape. Draft close to the
  conventions so the fine-tuning stays small.
- Diagrams stay **concept-level** (staggering, halos, mode structure),
  never API-level (no method names in diagrams), because they are the
  only artifacts the build cannot re-verify.
- `diagrams/INVENTORY.md` lists every diagram with one line stating
  what change would invalidate it. Reviewed when the named concept
  changes.

## 10. Linking

- **Link the first mention** of any concept that has its own page or
  glossary entry; later mentions on the same page stay unlinked.
- **Prefer linking over duplicating**: a derivation or definition
  lives on exactly one page; everyone else links it.
- Glossary terms via `:term:`; pages via `:doc:`; sections via
  `:ref:` with §4 labels. Every page appears in exactly one toctree.
- Guide pages end with "where to go next"; advanced pages open with
  prerequisites and close with related chapters.

## 11. Citations and references

Structurally hallucination-proof, not merely discouraged:

- All citations go through `sphinxcontrib-bibtex`: prose may only cite
  keys that exist in `docs/source/references.bib`
  (`:cite:t:` for author-as-subject, `:cite:p:` parenthetical).
- **A key enters the bib only through verification**: resolve the DOI
  (or, for the rare paperless source, the canonical URL), compare
  title/authors/year against the claim, and record the check as a
  `% verified: <date> via doi` comment on the entry. Peer-reviewed
  papers are preferred over repository links; cite the paper for the
  method and the repository only for the implementation.
- A writer (human or agent) who cannot verify a wanted citation writes
  `.. todo:: cite <what>` instead. Free-typed references in prose are
  a review blocker. [lint: `:cite:` keys ⊆ bib, no bare "et al." in
  prose]
- The References page renders the full bibliography; individual pages
  do not carry local bibliographies.

## 12. Enforcement

Phase 0.5 ships `docs/lint_docs.py` (name tentative), run in CI on
changed files next to `sphinx-build -W`, linkcheck, and doctest. It
checks, at minimum: the §2 banned patterns, §4 label uniqueness and
scheme, §5 admonition vocabulary, §11 citation-key existence, em
dashes outside literals, emoji anywhere, and **no open `REVIEW:`
markers** (the owner's in-file review comments; an unaddressed marker
blocks the merge — see the review workflow in AGENTS.md). Every new
renderer quirk or style violation we hit once becomes a check.
