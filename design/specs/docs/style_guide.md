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
| Semicolons joining prose clauses (owner ruling 2026-07-12) | Two sentences, or an explicit connective. Semicolons in code are unaffected. |

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
- **Symbols in prose are math, not code literals.** A field or
  variable referred to as a physical quantity is `:math:`u`` (or plain
  `u`), never ``` ``u`` ``` — double-backtick literal markup is for
  code identifiers and API names, and reads as clutter on a symbol
  (owner review 2026-07-22).
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
- Code follows the house style (double quotes, 79 chars). The prose
  between cells explains *why*; short inline comments above the
  nontrivial steps of a construction sequence say *what* each step
  does (owner preference 2026-07-20 — do not leave a multi-step
  block bare).

Gallery-script rules (owner review of the Phase 2 pilot, 2026-07-11):

- **Config comments follow real code.** A `# sphinx_gallery_...`
  comment placed between the docstring and the first `# %%` renders
  as an empty leading cell, and one at the tail of an rst comment
  block renders as prose; put it after at least one code line (e.g.
  between import groups), where `remove_config_comments` strips it.
- **Imports go through the root alias** (`import fridom as fr`), as
  everywhere. Importing a class directly is the exception, not the
  rule: use it only where the aliased path forces awkward wrapping
  and the direct import genuinely reads better.
- **Suppress bare object reprs** at the end of a code block
  (`_ = field.xr.plot(...)`); reprs that inform the reader (a
  `RunResult` after `model.run`) stay unsuppressed.
- **Build mechanics stay out of the prose**, and out of the code too.
  Examples ship at one resolution, the good one. The
  `FRIDOM_EXAMPLES_FAST` switch is retired (owner ruling
  2026-08-12); an example that is too slow at full resolution is
  made cheaper by choosing a smaller problem, not by branching on an
  environment variable.
- **Name tools in one working sentence** ("We use CDFViewer to render
  the vorticity animation from the zarr store"), not a capability
  pitch. §2's promotional-language ban applies to tools we ship too.

Further gallery-script rules (owner review of the revision pass,
2026-07-20):

- **Shortest public spelling.** Use the most convenient public
  surface: `fr.io.Writer` (root alias), not `fr.model.io.Writer`;
  uniform grids through `fr.spatial.cartesian.Grid(shape, extent,
  periodic=...)`, not hand-assembled `IntervalMesh` factors.
- **Descriptive state names.** Never one-letter state variables
  (`z`); name what the state is (`initial_state`, `balanced`).
- **Initial conditions in the open.** Build initial states explicitly
  in the example (sample analytic profiles with
  `grid.create_field(init=...)`, project, add modes) rather than
  calling an `initial_conditions` factory — the construction is part
  of what the page teaches.
- **Tuned coefficients get a why-comment.** A magic number (e.g. a
  biharmonic viscosity) is pulled out into a named variable carrying
  a short comment justifying its scaling; amplitudes and similar
  constants get named variables too, never inline literals.
- **Uniform eigenmode surface.** Pages spell `sw.eigenbasis(model)` /
  `nh.eigenbasis(model)` (never `from_model`) and family-string
  `mode("vortical" | "wave+" | ..., mode_number=...)`; build transform
  objects on their own line (`project_vortical =
  sw.transforms.VorticalProjection(eigenmodes)`) instead of one-line
  construct-and-call chains.

Concision rules (owner review of the barotropic-instability pass,
2026-07-22):

- **The docstring is a title plus one short sentence.** Sphinx-gallery
  shows that sentence as the card/tooltip text in the gallery, where
  only a few words fit; a multi-sentence opening paragraph is truncated
  there and reads as bloat on the page. Any longer framing belongs in
  the first `# %%` block — or nowhere.
- **Examples stay short; the guide owns the concepts.** Do not
  re-explain machinery the prose pages teach (how a `Writer` streams to
  a store, how a state is assembled, what a projection does). Repeating
  it once per example duplicates the guide across the whole gallery.
  Explain only what is specific to *this* experiment, and prefer a
  short inline comment on the line over a paragraph above the block.
  **Cutting prose never means cutting the step comments**: §7's "do not
  leave a multi-step block bare" still binds, and each code block may
  open with a one-line comment saying what it does ("project the jet
  onto the vortical subspace", "plot the initial condition").
- **No discussion residue.** Findings from the design conversation that
  produced the example — why one parameter regime was picked over
  another, how symmetric the saturated state turned out, benchmark
  numbers — are review artifacts, not reader content. They belong in
  `design/`, not in the script.

Rules from the equatorial-waves review (owner review, 2026-07-23):

- **Plain settings, library constructors.** Declare experiment
  settings as the numbers the reader should picture (a 6000 km by
  3000 km basin), not as derived geographic quantities (degrees of
  longitude times an Earth radius). And never hand-derive constants
  the API provides — `BetaPlaneCoriolis.from_latitude(0.0)`, not a
  hand-computed rotation rate and beta.
- **No overclaims in the physics prose.** When a modeling choice is
  one of convenience, say so ("simpler than adapting the analytic
  solutions to the bounded domain"); never justify it with an
  unverified impossibility claim ("has no closed form").
- **No editorial ratings.** Fame or importance framing ("the most
  famous equatorial mode") is voice, not physics; state what the
  mode does instead. And near-coincidences stay approximate in
  prose ("nearly equal frequency", not "twins at the same
  frequency") — exact-sounding claims about almost-degenerate
  quantities are overclaims too.

Rules from the wave-package review (owner review, 2026-07-24):

- **A justification appears once across the gallery.** When a
  sibling example already explains a shared choice (the internal-wave
  time step resolving N), later examples do not repeat the paragraph
  — the inline comment on the line (`# omega dt <= 0.1`) carries the
  reasoning alone.
- **Convention details take the smallest slot that holds them.** A
  formula with competing conventions (the Gaussian width) pins its
  meaning where it is defined — the docstring says "1/e radius
  150 m" — not in a prose paragraph about conventions. An equivalent
  built-in spelling (`gaussian_envelope`) is alternative syntax and
  goes in a `.. note::` callout after the code block, not into the
  running prose.

Rules from the rayleigh-taylor review (owner review, 2026-08-14):

- **The origin test.** Every sentence and every comment in an example
  has to be something a reader who never saw the review would need,
  either to follow the physics or to use the API. Prose that answers
  "why did you do it this way and not the other way" fails the test
  even when the answer is correct and interesting. The 2026-07-22 rule
  above says the same thing and did not hold, because the failure has a
  specific moment: while applying review feedback the writer's audience
  silently becomes the reviewer, and the fix documents the change that
  was just made instead of the model the reader is running. Three
  categories always come out.
  1. Defences of a choice against an alternative the example never
     shows ("the colorbar takes its width out of the figure, so the box
     aspect is set on the axes rather than through the figure aspect").
  2. Records of a tool bug or a workaround ("these come first because a
     colorrange placed ahead of them is silently discarded"). Those
     belong in §8, where they can also be retired when the tool is
     fixed. The two examples carrying that particular comment kept it
     for three weeks after CDFViewer 2026.8.1 made it false.
  3. Reassurances aimed at the reviewer ("small enough to be invisible
     in the opening frame"), which answer a question the reader never
     asked.

  What stays is what the reader can check against the page in front of
  them: a statement of what a figure shows ("Dark is the dense fluid
  resting on top"), a warning they will hit if they copy the code and
  change one thing, and a contrast that is the declared subject of the
  example.
- **Applying a review marker deletes or rewrites prose. It does not add
  prose.** If a fix leaves the file longer, the marker must have asked
  for an explanation. Otherwise the added sentences are residue. This
  one is checkable in the diff, which is why the `docs-review` skill
  carries it as a step rather than as a review criterion.
- **Route the finding, do not just delete it.** A workaround worth
  remembering goes in §8, a physics or API insight goes in the guide
  pages, and the rest goes in `design/`. The example is never the place
  of record for how the example came to be.
- **Never pass a keyword that restates the default.**
  `model.run(..., progress=False)` read as if it were silencing output,
  when the default reporter logs at debug level and prints nothing at
  the default log level. A keyword in an example tells the reader it
  matters, so pass only what changes behaviour the reader can see.

Rules from the rayleigh-taylor API review (owner review, 2026-08-14):

- **Fit the time step to the run window.** An example that ends with
  `run(runlen=...)` sets its step with
  `fr.model.fit_dt(runlen, max_dt, parts=frames)`, never by hand. The
  helper returns the largest `dt <= max_dt` that divides the window, so
  the rounding warning cannot fire, and `parts=frames` makes each frame
  a whole number of steps as well, so the samples are exactly uniform
  instead of jittering up to one step. It costs a few percent more
  steps (measured 0 to 6.7% across the gallery), which is the right
  trade. Where the window is known only after the model exists (a
  period read off an eigenmode), build with a provisional step and
  retune the bound one:
  `model.update_parameters({fr.model.params.TIME_STEP: fitted})`.
  Runs targeted with `steps=` are exact already and need none of this.
- **A bare element wherever a collection keyword is taken.**
  `outputs=writer`, `modules_extra=front`, `fields="b"`. The
  one-element tuple was ceremony, and its trailing comma is the kind of
  thing a reader copies wrong. A list or a tuple still means many.

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
- **A field plot forces its own data aspect** (owner review
  2026-08-13). xarray's `aspect=` sizes the *figure*, and the colorbar
  then takes part of that width, so a 2:1 box drawn with `aspect=2.0`
  renders at 1.73:1. Measured skews on the reviewed examples ran from
  0.86 to 0.83 before the fix. Keep a reference to the plot and force
  it on the axes:

  ```python
  plot = field.xr.isel(y=0, drop=True).plot(x="x", ...)
  plot.axes.set_aspect("equal")
  ```

- **Set `vmin`/`vmax` explicitly on a field that should read as
  one-sided.** A scheme that overshoots slightly (WENO on a sharp
  front, for instance) puts a few cells below zero, xarray then sees
  signed data and picks a symmetric diverging range, and the physical
  band is squeezed into half the colormap. This bit three reviewed
  examples.

### CDFViewer animations

Written against **CDFViewer 2026.8.3**, whose `cdfviewer` python
package is a project dependency (the docs CI fetches the viewer bundle
of the package version in `uv.lock`). An example records through the
package, never through a shell line:

```python
import cdfviewer as cv

_ = cv.record(
    "store.zarr", var="b", x="x", y="z", dims={"y": 0},
    plot_type="heatmap", ani_dim="time",
    kwargs={"colormap": "balance", "colorrange": (-1, 1),
            "title": "Buoyancy"},
    filename="store.mp4", framerate=24)
```

The parameters are named after the CLI options (`var` for `-v`,
`ani_dim` for `-a`, `dims` for `--dims`, `kwargs` for `--kwargs`,
`filename`/`framerate` for `-s`), so the viewer manual's command lines
translate a word at a time. Python values are written in the viewer's
keyword syntax by the package: a colormap or colour is the plain string
`"balance"`, tuples stay tuples, `cv.sym("fitzoom")` is for the few
keywords that insist on a symbol, and `cv.raw(...)` passes Julia source
through. Keep `_ =` in front of the call, as in front of the matplotlib
stills: it returns the path of the video, and a bare call would print it
into the rendered page. Two hazards of the 2026.7 line are fixed since
2026.8.1 and no longer constrain the examples: an expression-valued
keyword no longer arrives as a string, and `colorrange` is no longer
discarded when the size keywords are present. Keyword order is free.

- **Label the colorbar with `cbarlabel="auto"`.** It reads the field's
  own `long_name`, so the caption cannot drift from the model, and a
  literal string is a second place to keep the name correct. Plain
  `label=` reaches Makie's colorbar too but has no automatic mode.
- **Size the figure to the data.** The viewer's fixed chrome is 165 by
  120 pixels, so a square domain wants `figsize=(W, W - 45)`. Leaving
  the default 800 by 600 on a square field letterboxes it, measured at
  47% of the frame against 66% when sized, with the colorbar stranded
  110 pixels from the plot instead of 20. Nothing else reaches that
  gap, since `colgap`, `figure_padding`, `colorbargap` and
  `colorbarwidth` are all rejected.
- **Overlay a vector field with `over=["u,v"], over_plot=["quiver"]`**
  (or `"streamplot"`, `"contour"`, `"contourf"`, `"heatmap"`). The
  store must carry the components, so the writer needs them in
  `derived=`. Draw them in a flat colour with the `kwargs` entry
  `"over.color": "black"`, since by default
  they are coloured by speed and fight the field underneath. `arrows=`
  takes a target count per axis and is resolution independent; the
  default `(24, 16)` is anisotropic and wrong on a square domain.
  `minspeed=` hides slow arrows, which helps only where there is a
  genuinely quiescent region to clear.
- **An overlay changes the title** to a composite such as
  `Buoyancy / |(u, v)|`, so set `title=` explicitly whenever one is
  present.
- **A second overlay is `over2`** (`over=["v,w", "b_total"],
  over_plot=["quiver", "contour"]`), its keywords prefixed the same
  way (`"over2.levels": levels.tolist(), "over2.color": "white"`). A
  contour overlay takes a level count or an explicit list (the list
  needs CDFViewer 2026.8.4 or later); pass the list the stills use so
  the animation draws the same isopycnals.
- **Quiver arrows are sized and oriented on screen** (CDFViewer
  2026.8.4 or later). The reference speed of a frame spans nine
  tenths of the pixel gap between neighbouring arrows, and every
  arrow points where its vector points. For an animation prefer a
  fixed `"over.lengthscale"` (pixels per unit speed, read the speeds
  off the store), so the arrows grow with the forcing instead of
  being refitted to every frame; `"over.minspeed"` (data units)
  blanks a quiescent interior that would otherwise draw as dots. The
  arrow shape passes through to Makie (`"over.shaftwidth": 2,
  "over.tipwidth": 8, "over.tiplength": 8` reads well at 1200 px
  wide). The sample stride is an index stride, so on a stretched mesh
  the arrows crowd where the cells are fine.
- **On a section the vertical velocity is orders of magnitude below
  the horizontal one**, so true-direction arrows come out flat
  everywhere and the overturning cell does not show. Record the
  vertical component scaled by the ratio of the physical aspect to
  the drawn one (`derived={"w_drawn": lambda ms: ms.state["w"] * (LY
  / LZ / ASPECT)}` with `"aspect": ASPECT` in the kwargs), which draws
  the arrows tangent to the streamlines as the figure shows them, and
  say so in one sentence of the prose (`coastal_upwelling.py`).
- **Several videos from one code block** render side by side in a
  grid of up to three columns. Set `# sphinx_gallery_video_columns = N`
  in the block to change that, `1` stacking them at full width (the
  right choice for wide domains). The comment follows a code line
  like every config comment and is stripped from the page; one placed
  elsewhere in the file applies to every block without its own. Do
  not split videos over cells only to stack them.

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

**The humanizer pass is mandatory** (owner ruling 2026-07-23). Every
prose block an author writes or edits — rst prose, gallery `# %%`
blocks, docstrings — goes through the `humanizer` skill before the
content is handed to the owner for review, and again after any rewrite
made while applying his feedback. Its hard constraints are no em or en
dashes, no colons or semicolons in running prose, no curly quotes, and
none of the AI-tell vocabulary. Docs prose is technical register, so
the skill's "add personality" section does not apply: plain and
neutral is the correct human voice here, and no first person or
opinions enter the text. The mechanics live in the `docs-review`
skill, step 2.
