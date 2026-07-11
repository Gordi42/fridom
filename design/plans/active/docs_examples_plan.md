---
status: active
date: 2026-07-11
---

# Docs & examples rebuild plan

Owner-approved 2026-07-11. Replaces the deferred "Examples refresh" /
"Docs refresh" items in `cutover_parity_plan.md` with a concrete program.
The docs are rebuilt from scratch for the new stack; examples and code
snippets **execute at doc build time** so they cannot diverge from the
code.

## Settled decisions

- **Build system:** keep sphinx-gallery, upgrade to ≥ 0.17
  (`matplotlib_animations`, `parallel`, per-example `.md5` incremental
  skip). Examples actually run at build time — `@skip_on_doc_build`
  and the pre-rendered-media machinery (`custom_scraper.copy_media_files`,
  git-LFS videos) are retired. Low resolution is the accepted price.
- **Hosting:** GitHub Actions → GitHub Pages. RTD Community cannot do
  executed examples (15 min/build, no cache between builds, no Julia).
  Keep a thin RTD project (execution off via `SPHINX_QUICK_BUILD`)
  purely for free PR previews of prose/API changes.
- **Stills:** `f.xr.plot(...)` in the example; the stock sphinx-gallery
  matplotlib scraper picks the figures up.
- **Animations:** example writes zarr via `fr.io.Writer`; CDFViewer
  renders the mp4 (`cdfviewer <store> ... --record`), embedded with
  `sphinxcontrib-video`. The cdfviewer command is **visible in the
  example** as a one-liner — it documents the real workflow and
  showcases CDFViewer. Videos are rebuilt each full build and never
  committed.
- **CDFViewer in CI:** use the prebuilt Linux binaries from CDFViewer's
  GitHub releases (being added, 2026-07-11). Until they exist:
  `julia-actions/setup-julia` + `julia-actions/cache` + pinned CDFViewer
  SHA (~3–5 min warm, ~19 min on cache miss; xvfb + Mesa software GL,
  copy the recipe from CDFViewer's own CI).
- **No hold on the swap:** examples/docs are written against the current
  names (`fridom.nonhydro2` etc.) now; the Wave-C rename
  (`nonhydro2 → nonhydro`, `framework2 → framework`) is a mechanical
  find/replace pass over `examples/` and `docs/`.

## Review workflow (owner-mandated, 2026-07-11; made private same day)

All reader-facing content (`docs/` pages, `examples/` scripts) is
**100% owner-reviewed before it reaches dev**, and the review happens
**off GitHub**: the repo is public, and review discussion must not be
(this superseded a same-day PR-based design). Mechanics (also in
AGENTS.md, git workflow):

- One local `docs/<topic>` branch per page or chapter, small enough to
  review in one sitting; **never pushed until approved**, so the
  public history shows clean merges only.
- The agent builds a local preview (`make html`, later
  `sphinx-autobuild` for live reload) so review happens on rendered
  pages plus the git diff, not rst source alone.
- Feedback channels: Silvano edits the text directly on the branch
  (authoritative), or drops anchored markers at the exact spot —
  `.. REVIEW: sentence A should be B` in rst (a comment, invisible in
  the rendered page), `# REVIEW: ...` in example scripts,
  `<!-- REVIEW: ... -->` in Markdown. Agents sweep with
  `grep -rn "REVIEW:" docs examples`, apply each marker, delete it,
  and fold generalizable corrections into the style guide on the same
  branch.
- **Merge gate:** zero open markers and an explicit approval from
  Silvano in chat; the agent then does the mechanical merge onto dev,
  push, and branch cleanup.
- Purpose beyond quality: reviewing the docs is how the owner audits
  the public API surface; expect review to produce upstream
  change requests against `src/` semantics, which spin off as separate
  branches, not as workarounds in the page under review.

Docs *infrastructure* (conf.py, CI workflows, templates, scrapers) is
exempt and follows the normal branch-merge workflow.

## Prerequisites (state, 2026-07-11)

- CDFViewer zarr support: **done**, draft PR
  [Gordi42/CDFViewer.jl#1](https://github.com/Gordi42/CDFViewer.jl/pull/1)
  — merge, then tag/pin for CI. (Zarr v3 is detected-but-rejected;
  blocked upstream on ZarrDatasets' `Zarr = "0.9"` cap, not needed here.)
- Writer CF reference-date fix (whole seconds, DateTime-decodable):
  **done**, draft PR
  [Gordi42/fridom#1](https://github.com/Gordi42/fridom/pull/1) — merge.
- CDFViewer Linux release binaries: **in progress** (separate agent).

## Phase 1 — CI skeleton (content-independent, start now)

New `.github/workflows/docs.yml`:

1. Replicate today's RTD build on Actions and deploy to GitHub Pages
   (uv env, `sphinx-build`, `actions/deploy-pages`). Green before any
   content changes.
2. Add the CDFViewer leg (release binary download, or setup-julia +
   cache fallback) and ffmpeg.
3. `actions/cache` on the gallery output dir. Key MUST include
   `hashFiles('uv.lock', 'examples/**')` — sphinx-gallery's `.md5` skip
   hashes only script content, so dependency bumps must bust the cache.
4. Weekly scheduled full build (beats the 7-day cache eviction, catches
   silent divergence) + job-level `timeout-minutes` (sphinx-gallery has
   no per-example timeout).
5. PR strategy: PRs execute only changed examples (git-diff →
   `filename_pattern`, the MNE/scikit-learn trick) and/or a short-run
   env flag; full execution on main and the weekly cron.

Budget check: GHA public runners are 4 vCPU / 16 GB, 6 h/job, free;
Pages site ≤ 1 GB (a dozen low-res mp4s at 5–20 MB is fine).

### CI cost (settled 2026-07-11)

Owner review round-trips cost **zero CI**: the review loop is local
(local branch, local preview build; see Review workflow above), so
nothing runs on GitHub until the approved merge lands on dev. The
remaining CI rules:

- `tests.yml` skips docs-only changes (`paths-ignore` on `docs/**`,
  `examples/**`, `design/**`, `assets/**`, `**.md`; done 2026-07-11)
  and cancels superseded runs per branch — this keeps docs-only
  *merges to dev* from running the test suite.
- `docs.yml` runs the full executed build on push to dev, the weekly
  cron, and `workflow_dispatch`. Its PR trigger (paths-limited to
  `docs/**`/`examples/**`, cheap leg only: quick build or
  changed-examples per item 5) exists for **outside-contributor PRs**,
  not for owner review.
- The thin RTD project is likewise re-scoped: it provides rendered
  previews for public PRs from outside contributors; the owner reviews
  locally.

## Phase 2 — Pilot example, end-to-end

Port **one** example — `shallowwater/barotropic_instability` (2D,
cheap) — to the new stack at doc resolution, executing in CI:
run → `fr.io.Writer` zarr → visible `cdfviewer --record` line → mp4 in
the built page. The pilot fixes all conventions:

- resolution/runtime budget per example (target: ≤ ~2 min each on CI)
- how the visible cdfviewer line is executed (subprocess call in the
  example vs. scraper-executed literal block — decide here)
- the new gallery scraper replacing `copy_media_files`
- thumbnail selection, short-run env flag mechanics

Keep it to one example on purpose; every integration surprise lands here.

## Phase 3 — Port all examples (parallel, agent-friendly)

The remaining 12 examples plus the new `sw.eigenbasis` gallery example
(β slow-mode filtering, `projection_eigenmode_roadmap.md` §5). This is
the **parity shakedown** the cutover plan wants — API awkwardness found
while porting feeds the framework2 fixes
(`framework2-userfacing-awkwardness` audit), not workarounds in the
examples.

## Phase 4 — Prose docs rewrite

Getting started, installation, the 10 tutorials — rewritten for the new
stack; tutorials containing code are authored as executed gallery
scripts so one execution mechanism covers everything (tiny inline API
snippets on non-gallery pages: `sphinx.ext.doctest`). API reference:
adapt the custom autosummary/Jinja machinery (`load_modules.py`,
`_templates/autosummary/`) to the new package layout.

Normative inputs (added 2026-07-11): the page tree and per-page scopes
in [`../../specs/docs/structure.md`](../../specs/docs/structure.md),
the writing/figure/citation rules in
[`../../specs/docs/style_guide.md`](../../specs/docs/style_guide.md).
The style guide adds a small Phase 0.5 to this program: docs-lint
script, shared mplstyle + palette, references.bib workflow — all
content-independent and parallel to Phase 1.

## Phase 5 — Retirement

- delete LFS video/figure assets, `copy_media_files`,
  `@skip_on_doc_build` usage
- thin `.readthedocs.yaml` to the preview-only build (drop the git-lfs
  hack); old-stack doc pages go with Wave C
- versioning: none until the first stable release, then subdirectory
  builds + theme version switcher (sphinx-multiversion is dead;
  sphinx-polyversion if tooling is wanted)

## Reference numbers

- RTD Community: 15 min/build, 7 GB RAM, fresh container every build
  (no cache), `build.tools` has no Julia.
- sphinx-gallery ≥ 0.17: `matplotlib_animations = (True, "mp4")`
  (needs ffmpeg + sphinxcontrib-video), experimental `parallel`,
  `junit` per-example timing.
- CDFViewer CI timings (measured, its own repo): cold ~19 min,
  warm cache ~3–5 min, full docs render 3.4 min; binary route:
  ~30–60 s download + seconds of startup.
