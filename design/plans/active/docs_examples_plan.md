---
status: active
date: 2026-07-13
---

# Docs & examples rebuild plan

Owner-approved 2026-07-11. Replaces the deferred "Examples refresh" /
"Docs refresh" items in `cutover_parity_plan.md` with a concrete program.
The docs are rebuilt from scratch for the new stack; examples and code
snippets **execute at doc build time** so they cannot diverge from the
code.

Normative inputs: the page tree and per-page scopes in
[`../../specs/docs/structure.md`](../../specs/docs/structure.md), the
writing/figure/citation rules in
[`../../specs/docs/style_guide.md`](../../specs/docs/style_guide.md).
This file only says how the pages are built and shipped.

## Settled decisions

- **Build system:** sphinx-gallery ≥ 0.17; examples run at build time.
  `@skip_on_doc_build` and the pre-rendered-media machinery
  (`custom_scraper.copy_media_files`, git-LFS videos) are retired as the
  old examples are ported. Low resolution is the accepted price.
- **Hosting:** GitHub Actions → GitHub Pages. RTD Community cannot do
  executed examples (15 min/build, no cache between builds, no Julia);
  the thin RTD project stays only for free rendered previews of
  outside-contributor PRs (`SPHINX_QUICK_BUILD`, execution off).
- **Stills:** `field.xr.plot(...)` in the example; the stock
  sphinx-gallery matplotlib scraper picks the figures up, styled by
  `docs/source/fridom_docs.mplstyle` via the gallery reset hook.
- **Animations:** the example writes zarr via the writer, then runs a
  **visible** `cdfviewer <store> ... --record` line
  (`subprocess.run(..., shell=True, check=True)`); `video_scraper.py`
  embeds the mp4. The command documents the real workflow. Videos are
  rebuilt each full build and never committed.
- **CDFViewer in CI:** the prebuilt Linux release binary
  (`v2026.7.0`, `cdfviewer-linux-x86_64.tar.zst`), run under xvfb +
  Mesa software GL. Install + smoke test ~35 s.
- **Divergence guard:** `only_warn_on_example_error = False` — a raising
  example fails the build.
- **Budget rule:** an example stays under ~90 s locally to survive the
  CI slowdown factor; `FRIDOM_EXAMPLES_FAST` selects a smoke-sized run.
- **No hold on the swap:** examples/docs are written against the current
  names (`fridom.nonhydro2` etc.); the Wave-C rename is a mechanical
  find/replace over `examples/` and `docs/`.

## Review workflow

All reader-facing content (`docs/` pages, `examples/` scripts) is
owner-reviewed **before** it reaches dev, and the review happens **off
GitHub** (the repo is public; review discussion is not). The mechanics —
local `docs/<topic>` branch, never pushed until approved; local preview
build; handoff by projecting the branch onto the main checkout as
unstaged changes (`git restore --source=docs/<topic> -- docs/
examples/`); `REVIEW:` markers; merge gate of zero markers plus explicit
approval — are binding and written down in **AGENTS.md** (git workflow).
Owner review round-trips cost zero CI.

Two consequences worth keeping here:

- Docs *infrastructure* (conf.py, workflows, templates, scrapers) is
  exempt and follows the normal branch-merge workflow.
- Reviewing the docs is how the owner audits the public API surface;
  expect review to produce change requests against `src/` semantics,
  which spin off as separate branches, not as workarounds in the page.

## Landed

- **Prerequisites** — CDFViewer zarr support merged; CDFViewer Linux
  release binaries shipped (`v2026.7.0`); the writer's CF reference-date
  fix (whole seconds, DateTime-decodable) merged.
- **Phase 1, CI skeleton** (2026-07-11) — `.github/workflows/docs.yml`
  builds and deploys to <https://gordi42.github.io/fridom/> (~3.5 min);
  CDFViewer release binary + ffmpeg leg; `actions/cache` on the gallery
  output dir keyed on `uv.lock` + `examples/**`; weekly full build;
  `timeout-minutes`. The PR leg is the quick build (`SPHINX_QUICK_BUILD`,
  no gallery/autodoc). `tests.yml` skips docs-only changes. `conf.py`
  mocks only *uninstalled* dependencies. Repo settings: Pages
  `build_type=workflow`, `github-pages` deployment branch policy for
  `dev`.
- **Phase 2, pilot example** (merged to dev, review closed 2026-07-11) —
  `examples/shallowwater/barotropic_instability.py` runs on the new
  stack, writes zarr, renders its animation through the visible
  `cdfviewer --record` line, and is embedded in the built gallery
  (`afb39c9a`, reviews `2f52bb0d` / `a8deb61c`, infra `605e6698`,
  `cc40d5f2`, `6fe39e48`, `c6c7a949`). It fixes the conventions every
  later port follows: the budget rule, the visible cdfviewer line,
  `video_scraper.py`, the mplstyle reset hook,
  `# sphinx_gallery_thumbnail_number`, and the `examples/**` ruff
  per-file ignores.
- **Upstream fix from the pilot** — staggered coordinate names no longer
  leak into user plotting; examples now plot and animate on plain `x` /
  `y` (`477b80bc`).
- **Phase 3, equatorial_waves** (merged 2026-07-20) — rewritten on the
  shallowwater2 numeric eigenbasis (`sw.eigenbasis`), rendered via
  CDFViewer/zarr; LFS figure + videos dropped (`1db28aaf`).

## Open upstream items (found while porting; fix in `src/`, not in the page)

- `fr.io` root alias missing: the io_ops spec spells `fr.io.Writer`, the
  pilot writes `fr.model.io.Writer`.
- `pot_vort` is not implemented on shallowwater2 (the pilot animates
  `rel_vort` instead).

## Phase 0.5 — Content-independent groundwork (partly done)

Parallel to everything else; required by the style guide.

- Shared mplstyle + palette — **done** (`fridom_docs.mplstyle`, wired
  through the gallery reset hook).
- `references.bib` and the citation workflow (style guide §11) — open.
- docs-lint script for the **[lint]**-marked style rules — open.

## Phase 3 — Port the remaining examples

Eleven old-stack scripts still import `fridom.nonhydro` and still
carry `@skip_on_doc_build` plus committed LFS media, all under
`examples/nonhydro/` (barotropic_jet, convection_and_closures,
dancing_eddies, internal_wave_maker, multiple_wave_makers,
rayleigh_bénard_convection, rayleigh_taylor_instability,
single_internal_wave, symmetric_instability, tracers_and_eddies,
wave_package). Plus one new gallery example: `sw.eigenbasis`
(β slow-mode filtering, `projection_eigenmode_roadmap.md` §5) —
distinct from the landed equatorial_waves rewrite, which uses the
eigenbasis for equatorial wave modes, not slow-mode filtering.

Parallelizable one example per branch, each following the pilot's
conventions. This is the **parity shakedown** the cutover plan wants:
API awkwardness found while porting feeds the framework2 fixes
(`framework2-userfacing-awkwardness` audit), not workarounds in the
examples. A ported example drops its `figures/` and `videos/` LFS assets
in the same branch.

## Phase 4 — Prose docs rewrite

Nothing here has started: `docs/source/` is still the old tree
(`getting_started.rst`, `installation.rst`, `tutorials/{using_models,
creating_models,more_tutorials}`, `fridom_api.rst`), written against the
old stack.

Build out the page tree of `specs/docs/structure.md`: Home,
Installation, Getting Started, the nine Guide chapters, the four starred
Advanced chapters, both Models chapters, Verification (one component +
one model case), References, Glossary. Pages containing code are
authored as executed gallery scripts, so one execution mechanism covers
everything; tiny inline API snippets on non-gallery pages use
`sphinx.ext.doctest`. API reference: adapt the custom
autosummary/Jinja machinery (`load_modules.py`,
`_templates/autosummary/`) to the new package layout. Each Advanced
chapter gets its own short planning note under `design/specs/docs/`
before writing starts.

**No stub pages** (structure.md): an unwritten chapter appears in no
toctree; the backlog lives here, not in the reader's navigation. Old
pages are deleted as their replacements land, not left orphaned.

## Phase 5 — Retirement

Blocked on Phases 3–4; nothing done yet.

- delete the LFS video/figure assets, `custom_scraper.copy_media_files`
  (still in the `image_scrapers` tuple in `conf.py`), and the
  `@skip_on_doc_build` decorator plus its uses in
  `framework/utils/decorators.py`
- thin `.readthedocs.yaml` to the preview-only build (drop the git-lfs
  hack); old-stack doc pages go with Wave C
- deferred from Phase 1 item 5: PRs execute only *changed* examples
  (git-diff → `filename_pattern`, the MNE/scikit-learn trick) — worth
  doing once many examples actually execute
- versioning: none until the first stable release, then subdirectory
  builds + a theme version switcher (sphinx-multiversion is dead;
  sphinx-polyversion if tooling is wanted)

## Reference numbers

- RTD Community: 15 min/build, 7 GB RAM, fresh container every build
  (no cache), `build.tools` has no Julia.
- sphinx-gallery ≥ 0.17: `matplotlib_animations = (True, "mp4")`
  (needs ffmpeg + sphinxcontrib-video), experimental `parallel`,
  `junit` per-example timing.
- GHA public runners: 4 vCPU / 16 GB, 6 h/job, free; Pages site ≤ 1 GB.
- CDFViewer: release-binary route ~30–60 s download plus seconds of
  startup (the setup-julia fallback, ~19 min cold / ~3–5 min warm, was
  never needed).
