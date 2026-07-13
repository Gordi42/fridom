---
status: normative
date: 2026-07-13
---

# Grid abstraction redesign — Open threads

Part of the grid redesign notes; see [`00_overview.md`](00_overview.md)
for the document map. Concepts are in
[`01_concepts.md`](01_concepts.md), rules in
[`02_rules.md`](02_rules.md).

---

## 7. Open threads

**All twelve threads are resolved.** Each decision has been folded into
the normative section that owns it; this page is retained only as a map
from the former thread to where its resolution now lives, so the
section numbering (a stable identifier) stays valid.

This page predates the class-design phase: the *currently* open
questions live in the `## Open questions` sections of the four class
docs under [`classes/`](classes/README.md).

| # | Former thread | Resolution lives in |
|---|---------------|---------------------|
| 1 | Immersed-domain object | [§3.7](02_rules.md#37-boundaries-ii-immersed-masked-domains) (`grid.immersed`), [§2.6](01_concepts.md#26-grid--the-assembly-object) |
| 2 | Transform scheduling for nonlinear terms | [§3.12](02_rules.md#312-dealiasing) (explicit transform-once combinator; auto-scheduling rejected) |
| 3 | Reductions and integrals | [§3.13](02_rules.md#313-reductions-and-integrals) |
| 4 | Metric terms and vector calculus | [§2.4](01_concepts.md#24-field) (thin `VectorField`), [§6.3](05_validation.md#63-sphere--curvilinear) |
| 5 | Naming | [§8](00_overview.md#8-migration-strategy) (`fr.meshes`/`fr.operators`; the grid stack ships as `fridom.spatial`, not the once-planned `framework.grid`) |
| 6 | jax specifics | [§2.7](01_concepts.md#27-where-coordinate-data-lives) (checkpoint knob, dynamic stencil coefficients), [§3.1](02_rules.md#31-strict-space-algebra) (scalar-changing criterion) |
| 7 | Module-system touchpoints | [§5](04_decomposition.md#5-domain-decomposition) (halo-accounting trace), [§3.4](02_rules.md#34-generic-operator-dispatch) (override registration) |
| 8 | Eigenmode objects | [§2.5](01_concepts.md#25-operator--typed-maps-between-spaces) (model-side, out of scope for the grid redesign; landed as `fr.model.eigenbasis` — [`../model/07_open_threads.md`](../model/07_open_threads.md) thread 5) |
| 9 | Shape-changing stencil machinery | [§3.5](02_rules.md#35-shape-is-a-property-of-the-space) (slice-on-halo), [§5](04_decomposition.md#5-domain-decomposition) |
| 10 | Coordinate-mapping object | [§3.8](02_rules.md#38-boundaries-iii-terrain-following-boundary-fitted) (`CoordinateMapping`, `grid.metric`) — built out by ROADMAP 3.4 (charts, mapped/terrain-following grids, dynamic metrics) |
| 11 | Inhomogeneous BC machinery | [§3.6](02_rules.md#36-boundaries-i-conforming-bc-structure-vs-boundary-data) (extended space, `ghost_fill`, module interface) |
| 12 | Random fields | [§3.10](02_rules.md#310-discretizing-continuous-functions) (sharded per-shard keying), [§5](04_decomposition.md#5-domain-decomposition) |
