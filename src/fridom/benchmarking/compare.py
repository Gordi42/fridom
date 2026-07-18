"""Compare and render benchmark suite results."""
from __future__ import annotations

from dataclasses import dataclass

from fridom.benchmarking.result import (  # noqa: TC001
    BenchmarkResult,
    RunMetadata,
    SuiteResult,
)

DEFAULT_THRESHOLD = 0.05

# the noise-derived tolerance band is K times the baseline's own
# coefficient of variation; K = 3 is a ~3-sigma band on the per-case
# jitter (measured A100 CoV: 2-6% on small cases, <0.5% on large ones)
NOISE_TOLERANCE_K = 3.0

# an absolute floor (seconds, per measured chunk) below which a
# wall-time delta is treated as noise regardless of its relative size.
# A measured per-process slow mode adds ~0.8 ms per 50-step chunk
# (A100, 2026-07-18): every sample in an affected subprocess lands
# uniformly high, a fresh process reads normal, and it hits a random
# case each run. On sub-16 ms cases that ~0.8 ms is +5..9%, past the
# relative band, so the guard reds with no real regression; the
# relative tolerance cannot express it because the mode is absolute.
# The floor declares the harness's resolution limit (~one kernel launch
# per step at the 50-step chunk convention in
# benchmarks/model/bench_step.py); if the chunk length convention
# changes, revisit this value.
ABSOLUTE_FLOOR = 1.2e-3

# metadata fields that must match between the two runs for a
# comparison to be meaningful (a cpu run compared against a gpu
# baseline otherwise produces nonsense deltas silently)
ENV_GUARD_FIELDS = ("backend", "device_count", "device_kind", "jax_version")


# ================================================================
#  Environment guard
# ================================================================
@dataclass
class EnvMismatch:

    """
    A single mismatched environment field between two runs.

    Parameters
    ----------
    field : str
        The name of the mismatched metadata field.
    base : object | None
        The field value of the base run (None if absent).
    new : object | None
        The field value of the new run (None if absent).
    """

    field: str
    base: object | None
    new: object | None


def env_mismatches(
    base: RunMetadata, new: RunMetadata,
) -> list[EnvMismatch]:
    """
    Find environment fields that differ between two runs.

    Description
    -----------
    Compares the base and new metadata on the fields in
    ``ENV_GUARD_FIELDS`` (backend, device_count, device_kind,
    jax_version). A missing field (None on either side) counts as a
    mismatch, so an incompletely recorded run never silently passes
    the guard.

    Parameters
    ----------
    base : RunMetadata
        The metadata of the base run.
    new : RunMetadata
        The metadata of the new run.

    Returns
    -------
    list[EnvMismatch]
        One entry per mismatched field, in ``ENV_GUARD_FIELDS`` order;
        empty if the environments match.
    """
    mismatches = []
    for name in ENV_GUARD_FIELDS:
        base_val = getattr(base, name)
        new_val = getattr(new, name)
        if base_val is None or new_val is None or base_val != new_val:
            mismatches.append(EnvMismatch(name, base_val, new_val))
    return mismatches


def format_env_mismatches(mismatches: list[EnvMismatch]) -> str:
    """
    Render environment mismatches as a human-readable block.

    Parameters
    ----------
    mismatches : list[EnvMismatch]
        The mismatched fields to render.

    Returns
    -------
    str
        A multi-line description listing every mismatched field with
        both values.
    """
    lines = ["environment mismatch between base and new run:"]
    lines.extend(
        f"  {m.field}: base={m.base!r} new={m.new!r}" for m in mismatches
    )
    return "\n".join(lines)


# ================================================================
#  Value formatting
# ================================================================
def _format_seconds(value: float | None) -> str:
    """Format a duration in seconds with a human-readable unit."""
    if value is None:
        return "-"
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if abs(value) >= scale:
            return f"{value / scale:.2f} {unit}"
    return f"{value / 1e-9:.2f} ns"


def _format_bytes(value: int | None) -> str:
    """Format a number of bytes with a human-readable unit."""
    if value is None:
        return "-"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(size) < 1024:  # noqa: PLR2004
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def _format_rel(value: float | None) -> str:
    """Format a relative change as a signed percentage."""
    if value is None:
        return "-"
    return f"{value:+.1%}"


def _format_tolerance(
    case: CaseComparison, status: str, threshold: float,
) -> str:
    """
    Render the effective tolerance and the rule that set it.

    Description
    -----------
    Only comparable cases (ok/slower/faster) have a tolerance band;
    added, removed, and errored cases render as "-".

    The rule is normally "global" or "noise" (see
    ``effective_tolerance``); it reads "floor" when the relative band
    was exceeded but the absolute delta sat under ``ABSOLUTE_FLOOR``,
    so the case is "ok" only because the floor suppressed it. Surfacing
    it keeps floor suppressions visible in the report rather than
    silent.
    """
    if status not in ("ok", "slower", "faster"):
        return "-"
    tol, rule = case.effective_tolerance(threshold)
    rel = case.wall.rel
    if status == "ok" and rel is not None and abs(rel) > tol:
        rule = "floor"
    return f"{tol:.1%} ({rule})"


def _describe_run(mdata: RunMetadata) -> str:
    """Build a one-line description of a benchmark run."""
    commit = (mdata.commit or "unknown")[:8]
    if mdata.dirty:
        commit += "+"
    branch = f" ({mdata.branch})" if mdata.branch else ""
    backend = mdata.backend or "unknown"
    return (
        f"{commit}{branch} on {backend} "
        f"[{mdata.device_kind} x{mdata.device_count}]"
    )


# ================================================================
#  Comparison data model
# ================================================================
@dataclass
class MetricDelta:

    """
    The change of a single metric between two runs.

    Parameters
    ----------
    base : float | None
        The metric value of the base run (None if unavailable).
    new : float | None
        The metric value of the new run (None if unavailable).
    """

    base: float | None
    new: float | None

    @property
    def rel(self) -> float | None:
        """The relative change (new - base) / base, or None."""
        if self.base is None or self.new is None or self.base == 0:
            return None
        return (self.new - self.base) / self.base


def _memory_bytes(result: BenchmarkResult) -> int | None:
    """Return the memory metric: peak bytes if available, else temp."""
    if result.peak_bytes is not None:
        return result.peak_bytes
    return result.temp_bytes


@dataclass
class CaseComparison:

    """
    The comparison of one benchmark case between two runs.

    Parameters
    ----------
    full_name : str
        The full case name (including parameter values).
    base : BenchmarkResult | None
        The result of the base run (None if the case was added).
    new : BenchmarkResult | None
        The result of the new run (None if the case was removed).
    """

    full_name: str
    base: BenchmarkResult | None
    new: BenchmarkResult | None

    @property
    def wall(self) -> MetricDelta:
        """
        The change of the minimum wall time.

        Description
        -----------
        Environmental noise is one-sided: it only ever adds time, so
        the minimum over a case's samples is the least-contaminated
        estimator of its true cost (Chen & Revels, HPEC 2016). The
        rel/status/regression logic is all computed on this minimum;
        the median is still shown per run by ``format_suite``.
        """
        return MetricDelta(
            self.base.wall_min if self.base else None,
            self.new.wall_min if self.new else None,
        )

    @property
    def cov_base(self) -> float:
        """
        The coefficient of variation of the base wall times.

        Description
        -----------
        Defined as ``std(base.wall_times) / mean(base.wall_times)``.
        Degenerate cases (no base, empty/single-sample times, or a
        zero mean) yield 0.0 so the noise-derived tolerance falls back
        to the global threshold.
        """
        if self.base is None:
            return 0.0
        if len(self.base.wall_times) < 2:  # noqa: PLR2004
            return 0.0
        mean = self.base.wall_mean
        if not mean:
            return 0.0
        return self.base.wall_std / mean

    def effective_tolerance(
        self, threshold: float = DEFAULT_THRESHOLD,
    ) -> tuple[float, str]:
        """
        Return the per-case tolerance band and the rule that set it.

        Description
        -----------
        The band is ``max(threshold, NOISE_TOLERANCE_K * cov_base)``:
        the global threshold, widened to a noise-derived band whenever
        the baseline's own jitter exceeds it.

        Parameters
        ----------
        threshold : float, optional
            The global relative wall-time threshold (default: 0.05).

        Returns
        -------
        tuple[float, str]
            The effective tolerance and the rule that set it, one of
            "global" (the flat threshold) or "noise" (the CoV band).
        """
        noise = NOISE_TOLERANCE_K * self.cov_base
        if noise > threshold:
            return noise, "noise"
        return threshold, "global"

    @property
    def compile(self) -> MetricDelta:
        """The change of the compile time."""
        return MetricDelta(
            self.base.compile_time if self.base else None,
            self.new.compile_time if self.new else None,
        )

    @property
    def memory(self) -> MetricDelta:
        """The change of the memory metric (peak or temp bytes)."""
        return MetricDelta(
            _memory_bytes(self.base) if self.base else None,
            _memory_bytes(self.new) if self.new else None,
        )

    def status(self, threshold: float = DEFAULT_THRESHOLD) -> str:
        """
        Classify the case comparison.

        Description
        -----------
        The comparison uses the per-case effective tolerance
        (``effective_tolerance``), not the bare ``threshold``: a case
        whose minimum wall time moved by less than its own noise band
        counts as "ok" even past the global threshold, and vice versa.

        A case is flagged "slower"/"faster" only if the relative delta
        clears the effective tolerance *and* the absolute delta of the
        minima clears ``ABSOLUTE_FLOOR``. The floor suppresses the
        per-process slow mode (see ``ABSOLUTE_FLOOR``), which the
        relative band cannot express because it is absolute; it is well
        below any real tiny-case regression, so it never masks one.
        The suppression is symmetric: a spurious "faster" from a
        slow-mode-contaminated baseline is equally meaningless.

        Parameters
        ----------
        threshold : float, optional
            The global relative wall-time change above which a case
            counts as slower/faster, before the per-case noise band is
            applied (default: 0.05).

        Returns
        -------
        str
            One of "error", "added", "removed", "slower", "faster",
            or "ok".
        """
        base_error = self.base is not None and self.base.error is not None
        new_error = self.new is not None and self.new.error is not None
        if base_error or new_error:
            return "error"
        if self.base is None:
            return "added"
        if self.new is None:
            return "removed"
        rel = self.wall.rel
        tol, _ = self.effective_tolerance(threshold)
        if rel is not None:
            abs_delta = self.wall.new - self.wall.base
            if rel > tol and abs_delta > ABSOLUTE_FLOOR:
                return "slower"
            if rel < -tol and -abs_delta > ABSOLUTE_FLOOR:
                return "faster"
        return "ok"


@dataclass
class SuiteComparison:

    """
    The comparison of two benchmark suite runs.

    Parameters
    ----------
    base_metadata : RunMetadata
        The metadata of the base run.
    new_metadata : RunMetadata
        The metadata of the new run.
    cases : list[CaseComparison]
        The per-case comparisons.
    """

    base_metadata: RunMetadata
    new_metadata: RunMetadata
    cases: list[CaseComparison]

    def regressions(
        self, threshold: float = DEFAULT_THRESHOLD,
    ) -> list[CaseComparison]:
        """
        Return the cases that regressed (slower or errored).

        Parameters
        ----------
        threshold : float, optional
            The relative wall-time change above which a case counts
            as slower (default: 0.05).

        Returns
        -------
        list[CaseComparison]
            The regressed cases.
        """
        return [
            case for case in self.cases
            if case.status(threshold) in ("slower", "error")
        ]


def compare(base: SuiteResult, new: SuiteResult) -> SuiteComparison:
    """
    Compare two benchmark suite results.

    Description
    -----------
    Matches the cases of both runs by their full name. Cases present
    in only one of the runs are included and classified as "added" or
    "removed".

    Parameters
    ----------
    base : SuiteResult
        The base run (e.g. the main branch).
    new : SuiteResult
        The new run (e.g. a feature branch).

    Returns
    -------
    SuiteComparison
        The comparison; cases follow the order of the new run, with
        removed cases appended in the order of the base run.
    """
    base_by_name = {result.full_name: result for result in base.results}
    new_by_name = {result.full_name: result for result in new.results}

    cases = [
        CaseComparison(
            full_name=name,
            base=base_by_name.get(name),
            new=result,
        )
        for name, result in new_by_name.items()
    ]
    cases.extend(
        CaseComparison(full_name=name, base=result, new=None)
        for name, result in base_by_name.items()
        if name not in new_by_name
    )
    return SuiteComparison(
        base_metadata=base.metadata,
        new_metadata=new.metadata,
        cases=cases,
    )


# ================================================================
#  Rendering
# ================================================================
def _render_table(
    header: list[str],
    rows: list[list[str]],
    markdown: bool,
) -> str:
    """Render a table in markdown or aligned plain-text format."""
    if markdown:
        lines = [
            "| " + " | ".join(header) + " |",
            "|" + "|".join(" --- " for _ in header) + "|",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows)) if rows
        else len(header[i])
        for i in range(len(header))
    ]

    def render_row(row: list[str]) -> str:
        # first column left-aligned, the rest right-aligned
        cells = [row[0].ljust(widths[0])]
        cells += [cell.rjust(width)
                  for cell, width in zip(row[1:], widths[1:], strict=True)]
        return "  ".join(cells).rstrip()

    lines = [render_row(header)]
    lines.append("-" * len(lines[0]))
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def format_suite(suite: SuiteResult, *, markdown: bool = False) -> str:
    """
    Render the results of a single benchmark run as a table.

    Parameters
    ----------
    suite : SuiteResult
        The suite result to render.
    markdown : bool, optional
        Whether to render a markdown table instead of aligned plain
        text (default: False).

    Returns
    -------
    str
        The rendered report.
    """
    header = ["case", "wall", "std", "compile", "memory", "status"]
    rows = []
    for result in suite.results:
        status = "error" if result.error is not None else "ok"
        rows.append([
            result.full_name,
            _format_seconds(result.wall_median),
            _format_seconds(result.wall_std),
            _format_seconds(result.compile_time),
            _format_bytes(_memory_bytes(result)),
            status,
        ])
    lines = [
        f"run: {_describe_run(suite.metadata)}",
        "",
        _render_table(header, rows, markdown),
    ]
    return "\n".join(lines)


def format_comparison(
    comparison: SuiteComparison,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    markdown: bool = False,
) -> str:
    """
    Render a suite comparison as a table.

    Parameters
    ----------
    comparison : SuiteComparison
        The comparison to render.
    threshold : float, optional
        The relative wall-time change above which a case counts as
        slower/faster (default: 0.05).
    markdown : bool, optional
        Whether to render a markdown table instead of aligned plain
        text (default: False).

    Returns
    -------
    str
        The rendered report.
    """
    header = [
        "case", "wall (base)", "wall (new)", "d wall", "tol", "d compile",
        "d memory", "status",
    ]
    rows = []
    counts: dict[str, int] = {}
    for case in comparison.cases:
        status = case.status(threshold)
        counts[status] = counts.get(status, 0) + 1
        rows.append([
            case.full_name,
            _format_seconds(case.wall.base),
            _format_seconds(case.wall.new),
            _format_rel(case.wall.rel),
            _format_tolerance(case, status, threshold),
            _format_rel(case.compile.rel),
            _format_rel(case.memory.rel),
            status,
        ])
    summary = ", ".join(
        f"{count} {status}" for status, count in sorted(counts.items()))
    lines = [
        f"base: {_describe_run(comparison.base_metadata)}",
        f"new:  {_describe_run(comparison.new_metadata)}",
        "",
        _render_table(header, rows, markdown),
        "",
        f"{len(comparison.cases)} cases: {summary}",
    ]
    return "\n".join(lines)
