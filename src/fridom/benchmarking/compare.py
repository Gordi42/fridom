"""Compare and render benchmark suite results."""
from __future__ import annotations

from dataclasses import dataclass

from fridom.benchmarking.result import (  # noqa: TC001
    BenchmarkResult,
    RunMetadata,
    SuiteResult,
)

DEFAULT_THRESHOLD = 0.05


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
        """The change of the median wall time."""
        return MetricDelta(
            self.base.wall_median if self.base else None,
            self.new.wall_median if self.new else None,
        )

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

        Parameters
        ----------
        threshold : float, optional
            The relative wall-time change above which a case counts
            as slower/faster (default: 0.05).

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
        if rel is not None and rel > threshold:
            return "slower"
        if rel is not None and rel < -threshold:
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
        "case", "wall (base)", "wall (new)", "d wall", "d compile",
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
