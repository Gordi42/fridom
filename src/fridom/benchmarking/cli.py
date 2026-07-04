"""Command line interface for running and comparing benchmarks."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from fridom.benchmarking.compare import (
    DEFAULT_THRESHOLD,
    compare,
    format_comparison,
    format_suite,
)
from fridom.benchmarking.result import SuiteResult
from fridom.benchmarking.runner import run_suite
from fridom.benchmarking.suite import discover_cases


# ================================================================
#  Subcommands
# ================================================================
def _default_output(directory: Path, suite: SuiteResult) -> Path:
    """Build the default output path for a suite result."""
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    commit = (suite.metadata.commit or "unknown")[:8]
    return directory / "results" / f"{timestamp}-{commit}.json"


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a benchmark suite and save the results."""
    directory = Path(args.directory)
    suite = run_suite(
        directory,
        pattern=args.filter,
        reps=args.reps,
        warmup=args.warmup,
        isolate=not args.no_isolate,
        timeout=args.timeout,
        progress=print,
    )
    output = (
        Path(args.output) if args.output is not None
        else _default_output(directory, suite)
    )
    suite.save(output)
    print()
    print(format_suite(suite, markdown=args.markdown))
    print()
    print(f"results written to {output}")
    failed = [result for result in suite.results if result.error is not None]
    for result in failed:
        print()
        print(f"case {result.full_name} failed:")
        print(result.error)
    return 1 if failed else 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compare two benchmark suite results."""
    comparison = compare(
        SuiteResult.load(args.base), SuiteResult.load(args.new))
    print(format_comparison(
        comparison, threshold=args.threshold, markdown=args.markdown))
    if args.fail_on_regression and comparison.regressions(args.threshold):
        return 1
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """List all benchmark case instances of a suite directory."""
    for case in discover_cases(Path(args.directory)):
        for instance in case.instances():
            print(instance.full_name)
    return 0


# ================================================================
#  Parser
# ================================================================
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of the benchmarking CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m fridom.benchmarking",
        description="Run and compare fridom benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="run a benchmark suite directory")
    run.add_argument(
        "directory", nargs="?", default="benchmarks",
        help="the suite directory (default: benchmarks)")
    run.add_argument(
        "--filter", default=None,
        help="only run cases whose full name contains this substring")
    run.add_argument(
        "--reps", type=int, default=None,
        help="override the number of timed repetitions of all cases")
    run.add_argument(
        "--warmup", type=int, default=None,
        help="override the number of warmup calls of all cases")
    run.add_argument(
        "--no-isolate", action="store_true",
        help="run cases in-process instead of in fresh subprocesses")
    run.add_argument(
        "--timeout", type=float, default=None,
        help="timeout per isolated case in seconds")
    run.add_argument(
        "--markdown", action="store_true",
        help="render the report as a markdown table")
    run.add_argument(
        "-o", "--output", default=None,
        help="output path of the result JSON "
             "(default: <directory>/results/<timestamp>-<commit>.json)")
    run.set_defaults(func=_cmd_run)

    comp = subparsers.add_parser(
        "compare", help="compare two benchmark result files")
    comp.add_argument("base", help="the base result JSON")
    comp.add_argument("new", help="the new result JSON")
    comp.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="relative wall-time change flagged as slower/faster "
             f"(default: {DEFAULT_THRESHOLD})")
    comp.add_argument(
        "--markdown", action="store_true",
        help="render the report as a markdown table")
    comp.add_argument(
        "--fail-on-regression", action="store_true",
        help="exit with a non-zero code if any case is slower or "
             "errored")
    comp.set_defaults(func=_cmd_compare)

    lst = subparsers.add_parser(
        "list", help="list all benchmark case instances")
    lst.add_argument(
        "directory", nargs="?", default="benchmarks",
        help="the suite directory (default: benchmarks)")
    lst.set_defaults(func=_cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Entry point of the benchmarking CLI.

    Parameters
    ----------
    argv : list[str] | None, optional
        The command line arguments; None reads `sys.argv`
        (default: None).

    Returns
    -------
    int
        The exit code.
    """
    args = build_parser().parse_args(argv)
    return args.func(args)
