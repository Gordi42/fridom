"""Entry point to run a single benchmark case in a child process."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fridom.benchmarking.suite import CaseInstance, load_cases

N_ARGS = 5


def main(argv: list[str] | None = None) -> int:
    """
    Run a single benchmark case instance and write the result to disk.

    Description
    -----------
    Invoked by the benchmark runner as
    `python -m fridom.benchmarking._child` to isolate a benchmark case
    in its own process (clean memory peaks, no jit-cache cross-talk).

    Parameters
    ----------
    argv : list[str] | None, optional
        The command line arguments: the suite file, the case name, the
        parameter values as JSON, the reps/warmup overrides as JSON,
        and the output path for the result JSON; None reads
        `sys.argv` (default: None).

    Returns
    -------
    int
        The exit code: 0 on success, 1 on failure.
    """
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != N_ARGS:
        sys.stderr.write(
            "usage: python -m fridom.benchmarking._child "
            "<file> <case> <params-json> <overrides-json> <output>\n")
        return 1
    file, case_name, params_json, overrides_json, output = argv

    cases = [case for case in load_cases(file) if case.name == case_name]
    if not cases:
        sys.stderr.write(f"case '{case_name}' not found in {file}\n")
        return 1

    instance = CaseInstance(case=cases[0], params=json.loads(params_json))
    result = instance.run(**json.loads(overrides_json))
    Path(output).write_text(json.dumps(result.to_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
