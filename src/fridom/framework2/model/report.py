"""
The printable assembly report.

Description
-----------
Owning class spec: ``notes/framework2/model/classes/model.md``
(section "AssemblyReport (+ Fingerprint)"). ``model.report`` — a
host object printable without device sync, logged at INFO. The
section structure is fixed (the d4_1 mock is the format reference),
in order: header, fields, parameters, dispatch, schedule,
halo/layout, lint, and the run-start addendum (a placeholder until
the first ``run()`` appends the defaults-vs-user-initialized
provenance table — D1.1's logging promise; wave 4.2 seam:
:meth:`AssemblyReport.append_run_start`). Section *content* is
composed by the assembly pipeline (``assembly.assemble``); this
class owns structure and rendering only.
"""
# Wave 4 A: AssemblyReport
from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

#: the fixed section keys, in print order (model.md section 3)
SECTION_ORDER: Final[tuple[str, ...]] = (
    "header", "fields", "parameters", "dispatch", "schedule",
    "halo", "lint", "run_start")

#: printed section titles, keyed by section name
_TITLES: Final[dict[str, str]] = {
    "header": "Assembly",
    "fields": "Fields",
    "parameters": "Parameters",
    "dispatch": "Dispatch",
    "schedule": "Schedule",
    "halo": "Halo / layout",
    "lint": "Lint",
    "run_start": "Run start",
}

#: the assembly-time run-start placeholder (replaced at first run())
RUN_START_PLACEHOLDER: Final[str] = (
    "pending: the defaults-vs-user-initialized provenance table is "
    "appended at the first run()")


# ================================================================
#  AssemblyReport
# ================================================================
class AssemblyReport:

    """
    ``model.report`` — the printable assembly record.

    Description
    -----------
    Eight prebuilt text sections in fixed order (printable without
    any device sync — every value was stringified at assembly).
    ``header`` doubles as ``Model.__repr__``; ``section(name)`` is
    the targeted read; ``append_run_start`` is the one sanctioned
    post-assembly mutation (the first ``run()`` appends the
    defaults-vs-user-initialized provenance table).

    Parameters
    ----------
    sections : Mapping[str, str]
        The section texts, keyed by every name in
        :data:`SECTION_ORDER` (all keys required, no extras).
    """

    __slots__ = ("_sections",)

    def __init__(self, sections: Mapping[str, str]) -> None:
        """Validate the section keys and freeze the print order."""
        unknown = tuple(name for name in sections
                        if name not in SECTION_ORDER)
        if unknown:
            raise ValueError(
                f"unknown report section(s) {unknown}; sections "
                f"are {SECTION_ORDER}")
        missing = tuple(name for name in SECTION_ORDER
                        if name not in sections)
        if missing:
            raise ValueError(
                f"missing report section(s) {missing}; sections "
                f"are {SECTION_ORDER}")
        self._sections: dict[str, str] = {
            name: str(sections[name]) for name in SECTION_ORDER}

    # ================================================================
    #  Read surface
    # ================================================================
    @property
    def header(self) -> str:
        """The header section (grid/stepper/modules/name/digest)."""
        return self._sections["header"]

    def section(self, name: str) -> str:
        """
        Return one section's text.

        Parameters
        ----------
        name : str
            A section name from :data:`SECTION_ORDER`.

        Returns
        -------
        str
            The section text.

        Raises
        ------
        KeyError
            If ``name`` is not a report section (the message lists
            the valid names).
        """
        try:
            return self._sections[name]
        except KeyError:
            raise KeyError(
                f"no report section {name!r}; sections are "
                f"{SECTION_ORDER}") from None

    def __str__(self) -> str:
        """Render every section under its title, in fixed order."""
        blocks = []
        for name in SECTION_ORDER:
            title = _TITLES[name]
            rule = "-" * len(title)
            blocks.append(
                f"{title}\n{rule}\n{self._sections[name]}")
        return "\n\n".join(blocks)

    def __repr__(self) -> str:
        """Compact host-side summary (the header's first line)."""
        first = self._sections["header"].splitlines()[0]
        return f"<AssemblyReport: {first}>"

    # ================================================================
    #  The run-start addendum (wave 4.2 seam)
    # ================================================================
    def append_run_start(self, text: str) -> None:
        """
        Append the run-start addendum (first ``run()`` only).

        Description
        -----------
        The defaults-vs-user-initialized provenance table (D1.1's
        logging promise) replaces the assembly-time placeholder;
        repeated calls append further lines.

        Parameters
        ----------
        text : str
            The addendum text.
        """
        current = self._sections["run_start"]
        if current == RUN_START_PLACEHOLDER:
            self._sections["run_start"] = str(text)
        else:
            self._sections["run_start"] = f"{current}\n{text}"
