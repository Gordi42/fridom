"""Every ``fr.``/``nh.``/``sw.`` spelling written in the new stack resolves.

Docstrings and error messages advertise import paths. When a name moves
(or was never lifted to the depth the prose claims), the prose keeps
pointing at an ``AttributeError`` and an example copied from it fails at
import. This test walks the new-stack sources, harvests every dotted
``fr.``/``nh.``/``sw.`` token that appears anywhere in the file (prose,
comments, code), and resolves it by attribute walk.

The alias meaning is read per file: new-stack modules use
``import fridom as fr``, but a handful still carry the old-stack
``import fridom.framework as fr`` (a cutover leftover), and there the
same token means something else.
"""
import pathlib
import re

import pytest

import fridom
import fridom.nonhydro2
import fridom.shallowwater2

# The new stack only. ``framework``/``nonhydro``/``shallowwater`` are the
# old stack, where ``fr`` means ``fridom.framework``; they go away at
# cutover and are not held to this.
NEW_STACK = ("spatial", "model", "io", "ops", "nonhydro2", "shallowwater2")

DEFAULT_ALIASES = {
    "fr": "fridom",
    "nh": "fridom.nonhydro2",
    "sw": "fridom.shallowwater2",
}

TOKEN = re.compile(r"(?<![\w.])(fr|nh|sw)\.([A-Za-z_][A-Za-z0-9_.]*)")
ALIAS_IMPORT = re.compile(
    r"^import\s+([\w.]+)\s+as\s+(fr|nh|sw)\s*$", re.MULTILINE)

#: Spellings that are *meant* to fail on attribute access, or that a
#: concurrent change owns. Keyed ``"<relative path>::<token>"``.
KNOWN_DRIFT = frozenset({
    # a deliberate teaching AttributeError for the retired name
    "nonhydro2/params.py::nh.params.DSQR",
    # drift in files owned by concurrent work (2026-08-13); drop each
    # entry as its owner lands the fix
    "model/model.py::fr.params.TIME_STEP",
    "model/model.py::fr.terms",
    "model/model.py::fr.Module",
    "model/model.py::fr.modules.TendencyEnvelope",
    "model/modules/advection.py::fr.operators.Restriction",
    "spatial/coordinate_mapping.py::fr.modules.RotationCoriolis",
    "spatial/fields/scalar_field.py::fr.Real",
    "spatial/fields/scalar_field.py::fr.operators.diff",
    "spatial/fields/scalar_field.py::fr.operators.embed",
    "spatial/fields/scalar_field.py::fr.operators.as_profile",
    "spatial/fields/scalar_field.py::fr.operators.integrate",
    "spatial/fields/scalar_field.py::fr.operators.Fourier",
    "spatial/fields/vector_field.py::fr.operators.Divergence",
})

ROOT = pathlib.Path(fridom.__file__).parent
SOURCES = sorted(
    p for sub in NEW_STACK for p in (ROOT / sub).rglob("*.py"))


def _resolve(alias_target: str, path: str) -> bool:
    """Return whether ``<alias_target>.<path>`` resolves by getattr."""
    obj = fridom
    for part in alias_target.split(".")[1:] + path.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:  # noqa: BLE001 — any failure is a dead spelling
            return False
    return True


@pytest.mark.parametrize(
    "source", SOURCES, ids=[str(p.relative_to(ROOT)) for p in SOURCES])
def test_advertised_spellings_resolve(source: pathlib.Path):
    rel = str(source.relative_to(ROOT))
    text = source.read_text()
    aliases = dict(DEFAULT_ALIASES)
    aliases.update(
        {alias: target for target, alias in ALIAS_IMPORT.findall(text)})

    dead = set()
    for match in TOKEN.finditer(text):
        alias, path = match.group(1), match.group(2).rstrip(".")
        token = f"{alias}.{path}"
        if f"{rel}::{token}" in KNOWN_DRIFT:
            continue
        if not _resolve(aliases[alias], path):
            dead.add(token)
    assert not dead, (
        f"{rel} advertises import spellings that do not resolve: "
        f"{sorted(dead)}")


def test_sources_were_found():
    # a broken ROOT/glob would make the sweep above vacuously green
    assert len(SOURCES) > 100
