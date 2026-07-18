"""Force the negotiated storage-halo width below the traced demand.

Description
-----------
The storage width per coordinate name is the negotiated ``HaloSpec``
(``grid.decomposition.halo``), which is the pointwise maximum of the
per-name accumulated depth over the traced tendency chain. Passing an
explicit ``halo=`` spec to ``negotiate`` can only *raise* the width
(``merge_max``), so it cannot force a *narrower* store. The only clean
lever is to clamp the negotiated spec itself.

``force_halo(cap)`` is a context manager that patches the two seams
that decide the width:

* ``decomposition._negotiated_halo`` — the value stored into the
  ``TensorDecomposition`` at (re)negotiation, i.e. the actual storage
  width. Each per-name width is clamped to ``min(width, cap)``.
* ``Grid._demanded_halo`` — the freshly traced demand the *frozen*
  grid verifies against its recorded fingerprint. Clamped identically
  so the frozen-verify path (``_halo_violations(demand, record.halo)``)
  stays consistent with the clamped record instead of raising
  ``GridFrozenError`` (the record is width ``cap``, an unclamped demand
  of width 4 would look like a violation).

Both patches must stay active across the whole model lifecycle
(assembly + freeze + run/compile), so wrap the entire variant in one
``with force_halo(cap): ...`` block. Widths already at or below ``cap``
are untouched (the clamp is a floor-preserving ``min``).
"""
from __future__ import annotations

import contextlib

import fridom.spatial.decomposition.decomposition as _decomp
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid


def _clamp(spec: HaloSpec, cap: int) -> HaloSpec:
    """Return a copy of `spec` with every width clamped to <= cap."""
    return HaloSpec({name: min(width, cap)
                     for name, width in spec.widths})


@contextlib.contextmanager
def force_halo(cap: int):
    """Clamp every negotiated/verified halo width to <= `cap`."""
    orig_negotiated = _decomp._negotiated_halo  # noqa: SLF001
    orig_demanded = Grid._demanded_halo  # noqa: SLF001

    def clamped_negotiated(*args, **kwargs):
        return _clamp(orig_negotiated(*args, **kwargs), cap)

    def clamped_demanded(self, *args, **kwargs):
        return _clamp(orig_demanded(self, *args, **kwargs), cap)

    _decomp._negotiated_halo = clamped_negotiated  # noqa: SLF001
    Grid._demanded_halo = clamped_demanded  # noqa: SLF001
    try:
        yield
    finally:
        _decomp._negotiated_halo = orig_negotiated  # noqa: SLF001
        Grid._demanded_halo = orig_demanded  # noqa: SLF001
