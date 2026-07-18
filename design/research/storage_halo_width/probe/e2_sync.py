"""E2: does forcing a narrower halo insert a mid-chain sync?

Counts ``Grid.sync`` calls (each = one halo-exchange node inserted into
the traced graph) for baseline vs forced-narrow width, both for the
isolated advection tendency and for a full compiled step. Records the
field name + its ``halo_valid`` at each sync so an *extra* sync's
location in the chain is visible.
"""
from __future__ import annotations

import contextlib
import sys

import _common as C
from _force_halo import force_halo

import fridom as fr
from fridom.model.modules.advection import (
    CenteredAdvection,
    UpwindAdvection,
)
from fridom.spatial.grid import Grid


@contextlib.contextmanager
def count_syncs():
    log = []
    orig = Grid.sync

    def counting(self, field, boundary_data=None, *, materialize=False):
        try:
            valid = dict(field.halo_valid.widths)
        except Exception:  # noqa: BLE001
            valid = "?"
        log.append((getattr(field, "name", "?"), valid))
        return orig(self, field, boundary_data, materialize=materialize)

    Grid.sync = counting
    try:
        yield log
    finally:
        Grid.sync = orig


def probe(n, name, factory, cls, cap):
    print(f"--- {name} (cap={cap}) ---")
    build = (force_halo(cap) if cap is not None
             else contextlib.nullcontext())
    with build:
        model = C.make_model(n, factory())
        C.seed_state(model)
        print(f"  widths: {C.widths(model)}")
        # (a) advection tendency only
        with count_syncs() as log:
            _ = model.tendency(
                model.state, constraints=False,
                filter=fr.model.term_predicates.owned_by(cls))
        print(f"  advection-only Grid.sync calls: {len(log)}")
        for nm, valid in log:
            print(f"      sync field={nm!r} halo_valid={valid}")
        # (b) full step trace (includes pressure solve etc.)
        with count_syncs() as log2:
            model.run(steps=1, progress=False)
        print(f"  full-step  Grid.sync calls: {len(log2)}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    print(f"=== E2 sync behaviour, n={n}^3 ===\n")
    probe(n, "UpwindAdvection(5) baseline", C.upwind5,
          UpwindAdvection, None)
    probe(n, "UpwindAdvection(5) forced-3", C.upwind5,
          UpwindAdvection, 3)
    probe(n, "CenteredAdvection baseline", C.centered,
          CenteredAdvection, None)
    probe(n, "CenteredAdvection forced-1", C.centered,
          CenteredAdvection, 1)
