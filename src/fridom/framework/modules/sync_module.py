"""A module that synchronizes the model state."""
from __future__ import annotations

from functools import partial

import fridom.framework as fr


@partial(fr.utils.jaxify, dynamic=("_water_mask", ))
class SyncModule(fr.modules.Module):

    """A module that synchronizes the model state."""

    name = "Sync Module"

    def _on_setup(self) -> None:
        super()._on_setup()
        self._water_mask = self.mset.grid.water_mask

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        z = mz.z.sync()
        for f in z:
            z[f.name] = self._water_mask.apply_mask(f)

        mz.z = z
        return mz
