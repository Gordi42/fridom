"""Modules for creating animated output of the model."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from .model_plotter import ModelPlotter
    from .video_writer import VideoWriter

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.framework.modules.animation"

all_modules_by_origin = { }

all_imports_by_origin = {
    f"{base_path}.model_plotter": ["ModelPlotter"],
    f"{base_path}.video_writer": ["VideoWriter"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
