"""
Modules for creating animated output of the model
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import classes
    from .model_plotter import ModelPlotter
    from .live_animation import LiveAnimation
    from .video_writer import VideoWriter
    
# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.framework.modules.animation"

all_modules_by_origin = { }

all_imports_by_origin = { 
    f"{base_path}.model_plotter": ["ModelPlotter"],
    f"{base_path}.live_animation": ["LiveAnimation"],
    f"{base_path}.video_writer": ["VideoWriter"], 
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
