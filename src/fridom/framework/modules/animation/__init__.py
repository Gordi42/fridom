"""
Animation Module
================
Modules for creating animated output of the model

Classes
-------
`ModelPlotterBase`
    For creating the figure
`LiveAnimation`
    For live plotting of the model
`VideoWriter`
    For writing the animation to a video file
"""
# import classes
from .model_plotter import ModelPlotterBase
from .live_animation import LiveAnimation
from .video_writer import VideoWriter
