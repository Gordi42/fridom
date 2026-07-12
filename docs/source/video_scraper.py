"""sphinx-gallery scraper for videos rendered by executing examples.

An example that records an animation (e.g. through the visible
``cdfviewer ... --record`` line) leaves a video file next to itself.
After every code block this scraper moves each new video from the
example's directory into the generated gallery page's ``videos/``
folder and returns the rst that embeds it with ``sphinxcontrib-video``.
A block that produced several videos gets them side by side in a
``sphinx-design`` grid (up to three per row) instead of stacked
full-width. The old-stack ``copy_media_files`` scraper (pre-rendered
LFS media) coexists with this one until the last old example is
ported.
"""
import os
import shutil

VIDEO_EXTENSIONS = (".mp4", ".webm", ".gif")

VIDEO_RST = """
.. video:: videos/{name}
   :loop:
   :autoplay:
   :muted:
   :width: 100%
"""

GRID_ITEM_RST = """
   .. grid-item::

      .. video:: videos/{name}
         :loop:
         :autoplay:
         :muted:
         :width: 100%
"""


def purge_stray_videos(gallery_conf, fname):
    """Remove leftover videos before an example executes (reset hook).

    A previous manual run of the example (outside sphinx) leaves its
    rendered videos next to the script. Without this hook the scraper
    would sweep them at the example's FIRST block and embed every
    video twice: once at the top, once at its proper block.
    """
    # sphinx-gallery also invokes reset hooks without a concrete
    # example file (gallery setup/teardown); nothing to purge then
    src_dir = os.path.dirname(fname) if fname else ""
    if not src_dir or not os.path.isdir(src_dir):
        return
    for name in os.listdir(src_dir):
        if name.endswith(VIDEO_EXTENSIONS):
            os.remove(os.path.join(src_dir, name))


class VideoScraper:

    """Move newly created videos into the gallery and embed them."""

    def __call__(self, block, block_vars, gallery_conf):
        """Collect videos the last code block created."""
        src_dir = os.path.dirname(block_vars["src_file"])
        target_dir = os.path.dirname(block_vars["target_file"])
        names = []
        for name in sorted(os.listdir(src_dir)):
            if not name.endswith(VIDEO_EXTENSIONS):
                continue
            video_dir = os.path.join(target_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            shutil.move(os.path.join(src_dir, name),
                        os.path.join(video_dir, name))
            names.append(name)
        if len(names) <= 1:
            return "".join(VIDEO_RST.format(name=name) for name in names)
        columns = min(3, len(names))
        items = "".join(GRID_ITEM_RST.format(name=name) for name in names)
        return (f"\n.. grid:: 1 {min(2, columns)} {columns} {columns}"
                f"\n   :gutter: 2\n{items}")
