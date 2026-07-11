"""sphinx-gallery scraper for videos rendered by executing examples.

An example that records an animation (e.g. through the visible
``cdfviewer ... --record`` line) leaves a video file next to itself.
After every code block this scraper moves each new video from the
example's directory into the generated gallery page's ``videos/``
folder and returns the rst that embeds it with ``sphinxcontrib-video``.
The old-stack ``copy_media_files`` scraper (pre-rendered LFS media)
coexists with this one until the last old example is ported.
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


class VideoScraper:

    """Move newly created videos into the gallery and embed them."""

    def __call__(self, block, block_vars, gallery_conf):
        """Collect videos the last code block created."""
        src_dir = os.path.dirname(block_vars["src_file"])
        target_dir = os.path.dirname(block_vars["target_file"])
        rst = []
        for name in sorted(os.listdir(src_dir)):
            if not name.endswith(VIDEO_EXTENSIONS):
                continue
            video_dir = os.path.join(target_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            shutil.move(os.path.join(src_dir, name),
                        os.path.join(video_dir, name))
            rst.append(VIDEO_RST.format(name=name))
        return "\n".join(rst)
