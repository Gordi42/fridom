"""sphinx-gallery scraper for videos rendered by executing examples.

An example that records an animation (e.g. through the visible
``cdfviewer ... --record`` line) leaves a video file next to itself.
After every code block this scraper moves each new video from the
example's directory into the generated gallery page's ``videos/``
folder and returns the rst that embeds it with ``sphinxcontrib-video``.
A block that produced several videos gets them side by side in a
``sphinx-design`` grid (up to three per row) unless the example sets
``# sphinx_gallery_video_columns = N`` (see ``video_columns``), with
``1`` stacking them full-width. The old-stack ``copy_media_files``
scraper (pre-rendered LFS media) coexists with this one until the last
old example is ported.
"""
import os
import shutil

from sphinx_gallery.py_source_parser import extract_file_config

VIDEO_EXTENSIONS = (".mp4", ".webm", ".gif")

#: videos a block lays side by side unless the example says otherwise
DEFAULT_COLUMNS = 3

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


def purge_stray_outputs(gallery_conf, fname, when):
    """Remove example byproducts around each execution (reset hook).

    Runs before AND after every example (``reset_modules_order:
    "both"``). Before: a previous manual run of the example (outside
    sphinx) leaves its rendered videos next to the script, and without
    this hook the scraper would sweep them at the example's FIRST
    block and embed every video twice. After: zarr stores written by
    the example's ``Writer`` (and any video the scraper did not move)
    would otherwise litter the source tree once the build finishes.

    sphinx-gallery passes ``fname`` as a bare filename with no
    directory, so the example directories are resolved from the
    gallery config instead. The purge must not recurse: the unported
    old-stack examples keep committed pre-rendered media in
    ``videos/`` subdirectories. It must not run per code block
    either: an example writes its zarr store in one block and renders
    it with cdfviewer in a later one.
    """
    roots = gallery_conf["examples_dirs"]
    if not isinstance(roots, list):
        roots = [roots]
    base = gallery_conf["src_dir"]
    for root in roots:
        if not os.path.isabs(root):
            root = os.path.join(base, root)
        if not os.path.isdir(root):
            continue
        subdirs = [entry.path for entry in os.scandir(root)
                   if entry.is_dir()]
        for gallery_dir in [root, *subdirs]:
            for entry in os.scandir(gallery_dir):
                if entry.is_file() and entry.name.endswith(
                        VIDEO_EXTENSIONS):
                    os.remove(entry.path)
                elif entry.is_dir() and entry.name.endswith(".zarr"):
                    shutil.rmtree(entry.path)


class VideoScraper:

    """Move newly created videos into the gallery and embed them."""

    def __call__(self, block, block_vars, gallery_conf):
        """Collect videos the last code block created."""
        src_dir = os.path.dirname(block_vars["src_file"])
        target_dir = os.path.dirname(block_vars["target_file"])
        names = []
        # in the order the block recorded them, not alphabetically
        videos = [name for name in os.listdir(src_dir)
                  if name.endswith(VIDEO_EXTENSIONS)]
        videos.sort(key=lambda name: os.path.getmtime(
            os.path.join(src_dir, name)))
        for name in videos:
            video_dir = os.path.join(target_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            shutil.move(os.path.join(src_dir, name),
                        os.path.join(video_dir, name))
            names.append(name)
        columns = min(video_columns(block, block_vars), len(names))
        if columns <= 1:
            return "".join(VIDEO_RST.format(name=name) for name in names)
        items = "".join(GRID_ITEM_RST.format(name=name) for name in names)
        return (f"\n.. grid:: 1 {min(2, columns)} {columns} {columns}"
                f"\n   :gutter: 2\n{items}")


def video_columns(block, block_vars):
    """Return the video columns an example asks for (default 3).

    The in-file config comment ``# sphinx_gallery_video_columns = N``
    sets the most videos a block lays side by side, so ``1`` stacks
    them at full width. The code block's own comment wins over one
    placed anywhere else in the example, and sphinx-gallery strips the
    comment from the rendered code like its own config comments.
    """
    content = getattr(block, "content", None)
    if content is None:
        content = block[1]
    block_conf = extract_file_config(content)
    columns = block_conf.get(
        "video_columns",
        block_vars.get("file_conf", {}).get("video_columns", DEFAULT_COLUMNS))
    if isinstance(columns, bool) or not isinstance(columns, int) \
            or columns < 1:
        msg = ("sphinx_gallery_video_columns must be a positive integer, "
               f"got {columns!r}")
        raise ValueError(msg)
    return columns
