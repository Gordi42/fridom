import re
from textwrap import indent
from sphinx_gallery.scrapers import figure_rst
from sphinx_gallery.utils import optipng
from pathlib import PurePosixPath




# The following strings are used when we have several pictures: we use
# an html div tag that our CSS uses to turn the lists into horizontal
# lists.
HLIST_HEADER = """
.. rst-class:: sphx-glr-horizontal

"""

HLIST_IMAGE_MATPLOTLIB = """
    *
"""

HLIST_SG_TEMPLATE = """
    *

      .. image-sg:: /%s
          :alt: %s
          :srcset: %s
          :class: sphx-glr-multi-img
"""

SG_IMAGE = """
.. image-sg:: /%s
   :alt: %s
   :srcset: %s
   :class: sphx-glr-single-img
"""

# keep around for back-compat:
SINGLE_IMAGE = """
 .. image:: /%s
     :alt: %s
     :class: sphx-glr-single-img
"""



def pil_scraper(block, block_vars, gallery_conf, **kwargs):
    """Scrape Matplotlib images.

    Parameters
    ----------
    block : sphinx_gallery.py_source_parser.Block
        The code block to be executed. Format (label, content, lineno).
    block_vars : dict
        Dict of block variables.
    gallery_conf : dict
        Contains the configuration of Sphinx-Gallery
    **kwargs : dict
        Additional keyword arguments to pass to
        :meth:`~matplotlib.figure.Figure.savefig`, e.g. ``format='svg'``. The
        ``format`` keyword argument in particular is used to set the file
        extension of the output file (currently only 'png', 'jpg', 'svg',
        'gif', and 'webp' are supported).

        This is not used internally, but intended for use when overriding the scraper.

    Returns
    -------
    rst : str
        The reStructuredText that will be rendered to HTML containing
        the images. This is often produced by :func:`figure_rst`.
    """
    from PIL import Image
    image_path_iterator = block_vars["image_path_iterator"]
    image_rsts = []
    srcset = gallery_conf["image_srcset"]

    env_vars = block_vars['example_globals']
    images = []
    for key, value in env_vars.items():
        if isinstance(value, Image.Image):
            images.append(value)

    for img, image_path in zip(images, image_path_iterator):
        image_path = PurePosixPath(image_path)

        if "format" in kwargs:
            image_path = image_path.with_suffix("." + kwargs["format"])

        # save the figures, and populate the srcsetpaths
        img.save(image_path)
        srcsetpaths = {0: image_path}
        srcsetpaths = [srcsetpaths]

        image_rsts.append(
            figure_rst(
                [image_path],
                gallery_conf["src_dir"],
                "",
                srcsetpaths=srcsetpaths,
            )
        )

    rst = ""
    if len(image_rsts) == 1:
        rst = image_rsts[0]
    elif len(image_rsts) > 1:
        image_rsts = [
            re.sub(r":class: sphx-glr-single-img", ":class: sphx-glr-multi-img", image)
            for image in image_rsts
        ]
        image_rsts = [
            HLIST_IMAGE_MATPLOTLIB + indent(image, " " * 6) for image in image_rsts
        ]
        rst = HLIST_HEADER + "".join(image_rsts)
    return rst