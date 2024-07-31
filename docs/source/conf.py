import os
import sys
import os
import inspect
from urllib.parse import quote

sys.path.insert(0, os.path.abspath('../../src'))

# generate the rst files
import auto_generate_api

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fridom'
copyright = '2024, Silvano Rosenau'
author = 'Silvano Rosenau'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'myst_parser',
]
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
auto_summary_generate = True
autosummary_generate_overwrite = False  # Prevent overwriting existing files
autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = []
source_suffix = '.rst'

pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/Gordi42/FRIDOM",
    "use_repository_button": True,
}
html_title = "FRIDOM - Documentation"
 

html_static_path = ['_static']
autodoc_mock_imports = [
        'numpy', 'scipy', 'mpi4py', 'IPython', 'jax', 'cupy', 'coloredlogs', 
        'netCDF4', 'matplotlib']

# default_role = 'literal'
# MyST configuration
myst_enable_extensions = [
    "deflist",  # for definition lists
    "colon_fence",  # for fenced code blocks with colons
    "html_admonition",  # for admonitions with HTML
    "html_image",  # for HTML image syntax
]

# Optional: configure other MyST options
myst_heading_anchors = 3  # create anchors for headers up to level 3
