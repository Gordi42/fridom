import os
import sys
import os
import inspect
import shutil
from urllib.parse import quote
from unittest.mock import patch, MagicMock
from jinja2.filters import FILTERS
from importlib.util import spec_from_file_location, module_from_spec

src_base_path = "../../src"

sys.path.insert(0, os.path.abspath(src_base_path))
sys.path.insert(0, os.path.abspath('../..'))


# generate the rst files
# import auto_examples
import load_modules
shutil.rmtree("auto_api", ignore_errors=True)

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
    'sphinx_gallery.gen_gallery',
    'sphinxcontrib.youtube',
    'myst_parser',
]
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'download_all_examples': False,
    'plot_gallery': 'False',
    'show_signature': False,
    "notebook_extensions": {},
    "default_thumb_file": "_static/fridom-title.png",
    'backreferences_dir'  : 'gen_modules/backreferences',
    'doc_module'          : ('fridom', ),
}

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
html_logo = "_static/fridom-title.png"
html_theme_options = {
    "repository_url": "https://github.com/Gordi42/FRIDOM",
    "use_repository_button": True,
    "logo": {
        "image_dark": "_static/fridom-title-dark.png",
    }
}
html_title = "FRIDOM - Documentation"
 

html_static_path = ['_static']
autodoc_mock_imports = [
        'numpy', 'scipy', 'mpi4py', 'IPython', 'jax', 'cupy', 'coloredlogs', 
        'netCDF4', 'matplotlib', 'lazypimp']

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


# Create custom Jinja2 filters
mocks = {mock: MagicMock() for mock in autodoc_mock_imports}

def is_init(fullname):
    with patch.dict('sys.modules', mocks):
        try:
            # Get the module object
            module = __import__(fullname, fromlist=[''])
            # Get the file path of the module
            module_file = inspect.getfile(module)
            # Check if the file is an __init__.py
            return os.path.basename(module_file) == '__init__.py'
        except ImportError:
            return False

def item_name(fullname):
    return fullname.split('.')[-1]

def get_submodules(fullname):
    with patch.dict('sys.modules', mocks):
        try:
            # Get the module object
            module = __import__(fullname, fromlist=[''])
            all_imports_dict = module.all_modules_by_origin
            all_imports_list = []
            for base, imports in all_imports_dict.items():
                all_imports_list.extend(base + "." + imp for imp in imports)
            return all_imports_list
        except ImportError:
            return []

def can_import_from_package(base, imp):
    try:
        modname = ".".join(base.split(".")[:-1])
        mod = __import__(modname, fromlist=[''])
        cls = getattr(mod, imp)
        return True
    except:
        return False

def can_import(base, imp):
    try:
        mod = __import__(base, fromlist=[''])
        cls = getattr(mod, imp)
        return True
    except:
        return False

def get_imports(fullname):
    with patch.dict('sys.modules', mocks):
        try:
            # Get the module object
            module = __import__(fullname, fromlist=[''])
            all_imports_dict = module.all_imports_by_origin
            all_imports_list = []
            for base, imports in all_imports_dict.items():
                if not can_import(base, imports[0]):
                    base = ".".join(base.split(".")[:-1])
                # if can_import_from_package(base, imports[0]):
                #     base = ".".join(base.split(".")[:-1])
                all_imports_list.extend(base + "." + imp for imp in imports)
            print(fullname, all_imports_list)
            return all_imports_list
        except ImportError:
            return []

def shorten(fullname):
    base, imp = fullname.rsplit(".", 1)
    if can_import_from_package(base, imp):
        base = ".".join(base.split(".")[:-1])
    return base + "." + imp

def doc_summary_module(fullname):
    with patch.dict('sys.modules', mocks):
        try:
            # Get the module object
            module = __import__(fullname, fromlist=[''])
            doc = module.__doc__
            if doc is None:
                return ""
            for line in doc.split("\n"):
                if line.strip():
                    return line.strip()
        except:
            return ""

import importlib
def doc_summary_import(fullname):
    with patch.dict('sys.modules', mocks):
        try:
            mod_name = ".".join(fullname.split(".")[:-1])
            mod = importlib.import_module(mod_name)
            doc = inspect.getdoc(getattr(mod, fullname.split(".")[-1]))
            if doc is None:
                return ""
            for line in doc.split("\n"):
                if line.strip():
                    return line.strip()
        except Exception as e:
            print(f"Error importing {fullname}")
            print(e)
            return ""

def split_by_parent(fullnames: list):
    bases, imps = zip(*[imp.rsplit(".", 1) for imp in fullnames])
    c = {}
    # Iterate over both lists simultaneously to sort the imports by their base
    for key, value in zip(bases, imps):
        # If the key already exists in the dictionary, append the value to the list
        if key in c:
            c[key].append(value)
        # If the key doesn't exist, create a new list with the current value
        else:
            c[key] = [value]

    # Convert the dictionary to a list of tuples
    vars = []
    for key, value in c.items():
        vars.append((key, value))
    return vars

FILTERS['is_init'] = is_init
FILTERS['item_name'] = item_name
FILTERS['get_submodules'] = get_submodules
FILTERS['get_imports'] = get_imports
FILTERS['doc_summary_module'] = doc_summary_module
FILTERS['doc_summary_import'] = doc_summary_import
FILTERS['split_by_parent'] = split_by_parent
FILTERS['shorten'] = shorten