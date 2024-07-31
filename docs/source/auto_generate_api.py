import sys
import os
import shutil
from importlib.util import spec_from_file_location, module_from_spec
import ast
from enum import Enum, auto

api_base_path = "reference"
src_base_path = "../../src"

# remove the api directory if it exists
shutil.rmtree(api_base_path, ignore_errors=True)

# create the api directory
os.makedirs(api_base_path, exist_ok=True)

# find all __init__.py files
init_files = []
for root, dirs, files in os.walk(src_base_path):
    for file in files:
        if file == "__init__.py":
            init_files.append(root)
    
def open_init(package: str):
    try:
        path = os.path.join(src_base_path, package.replace(".", "/"))
        file_path = os.path.join(path, "__init__.py")
        spec = spec_from_file_location("__init__", file_path)
        mod = module_from_spec(spec)
        sys.modules["__init__"] = mod
        spec.loader.exec_module(mod)
        all_mods = mod.all_modules_by_origin
        all_imports = mod.all_imports_by_origin
    except:
        print(f"Error importing {package}")
        all_mods = {}
        all_imports = {}
    return all_mods, all_imports

def create_package_rst(file):
    rst = file[len(src_base_path)+1:].replace("/", ".")
    f = open(os.path.join(api_base_path, rst + ".rst"), "w")

    f.write(f"{rst}\n")
    f.write("=" * len(rst) + "\n\n")

    f.write(".. currentmodule:: " + rst + "\n\n")

    f.write("Module Documentation\n")
    f.write("--------------------\n")
    f.write(".. automodule:: " + rst + "\n")
    f.write("    :no-members:\n\n")

    all_mods, all_imports = open_init(rst)
    if all_mods:
        f.write("Submodules\n")
        f.write("----------\n")
        f.write(".. autosummary::\n")
        f.write("    :toctree:\n\n")
        for base, mods in all_mods.items():
            for mod in mods:
                f.write(f"    {mod}\n")
        f.write("\n")

    if all_imports:
        f.write("Classes, Functions, and Variables\n")
        f.write("---------------------------------\n")

        f.write(".. autosummary::\n")
        f.write("    :toctree:\n\n")
        for base, imports in all_imports.items():
            for imp in imports:
                f.write(f"    {imp}\n")


for init_file in init_files:
    create_package_rst(init_file)
