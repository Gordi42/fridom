import sys
import os
import shutil
from importlib.util import spec_from_file_location, module_from_spec
import ast
from enum import Enum, auto

api_base_path = "docs/source/api"
src_base_path = "src"

# change working directory to the root of the project
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# remove the api directory if it exists
shutil.rmtree(api_base_path, ignore_errors=True)

def open_init(path: str):
    try:
        file_path = os.path.join(path, "__init__.py")
        spec = spec_from_file_location("__init__", file_path)
        mod = module_from_spec(spec)
        sys.modules["__init__"] = mod
        spec.loader.exec_module(mod)
        all_mods = mod.all_modules_by_origin
        all_imports = mod.all_imports_by_origin
    except:
        print(f"Error importing {path}")
        all_mods = {}
        all_imports = {}
    return all_mods, all_imports

def mod_is_package(modpath: str, modname: str):
    path = os.path.join(src_base_path, modpath.replace(".", "/"))
    path = os.path.join(path, modname)
    return os.path.exists(os.path.join(path, "__init__.py"))

class ImportType(Enum):
    MODULE = auto()
    PACKAGE = auto()
    CLASS = auto()
    FUNCTION = auto()
    VARIABLE = auto()

def get_import_type(modpath: str, impname: str):
    file_path = os.path.join(src_base_path, modpath.replace(".", "/"))

    full_path = os.path.join(file_path, impname)
    if os.path.exists(os.path.join(full_path, "__init__.py")):
        return ImportType.PACKAGE
    
    if os.path.exists(full_path + ".py"):
        return ImportType.MODULE

    if not os.path.exists(file_path + ".py"):
        print(f"Error: {file_path}.py does not exist")
        return ImportType.MODULE

    with open(file_path + ".py") as f:
        source = f.read()

    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == impname:
            return ImportType.CLASS
        elif isinstance(node, ast.FunctionDef) and node.name == impname:
            return ImportType.FUNCTION
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == impname:
                    return ImportType.VARIABLE
    
    print(f"Error: {impname} not found in {file_path}.py")
    return ImportType.MODULE

def create_index_rst(modpath, modname, doc_path):
    path = os.path.join(src_base_path, modpath.replace(".", "/"))
    path = os.path.join(path, modname)

    # create a index.rst file
    with open(os.path.join(doc_path, "index.rst"), "w") as f:
        module_path = os.path.join(modpath.replace(".", "/"), modname)
        import_path = module_path.replace("/", ".")

        f.write(f"{modname}\n")
        f.write("="*len(modname))
        f.write("\n")
        f.write("\n")

        f.write(f".. automodule:: {import_path}\n")
        f.write("   :members:\n")
        f.write("   :undoc-members:\n")
        f.write("   :show-inheritance:\n")

        # import `base_mod` from the module
        all_mods, all_imports = open_init(path)

        # check if all_mods is not empty
        if all_mods:
            f.write("\n")
            f.write(".. toctree::\n")
            f.write("   :caption: Submodules:\n")
            f.write("   :maxdepth: 1\n")
            f.write("\n")
            for base, mods in all_mods.items():
                for mod in mods:
                    if get_import_type(base, mod) == ImportType.PACKAGE:
                        f.write(f"   {mod}/index\n")
                        create_module_rst(base, mod, os.path.join(doc_path, mod))
                    else:
                        f.write(f"   {mod}\n")
                        create_module_rst(base, mod, doc_path)

        # check if all_imports is not empty
        if all_imports:
            f.write("\n")
            f.write(".. toctree::\n")
            f.write("   :caption: Imports:\n")
            f.write("   :maxdepth: 1\n")
            f.write("\n")
            for base, imports in all_imports.items():
                for imp in imports:
                    f.write(f"   {imp}\n")
                    create_module_rst(base, imp, doc_path)

def create_import_rst(modpath, modname, doc_path):
    path = os.path.join(src_base_path, modpath.replace(".", "/"))
    path = os.path.join(path, modname)
    module_path = os.path.join(modpath.replace(".", "/"), modname)
    import_path = module_path.replace("/", ".")
    import_type = get_import_type(modpath, modname)

    f = open(os.path.join(doc_path, f"{modname}.rst"), "w")
    f.write(f"{modname}\n")
    f.write("="*len(modname))
    f.write("\n")
    f.write("\n")

    if import_type == ImportType.MODULE:
        f.write(f".. automodule:: {import_path}\n")
        f.write("   :members:\n")
        f.write("   :undoc-members:\n")
        f.write("   :show-inheritance:\n")
    elif import_type == ImportType.CLASS:
        f.write(f".. autoclass:: {import_path}\n")
        f.write("   :members:\n")
        f.write("   :undoc-members:\n")
        f.write("   :show-inheritance:\n")
    elif import_type == ImportType.FUNCTION:
        f.write(f".. autofunction:: {import_path}\n")
    elif import_type == ImportType.VARIABLE:
        f.write(f".. autodata:: {import_path}\n")

def create_module_rst(modpath: str, modname: str, doc_path: str):
    os.makedirs(doc_path, exist_ok=True)
    # first check if the file is a module or a package (i.e. has a __init__.py file)
    path = os.path.join(src_base_path, modpath.replace(".", "/"))
    path = os.path.join(path, modname)
    is_package = os.path.exists(os.path.join(path, "__init__.py"))

    if is_package:
        create_index_rst(modpath, modname, doc_path)
    else:
        create_import_rst(modpath, modname, doc_path)


create_module_rst("", "fridom", api_base_path)