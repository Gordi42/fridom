import sys
import os
import shutil
from importlib.util import spec_from_file_location, module_from_spec

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

def create_index_rst(modpath, modname, doc_path):
    path = os.path.join(src_base_path, modpath.replace(".", "/"))
    path = os.path.join(path, modname)

    # create a index.rst file
    with open(os.path.join(doc_path, "index.rst"), "w") as f:
        module_path = os.path.join(modpath.replace(".", "/"), modname)
        import_path = module_path.replace("/", ".")

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
                    if mod_is_package(base, mod):
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

    # create a index.rst file
    with open(os.path.join(doc_path, f"{modname}.rst"), "w") as f:
        module_path = os.path.join(modpath.replace(".", "/"), modname)
        import_path = module_path.replace("/", ".")

        f.write(f".. automodule:: {import_path}\n")
        f.write("   :members:\n")
        f.write("   :undoc-members:\n")
        f.write("   :show-inheritance:\n")

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

exit()



# find all "__init__.py" files in the src directory
all_init_files = []
for root, dirs, files in os.walk(src_base_path):
    for file in files:
        if file == "__init__.py":
            all_init_files.append(os.path.join(root, file))

for filepath in all_init_files:
    path, file = os.path.split(filepath)
    # format the path to a python import
    # src/fridom/framework => fridom.framework
    import_path = path[4:].replace("/", ".")
    # remove the src_base_path from the path
    path = path[len(src_base_path)+1:]
    api_path = os.path.join(api_base_path, path)

    # create the directory
    os.makedirs(api_path, exist_ok=True)

    # create a index.rst file
    with open(os.path.join(api_path, "index.rst"), "w") as f:
        f.write(f".. automodule:: {import_path}\n")
        f.write("   :members:\n")
        f.write("   :undoc-members:\n")
        f.write("   :show-inheritance:\n")

        # import `base_mod` from the module
        spec = spec_from_file_location("__init__", filepath)
        mod = module_from_spec(spec)
        sys.modules["__init__"] = mod
        spec.loader.exec_module(mod)
        all_mods = mod.all_modules_by_origin

        f.write("\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 1\n")
        for base, mods in all_mods.items():
            print(base, mods)


        
