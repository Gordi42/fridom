import sys
import os
from importlib.util import spec_from_file_location, module_from_spec

src_base_path = "../../src"

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
    except:
        print(f"Error importing {package}")

[open_init(file[len(src_base_path)+1:].replace("/", ".")) for file in init_files]