"""Test the imports in the fridom.io __init__ file."""
import pytest

import fridom.io as test_module

all_imports = []
for _mods in test_module.all_modules_by_origin.values():
    all_imports.extend(_mods)
for _attrs in test_module.all_imports_by_origin.values():
    all_imports.extend(_attrs)


@pytest.mark.parametrize("import_name", all_imports)
def test_module_import(import_name):
    attr = getattr(test_module, import_name)
    assert attr is not None
