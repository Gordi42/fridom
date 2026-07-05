"""Test the imports in fridom/framework/domain_decomposition/__init__.py."""
import pytest

import fridom.framework.domain_decomposition as test_module

all_imports = []
for mod in test_module.all_modules_by_origin.values():
    all_imports.extend(mod)
for mod in test_module.all_imports_by_origin.values():
    all_imports.extend(mod)

@pytest.mark.parametrize("import_name", all_imports)
def test_module_import(import_name):
    assert getattr(test_module, import_name) is not None
