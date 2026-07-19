"""Test the imports in the fridom/__init__.py file."""
import pytest

import fridom as test_module

all_imports = []
for mod in test_module.all_modules_by_origin.values():
    all_imports.extend(mod)
for mod in test_module.all_imports_by_origin.values():
    all_imports.extend(mod)

@pytest.mark.parametrize("import_name", all_imports)
def test_module_import(import_name):
    attr = getattr(test_module, import_name)
    assert attr is not None


def test_io_root_alias():
    # ``fr.io`` is a lazy root alias for ``fridom.model.io``; both
    # spellings must resolve to the same module.
    assert test_module.io is test_module.model.io
    assert test_module.io.Writer is test_module.model.io.Writer
