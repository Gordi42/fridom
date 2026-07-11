"""Test the imports in the fridom/model/modules/__init__.py file."""
import pytest

import fridom.model.modules as test_module


def _names(items):
    """Yield exposed names, normalizing lazypimp ``{alias: mod}`` items."""
    for item in items:
        if isinstance(item, dict):
            yield from item.keys()
        else:
            yield item


all_imports = []
for mod in test_module.all_modules_by_origin.values():
    all_imports.extend(_names(mod))
for mod in test_module.all_imports_by_origin.values():
    all_imports.extend(_names(mod))


@pytest.mark.parametrize("import_name", all_imports)
def test_module_import(import_name):
    attr = getattr(test_module, import_name)
    assert attr is not None
