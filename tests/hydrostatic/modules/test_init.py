"""Test the imports of the fridom/hydrostatic/modules package."""
import pytest

import fridom.hydrostatic.modules


def _names(items):
    """Yield exposed names, normalizing lazypimp ``{alias: mod}`` items."""
    for item in items:
        if isinstance(item, dict):
            yield from item.keys()
        else:
            yield item


def _all_imports(module):
    names = []
    for mod in module.all_modules_by_origin.values():
        names.extend(_names(mod))
    for mod in module.all_imports_by_origin.values():
        names.extend(_names(mod))
    return names


all_imports = [
    pytest.param(name, id=name)
    for name in _all_imports(fridom.hydrostatic.modules)
]


@pytest.mark.parametrize("import_name", all_imports)
def test_module_import(import_name):
    attr = getattr(fridom.hydrostatic.modules, import_name)
    assert attr is not None
