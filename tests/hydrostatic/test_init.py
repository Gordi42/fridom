"""Test the imports of fridom/hydrostatic and its modules package."""
import pytest

import fridom.hydrostatic
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
    pytest.param(module, name, id=f"{module.__name__}.{name}")
    for module in (fridom.hydrostatic, fridom.hydrostatic.modules)
    for name in _all_imports(module)
]


@pytest.mark.parametrize(("module", "import_name"), all_imports)
def test_module_import(module, import_name):
    attr = getattr(module, import_name)
    assert attr is not None
