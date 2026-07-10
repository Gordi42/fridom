"""Test the imports of the fridom.nonhydro2 __init__ files."""
import pytest

import fridom.nonhydro2 as nh
import fridom.nonhydro2.modules as nh_modules


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


@pytest.mark.parametrize("import_name", _all_imports(nh))
def test_package_import(import_name):
    assert getattr(nh, import_name) is not None


@pytest.mark.parametrize("import_name", _all_imports(nh_modules))
def test_modules_import(import_name):
    assert getattr(nh_modules, import_name) is not None
