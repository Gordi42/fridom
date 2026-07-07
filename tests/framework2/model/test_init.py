"""Test the imports in the framework2 model-layer __init__ files."""
import importlib

import pytest

PACKAGES = [
    "fridom.framework2.model",
    "fridom.framework2.model.time_steppers",
    "fridom.framework2.model.closures",
    "fridom.framework2.transforms",
    "fridom.framework2.io",
    "fridom.framework2.ops",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_imports(package):
    module = importlib.import_module(package)
    assert module is not None


@pytest.mark.parametrize("package", PACKAGES)
def test_lazy_reexports(package):
    """Every name in the lazypimp tables resolves."""
    module = importlib.import_module(package)
    names = []
    for mods in module.all_modules_by_origin.values():
        names.extend(mods)
    for attrs in module.all_imports_by_origin.values():
        names.extend(attrs)
    for name in names:
        assert getattr(module, name) is not None
