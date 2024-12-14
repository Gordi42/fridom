"""Test the imports in the fridom/framework/domain_decomposition/__init__.py file."""
import pytest

import fridom.framework.domain_decomposition as test_module

all_imports = []
for mod in test_module.all_modules_by_origin.values():
    all_imports.extend(mod)
for mod in test_module.all_imports_by_origin.values():
    all_imports.extend(mod)

@pytest.mark.parametrize("import_name", all_imports)
def test_module_import(import_name):
    # the jaxdecomp module is allowed to fail to import if jax is not installed
    try:
        attr = getattr(test_module, import_name)
    except ImportError as e:
        if import_name == "JaxDecomposition":
            return
        raise ImportError from e

    assert attr is not None
