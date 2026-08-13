"""Test the imports in the fridom/__init__.py file."""
import os
import subprocess
import sys

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


def test_io_ops_top_level():
    # ``fr.io`` and ``fr.ops`` are real top-level packages (rehomed out
    # of ``fr.model``); their headline surfaces resolve.
    assert test_module.io.Writer is not None
    assert test_module.ops.Session is not None


# ================================================================
#  Double precision (x64)
# ================================================================
# ``import fridom`` itself must enable x64 -- not some subpackage on the
# way. jax reads the flag when an array is created, so anything built
# before it is set is silently float32, and the flag's previous home
# (``fridom/framework/__init__.py``) disappears with the cutover.
#
# Asserting this in-process would be vacuous: conftest and every other
# test file have long since imported fridom subpackages, so the flag is
# on no matter where it was set. The probe therefore runs in a fresh
# interpreter that imports nothing but ``fridom``.
_X64_PROBE = """
import fridom
import jax
import jax.numpy as jnp

# jnp.result_type(float) is exactly what fr.utils.dtype_real() returns;
# it is used here so the probe imports no fridom subpackage.
print(jax.config.jax_enable_x64, jnp.result_type(float))
"""


def test_root_import_enables_x64():
    # FRIDOM_DISABLE_COMPILE_CACHE also pins that the flag is set
    # independently of the compile-cache configuration (and keeps the
    # probe off the user's cache directory).
    env = os.environ | {"FRIDOM_DISABLE_COMPILE_CACHE": "1"}
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _X64_PROBE],
        capture_output=True, text=True, check=False, timeout=300, env=env)
    assert proc.returncode == 0, proc.stderr
    last_line = proc.stdout.strip().splitlines()[-1]
    assert last_line.split() == ["True", "float64"], proc.stdout
