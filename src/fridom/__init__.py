import sys
from types import ModuleType

# load submodules on demand
class _module(ModuleType):
    def __getattr__(self, name):
        __import__("fridom." + name)
        return ModuleType.__getattribute__(self, name)

sys.modules[__name__].__class__ = _module
__all__ = ["framework", "nonhydro", "shallowwater"]