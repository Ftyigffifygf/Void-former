"""Pytest configuration to alias top-level modules under 'voidformer' namespace."""

import sys
import types
from pathlib import Path

repo_root = Path(__file__).parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

class VoidFormerFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname == "voidformer":
            from importlib.machinery import ModuleSpec
            from importlib.abc import Loader

            class VoidFormerLoader(Loader):
                def create_module(self, spec):
                    mod = types.ModuleType("voidformer")
                    mod.__path__ = [str(repo_root)]
                    return mod

                def exec_module(self, module):
                    pass

            return ModuleSpec("voidformer", VoidFormerLoader(), is_package=True)
        return None

sys.meta_path.insert(0, VoidFormerFinder())
