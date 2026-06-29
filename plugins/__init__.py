"""Plugin system - auto-discovers and loads plugins from plugins/ directory"""

import logging
logger = logging.getLogger(__name__)

import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any

PLUGINS_DIR = Path(__file__).parent


class Plugin:
    name: str = ""
    description: str = ""

    def on_load(self, agent: Any):
        pass

    def get_tools(self) -> list[Any]:
        return []

    def get_commands(self) -> list[dict]:
        return []


class PluginRegistry:
    def __init__(self):
        self.plugins: list[Plugin] = []

    def discover_and_load(self, agent: Any):
        self.plugins = []
        plugin_dirs = [PLUGINS_DIR, PLUGINS_DIR / "examples"]

        for pdir in plugin_dirs:
            if not pdir.exists():
                continue
            for f in sorted(pdir.glob("*.py")):
                if f.name.startswith("_"):
                    continue
                self._load_plugin(f, agent)

        logger.info(f"[plugins] Loaded {len(self.plugins)} plugin(s)")

    def _load_plugin(self, filepath: Path, agent: Any):
        try:
            module_name = f"plugins.{filepath.stem}"
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if not spec or not spec.loader:
                return
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and issubclass(obj, Plugin)
                        and obj is not Plugin):
                    instance = obj()
                    instance.on_load(agent)
                    self.plugins.append(instance)
                    name_str = instance.name or filepath.stem
                    logger.info(f"[plugins] Loaded: {name_str}")

        except Exception as e:
            logger.error(f"[plugins] Failed to load {filepath.name}: {e}")
