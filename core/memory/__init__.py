"""core/memory — Memory System

Two-layer architecture:
  NeuralMemory       — fast, SQLite-backed (Pillar 2a)
  ObsidianBridge     — permanent, markdown knowledge graph (Pillar 2b)
  ConversationMemory — conversation history store (kept for backwards-compat)
"""
from core.memory.neural_memory import NeuralMemory, MemoryNode, MemoryQuery, get_neural_memory
from core.memory.obsidian_bridge import ObsidianBridge, get_obsidian_bridge

# Backwards-compatibility: agent.py and web.py import ConversationMemory from core.memory
import os as _os
import importlib.util as _ilu

_mem_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "memory.py")
_spec = _ilu.spec_from_file_location("core._memory_compat", _mem_path)
_mem_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mem_mod)
ConversationMemory = _mem_mod.ConversationMemory
Message = _mem_mod.Message

__all__ = [
    "NeuralMemory", "MemoryNode", "MemoryQuery", "get_neural_memory",
    "ObsidianBridge", "get_obsidian_bridge",
    "ConversationMemory", "Message",
]
