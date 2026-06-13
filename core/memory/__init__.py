"""core/memory — Neural Memory System

Two-layer architecture:
  NeuralMemory     — conscious, fast, SQLite-backed
  ObsidianBridge   — subconscious, permanent, markdown knowledge graph
"""
from core.memory.neural_memory import NeuralMemory, MemoryNode, MemoryQuery, get_neural_memory
from core.memory.obsidian_bridge import ObsidianBridge, get_obsidian_bridge

__all__ = [
    "NeuralMemory",
    "MemoryNode",
    "MemoryQuery",
    "get_neural_memory",
    "ObsidianBridge",
    "get_obsidian_bridge",
]
