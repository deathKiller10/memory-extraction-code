"""
BAPCA -- type-aware persistent memory for LLM agents.

Two decisions, deliberately separated:
    what to KEEP     -> Ebbinghaus salience (weight x frequency x time decay)
    what to RETRIEVE -> memory type

"""

from .memory import Episode, MemoryNode, MemoryType, cosine
from .store import EpisodicStore, RetrievalConfig, Retrieved
from .embeddings import Embedder, HashEmbedder
from .cache import DiskCache

__version__ = "0.1.0"

__all__ = [
    "Episode",
    "MemoryNode",
    "MemoryType",
    "cosine",
    "EpisodicStore",
    "RetrievalConfig",
    "Retrieved",
    "Embedder",
    "HashEmbedder",
    "DiskCache",
]
