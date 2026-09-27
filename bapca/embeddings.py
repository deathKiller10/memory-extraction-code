"""
Sentence embeddings -- the thing Review 1 did not have.

The notebook used three hand-typed 3-dimensional vectors:
    [0.9, 0.1, 0.1]  [0.1, 0.9, 0.1]  [0.1, 0.1, 0.9]
which is why its "cue-trigger semantic disconnect" demo actually had a
cosine similarity of 0.994 between query and answer. Real embeddings are the
first thing that makes the evaluation mean anything.

all-MiniLM-L6-v2: 384 dimensions, ~80MB, runs on CPU, free forever, no API key.
Embeddings are cached to disk like everything else -- re-embedding the same
corpus on every run is slow, and slow means we stop running experiments.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import numpy as np

from .cache import DiskCache

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache = DiskCache(f"embeddings/{model_name.split('/')[-1]}")
        self._model = None  # loaded lazily: importing torch takes a few seconds

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}||{text}".encode("utf-8")).hexdigest()

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Embed a batch, hitting the model only for texts we have not seen."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        vectors: list[Optional[np.ndarray]] = []
        missing_idx: list[int] = []

        for i, text in enumerate(texts):
            cached = self.cache.get(self._key(text))
            if cached is None:
                vectors.append(None)
                missing_idx.append(i)
            else:
                vectors.append(np.asarray(cached, dtype=np.float32))

        if missing_idx:
            model = self._load()
            fresh = model.encode(
                [texts[i] for i in missing_idx],
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            for slot, vector in zip(missing_idx, np.asarray(fresh, dtype=np.float32)):
                vectors[slot] = vector
                self.cache.put(self._key(texts[slot]), vector.tolist())

        return np.vstack([v for v in vectors if v is not None])

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class HashEmbedder:
    """
    Deterministic fake embedder for unit tests and offline development.

    Never use this for results -- it has no semantics. It exists so the test
    suite runs in a second with no model download and no network.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode_one(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self.dim).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([self.encode_one(t) for t in texts])
