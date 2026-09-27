"""
Memory nodes and the Ebbinghaus salience score.

This is the part of the Review 1 notebook that was actually correct, moved into
a real module and extended with the episodic 4-tuple (Huet et al., 2025) and a
memory *type* (Li et al., LoCoMo-Plus 2026).

The type is the core of our contribution: it decides HOW a memory is retrieved,
separately from the salience score, which decides WHETHER it is kept at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np


class MemoryType(str, Enum):
    """
    How a memory is allowed to be retrieved.

    FACT is the classic RAG case: look it up when the query looks like it.

    The other four are "standing" memories -- latent constraints on future
    behaviour. LoCoMo-Plus builds its benchmark by deliberately REMOVING
    lexical and semantic overlap between the cue and the later query
    (their section 4.4: BM25 + MPNet filtering). So a standing memory
    provably cannot be found by query similarity. It has to be carried.
    """

    FACT = "fact"            # "I bought the Skinn Amalfi Bleu perfume."
    STATE = "state"          # "I've been anxious about money lately."
    GOAL = "goal"            # "I'm preparing for an important exam."
    VALUE = "value"          # "I don't eat meat."
    CONSTRAINT = "constraint"  # "I want to minimise distractions until June."

    @property
    def is_standing(self) -> bool:
        return self is not MemoryType.FACT


# Per-type decay rates (lambda, per day).
#
# Biological motivation: episodic detail fades faster than the dispositions and
# goals it taught you. Behaviourally this is also what we need -- a constraint
# that decays as fast as a shopping fact is useless a month later.
#
# These are the defaults; every experiment sweeps them, and the ablation
# "--no-type-decay" collapses them all to DEFAULT_LAMBDA.
DEFAULT_LAMBDA = 0.03
TYPE_LAMBDA: dict[MemoryType, float] = {
    MemoryType.FACT: 0.03,
    MemoryType.STATE: 0.02,
    MemoryType.GOAL: 0.01,
    MemoryType.VALUE: 0.005,
    MemoryType.CONSTRAINT: 0.01,
}


@dataclass
class Episode:
    """The 4-tuple of Huet et al. (2025). Any field may be None."""

    time: Optional[str] = None      # t   - when it happened
    space: Optional[str] = None     # s   - where
    entity: Optional[str] = None    # ent - who/what it is about
    content: Optional[str] = None   # c   - what happened
    detail: Optional[str] = None    # d   - distinguishing detail

    def is_empty(self) -> bool:
        return not any(asdict(self).values())


@dataclass
class MemoryNode:
    """One memory. `text` is the surface form; everything else is bookkeeping."""

    id: int
    text: str
    embedding: np.ndarray
    weight: float                       # w0, base semantic importance in [0.1, 1.0]
    created: datetime
    last_accessed: datetime
    mem_type: MemoryType = MemoryType.FACT
    frequency: int = 1                  # c, times restated or successfully recalled
    episode: Episode = field(default_factory=Episode)
    source_turn: Optional[int] = None    # index into the source dialogue, for auditing
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- salience -------------------------------------------------------

    def lambda_for(self, per_type_decay: bool = True) -> float:
        if not per_type_decay:
            return DEFAULT_LAMBDA
        return TYPE_LAMBDA.get(self.mem_type, DEFAULT_LAMBDA)

    def score(
        self,
        now: datetime,
        *,
        use_frequency: bool = True,
        per_type_decay: bool = True,
    ) -> float:
        """
        Ebbinghaus salience:  S = w * c * exp(-lambda * dt),  dt in days.

        `use_frequency=False` and `per_type_decay=False` exist so the ablation
        study can switch each term off without touching a second code path.
        """
        dt_days = max(0.0, (now - self.last_accessed).total_seconds() / 86400.0)
        c = float(self.frequency) if use_frequency else 1.0
        return self.weight * c * math.exp(-self.lambda_for(per_type_decay) * dt_days)

    def reinforce(self, now: datetime, weight_bump: float = 0.0) -> None:
        """Pattern completion / successful recall: bump c, reset the clock."""
        self.frequency += 1
        self.last_accessed = now
        if weight_bump:
            self.weight = float(np.clip(self.weight + weight_bump, 0.1, 1.0))

    def age_days(self, now: datetime) -> float:
        return (now - self.last_accessed).total_seconds() / 86400.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("embedding")
        d["mem_type"] = self.mem_type.value
        d["created"] = self.created.isoformat()
        d["last_accessed"] = self.last_accessed.isoformat()
        return d

    def as_context_line(self) -> str:
        """How this memory is rendered into the LLM prompt."""
        if self.mem_type.is_standing:
            return f"[{self.mem_type.value}] {self.text}"
        return self.text


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
