"""
The episodic store: write, consolidate, prune, retrieve.

Two decisions, deliberately separated -- this separation is the paper's claim:

  WHAT TO KEEP     -> Ebbinghaus salience (weight x frequency x time decay)
  WHAT TO RETRIEVE -> memory type

Facts are retrieved by similarity to the query, like any RAG system.
Standing memories (state / goal / value / constraint) are NOT, because the
benchmark we target has had cue-query similarity deliberately filtered out.
They are carried forward while their salience holds, under a fixed budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import numpy as np

from .memory import Episode, MemoryNode, MemoryType, cosine


@dataclass
class RetrievalConfig:
    """Everything the ablation study needs to switch off, in one place."""

    top_k: int = 5                    # similarity hits returned for FACT memories
    sim_floor: float = 0.25           # minimum similarity for a FACT to be returned
    dedup_threshold: float = 0.85     # pattern completion / auto-association
    prune_threshold: float = 0.20     # tau_prune; below this a memory is archived
    standing_budget: int = 8          # max standing memories carried per query
    standing_floor: float = 0.15      # min salience for a standing memory to ride along

    # --- ablation switches -------------------------------------------------
    use_type_routing: bool = True     # OFF => everything retrieved by similarity (flat RAG behaviour)
    use_decay: bool = True            # OFF => no time decay, nothing is ever pruned
    use_frequency: bool = True        # OFF => c is fixed at 1
    per_type_decay: bool = True       # OFF => one lambda for all types


@dataclass
class Retrieved:
    similarity_hits: list[MemoryNode]
    standing_hits: list[MemoryNode]

    @property
    def all(self) -> list[MemoryNode]:
        return self.standing_hits + self.similarity_hits

    def as_context(self) -> str:
        if not self.all:
            return ""
        lines = []
        if self.standing_hits:
            lines.append("What you know about this user (standing context):")
            lines += [f"- {m.as_context_line()}" for m in self.standing_hits]
        if self.similarity_hits:
            if lines:
                lines.append("")
            lines.append("Relevant things they told you earlier:")
            lines += [f"- {m.text}" for m in self.similarity_hits]
        return "\n".join(lines)

    def token_estimate(self) -> int:
        """Rough proxy for injected context cost. ~4 chars per token."""
        return len(self.as_context()) // 4


class EpisodicStore:
    """
    A list of MemoryNodes with decay-based retention and type-aware retrieval.

    Deliberately NOT a vector DB. At the scale of one LoCoMo-Plus conversation
    (a few hundred memories) a brute-force numpy scan is exact, instant, and has
    no index-build step to explain away in the paper.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.nodes: list[MemoryNode] = []
        self.archive: list[MemoryNode] = []   # pruned, not deleted -- so we can report on it
        self._next_id = 1

    # ---- write ----------------------------------------------------------

    def write(
        self,
        text: str,
        embedding: np.ndarray,
        *,
        when: datetime,
        weight: float = 0.5,
        mem_type: MemoryType = MemoryType.FACT,
        episode: Optional[Episode] = None,
        source_turn: Optional[int] = None,
    ) -> MemoryNode:
        """
        Consolidate one observation.

        If it is near-identical to something already stored (same type), that is
        pattern completion: reinforce the existing trace instead of duplicating.
        Note we keep the ORIGINAL text -- the Review 1 notebook printed the new
        text while keeping the old one, which made its logs misleading.
        """
        for node in self.nodes:
            if node.mem_type is not mem_type:
                continue
            if cosine(node.embedding, embedding) > self.config.dedup_threshold:
                node.reinforce(when)
                return node

        node = MemoryNode(
            id=self._next_id,
            text=text,
            embedding=np.asarray(embedding, dtype=np.float32),
            weight=float(np.clip(weight, 0.1, 1.0)),
            created=when,
            last_accessed=when,
            mem_type=mem_type,
            episode=episode or Episode(),
            source_turn=source_turn,
        )
        self.nodes.append(node)
        self._next_id += 1
        return node

    # ---- consolidation ---------------------------------------------------

    def prune(self, now: datetime) -> list[MemoryNode]:
        """
        Archive everything below tau_prune. Returns what was archived.

        Unlike the Review 1 notebook this is non-destructive: pruned nodes move
        to self.archive so we can report pruning behaviour in the paper and, if
        needed, audit a wrong answer against what had been dropped.
        """
        if not self.config.use_decay:
            return []

        kept, dropped = [], []
        for node in self.nodes:
            s = node.score(
                now,
                use_frequency=self.config.use_frequency,
                per_type_decay=self.config.per_type_decay,
            )
            (kept if s >= self.config.prune_threshold else dropped).append(node)

        self.nodes = kept
        self.archive.extend(dropped)
        return dropped

    # ---- retrieve --------------------------------------------------------

    def retrieve(self, query_embedding: np.ndarray, now: datetime) -> Retrieved:
        self.prune(now)

        def salience(n: MemoryNode) -> float:
            if not self.config.use_decay:
                return n.weight * (n.frequency if self.config.use_frequency else 1)
            return n.score(
                now,
                use_frequency=self.config.use_frequency,
                per_type_decay=self.config.per_type_decay,
            )

        # Ablation: with type routing off, every memory competes on similarity
        # alone. That is exactly the flat-RAG baseline, which is the point --
        # the comparison isolates our contribution rather than confounding it
        # with a different retriever or a different embedding model.
        if self.config.use_type_routing:
            standing_pool = [n for n in self.nodes if n.mem_type.is_standing]
            fact_pool = [n for n in self.nodes if not n.mem_type.is_standing]
        else:
            standing_pool, fact_pool = [], list(self.nodes)

        standing = [n for n in standing_pool if salience(n) >= self.config.standing_floor]
        standing.sort(key=salience, reverse=True)
        standing = standing[: self.config.standing_budget]

        scored = [(cosine(query_embedding, n.embedding), n) for n in fact_pool]
        scored = [(s, n) for s, n in scored if s >= self.config.sim_floor]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        hits = [n for _, n in scored[: self.config.top_k]]

        return Retrieved(similarity_hits=hits, standing_hits=standing)

    # ---- feedback --------------------------------------------------------

    def reinforce_used(
        self,
        used: Iterable[MemoryNode],
        now: datetime,
        *,
        correct: bool,
        weight_bump: float = 0.05,
    ) -> None:
        """
        Correctness-gated salience update (the ERM idea, simplified).

        A memory that was in context when the judge said "correct" gets its
        weight and frequency bumped; on an incorrect answer we leave it alone
        and let ordinary decay handle it. We do NOT punish on failure -- with a
        noisy LLM judge that would amplify judge error into the memory state.
        """
        if not correct:
            return
        for node in used:
            node.reinforce(now, weight_bump=weight_bump)

    # ---- reporting -------------------------------------------------------

    def stats(self, now: datetime) -> dict:
        by_type: dict[str, int] = {}
        for n in self.nodes:
            by_type[n.mem_type.value] = by_type.get(n.mem_type.value, 0) + 1
        return {
            "retained": len(self.nodes),
            "archived": len(self.archive),
            "total_written": len(self.nodes) + len(self.archive),
            "by_type": by_type,
            "mean_salience": (
                float(np.mean([n.score(now) for n in self.nodes])) if self.nodes else 0.0
            ),
        }
