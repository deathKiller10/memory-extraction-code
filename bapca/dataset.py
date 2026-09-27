"""
Loader for the LoCoMo-Plus evaluation samples.

Written against the schema confirmed by scripts/build_eval_set.py on the real
file, not against anything assumed:

    input_prompt  str   the full stitched conversation, trigger appended last
    trigger       str   the query utterance (also the last line of input_prompt)
    evidence      str   the cue turns that the response must reflect
    category      str   Cognitive | single-hop | multi-hop | temporal | ...
    answer        str   present on 1,542 of 2,387 samples -- NEVER on Cognitive
    time_gap      str   Cognitive only, e.g. "two weeks later"

The 401 Cognitive samples are our evaluation universe. They have no reference
answer by design: correctness is whether the response reflects the evidence,
judged by an LLM, which is why our judge reuses the benchmark's own rubric.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_PATH = Path("third_party/Locomo-Plus/data/unified_input_samples_v2.json")
COGNITIVE = "Cognitive"


@dataclass(frozen=True)
class Sample:
    index: int          # position in the source file: our stable sample id
    input_prompt: str
    trigger: str
    evidence: str
    category: str
    answer: Optional[str] = None
    time_gap: Optional[str] = None

    @property
    def is_cognitive(self) -> bool:
        return self.category == COGNITIVE

    def approx_tokens(self) -> int:
        return len(self.input_prompt) // 4


class LocomoPlus:
    """The evaluation set, with reproducible subsetting."""

    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found.\n"
                "Build it first:  python scripts/build_eval_set.py"
            )
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = next((v for v in raw.values() if isinstance(v, list)), [])

        self.samples: list[Sample] = [
            Sample(
                index=i,
                input_prompt=record.get("input_prompt", ""),
                trigger=record.get("trigger", ""),
                evidence=record.get("evidence", ""),
                category=record.get("category", "?"),
                answer=record.get("answer"),
                time_gap=record.get("time_gap"),
            )
            for i, record in enumerate(raw)
            if isinstance(record, dict)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def cognitive(self) -> list[Sample]:
        return [s for s in self.samples if s.is_cognitive]

    def subset(self, n: int, seed: int = 42, category: str = COGNITIVE) -> list[Sample]:
        """
        A fixed, reproducible slice.

        Free-tier quota will not cover all 401 items across every condition and
        seed, so the paper evaluates a subset. A subset chosen with a stated
        seed and reported honestly is fine; a silently chosen one is not, which
        is why the seed is a required part of the identity of any result.
        """
        pool = [s for s in self.samples if s.category == category]
        if n >= len(pool):
            return pool
        rng = random.Random(seed)
        return sorted(rng.sample(pool, n), key=lambda s: s.index)

    def stats(self) -> dict:
        cognitive = self.cognitive()
        by_category: dict[str, int] = {}
        for sample in self.samples:
            by_category[sample.category] = by_category.get(sample.category, 0) + 1
        tokens = sorted(s.approx_tokens() for s in cognitive)
        return {
            "total": len(self.samples),
            "cognitive": len(cognitive),
            "by_category": by_category,
            "cognitive_median_tokens": tokens[len(tokens) // 2] if tokens else 0,
            "cognitive_max_tokens": tokens[-1] if tokens else 0,
        }
