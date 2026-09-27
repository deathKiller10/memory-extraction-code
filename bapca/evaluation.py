"""
Conditions, response generation, and judging.

THE PILOT'S QUESTION
--------------------
Every system on LoCoMo-Plus Cognitive scores 15-26%, including full-context
models that do no retrieval at all and have the cue sitting in the prompt. So
which is broken: FINDING the constraint, or APPLYING it?

Four conditions answer that, on identical items:

    A  no_memory          trigger alone                    floor
    B  full_context       the benchmark's own prompt       the comparable number
    C  oracle_cue         trigger + the cue verbatim       perfect retrieval
    D  oracle_constraint  trigger + the cue restated as
                          an explicit standing constraint  perfect retrieval
                                                           AND perfect salience

  C == B  -> retrieval is not the bottleneck. Having the cue is not enough.
  D  > C  -> making the constraint explicit is what helps. That is our thesis,
             and the extraction step is where our contribution lives.
  C  > B  -> retrieval IS the bottleneck after all, and the original plan
             (better retrieval) was right. We would need to know that.

C and D are oracles: they are handed the correct cue. They are not our system,
they are the ceiling our system would be aiming at, measured before we spend
three weeks building toward it.

NO TASK DISCLOSURE
------------------
LoCoMo-Plus section 5.3 presents queries as natural continuations of the
dialogue, with no hint that memory is being tested -- telling the model "this
is a memory task" changes its behaviour and breaks comparability. Our system
prompt says nothing about memory, recall, or evaluation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Callable, Optional

from .dataset import Sample
from .llm import LLM

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

ASSISTANT_SYSTEM = (
    "You are a helpful conversational assistant talking with a friend. "
    "Reply naturally to their latest message in two or three sentences."
)


def prompt_no_memory(sample: Sample) -> str:
    return sample.trigger


def prompt_full_context(sample: Sample) -> str:
    """Exactly what the benchmark evaluates. Do not reformat it."""
    return sample.input_prompt


def prompt_oracle_cue(sample: Sample) -> str:
    """Perfect retrieval: the cue turns, verbatim, as prior conversation."""
    return (
        "Earlier in your conversation:\n"
        f"{sample.evidence}\n\n"
        f"{sample.trigger}"
    )


def prompt_oracle_constraint(sample: Sample, constraint: str) -> str:
    """Perfect retrieval AND perfect salience: the cue restated as a standing fact."""
    return (
        "What you know about this person:\n"
        f"- {constraint}\n\n"
        f"{sample.trigger}"
    )


# ---------------------------------------------------------------------------
# Constraint extraction -- the step our system would perform at write time
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "You turn conversational snippets into short standing notes about a person."
)

EXTRACT_PROMPT = """From the conversation snippet below, write ONE short sentence \
capturing what it implies about this person that would still matter weeks later \
- a goal, value, ongoing state, or constraint on how they want to live.

Write it as a standing note about them, in the third person. Do not mention the \
conversation, and do not add advice. If the snippet implies nothing lasting, \
write exactly: NONE

Snippet:
{evidence}

Standing note:"""


def extract_constraint(llm: LLM, sample: Sample) -> str:
    """
    Distil the cue into a standing note.

    Run on the judge's provider: the prompt is tiny, and it keeps the
    generator's larger daily quota for the calls that actually need it.
    """
    text = llm.complete(
        EXTRACT_PROMPT.format(evidence=sample.evidence),
        system=EXTRACT_SYSTEM,
        max_tokens=160,   # a one-sentence note; the judge provider is TPM-bound
    ).strip()
    text = text.strip('"').strip()
    return "" if text.upper().startswith("NONE") else text


# ---------------------------------------------------------------------------
# Judging -- LoCoMo-Plus's own rubric, reused verbatim for comparability
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a Memory Awareness Judge. Your task: Judge whether the Model "
    "Prediction considers or is linked to the Evidence."
)

JUDGE_PROMPT = """Evidence:
{evidence}

User message:
{trigger}

Model Prediction:
{prediction}

Scoring:
- "correct": the Prediction explicitly or implicitly reflects or uses the \
Evidence (the memory or constraint).
- "wrong": the Prediction does not show such a link to the Evidence.

Respond with JSON only:
{{"label": "correct" | "wrong", "reason": "<does the prediction relate to the evidence?>"}}"""

_JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)

# Truncation-proof fallback: the label is readable long before the JSON closes.
# '{"label": "correct", "reason": "The prediction acknowledges her...' has no
# closing brace, but the verdict in it is unambiguous.
_LABEL_ONLY = re.compile(r'["\']?label["\']?\s*[:=]\s*["\']?(correct|wrong)\b',
                         re.IGNORECASE)


@dataclass
class Verdict:
    label: str          # "correct" | "wrong" | "unparsed"
    reason: str
    raw: str

    @property
    def score(self) -> float:
        return 1.0 if self.label == "correct" else 0.0

    @property
    def parsed(self) -> bool:
        return self.label in ("correct", "wrong")


def parse_verdict(raw: str) -> Verdict:
    """
    Extract the judge's decision.

    A model that ignores the format must never be silently scored as "wrong" --
    that would quietly bias every result downward and look like a real finding.
    Unparsed verdicts are labelled as such and counted separately.
    """
    match = _JSON_BLOCK.search(raw or "")
    if match:
        try:
            data = json.loads(match.group())
            label = str(data.get("label", "")).strip().lower()
            if label in ("correct", "wrong"):
                return Verdict(label, str(data.get("reason", "")), raw)
        except json.JSONDecodeError:
            pass

    # The JSON never closed (truncated output). The label is still explicit.
    label_match = _LABEL_ONLY.search(raw or "")
    if label_match:
        return Verdict(label_match.group(1).lower(), "label recovered from truncated output", raw)

    # Fall back to an unambiguous bare word, and only that.
    lowered = (raw or "").strip().lower()
    if lowered.startswith("correct") or lowered == '"correct"':
        return Verdict("correct", "bare label", raw)
    if lowered.startswith("wrong") or lowered == '"wrong"':
        return Verdict("wrong", "bare label", raw)
    return Verdict("unparsed", "judge did not return a usable label", raw)


def judge_response(llm: LLM, sample: Sample, prediction: str,
                   max_tokens: int = 256) -> Verdict:
    raw = llm.complete(
        JUDGE_PROMPT.format(
            evidence=sample.evidence,
            trigger=sample.trigger,
            prediction=prediction,
        ),
        system=JUDGE_SYSTEM,
        # 256 suits the Groq judge, which is TPM-bound. Gemini has a 250k
        # window and its models may think before answering, so the secondary
        # judge passes a larger budget rather than getting truncated.
        max_tokens=max_tokens,
    )
    return parse_verdict(raw)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    key: str
    label: str
    build: Callable[[Sample, Optional[str]], str]
    needs_extraction: bool = False


CONDITIONS: list[Condition] = [
    Condition("no_memory", "A. No memory (trigger only)",
              lambda s, c: prompt_no_memory(s)),
    Condition("full_context", "B. Full context (benchmark default)",
              lambda s, c: prompt_full_context(s)),
    Condition("oracle_cue", "C. Oracle cue verbatim",
              lambda s, c: prompt_oracle_cue(s)),
    Condition("oracle_constraint", "D. Oracle constraint (extracted)",
              lambda s, c: prompt_oracle_constraint(s, c or ""), needs_extraction=True),
]

CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}


@dataclass
class Result:
    condition: str
    sample_index: int
    prompt_tokens: int
    prediction: str
    label: str
    reason: str
    constraint: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
