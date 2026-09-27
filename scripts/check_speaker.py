"""
Does `target_speaker` resolve who "A" is, on every item?

    python scripts/check_speaker.py

Free: no API calls, no embedder. Needs only the dataset.

WHY. Four selection failures in the n=100 events run carried notes about the
OTHER person in the conversation (#2008, #2030, #2096, #2363). The re-ranker
never learns who it is answering, because the benchmark anonymises the query's
speaker as `A:`. Printing the raw data on 4 Sep showed the conversation is
written `Name said, "..."` and the trigger's own text is the final line,
attributed by name -- so `A` is recoverable from the system's own input.

This checks that claim before a day of quota is spent on it. Two questions:

  1. Does the resolver name someone on all 100 items?
  2. Does that name match the speaker of the gold `evidence` line?

Question 2 uses `evidence` ONLY to validate the resolver here. The resolver
itself never sees it, and neither will the re-ranker -- otherwise the fix would
be an oracle and the result unreportable.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.dataset import LocomoPlus
from bapca.pipeline import target_speaker
from bapca.stats import Proportion


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def gold_speaker(evidence: str) -> str | None:
    """The evidence field uses `Name：` with a FULLWIDTH colon (U+FF1A) -- this
    dataset's format has been guessed wrong five times, so split on both."""
    first = (evidence or "").splitlines()[0] if evidence else ""
    match = re.match(r"^\s*([^:：]{1,30})[:：]", first)
    return match.group(1).strip() if match else None


def main() -> int:
    samples = LocomoPlus().subset(100, seed=42)
    print(f"Items: {len(samples)}")

    resolved, agree, disagree, unresolved = [], [], [], []
    for s in samples:
        who = target_speaker(s.input_prompt, s.trigger)
        gold = gold_speaker(s.evidence)
        if who is None:
            unresolved.append((s.index, gold))
            continue
        resolved.append(s.index)
        if gold and who.lower() == gold.lower():
            agree.append(s.index)
        else:
            disagree.append((s.index, who, gold))

    banner("1. Does the resolver name anyone?")
    print(f"  resolved from input alone   {Proportion(len(resolved), len(samples))}")
    if unresolved:
        print(f"  unresolved: {[i for i, _ in unresolved][:12]}")

    banner("2. Is it the right person? (validated against gold evidence)")
    if resolved:
        print(f"  matches the gold cue's speaker  {Proportion(len(agree), len(resolved))}")
    for index, who, gold in disagree[:10]:
        print(f"    #{index}  resolver said {who!r}, gold cue is {gold!r}")

    banner("Verdict")
    rate = len(agree) / max(1, len(samples))
    if rate >= 0.95:
        print(f"  {100*rate:.0f}% correct from the system's own input. A person-aware")
        print("  re-rank prompt is worth one day of quota (100 re-rank + 100 judge,")
        print("  ~150,000 tokens; the extraction cache is untouched).")
    elif rate >= 0.8:
        print(f"  {100*rate:.0f}% correct. Usable, but the re-rank prompt must PREFER")
        print("  the named person rather than filter on them, or the misses cost more")
        print("  than the hits gain.")
    else:
        print(f"  Only {100*rate:.0f}% correct. Do not build on this. Report the")
        print("  anonymous speaker as a benchmark limitation instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
