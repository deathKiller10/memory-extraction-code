"""
Compare two run_system.py result files on the items they BOTH contain.

    python scripts/compare_runs.py results/A.json results/B.json

Free: no API calls, no dataset, no embedder. It reads only fields every row
already carries, so it works on a PARTIAL file -- which is the point. Bug 16
means a cold extraction pass stops partway through the day, and the question
"did the new extraction prompt change anything?" should not have to wait for
all 100 items.

WHY PAIRED, ALWAYS. This project has already retracted one claim built on an
unpaired comparison (HANDOFF 4.7 -> 4.8: a -12 point "distractor cost" that
turned out to be item difficulty). And the 100 items are drawn from only about
ten base conversations, so a handful of items is not a handful of independent
observations -- ten consecutive indices may span only four or five
conversations. This script reports the item count and says so plainly.

Reported for each run, on the shared items only:

    score          label == "correct"
    recall/window  cue_selected     -- a note from the cue's window got in
    recall/cosine  evidence_found   -- a note RESEMBLING the cue got in
    notes/item     notes_extracted  -- store size; the dilution check
    tokens         prompt_tokens

`evidence_found` is the one to quote: HANDOFF 4.15 measured it as the best
predictor of a correct answer (48.1-point gap, against 43.8 for the window
metric).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.stats import Proportion, mcnemar

SCORED = ("correct", "wrong")


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def load(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["sample_index"]: r for r in rows if r.get("label") in SCORED}


def median(values: list) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def flags(rows: dict, ids: list, field: str) -> list:
    return [bool(rows[i].get(field)) for i in ids]


def correct(rows: dict, ids: list) -> list:
    return [rows[i]["label"] == "correct" for i in ids]


def identity(rows: dict, ids: list, field: str) -> str:
    """score = recall * P(ok|cue) + (1-recall) * P(ok|no cue), spelled out."""
    if not ids:
        return "  (no items)"
    with_cue = [i for i in ids if rows[i].get(field)]
    without = [i for i in ids if not rows[i].get(field)]
    recall = len(with_cue) / len(ids)
    p_with = (sum(rows[i]["label"] == "correct" for i in with_cue) / len(with_cue)
              if with_cue else 0.0)
    p_without = (sum(rows[i]["label"] == "correct" for i in without) / len(without)
                 if without else 0.0)
    return (f"  recall {100*recall:5.1f}%  x  P(ok|cue) {100*p_with:5.1f}% "
            f"({len(with_cue)})  +  P(ok|no cue) {100*p_without:5.1f}% "
            f"({len(without)})  =  {100*(recall*p_with + (1-recall)*p_without):5.1f}%")


def named_verdict(r, name_a: str, name_b: str, alpha: float = 0.05) -> str:
    """
    McNemar.verdict() says "A better" / "B better". Bug 17: a positional
    label once put the headline in the paper backwards, so the winner is
    named by its FILE here, never by position. (27 Sep)
    """
    if r.discordant == 0:
        return "identical on every item"
    if r.p_value < alpha:
        winner = name_b if r.only_b > r.only_a else name_a
        return f"different (p={r.p_value:.4f}), better: {winner}"
    leader = (name_b if r.only_b > r.only_a else
              name_a if r.only_a > r.only_b else "neither")
    return (f"no significant difference (p={r.p_value:.4f}); "
            f"more items won by: {leader}")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        print("Give two result files.")
        return 1
    paths = [Path(sys.argv[1]), Path(sys.argv[2])]
    for p in paths:
        if not p.exists():
            print(f"Not found: {p}")
            return 1

    a, b = load(paths[0]), load(paths[1])
    shared = sorted(set(a) & set(b))

    banner("What is being compared")
    print(f"  A  {paths[0].name:<48} {len(a):>4} scored items")
    print(f"  B  {paths[1].name:<48} {len(b):>4} scored items")
    print(f"\n  items in BOTH: {len(shared)}   <- every number below is on these only")
    if not shared:
        print("\n  Nothing in common. Are these the same --n and --seed?")
        return 1
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a:
        print(f"  only in A: {len(only_a)}")
    if only_b:
        print(f"  only in B: {len(only_b)}")

    banner(f"Side by side, paired on {len(shared)} identical items")
    print(f"  {'':<26} {'A':>26} {'B':>26}")
    print(f"  {'-' * 78}")
    print(f"  {'score (correct)':<26} "
          f"{str(Proportion(sum(correct(a, shared)), len(shared))):>26} "
          f"{str(Proportion(sum(correct(b, shared)), len(shared))):>26}")
    for title, field in [("recall / window", "cue_selected"),
                         ("recall / cosine", "evidence_found")]:
        pa = Proportion(sum(flags(a, shared, field)), len(shared))
        pb = Proportion(sum(flags(b, shared, field)), len(shared))
        print(f"  {title:<26} {str(pa):>26} {str(pb):>26}")
    for title, field in [("notes / item (median)", "notes_extracted"),
                         ("prompt tokens (median)", "prompt_tokens")]:
        va = median([a[i].get(field, 0) for i in shared])
        vb = median([b[i].get(field, 0) for i in shared])
        print(f"  {title:<26} {va:>26.0f} {vb:>26.0f}")

    banner("Paired tests (exact McNemar, two-sided)")
    for title, field in [("correct", None), ("cue_selected", "cue_selected"),
                         ("evidence_found", "evidence_found")]:
        if field is None:
            xa, xb = correct(a, shared), correct(b, shared)
        else:
            xa, xb = flags(a, shared, field), flags(b, shared, field)
        r = mcnemar(xa, xb)
        print(f"  {title:<16} {named_verdict(r, paths[0].name, paths[1].name)}")
        print(f"  {'':<16} only {paths[0].name}: {r.only_a}   "
              f"only {paths[1].name}: {r.only_b}   both: {r.both}")

    banner("The identity, on the shared items, by content (evidence_found)")
    print("  A:" + identity(a, shared, "evidence_found"))
    print("  B:" + identity(b, shared, "evidence_found"))

    if len(shared) <= 25:
        banner("Item by item (the overlap is small enough to read)")
        print(f"  {'item':>6}  {'A':<9} {'B':<9}  {'A cue':<7} {'B cue':<7}  notes A/B")
        for i in shared:
            print(f"  #{i:<5}  {a[i]['label']:<9} {b[i]['label']:<9}  "
                  f"{'yes' if a[i].get('evidence_found') else '-':<7} "
                  f"{'yes' if b[i].get('evidence_found') else '-':<7}  "
                  f"{a[i].get('notes_extracted', 0)}/{b[i].get('notes_extracted', 0)}")

    banner("How much of this is real")
    print(f"  n = {len(shared)} items. The 100 items come from about ten base")
    print("  conversations, so items are CLUSTERED: a run of consecutive indices may")
    print("  span only a few conversations, and the effective sample size is smaller")
    print("  than the item count. Treat anything under ~30 shared items as a reason")
    print("  to keep going, not as a result to write down.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
