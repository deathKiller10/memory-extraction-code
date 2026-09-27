"""
Are three WRONG notes worse than an empty prompt?

    python scripts/distractor_cost.py results/<run>.json

Free: no API calls, no dataset, no embedder.

WHY THIS SCRIPT EXISTS. The n=100 events run scores 10.7% on the items where
the cue was not delivered, against condition A's 37.0% floor. Read naively that
says our own distractors are actively harmful. The project has already published
and retracted exactly that claim once (HANDOFF 4.7 -> 4.8): the -12 points were
item difficulty, because the failed items are not a random sample and condition
A was measured on all 100. The paired difference was -1.5 points, p=1.0000.

So the only admissible test restricts A to the SAME items and pairs them.
`analyse_rerank.py` does this for the v1 forced-slate run and hardcodes those
file paths, so it cannot answer the question for any newer run. This script
takes the run as an argument.

It reports the test twice, once per delivery metric, because they disagree:

    cue_selected    a note from the cue's WINDOW reached the prompt
    evidence_found  a note RESEMBLING the cue reached it   <- quote this one
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.stats import Proportion, mcnemar

RESULTS = Path("results")
SCORED = ("correct", "wrong")


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    run_path = Path(sys.argv[1])
    pilot_path = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "pilot_n100_seed42.json"
    for p in (run_path, pilot_path):
        if not p.exists():
            print(f"Not found: {p}")
            return 1

    banner("Files actually read (check these are the ones you meant)")
    print(f"  run:   {run_path}")
    print(f"  pilot: {pilot_path}")

    ours = {r["sample_index"]: r for r in json.loads(run_path.read_text(encoding="utf-8"))
            if r.get("label") in SCORED}
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    floor = {r["sample_index"]: r["label"] == "correct" for r in pilot
             if r.get("condition") == "no_memory" and r.get("label") in SCORED}

    shared = sorted(set(ours) & set(floor))
    print(f"\n  items in both: {len(shared)}")
    if len(shared) < 10:
        print("  Too few shared items to test.")
        return 1

    banner("Reference: A vs E on ALL shared items")
    a_all = [floor[i] for i in shared]
    e_all = [ours[i]["label"] == "correct" for i in shared]
    print(f"  A. no memory      {Proportion(sum(a_all), len(a_all))}")
    print(f"  E. our system     {Proportion(sum(e_all), len(e_all))}")
    t = mcnemar(a_all, e_all)
    print(f"  paired: {'E better' if t.only_b > t.only_a else 'A better'} "
          f"(p={t.p_value:.4f})   A only {t.only_a}, E only {t.only_b}")

    for metric, name in [("evidence_found", "content (evidence_found) -- QUOTE THIS"),
                         ("cue_selected", "window (cue_selected)")]:
        failed = [i for i in shared if not ours[i].get(metric)]
        banner(f"The real test, by {name}: {len(failed)} items where the cue "
               f"was NOT delivered")
        if len(failed) < 10:
            print(f"  Only {len(failed)} such items -- too few to test.")
            continue
        a = [floor[i] for i in failed]
        e = [ours[i]["label"] == "correct" for i in failed]
        pa, pe = Proportion(sum(a), len(a)), Proportion(sum(e), len(e))
        print(f"  A. empty prompt, these items       {pa}")
        print(f"  E. 3 notes, none of them right     {pe}")
        print(f"  paired difference: {100*(pe.rate - pa.rate):+.1f} points")
        t = mcnemar(a, e)
        # verdict() labels its lists positionally as "A"/"B"; here A really is
        # condition A and B is ours, but say so explicitly -- bug 17 was
        # exactly this collision reading backwards.
        words = (t.verdict().replace("A better", "A (no memory) better")
                            .replace("B better", "E (ours) better"))
        print(f"  McNemar: {words}")
        print(f"    A only {t.only_a}, E only {t.only_b}, both {t.both}, "
              f"neither {t.neither}")

    banner("How to read this")
    print("  If E is significantly BELOW A on identical items, three irrelevant")
    print("  standing notes are measurably worse than an empty prompt, and that")
    print("  is a real cost to report alongside the headline.")
    print("  If it is not significant, the raw gap is item difficulty and the")
    print("  claim must be dropped -- as it was in HANDOFF 4.8.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
