"""
Pair a run_system.py result file against ONE condition of a run_pilot.py file.

    python scripts/paired_vs_pilot.py results/<system>.json results/<pilot>.json
    python scripts/paired_vs_pilot.py results/<system>.json results/<pilot>.json --condition oracle_cue

Free: no API calls, no dataset, no embedder. Works on a PARTIAL system file,
which is the point -- the 401-item run lands ~120 items a day and the question
"is the headline holding?" should not have to wait for all of them.

WHY THIS SCRIPT EXISTS SEPARATELY FROM compare_runs.py. That script compares
two run_system.py files, which share a schema. A pilot file does not: its rows
carry a `condition` key, several conditions live in one file, and it uses two
extra labels (`no_constraint`, `unparsed`) that run_system.py never emits.
Pairing them by hand is exactly the kind of thing this project has already got
wrong once, so it lives in a script that prints what it found before it uses
it.

THREE WARNINGS THIS SCRIPT PRINTS AND YOU SHOULD READ EVERY TIME:

1. OPTIONAL STOPPING. Looking at a p-value on a partial run and stopping when
   it looks good is how a result stops being a result. The full 401 is being
   run regardless of what this prints. This is a progress check, not the
   reported number.

2. CLUSTERING. The Cognitive items come from roughly ten base conversations.
   The system run walks them in index order, so a partial file is not a random
   subset -- it is the first k conversations. Blocks of items can be easier or
   harder than the whole, and the running score will drift for that reason
   alone.

3. NAME THE WINNER. HANDOFF bug 17: positional "A"/"B" labels in a McNemar
   verdict once printed this project's headline backwards and nearly published
   it. This script states which named condition won, and computes the exact
   two-sided p-value independently of bapca.stats so the two can be compared.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.stats import Proportion, mcnemar


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def exact_mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided binomial test on the discordant pairs, from scratch.

    b and c are the two discordant counts. Under H0 each discordant pair is a
    fair coin, so the two-sided p is the total probability of every outcome no
    more likely than the observed one. Computed here independently of
    bapca.stats so that a disagreement between the two is visible rather than
    silent.
    """
    n = b + c
    if n == 0:
        return 1.0
    probs = [math.comb(n, k) * 0.5 ** n for k in range(n + 1)]
    observed = probs[b]
    return min(1.0, sum(p for p in probs if p <= observed + 1e-12))


def describe(path: Path, rows: list[dict], title: str) -> None:
    print(f"\n  {title}")
    print(f"    file    {path}")
    print(f"    rows    {len(rows)}")
    if not rows:
        return
    print(f"    keys    {sorted(rows[0])}")
    labels = Counter(r.get("label") for r in rows)
    print(f"    labels  {dict(labels)}")
    conds = Counter(r.get("condition") for r in rows if "condition" in r)
    if conds:
        print(f"    conditions  {dict(conds)}")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    condition = "full_context"
    if "--condition" in sys.argv:
        condition = sys.argv[sys.argv.index("--condition") + 1]
    if len(args) != 2:
        print(__doc__)
        print("Give a system result file and a pilot result file.")
        return 1

    sys_path, pilot_path = Path(args[0]), Path(args[1])
    for p in (sys_path, pilot_path):
        if not p.exists():
            print(f"Not found: {p}")
            return 1

    sys_rows = json.loads(sys_path.read_text(encoding="utf-8"))
    pilot_rows = json.loads(pilot_path.read_text(encoding="utf-8"))

    banner("What is in these files (printed before anything is assumed)")
    describe(sys_path, sys_rows, "SYSTEM (condition E, ours)")
    describe(pilot_path, pilot_rows, "PILOT (baselines)")

    available = sorted({r["condition"] for r in pilot_rows if "condition" in r})
    if condition not in available:
        print(f"\n  '{condition}' is not in this pilot file.")
        print(f"  Available: {available}")
        print(f"  Re-run with --condition <one of those>.")
        return 1

    # ---- build the two score maps ------------------------------------------
    # run_system.py emits only correct/wrong. Anything else is not a scored
    # item and is dropped rather than counted as a failure.
    ours: dict[int, bool] = {
        r["sample_index"]: r["label"] == "correct"
        for r in sys_rows
        if r.get("label") in ("correct", "wrong")
    }

    # run_pilot.py's convention, copied deliberately rather than reinvented:
    #   no_constraint -> the item had no gold constraint; scored as FALSE
    #   unparsed      -> the judge's reply could not be read; DROPPED
    theirs: dict[int, bool] = {}
    dropped_unparsed = 0
    for r in pilot_rows:
        if r.get("condition") != condition:
            continue
        label = r.get("label")
        if label == "unparsed":
            dropped_unparsed += 1
            continue
        if label == "no_constraint":
            theirs[r["sample_index"]] = False
            continue
        if label in ("correct", "wrong"):
            theirs[r["sample_index"]] = label == "correct"

    shared = sorted(set(ours) & set(theirs))

    banner(f"Paired on the items BOTH files scored: ours vs {condition}")
    print(f"  ours          {len(ours):>4} scored items")
    print(f"  {condition:<13} {len(theirs):>4} scored items"
          + (f"   ({dropped_unparsed} dropped as unparsed)" if dropped_unparsed else ""))
    print(f"\n  items in BOTH: {len(shared)}   <- every number below is on these only")
    if not shared:
        print("\n  Nothing in common. Same --n and --seed?")
        return 1

    k_ours = sum(ours[i] for i in shared)
    k_them = sum(theirs[i] for i in shared)
    p_ours = Proportion(k_ours, len(shared))
    p_them = Proportion(k_them, len(shared))
    delta = 100 * (k_ours - k_them) / len(shared)

    print(f"\n  ours (E)      {p_ours}")
    print(f"  {condition:<13} {p_them}")
    print(f"  difference    {delta:+.1f} points")

    # ---- discordant pairs, spelled out -------------------------------------
    ours_only = sum(1 for i in shared if ours[i] and not theirs[i])
    them_only = sum(1 for i in shared if theirs[i] and not ours[i])
    both = sum(1 for i in shared if ours[i] and theirs[i])
    neither = len(shared) - both - ours_only - them_only

    banner("Exact McNemar, two-sided")
    print(f"  both correct                 {both:>4}")
    print(f"  ours correct, {condition} wrong   {ours_only:>4}   <- discordant")
    print(f"  {condition} correct, ours wrong   {them_only:>4}   <- discordant")
    print(f"  both wrong                   {neither:>4}")

    p_here = exact_mcnemar_p(ours_only, them_only)
    print(f"\n  p = {p_here:.4f}   (computed in this script from math.comb)")
    print(f"  bapca.stats.mcnemar says:  "
          f"{mcnemar([ours[i] for i in shared], [theirs[i] for i in shared])}")
    print("  ^ if those two p-values disagree, stop and find out why before "
          "believing either.")

    if ours_only == them_only:
        winner = "neither -- the discordant counts are equal"
    elif ours_only > them_only:
        winner = "OURS (condition E)"
    else:
        winner = f"{condition.upper()}"
    sig = "significant" if p_here < 0.05 else "NOT significant"
    print(f"\n  More items won by: {winner}")
    print(f"  At alpha = 0.05 this is {sig}.")

    banner("Before you quote any of this")
    print(f"  1. OPTIONAL STOPPING. This is {len(shared)} items of a run that is")
    print("     still going. The reported number is the finished run, whatever")
    print("     this says. Do not stop the run because this looks good.")
    print("  2. CLUSTERING. The system run walks items in index order and the")
    print("     Cognitive items come from ~10 base conversations, so a partial")
    print("     file is the first few conversations, not a random subset.")
    print("  3. The winner above is named, not positional. Check it reads the")
    print("     way you expect before it goes anywhere near the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
