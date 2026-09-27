"""
Why does the score stay at ~42% however much recall improves?

Free: reads saved results and the dataset. No API calls.

    python scripts/marginal_value.py

Three runs of condition E now exist, and together they trace a curve:

    run                  cue reaches prompt   correct WITH   correct WITHOUT   score
    permissive                   16.0%            87.5%           33.3%        42.0%
    forced slate                 32.0%            78.1%           25.0%        42.0%
    forced + no trigger          38.4%            68.4%           24.6%        41.4%

Recall rose 22 points. Accuracy-when-delivered fell 19. The product is flat.

Hypothesis under test: retrieval difficulty and utilisation difficulty are
CORRELATED. The cue notes that are easiest to find are also the ones the
generator can most easily use, so each extra point of recall delivers a
progressively less useful note. If that holds, pushing recall further is not
worth doing, and what remains is note QUALITY -- measured against condition D,
which carried the same fact in better words and scored 87.6%.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.stats import Proportion, mcnemar

RESULTS = Path("results")
RUNS = [
    ("permissive",          RESULTS / "system_n100_seed42_llm3_d1_PERMISSIVE.json"),
    ("forced slate",        RESULTS / "system_n100_seed42_llm3_d1.json"),
    ("forced + no trigger", RESULTS / "system_n100_seed42_llm3_d1_notrig.json"),
]
PILOT = RESULTS / "pilot_n100_seed42.json"


def banner(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def load(path):
    if not path.exists():
        print(f"  MISSING: {path}")
        return {}
    return {r["sample_index"]: r for r in json.loads(path.read_text(encoding="utf-8"))
            if r["label"] in ("correct", "wrong")}


def main() -> int:
    runs = [(n, load(p)) for n, p in RUNS]
    runs = [(n, r) for n, r in runs if r]
    if len(runs) < 2:
        print("Need at least two runs.")
        return 1

    banner("1. score = recall x P(ok | cue) + (1 - recall) x P(ok | no cue)")
    print(f"  {'run':<22} {'recall':>8} {'P(ok|cue)':>11} {'P(ok|no cue)':>13} "
          f"{'predicted':>10} {'actual':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*11} {'-'*13} {'-'*10} {'-'*8}")
    for name, rows in runs:
        loc = [r for r in rows.values() if r.get("cue_window") is not None]
        w = [r for r in loc if r["cue_selected"]]
        wo = [r for r in loc if not r["cue_selected"]]
        if not w or not wo:
            continue
        sel = len(w) / len(loc)
        pw = sum(r["label"] == "correct" for r in w) / len(w)
        pwo = sum(r["label"] == "correct" for r in wo) / len(wo)
        act = sum(r["label"] == "correct" for r in rows.values()) / len(rows)
        print(f"  {name:<22} {100*sel:>7.1f}% {100*pw:>10.1f}% {100*pwo:>12.1f}% "
              f"{100*(sel*pw + (1-sel)*pwo):>9.1f}% {100*act:>7.1f}%")
    print("\n  Recall climbs; P(ok|cue) falls by almost as much; the product is flat.")

    banner("2. What is each ADDITIONAL delivered cue note worth?")
    shared = sorted(set.intersection(*[set(r) for _, r in runs]))
    tiers = {}
    for i in shared:
        got = [r[i]["cue_selected"] for _, r in runs]
        if all(got):
            key = "delivered by every run (easiest)"
        elif got[-1] and not got[0]:
            key = "only once recall was pushed (harder)"
        elif any(got):
            key = "inconsistent across runs"
        else:
            key = "never delivered"
        tiers.setdefault(key, []).append(i)

    last = runs[-1][1]
    print(f"  {'tier':<38} {'n':>4} {'score in the newest run':>26}")
    print(f"  {'-'*38} {'-'*4} {'-'*26}")
    for key in ["delivered by every run (easiest)",
                "only once recall was pushed (harder)",
                "inconsistent across runs", "never delivered"]:
        ids = tiers.get(key, [])
        if not ids:
            continue
        p = Proportion(sum(last[i]["label"] == "correct" for i in ids), len(ids))
        print(f"  {key:<38} {len(ids):>4} {str(p):>26}")
    print("\n  If 'easiest' scores far above 'harder', the notes that are simple to")
    print("  retrieve are also the ones the generator can use, and buying more recall")
    print("  buys progressively less.")

    banner("3. When we DO deliver the cue, how much worse is our note than D's?")
    if PILOT.exists():
        pilot = json.loads(PILOT.read_text(encoding="utf-8"))
        d = {r["sample_index"]: r["label"] == "correct" for r in pilot
             if r["condition"] == "oracle_constraint" and r["label"] in ("correct", "wrong")}
        ids = [i for i in last if last[i]["cue_selected"] and i in d]
        if len(ids) >= 10:
            ours = Proportion(sum(last[i]["label"] == "correct" for i in ids), len(ids))
            theirs = Proportion(sum(d[i] for i in ids), len(ids))
            print(f"  items where we delivered a note from the cue window: {len(ids)}")
            print(f"\n  D. oracle constraint (gold wording): {theirs}")
            print(f"  E. our extracted note:               {ours}")
            print(f"  difference: {100*(ours.rate - theirs.rate):+.1f} points")
            t = mcnemar([d[i] for i in ids],
                        [last[i]["label"] == "correct" for i in ids])
            print(f"  paired: {t.verdict()}  (D only {t.only_a}, E only {t.only_b})")
            print("\n  Same items, same underlying fact, different words. A gap here is")
            print("  note QUALITY, not retrieval -- the one lever the recall experiments")
            print("  cannot touch.")

    banner("4. We delivered a cue-window note and still got it wrong")
    try:
        from bapca.dataset import LocomoPlus
        samples = {s.index: s for s in LocomoPlus().subset(100, seed=42)}
    except Exception as exc:
        print(f"  dataset unavailable ({exc.__class__.__name__}: {exc})")
        return 0
    bad = [i for i in sorted(last) if last[i]["cue_selected"] and last[i]["label"] == "wrong"]
    print(f"  {len(bad)} such items. First 6:\n")
    for i in bad[:6]:
        s = samples.get(i)
        print(f"  #{i}")
        if s:
            print(f"    trigger:   {s.trigger[:96]}")
            print(f"    GOLD cue:  {(s.evidence.splitlines() or [''])[0][:96]}")
        for note in last[i]["carried"]:
            print(f"    our note:  {note[:96]}")
        print()
    print("  Read these. If our note states the same fact as the gold cue, the loss is")
    print("  in the generator. If it is vaguer, the loss is extraction, and")
    print("  EXTRACT_PROMPT is the thing to change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
