"""
Post-hoc analysis of the re-ranker recall fix. Costs nothing: reads the saved
result JSONs and re-segments the conversations on CPU. No API calls.

    python scripts/analyse_rerank.py

Answers four questions the run summaries cannot:

  1. Did the similarity backfill ever fire? (hybrid vs forced-slate llm)
  2. Doubling recall moved the score by zero. Where did the gain go?
  3. Do wrong notes cost more than no notes at all?
  4. Does extraction ever decline a window, or does it note everything?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.stats import Proportion, mcnemar

RESULTS = Path("results")

PERMISSIVE = RESULTS / "system_n100_seed42_llm3_d1_PERMISSIVE.json"
FORCED_LLM = RESULTS / "system_n100_seed42_llm3_d1.json"
HYBRID     = RESULTS / "system_n100_seed42_hybrid3_d1.json"
PILOT      = RESULTS / "pilot_n100_seed42.json"


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def load(path: Path) -> dict:
    if not path.exists():
        print(f"  MISSING: {path}")
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["sample_index"]: r for r in rows if r["label"] in ("correct", "wrong")}


def correct(rows: dict) -> dict:
    return {i: r["label"] == "correct" for i, r in rows.items()}


def main() -> int:
    perm, forced, hyb = load(PERMISSIVE), load(FORCED_LLM), load(HYBRID)

    # ---- 1. did the backfill ever fire? -----------------------------------
    banner("1. Did the similarity backfill ever fire?")
    if forced and hyb:
        shared = sorted(set(forced) & set(hyb))
        same_label = sum(forced[i]["label"] == hyb[i]["label"] for i in shared)
        same_notes = sum(forced[i]["carried"] == hyb[i]["carried"] for i in shared)
        short = sum(1 for i in shared if forced[i]["notes_carried"] < 3)
        print(f"  items compared:                    {len(shared)}")
        print(f"  identical judge label:             {same_label}/{len(shared)}")
        print(f"  identical note text in the prompt: {same_notes}/{len(shared)}")
        print(f"  forced-slate items with <3 picks:  {short}/{len(shared)}")
        if short == 0:
            print("\n  The model returned a full slate on every item, so hybrid had no"
                  "\n  empty slot to fill. The backfill is dead code at top_k=3; the"
                  "\n  forced-slate PROMPT did all of the work.")

    # ---- 2. where did the recall gain go? ---------------------------------
    banner("2. Recall doubled. Where did the score gain go?")
    print(f"  {'run':<22} {'cue reached prompt':<28} {'correct WITH':<28} {'correct WITHOUT'}")
    print(f"  {'-'*22} {'-'*28} {'-'*28} {'-'*28}")
    for name, rows in [("permissive (16%)", perm), ("forced slate", forced)]:
        if not rows:
            continue
        loc = [r for r in rows.values() if r.get("cue_window") is not None]
        with_cue = [r for r in loc if r["cue_selected"]]
        without  = [r for r in loc if not r["cue_selected"]]
        sel = Proportion(len(with_cue), len(loc))
        a = Proportion(sum(r["label"] == "correct" for r in with_cue), len(with_cue))
        b = Proportion(sum(r["label"] == "correct" for r in without), len(without))
        print(f"  {name:<22} {str(sel):<28} {str(a):<28} {str(b)}")

    if perm and forced:
        shared = sorted(set(perm) & set(forced))
        cp, cf = correct(perm), correct(forced)
        test = mcnemar([cp[i] for i in shared], [cf[i] for i in shared])
        print(f"\n  permissive vs forced slate (paired, n={len(shared)}): {test.verdict()}")
        print(f"    both correct {test.both}, permissive only {test.only_a}, "
              f"forced only {test.only_b}, neither {test.neither}")
        gained = [i for i in shared if cf[i] and not cp[i]]
        lost   = [i for i in shared if cp[i] and not cf[i]]
        print(f"    items the fix WON:  {len(gained)}")
        print(f"    items the fix LOST: {len(lost)}")
        print("\n  The marginal score is identical, so any movement here is items"
              "\n  trading places, not net progress.")

    # ---- 3. are wrong notes worse than no notes? --------------------------
    banner("3. Are wrong notes worse than no memory at all?")
    if PILOT.exists() and forced:
        pilot = json.loads(PILOT.read_text(encoding="utf-8"))
        floor = [r["label"] == "correct" for r in pilot
                 if r["condition"] == "no_memory" and r["label"] in ("correct", "wrong")]
        a_floor = Proportion(sum(floor), len(floor))
        loc = [r for r in forced.values() if r.get("cue_window") is not None]
        without = [r for r in loc if not r["cue_selected"]]
        b = Proportion(sum(r["label"] == "correct" for r in without), len(without))
        print(f"  A. no memory, 0 notes in the prompt:      {a_floor}")
        print(f"  E. 3 notes, none of them the right one:   {b}")
        gap = 100 * (b.rate - a_floor.rate)
        print(f"\n  unpaired difference: {gap:+.1f} points")
        print("\n  CONFOUND: those 68 items are not a random sample. They are the items")
        print("  where selection FAILED, which may simply be the harder ones. Condition A")
        print("  above is measured on all 100. The honest test restricts A to the same 68")
        print("  items and pairs them:")

        a_by_item = {r["sample_index"]: r["label"] == "correct" for r in pilot
                     if r["condition"] == "no_memory" and r["label"] in ("correct", "wrong")}
        ids = [r["sample_index"] for r in without if r["sample_index"] in a_by_item]
        if len(ids) >= 10:
            a_same = Proportion(sum(a_by_item[i] for i in ids), len(ids))
            e_same = Proportion(sum(forced[i]["label"] == "correct" for i in ids), len(ids))
            print(f"\n  A. no memory, SAME {len(ids)} items:        {a_same}")
            print(f"  E. 3 wrong notes, same items:         {e_same}")
            print(f"  paired difference: {100 * (e_same.rate - a_same.rate):+.1f} points")
            test = mcnemar([a_by_item[i] for i in ids],
                           [forced[i]["label"] == "correct" for i in ids])
            print(f"  McNemar (A vs E on these items): {test.verdict()}")
            print(f"    A only {test.only_a}, E only {test.only_b}, "
                  f"both {test.both}, neither {test.neither}")
            print("\n  THIS is the number to quote. If E is significantly below A on the")
            print("  same items, then three irrelevant standing notes are measurably worse")
            print("  than an empty prompt, and that bounds any pure-recall fix. If it is")
            print("  not significant, the -12 points above is mostly item difficulty and")
            print("  the claim must be dropped.")
        else:
            print("  (too few overlapping items to pair)")

    # ---- 4. does extraction ever decline a window? ------------------------
    banner("4. Does extraction ever decline a window?")
    try:
        from bapca.dataset import LocomoPlus
        from bapca.pipeline import segment
        data = LocomoPlus()
        samples = {s.index: s for s in data.subset(100, seed=42)}
        rows = forced or hyb
        ratios = []
        for i, r in rows.items():
            s = samples.get(i)
            if s is None:
                continue
            n_windows = len(segment(s.input_prompt, 12))
            ratios.append((n_windows, r["notes_extracted"]))
        if ratios:
            # Median each column independently. Sorting the PAIRS and reading
            # one row reports the notes belonging to the median-window item,
            # which is not the median note count -- that bug printed 35 against
            # the run's own correct median of 29.
            windows = sorted(w for w, _ in ratios)
            notes = sorted(n for _, n in ratios)
            per_item = sorted(n / w for w, n in ratios if w)
            total_w = sum(w for w, _ in ratios)
            total_n = sum(n for _, n in ratios)
            mid = len(ratios) // 2
            print(f"  windows per item (median):  {windows[mid]}")
            print(f"  notes   per item (median):  {notes[mid]}")
            print(f"  kept per item (median):     {per_item[mid]:.1%}")
            print(f"  overall notes / windows:    {total_n}/{total_w} = "
                  f"{total_n / max(1, total_w):.1%}")
            print("\n  EXTRACT_PROMPT says 'Most excerpts reveal nothing... write exactly:")
            print("  NONE'. Measured at 68.7%, so it declines about one window in three --")
            print("  it is filtering, but nothing like 'most'. The question that decides")
            print("  the next experiment is not how MANY notes survive but whether the")
            print("  survivors are about the right person and say anything specific.")
            print("  Run scripts/inspect_notes.py for that.")
    except Exception as exc:                      # dataset not present locally
        print(f"  skipped ({exc.__class__.__name__}: {exc})")

    # ---- 6. where exactly did the null come from? -------------------------
    banner("6. The null, decomposed by what happened to the cue note")
    if perm and forced:
        shared = sorted(set(perm) & set(forced))
        groups = {"cue in NEITHER run": [], "cue GAINED by the fix": [],
                  "cue LOST by the fix": [], "cue in BOTH runs": []}
        for i in shared:
            a, b = perm[i]["cue_selected"], forced[i]["cue_selected"]
            key = ("cue in BOTH runs" if a and b else
                   "cue GAINED by the fix" if b else
                   "cue LOST by the fix" if a else "cue in NEITHER run")
            groups[key].append(i)

        print(f"  {'group':<24} {'n':>4}  {'permissive':>12} {'forced':>12}   "
              f"{'notes carried, permissive':>26}")
        print(f"  {'-'*24} {'-'*4}  {'-'*12} {'-'*12}   {'-'*26}")
        for key, ids in groups.items():
            if not ids:
                continue
            a = Proportion(sum(perm[i]["label"] == "correct" for i in ids), len(ids))
            b = Proportion(sum(forced[i]["label"] == "correct" for i in ids), len(ids))
            carried = sorted(perm[i]["notes_carried"] for i in ids)
            print(f"  {key:<24} {len(ids):>4}  {100*a.rate:>11.1f}% {100*b.rate:>11.1f}%   "
                  f"median {carried[len(carried)//2]:>2}, mean "
                  f"{sum(carried)/len(carried):>4.1f}")

        neither = groups["cue in NEITHER run"]
        if len(neither) >= 10:
            print("\n  THE CLEAN TEST OF DISTRACTOR COST.")
            print("  In 'cue in NEITHER run' the same items got no useful note either way;")
            print("  the only thing that changed is how many USELESS notes were in the")
            print("  prompt. Permissive carried the median shown above, forced carried 3.")
            test = mcnemar([perm[i]["label"] == "correct" for i in neither],
                           [forced[i]["label"] == "correct" for i in neither])
            print(f"\n  paired, n={len(neither)}: {test.verdict()}")
            print(f"    permissive only {test.only_a}, forced only {test.only_b}, "
                  f"both {test.both}, neither {test.neither}")
            print("\n  This is paired on identical items, so item difficulty cannot")
            print("  explain it -- unlike the A-vs-E comparison in section 3.")

        # Pooled: every item whose cue status did NOT change between runs.
        # NEITHER (n=67) and BOTH (n=15) are disjoint and both measure the same
        # thing -- what extra distractors cost -- so pooling them is the
        # properly powered version of the test above.
        unchanged = groups["cue in NEITHER run"] + groups["cue in BOTH runs"]
        if len(unchanged) >= 20:
            a = Proportion(sum(perm[i]["label"] == "correct" for i in unchanged), len(unchanged))
            b = Proportion(sum(forced[i]["label"] == "correct" for i in unchanged), len(unchanged))
            test = mcnemar([perm[i]["label"] == "correct" for i in unchanged],
                           [forced[i]["label"] == "correct" for i in unchanged])
            print(f"\n  POOLED (cue status unchanged either way), n={len(unchanged)}:")
            print(f"    permissive: {a}")
            print(f"    forced:     {b}")
            print(f"    {100 * (b.rate - a.rate):+.1f} points")
            print(f"    {test.verdict()}  "
                  f"(permissive only {test.only_a}, forced only {test.only_b})")
            print("\n    These items gained nothing from the fix and lost slots to it.")

        gained = groups["cue GAINED by the fix"]
        if len(gained) >= 5:
            a = Proportion(sum(perm[i]["label"] == "correct" for i in gained), len(gained))
            b = Proportion(sum(forced[i]["label"] == "correct" for i in gained), len(gained))
            print(f"\n  AND WHAT THE CUE BOUGHT, on the {len(gained)} items that gained it:")
            print(f"    without it (permissive): {a}")
            print(f"    with it (forced):        {b}")
            print(f"    {100*(b.rate - a.rate):+.1f} points on the items the fix actually helped.")
            print("\n  If this is strongly positive while the total is flat, the fix DID")
            print("  work where it fired, and the loss is elsewhere -- which is what")
            print("  section 6's first table locates.")

    # ---- 5. eyeball the notes --------------------------------------------
    banner("5. What do the carried notes actually look like?")
    rows = forced or hyb
    for i in sorted(rows)[:3]:
        r = rows[i]
        flag = "CUE PRESENT" if r["cue_selected"] else "cue missing"
        print(f"\n  #{i}  {r['label']}  ({flag})")
        for line in r["carried"]:
            print(f"    - {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
