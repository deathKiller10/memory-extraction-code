"""
Does "the cue reached the prompt" actually mean the cue reached the prompt?

Free: reads saved results and the dataset. No API calls.

    python scripts/true_recall.py

WHY. `cue_selected` is computed in run_system.py as

    cue_window in {n.source_turn for n in chosen}

-- a note extracted from the window that CONTAINS the cue was selected. That is
not the same as the cue itself reaching the prompt. A window is ~12 turns of
dialogue holding many facts; extraction writes ONE note per window; it can
easily write about a different fact in that window.

The evidence that this happens, from marginal_value.py section 4 -- five of six
inspected failures had `cue_selected = True` and carried nothing resembling the
cue:

    #2037  gold: migraine from screens -> bought blue-light glasses
           ours: three notes about running a dance studio

    #2121  gold: sore wrists -> bought an ergonomic keyboard
           ours: three notes about her dogs

If this is systematic, `cue_selected` OVERSTATES delivery, exactly as the old
cosine `hit_rate` UNDERSTATED it (bug 11). And it would explain the invariance
more simply than "retrieval difficulty correlates with utilisation difficulty":
pushing recall adds items where a USELESS note from the right window reached the
prompt, so P(ok | cue) falls mechanically.

Three opinions on the same question are compared here:
  cue_selected   window-based, what the runs report
  evidence_found cosine >= 0.5 between a carried note and the gold cue (already
                 recorded in every row)
  lexical        share of the gold cue's content words present in some carried note
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.stats import Proportion

RESULTS = Path("results")
RUNS = [
    ("permissive",          RESULTS / "system_n100_seed42_llm3_d1_PERMISSIVE.json"),
    ("forced slate",        RESULTS / "system_n100_seed42_llm3_d1.json"),
    ("forced + no trigger", RESULTS / "system_n100_seed42_llm3_d1_notrig.json"),
    ("events extraction",   RESULTS / "system_n100_seed42_llm3_d1_notrig_events.json"),
]

_STOP = set("""a an and are as at be been but by for from had has have he her hers him his i if
in into is it its me my of on or our she that the their them they this to was we were what
when which who will with you your not no so just still really very about after since every
day get got make made take took go went come came been being do does did""".split())


def words(text: str) -> set[str]:
    body = re.sub(r"^\[\w+\]\s*", "", text or "").lower()
    body = re.sub(r"[^a-z0-9 ]+", " ", body)
    return {w for w in body.split() if len(w) > 2 and w not in _STOP}


def gold_words(evidence: str) -> set[str]:
    out: set[str] = set()
    for line in (evidence or "").splitlines():
        out |= words(re.split(r"[:：]", line, maxsplit=1)[-1])
    return out


def best_overlap(carried, evidence) -> float:
    gold = gold_words(evidence)
    if not gold:
        return 0.0
    return max((len(gold & words(n)) / len(gold) for n in carried), default=0.0)


def banner(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def main() -> int:
    try:
        from bapca.dataset import LocomoPlus
        samples = {s.index: s for s in LocomoPlus().subset(100, seed=42)}
    except Exception as exc:
        print(f"dataset unavailable ({exc.__class__.__name__}: {exc})")
        return 1

    # It used to ignore sys.argv entirely and always read the three hardcoded
    # runs below. Passing it the events run therefore printed v1's numbers
    # under the events run's name -- the bug-12 family again, a script quietly
    # reporting a previous experiment. (4 Sep)
    wanted = list(RUNS)
    if len(sys.argv) > 1:
        wanted = [(Path(a).stem, Path(a)) for a in sys.argv[1:]]

    print("Files actually read (check these are the ones you meant):")
    for _, path in wanted:
        print(f"  {path}   {'ok' if path.exists() else 'MISSING'}")

    runs = []
    for name, path in wanted:
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        rows = {r["sample_index"]: r for r in json.loads(path.read_text(encoding="utf-8"))
                if r["label"] in ("correct", "wrong")}
        runs.append((name, rows))
    if not runs:
        return 1

    last_name, last = runs[-1]

    # ---- 1. do the three metrics agree? -----------------------------------
    banner(f"1. Three ways to ask 'did the cue reach the prompt?'  ({last_name})")
    print(f"  {'threshold':<28} {'says delivered':>26}")
    print(f"  {'-'*28} {'-'*26}")
    sel = [r["cue_selected"] for r in last.values()]
    ev = [bool(r.get("evidence_found")) for r in last.values()]
    print(f"  {'cue_selected (window based)':<28} {str(Proportion(sum(sel), len(sel))):>26}")
    print(f"  {'evidence_found (cosine .5)':<28} {str(Proportion(sum(ev), len(ev))):>26}")
    for th in (0.3, 0.4, 0.5, 0.6):
        lex = [best_overlap(r["carried"], samples[i].evidence) >= th
               for i, r in last.items() if i in samples]
        print(f"  {'lexical overlap >= ' + f'{th:.0%}':<28} "
              f"{str(Proportion(sum(lex), len(lex))):>26}")

    # ---- 2. which metric predicts being right? ----------------------------
    banner("2. Which of them actually predicts a correct answer?")
    print("  A metric that means something should separate correct from wrong.\n")
    print(f"  {'metric':<30} {'score when TRUE':>26} {'score when FALSE':>26}")
    print(f"  {'-'*30} {'-'*26} {'-'*26}")

    def split(flagfn, label):
        yes = [r["label"] == "correct" for i, r in last.items()
               if i in samples and flagfn(i, r)]
        no = [r["label"] == "correct" for i, r in last.items()
              if i in samples and not flagfn(i, r)]
        if yes and no:
            print(f"  {label:<30} {str(Proportion(sum(yes), len(yes))):>26} "
                  f"{str(Proportion(sum(no), len(no))):>26}")

    split(lambda i, r: r["cue_selected"], "cue_selected (window)")
    split(lambda i, r: bool(r.get("evidence_found")), "evidence_found (cosine)")
    split(lambda i, r: best_overlap(r["carried"], samples[i].evidence) >= 0.4,
          "lexical overlap >= 40%")
    print("\n  The metric with the WIDEST gap is the one tracking what matters.")

    # ---- 3. where the window metric lies ----------------------------------
    banner("3. cue_selected = True, but nothing resembling the cue was carried")
    liars = [i for i, r in last.items()
             if i in samples and r["cue_selected"]
             and best_overlap(r["carried"], samples[i].evidence) < 0.25]
    total_sel = sum(1 for r in last.values() if r["cue_selected"])
    print(f"  {len(liars)} of {total_sel} 'delivered' items carry no trace of the cue "
          f"({100*len(liars)/max(1,total_sel):.0f}%)")
    if liars:
        p = Proportion(sum(last[i]["label"] == "correct" for i in liars), len(liars))
        clean = [i for i, r in last.items()
                 if i in samples and r["cue_selected"] and i not in liars]
        q = Proportion(sum(last[i]["label"] == "correct" for i in clean), len(clean))
        print(f"\n  score on those:                    {p}")
        print(f"  score when the cue really is there: {q}")
        print("\n  If the second is far higher, cue_selected is diluted by items where")
        print("  extraction wrote about a DIFFERENT fact from the right window, and the")
        print("  true delivery rate is the smaller number.")
        print("\n  Examples:")
        for i in liars[:5]:
            print(f"\n    #{i}  overlap "
                  f"{best_overlap(last[i]['carried'], samples[i].evidence):.0%}")
            print(f"      GOLD: {(samples[i].evidence.splitlines() or [''])[0][:88]}")
            for n in last[i]["carried"]:
                print(f"      ours: {n[:88]}")

    # ---- 4. redo the identity with the honest metric -----------------------
    banner("4. The identity again, using lexical delivery instead of the window")
    print(f"  {'run':<22} {'recall':>8} {'P(ok|cue)':>11} {'P(ok|no cue)':>13} {'score':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*11} {'-'*13} {'-'*8}")
    for name, rows in runs:
        ids = [i for i in rows if i in samples]
        got = {i: best_overlap(rows[i]["carried"], samples[i].evidence) >= 0.4 for i in ids}
        w = [i for i in ids if got[i]]
        wo = [i for i in ids if not got[i]]
        if not w or not wo:
            continue
        pw = sum(rows[i]["label"] == "correct" for i in w) / len(w)
        pwo = sum(rows[i]["label"] == "correct" for i in wo) / len(wo)
        act = sum(rows[i]["label"] == "correct" for i in ids) / len(ids)
        print(f"  {name:<22} {100*len(w)/len(ids):>7.1f}% {100*pw:>10.1f}% "
              f"{100*pwo:>12.1f}% {100*act:>7.1f}%")
    print("\n  If P(ok|cue) is roughly CONSTANT here while it fell under the window")
    print("  metric, then nothing mysterious is happening: real delivery barely moved,")
    print("  and the earlier 'diminishing returns' story was a measurement artefact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
