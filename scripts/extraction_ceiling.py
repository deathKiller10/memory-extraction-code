"""
Is the cue ever written down at all? And if it is, why is it not selected?

Free of API calls. Uses the local MiniLM embedder (no network after the first
model download) and the notes now recorded in every row.

    python scripts/extraction_ceiling.py

WHY THIS, AND WHY NOW. The pipeline reports two stages:

    cue_extracted  a note was extracted from the WINDOW holding the cue   91%
    cue_selected   a note from that window reached the prompt             38%

true_recall.py showed both are provenance, not content: 9 of 38 "delivered"
items carry nothing resembling the cue, because a window is ~12 turns and
extraction writes ONE note for it -- often about that window's dominant topic
instead of the cue line. So the real 91% is unknown and is certainly lower.

This measures the content version of both stages, using the same cosine >= 0.5
test as the `evidence_found` field, which true_recall.py found to be the best
predictor of a correct answer (a 48-point gap, wider than the window metric's
43.8 and far wider than lexical overlap's 30.9).

    written   does ANY extracted note resemble the gold cue?
    chosen    did one of those reach the prompt?

The split decides where the remaining quota goes:

  low `written`  -> extraction is the bottleneck. One note per 12-turn window
                    cannot carry a single inserted line. Let EXTRACT_PROMPT
                    return up to 3 notes per window: same ~569 calls, one day.
  high `written`, low `chosen` -> the re-ranker is the bottleneck, and the notes
                    it needs are already sitting in the store.

Requires `all_notes`, added to the row on 31 Aug, so it works on the
`_notrig` run and any later one.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.stats import Proportion

RESULTS = Path("results")
DEFAULT = RESULTS / "system_n100_seed42_llm3_d1_notrig.json"
THRESHOLD = 0.5


def banner(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def subset_size(path: Path, n_rows: int) -> tuple[int, str]:
    """How many items the run covered, and where that number came from.

    BUG 21 (18 Sep). This used to be the literal 100. Run against the 401-item
    result file it therefore analysed only the 100-item seed-42 subset, dropped
    301 rows without a word, and printed `items 100` immediately above window
    figures computed over all 401 -- two different denominators, one table.
    The exported ceiling_*.json then carried n=100 into the paper as though it
    described the full run.

    Same family as bug 18: an analysis input decided by a literal rather than
    by the file being analysed. The size now comes from the file, and the
    caller reports every row it had to drop.
    """
    if "--n" in sys.argv:
        return int(sys.argv[sys.argv.index("--n") + 1]), "--n on the command line"
    m = re.search(r"_n(\d+)_", path.name)
    if m:
        return int(m.group(1)), f"the filename ({path.name})"
    return n_rows, "the number of scored rows in the file"


def main() -> int:
    argv = sys.argv[1:]
    if "--n" in argv:                      # drop --n and its value first, so
        i = argv.index("--n")              # the number is never mistaken for
        del argv[i:i + 2]                  # the result-file path
    positional = [a for a in argv if not a.startswith("--")]
    path = Path(positional[0]) if positional else DEFAULT
    print(f"Reading: {path}")
    if not path.exists():
        print(f"Not found: {path}")
        return 1
    rows = {r["sample_index"]: r for r in json.loads(path.read_text(encoding="utf-8"))
            if r["label"] in ("correct", "wrong")}
    if not any("all_notes" in r for r in rows.values()):
        print(f"{path.name} has no `all_notes` field -- it predates the 31 Aug change.")
        print("Re-run with the current run_system.py to use this script.")
        return 1

    from bapca.dataset import LocomoPlus
    from bapca.embeddings import Embedder
    from bapca.memory import cosine

    n_run, source = subset_size(path, len(rows))
    print(f"Dataset subset: {n_run} items, seed 42   (taken from {source})")
    samples = {s.index: s for s in LocomoPlus().subset(n_run, seed=42)}

    # Loud about anything it cannot analyse, rather than quietly shrinking n.
    missing_sample = [i for i in rows if i not in samples]
    missing_notes = [i for i, r in rows.items() if "all_notes" not in r]
    if missing_sample:
        print(f"\n  !! {len(missing_sample)} of {len(rows)} scored rows are NOT in "
              f"the {n_run}-item subset and will be DROPPED.")
        print(f"     first few: {sorted(missing_sample)[:8]}")
        print("     If that is not what you meant, pass --n <the run's --n>.")
    if missing_notes:
        print(f"\n  !! {len(missing_notes)} rows have no `all_notes` field and "
              "will be DROPPED.")

    emb = Embedder()

    print(f"Embedding notes from {path.name} (local model, no API calls)...")
    written, chosen, both = [], [], []
    for i, r in rows.items():
        s = samples.get(i)
        if not s or "all_notes" not in r:
            continue
        gold = emb.encode_one(s.evidence)
        w = max((cosine(gold, emb.encode_one(n)) for n in r["all_notes"]), default=0.0)
        c = max((cosine(gold, emb.encode_one(n)) for n in r["carried"]), default=0.0)
        written.append((i, w >= THRESHOLD))
        chosen.append((i, c >= THRESHOLD))
        both.append((i, w >= THRESHOLD, c >= THRESHOLD))

    banner(f"The pipeline, by CONTENT rather than provenance (cosine >= {THRESHOLD})")
    nw = sum(1 for _, w in written if w)
    nc = sum(1 for _, c in chosen if c)
    n = len(written)
    print(f"  items                                        {n}")
    print(f"  cue is WRITTEN somewhere in the store         {Proportion(nw, n)}")
    print(f"  cue REACHES the prompt                        {Proportion(nc, n)}")
    # These used to be the literals 91% and 38% -- v1's numbers, printed
    # unchanged next to the events run's 86% and 69% as though they described
    # it. Same family as bug 12: a script quietly reporting a previous
    # experiment. Read them from the file being analysed. (4 Sep)
    win_ext = sum(1 for r in rows.values() if r.get("cue_extracted"))
    win_sel = sum(1 for r in rows.values() if r.get("cue_selected"))
    print(f"\n  this run's cue_extracted (window)             "
          f"{Proportion(win_ext, len(rows))}")
    print(f"  this run's cue_selected  (window)             "
          f"{Proportion(win_sel, len(rows))}")
    print("\n  The window figures count a note from the right neighbourhood. These")
    print("  count a note that resembles the cue itself.")

    banner("Where the loss is")
    if nw:
        print(f"  of the {nw} items where the cue IS written down,")
        print(f"  it reached the prompt in {Proportion(sum(1 for _, w, c in both if w and c), nw)}")
        print("\n  That is the re-ranker's true job: it sees ~29 notes and must find this")
        print("  one. Chance at top-3 of 29 is about 10%.")
    lost = n - nw
    print(f"\n  and on {lost} items the cue was NEVER written down at all,")
    print("  so no selector however good could have retrieved it.")
    print("\n  BUDGET: extraction can add at most "
          f"{100*lost/n:.0f} points of recall; the re-ranker at most "
          f"{100*(nw - nc)/n:.0f}.")

    banner("Score by stage")
    for label, ids in [
        ("cue written AND chosen", [i for i, w, c in both if w and c]),
        ("cue written, NOT chosen", [i for i, w, c in both if w and not c]),
        ("cue never written", [i for i, w, c in both if not w]),
    ]:
        if ids:
            p = Proportion(sum(rows[i]["label"] == "correct" for i in ids), len(ids))
            print(f"  {label:<26} {len(ids):>4} items   {p}")
    print("\n  'written, not chosen' is the recoverable population: the note exists,")
    print("  the system simply failed to pick it.")

    # --- the human check on the metric -----------------------------------
    #
    # 2 Sep. `written` jumped from 48.5% (v1) to 100% on the first 10 items of
    # the events run, and `evidence_found` is a cosine against the gold cue.
    # The events prompt asks for "what happened AND what they now do", which is
    # the SHAPE of the gold cue, so notes could score higher simply by matching
    # its form rather than its content. Cosine cannot tell those apart. Reading
    # them can, so print them and read them.
    banner("Cue WRITTEN -- read the note against the gold cue and judge for yourself")
    hits = [i for i, w, c in both if w][:6]
    if not hits:
        print("  (none)")
    for i in hits:
        s_ = samples[i]
        gold = emb.encode_one(s_.evidence)
        ranked = sorted(rows[i]["all_notes"],
                        key=lambda t: cosine(gold, emb.encode_one(t)), reverse=True)
        chosen_flag = "chosen" if any(c for j, w, c in both if j == i and c) else "NOT chosen"
        print(f"\n  #{i}   ({chosen_flag}, judged {rows[i]['label']})")
        print(f"    GOLD cue: {(s_.evidence.splitlines() or [''])[0][:100]}")
        print(f"    best note: {cosine(gold, emb.encode_one(ranked[0])):.2f}  {ranked[0][:100]}")
    print("\n  Ask of each pair: is the note the SAME FACT as the cue, or merely the")
    print("  same shape -- an event plus an adaptation, about something else? If the")
    print("  second, `written` is measuring the prompt's format and not its content.")

    banner("Cue written but NOT chosen -- the recoverable population")
    stuck = [i for i, w, c in both if w and not c][:4]
    if not stuck:
        print("  (none -- the re-ranker found every cue that was written)")
    for i in stuck:
        s_ = samples[i]
        gold = emb.encode_one(s_.evidence)
        print(f"\n  #{i}   (judged {rows[i]['label']})")
        print(f"    GOLD cue: {(s_.evidence.splitlines() or [''])[0][:100]}")
        best = max(rows[i]["all_notes"], key=lambda t: cosine(gold, emb.encode_one(t)))
        print(f"    in the store, unused: {cosine(gold, emb.encode_one(best)):.2f}  {best[:88]}")
        for t in rows[i]["carried"]:
            print(f"    carried instead:      {cosine(gold, emb.encode_one(t)):.2f}  {t[:88]}")

    banner("Cue never written -- what did extraction write instead?")
    misses = [i for i, w, c in both if not w][:4]
    if not misses:
        print("  (none -- extraction wrote something resembling the cue every time)")
    for i in misses:
        s = samples[i]
        print(f"\n  #{i}")
        print(f"    GOLD cue: {(s.evidence.splitlines() or [''])[0][:92]}")
        gold = emb.encode_one(s.evidence)
        ranked = sorted(rows[i]["all_notes"],
                        key=lambda t: cosine(gold, emb.encode_one(t)), reverse=True)
        print(f"    closest {min(3, len(ranked))} of {len(rows[i]['all_notes'])} notes in the store:")
        for t in ranked[:3]:
            print(f"      {cosine(gold, emb.encode_one(t)):.2f}  {t[:84]}")
    print("\n  If the closest note is a vague trait while the cue is a concrete event,")
    print("  EXTRACT_PROMPT is generalising the wrong way and that is the thing to fix.")

    # --export writes the figures where paper_numbers.py can read them, so the
    # paper never has these typed into its prose. They cannot be recomputed on
    # a laptop (they need the sentence-transformers embedder), which is exactly
    # why they have to be exported rather than remembered.
    if "--export" in sys.argv:
        out = RESULTS / f"ceiling_{path.stem}.json"
        written_and_chosen = [i for i, w, c in both if w and c]
        written_not = [i for i, w, c in both if w and not c]
        never = [i for i, w, c in both if not w]
        def score(ids):
            return (sum(rows[i]["label"] == "correct" for i in ids) / len(ids)) if ids else 0.0
        out.write_text(json.dumps({
            "run": path.name, "threshold": THRESHOLD, "items": n,
            "written": nw, "chosen": nc,
            "chosen_given_written": (sum(1 for _, w, c in both if w and c) / nw) if nw else 0.0,
            "never_written": len(never),
            "n_written_and_chosen": len(written_and_chosen),
            "n_written_not_chosen": len(written_not),
            "score_written_and_chosen": score(written_and_chosen),
            "score_written_not_chosen": score(written_not),
            "score_never_written": score(never),
            "budget_extraction_points": 100 * len(never) / n,
            "budget_reranker_points": 100 * (nw - nc) / n,
        }, indent=1), encoding="utf-8")
        print(f"\n  Exported to {out} for scripts/paper_numbers.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
