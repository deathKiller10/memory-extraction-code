"""
Did prompt A or prompt B write down more of what the questions need?

    python scripts/written_compare.py results/extract_only_v1.json results/extract_only_events.json
    ... --export transfer      also write results/written_transfer.json for paper_numbers.py

Reads two files from scripts/extract_only.py. Free: no API calls, only the
local embedder.

WRITTEN, for one question, means some note in its conversation's store has
cosine >= 0.5 with the gold evidence -- exactly the paper's content measure
(extraction_ceiling.py, "cue is WRITTEN somewhere in the store"), so the number
is read the same way as the headline's 85.0%.

A second, cruder check is printed beside it, because a cosine threshold is one
choice among many: ANSWER IN A NOTE -- at least 60% of the gold answer's
content words appear in one single note. Yes/no answers are skipped.

Three things this prints on purpose:
  * the model each file was extracted with -- the two must match, or the
    comparison is between models, not prompts;
  * the winner by FILE NAME, never "A"/"B" (bug 17);
  * a per-CONVERSATION comparison. The ~1,540 questions come from ten
    conversations, and every question about one conversation shares one store.
    The per-question McNemar test treats them as independent; the
    per-conversation count (how many of the ten each prompt wins) is the
    honest check on that, and the one to quote if the two disagree.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.dataset import LocomoPlus
from bapca.stats import Proportion, mcnemar

THRESHOLD = 0.5
_STOP = set("the and for her his with that this was are from she they them their "
            "has have had into about who what when where which how did does".split())


def content_words(text: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
            if len(w) > 2 and w not in _STOP]


def answer_in_a_note(answer, notes) -> bool | None:
    if not answer:
        return None
    want = content_words(str(answer))
    if not want or set(want) <= {"yes", "not"}:
        return None
    for note in notes:
        have = set(content_words(note))
        if sum(w in have for w in want) / len(want) >= 0.6:
            return True
    return False


_BASE: dict[str, int] = {}


def conversation_id(row, sample) -> int:
    """The row's conversation. extract_only.py files record it; run_system.py
    files (the Cognitive runs) do not, so it is recovered from the item's own
    prompt: its two most frequent speakers name the base conversation. The
    opening lines do not -- a stitched cue can land there, and that split ten
    conversations into fourteen. Checked on the n=100 Cognitive subset: ten
    groups. (2 Oct)"""
    if row.get("conversation") is not None:
        return int(row["conversation"])
    speakers = Counter(re.findall(r"^(\w+) said,", sample.input_prompt, re.M))
    key = "|".join(sorted(name for name, _ in speakers.most_common(2)))
    return _BASE.setdefault(key, len(_BASE))


def load(path: Path) -> dict[int, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(r["sample_index"]): r for r in rows}


def named_verdict(r, name_a: str, name_b: str, alpha: float = 0.05) -> str:
    if r.discordant == 0:
        return "identical on every question"
    leader = name_b if r.only_b > r.only_a else name_a if r.only_a > r.only_b else "neither"
    if r.p_value < alpha:
        return f"different (p={r.p_value:.2g}), better: {leader}"
    return f"no significant difference (p={r.p_value:.2g}); more won by: {leader}"


def main(argv=None, embedder=None, samples=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    export_name = None
    if "--export" in argv:                 # drop the flag and its value first
        k = argv.index("--export")
        export_name = argv[k + 1]
        del argv[k:k + 2]
    if len(argv) != 2:
        print(__doc__)
        return 1
    pa, pb = Path(argv[0]), Path(argv[1])
    export = dict(file_a=pa.name, file_b=pb.name)
    a, b = load(pa), load(pb)
    na, nb = pa.name, pb.name

    print("Files:")
    for name, rows in ((na, a), (nb, b)):
        models = sorted({r.get("model", "?") for r in rows.values()})
        prompts = sorted({r.get("extract", "?") for r in rows.values()})
        convs = len({r.get("conversation") for r in rows.values()})
        print(f"  {name:<34} {len(rows):>5} questions, "
              f"{convs if convs > 1 else '?'} conversations, "
              f"prompt {prompts}, model {models}")
        if models == ["?"]:
            print(f"  {'':<34} (model not recorded in this file -- check HANDOFF)")
    ma = {r.get("model") for r in a.values()}
    mb = {r.get("model") for r in b.values()}
    if ma != mb:
        print("\n  *** THE TWO FILES WERE EXTRACTED WITH DIFFERENT MODELS. ***")
        print("  This compares models, not prompts. Do not report it as a prompt result.")

    shared = sorted(set(a) & set(b))
    print(f"\n  questions in BOTH: {len(shared)}   <- every number below is on these only")
    if not shared:
        return 1

    if samples is None:
        samples = {s.index: s for s in LocomoPlus()}
    if embedder is None:
        from bapca.embeddings import Embedder
        embedder = Embedder()
    from bapca.memory import cosine
    memo: dict[str, object] = {}

    def enc(t: str):
        if t not in memo:
            memo[t] = embedder.encode_one(t)
        return memo[t]

    def written(row, sample) -> bool:
        gold = enc(sample.evidence)
        return max((cosine(gold, enc(n)) for n in row["all_notes"]), default=0.0) >= THRESHOLD

    wa, wb, xa, xb, cat_of, conv_of = {}, {}, {}, {}, {}, {}
    for i in shared:
        s = samples[i]
        wa[i], wb[i] = written(a[i], s), written(b[i], s)
        xa[i], xb[i] = answer_in_a_note(s.answer, a[i]["all_notes"]), \
            answer_in_a_note(s.answer, b[i]["all_notes"])
        cat_of[i], conv_of[i] = s.category, conversation_id(a[i], s)

    print(f"\n{'=' * 74}\nWRITTEN (cosine >= {THRESHOLD} to the gold evidence), paired\n{'=' * 74}")
    print(f"  {'':<16} {na:>28} {nb:>28}")
    groups = [("all", shared)] + [
        (c, [i for i in shared if cat_of[i] == c])
        for c in sorted(set(cat_of.values()))]
    for label, ids in groups:
        if not ids:
            continue
        pa_ = Proportion(sum(wa[i] for i in ids), len(ids))
        pb_ = Proportion(sum(wb[i] for i in ids), len(ids))
        print(f"  {label:<16} {str(pa_):>28} {str(pb_):>28}")
    r = mcnemar([wa[i] for i in shared], [wb[i] for i in shared])
    export.update(
        prompt_a=sorted({str(x.get("extract", "?")) for x in a.values()}),
        prompt_b=sorted({str(x.get("extract", "?")) for x in b.values()}),
        model_a=sorted({str(x.get("model", "?")) for x in a.values()}),
        model_b=sorted({str(x.get("model", "?")) for x in b.values()}),
        n=len(shared), written_a=sum(wa[i] for i in shared),
        written_b=sum(wb[i] for i in shared), only_a=r.only_a, only_b=r.only_b,
        both=r.both, p=r.p_value, threshold=THRESHOLD,
        per_category={c: dict(n=len(ids), written_a=sum(wa[i] for i in ids),
                              written_b=sum(wb[i] for i in ids))
                      for c, ids in groups[1:] if ids})
    print(f"\n  per question:  {named_verdict(r, na, nb)}")
    print(f"                 only {na}: {r.only_a}   only {nb}: {r.only_b}   both: {r.both}")

    print(f"\n{'=' * 74}\nPer CONVERSATION (the clustering check)\n{'=' * 74}")
    by_conv = defaultdict(list)
    for i in shared:
        by_conv[conv_of[i]].append(i)
    wins_a = wins_b = ties = 0
    for c in sorted(by_conv):
        ids = by_conv[c]
        ra = sum(wa[i] for i in ids) / len(ids)
        rb = sum(wb[i] for i in ids) / len(ids)
        mark = "=" if abs(ra - rb) < 1e-9 else (">" if ra > rb else "<")
        wins_a += mark == ">"
        wins_b += mark == "<"
        ties += mark == "="
        print(f"  conversation {c:>2}  ({len(ids):>3} q)   {100*ra:5.1f}%  {mark}  {100*rb:5.1f}%")
    print(f"\n  conversations won:  {na}: {wins_a}   {nb}: {wins_b}   tied: {ties}")
    decided = wins_a + wins_b
    if decided:
        from math import comb
        k = max(wins_a, wins_b)
        p = min(1.0, 2 * sum(comb(decided, j) for j in range(k, decided + 1)) / 2 ** decided)
        print(f"  sign test over {decided} decided conversations: p = {p:.3g}")
        print("  If this and the per-question test disagree, quote THIS one.")
        export.update(sign_p=p)
    export.update(conversations=len(by_conv), conv_won_a=wins_a,
                  conv_won_b=wins_b, conv_ties=ties)

    print(f"\n{'=' * 74}\nANSWER IN A NOTE (cruder check; yes/no answers skipped)\n{'=' * 74}")
    ids = [i for i in shared if xa[i] is not None and xb[i] is not None]
    print(f"  checkable questions: {len(ids)}")
    if ids:
        print(f"  {na:<34} {Proportion(sum(xa[i] for i in ids), len(ids))}")
        print(f"  {nb:<34} {Proportion(sum(xb[i] for i in ids), len(ids))}")
        r2 = mcnemar([xa[i] for i in ids], [xb[i] for i in ids])
        print(f"  per question:  {named_verdict(r2, na, nb)}")

    if export_name:
        out = Path("results") / f"written_{export_name}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(export, indent=1), encoding="utf-8")
        print(f"\n  Exported to {out} for scripts/paper_numbers.py")

    print("\n  WRITTEN is a ceiling, not a score: it says the fact was stored, not that")
    print("  the system would select it or answer with it. It is the measure of the")
    print("  extraction PROMPT, which is the thing under test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
