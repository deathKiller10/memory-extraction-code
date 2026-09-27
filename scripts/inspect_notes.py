"""
Are the notes in the prompt about the right person?

Costs nothing: reads the saved results and the dataset. No API calls.

    python scripts/inspect_notes.py            # measurement + 8 examples
    python scripts/inspect_notes.py --dump 25

FIRST VERSION OF THIS SCRIPT FAILED, and how it failed matters. It looked for
"Name:" with an ASCII colon. The data uses a FULLWIDTH colon (U+FF1A), and the
trigger's speaker is the anonymous label "A", not a name. bapca/pipeline.py's
locate_evidence_window already splits on [:\uff1a]; this script did not.

So the target person is taken from the EVIDENCE line, which names them
("Joanna\uff1aSince my cousin got diagnosed..."), and the vocabulary of real names
is built from all 100 evidence lines at once. That does not depend on the
input_prompt's format at all. Section 0 dumps the raw text with repr() so the
format is visible rather than assumed -- read it before trusting the rest.

The question: EXTRACT_PROMPT anchors each note on "the FIRST speaker" of a
window, but _content_defined picks window boundaries from a hash of each line,
so "first" is a different person from window to window. Item #1998 carried two
notes about John and one about Maria. This measures how often that happens.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.dataset import LocomoPlus
from bapca.stats import Proportion, mcnemar

RESULTS = Path("results")
DEFAULT = RESULTS / "system_n100_seed42_llm3_d1.json"

COLONS = ":\uff1a"                       # ASCII and fullwidth
_PREFIX = re.compile(rf"^\s*([^\s{COLONS}][^{COLONS}]{{0,30}}?)\s*[{COLONS}]")
_STOP = {"DATE", "CONVERSATION", "SPEAKER", "USER", "ASSISTANT", "NOTE", "A", "B"}


def prefix_name(line: str) -> str | None:
    """The 'Name:' or 'Name\uff1a' label starting this line, if it looks like a name."""
    m = _PREFIX.match(line or "")
    if not m:
        return None
    name = m.group(1).strip()
    if not name or name.upper() in _STOP or len(name) > 24:
        return None
    return name if re.fullmatch(r"[A-Z][A-Za-z'\-. ]*", name) else None


_STOPWORDS = set("""a an and are as at be been but by for from had has have he her hers him his
i if in into is it its me my of on or our she that the their them they this to was we were
what when which who will with you your not no so just still really very about after before
""".split())


def content_words(text: str) -> set[str]:
    body = re.sub(r"^\[\w+\]\s*", "", text or "").lower()
    body = re.sub(r"[^a-z0-9 ]+", " ", body)
    return {w for w in body.split() if len(w) > 2 and w not in _STOPWORDS}


def echo_score(note: str, trigger: str) -> float:
    """
    How much of this note is just the trigger said back?

    input_prompt carries the trigger as its LAST line (dataset.py: "trigger
    appended last"), and run_system.py segments input_prompt, so extraction
    sees the query and can write a standing note describing it. Such a note
    looks maximally relevant to the query -- because it IS the query -- and
    wins a slot while carrying no memory at all.
    """
    n, t = content_words(note), content_words(trigger)
    return len(n & t) / len(n) if n else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DEFAULT)
    ap.add_argument("--dump", type=int, default=8)
    args = ap.parse_args()

    if not args.results.exists():
        print(f"Not found: {args.results}")
        return 1
    rows = {r["sample_index"]: r for r in json.loads(args.results.read_text(encoding="utf-8"))
            if r["label"] in ("correct", "wrong")}
    data = LocomoPlus()
    samples = {s.index: s for s in data.subset(100, seed=42)}

    # ---- 0. what does the data actually look like? -------------------------
    print("=" * 74)
    print("0. Raw format -- repr() so nothing is assumed")
    print("=" * 74)
    first = sorted(rows)[0]
    s0 = samples[first]
    lines = [ln for ln in s0.input_prompt.splitlines() if ln.strip()]
    print(f"  #{first}: {len(lines)} non-blank lines in input_prompt")
    for ln in lines[:8]:
        print(f"    {ln[:96]!r}")
    print("    ...")
    for ln in lines[-3:]:
        print(f"    {ln[:96]!r}")
    print(f"\n  trigger:  {s0.trigger[:96]!r}")
    print(f"  evidence: {(s0.evidence.splitlines() or [''])[0][:96]!r}")

    tail_hits = 0
    for i, sm in samples.items():
        body = content_words(sm.trigger)
        if not body:
            continue
        last = " ".join(ln for ln in sm.input_prompt.splitlines() if ln.strip())[-600:]
        if len(body & content_words(last)) / len(body) >= 0.6:
            tail_hits += 1
    print(f"\n  items whose TRIGGER appears at the end of input_prompt: "
          f"{tail_hits}/{len(samples)}")
    print("  run_system.py segments input_prompt, so extraction reads the query too.")

    # ---- 1. the name vocabulary --------------------------------------------
    evidence_name: dict[int, str] = {}
    for i, s in samples.items():
        for line in s.evidence.splitlines():
            n = prefix_name(line)
            if n:
                evidence_name[i] = n
                break
    vocab = sorted({n for n in evidence_name.values()}, key=len, reverse=True)

    prompt_labels = Counter()
    for s in samples.values():
        for line in s.input_prompt.splitlines():
            n = prefix_name(line)
            if n:
                prompt_labels[n] += 1

    print("\n" + "=" * 74)
    print("1. Names")
    print("=" * 74)
    print(f"  items whose evidence names a speaker: {len(evidence_name)}/{len(samples)}")
    print(f"  distinct names in evidence ({len(vocab)}): {', '.join(vocab[:20])}")
    print(f"  name-like labels found in input_prompt: "
          f"{dict(prompt_labels.most_common(8)) or 'NONE -- prompt uses A:/B: labels'}")
    if not evidence_name:
        print("\n  Could not name anyone. Read section 0 and tell Claude the format.")
        return 0

    def names_in(text: str) -> set[str]:
        body = re.sub(r"^\[\w+\]\s*", "", text)
        return {n for n in vocab if re.search(rf"\b{re.escape(n)}\b", body)}

    # ---- 2. measurement ----------------------------------------------------
    on = off = anon = 0
    dirty_items, clean_items = [], []
    multi = Counter()

    for i, r in rows.items():
        target = evidence_name.get(i)
        if not target:
            continue
        seen_names, off_here = set(), 0
        for note in r["carried"]:
            found = names_in(note)
            seen_names |= found
            if not found:
                anon += 1
            elif target in found:
                on += 1
            else:
                off += 1
                off_here += 1
        multi[len(seen_names)] += 1
        (dirty_items if off_here else clean_items).append(i)

    total = on + off + anon
    print("\n" + "=" * 74)
    print("2. Are the carried notes about the person the cue is about?")
    print("=" * 74)
    print(f"  notes naming the TARGET person: {Proportion(on, total)}")
    print(f"  notes naming SOMEONE ELSE:      {Proportion(off, total)}")
    print(f"  notes naming nobody:            {Proportion(anon, total)}")
    print(f"\n  distinct people per 3-note prompt: "
          f"{dict(sorted(multi.items()))}   (1 is clean, 2+ is mixed)")
    print(f"  items carrying >=1 wrong-person note: "
          f"{Proportion(len(dirty_items), len(dirty_items) + len(clean_items))}")

    if clean_items and dirty_items:
        c = Proportion(sum(rows[i]["label"] == "correct" for i in clean_items), len(clean_items))
        d = Proportion(sum(rows[i]["label"] == "correct" for i in dirty_items), len(dirty_items))
        print(f"\n  score, no wrong-person note: {c}")
        print(f"  score, >=1 wrong-person note: {d}")
        print(f"  difference: {100 * (d.rate - c.rate):+.1f} points")
        print("  Observational, not causal: items differ. Treat as a lead, not a result.")

    # ---- 2b. is the model just quoting the query back? ---------------------
    print("\n" + "=" * 74)
    print("2b. Notes that are the TRIGGER restated, not a memory")
    print("=" * 74)
    by_slot = {0: 0, 1: 0, 2: 0}
    echoes = 0
    items_with_echo = []
    for i, r in rows.items():
        sm = samples.get(i)
        if not sm:
            continue
        hit = False
        for slot, note in enumerate(r["carried"]):
            if echo_score(note, sm.trigger) >= 0.6:
                echoes += 1
                hit = True
                if slot in by_slot:
                    by_slot[slot] += 1
        if hit:
            items_with_echo.append(i)
    n_notes = sum(len(r["carried"]) for r in rows.values())
    print(f"  carried notes that restate the trigger (>=60% of their content"
          f" words):\n    {Proportion(echoes, n_notes)}")
    print(f"  items carrying at least one: {Proportion(len(items_with_echo), len(rows))}")
    print(f"  which slot (0 = the re-ranker's TOP pick): {by_slot}")
    if items_with_echo:
        clean = [rows[i]["label"] == "correct" for i in rows if i not in items_with_echo]
        dirty = [rows[i]["label"] == "correct" for i in items_with_echo]
        sel_clean = [rows[i]["cue_selected"] for i in rows if i not in items_with_echo]
        sel_dirty = [rows[i]["cue_selected"] for i in items_with_echo]
        print(f"\n  score,  no echo note: {Proportion(sum(clean), len(clean))}")
        print(f"  score, has echo note: {Proportion(sum(dirty), len(dirty))}")
        print(f"  cue reached prompt,  no echo: {Proportion(sum(sel_clean), len(sel_clean))}")
        print(f"  cue reached prompt, has echo: {Proportion(sum(sel_dirty), len(sel_dirty))}")
        print("\n  Every echo note burns one of the three slots on a paraphrase of the")
        print("  question. If the echo rate is high, effective top_k is 2, not 3, and")
        print("  the cheapest available fix is to stop segmenting the trigger.")
        print("\n  Examples:")
        for i in items_with_echo[:4]:
            sm = samples[i]
            print(f"\n    #{i}  trigger: {sm.trigger[:88]}")
            for note in rows[i]["carried"]:
                sc = echo_score(note, sm.trigger)
                if sc >= 0.6:
                    print(f"      ECHO {sc:.0%}: {note[:96]}")

    # ---- 3. read them ------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"3. {args.dump} items in full")
    print("=" * 74)
    for i in sorted(rows)[:args.dump]:
        r, s = rows[i], samples.get(i)
        target = evidence_name.get(i)
        print(f"\n  #{i}  {r['label']}  "
              f"({'CUE PRESENT' if r['cue_selected'] else 'cue missing'})  target={target}")
        if s:
            print(f"    trigger:  {s.trigger[:100]}")
            print(f"    evidence: {(s.evidence.splitlines() or [''])[0][:100]}")
        for note in r["carried"]:
            found = names_in(note)
            mark = "  " if target in found else ("!!" if found else " ?")
            print(f"    {mark} {note}")
    print("\n  !! = names a different person    ? = names nobody")
    return 0


if __name__ == "__main__":
    sys.exit(main())
