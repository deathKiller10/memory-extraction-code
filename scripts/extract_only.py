"""
Run ONLY the extraction stage over the non-Cognitive LoCoMo-Plus splits, and
save what each prompt wrote down. No answering, no re-ranking, no judging.

    python scripts/extract_only.py --extract events --dry-run
    python scripts/extract_only.py --extract events
    python scripts/extract_only.py --extract v1
    python scripts/written_compare.py results/extract_only_v1.json results/extract_only_events.json

WHY THIS AND NOT run_system.py (27 Sep, HANDOFF 16.4)
The question a reviewer will ask is whether the corrected extraction prompt
only works on the split it was written around. That is a question about what
the prompt WRITES DOWN, and on these splits the answer can be measured for every
question at once:

  * All 1,540 answerable non-Cognitive questions are asked about the same ten
    conversations. Once the question line is removed, every question about a
    conversation sees the SAME memory store, so each prompt is paid for once
    per conversation, not once per question.
  * run_system.py on these splits cuts windows at the DATE lines -- whole
    sessions of ~27 turns -- and extraction writes ONE note per window. That
    ceiling (about 26 notes for a whole conversation) hits both prompts alike
    and hides any difference between them. Measured on the first 75 single-hop
    items: the gold answer was in the store on ~15 of 75.

So windows here are cut EXACTLY as on the Cognitive split: the DATE and
CONVERSATION header lines are dropped (the Cognitive prompts do not carry
them) and the remaining turns are content-defined, target 12 turns -- the
same `_content_defined` the headline run used. The comparison then differs
from the paper's setup in the split and nothing else.

Removing the question: `strip_trigger` deliberately refuses triggers under
five words, which left 13 short questions ("How old is Max?") inside their own
store. Here the question is always the literal last line `Question: <trigger>`,
so exactly that line is removed and nothing else; the script asserts that the
questions then collapse onto a small number of conversations and prints the
count.

Output: results/extract_only_<extract>.json, one row per question:
    sample_index, category, extract, conversation (0..9), all_notes
Safe to stop and resume: every extraction call is cached on disk, and a
conversation is written to the file only when all its windows are done.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.dataset import LocomoPlus
from bapca.pipeline import (EXTRACT_PROMPTS, Window, _DATE_LINE, _content_defined,
                            extract_notes)

RESULTS_DIR = Path("results")
CATEGORIES = ["single-hop", "multi-hop", "temporal", "common-sense"]
WINDOW_TURNS = 12

# Same calibration as run_system.py: 778 tokens measured on 529-token windows.
TOKENS_PER_EXTRACTION = 778
MEASURED_WINDOW_TOKENS = 529
GROQ_TPD = 200_000


def conversation_text(sample) -> str:
    """The conversation with the question line removed -- that line only."""
    lines = sample.input_prompt.rstrip("\n").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    question = "Question: " + sample.trigger.strip()
    if lines and lines[-1].strip() == question:
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines)


def windows_like_cognitive(text: str) -> list[Window]:
    lines = [l for l in text.splitlines()
             if l.strip() and not _DATE_LINE.match(l) and l.strip() != "CONVERSATION:"]
    return _content_defined(lines, WINDOW_TURNS)


def group_by_conversation(samples):
    """{conversation text: [samples]} in a stable order."""
    groups: dict[str, list] = {}
    for s in samples:
        groups.setdefault(conversation_text(s), []).append(s)
    return groups


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", choices=sorted(EXTRACT_PROMPTS), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-conversations", type=int, default=20,
                        help="refuse to run if the questions do not collapse onto at "
                             "most this many conversations (a sign the question line "
                             "was not removed)")
    args = parser.parse_args()

    data = LocomoPlus()
    samples = [s for s in data if s.category in CATEGORIES and s.evidence.strip()]
    groups = group_by_conversation(samples)
    print(f"Loaded {len(data)} samples; {len(samples)} answerable non-Cognitive "
          f"questions with evidence, over {len(groups)} conversations.")
    if len(groups) > args.max_conversations:
        print(f"  Expected about ten conversations, got {len(groups)}: the question "
              "line was not removed from some items. Stopping.")
        return 1

    out = RESULTS_DIR / f"extract_only_{args.extract}.json"
    rows = []
    if out.exists():
        rows = json.loads(out.read_text(encoding="utf-8"))
    done_convs = {r["conversation"] for r in rows}

    windows = {i: windows_like_cognitive(text) for i, text in enumerate(groups)}
    todo = [i for i in windows if i not in done_convs]
    n_windows = sum(len(windows[i]) for i in todo)
    texts = [w.text for i in todo for w in windows[i]]
    mean = sum(len(t) // 4 for t in texts) / max(1, len(texts))
    per_call = round(TOKENS_PER_EXTRACTION - MEASURED_WINDOW_TOKENS + mean)
    tokens = n_windows * per_call

    banner(f"Extraction only -- prompt '{args.extract}'")
    print(f"  conversations: {len(windows)}   already finished: {len(done_convs)}")
    print(f"  windows to extract: {n_windows:,}  (content-defined, target "
          f"{WINDOW_TURNS} turns, as on the Cognitive split)")
    print(f"  Groq tokens: ~{tokens:,}  ({per_call}/extraction, scaled from the 778 "
          "measured on Cognitive windows; cached windows cost nothing, so this is "
          "an upper bound)")
    print(f"  at {GROQ_TPD:,}/day: about {tokens / GROQ_TPD:.1f} days")
    print(f"  results file: {out.name}")
    if args.dry_run:
        return 0

    from bapca.llm import DailyQuotaExhausted, TransientProviderError, judge
    llm = judge(verbose=True)      # extraction runs on the judge provider, as in run_system
    print(f"  extraction model: {llm.model}")

    for i in todo:
        group = list(groups.values())[i]
        try:
            notes = extract_notes(llm, windows[i], prompt=EXTRACT_PROMPTS[args.extract])
        except DailyQuotaExhausted as exc:
            banner("Out of quota for today -- this is a pause, not a failure")
            print(f"  {exc}\n")
            print(f"  {len(done_convs)} of {len(windows)} conversations are finished in "
                  f"{out.name}; every extraction made today is cached.")
            print("  Run the identical command after the reset:\n")
            print(f"    !python scripts/extract_only.py --extract {args.extract}\n")
            return 0
        except TransientProviderError as exc:
            print(f"  conversation {i}: provider hiccup -- {exc}\n"
                  "  Re-run the identical command in a few minutes; it resumes.")
            return 0
        all_notes = [f"[{n.mem_type.value}] {n.text}" for n in notes]
        for s in group:
            rows.append(dict(sample_index=s.index, category=s.category,
                             extract=args.extract, conversation=i,
                             model=llm.model, all_notes=all_notes))
        done_convs.add(i)
        RESULTS_DIR.mkdir(exist_ok=True)
        out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"  conversation {i + 1}/{len(windows)}: {len(windows[i])} windows -> "
              f"{len(notes)} notes, {len(group)} questions recorded", flush=True)

    banner("Done")
    print(f"  {len(rows)} questions over {len(done_convs)} conversations in {out.name}")
    print(f"  model: {llm.model}  -- both prompts must show the SAME model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
