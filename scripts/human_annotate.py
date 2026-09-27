"""
Label responses by hand. This is the gold standard, and it is free.

    python scripts/human_annotate.py --annotator annotator2 --n 60
    python scripts/human_annotate.py --report        # once you have both sets

WHY THIS MATTERS MORE THAN ANOTHER LLM JUDGE
--------------------------------------------
Our automated judges disagreed with each other on 5 of the first 9 items --
barely better than chance -- and both marked "no memory" responses correct far
more often than seems possible for a model that was shown nothing. Adding a
third LLM would not settle that. Two humans would.

LoCoMo-Plus validated their judge with exactly this: two human annotators, and
agreement reported between each annotator and the judge. Doing the same makes
our numbers defensible, and it is the one experiment on this project that no
rate limit can interfere with.

BLINDING
--------
Items are shuffled and the condition is hidden. You are shown only the cue, the
user's message, and a response. If you knew which condition produced it you
would unconsciously grade the full-context ones more generously, and the whole
exercise would be worthless.

Split the work: run it twice with different --annotator names, ideally without
discussing items in between. Then --report gives you inter-annotator agreement
(how consistent you two are) and agreement with the automatic judge (whether it
can be trusted).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.dataset import LocomoPlus
from bapca.evaluation import CONDITIONS
from bapca.stats import Proportion, cohens_kappa

RESULTS_DIR = Path("results")
ANNOTATION_DIR = RESULTS_DIR / "annotations"

PROMPT = """
Does this response reflect or use the earlier note?

  [y] yes  - it clearly or implicitly acts on what the person said before
  [n] no   - it does not show that link
  [s] skip - genuinely cannot tell
  [q] quit - save and stop (you can resume later)
"""


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def build_queue(results: list[dict], per_condition: int, seed: int) -> list[dict]:
    """Stratified by condition, then shuffled so the order gives nothing away."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        if row["label"] in ("correct", "wrong"):
            grouped[row["condition"]].append(row)

    rng = random.Random(seed)
    queue: list[dict] = []
    for condition in CONDITIONS:
        rows = grouped.get(condition.key, [])
        queue.extend(rows if len(rows) <= per_condition else rng.sample(rows, per_condition))
    rng.shuffle(queue)
    return queue


def annotate(args) -> int:
    results = json.loads(args.results.read_text(encoding="utf-8"))
    queue = build_queue(results, args.n // len(CONDITIONS), args.seed)

    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    out = ANNOTATION_DIR / f"{args.annotator}.json"
    done: dict[str, str] = json.loads(out.read_text()) if out.exists() else {}

    data = LocomoPlus()
    by_index = {s.index: s for s in data}

    remaining = [r for r in queue if f"{r['condition']}:{r['sample_index']}" not in done]
    banner(f"{len(done)} already labelled, {len(remaining)} to go")
    print("The condition is hidden on purpose. Judge only what you see.")
    print(PROMPT)

    for i, row in enumerate(remaining, 1):
        sample = by_index[row["sample_index"]]
        key = f"{row['condition']}:{row['sample_index']}"

        print(f"\n{'-' * 74}\n[{i}/{len(remaining)}]  item {row['sample_index']}")
        print(f"\nEarlier note (the cue):\n  {sample.evidence}")
        print(f"\nTheir message now:\n  {sample.trigger}")
        print(f"\nThe response:\n  {row['prediction']}")

        while True:
            try:
                answer = input("\n  reflects the note? [y/n/s/q] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "q"
            if answer in ("y", "n", "s", "q"):
                break
            print("  please type y, n, s or q")

        if answer == "q":
            break
        done[key] = {"y": "correct", "n": "wrong", "s": "skip"}[answer]
        out.write_text(json.dumps(done, indent=1), encoding="utf-8")

    print(f"\nSaved {len(done)} labels to {out}")
    if len(done) < len(queue):
        print(f"{len(queue) - len(done)} left. Re-run the same command to continue.")
    return 0


def report(args) -> int:
    results = json.loads(args.results.read_text(encoding="utf-8"))
    machine = {f"{r['condition']}:{r['sample_index']}": r for r in results}

    files = sorted(ANNOTATION_DIR.glob("*.json"))
    if not files:
        print(f"No annotations in {ANNOTATION_DIR}. Run with --annotator <name> first.")
        return 1

    sets = {f.stem: json.loads(f.read_text()) for f in files}
    for name, labels in sets.items():
        usable = [v for v in labels.values() if v in ("correct", "wrong")]
        print(f"  {name}: {len(labels)} labelled ({len(usable)} usable, "
              f"{len(labels) - len(usable)} skipped)")

    # ---- how consistent are the two humans with each other? ----------------
    if len(sets) >= 2:
        banner("Inter-annotator agreement")
        names = list(sets)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = sets[names[i]], sets[names[j]]
                shared = [k for k in a if k in b
                          and a[k] in ("correct", "wrong") and b[k] in ("correct", "wrong")]
                if not shared:
                    print(f"  {names[i]} vs {names[j]}: no overlapping items")
                    continue
                result = cohens_kappa([a[k] == "correct" for k in shared],
                                      [b[k] == "correct" for k in shared])
                print(f"  {names[i]} vs {names[j]}: n={result.n}, "
                      f"raw {100 * result.raw:.1f}%, kappa {result.kappa:.3f} "
                      f"({result.reading()})")
        print("\n  If you two cannot agree, the task itself is ambiguous and no LLM")
        print("  judge can be expected to do better. That would be a finding worth")
        print("  reporting, not a problem to hide.")

    # ---- does the automatic judge match the humans? ------------------------
    banner("Automatic judge vs humans")
    for name, labels in sets.items():
        shared = [k for k, v in labels.items() if v in ("correct", "wrong") and k in machine]
        if not shared:
            continue
        result = cohens_kappa([machine[k]["label"] == "correct" for k in shared],
                              [labels[k] == "correct" for k in shared])
        print(f"  judge vs {name}: n={result.n}, raw {100 * result.raw:.1f}%, "
              f"kappa {result.kappa:.3f} ({result.reading()})")
        print(f"    judge said correct, human said wrong: {result.a_only}")
        print(f"    human said correct, judge said wrong: {result.b_only}")
        if result.a_only > 2 * max(1, result.b_only):
            print("    -> the judge is more lenient than you are; absolute rates are inflated")

    # ---- and does our conclusion survive human labels? ---------------------
    banner("Condition rates under human labels")
    pooled: dict[str, list[bool]] = defaultdict(list)
    for labels in sets.values():
        for key, value in labels.items():
            if value in ("correct", "wrong"):
                pooled[key.split(":")[0]].append(value == "correct")

    # Both columns must cover the SAME items, or we are comparing a 15-item
    # human sample against a 100-item machine average and calling it agreement.
    labelled_keys = {k for labels in sets.values() for k, v in labels.items()
                     if v in ("correct", "wrong")}

    print(f"  {'Condition':<34} {'human':<28} {'judge, same items':<28} {'gap'}")
    print(f"  {'-' * 34} {'-' * 28} {'-' * 28} {'-' * 6}")
    for condition in CONDITIONS:
        human = pooled.get(condition.key, [])
        auto = [machine[k]["label"] == "correct" for k in labelled_keys
                if k.split(":")[0] == condition.key and k in machine]
        if not human:
            continue
        gap = 100 * (sum(auto) / len(auto) - sum(human) / len(human)) if auto else 0.0
        print(f"  {condition.label:<34} {str(Proportion(sum(human), len(human))):<28} "
              f"{str(Proportion(sum(auto), len(auto))):<28} {gap:+5.1f}")

    print("\n  The 'gap' column is how much the judge inflates each condition.")
    print("  If it is large where there is no evidence to use and small where there")
    print("  is, the judge is rewarding topical plausibility, not memory -- and our")
    print("  effect sizes survive even though the absolute numbers do not.")
    print("\n  The ordering is what matters. If the human column ranks the conditions")
    print("  the same way, the pilot's conclusion holds and only the absolute")
    print("  numbers were inflated -- which we then report honestly.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=RESULTS_DIR / "pilot_n100_seed42.json")
    parser.add_argument("--annotator", type=str, help="your name, e.g. annotator2")
    parser.add_argument("--n", type=int, default=60, help="items total, split evenly")
    parser.add_argument("--seed", type=int, default=7,
                        help="same seed for both annotators, or you label different items")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if args.report:
        return report(args)
    if not args.annotator:
        parser.error("give --annotator <name>, or --report")
    return annotate(args)


if __name__ == "__main__":
    sys.exit(main())
