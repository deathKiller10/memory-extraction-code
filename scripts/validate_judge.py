"""
Is our judge trustworthy? Nothing in the pilot means anything until we know.

    python scripts/validate_judge.py --per-condition 25

THE PROBLEM THIS EXISTS TO INVESTIGATE
--------------------------------------
The pilot's no-memory condition scored 37%. That condition gives the model
nothing but the trigger utterance -- no conversation, no cue, no context of any
kind. It cannot possibly be reflecting the evidence. A judge that marks it
"correct" 37% of the time is rewarding responses that merely sound topically
plausible.

For comparison, LoCoMo-Plus reports gpt-4o at 21.05 and gemini-2.5-pro at 26.06
with the FULL conversation available. Our no-memory floor sits above their best
full-context system. Our absolute numbers are therefore not comparable to
theirs, and quoting them side by side would be misleading.

What this script does:

  1. Re-judges a stratified subset with a second, stronger judge from a
     different tier (Gemini flash rather than Groq's 27B Qwen).
  2. Reports Cohen's kappa, not just raw agreement -- when 85% of labels are
     "correct", two judges agree ~75% of the time by luck alone.
  3. Recomputes every condition's rate under the second judge, so we can see
     whether the ORDERING survives even if the absolute numbers move.
  4. Prints the actual no-memory responses that were marked correct, so we can
     read them and judge the judge ourselves.

Point 3 is the one that decides the project. If the ordering holds under a
stricter judge, the pilot's conclusion stands and only the absolute numbers
were inflated. If the ordering collapses, we have no result yet.
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
from bapca.evaluation import CONDITIONS, judge_response
from bapca.llm import judge_secondary
from bapca.stats import Proportion, cohens_kappa, mcnemar

RESULTS_DIR = Path("results")


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/run_pilot.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def stratify(results: list[dict], per_condition: int, seed: int) -> list[dict]:
    """Equal numbers per condition, so agreement is not dominated by one of them."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        if row["label"] in ("correct", "wrong"):
            grouped[row["condition"]].append(row)

    rng = random.Random(seed)
    picked: list[dict] = []
    for condition in CONDITIONS:
        rows = grouped.get(condition.key, [])
        picked.extend(rows if len(rows) <= per_condition
                      else rng.sample(rows, per_condition))
    return picked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=RESULTS_DIR / "pilot_n100_seed42.json")
    parser.add_argument("--per-condition", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results = load_results(args.results)
    subset = stratify(results, args.per_condition, args.seed)
    print(f"Loaded {len(results)} judged responses; re-judging {len(subset)}.")

    if args.dry_run:
        print(f"\nWould cost {len(subset)} calls to the secondary judge "
              f"(Gemini flash, RPD 250). Spends nothing now.")
        return 0

    data = LocomoPlus()
    by_index = {s.index: s for s in data}
    jdg2 = judge_secondary(verbose=True)
    print(f"Secondary judge: {jdg2.model} ({jdg2.provider.name})\n")

    rows = []
    for i, row in enumerate(subset, 1):
        sample = by_index[row["sample_index"]]
        # Gemini has a 250k window and may think before answering; the 256
        # tokens that suit the Groq judge got this one truncated mid-JSON.
        verdict = judge_response(jdg2, sample, row["prediction"], max_tokens=1024)
        rows.append({**row, "label2": verdict.label, "reason2": verdict.reason})
        print(f"  [{i:>3}/{len(subset)}] {row['condition']:<18} "
              f"primary={row['label']:<8} secondary={verdict.label}", flush=True)

    out = args.results.with_name(args.results.stem + "_judge2.json")
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    unparsed = [r for r in rows if r["label2"] not in ("correct", "wrong")]
    if unparsed:
        print(f"\n  WARNING: {len(unparsed)} of {len(rows)} secondary verdicts were "
              f"unparseable and are excluded.")
        print(f"  If that fraction is large the agreement figure is not trustworthy.")
    usable = [r for r in rows if r["label2"] in ("correct", "wrong")]
    primary = [r["label"] == "correct" for r in usable]
    secondary = [r["label2"] == "correct" for r in usable]

    banner("Do the two judges agree?")
    overall = cohens_kappa(primary, secondary)
    print(f"  n={overall.n}   raw agreement {100 * overall.raw:.1f}%   "
          f"kappa {overall.kappa:.3f} ({overall.reading()})")
    print(f"  primary-only correct: {overall.a_only}   "
          f"secondary-only correct: {overall.b_only}")
    if overall.a_only > 2 * max(1, overall.b_only):
        print("\n  The primary judge is markedly more lenient. Absolute rates from")
        print("  the pilot are inflated and must not be compared to published numbers.")

    banner("Does the ordering survive the stricter judge?")
    print(f"  {'Condition':<34} {'primary':<26} {'secondary'}")
    print(f"  {'-' * 34} {'-' * 26} {'-' * 26}")
    outcomes = {}
    for condition in CONDITIONS:
        group = [r for r in usable if r["condition"] == condition.key]
        if not group:
            continue
        p = [r["label"] == "correct" for r in group]
        s = [r["label2"] == "correct" for r in group]
        outcomes[condition.key] = s
        print(f"  {condition.label:<34} {str(Proportion(sum(p), len(p))):<26} "
              f"{Proportion(sum(s), len(s))}")

    print()
    for a, b in [("no_memory", "full_context"), ("full_context", "oracle_cue"),
                 ("oracle_cue", "oracle_constraint")]:
        if a in outcomes and b in outcomes and len(outcomes[a]) == len(outcomes[b]):
            print(f"  {a} vs {b} (secondary judge): {mcnemar(outcomes[a], outcomes[b]).verdict()}")
    print("\n  Note: these are UNPAIRED across conditions (different items were")
    print("  sampled per condition), so read them as a direction check, not a test.")

    banner("The no-memory responses our primary judge called correct")
    print("  These had no conversation, no cue, nothing but the trigger line.")
    print("  If they read as generic sympathy rather than evidence-aware, the")
    print("  rubric is being applied too loosely.\n")
    shown = 0
    for row in rows:
        if row["condition"] != "no_memory" or row["label"] != "correct":
            continue
        sample = by_index[row["sample_index"]]
        agrees = row["label2"] == "correct"
        print(f"  --- item #{row['sample_index']}  (secondary judge: {row['label2']})")
        print(f"      evidence : {sample.evidence[:150]}")
        print(f"      trigger  : {sample.trigger[:150]}")
        print(f"      response : {row['prediction'][:260]}")
        print(f"      primary reason: {row['reason'][:170]}\n")
        shown += 1
        if shown >= 6:
            break
    if not shown:
        print("  (none in this subset)")

    print(f"\n  Secondary judge stats: {jdg2.stats()}")
    print(f"  Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
