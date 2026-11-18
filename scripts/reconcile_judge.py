"""
What would the humans have scored? Reconciling our numbers with LoCoMo-Plus.

    python scripts/reconcile_judge.py [--export]

Free, local, offline: reads results/annotations/*.json and the judge labels
already stored in the result files. No API calls, no dataset, no embedder.

WHY. LoCoMo-Plus (Li et al., 2026, Table 1) reports every system on the
Cognitive split at 10-26% (best: gemini-2.5-pro, full context, 26.06). Our
judge puts full context at 45.9% and no memory at 37.0% -- above every system
they report. Their judge (gemini-2.5-flash) agrees with humans at kappa
0.80-0.82; ours at 0.56-0.60, and is more lenient on weak conditions.

METHOD (calibration, not a new judge). Every annotation batch was drawn at
random WITHIN a condition, independently of the judge's label
(build_annotation_batch.py). So for each condition and each annotator,

    ppv = P(human correct | judge correct)
    fo  = P(human correct | judge wrong)

are unbiased estimates, and the human-equivalent rate on any set judged by
the same judge on the same condition is

    human_hat = judge_rate * ppv + (1 - judge_rate) * fo.

This corrects for the annotated items not being the whole set. It assumes
the judge errs the same way on the 401 as on the n=100 subset the annotated
items came from (same judge, same prompts, same generator).

Intervals: bootstrap over annotated items (within condition) and over the
target items, fixed seed, so the number is reproducible.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = Path("results")
ANNOTATIONS = RESULTS / "annotations"
JOINT = "annotator2"   # the joint session: one agreed label, not one annotator
SCORED = ("correct", "wrong")

# Where the annotated predictions came from (their judge labels).
CALIBRATION_FILES = (RESULTS / "pilot_n100_seed42.json",
                     RESULTS / "system_n100_seed42_llm3_d1_notrig_events.json")
# Where to apply the calibration: the headline files where they exist.
TARGETS = {
    "no_memory":         (RESULTS / "pilot_n100_seed42.json", "no_memory"),
    "full_context":      (RESULTS / "pilot_n401_seed42_full_context.json", "full_context"),
    "system_llm":        (RESULTS / "system_n401_seed42_llm3_d1_notrig_events.json", "system_llm"),
    "oracle_cue":        (RESULTS / "pilot_n100_seed42.json", "oracle_cue"),
    "oracle_constraint": (RESULTS / "pilot_n100_seed42.json", "oracle_constraint"),
}
BOOT = 2000
SEED = 7


def judge_labels(paths) -> dict:
    out = {}
    for path in paths:
        for r in json.loads(Path(path).read_text(encoding="utf-8")):
            if r.get("label") in SCORED:
                out[f"{r['condition']}:{r['sample_index']}"] = r["label"] == "correct"
    return out


def target_labels(path, condition) -> list:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [r["label"] == "correct" for r in rows
            if r.get("condition") == condition and r.get("label") in SCORED]


def annotators(ann_dir=ANNOTATIONS) -> dict:
    people = {}
    for path in sorted(Path(ann_dir).glob("*.json")):
        if path.stem == JOINT:
            continue
        who = path.stem.split("_")[0]
        table = json.loads(path.read_text(encoding="utf-8"))
        people.setdefault(who, {}).update(
            {k: v == "correct" for k, v in table.items() if v in SCORED})
    return people


def calibrate(pairs, judge_rate):
    """pairs: list of (judge_correct, human_correct). Returns (ppv, fo, human_hat)."""
    jc = [h for j, h in pairs if j]
    jw = [h for j, h in pairs if not j]
    if not jc or not jw:
        raise ValueError("need annotated items with both judge verdicts")
    ppv = sum(jc) / len(jc)
    fo = sum(jw) / len(jw)
    return ppv, fo, judge_rate * ppv + (1 - judge_rate) * fo


def bootstrap(pairs, target, rng, reps=BOOT):
    vals = []
    n_p, n_t = len(pairs), len(target)
    for _ in range(reps):
        p = [pairs[rng.randrange(n_p)] for _ in range(n_p)]
        t = [target[rng.randrange(n_t)] for _ in range(n_t)]
        try:
            vals.append(calibrate(p, sum(t) / n_t)[2])
        except ValueError:
            continue
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def main() -> int:
    export = "--export" in sys.argv
    print("Files actually read:")
    for p in sorted(ANNOTATIONS.glob("*.json")):
        print(f"  {p}")
    for p in sorted({str(x) for x in CALIBRATION_FILES} |
                    {str(v[0]) for v in TARGETS.values()}):
        print(f"  {p}")

    judge = judge_labels(CALIBRATION_FILES)
    people = annotators()
    rng = random.Random(SEED)
    out = {"method": "calibrated: judge_rate*ppv + (1-judge_rate)*fo",
           "conditions": {}}

    print(f"\n{'condition':<18} {'set':>5} {'judge':>7} | "
          + " | ".join(f"{w:^30}" for w in sorted(people)) + " | mean")
    for cond, (path, key) in TARGETS.items():
        target = target_labels(path, key)
        jr = sum(target) / len(target)
        row = {"target_file": str(path), "n": len(target), "judge": jr,
               "annotators": {}}
        cells, hats = [], []
        for who in sorted(people):
            table = people[who]
            pairs = [(judge[k], table[k]) for k in table
                     if k.startswith(cond + ":") and k in judge]
            ppv, fo, hat = calibrate(pairs, jr)
            lo, hi = bootstrap(pairs, target, rng)
            row["annotators"][who] = {"n_annotated": len(pairs), "ppv": ppv,
                                      "fo": fo, "human_hat": hat,
                                      "ci": [lo, hi]}
            hats.append(hat)
            cells.append(f"{100*hat:5.1f} [{100*lo:4.1f},{100*hi:4.1f}] n={len(pairs):<3}")
        row["human_hat_mean"] = sum(hats) / len(hats)
        out["conditions"][cond] = row
        print(f"{cond:<18} {len(target):>5} {100*jr:6.1f}% | " + " | ".join(cells)
              + f" | {100*row['human_hat_mean']:5.1f}")

    print("\nppv = P(human says correct | judge says correct); "
          "fo = P(human says correct | judge says wrong)")
    for cond, row in out["conditions"].items():
        print(f"  {cond:<18} " + "   ".join(
            f"{w}: ppv {a['ppv']:.2f} fo {a['fo']:.2f}"
            for w, a in row["annotators"].items()))

    b = out["conditions"]["full_context"]["human_hat_mean"]
    e = out["conditions"]["system_llm"]["human_hat_mean"]
    print(f"\nHuman-equivalent, all 401: full context {100*b:.1f}%, "
          f"ours {100*e:.1f}%, gap {100*(e-b):+.1f} points.")
    print("LoCoMo-Plus Table 1 (their judge, their generators): "
          "best full context 26.06 (gemini-2.5-pro).")

    if export:
        dest = RESULTS / "reconcile_judge.json"
        dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
