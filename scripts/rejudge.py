"""
Second judge (19 Nov): re-judge the answers the paper's headline rests on with
a judge from another family, and check that the conclusions survive.

    python scripts/rejudge.py --plan      # rows done; free
    python scripts/rejudge.py             # run until quota stops it (resumes)
    python scripts/rejudge.py --report    # agreement + paired tests; free

WHY. Every score in the paper comes from one judge (Qwen on Groq), which
agrees with the two human annotators at kappa 0.56-0.60. The 29 Aug
validate_judge.py run re-judged a stratified 100-row sample of the pilot only.
This re-judges EVERY answer of the three conditions the headline compares on
the grid's 100 items: no memory and full context (pilot_n100_seed42.json) and
our system (events x llm, system_n100_seed42_llm3_d1_notrig_events.json).
The answers are the paper's; only the judge changes.

Judge: llm.judge_secondary (GPT-OSS on Groq -- a different family from Qwen,
and its own 200k-tokens-a-day quota, so it can run alongside
run_second_model.py). Output: results/judge2_n100.json, one row per answer,
saved after every row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.stats import Proportion, cohens_kappa, mcnemar

RESULTS = Path("results")
OUT = RESULTS / "judge2_n100.json"
SOURCES = [
    ("no_memory", RESULTS / "pilot_n100_seed42.json", "no_memory"),
    ("full_context", RESULTS / "pilot_n100_seed42.json", "full_context"),
    ("ours", RESULTS / "system_n100_seed42_llm3_d1_notrig_events.json", None),
]
SCORED = ("correct", "wrong")


def todo_rows() -> list[dict]:
    """Every answer to re-judge, in a fixed order, from the paper's files."""
    rows = []
    for name, path, condition in SOURCES:
        data = json.loads(path.read_text(encoding="utf-8"))
        for r in data:
            if condition is not None and r.get("condition") != condition:
                continue
            if r.get("label") not in SCORED:
                continue          # unparsed / no_constraint: not part of any score
            rows.append({"condition": name, "file": path.name,
                         "sample_index": r["sample_index"], "prediction": r["prediction"],
                         "label": r["label"]})
    return rows


def load_done() -> list[dict]:
    if not OUT.exists():
        return []
    return json.loads(OUT.read_text(encoding="utf-8"))


def report(rows: list[dict]) -> dict:
    """Agreement per condition and the headline's paired tests under each judge."""
    use = [r for r in rows if r.get("label2") in SCORED]
    out = {"n_rows": len(rows), "n_scored": len(use), "conditions": {}, "paired": {}}
    print(f"\n  rows judged twice: {len(use)} (of {len(rows)}; "
          f"{len(rows) - len(use)} second verdicts unparsed and excluded)")
    if use:
        out["judge2"] = sorted({r.get("judge2", "?") for r in use})
        k = cohens_kappa([r["label"] == "correct" for r in use],
                         [r["label2"] == "correct" for r in use])
        out["kappa_all"] = round(k.kappa, 3)
        print(f"  agreement, all rows: raw {100 * k.raw:.1f}%  kappa {k.kappa:.3f} "
              f"({k.reading()})")
    by = {}
    for name, _, _ in SOURCES:
        group = [r for r in use if r["condition"] == name]
        if not group:
            continue
        by[name] = {r["sample_index"]: r for r in group}
        p1 = Proportion(sum(r["label"] == "correct" for r in group), len(group))
        p2 = Proportion(sum(r["label2"] == "correct" for r in group), len(group))
        k = cohens_kappa([r["label"] == "correct" for r in group],
                         [r["label2"] == "correct" for r in group])
        out["conditions"][name] = {"n": len(group), "judge1": round(100 * p1.rate, 1),
                                   "judge2": round(100 * p2.rate, 1),
                                   "kappa": round(k.kappa, 3)}
        print(f"  {name:<13} judge 1 {str(p1):<24} judge 2 {str(p2):<24} kappa {k.kappa:.3f}")
    for a, b in (("ours", "no_memory"), ("ours", "full_context")):
        if a not in by or b not in by:
            continue
        common = sorted(set(by[a]) & set(by[b]))
        for key, label in (("judge1", "label"), ("judge2", "label2")):
            t = mcnemar([by[a][i][label] == "correct" for i in common],
                        [by[b][i][label] == "correct" for i in common])
            out["paired"][f"{a}_vs_{b}_{key}"] = {
                "n": len(common), "only_a": t.only_a, "only_b": t.only_b,
                "p": t.p_value}
            print(f"  {a} vs {b:<13} ({key}, n={len(common)}): "
                  f"{t.only_a} vs {t.only_b} discordant, p={t.p_value:.4g}")
    return out


def main() -> int:
    rows = load_done()
    done = {(r["condition"], r["sample_index"]) for r in rows}
    todo = [r for r in todo_rows() if (r["condition"], r["sample_index"]) not in done]
    print(f"second judge: {len(rows)} rows done, {len(todo)} to go  -> {OUT.name}")
    if "--plan" in sys.argv:
        return 0
    if "--report" in sys.argv:
        result = report(rows)
        if "--export" in sys.argv:
            (RESULTS / "judge2_report.json").write_text(json.dumps(result, indent=1),
                                                         encoding="utf-8")
            print(f"\n  exported results/judge2_report.json")
        return 0

    from bapca.dataset import LocomoPlus
    from bapca.evaluation import judge_response
    from bapca.llm import DailyQuotaExhausted, TransientProviderError, judge_secondary
    by_index = {s.index: s for s in LocomoPlus()}
    jdg2 = judge_secondary(verbose=True)
    print(f"  second judge: {jdg2.model} ({jdg2.provider.name})\n")
    stalled = 0
    for i, r in enumerate(todo, 1):
        try:
            v = judge_response(jdg2, by_index[r["sample_index"]], r["prediction"],
                               max_tokens=1024)
        except DailyQuotaExhausted as exc:
            print(f"\n  Out of quota for today ({exc}).")
            print(f"  {len(rows)} rows saved in {OUT.name}. Run the same command "
                  "tomorrow; it resumes.")
            return 0
        except TransientProviderError as exc:
            stalled += 1
            print(f"  [{i}/{len(todo)}] provider hiccup ({stalled} in a row): {exc}")
            if stalled >= 3:
                print("  Provider is down -- stopping. Re-run later; it resumes.")
                return 0
            continue
        stalled = 0
        rows.append({**{k: r[k] for k in ("condition", "file", "sample_index", "label")},
                     "label2": v.label, "reason2": v.reason, "judge2": jdg2.model})
        OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"  [{i:>3}/{len(todo)}] {r['condition']:<13} #{r['sample_index']:<5} "
              f"judge1={r['label']:<8} judge2={v.label}", flush=True)
    print("\nAll rows judged twice.")
    report(rows)
    print("\nNext: python scripts/rejudge.py --report --export")
    return 0


if __name__ == "__main__":
    sys.exit(main())
