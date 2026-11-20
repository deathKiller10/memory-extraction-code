"""
Do the paper's claims survive a second answering model? (19 Nov)

    python scripts/second_model_table.py            # works with any finished steps
    python scripts/second_model_table.py --export   # -> results/second_model.json

Reads the paper's files (Gemini flash-lite answered) and the run_second_model.py
files (Cohere Command A answered; same notes, selections and judge), pairs items by
sample_index, and tests each claim on the second model's answers with the paper's
test (exact two-sided McNemar):

  C1  no prior-work cell beats no memory          (paper: 0 of 9 significantly better)
  C2  every prior-work cell is below full context (paper: 9 of 9 significantly worse;
                                                   here 3 cells, the hybrid column)
  C3  our system beats no memory and full context
  C4  for our notes the selector matters           (ours vs prior work's hybrid)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bapca.stats import mcnemar

RESULTS = Path("results")
SCORED = ("correct", "wrong")
PAPER_PILOT = "pilot_n100_seed42.json"
YUAN = ("raw", "facts", "summary")


def _labels(path: Path, condition: str | None = None) -> dict:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["sample_index"]: r["label"] == "correct" for r in rows
            if r.get("label") in SCORED
            and (condition is None or r.get("condition") == condition)}


def _models(path: Path) -> set:
    if not path.exists():
        return set()
    return {r.get("generator", "?") for r in json.loads(path.read_text(encoding="utf-8"))}


def load() -> dict:
    import run_second_model as rsm
    pilot_m = rsm.target(("pilot", None))
    cells = {"no_memory": (_labels(RESULTS / PAPER_PILOT, "no_memory"),
                           _labels(pilot_m, "no_memory")),
             "full_context": (_labels(RESULTS / PAPER_PILOT, "full_context"),
                              _labels(pilot_m, "full_context"))}
    models = {"second": _models(pilot_m)}
    for policy, select in rsm.QUEUE[1:]:
        paper = RESULTS / f"system_n100_seed42_{select}3_d1_notrig_{policy}.json"
        mine = rsm.target((policy, select))
        cells[f"{policy}_{select}"] = (_labels(paper), _labels(mine))
        models["second"] |= _models(mine)
    return {"cells": cells, "models": models}


def rate(d: dict) -> float | None:
    return round(100 * sum(d.values()) / len(d), 1) if d else None


def paired(a: dict, b: dict) -> dict | None:
    common = sorted(set(a) & set(b))
    if not common:
        return None
    t = mcnemar([a[i] for i in common], [b[i] for i in common])
    return {"n": len(common), "only_a": t.only_a, "only_b": t.only_b, "p": t.p_value}


def main() -> int:
    data = load()
    cells = data["cells"]
    print(f"answering models seen in the second-model files: {sorted(data['models']['second'] - {'?'}) or 'none yet'}")
    if len(data["models"]["second"] - {"?"}) > 1:
        print("STOP: the second-model files mix answering models; do not use this table.")
        return 1
    print(f"\n  {'cell':<18} {'Gemini (paper)':>15} {'Cohere':>9} {'n':>4}")
    out = {"rates": {}, "tests": {}, "models": sorted(data["models"]["second"])}
    for name, (g, m) in cells.items():
        out["rates"][name] = {"gemini": rate(g), "second": rate(m), "n_second": len(m)}
        print(f"  {name:<18} {str(rate(g)):>15} {str(rate(m)):>9} {len(m):>4}")

    for gen, idx in (("gemini", 0), ("second", 1)):
        A, B = cells["no_memory"][idx], cells["full_context"][idx]
        ours = cells["events_llm"][idx]
        tests = {}
        for pol in YUAN:
            c = cells[f"{pol}_pool"][idx]
            tests[f"{pol}_pool_vs_no_memory"] = paired(c, A)
            tests[f"{pol}_pool_vs_full_context"] = paired(c, B)
        tests["ours_vs_no_memory"] = paired(ours, A)
        tests["ours_vs_full_context"] = paired(ours, B)
        tests["ours_vs_events_pool"] = paired(ours, cells["events_pool"][idx])
        out["tests"][gen] = tests

        def sig_better(t):
            return t is not None and t["p"] < 0.05 and t["only_a"] > t["only_b"]

        def sig_worse(t):
            return t is not None and t["p"] < 0.05 and t["only_b"] > t["only_a"]

        have = [p for p in YUAN if tests[f"{p}_pool_vs_no_memory"]]
        print(f"\n  --- claims, answered by {gen} ---")
        print(f"  C1 prior-work cells significantly better than no memory: "
              f"{sum(sig_better(tests[f'{p}_pool_vs_no_memory']) for p in have)} of {len(have)}")
        print(f"  C2 prior-work cells significantly worse than full context: "
              f"{sum(sig_worse(tests[f'{p}_pool_vs_full_context']) for p in have)} of {len(have)}")
        for key in ("ours_vs_no_memory", "ours_vs_full_context", "ours_vs_events_pool"):
            t = tests[key]
            print(f"  {'C3' if 'events' not in key else 'C4'} {key:<22} " +
                  (f"{t['only_a']} vs {t['only_b']} discordant, p={t['p']:.4g} (n={t['n']})"
                   if t else "--"))
    if "--export" in sys.argv:
        (RESULTS / "second_model.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("\n  exported results/second_model.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
