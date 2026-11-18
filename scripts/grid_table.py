"""
The write-policy x selection grid, as one table.

    python scripts/grid_table.py [--n 100] [--export]

Free, local, offline: reads results/system_n{N}_seed42_*_d1_notrig*.json.
Missing cells are printed as missing, never filled in.

ROWS are write policies, COLUMNS selectors, exactly as Yuan, Su & Yao
(arXiv 2603.02473, Table 2) lay out their LoCoMo grid, so the two can be
read side by side. Their question: which axis moves accuracy more? Their
answer on LoCoMo: the retriever (~20 points) over the write strategy (3-8).

Two extra columns that their design cannot have:
  oracle_window  answer from every note written from the cue's window --
                 the write stage priced with selection removed (a ceiling);
  llm            our own re-ranker over the whole store (the headline's).

The axis comparison uses only the three selectors both papers share
(similar = their Cosine, bm25, pool = their Hybrid+Rerank).
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.stats import Proportion, mcnemar

RESULTS = Path("results")
POLICIES = ("v1", "events", "facts", "summary", "raw")
SHARED = ("similar", "bm25", "pool")            # the selectors Yuan et al. also ran
EXTRA = ("llm", "oracle_window")
SELECTORS = SHARED + EXTRA
LABEL = {"similar": "cosine", "bm25": "BM25", "pool": "hybrid", "llm": "ours(llm)",
         "oracle_window": "ORACLE-win"}


def cell_path(n: int, policy: str, selector: str, top_k: int = 3) -> Path:
    suffix = "" if policy == "v1" else f"_{policy}"
    return RESULTS / f"system_n{n}_seed42_{selector}{top_k}_d1_notrig{suffix}.json"


def load(path: Path):
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["sample_index"]: r for r in rows if r.get("label") in ("correct", "wrong")}


def models(rows) -> set:
    return {(r.get("generator", "?"), r.get("judge", "?")) for r in rows.values()}


def main() -> int:
    n = 100
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    export = "--export" in sys.argv

    cells, files = {}, []
    for pol in POLICIES:
        for sel in SELECTORS:
            path = cell_path(n, pol, sel)
            rows = load(path)
            if rows is not None:
                cells[(pol, sel)] = rows
                files.append(path)
    print("Files actually read:")
    for f in files:
        print(f"  {f}")
    if not cells:
        print("  none -- nothing to tabulate yet.")
        return 0

    # Every cell must come from the same generator and judge, or the grid
    # compares models, not stages. Rows from before 24 Oct record no model
    # ('?'); those are allowed but flagged.
    seen = set().union(*(models(r) for r in cells.values()))
    known = {m for m in seen if "?" not in m}
    if len(known) > 1:
        print(f"\n  REFUSING: cells were produced by different models: {sorted(known)}")
        print("  Re-run the odd cells with the same models before comparing.")
        return 1
    if any("?" in m for m in seen):
        print("\n  note: some files predate per-row model recording ('?'); check HANDOFF"
              " that they used the same models.")

    complete = sorted(set.intersection(*(set(r) for r in cells.values())))
    print(f"\n  items scored in every present cell: {len(complete)}")

    corner = "write \\ select"
    print(f"\n  {corner:<10}" + "".join(f"{LABEL[s]:>13}" for s in SELECTORS))
    table = {}
    for pol in POLICIES:
        line = f"  {pol:<10}"
        for sel in SELECTORS:
            rows = cells.get((pol, sel))
            if rows is None:
                line += f"{'--':>13}"
                continue
            hits = [rows[i]["label"] == "correct" for i in complete]
            prop = Proportion(sum(hits), len(hits))
            table[(pol, sel)] = prop
            line += f"{100 * prop.rate:12.1f}%"
        print(line)

    # ---- which axis matters? Yuan et al.'s comparison, on shared selectors
    print("\n  Range of accuracy (max - min), as Yuan et al. report it:")
    write_ranges, select_ranges = {}, {}
    for sel in SHARED:
        vals = [table[(p, sel)].rate for p in POLICIES if (p, sel) in table]
        if len(vals) >= 2:
            write_ranges[sel] = max(vals) - min(vals)
            print(f"    across WRITE policies, selector {LABEL[sel]:<8}: "
                  f"{100 * write_ranges[sel]:5.1f} points ({len(vals)} policies)")
    for pol in POLICIES:
        vals = [table[(pol, s)].rate for s in SHARED if (pol, s) in table]
        if len(vals) >= 2:
            select_ranges[pol] = max(vals) - min(vals)
            print(f"    across SELECTORS, write policy {pol:<8}: "
                  f"{100 * select_ranges[pol]:5.1f} points ({len(vals)} selectors)")
    if write_ranges and select_ranges:
        mw = sum(write_ranges.values()) / len(write_ranges)
        ms = sum(select_ranges.values()) / len(select_ranges)
        print(f"\n    mean range across write policies: {100 * mw:5.1f} points")
        print(f"    mean range across selectors:      {100 * ms:5.1f} points")
        print("    (LoCoMo, Yuan et al.: selectors ~20, write policies 3-8)")

    # ---- the write ceiling, and how much each selector leaves on the table
    print("\n  Write ceiling (oracle window) and the selection gap:")
    for pol in POLICIES:
        ceil = table.get((pol, "oracle_window"))
        if ceil is None:
            continue
        rest = [(s, table[(pol, s)]) for s in SHARED + ("llm",) if (pol, s) in table]
        gaps = ", ".join(f"{LABEL[s]} {100 * (ceil.rate - p.rate):+.1f}" for s, p in rest)
        print(f"    {pol:<8} ceiling {100 * ceil.rate:5.1f}%   gap: {gaps or '--'}")

    # ---- paired tests: best vs worst within each axis, named by cell
    print("\n  Paired exact McNemar (same items), best vs worst cell on each line:")
    tests = {}
    def named(a, b):
        ra, rb = cells[a], cells[b]
        t = mcnemar([ra[i]["label"] == "correct" for i in complete],
                    [rb[i]["label"] == "correct" for i in complete])
        win = a if t.only_a > t.only_b else (b if t.only_b > t.only_a else None)
        return t, win
    for sel in SHARED:
        present = [(p, sel) for p in POLICIES if (p, sel) in table]
        if len(present) >= 2:
            hi = max(present, key=lambda c: table[c].rate)
            lo = min((c for c in present if c != hi), key=lambda c: table[c].rate)
            t, win = named(hi, lo)
            tests[f"write@{sel}"] = dict(best=hi, worst=lo, p=t.p_value,
                                         only_best=t.only_a, only_worst=t.only_b)
            print(f"    selector {LABEL[sel]:<7}: {hi[0]} vs {lo[0]}  p={t.p_value:.4f} "
                  f"({t.only_a} vs {t.only_b} discordant; more correct: {win[0] if win else 'neither'})")
    for pol in POLICIES:
        present = [(pol, s) for s in SHARED if (pol, s) in table]
        if len(present) >= 2:
            hi = max(present, key=lambda c: table[c].rate)
            # A tie must not pair a cell with itself (27 Oct: "cosine vs cosine").
            lo = min((c for c in present if c != hi), key=lambda c: table[c].rate)
            t, win = named(hi, lo)
            tests[f"select@{pol}"] = dict(best=hi, worst=lo, p=t.p_value,
                                          only_best=t.only_a, only_worst=t.only_b)
            print(f"    policy {pol:<8}: {LABEL[hi[1]]} vs {LABEL[lo[1]]}  p={t.p_value:.4f} "
                  f"({t.only_a} vs {t.only_b} discordant; more correct: {LABEL[win[1]] if win else 'neither'})")

    if export:
        out = dict(n_items=len(complete), files=[str(f) for f in files],
                   models=sorted(known) or ["not recorded"],
                   cells={f"{p}|{s}": dict(correct=v.successes, total=v.total)
                          for (p, s), v in table.items()},
                   write_ranges=write_ranges, select_ranges=select_ranges,
                   tests={k: {**v, "best": "|".join(v["best"]),
                              "worst": "|".join(v["worst"])} for k, v in tests.items()})
        dest = RESULTS / "grid.json"
        dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\n  wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
