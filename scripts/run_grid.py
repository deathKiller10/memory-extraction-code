"""
Run the write-policy x selection grid, one cell after another, across days.

    python scripts/run_grid.py --plan      # the queue and what is done; free
    python scripts/run_grid.py             # run until quota stops it

Every cell is an ordinary run_system.py run (n=100, seed 42, top-k 3,
--strip-trigger, days-per-window 1), so it caches, resumes and names its file
exactly as every earlier run did. Run the SAME command every day: finished
cells are skipped, a half-finished one resumes, and the run stops at the first
cell that does not complete (quota or provider down). Never add --fresh.

ORDER (most informative per token first; see HANDOFF section 20.1):
  phase 1  v1, events, raw  -- no new extraction (v1/events cached, raw free)
  phase 2  summary          -- Yuan et al.'s MemGPT-style extraction, ~0.4M tokens
  phase 3  facts            -- Yuan et al.'s Mem0-style extraction, ~0.9M tokens
  (summary moved first on 27 Oct: measured cheaper, and facts had to restart
   at a 2,048-token cap -- HANDOFF 20.7)
  phase 4  BM25 column      -- optional, all five policies
Within a policy: the oracle-window ceiling first, then cosine, then hybrid.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results")
N, SEED, TOP_K = 100, 42, 3

PLAN = (
    [("events", "oracle_window"), ("v1", "oracle_window"),
     ("raw", "oracle_window"), ("raw", "similar"), ("raw", "pool"),
     ("events", "similar"), ("events", "pool"),
     ("v1", "similar"), ("v1", "pool")]
    + [("summary", s) for s in ("oracle_window", "similar", "pool")]
    # 3 Nov sensitivity check: the store's near-duplicate merge dropped the
    # cue window's summary on 11/100 items. Cheap -- summaries are cached and
    # only the affected items get new prompts. HANDOFF 20.8.
    + [("summary", "oracle_window", "nomerge")]
    + [("facts", s) for s in ("oracle_window", "similar", "pool")]
    # facts x llm was added 25 Oct and REMOVED 26 Oct: facts writes ~12 per
    # window (529 on the first item, not ~150), and our re-ranker lists every
    # note in ONE request -- ~8k tokens, over Groq's 8,000/minute ceiling for a
    # single request, so every item would be refused. HANDOFF 20.6.
    + [(p, "bm25") for p in ("events", "v1", "raw", "facts", "summary")]
    + [("summary", s, "nomerge") for s in ("similar", "pool")]
)


def cells():
    """PLAN as (policy, select, variant) -- variant is "" or "nomerge"."""
    return [(c[0], c[1], c[2] if len(c) > 2 else "") for c in PLAN]


def cell_file(policy: str, select: str, variant: str = "") -> Path:
    suffix = "" if policy == "v1" else f"_{policy}"
    if variant:
        suffix += f"_{variant}"
    return RESULTS / f"system_n{N}_seed{SEED}_{select}{TOP_K}_d1_notrig{suffix}.json"


def done_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return 0


def _row_ok(row: dict, policy: str, budgets: dict, protocol: int) -> bool:
    if row.get("write_budget") != budgets[policy]:
        return False
    if policy != "facts":
        return True
    if row.get("facts_protocol") == protocol:
        return True
    # A v2 row (6-7 Nov) that split nothing and kept nothing truncated used only
    # whole windows at 2048 with complete replies -- byte-identical under v3.
    return (row.get("facts_protocol") == 2 and not row.get("facts_split_windows")
            and not row.get("facts_truncated_kept"))


def _superseded(row: dict) -> bool:
    """v2 facts rows that DID split: the rule v3 replaced. Dropped and redone."""
    return (row.get("facts_protocol") == 2 and row.get("write_budget") == 2048
            and bool(row.get("facts_split_windows") or row.get("facts_truncated_kept")))


def _rules():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bapca.pipeline import FACTS_PROTOCOL, WRITE_BUDGETS
    return WRITE_BUDGETS, FACTS_PROTOCOL


def stale_rows(path: Path, policy: str) -> int:
    """Rows of a facts/summary cell made under an old output cap or an old facts
    rule (700-token cap, 26 Oct; facts rule 1, before 6 Nov). The run stops
    rather than mix them with current rows."""
    budgets, protocol = _rules()
    if policy not in budgets or not path.exists():
        return 0
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    return sum(not _row_ok(r, policy, budgets, protocol) for r in rows)


def drop_superseded(path: Path, policy: str) -> int:
    """Remove facts rule-v2 rows that split windows, after saving the whole file
    as results/OLD_superseded_<name>_<time>.json. Returns rows dropped."""
    if policy != "facts" or not path.exists():
        return 0
    rows = json.loads(path.read_text(encoding="utf-8"))
    drop = [r for r in rows if _superseded(r)]
    if not drop:
        return 0
    import time as _time
    backup = path.parent / f"OLD_superseded_{path.stem}_{_time.strftime('%Y%m%d-%H%M')}.json"
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    keep = [r for r in rows if not _superseded(r)]
    path.write_text(json.dumps(keep, indent=1), encoding="utf-8")
    print(f"  {path.name}: dropped {len(drop)} rule-v2 row(s) that split windows "
          f"({', '.join(str(r['sample_index']) for r in drop)}); they will be redone.")
    print(f"    backup of the file as it was: {backup.name}")
    return len(drop)


def command(policy: str, select: str, variant: str = "") -> list[str]:
    cmd = [sys.executable, "scripts/run_system.py", "--n", str(N), "--seed", str(SEED),
           "--select", select, "--top-k", str(TOP_K), "--strip-trigger",
           "--extract", policy]
    if variant == "nomerge":
        cmd.append("--no-merge")
    return cmd


CODE_VERSION = "8 Nov 2026"     # bump whenever the grid code changes (HANDOFF 20.14)


def main() -> int:
    plan_only = "--plan" in sys.argv
    _, protocol = _rules()
    print(f"grid code of {CODE_VERSION}: facts rule v{protocol}, cache reads batched\n")
    print(f"{'#':>2}  {'policy':<8} {'select':<14} {'rows':>5}  file")
    for i, (pol, sel, var) in enumerate(cells(), 1):
        path = cell_file(pol, sel, var)
        print(f"{i:>2}  {pol:<8} {sel:<14} {done_rows(path):>5}  {path.name}")
    if plan_only:
        return 0

    # Facts rows made under rule v2 that split windows are superseded by v3
    # (8 Nov): drop them (with a backup) so they are redone. Then check EVERY
    # cell before anything runs, so a stale file is caught today.
    for p, s, v in cells():
        drop_superseded(cell_file(p, s, v), p)
    stale = [(cell_file(p, s, v), stale_rows(cell_file(p, s, v), p)) for p, s, v in cells()]
    stale = [(path, n) for path, n in stale if n]
    if stale:
        for path, n in stale:
            print(f"\n!!! {path.name} has {n} row(s) written under an old output cap")
        print("    or an old facts extraction rule (before 6 Nov). Rename that file in")
        print("    Drive's results/ folder (e.g. put OLD_ in front), then run this again.")
        print("    Nothing was run.")
        return 1

    for pol, sel, var in cells():
        path = cell_file(pol, sel, var)
        if done_rows(path) >= N:
            continue
        name = f"{pol} x {sel}" + (f" ({var})" if var else "")
        print(f"\n>>> cell {name}: {done_rows(path)}/{N} done, running\n", flush=True)
        subprocess.run(command(pol, sel, var), check=False)
        rows = done_rows(path)
        if rows < N:
            # A few items can be skipped for good (a prompt too large, a
            # provider error), so "not 100" is not always "out of quota". The
            # run_system output above says which; either way, stop here.
            print(f"\n<<< {name} stopped at {rows}/{N}. Read the lines above:")
            print("    'Out of quota' -> run this same command tomorrow; it resumes.")
            print("    anything else  -> stop and check before re-running.")
            return 0
    print("\nAll cells complete. Next: python scripts/grid_table.py --export")
    return 0


if __name__ == "__main__":
    sys.exit(main())
