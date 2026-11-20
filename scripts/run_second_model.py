"""
The second-answering-model check (19 Nov), one step after another, across days.

    python scripts/run_second_model.py --plan     # the queue and what is done; free
    python scripts/run_second_model.py --check    # is the extraction cache warm? free
    python scripts/run_second_model.py            # run until quota stops it

WHY. Every number in the paper is answered by one model (Gemini flash-lite)
and judged by one (Qwen on Groq). A reviewer will ask whether the findings
are about memory or about that answering model. This re-answers the cells
the claims rest on with a third vendor and family (Cohere Command A), changing NOTHING
else: the notes, the selections and the judge are the paper's, and the notes
and selections come from the cache.

THE CLAIMS AND THE STEPS THAT TEST THEM (n=100, seed 42, the grid's subset):
  pilot          no memory and full context  -- the two baselines
  events x llm   our system                  -- beats both baselines?
  raw/facts/summary x pool
                 prior work's write strategies under its best retriever
                 (hybrid re-rank)            -- still no better than no memory?
  events x pool  our notes, prior work's selector -- does the selector matter?

Each step is an ordinary run_pilot.py / run_system.py run with
--generator cohere, so it caches, resumes and names its own file
(..._gencohere.json; it can never touch a file behind the paper). Run the
SAME command every day: finished steps are skipped, a half-finished one
resumes, and the run stops at the first step that does not complete.
Never add --fresh.

COST. Cohere (answering): free trial key, 20 requests/minute and 1,000
calls a MONTH; this run makes 700 (cached answers cost nothing on a rerun).
Groq
(judging, plus re-ranking for pool/llm, all re-ranks cached): ~100 judge calls
per step, ~90k tokens, against 200k a day -- about two steps a day, so
three to four days. Facts windows refused for Groq's output cap are now
remembered (llm.py, 19 Nov), so a facts step no longer re-pays ~1,000
refused requests.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results")
N, SEED, TOP_K = 100, 42, 3
GEN = "cohere"
CODE_VERSION = "19 Nov 2026"          # bump whenever this check's code changes

QUEUE = [
    ("pilot", None),
    ("events", "llm"),
    ("raw", "pool"),
    ("facts", "pool"),
    ("summary", "pool"),
    ("events", "pool"),
]
PILOT_CONDITIONS = ("no_memory", "full_context")


def target(step) -> Path:
    policy, select = step
    if policy == "pilot":
        tag = "+".join(sorted(PILOT_CONDITIONS))
        return RESULTS / f"pilot_n{N}_seed{SEED}_{tag}_gen{GEN}.json"
    suffix = "" if policy == "v1" else f"_{policy}"
    return RESULTS / f"system_n{N}_seed{SEED}_{select}{TOP_K}_d1_notrig{suffix}_gen{GEN}.json"


def needed(step) -> int:
    return N * len(PILOT_CONDITIONS) if step[0] == "pilot" else N


def command(step) -> list[str]:
    policy, select = step
    if policy == "pilot":
        return [sys.executable, "scripts/run_pilot.py", "--n", str(N), "--seed", str(SEED),
                "--conditions", *PILOT_CONDITIONS, "--generator", GEN]
    return [sys.executable, "scripts/run_system.py", "--n", str(N), "--seed", str(SEED),
            "--select", select, "--top-k", str(TOP_K), "--strip-trigger",
            "--extract", policy, "--generator", GEN]


def done_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return 0


def check_cache(items: int = 3, jdg=None, samples=None) -> bool:
    """
    Is every note these steps need already in the extraction cache? Runs the
    real write step on the first few items with the network BLOCKED: a cached
    window returns at once, an uncached one would have to call Groq and is
    refused here instead. Costs nothing (one model-list call to resolve the
    judge). A cold cache would quietly spend days of Groq quota re-extracting.

    Facts windows that Groq refused for its output cap (rule v3) never had
    their first, 2048-token request cached -- only the 936-token retry. So an
    uncached request in "raise" mode is answered here as the refusal it was,
    and the check follows the retry, exactly as the run will. Nothing is
    written to the cache while checking: a refusal marker for a window that
    was simply never extracted would change how the real run extracts it.

    Over Drive each cache file takes a moment to read, and a facts item is ~40
    windows, so this takes a few minutes; it prints a line per item (20 Nov:
    with no output it looked frozen and was stopped).
    """
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bapca.llm import OutputCapExceeded
    from bapca.pipeline import segment, write_notes
    import run_system

    class WouldSpend(Exception):
        pass

    def blocked(messages, temperature, max_tokens, on_output_cap="shrink"):
        if on_output_cap == "raise":
            raise OutputCapExceeded(1000, max_tokens, max_tokens)
        raise WouldSpend()

    if jdg is None:
        from bapca.llm import judge
        jdg = judge(verbose=False)
    if samples is None:
        from bapca.dataset import LocomoPlus
        samples = LocomoPlus().subset(N, seed=SEED)[:items]
    print(f"  judge (writes the notes): {jdg.model}")
    real_call, real_put = jdg._call_with_backoff, jdg.cache.put
    jdg._call_with_backoff = blocked
    jdg.cache.put = lambda *_a, **_k: None
    ok = True
    try:
        for policy in sorted({p for p, _ in QUEUE if p not in ("pilot", "raw")}):
            warm = 0
            for k, s in enumerate(samples, 1):
                started = time.time()
                windows = segment(run_system._source(s, True), 12)
                try:
                    write_notes(jdg, windows, policy)
                    warm += 1
                    state = "cached"
                except WouldSpend:
                    state = "NOT cached"
                print(f"    {policy:<8} item {k}/{len(samples)}: {len(windows)} windows "
                      f"{state} ({time.time() - started:.0f} s)", flush=True)
            print(f"  {policy:<8} {warm}/{len(samples)} items fully cached", flush=True)
            ok &= warm == len(samples)
    finally:
        jdg._call_with_backoff, jdg.cache.put = real_call, real_put
    return ok


def main() -> int:
    print(f"second-model check, code of {CODE_VERSION}: answering model = {GEN}\n")
    print(f"{'#':>2}  {'step':<16} {'rows':>9}  file")
    for i, step in enumerate(QUEUE, 1):
        name = "pilot (A, B)" if step[0] == "pilot" else f"{step[0]} x {step[1]}"
        path = target(step)
        print(f"{i:>2}  {name:<16} {done_rows(path):>4}/{needed(step):<4}  {path.name}")
    if "--plan" in sys.argv:
        return 0
    if "--check" in sys.argv:
        print("\nExtraction cache (network blocked; a few minutes over Drive):", flush=True)
        ok = check_cache()
        print("\nOK -- the notes are cached; the run pays only for answering and judging."
              if ok else
              "\nSTOP -- some notes are NOT cached. Check that section 5 printed the Drive "
              "cache, then tell Claude. Do not start the run.")
        return 0 if ok else 1

    for step in QUEUE:
        path = target(step)
        if done_rows(path) >= needed(step):
            continue
        name = "pilot (no memory, full context)" if step[0] == "pilot" else f"{step[0]} x {step[1]}"
        print(f"\n>>> {name}: {done_rows(path)}/{needed(step)} done, running\n", flush=True)
        subprocess.run(command(step), check=False)
        rows = done_rows(path)
        if rows < needed(step):
            print(f"\n<<< {name} stopped at {rows}/{needed(step)}. Read the lines above:")
            print("    'Out of quota' -> run this same command tomorrow; it resumes.")
            print("    anything else  -> stop and check before re-running.")
            return 0
    print("\nAll steps complete. Next: python scripts/second_model_table.py --export")
    return 0


if __name__ == "__main__":
    sys.exit(main())
