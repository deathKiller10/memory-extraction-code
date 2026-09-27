"""
Run LoCoMo-Plus's own build pipeline and inspect what it produces.

Their `unified_input.py` stitches each cue/trigger pair into a real LoCoMo
conversation at the right temporal offset, and emits the prompts that models are
actually evaluated on. It takes no arguments, reads only local files, and needs
no API key -- so we can run it as-is and evaluate on exactly the input the
benchmark authors intended, rather than reconstructing it ourselves.

    python scripts/build_eval_set.py

Two things this measures that decide our design:

  * Prompt length. These are multi-session dialogues; if the full-context
    baseline runs to tens of thousands of tokens, "we match full context at a
    fraction of the tokens" is a claim with real headroom. If they are short,
    the efficiency argument is much weaker and we should know now.
  * How many Cognitive-category samples exist. That is our evaluation universe,
    and it sets what a "fixed subset" can honestly be.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import median

REPO_DIR = Path("third_party/Locomo-Plus")
DATA_DIR = REPO_DIR / "data"
OUTPUT = DATA_DIR / "unified_input_samples_v2.json"


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def ensure_save_flag() -> None:
    """
    Their script only writes the JSON when SAVE_JSON_FOR_INSPECTION is True.

    We flip it in place and keep a .orig backup rather than reimplementing the
    builder -- the whole point is to use their construction, not our copy of it.
    """
    script = DATA_DIR / "unified_input.py"
    source = script.read_text(encoding="utf-8")
    if re.search(r"SAVE_JSON_FOR_INSPECTION\s*=\s*True", source):
        print("  SAVE_JSON_FOR_INSPECTION already True")
        return
    patched, count = re.subn(
        r"SAVE_JSON_FOR_INSPECTION\s*=\s*False",
        "SAVE_JSON_FOR_INSPECTION = True",
        source,
    )
    if count:
        backup = script.with_suffix(".py.orig")
        if not backup.exists():
            shutil.copy2(script, backup)
        script.write_text(patched, encoding="utf-8")
        print(f"  set SAVE_JSON_FOR_INSPECTION = True (backup at {backup.name})")
    else:
        print("  WARNING: could not find SAVE_JSON_FOR_INSPECTION; running anyway")


def run_builder() -> bool:
    banner("Running their unified_input.py")
    ensure_save_flag()
    result = subprocess.run(
        [sys.executable, "unified_input.py"],
        cwd=DATA_DIR, capture_output=True, text=True, timeout=900,
    )
    if result.stdout.strip():
        print("\n  stdout:")
        for line in result.stdout.strip().splitlines()[-25:]:
            print(f"    {line}")
    if result.returncode != 0:
        print(f"\n  FAILED (exit {result.returncode}):")
        for line in (result.stderr or "").strip().splitlines()[-25:]:
            print(f"    {line}")
        return False
    return True


def approx_tokens(text: str) -> int:
    """~4 characters per token. Good enough to size a context budget."""
    return len(text) // 4


def distribution(values: list[int], label: str) -> None:
    if not values:
        return
    values = sorted(values)
    pick = lambda p: values[min(len(values) - 1, int(p * len(values)))]  # noqa: E731
    print(f"    {label:<26} n={len(values):<5} min={values[0]:>7,}  "
          f"p25={pick(.25):>7,}  median={int(median(values)):>7,}  "
          f"p75={pick(.75):>7,}  p95={pick(.95):>7,}  max={values[-1]:>8,}")


def describe(value, indent: int = 2, depth: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        if depth >= 2:
            return f"dict({len(value)} keys)"
        inner = "\n".join(
            f"{pad}{k}: {describe(v, indent + 2, depth + 1)}" for k, v in value.items()
        )
        return f"dict({len(value)} keys)\n{inner}"
    if isinstance(value, list):
        return f"list({len(value)})" if not value else \
            f"list({len(value)}) of {describe(value[0], indent + 2, depth + 1)}"
    if isinstance(value, str):
        return f"str(len={len(value)}) {value.replace(chr(10), ' ')[:64]!r}"
    return f"{type(value).__name__}({value})"


def main() -> int:
    if not DATA_DIR.exists():
        print(f"{DATA_DIR} not found. Run scripts/inspect_dataset.py first.")
        return 1

    if not OUTPUT.exists() and not run_builder():
        return 1
    if not OUTPUT.exists():
        print(f"\n{OUTPUT} was still not produced.")
        return 1

    samples = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if isinstance(samples, dict):
        samples = next((v for v in samples.values() if isinstance(v, list)), [])

    banner(f"unified_input_samples_v2.json -- {len(samples)} samples")
    print("\nFirst sample:")
    print(describe(samples[0]))

    keys = Counter(k for s in samples if isinstance(s, dict) for k in s)
    print(f"\nField presence:")
    for key, count in keys.most_common():
        flag = "" if count == len(samples) else "   <-- optional"
        print(f"    {key:<20} {count:>5}{flag}")

    categories = Counter(s.get("category", "?") for s in samples)
    print(f"\nCategories:")
    for name, count in categories.most_common():
        print(f"    {name:<20} {count:>5}  ({100 * count / len(samples):4.1f}%)")

    banner("Prompt size -- does the efficiency claim have headroom?")
    cognitive = [s for s in samples if s.get("category") == "Cognitive"]
    other = [s for s in samples if s.get("category") != "Cognitive"]
    for label, group in (("Cognitive", cognitive), ("all other categories", other)):
        if group:
            distribution([approx_tokens(s.get("input_prompt", "")) for s in group],
                         f"{label} (~tokens)")

    if cognitive:
        distribution([approx_tokens(s.get("evidence", "")) for s in cognitive],
                     "Cognitive evidence")
        distribution([approx_tokens(s.get("trigger", "")) for s in cognitive],
                     "Cognitive trigger")

        ratio = (median([approx_tokens(s.get("input_prompt", "")) for s in cognitive])
                 / max(1, median([approx_tokens(s.get("evidence", "")) for s in cognitive])))
        print(f"\n  Full prompt is ~{ratio:.0f}x the size of the cue that actually matters.")
        print("  That ratio is the ceiling on our efficiency claim.")

        banner("One complete Cognitive sample")
        sample = cognitive[0]
        for field in ("category", "time_gap", "trigger", "evidence"):
            if field in sample:
                print(f"\n  {field}:\n    {str(sample[field])[:600]}")
        prompt = sample.get("input_prompt", "")
        print(f"\n  input_prompt: {approx_tokens(prompt):,} tokens, first 700 chars:\n")
        print("    " + prompt[:700].replace("\n", "\n    "))
        print(f"\n    [...]\n\n    ...last 400 chars:\n")
        print("    " + prompt[-400:].replace("\n", "\n    "))

    banner("Next")
    print("  With this schema confirmed, the loader and the pilot experiment can be")
    print("  written against reality. The pilot answers one question before we")
    print("  build anything else: is the bottleneck FINDING the cue, or APPLYING it?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
