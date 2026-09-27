"""
Look at the real LoCoMo-Plus data before writing a line of loader code.

Guessing a schema and writing a parser against the guess is how you end up with
a loader that silently mis-reads half the fields and an experiment that runs
cleanly on the wrong data. So: clone, inspect, report. The loader comes after.

    python scripts/inspect_dataset.py

There is one measurement here that is not just plumbing. LoCoMo-Plus ships the
cue-query similarity scores it used to FILTER the benchmark (mpnet, bge, bm25).
Those numbers are the quantitative case for our whole approach: they show how
far apart cue and query were deliberately pushed, and therefore how little a
similarity retriever has to work with. If the distribution looks the way we
expect, it becomes a figure in the paper.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

REPO_URL = "https://github.com/xjtuleeyf/Locomo-Plus.git"
REPO_DIR = Path("third_party/Locomo-Plus")


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def ensure_repo() -> bool:
    if REPO_DIR.exists():
        print(f"Repository already present at {REPO_DIR}")
        return True
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {REPO_URL} ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Clone failed:\n{result.stderr}")
        return False
    return True


def describe(value, indent: int = 2, depth: int = 0) -> str:
    """One-line shape description, recursing a little into containers."""
    pad = " " * indent
    if isinstance(value, dict):
        if depth >= 2:
            return f"dict({len(value)} keys)"
        inner = "\n".join(
            f"{pad}{k}: {describe(v, indent + 2, depth + 1)}" for k, v in value.items()
        )
        return f"dict({len(value)} keys)\n{inner}"
    if isinstance(value, list):
        if not value:
            return "list(empty)"
        return f"list({len(value)}) of {describe(value[0], indent + 2, depth + 1)}"
    if isinstance(value, str):
        preview = value.replace("\n", " ")[:70]
        return f"str(len={len(value)}) {preview!r}"
    return f"{type(value).__name__}({value})"


def load(path: Path):
    if not path.exists():
        print(f"  MISSING: {path}")
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def summarise_field(records: list[dict], field: str, limit: int = 12) -> None:
    values = [r.get(field) for r in records if isinstance(r, dict) and field in r]
    if not values:
        return
    counts = Counter(str(v) for v in values)
    print(f"\n  {field} -- {len(counts)} distinct across {len(values)} records:")
    for value, count in counts.most_common(limit):
        share = 100 * count / len(values)
        print(f"    {value[:46]:<48} {count:>5}  ({share:4.1f}%)")
    if len(counts) > limit:
        print(f"    ... and {len(counts) - limit} more")


def summarise_numeric(records: list[dict], path: tuple[str, ...]) -> None:
    values = []
    for record in records:
        node = record
        for part in path:
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if isinstance(node, (int, float)):
            values.append(float(node))
    if not values:
        return
    values.sort()
    name = ".".join(path)
    pct = lambda p: values[min(len(values) - 1, int(p * len(values)))]  # noqa: E731
    print(f"    {name:<28} n={len(values):<6} min={values[0]:+.3f}  "
          f"p25={pct(.25):+.3f}  median={median(values):+.3f}  "
          f"p75={pct(.75):+.3f}  max={values[-1]:+.3f}  mean={mean(values):+.3f}")


def inspect_locomo_plus(records) -> None:
    banner("locomo_plus.json -- the cue/trigger pairs")

    if isinstance(records, dict):
        print(f"Top level is a dict with keys: {list(records.keys())[:20]}")
        for key, value in records.items():
            if isinstance(value, list) and value:
                print(f"Using records from key {key!r}")
                records = value
                break

    if not isinstance(records, list):
        print(f"Unexpected top-level type: {type(records).__name__}")
        return

    print(f"\n{len(records)} records.")
    print("\nFirst record, field by field:")
    print(describe(records[0]))

    keys = Counter(k for r in records if isinstance(r, dict) for k in r)
    print(f"\nField presence across all {len(records)} records:")
    for key, count in keys.most_common():
        flag = "" if count == len(records) else "   <-- not on every record"
        print(f"    {key:<28} {count:>5}{flag}")

    for field in ("relation_type", "time_gap", "model_name", "category", "constraint_type"):
        summarise_field(records, field)

    # The number that matters for the paper.
    banner("Cue-query similarity: how much signal a retriever actually has")
    print("  LoCoMo-Plus filtered these pairs to REMOVE lexical/semantic overlap")
    print("  (their section 4.4). These are the scores that survived that filter.\n")
    for path in [
        ("final_similarity_score",),
        ("scores", "mpnet"),
        ("scores", "bge"),
        ("scores", "bm25"),
        ("scores", "combined"),
    ]:
        summarise_numeric(records, path)
    print("\n  Read this against our own measurement (cosine 0.047 on the worked")
    print("  example). Low values here = similarity retrieval is structurally")
    print("  handicapped on this benchmark, which is exactly our thesis.")


def inspect_locomo10(data) -> None:
    banner("locomo10.json -- the long dialogues cues get inserted into")
    if data is None:
        return
    if isinstance(data, list):
        print(f"{len(data)} conversations.")
        sample = data[0]
    elif isinstance(data, dict):
        print(f"dict with {len(data)} keys: {list(data.keys())[:10]}")
        sample = next(iter(data.values()))
    else:
        print(f"Unexpected type: {type(data).__name__}")
        return
    print("\nFirst conversation, shape:")
    print(describe(sample))


def main() -> int:
    if not ensure_repo():
        print("\nCould not fetch the dataset. Check network access, or clone it")
        print(f"manually into {REPO_DIR}")
        return 1

    banner("Repository contents")
    for path in sorted(REPO_DIR.rglob("*")):
        if ".git/" in str(path) or path.is_dir():
            continue
        rel = path.relative_to(REPO_DIR)
        size = path.stat().st_size
        marker = "  <--" if path.suffix == ".json" else ""
        print(f"  {str(rel):<62} {size:>10,} B{marker}")

    data_dir = REPO_DIR / "data"
    plus = load(data_dir / "locomo_plus.json")
    if plus is not None:
        inspect_locomo_plus(plus)
    inspect_locomo10(load(data_dir / "locomo10.json"))

    banner("What this tells us for the loader")
    print("  * Field names above are the contract -- the loader is written against")
    print("    these, not against anything assumed.")
    print("  * `unified_input_samples_v2.json` is gitignored in their repo: the long")
    print("    dialogues are BUILT by data/build_conv.py + data/unified_input.py.")
    print("    Next step is to run those and inspect the result, not to guess it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
