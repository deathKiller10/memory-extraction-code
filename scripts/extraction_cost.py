"""
How many extraction calls one item costs -- counted from the data.

    python scripts/extraction_cost.py [--n 401] [--export]

Needs the dataset (third_party/Locomo-Plus), no API, no embedder.

BUG 25 (24 Oct). paper_numbers.py set \\WindowsPerItem to the median of
`notes_extracted` -- the notes that SURVIVED, after every window that answered
NONE was dropped. But extraction makes one call per WINDOW, NONE or not
(pipeline.extract_notes). On the 401 split that is 41 calls per item, not 30,
so the paper's token-cost paragraph under-stated the write-time cost by ~37%.
This counts windows exactly as run_system.py cuts them (strip-trigger on,
12-turn content-defined windows) and exports the count, so the macro is read
from the data instead of from a proxy.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.dataset import LocomoPlus
from bapca.pipeline import DEFAULT_WINDOW_TURNS, segment, strip_trigger

RESULTS = Path("results")


def windows_per_item(samples, window_turns=DEFAULT_WINDOW_TURNS):
    per_item, unique = [], set()
    for s in samples:
        ws = segment(strip_trigger(s.input_prompt, s.trigger), window_turns)
        per_item.append(len(ws))
        unique.update(w.text for w in ws)
    return per_item, unique


def main() -> int:
    n = 401
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    samples = LocomoPlus().subset(n, seed=42, category="Cognitive")
    per_item, unique = windows_per_item(samples)
    med = statistics.median(per_item)
    print(f"items {len(samples)}: windows per item median {med:g} "
          f"(min {min(per_item)}, max {max(per_item)}); "
          f"distinct windows across the set {len(unique):,}")
    if "--export" in sys.argv:
        # n=401 keeps its historical name (paper_numbers reads it for the token
        # paragraph); any other n gets its own file, e.g. the grid's n=100.
        dest = RESULTS / ("extraction_cost.json" if n == 401 else f"extraction_cost_n{n}.json")
        dest.write_text(json.dumps(dict(
            n_items=len(samples), seed=42, window_turns=DEFAULT_WINDOW_TURNS,
            strip_trigger=True, median_windows_per_item=med,
            min_windows=min(per_item), max_windows=max(per_item),
            distinct_windows=len(unique)), indent=1), encoding="utf-8")
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
