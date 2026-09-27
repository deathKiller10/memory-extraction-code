"""
Build a blind annotation batch as a self-contained HTML page.

    python scripts/build_annotation_batch.py            # run in Colab: needs the dataset
    -> paper/annotation_batch.html

Free: no API calls, no embedder. It needs `LocomoPlus`, so it runs in Colab.
Download the HTML, open it in a browser on any machine, annotate offline, and
press Download to get a JSON that drops straight into results/annotations/.

WHY A NEW BATCH. Measured on 6 Sep (`judge_agreement.py`), the existing study
is n=15 per condition on the jointly annotated set. At 4 discordant pairs you
need 5-1 to reach p<0.05, which is arithmetically impossible. It cannot support
or refute its own conclusions.

AND CONDITION E HAS NEVER BEEN ANNOTATED BY A HUMAN AT ALL. The paper's
headline is condition E's score. Every human label so far covers the four pilot
conditions only. That is the single largest hole in the study.

DESIGN. 100 items, stratified, weighted toward E because it starts from zero:

    E (ours)            40      <- currently 0 human labels
    no memory           15
    full context        15
    oracle cue          15
    oracle constraint   15

BOTH annotators label ALL 100 independently. That gives Cohen's kappa on n=100
(against the current n=40), and brings every condition to roughly 40 human
labels. Items already annotated by anyone are excluded, so this extends the
study rather than re-covering it.

The condition is hidden and the order is shuffled. An annotator who can tell
which system produced a response is not blind, and the validation is worthless.
"""

from __future__ import annotations

import html
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.dataset import LocomoPlus

RESULTS = Path("results")
ANNOTATIONS = RESULTS / "annotations"
OUT = Path("paper/annotation_batch.html")

SOURCES = [
    (RESULTS / "pilot_n100_seed42.json", None),
    (RESULTS / "system_n100_seed42_llm3_d1_notrig_events.json", "system_llm"),
]
WEIGHTS = {"system_llm": 40, "no_memory": 15, "full_context": 15,
           "oracle_cue": 15, "oracle_constraint": 15}
SEED = 11


def already_annotated() -> set:
    seen = set()
    if ANNOTATIONS.exists():
        for path in ANNOTATIONS.glob("*.json"):
            seen |= set(json.loads(path.read_text(encoding="utf-8")))
    return seen


def collect() -> dict:
    grouped: dict = {}
    for path, only in SOURCES:
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        print(f"  {path}")
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("label") not in ("correct", "wrong"):
                continue
            cond = row.get("condition")
            if only and cond != only:
                continue
            grouped.setdefault(cond, []).append(row)
    return grouped


def matched_queue(grouped, done) -> list:
    """
    The SAME sample indices that were annotated under condition E, but under
    full context.

    Batch 2 gave us human labels for condition E (40 items) and for full
    context (15) -- but on DIFFERENT items, so the human E-vs-B comparison is
    unpaired, and this project has retracted three claims that rested on
    unpaired comparisons. Re-annotating the identical indices under B makes the
    headline's human validation a paired McNemar instead of two rates side by
    side.
    """
    want = sorted(int(k.split(":")[1]) for k in done if k.startswith("system_llm:"))
    by_index = {r["sample_index"]: r for r in grouped.get("full_context", [])}
    queue, absent, already = [], [], []
    for index in want:
        key = f"full_context:{index}"
        if key in done:
            already.append(index)
        elif index in by_index:
            queue.append(by_index[index])
        else:
            absent.append(index)
    print(f"\n  condition-E items already annotated:      {len(want)}")
    print(f"  of those, full context not yet annotated: {len(queue)}")
    if already:
        print(f"  already annotated under full context:     {len(already)} (skipped)")
    if absent:
        print(f"  WARNING: no full-context row for {len(absent)} of them: {absent[:8]}")
    return queue


def main() -> int:
    matched = "--matched" in sys.argv
    print("Files actually read (check these are the ones you meant):")
    grouped = collect()
    done = already_annotated()
    print(f"\n  already annotated by someone: {len(done)} items (excluded)")

    if matched:
        queue = matched_queue(grouped, done)
        random.Random(SEED).shuffle(queue)
        return write_page(queue, Path("paper/annotation_batch_matched.html"))

    rng = random.Random(SEED)
    queue = []
    for cond, want in WEIGHTS.items():
        pool = [r for r in grouped.get(cond, [])
                if f"{cond}:{r['sample_index']}" not in done]
        take = pool if len(pool) <= want else rng.sample(pool, want)
        if len(pool) < want:
            print(f"  WARNING: {cond} has only {len(pool)} unannotated items, wanted {want}")
        queue.extend(take)
        print(f"  {cond:<22} {len(take):>3} of {len(pool)} available")
    rng.shuffle(queue)

    return write_page(queue, OUT)


def write_page(queue, out_path) -> int:
    if not queue:
        print("\n  Nothing to annotate -- every item is already labelled.")
        return 0
    samples = {s.index: s for s in LocomoPlus()}
    items = []
    for row in queue:
        s = samples.get(row["sample_index"])
        if not s:
            continue
        items.append({
            "key": f"{row['condition']}:{row['sample_index']}",
            "cue": s.evidence,
            "message": s.trigger,
            "response": row["prediction"],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # A distinct localStorage key per batch, or the second batch resumes into
    # the first one's saved answers and silently skips every item.
    page = (PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
                .replace("__STORAGE_KEY__", "bapca_annotation_" + out_path.stem))
    out_path.write_text(page, encoding="utf-8")
    print(f"\nWrote {out_path}  ({len(items)} items)")
    print("\nDownload it, open it in a browser, and annotate. Both of you label ALL")
    print("of them, independently. Press Download when finished (or part-way -- it")
    print("saves as you go and resumes).")
    return 0


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>BAPCA annotation</title><style>
:root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#666;--line:#ddd;--card:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#16161a;--fg:#eee;--mut:#999;--line:#333;--card:#1f1f24}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:24px}
.bar{height:5px;background:var(--line);border-radius:3px;overflow:hidden;margin:10px 0 22px}
.bar>i{display:block;height:100%;background:#3b7d4f;width:0;transition:width .2s}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px;margin-bottom:16px}
.lab{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin-bottom:6px}
.txt{white-space:pre-wrap}
button{font:inherit;padding:10px 20px;margin-right:8px;border:1px solid var(--line);
border-radius:8px;background:var(--card);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--mut)}
.y{border-color:#3b7d4f}.n{border-color:#a4453a}
small{color:var(--mut)}
#done{display:none;text-align:center;padding:40px 0}
</style></head><body><div class="wrap">
<h2>Does the response reflect the earlier note?</h2>
<p><small>The condition is hidden and the order is shuffled, on purpose. Judge only
what you see. Keys: <b>Y</b> yes &middot; <b>N</b> no &middot; <b>S</b> skip &middot;
<b>&larr;</b> back. Your work is saved in this browser as you go.</small></p>
<div class="bar"><i id="fill"></i></div>
<div id="live">
  <div class="card"><div class="lab">Earlier note (the cue)</div><div class="txt" id="cue"></div></div>
  <div class="card"><div class="lab">Their message now</div><div class="txt" id="msg"></div></div>
  <div class="card"><div class="lab">The response</div><div class="txt" id="rsp"></div></div>
  <p><button class="y" onclick="mark('correct')">Yes &nbsp;<small>Y</small></button>
     <button class="n" onclick="mark('wrong')">No &nbsp;<small>N</small></button>
     <button onclick="mark('skip')">Skip &nbsp;<small>S</small></button>
     <button onclick="back()">&larr; Back</button></p>
</div>
<div id="done"><h3>All done.</h3></div>
<p><small id="count"></small></p>
<p><input id="who" placeholder="your name, e.g. annotator2" style="padding:9px;border-radius:8px;
border:1px solid var(--line);background:var(--card);color:var(--fg)">
<button onclick="save()">Download labels</button></p>
</div><script>
const ITEMS = __ITEMS__;
const KEY = "__STORAGE_KEY__";
let labels = {}, i = 0;
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { labels = {}; }
function first(){ let k=0; while(k<ITEMS.length && labels[ITEMS[k].key]) k++; return k; }
function show(){
  const total = ITEMS.length, n = Object.keys(labels).length;
  document.getElementById("fill").style.width = (100*n/total)+"%";
  document.getElementById("count").textContent = n+" of "+total+" labelled";
  if (i >= total){ document.getElementById("live").style.display="none";
                   document.getElementById("done").style.display="block"; return; }
  document.getElementById("live").style.display="block";
  document.getElementById("done").style.display="none";
  const it = ITEMS[i];
  document.getElementById("cue").textContent = it.cue;
  document.getElementById("msg").textContent = it.message;
  document.getElementById("rsp").textContent = it.response;
}
function mark(v){ labels[ITEMS[i].key]=v;
  try{localStorage.setItem(KEY, JSON.stringify(labels));}catch(e){}
  i++; show(); }
function back(){ if(i>0){ i--; show(); } }
document.addEventListener("keydown", e=>{
  const k=e.key.toLowerCase();
  if(k==="y")mark("correct"); else if(k==="n")mark("wrong");
  else if(k==="s")mark("skip"); else if(e.key==="ArrowLeft")back();
});
function save(){
  const who=(document.getElementById("who").value||"annotator").trim().toLowerCase()
            .replace(/[^a-z0-9_]+/g,"_");
  const blob=new Blob([JSON.stringify(labels,null,1)],{type:"application/json"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob); a.download=who+"_"+KEY.replace("bapca_annotation_","")+".json"; a.click();
}
i = first(); show();
</script></body></html>"""


if __name__ == "__main__":
    sys.exit(main())
