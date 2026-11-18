"""
Measure what one write call really costs -- Groq's own usage numbers.

    python scripts/measure_call_cost.py            # ~6 calls, ~10-15k tokens

Run in Colab (needs the dataset and the API key). Uncached on purpose: the
point is to see real usage, so nothing is read from or written to the cache.

WHY (26 Oct). The first facts item took 26 minutes for ~43 extraction calls
and wrote 529 facts -- ~12 per window, against the ~150 per item the plan
assumed. Before spending days on phases 2-3 we need, per call: prompt and
completion tokens as Groq counts them, whether the output hit max_tokens
(finish_reason "length" -> a truncated fact list), and wall time. It also
sizes the facts x llm re-rank request from the first facts row already on
disk, against Groq's 8,000 tokens-per-minute ceiling for a single request.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.dataset import LocomoPlus
from bapca.llm import estimate_tokens, judge
from bapca.pipeline import (EXTRACT_PROMPT_EVENTS, EXTRACT_SYSTEM, FACTS_MAX_TOKENS,
                            FACTS_PROMPT, FACTS_SYSTEM, RERANK_PROMPT,
                            SUMMARY_MAX_TOKENS, SUMMARY_NO_DATE, SUMMARY_PROMPT,
                            SUMMARY_SYSTEM, parse_facts, segment, strip_trigger)

FACTS_FILE = Path("results/system_n100_seed42_oracle_window3_d1_notrig_facts.json")


def call(llm, prompt, system, max_tokens):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    started = time.time()
    resp = llm._create(messages, llm.temperature, max_tokens)
    took = time.time() - started
    u = resp.usage
    reasoning = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None)
    text = resp.choices[0].message.content or ""
    return dict(prompt=u.prompt_tokens, completion=u.completion_tokens, reasoning=reasoning,
                finish=resp.choices[0].finish_reason, seconds=round(took, 1), text=text)


def main() -> int:
    jdg = judge(verbose=False)
    print(f"judge model: {jdg.model}")
    sample = LocomoPlus().subset(100, seed=42, category="Cognitive")[0]
    windows = segment(strip_trigger(sample.input_prompt, sample.trigger), 12)[:3]

    jobs = []
    for w in windows:
        jobs.append(("events", EXTRACT_PROMPT_EVENTS.format(window=w.text[:6000]),
                     EXTRACT_SYSTEM, 160))
        jobs.append(("facts", FACTS_PROMPT.format(conversation=w.text[:6000]),
                     FACTS_SYSTEM, FACTS_MAX_TOKENS))
        jobs.append(("summary", SUMMARY_PROMPT.format(conversation=w.text[:6000],
                                                      timestamp=SUMMARY_NO_DATE),
                     SUMMARY_SYSTEM, SUMMARY_MAX_TOKENS))
    print(f"\n{'policy':<8} {'prompt':>6} {'compl':>6} {'reason':>6} {'total':>6} "
          f"{'finish':<7} {'sec':>5}  output")
    totals = {}
    for policy, prompt, system, budget in jobs:
        r = call(jdg, prompt, system, budget)
        total = r["prompt"] + r["completion"]
        totals.setdefault(policy, []).append(total)
        extra = (f"{len(parse_facts(r['text'], 0))} facts" if policy == "facts"
                 else f"{len(r['text'])} chars")
        print(f"{policy:<8} {r['prompt']:>6} {r['completion']:>6} {str(r['reasoning']):>6} "
              f"{total:>6} {r['finish']:<7} {r['seconds']:>5}  {extra}", flush=True)
        time.sleep(15)          # stay well under 8,000 tokens per minute
    print("\nmean total tokens per call:")
    for policy, vals in totals.items():
        print(f"  {policy:<8} {sum(vals) / len(vals):,.0f}")

    if FACTS_FILE.exists():
        rows = json.loads(FACTS_FILE.read_text(encoding="utf-8"))
        if rows:
            notes = rows[0]["all_notes"]
            listing = "\n".join(f"  {i + 1}. {n}" for i, n in enumerate(notes))
            est = estimate_tokens(RERANK_PROMPT + listing) + 160
            print(f"\nfacts x llm: item #{rows[0]['sample_index']} has {len(notes)} facts; "
                  f"one re-rank request ~{est:,} tokens against Groq's 8,000/minute "
                  f"ceiling -> {'REFUSED (PromptTooLarge)' if est > 8000 else 'fits'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
