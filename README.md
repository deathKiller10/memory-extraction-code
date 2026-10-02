# Code and results for the anonymous TMLR submission

An extract–store–retrieve memory pipeline for LLM agents, evaluated on the
**Cognitive** split of LoCoMo-Plus. This repository contains the code, the
exact result files every number in the paper is computed from, and the tests.

Everything runs on free tiers: CPU only, plus free API keys for Groq and
Google Gemini. No paid API is used anywhere.

## Layout

| path | what it is |
|---|---|
| `bapca/` | the library: dataset loader, segmentation, extraction, memory store, selection, judging, statistics |
| `scripts/run_system.py` | runs our system (condition E) and writes `results/system_*.json` |
| `scripts/run_pilot.py` | runs the baselines (no memory, full context, and the two oracles) |
| `scripts/paired_vs_pilot.py` | paired exact McNemar test of E against one baseline |
| `scripts/compare_runs.py` | paired comparison of two `run_system.py` files |
| `scripts/extraction_ceiling.py` | the stage decomposition (was the cue written, was it selected) |
| `scripts/judge_agreement.py` | judge-vs-human agreement from `results/annotations/` |
| `scripts/paper_numbers.py` | regenerates every figure in the paper from `results/` |
| `results/` | the raw per-item outputs behind every reported number |
| `results/annotations/` | blind human labels: `annotator1_*`, `annotator2_*`, and `joint.json` (a jointly labelled session) |
| `tests/` | unit tests (`pytest`, runs in a few seconds, no network) |

## Setup

```
pip install -r requirements.txt
git clone https://github.com/xjtuleeyf/Locomo-Plus.git third_party/Locomo-Plus
python scripts/build_eval_set.py      # runs the benchmark's own builder
export GROQ_API_KEY=...               # extraction, re-ranking, judging
export GEMINI_API_KEY=...             # answer generation
```

The exact model identifiers and run dates are given in the paper's Method
section.

## Reproducing the headline

```
python scripts/run_system.py --n 401 --select llm --top-k 3 --strip-trigger --extract events
python scripts/run_pilot.py  --n 401 --conditions full_context
python scripts/paired_vs_pilot.py \
    results/system_n401_seed42_llm3_d1_notrig_events.json \
    results/pilot_n401_seed42_full_context.json
```

A full run exceeds one day of the Groq free tier. It stops cleanly when the
daily quota is reached; re-running the identical command resumes it. LLM
outputs are not bit-for-bit deterministic across providers and dates, which is
why the exact outputs we scored are included in `results/`.

## Recomputing the paper's numbers without any API call

```
python scripts/paired_vs_pilot.py results/system_n401_seed42_llm3_d1_notrig_events.json results/pilot_n401_seed42_full_context.json
python scripts/extraction_ceiling.py results/system_n401_seed42_llm3_d1_notrig_events.json
python scripts/judge_agreement.py --export   # rewrites results/judge_agreement.json
python scripts/paper_numbers.py       # writes paper/numbers.tex
```

`extraction_ceiling.py` loads a sentence-transformers embedder (CPU is fine);
the others read only `results/`.

## Does the extraction fix transfer beyond the Cognitive split?

The corrected extraction prompt is compared with the original on every
answerable question of the other four LoCoMo-Plus splits. Only extraction runs
(no answering or judging); about 1.6 days of the Groq free tier per prompt.

```
python scripts/extract_only.py --extract events
python scripts/extract_only.py --extract v1
python scripts/written_compare.py results/extract_only_v1.json results/extract_only_events.json --export transfer
python scripts/written_compare.py results/system_n100_seed42_llm3_d1_notrig.json \
    results/system_n100_seed42_llm3_d1_notrig_events.json --export cognitive
```

The extraction outputs we measured are included in `results/`, so the two
`written_compare.py` lines can be re-run without any API call.

## Tests

```
pytest -q
```
