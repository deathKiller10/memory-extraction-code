"""
Condition E: our system, with no oracle.

    python scripts/run_system.py --dry-run     # exact cost, spends nothing
    python scripts/run_system.py --n 100

E receives the same conversation the full-context baseline receives, and has to
find the constraint itself: segment, extract, score, carry the top few. It is
compared against the pilot's existing results on the same items, so the numbers
line up directly.

Two outcomes are worth having:

  E close to D (~83% human)  -> the pipeline works end to end, and the oracle
                                was not doing the heavy lifting.
  E well below D             -> the components work but the search does not.
                                The `note carried` rate tells us which half is
                                broken, which is why we measure it separately.

Either way we learn something publishable. A negative result with a clean
diagnosis is a paper; a positive one without a diagnosis is not.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.dataset import LocomoPlus
from bapca.embeddings import Embedder
from bapca.evaluation import ASSISTANT_SYSTEM, judge_response
from bapca.llm import (DailyQuotaExhausted, PromptTooLarge,
                       TransientProviderError, generator, judge)
from bapca.pipeline import (EXTRACT_PROMPTS, RERANK_PROMPTS, SystemConfig, build_store,
                            estimate_unique_windows,
                            extract_notes, hit_rate, locate_evidence_window,
                            render_prompt, segment, select_notes,
                            strip_trigger)
from bapca.stats import Proportion, mcnemar

RESULTS_DIR = Path("results")

# Groq free tier, observed from a 429 on 1 Sep 2026:
#   "on tokens per day (TPD): Limit 200000, Used 199303"
GROQ_TPD = 200_000
# Measured the same day: 199,303 tokens bought 237 cold extraction calls
# plus 10 re-ranks and 10 judgings.
TOKENS_PER_EXTRACTION = 778
# That 778 was measured on Cognitive windows, whose distinct texts average 529
# tokens (len/4). The rest -- instructions plus the note written back -- is a
# fixed ~249 per call. Non-Cognitive items are segmented at real session
# boundaries and their windows average ~836 tokens, so a flat 778 under-quoted
# a non-Cognitive run by ~40%. The estimate now scales with the windows
# actually being priced, and reproduces 778 on the Cognitive windows it was
# measured on. (27 Sep)
MEASURED_WINDOW_TOKENS = 529
EXTRACTION_OVERHEAD = TOKENS_PER_EXTRACTION - MEASURED_WINDOW_TOKENS


def tokens_per_extraction(window_texts) -> int:
    """Estimated Groq tokens for one extraction call on windows like these."""
    texts = list(window_texts)
    if not texts:
        return TOKENS_PER_EXTRACTION
    mean = sum(len(t) // 4 for t in texts) / len(texts)
    return round(EXTRACTION_OVERHEAD + mean)


def resume_command(args) -> str:
    """
    The exact command that resumes this run, printed when quota runs out.

    It used to rebuild the flags by hand and left out --category and --seed.
    On a non-Cognitive run, pasting the printed line the next day would have
    pointed at the Cognitive results file instead of resuming -- the same
    one-flag-missing shape as bug 12. Every flag that goes into the results
    filename must be here. (27 Sep)
    """
    flags = f"--n {args.n} --select {args.select} --top-k {args.top_k}"
    if args.seed != 42:
        flags += f" --seed {args.seed}"
    if args.carry:
        flags += f" --carry {args.carry}"
    if args.window_turns != 12:
        flags += f" --window-turns {args.window_turns}"
    if args.days_per_window != 1.0:
        flags += f" --days-per-window {args.days_per_window:g}"
    if args.strip_trigger:
        flags += " --strip-trigger"
    if args.extract != "v1":
        flags += f" --extract {args.extract}"
    if args.rerank != "v1":
        flags += f" --rerank {args.rerank}"
    if args.category != "Cognitive":
        flags += f" --category {args.category}"
    return f"!python scripts/run_system.py {flags}"

# One overloaded model should cost one item, not the session. Several in a
# row means the provider is genuinely down, and continuing would pay for
# extraction and re-ranking on items whose generation will fail anyway.
MAX_CONSECUTIVE_STALLS = 3



def _source(sample, strip: bool) -> str:
    """The text the memory is built from: the conversation, and -- unless
    --strip-trigger is set -- the query appended to the end of it."""
    return strip_trigger(sample.input_prompt, sample.trigger) if strip else sample.input_prompt


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def dry_run(samples, config, strip: bool = False, select: str = "llm",
            extract: str = "v1") -> None:
    banner("Dry run -- exact cost, not an estimate")

    import dataclasses
    sources = [_source(s, strip) for s in samples]
    priced = [dataclasses.replace(s, input_prompt=src) for s, src in zip(samples, sources)]
    total, unique = estimate_unique_windows(priced, config.window_turns)

    # Subtract the windows a previous run of the SAME configuration already
    # paid for. Without this the estimate is a cold-cache figure: for n=401 it
    # reported 1,731 calls, 1.3M tokens, "7 DAYS" and "OVER the request limit
    # by 731" for a job that actually needs 962 calls, 732k tokens and 4 days,
    # because 569 of the 929 windows were cached from the n=100 run. An
    # estimate that over-quotes by 80% is the mirror of bug 16 and just as
    # likely to make a decision go the wrong way. (15 Sep)
    done_items = set()
    for path in sorted(RESULTS_DIR.glob("system_*.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for row in rows:
            if (row.get("extract") == extract
                    and bool(row.get("strip_trigger")) == bool(strip)):
                done_items.add(row["sample_index"])

    cached_windows = 0
    if done_items:
        already = [pr for pr, s in zip(priced, samples) if s.index in done_items]
        if already:
            _, cached_windows = estimate_unique_windows(already, config.window_turns)
        print(f"\n  {len(done_items)} of these items were already run with the same")
        print(f"  extraction and trigger settings; {cached_windows:,} of the windows")
        print("  below are therefore already cached and cost nothing.")

    dated = sum(1 for w in segment(sources[0], config.window_turns) if w.dated)

    how = ("session boundaries" if dated
           else f"content-defined windows, target {config.window_turns} turns")
    print(f"\n  items: {len(samples)}")
    print(f"  segmentation: {how}")
    print(f"  windows in total:  {total:,}")
    print(f"  distinct windows:   {unique:,}")
    print(f"  saved by dedup:     {total - unique:,} "
          f"({100*(total-unique)/max(1,total):.0f}%)")
    # This line used to read "929 <- what we actually pay for" while the Groq
    # line below said 360. Two numbers, one label, disagreeing on the same
    # screen. Say the payable figure once, here, and let the breakdown agree.
    print(f"  already cached:     {cached_windows:,}")
    print(f"  NEW windows to extract: {max(0, unique - cached_windows):,}"
          "   <- what we actually pay for")
    print("\n  The ten base conversations repeat across items, and extraction is")
    print("  keyed by window text, so the repeats cost nothing.")

    # Three kinds of Groq call, not two. The re-ranker runs on the judge
    # provider (`select_notes(..., llm=jdg)`), so `--select llm` and `hybrid`
    # each add one call per item. Omitting them under-reported the cost of the
    # `--extract events` run by 100 calls against an RPD of 1,000, which is the
    # difference between "one run fits today" and "it does not".
    new_windows = max(0, unique - cached_windows)
    new_items = len(samples) - len(done_items)
    rerank = new_items if select in ("llm", "hybrid") else 0
    groq = new_windows + rerank + new_items  # extraction + re-ranking + judging
    gemini = new_items                      # generation
    print(f"\n  Groq   (extract {new_windows:,} + re-rank {rerank:,} + judge "
          f"{new_items:,}): {groq:,} calls  | RPD 1,000")
    print(f"  Gemini (generate):        {gemini:,} calls  | RPD 1,000")

    # Measured, not assumed. The 1 Sep events run reported 199,303 Groq tokens
    # for 237 cold extractions plus 10 re-ranks and 10 judgings, i.e. ~778
    # tokens per extraction call. The previous constant here was 460, which
    # under-priced the day by 40% and hid the fact that the limit which
    # actually stops a run is TOKENS per day, not requests. Bug 16.
    per_call = tokens_per_extraction(
        {w.text for src in sources for w in segment(src, config.window_turns)})
    groq_tokens = new_windows * per_call + rerank * 600 + new_items * 900
    minutes = max(groq / 30, groq_tokens / 8000) + len(samples) / 15
    print(f"\n  Groq tokens: ~{groq_tokens:,}  ({per_call}/extraction: "
          f"{TOKENS_PER_EXTRACTION} measured 1 Sep on Cognitive windows, "
          "scaled to these windows' length)")
    print(f"  expect roughly {minutes:.0f} minutes of throughput "
          "(Groq is token-bound at 8,000/min)")

    tpd = GROQ_TPD
    days = math.ceil(groq_tokens / tpd)
    print(f"\n  Groq TOKENS PER DAY: {tpd:,}. This run needs ~{groq_tokens:,}.")
    if days > 1:
        print(f"  ==> {days} DAYS, not one. The run will stop partway on each of the")
        print("      first days; re-run the identical command after the reset and it")
        print("      resumes. Every completed extraction is cached, so no work repeats.")
        print(f"      A single day covers roughly "
              f"{int(tpd / max(1, groq_tokens) * max(1, new_items))} more of these items.")
    else:
        print("  ==> fits in one day.")

    if groq > 1000:
        print(f"\n  Also OVER the Groq request limit by {groq - 1000}.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--carry", type=int, default=0,
                        help="standing notes per query; 0 carries every survivor")
    parser.add_argument("--window-turns", type=int, default=12)
    parser.add_argument("--days-per-window", type=float, default=1.0,
                        help="how fast to age older windows; 7.0 pruned most notes")
    parser.add_argument("--select", choices=["carry", "similar", "llm", "hybrid"],
                        default="carry",
                        help="carry everything, retrieve by similarity, let a model "
                             "choose, or a model with similarity filling empty slots")
    parser.add_argument("--top-k", type=int, default=3,
                        help="notes to retrieve when --select similar/llm/hybrid")
    parser.add_argument("--extract", choices=sorted(EXTRACT_PROMPTS), default="v1",
                        help="extraction wording. 'v1' discards one-off events, which "
                             "is every gold cue in this benchmark; 'events' asks for the "
                             "event AND the change it caused. Switching invalidates the "
                             "extraction cache: ~569 calls, one day of quota.")
    parser.add_argument("--rerank", choices=sorted(RERANK_PROMPTS), default="v1",
                        help="re-rank wording. 'person' names the speaker the "
                             "system is replying to, recovered from the "
                             "conversation itself (never from the gold evidence "
                             "field). Four of the n=100 selection failures "
                             "carried notes about the other speaker. Invalidates "
                             "only the re-rank + judge cache: ~150,000 tokens.")
    parser.add_argument("--strip-trigger", action="store_true",
                        help="remove the query from the end of the conversation before "
                             "segmenting, so extraction cannot write a note that merely "
                             "restates the question (see pipeline.strip_trigger)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--category", default="Cognitive",
                        help="which LoCoMo-Plus split to evaluate. The loader "
                             "already supports every category in the file; "
                             "this exposes it. Use a non-Cognitive split to "
                             "test whether the events extraction prompt helps "
                             "anywhere other than the split it was written "
                             "around. NOTE: only Cognitive strips lexical "
                             "overlap between cue and query, so a score on "
                             "another split is NOT comparable to the headline "
                             "-- what transfers is the v1-vs-events DIFFERENCE, "
                             "not the level.")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any partial results and start over")
    args = parser.parse_args()

    config = SystemConfig(window_turns=args.window_turns, carry=args.carry,
                          days_per_window=args.days_per_window)
    data = LocomoPlus()
    samples = data.subset(args.n, seed=args.seed, category=args.category)
    if not samples:
        available = sorted({s.category for s in data})
        print(f"No samples in category {args.category!r}.")
        print(f"Categories present in this file: {available}")
        return 1
    print(f"Loaded {len(data)} samples; evaluating {len(samples)} "
          f"{args.category} items (seed {args.seed}).")

    if args.dry_run:
        dry_run(samples, config, strip=args.strip_trigger, select=args.select,
                extract=args.extract)
        return 0

    embedder = Embedder()
    gen, jdg = generator(verbose=True), judge(verbose=True)
    print(f"  generator: {gen.model}\n  judge:     {jdg.model}")

    RESULTS_DIR.mkdir(exist_ok=True)
    if args.select == "carry":
        tag = f"carry{args.carry or 'all'}"
    else:
        tag = f"{args.select}{args.top_k}"      # similar3, llm3, ...
    tag += f"_d{args.days_per_window:g}"
    if args.strip_trigger:
        tag += "_notrig"
    if args.extract != "v1":
        tag += f"_{args.extract}"
    if args.rerank != "v1":
        tag += f"_{args.rerank}"       # bug 12: every configuration, its own file
    if args.category != "Cognitive":
        # bug 12 again: a different split is a different experiment and must
        # never share a results file with the Cognitive run.
        tag += "_" + args.category.replace(" ", "").replace("-", "")
    out = RESULTS_DIR / f"system_n{len(samples)}_seed{args.seed}_{tag}.json"

    # Resume. A disconnected Colab runtime is the normal case, not the
    # exception, and the cached calls make a restart free -- but re-walking
    # 4,000 cached windows still wastes minutes. Skip what is already recorded.
    rows: list[dict] = []
    if out.exists() and not args.fresh:
        rows = json.loads(out.read_text(encoding="utf-8"))
        print(f"\n  Resuming: {len(rows)} items already recorded in {out.name}")
    finished = {r["sample_index"] for r in rows}

    banner(f"Running condition E -- progress saved to {out}")
    started = time.time()
    stalled = 0

    for i, sample in enumerate(samples, 1):
        if sample.index in finished:
            continue
        try:
            windows = segment(_source(sample, args.strip_trigger), config.window_turns)
            notes = extract_notes(jdg, windows,  # the judge provider: small prompts
                                  prompt=EXTRACT_PROMPTS[args.extract])
            store = build_store(notes, embedder, config)
            chosen = select_notes(store, sample, embedder, mode=args.select,
                                  top_k=args.top_k, llm=jdg, rerank=args.rerank)
        except DailyQuotaExhausted as exc:
            # Bug 16: a cold extraction pass costs ~3x the Groq daily TOKEN
            # budget, so stopping partway through is the normal case, not a
            # failure. Say what to do instead of printing a traceback.
            banner("Out of quota for today -- this is a pause, not a failure")
            print(f"  {exc}\n")
            print(f"  {len(rows)} of {len(samples)} items are recorded in {out.name}.")
            print("  Every extraction made today is cached on Drive, so tomorrow")
            print("  resumes where this stopped. Run the identical command:\n")
            print(f"    {resume_command(args)}\n")
            print("  Do NOT add --fresh: it would discard the finished items.")
            print("\n  Free analysis you can run right now on the partial file:")
            print(f"    !python scripts/extraction_ceiling.py {out}")
            return 0
        except TransientProviderError as exc:
            stalled += 1
            print(f"  [{i}/{len(samples)}] #{sample.index}: provider hiccup during extraction/selection "
                  f"({stalled} in a row) -- skipping this item\n      {exc}")
            if stalled >= MAX_CONSECUTIVE_STALLS:
                banner("Provider is down -- stopping cleanly")
                print(f"  {MAX_CONSECUTIVE_STALLS} items in a row failed. Continuing would pay for")
                print("  extraction and re-ranking on items that cannot finish.")
                print(f"\n  {len(rows)} of {len(samples)} items are recorded in {out.name}.")
                print("  Wait a few minutes and re-run the IDENTICAL command; it resumes.")
                return 0
            continue
        prompt, tokens, carried = render_prompt(sample, chosen)

        # Exact diagnostic: which window held the cue, and did a note from it
        # survive into the prompt? Replaces a cosine threshold that compared
        # raw dialogue against a paraphrase and under-reported.
        cue_window = locate_evidence_window(windows, sample.evidence)
        cue_extracted = cue_window is not None and cue_window in {n.window_index for n in notes}
        cue_selected = cue_window is not None and cue_window in {n.source_turn for n in chosen}

        try:
            prediction = gen.complete(prompt, system=ASSISTANT_SYSTEM, max_tokens=300)
            verdict = judge_response(jdg, sample, prediction)
        except PromptTooLarge as exc:
            print(f"  [{i}/{len(samples)}] #{sample.index}: {exc}")
            continue
        except DailyQuotaExhausted as exc:
            banner("Out of quota for today -- this is a pause, not a failure")
            print(f"  {exc}\n")
            print(f"  {len(rows)} of {len(samples)} items are recorded in {out.name}.")
            print("  Re-run the identical command after the daily reset; it resumes:\n")
            print(f"    {resume_command(args)}\n")
            print("  Do NOT add --fresh: it would discard the finished items.")
            return 0
        except TransientProviderError as exc:
            stalled += 1
            print(f"  [{i}/{len(samples)}] #{sample.index}: provider hiccup during generation/judging "
                  f"({stalled} in a row) -- skipping this item\n      {exc}")
            if stalled >= MAX_CONSECUTIVE_STALLS:
                banner("Provider is down -- stopping cleanly")
                print(f"  {MAX_CONSECUTIVE_STALLS} items in a row failed. Continuing would pay for")
                print("  extraction and re-ranking on items that cannot finish.")
                print(f"\n  {len(rows)} of {len(samples)} items are recorded in {out.name}.")
                print("  Wait a few minutes and re-run the IDENTICAL command; it resumes.")
                return 0
            continue

        stalled = 0
        found = hit_rate(carried, sample.evidence, embedder)

        rows.append(dict(condition=f"system_{args.select}", sample_index=sample.index,
                         prompt_tokens=tokens, prediction=prediction,
                         label=verdict.label, reason=verdict.reason,
                         notes_extracted=len(notes), notes_carried=len(carried),
                         notes_pruned=len(store.archive),
                         cue_window=cue_window, cue_extracted=cue_extracted,
                         cue_selected=cue_selected,
                         carried=carried, evidence_found=bool(found),
                         strip_trigger=bool(args.strip_trigger),
                         extract=args.extract, rerank=args.rerank,
                         all_notes=[f"[{n.mem_type.value}] {n.text}" for n in notes]))
        out.write_text(json.dumps(rows, indent=1), encoding="utf-8")

        print(f"  [{i:>3}/{len(samples)}] #{sample.index:<5} {verdict.label:<9} "
              f"notes={len(notes):>2} carried={len(carried)} "
              f"cue={'ext' if cue_extracted else '---'}/"
              f"{'sel' if cue_selected else '---'} {tokens:>5}tok "
              f"{(time.time()-started)/60:5.1f}min api={gen.calls_made + jdg.calls_made}",
              flush=True)

    # ---- results -----------------------------------------------------------
    banner("Condition E vs the pilot")
    scored = [r for r in rows if r["label"] in ("correct", "wrong")]
    if not scored:
        print("  No scored results.")
        return 1

    ours = {r["sample_index"]: r["label"] == "correct" for r in scored}
    carried_ok = [r["evidence_found"] for r in scored]
    tokens = sorted(r["prompt_tokens"] for r in scored)

    # A pilot restricted to some conditions writes a tagged filename
    # (pilot_n401_seed42_full_context.json). Fall back to it rather than
    # silently skipping the E-vs-B comparison the run exists to make.
    pilot_path = RESULTS_DIR / f"pilot_n{len(samples)}_seed{args.seed}.json"
    if args.category != "Cognitive":
        # The pilot files are Cognitive-only. Matching them by --n and --seed
        # alone printed the Cognitive baselines beside a non-Cognitive score as
        # if they were the same items -- bug 18's shape. (27 Sep)
        print(f"\n  No A-D baselines: the pilot files are Cognitive-only, and this"
              f" run is {args.category}.")
        pilot_path = RESULTS_DIR / "__no_pilot_for_this_category__.json"
    elif not pilot_path.exists():
        alternatives = sorted(RESULTS_DIR.glob(
            f"pilot_n{len(samples)}_seed{args.seed}_*.json"))
        if alternatives:
            pilot_path = alternatives[0]
            print(f"\n  baselines from {pilot_path.name}")
    print(f"\n  {'Condition':<36} {'judge score':<26} {'~tokens'}")
    print(f"  {'-'*36} {'-'*26} {'-'*8}")
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        for key, label in [("no_memory", "A. No memory"),
                           ("full_context", "B. Full context"),
                           ("oracle_cue", "C. Oracle cue"),
                           ("oracle_constraint", "D. Oracle constraint")]:
            hits = [r["label"] == "correct" for r in pilot
                    if r["condition"] == key and r["label"] in ("correct", "wrong")]
            toks = sorted(r["prompt_tokens"] for r in pilot if r["condition"] == key)
            if hits:
                print(f"  {label:<36} {str(Proportion(sum(hits), len(hits))):<26} "
                      f"{toks[len(toks)//2]:>7,}")
    print(f"  {'E. Our system (no oracle)':<36} "
          f"{str(Proportion(sum(ours.values()), len(ours))):<26} "
          f"{tokens[len(tokens)//2]:>7,}")

    banner("Where does it fail?")
    located = [r for r in scored if r.get("cue_window") is not None]
    print(f"  cue window located (measurement sanity): {len(located)}/{len(scored)}")
    if located:
        extracted = [r["cue_extracted"] for r in located]
        selected = [r["cue_selected"] for r in located]
        print(f"  a note WAS extracted from the cue window: "
              f"{Proportion(sum(extracted), len(extracted))}")
        print(f"  ...and it reached the prompt:             "
              f"{Proportion(sum(selected), len(selected))}")

        with_cue = [r for r in located if r["cue_selected"]]
        without = [r for r in located if not r["cue_selected"]]
        if with_cue and without:
            a = Proportion(sum(r["label"] == "correct" for r in with_cue), len(with_cue))
            b = Proportion(sum(r["label"] == "correct" for r in without), len(without))
            print(f"\n  answered well WITH the cue note:    {a}")
            print(f"  answered well WITHOUT it:           {b}")
            print(f"  having it is worth {100*(a.rate - b.rate):+.1f} points.")
            print("  Condition D, with exactly one correct note, scored 87.6% against")
            print("  a 37% floor. If the gap here is small, the note is not the")
            print("  problem -- the other notes around it are.")
    extracted = sorted(r["notes_extracted"] for r in scored)
    pruned = sorted(r.get("notes_pruned", 0) for r in scored)
    carried_n = sorted(r["notes_carried"] for r in scored)
    print(f"\n  notes extracted per item: median {extracted[len(scored)//2]}")
    print(f"  notes pruned by decay:    median {pruned[len(scored)//2]}"
          "   <- deleted before selection ran")
    print(f"  notes carried:            median {carried_n[len(scored)//2]}")
    chance = carried_n[len(scored)//2] / max(1, extracted[len(scored)//2])
    print(f"\n  picking at random would carry the right note {chance:.1%} of the time.")
    print(f"  we carried it {sum(carried_ok)/len(carried_ok):.1%} of the time.")

    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        for key, label in [("full_context", "B full context"),
                           ("oracle_constraint", "D oracle constraint")]:
            other = {r["sample_index"]: r["label"] == "correct" for r in pilot
                     if r["condition"] == key and r["label"] in ("correct", "wrong")}
            shared = sorted(set(ours) & set(other))
            if len(shared) >= 10:
                test = mcnemar([other[i] for i in shared], [ours[i] for i in shared])
                # McNemar.verdict() says "A better" / "B better" meaning its FIRST
                # and SECOND list. Here the first is the baseline and the second is
                # ours -- and the conditions are ALSO named A..E, so "B better"
                # reads as "condition B, full context, better" when it means the
                # exact opposite. That collision would have put a wrong sentence in
                # the paper. Name the winning condition instead. (4 Sep)
                if test.discordant == 0:
                    words = "identical on every item"
                elif test.p_value < 0.05:
                    winner = "E (ours)" if test.only_b > test.only_a else label
                    words = f"different (p={test.p_value:.4f}), {winner} better"
                else:
                    words = f"no significant difference (p={test.p_value:.4f})"
                print(f"\n  E vs {label} (paired, n={len(shared)}): {words}")
                print(f"       {label} only: {test.only_a}   E only: {test.only_b}   "
                      f"both: {test.both}   neither: {test.neither}")

    print(f"\n  generator: {gen.stats()}\n  judge:     {jdg.stats()}")
    print(f"\n  Raw results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
