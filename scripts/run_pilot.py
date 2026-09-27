"""
The pilot: is the bottleneck FINDING the constraint, or APPLYING it?

    python scripts/run_pilot.py --dry-run     # cost estimate, spends nothing
    python scripts/run_pilot.py --n 40        # the real thing

Four conditions on identical items. A no memory, B full context (the
benchmark's own prompt), C the cue handed over verbatim, D the cue restated as
an explicit standing constraint. C and D are oracles -- they get the right cue
for free. They measure the ceiling our system would be building toward, before
we spend three weeks building toward it.

Everything is cached, so a re-run costs nothing and a crash loses nothing.
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
from bapca.evaluation import (ASSISTANT_SYSTEM, CONDITIONS, CONDITIONS_BY_KEY,
                              extract_constraint,
                              judge_response)
from bapca.llm import DailyQuotaExhausted, PromptTooLarge, estimate_tokens, generator, judge
from bapca.stats import McNemar, Proportion, mcnemar, min_detectable_gap

RESULTS_DIR = Path("results")


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def dry_run(samples, gen_limits, judge_limits) -> None:
    banner("Dry run -- what this would cost")

    gen_calls = len(samples) * len(CONDITIONS)
    judge_calls = gen_calls
    # Only oracle_constraint extracts. Counting one extraction per item
    # regardless of --conditions over-priced a full_context-only run by 401
    # calls and ~124,000 tokens. (15 Sep)
    extract_calls = len(samples) if any(c.needs_extraction for c in CONDITIONS) else 0

    # Price each selected condition's own prompt, not all four.
    gen_tokens = 0
    for sample in samples:
        for condition in CONDITIONS:
            if condition.key == "full_context":
                gen_tokens += estimate_tokens(sample.input_prompt)
            elif condition.key == "oracle_cue":
                gen_tokens += estimate_tokens(sample.trigger + sample.evidence)
            else:                                   # no_memory, oracle_constraint
                gen_tokens += estimate_tokens(sample.trigger) + 60

    print(f"\n  items: {len(samples)}   conditions: {len(CONDITIONS)}")
    print(f"\n  generator (Gemini)  {gen_calls:>5} calls, ~{gen_tokens:>9,} tokens "
          f"| RPD {gen_limits.rpd}, TPM {gen_limits.tpm:,}")
    print(f"  judge     (Groq)    {judge_calls + extract_calls:>5} calls "
          f"({judge_calls} judging + {extract_calls} extraction) "
          f"| RPD {judge_limits.rpd}")

    biggest = max(s.approx_tokens() for s in samples)
    print(f"\n  largest single prompt: ~{biggest:,} tokens "
          f"({'fits' if biggest < gen_limits.tpm else 'DOES NOT FIT'} in "
          f"{gen_limits.tpm:,} TPM)")

    # The judge is TOKEN-bound, not request-bound: Groq allows 8,000 tokens per
    # minute, and each verdict costs the evidence, the trigger, the prediction,
    # the rubric and the reserved output. Estimating it on RPM alone said 17
    # minutes when the truth was nearer 40.
    judge_tokens = 0
    for sample in samples:
        judged = (estimate_tokens(sample.evidence + sample.trigger)
                  + 300      # the prediction being judged
                  + 120      # rubric boilerplate
                  + 256)     # reserved output
        judge_tokens += judged * len(CONDITIONS)
        if extract_calls:
            judge_tokens += estimate_tokens(sample.evidence) + 100 + 160

    gen_minutes = max(gen_calls / gen_limits.rpm, gen_tokens / gen_limits.tpm)
    judge_minutes = max((judge_calls + extract_calls) / judge_limits.rpm,
                        judge_tokens / judge_limits.tpm)
    print(f"\n  generator pacing: ~{gen_minutes:4.0f} min "
          f"({'request' if gen_calls / gen_limits.rpm > gen_tokens / gen_limits.tpm else 'token'}-bound)")
    print(f"  judge pacing:     ~{judge_minutes:4.0f} min "
          f"(~{judge_tokens:,} tokens, "
          f"{'request' if (judge_calls + extract_calls) / judge_limits.rpm > judge_tokens / judge_limits.tpm else 'token'}-bound)")
    print(f"  calls run in sequence, so expect roughly "
          f"{gen_minutes + judge_minutes:.0f} min end to end.")

    over = []
    if gen_calls > gen_limits.rpd:
        over.append(f"generator {gen_calls} > RPD {gen_limits.rpd}")
    if judge_calls + extract_calls > judge_limits.rpd:
        over.append(f"judge {judge_calls + extract_calls} > RPD {judge_limits.rpd}")
    if over:
        print("\n  OVER DAILY REQUEST LIMIT: " + "; ".join(over))

    # Bug 16, which this script never got. Groq caps TOKENS per day, and that
    # is what stops a run -- requests never were the constraint. Reporting
    # "within daily limits" on a request count alone said 86 minutes for a job
    # that needs two days.
    if judge_limits.tpd:
        days = math.ceil(judge_tokens / judge_limits.tpd)
        print(f"\n  judge TOKENS PER DAY: {judge_limits.tpd:,}. "
              f"This needs ~{judge_tokens:,}.")
        if days > 1:
            print(f"  ==> {days} DAYS, not one sitting. It stops partway each day;")
            print("      re-run the identical command after the reset and it resumes.")
        else:
            print("  ==> fits in one day.")
    if gen_limits.tpd is None and gen_tokens > 1_000_000:
        print(f"\n  NOTE: the generator needs ~{gen_tokens:,} tokens and its daily")
        print("  token cap is not published. If it stops, resume the same way.")
    print(f"\n  Detectable gap at n={len(samples)}: differences smaller than "
          f"~{100 * min_detectable_gap(len(samples)):.0f} points are noise.")


def run(samples, gen, jdg, out_path: Path) -> list[dict]:
    # Resume. A run this size stops on the daily token budget, so starting from
    # an empty list and overwriting the file would discard a day's work -- the
    # cached calls make it cheap to redo but not free in wall clock, and a
    # crash mid-write would lose it outright.
    results: list[dict] = []
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"\n  Resuming: {len(results)} rows already in {out_path.name}")
    finished = {(r["condition"], r["sample_index"]) for r in results}

    constraints: dict[int, str] = {}
    started = time.time()
    total = len(samples) * len(CONDITIONS)
    done = 0

    for sample in samples:
        for condition in CONDITIONS:
            done += 1
            if (condition.key, sample.index) in finished:
                continue
            constraint = None
            if condition.needs_extraction:
                if sample.index not in constraints:
                    constraints[sample.index] = extract_constraint(jdg, sample)
                constraint = constraints[sample.index]
                if not constraint:
                    # The extractor found nothing lasting in the cue. Recording
                    # this honestly matters: it is a failure mode of our own
                    # method, not a sample to quietly drop.
                    results.append(dict(
                        condition=condition.key, sample_index=sample.index,
                        prompt_tokens=0, prediction="", label="no_constraint",
                        reason="extractor returned NONE", constraint=""))
                    continue

            prompt = condition.build(sample, constraint)
            try:
                prediction = gen.complete(prompt, system=ASSISTANT_SYSTEM,
                                          max_tokens=300)
            except PromptTooLarge as exc:
                print(f"  [{done}/{total}] {condition.key} #{sample.index}: {exc}")
                continue
            except DailyQuotaExhausted as exc:
                banner("Out of quota for today -- this is a pause, not a failure")
                print(f"  {exc}\n")
                print(f"  {len(results)} of {total} rows are recorded in "
                      f"{out_path.name}.")
                print("  Re-run the IDENTICAL command after the daily reset; "
                      "it resumes.")
                return results

            try:
                verdict = judge_response(jdg, sample, prediction)
            except DailyQuotaExhausted as exc:
                banner("Out of quota for today -- this is a pause, not a failure")
                print(f"  {exc}\n")
                print(f"  {len(results)} of {total} rows are recorded in "
                      f"{out_path.name}.")
                print("  Re-run the IDENTICAL command after the daily reset; "
                      "it resumes.")
                return results
            results.append(dict(
                condition=condition.key, sample_index=sample.index,
                prompt_tokens=estimate_tokens(prompt), prediction=prediction,
                label=verdict.label, reason=verdict.reason, constraint=constraint))

            elapsed = time.time() - started
            print(f"  [{done:>3}/{total}] {condition.key:<18} #{sample.index:<5} "
                  f"{verdict.label:<9} {elapsed / 60:5.1f} min  "
                  f"api={gen.calls_made + jdg.calls_made}", flush=True)

            out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")

    return results


def report(results: list[dict], samples) -> None:
    banner("Results")

    by_condition = {c.key: {} for c in CONDITIONS}
    unparsed = {c.key: 0 for c in CONDITIONS}
    no_constraint = 0
    tokens = {c.key: [] for c in CONDITIONS}

    for row in results:
        key = row["condition"]
        if row["label"] == "no_constraint":
            no_constraint += 1
            by_condition[key][row["sample_index"]] = False
            continue
        if row["label"] == "unparsed":
            unparsed[key] += 1
            continue
        by_condition[key][row["sample_index"]] = row["label"] == "correct"
        tokens[key].append(row["prompt_tokens"])

    # Only items every condition answered, so the comparison stays paired.
    common = set.intersection(*(set(v) for v in by_condition.values() if v)) \
        if all(by_condition.values()) else set()
    common = sorted(common)
    print(f"\n  {len(common)} items answered under every condition "
          f"(of {len(samples)} attempted)")
    if no_constraint:
        print(f"  {no_constraint} items where the extractor found no standing "
              f"constraint (scored as failures, not dropped)")
    if any(unparsed.values()):
        print(f"  unparsed judge verdicts: {dict(unparsed)}")
    if not common:
        print("\n  Not enough overlapping results to compare. Re-run.")
        return

    print(f"\n  {'Condition':<34} {'Constraint consistency':<28} {'~tokens/query'}")
    print(f"  {'-' * 34} {'-' * 28} {'-' * 13}")
    outcomes = {}
    for condition in CONDITIONS:
        hits = [by_condition[condition.key][i] for i in common]
        outcomes[condition.key] = hits
        median_tokens = sorted(tokens[condition.key])[len(tokens[condition.key]) // 2] \
            if tokens[condition.key] else 0
        print(f"  {condition.label:<34} {str(Proportion(sum(hits), len(hits))):<28} "
              f"{median_tokens:>10,}")

    banner("What this means")
    gap = 100 * min_detectable_gap(len(common))
    print(f"  At n={len(common)}, ignore any gap smaller than ~{gap:.0f} points.\n")

    comparisons = [
        ("full_context", "oracle_cue",
         "Does handing over the cue beat having it buried in context?",
         "Retrieval IS the bottleneck -- better retrieval would pay off.",
         "Retrieval is NOT the bottleneck. The cue being present is not enough."),
        ("oracle_cue", "oracle_constraint",
         "Does restating the cue as an explicit constraint beat the raw cue?",
         "Salience is what matters. Extraction is our contribution.",
         "Restating it adds nothing measurable at this sample size."),
        ("no_memory", "full_context",
         "Sanity check: does the full conversation beat no memory at all?",
         "The benchmark behaves as published.",
         "Something is wrong with the setup -- investigate before trusting anything."),
    ]

    # --conditions can leave a comparison with no data on one side. Skipping
    # it is correct; raising KeyError after 27 minutes of completed work, as
    # this did on the first n=401 run, is not. The rows were already saved --
    # only the summary died -- but a crash at the end of a long run reads like
    # lost work, which is its own cost. (15 Sep)
    for a_key, b_key, question, if_sig, if_not in comparisons:
        if a_key not in outcomes or b_key not in outcomes:
            missing = [k for k in (a_key, b_key) if k not in outcomes]
            print(f"  {question}")
            print(f"    skipped: {', '.join(missing)} was not run\n")
            continue
        test: McNemar = mcnemar(outcomes[a_key], outcomes[b_key])
        print(f"  {question}")
        print(f"    {a_key} vs {b_key}: {test.verdict()}")
        print(f"    both {test.both}, only-{a_key} {test.only_a}, "
              f"only-{b_key} {test.only_b}, neither {test.neither}")
        print(f"    -> {if_sig if test.p_value < 0.05 else if_not}\n")

    if "full_context" in outcomes and "oracle_constraint" in outcomes:
        full = sum(outcomes["full_context"]) / len(common)
        constraint = sum(outcomes["oracle_constraint"]) / len(common)
        med_full = sorted(tokens["full_context"])[len(tokens["full_context"]) // 2] \
            if tokens["full_context"] else 0
        med_con = sorted(tokens["oracle_constraint"])[len(tokens["oracle_constraint"]) // 2] \
            if tokens["oracle_constraint"] else 1
        print(f"  Efficiency: full context ~{med_full:,} tokens at {100 * full:.1f}%, "
              f"extracted constraint ~{med_con:,} tokens at {100 * constraint:.1f}% "
              f"({med_full / max(1, med_con):.0f}x fewer tokens).")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="items to evaluate")
    parser.add_argument("--conditions", nargs="+", default=None,
                        metavar="KEY",
                        help="run only these conditions, e.g. --conditions "
                             "full_context. Added 15 Sep to scale the headline "
                             "comparison to all 401 items without paying for "
                             "the two oracle conditions, which are context "
                             "rather than the claim. Without it, --n 401 costs "
                             "four times as much and over a week of quota.")
    parser.add_argument("--seed", type=int, default=42, help="subset seed (report this)")
    parser.add_argument("--dry-run", action="store_true", help="estimate cost, spend nothing")
    args = parser.parse_args()

    if args.conditions:
        unknown = [k for k in args.conditions if k not in CONDITIONS_BY_KEY]
        if unknown:
            parser.error("unknown condition(s): %s. Choose from: %s"
                         % (", ".join(unknown), ", ".join(CONDITIONS_BY_KEY)))
        # Mutate the shared list in place so dry_run, the loop and the summary
        # all see the same set -- they each read CONDITIONS by name.
        keep = [c for c in CONDITIONS if c.key in args.conditions]
        CONDITIONS[:] = keep
        print("Running ONLY: %s" % ", ".join(c.key for c in CONDITIONS))

    data = LocomoPlus()
    print(f"Loaded {len(data)} samples; {len(data.cognitive())} are Cognitive.")
    samples = data.subset(args.n, seed=args.seed)
    print(f"Evaluating {len(samples)} items (seed {args.seed}).")

    from bapca.llm import ROLES
    if args.dry_run:
        dry_run(samples, ROLES["generator"][2], ROLES["judge"][2])
        return 0

    gen, jdg = generator(verbose=True), judge(verbose=True)
    print(f"\n  generator: {gen.model} ({gen.provider.name})")
    print(f"  judge:     {jdg.model} ({jdg.provider.name})")

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = ""
    if args.conditions and len(args.conditions) < 4:
        tag = "_" + "+".join(sorted(args.conditions))
    out_path = RESULTS_DIR / f"pilot_n{len(samples)}_seed{args.seed}{tag}.json"

    banner(f"Running -- progress saved to {out_path} after every item")
    results = run(samples, gen, jdg, out_path)

    report(results, samples)
    print(f"\n  generator: {gen.stats()}")
    print(f"  judge:     {jdg.stats()}")
    print(f"\n  Raw results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
