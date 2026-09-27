"""
Emit every number the paper quotes as a LaTeX macro.

    python scripts/paper_numbers.py            # writes paper/numbers.tex

Free: no API calls, no embedder, no dataset. Reads only the results files.

WHY THIS EXISTS. The plan is to evaluate all 401 Cognitive items after the
top_k curve. That changes EVERY figure in the paper -- the headline, the
baselines, every confidence interval, every p-value. A paper with numbers typed
into its prose would have to be rewritten line by line, and this project has
already lost three claims to numbers that came from the wrong place (bugs 12,
17, 18).

So the prose says \\EventsScore and this file defines it. Re-run after any
experiment and every figure in the text updates at once.

Anything it cannot compute is emitted as \\todo{...}, which renders in the PDF
as a visible [TODO: ...] marker. A missing number must be impossible to miss,
not silently absent.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.stats import Proportion, mcnemar

RESULTS = Path("results")
OUT = Path("paper/numbers.tex")
SCORED = ("correct", "wrong")

RUNS = {
    "Permissive": "system_n100_seed42_llm3_d1_PERMISSIVE.json",
    "Forced":     "system_n100_seed42_llm3_d1.json",
    "Hybrid":     "system_n100_seed42_hybrid3_d1.json",
    "NoTrig":     "system_n100_seed42_llm3_d1_notrig.json",
    "Events":     "system_n100_seed42_llm3_d1_notrig_events.json",
    "Person":     "system_n100_seed42_llm3_d1_notrig_events_person.json",
    "TopKOne":    "system_n100_seed42_llm1_d1_notrig_events.json",
    "TopKFive":   "system_n100_seed42_llm5_d1_notrig_events.json",
    # The full Cognitive split, for the robustness paragraph. Absent until run.
    "Full":       "system_n401_seed42_llm3_d1_notrig_events.json",
}
PILOT = "pilot_n100_seed42.json"
# Condition B re-run over all 401 items. Only full_context: the headline is
# E vs B, and the oracle conditions are context rather than the claim.
PILOT_FULL = "pilot_n401_seed42_full_context.json"
PILOT_CONDITIONS = {"NoMemory": "no_memory", "FullContext": "full_context",
                    "OracleCue": "oracle_cue", "OracleConstraint": "oracle_constraint"}

lines: list = []
missing: list = []


def macro(name, value):
    lines.append("\\newcommand{\\%s}{%s}" % (name, value))


TEX_SPECIALS = {"_": "\\_", "#": "\\#", "%": "\\%", "&": "\\&",
                "$": "\\$", "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}",
                "^": "\\textasciicircum{}"}


def tex_escape(text):
    """A `\todo` reason names a results file, and those names are full of
    underscores. An unescaped `_` in text mode is a hard LaTeX error, so the
    paper would not compile at all -- found on 13 Sep when the document was
    first built. Escape before emitting, never after."""
    return "".join(TEX_SPECIALS.get(ch, ch) for ch in str(text))


def todo(name, why):
    lines.append("\\newcommand{\\%s}{\\todo{%s}}" % (name, tex_escape(why)))
    missing.append(name)


def pct(x):
    return "%.1f" % (100 * x)


def tex_pvalue(p):
    """A p-value the way a paper prints one, not the way Python does.

    "%.4g" turned 0.00006428 into the string `6.428e-05`, which LaTeX sets as
    literal text -- "6.428e-05" appearing mid-sentence in the PDF. Found 18 Sep.

    Plain decimals only, deliberately. A macro that returned "< 0.0001" would
    render as "p=< 0.0001" wherever the prose writes $p{=}\\FullP{}$, and one
    that returned math-mode markup would break in any caption that is not
    already in math mode. Digits are the only form that is safe everywhere.

    Four decimals is the paper convention, but a fixed four turned 0.0000643
    into "0.0001" and a fixed five into "0.00006", both of which throw away the
    figure a reader wants. So: four decimals, extended only as far as it takes
    to show two significant digits.
    """
    if p <= 0:
        return "0.0000"
    places = max(4, 1 - int(math.floor(math.log10(p))))
    return ("%." + str(places) + "f") % p


def load(filename):
    path = RESULTS / filename
    if not path.exists():
        return {}
    return {r["sample_index"]: r for r in json.loads(path.read_text(encoding="utf-8"))
            if r.get("label") in SCORED}


def median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return float(ordered[mid]) if len(ordered) % 2 else (ordered[mid-1] + ordered[mid]) / 2


def emit_proportion(prefix, p):
    low, high = p.wilson()
    macro(prefix + "Score", pct(p.rate))
    macro(prefix + "Lo", pct(low))
    macro(prefix + "Hi", pct(high))
    macro(prefix + "N", p.total)
    macro(prefix + "K", p.successes)
    macro(prefix + "CI", "%s\\,[%s, %s]" % (pct(p.rate), pct(low), pct(high)))


SUFFIXES = ("Score", "Lo", "Hi", "N", "K", "CI", "Tokens")
RUN_SUFFIXES = SUFFIXES + ("Notes", "Recall", "RecallContent", "POkCue", "POkNoCue")


def main():
    print("Reading:")
    lines.append("% Generated by scripts/paper_numbers.py -- do not edit by hand.")
    lines.append("% Re-run after any experiment; every figure in the text follows.")
    lines.append("\\providecommand{\\todo}[1]{\\textbf{[TODO: #1]}}")

    pilot_path = RESULTS / PILOT
    pilot_rows = {}
    if pilot_path.exists():
        print("  %s" % pilot_path)
        raw = json.loads(pilot_path.read_text(encoding="utf-8"))
        for key in PILOT_CONDITIONS.values():
            pilot_rows[key] = {r["sample_index"]: r for r in raw
                               if r.get("condition") == key and r.get("label") in SCORED}
    for name, key in PILOT_CONDITIONS.items():
        rows = pilot_rows.get(key, {})
        if rows:
            emit_proportion(name, Proportion(
                sum(r["label"] == "correct" for r in rows.values()), len(rows)))
            macro(name + "Tokens", "{:,}".format(
                int(median([r.get("prompt_tokens", 0) for r in rows.values()]))))
        else:
            for suffix in SUFFIXES:
                todo(name + suffix, "%s missing from %s" % (key, PILOT))

    loaded = {}
    for name, filename in RUNS.items():
        rows = load(filename)
        if not rows:
            for suffix in RUN_SUFFIXES:
                todo(name + suffix, "%s not found" % filename)
            continue
        print("  %s" % (RESULTS / filename))
        loaded[name] = rows
        emit_proportion(name, Proportion(
            sum(r["label"] == "correct" for r in rows.values()), len(rows)))
        macro(name + "Tokens", int(median([r.get("prompt_tokens", 0) for r in rows.values()])))
        macro(name + "Notes", int(median([r.get("notes_extracted", 0) for r in rows.values()])))
        macro(name + "Recall",
              pct(sum(1 for r in rows.values() if r.get("cue_selected")) / len(rows)))
        with_cue = [r for r in rows.values() if r.get("evidence_found")]
        without = [r for r in rows.values() if not r.get("evidence_found")]
        macro(name + "RecallContent", pct(len(with_cue) / len(rows)))
        macro(name + "POkCue",
              pct(sum(r["label"] == "correct" for r in with_cue) / len(with_cue)) if with_cue else "0.0")
        macro(name + "POkNoCue",
              pct(sum(r["label"] == "correct" for r in without) / len(without)) if without else "0.0")

    # The size of the one intervention that worked: the writing stage.
    if "Events" in loaded and "NoTrig" in loaded:
        e = sum(r["label"] == "correct" for r in loaded["Events"].values()) / len(loaded["Events"])
        n = sum(r["label"] == "correct" for r in loaded["NoTrig"].values()) / len(loaded["NoTrig"])
        macro("DeltaExtraction", "%.1f" % (100 * (e - n)))
    else:
        todo("DeltaExtraction", "needs the notrig and events runs")

    # The paired distractor test, on the items where the cue did NOT arrive.
    # This claim has been made and withdrawn twice on unpaired comparisons
    # (once as -12 points, once as -26); only the paired form is admissible,
    # and it is computed here rather than quoted from a script's printout.
    floor = pilot_rows.get("no_memory", {})
    if "Events" in loaded and floor:
        ours = loaded["Events"]
        failed = sorted(i for i in set(ours) & set(floor) if not ours[i].get("evidence_found"))
        if len(failed) >= 10:
            a = [floor[i]["label"] == "correct" for i in failed]
            e = [ours[i]["label"] == "correct" for i in failed]
            t = mcnemar(a, e)
            macro("DistractorN", len(failed))
            macro("DistractorFloor", pct(sum(a) / len(a)))
            macro("DistractorOurs", pct(sum(e) / len(e)))
            macro("DistractorDelta", "%.1f" % (100 * (sum(e) - sum(a)) / len(failed)))
            macro("DistractorP", "%.4f" % t.p_value)
            macro("DistractorAOnly", t.only_a)
            macro("DistractorEOnly", t.only_b)
        else:
            for name in ("DistractorN", "DistractorFloor", "DistractorOurs",
                         "DistractorDelta", "DistractorP", "DistractorAOnly",
                         "DistractorEOnly"):
                todo(name, "too few undelivered items to test")
    else:
        for name in ("DistractorN", "DistractorFloor", "DistractorOurs",
                     "DistractorDelta", "DistractorP", "DistractorAOnly",
                     "DistractorEOnly"):
            todo(name, "needs the events run and the pilot")

    big_e = load(RUNS["Full"])
    big_b = {}
    if (RESULTS / PILOT_FULL).exists():
        print("  %s" % (RESULTS / PILOT_FULL))
        big_b = {r["sample_index"]: r for r in
                 json.loads((RESULTS / PILOT_FULL).read_text(encoding="utf-8"))
                 if r.get("condition") == "full_context" and r.get("label") in SCORED}
    if big_e and big_b:
        ids = sorted(set(big_e) & set(big_b))
        t401 = mcnemar([big_b[i]["label"] == "correct" for i in ids],
                       [big_e[i]["label"] == "correct" for i in ids])
        # NOTE THE PREFIX. These are "FullPaired*", not "Full*".
        #
        # `Full` is already a key in RUNS, so the per-run loop emits \FullN for
        # the number of scored rows in E's own file. Emitting \FullN a second
        # time here made numbers.tex define the same command twice, which is a
        # hard LaTeX error ("Command \FullN already defined") -- the document
        # would not compile at all. Found 18 Sep, before the first build of the
        # 401 numbers. The two counts are also genuinely different things:
        # \FullN is how many items E scored, \FullPairedN is how many items
        # BOTH conditions scored, and only the second belongs in a paired test.
        macro("FullPairedN", len(ids))
        macro("FullEScore", pct(sum(big_e[i]["label"] == "correct" for i in ids) / len(ids)))
        macro("FullBScore", pct(sum(big_b[i]["label"] == "correct" for i in ids) / len(ids)))
        macro("FullDelta", "%.1f" % (
            100 * (sum(big_e[i]["label"] == "correct" for i in ids)
                   - sum(big_b[i]["label"] == "correct" for i in ids)) / len(ids)))
        macro("FullP", tex_pvalue(t401.p_value))
        macro("FullBOnly", t401.only_a)
        macro("FullEOnly", t401.only_b)
        # Condition B's cost on the FULL split. The n=100 pilot's
        # \FullContextTokens is a different median over a different set, and
        # quoting it beside a 401-item score would misstate the compression by
        # about six per cent.
        b_tokens = int(median([big_b[i].get("prompt_tokens", 0) for i in ids]))
        e_tokens = int(median([big_e[i].get("prompt_tokens", 0) for i in ids]))
        macro("FullBTokens", "{:,}".format(b_tokens))
        macro("FullETokens", "{:,}".format(e_tokens))
        macro("FullCompression", "%.0f" % (b_tokens / e_tokens) if e_tokens else "0")
    else:
        for name in ("FullPairedN", "FullEScore", "FullBScore", "FullDelta",
                     "FullP", "FullBOnly", "FullEOnly", "FullBTokens",
                     "FullETokens", "FullCompression"):
            todo(name, "the 401-item run has not been done")

    head, full = loaded.get("Events", {}), pilot_rows.get("full_context", {})
    if head and full:
        shared = sorted(set(head) & set(full))
        t = mcnemar([full[i]["label"] == "correct" for i in shared],
                    [head[i]["label"] == "correct" for i in shared])
        macro("HeadlineN", len(shared))
        macro("HeadlineP", "%.4f" % t.p_value)
        macro("HeadlineFullOnly", t.only_a)
        macro("HeadlineOursOnly", t.only_b)
        e_tok = median([r.get("prompt_tokens", 0) for r in head.values()])
        b_tok = median([r.get("prompt_tokens", 0) for r in full.values()])
        macro("CompressionRatio", "%.0f" % (b_tok / max(1.0, e_tok)))
        e_rate = sum(r["label"] == "correct" for r in head.values()) / len(head)
        b_rate = sum(r["label"] == "correct" for r in full.values()) / len(full)
        macro("DeltaEB", "%.1f" % (100 * (e_rate - b_rate)))
    else:
        for name in ("HeadlineN", "HeadlineP", "HeadlineFullOnly",
                     "HeadlineOursOnly", "CompressionRatio", "DeltaEB"):
            todo(name, "needs the events run and the pilot")

    # Stage decomposition, exported by extraction_ceiling.py --export (it needs
    # the embedder, so it runs in Colab and leaves its figures behind).
    #
    # Prefer the FULL run's ceiling; fall back to the 100-item one. Each export
    # is named after the run it describes, so the file that exists decides the
    # n -- and \CeilingItems carries that n into the caption, so the table can
    # never silently claim an n it was not computed on. Bug 18: an analysis
    # input is never chosen without the code saying out loud which file it used.
    ceiling_candidates = [RESULTS / ("ceiling_" + RUNS["Full"]),
                          RESULTS / ("ceiling_" + RUNS["Events"])]
    ceiling = next((p for p in ceiling_candidates if p.exists()),
                   ceiling_candidates[-1])
    if ceiling.exists():
        print("  %s   <-- stage decomposition comes from THIS file" % ceiling)
        for other in ceiling_candidates:
            if other != ceiling and other.exists():
                print("      (ignoring %s)" % other.name)
        c = json.loads(ceiling.read_text(encoding="utf-8"))
        macro("CeilingItems", c["items"])
        macro("CeilingWritten", pct(c["written"] / c["items"]))
        macro("CeilingChosen", pct(c["chosen"] / c["items"]))
        macro("CeilingChosenGivenWritten", pct(c["chosen_given_written"]))
        macro("CeilingNeverWritten", c["never_written"])
        macro("CeilingWrittenChosenN", c["n_written_and_chosen"])
        macro("CeilingWrittenNotChosenN", c["n_written_not_chosen"])
        macro("CeilingWrittenChosenScore", pct(c["score_written_and_chosen"]))
        macro("CeilingWrittenNotChosenScore", pct(c["score_written_not_chosen"]))
        macro("CeilingNeverWrittenScore", pct(c["score_never_written"]))
        macro("BudgetExtraction", "%.0f" % c["budget_extraction_points"])
        macro("BudgetReranker", "%.0f" % c["budget_reranker_points"])
    else:
        print("  no ceiling export found; looked for:")
        for p in ceiling_candidates:
            print("      %s" % p.name)
        for name in ("CeilingItems", "CeilingWritten", "CeilingChosen",
                     "CeilingChosenGivenWritten",
                     "CeilingNeverWritten", "CeilingWrittenChosenN",
                     "CeilingWrittenNotChosenN", "CeilingWrittenChosenScore",
                     "CeilingWrittenNotChosenScore", "CeilingNeverWrittenScore",
                     "BudgetExtraction", "BudgetReranker"):
            todo(name, "run extraction_ceiling.py --export in Colab")

    # ---- what the compression figure does NOT count -------------------------
    #
    # Reviewer objection we agree with: 178x compares the ANSWERING prompt only.
    # The system also spends one extraction call per window at write time. That
    # per-call cost is a measured constant living in scripts/run_system.py
    # (it is what the quota planner budgets with), so it is read out of that
    # file rather than retyped here -- bug 18's rule applied to a constant.
    runner = Path("scripts/run_system.py")
    per_extraction = None
    if runner.exists():
        m = re.search(r"TOKENS_PER_EXTRACTION\s*=\s*([\d_]+)",
                      runner.read_text(encoding="utf-8"))
        if m:
            per_extraction = int(m.group(1).replace("_", ""))
    if per_extraction and "Full" in loaded:
        windows = int(median([r.get("notes_extracted", 0)
                              for r in loaded["Full"].values()]))
        macro("TokensPerExtraction", "{:,}".format(per_extraction))
        macro("WindowsPerItem", windows)
        macro("ExtractionTokensPerItem",
              "{:,}".format(windows * per_extraction))
    else:
        for name in ("TokensPerExtraction", "WindowsPerItem",
                     "ExtractionTokensPerItem"):
            todo(name, "needs scripts/run_system.py and the 401-item run")

    # ---- configuration, read from the code the experiments actually ran ----
    #
    # A Method section that disagrees with the code is a retraction waiting to
    # happen, so these come from SystemConfig rather than from memory.
    try:
        from bapca.pipeline import SystemConfig
        cfg = SystemConfig()
        macro("WindowTurns", cfg.window_turns)
        macro("DaysPerWindow", ("%g" % cfg.days_per_window))
        macro("PruneThreshold", ("%g" % cfg.prune_threshold))
        macro("CarrySetting", cfg.carry)
    except Exception as exc:                                   # noqa: BLE001
        for name in ("WindowTurns", "DaysPerWindow", "PruneThreshold", "CarrySetting"):
            todo(name, "could not import SystemConfig: %s" % exc.__class__.__name__)
    # Size of the Cognitive split. Not derivable without the dataset, which is
    # not on every machine that builds the paper; documented here so it has one
    # home rather than being typed into the prose.
    macro("CognitiveTotal", 401)
    macro("BenchmarkTotal", "2,387")

    # ---- human validation -------------------------------------------------
    #
    # The paired human comparison of condition E against full context, on the
    # SAME sample indices. Computed here rather than quoted from a script's
    # printout, and paired because this project has retracted three claims
    # built on unpaired comparisons.
    ANN = RESULTS / "annotations"
    people = {}
    if ANN.exists():
        for path in sorted(ANN.glob("*.json")):
            if path.stem == "joint":      # the joint session, not one rater
                continue
            people.setdefault(path.stem.split("_")[0], {}).update(
                {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items()
                 if v in SCORED})
    raters = sorted(people)
    if len(raters) == 2:
        print("  %s (%d annotation files)" % (ANN, len(list(ANN.glob("*.json")))))
        def idx_for(who):
            t = people[who]
            return ({int(k.split(":")[1]) for k in t if k.startswith("system_llm:")}
                    & {int(k.split(":")[1]) for k in t if k.startswith("full_context:")})
        shared = sorted(idx_for(raters[0]) & idx_for(raters[1]))
        agree = [i for i in shared
                 if people[raters[0]][f"system_llm:{i}"] == people[raters[1]][f"system_llm:{i}"]
                 and people[raters[0]][f"full_context:{i}"] == people[raters[1]][f"full_context:{i}"]]
        judge_all = {}
        for path in (RESULTS / PILOT, RESULTS / RUNS["Events"]):
            if path.exists():
                for r in json.loads(path.read_text(encoding="utf-8")):
                    if r.get("label") in SCORED:
                        judge_all[f"{r['condition']}:{r['sample_index']}"] = r["label"]

        def emit_pair(prefix, ids, table):
            e = [table[f"system_llm:{i}"] == "correct" for i in ids]
            b = [table[f"full_context:{i}"] == "correct" for i in ids]
            t = mcnemar(b, e)
            macro(prefix + "N", len(ids))
            macro(prefix + "E", pct(sum(e) / len(e)))
            macro(prefix + "B", pct(sum(b) / len(b)))
            macro(prefix + "Delta", "%.1f" % (100 * (sum(e) - sum(b)) / len(ids)))
            macro(prefix + "P", "%.4f" % t.p_value)
            macro(prefix + "BOnly", t.only_a)
            macro(prefix + "EOnly", t.only_b)

        if len(agree) >= 10:
            emit_pair("HumanPaired", agree, people[raters[0]])
            emit_pair("JudgePaired", agree, judge_all)
            # Never the raters' names: TMLR is double-blind and this macro
            # was being pasted into the submission source. (27 Sep)
            macro("HumanRaters", "two blind annotators")
            macro("HumanKappaN", len(shared))
        else:
            for name in ("HumanPairedN", "HumanPairedE", "HumanPairedB",
                         "HumanPairedDelta", "HumanPairedP", "HumanPairedBOnly",
                         "HumanPairedEOnly", "JudgePairedN", "JudgePairedE",
                         "JudgePairedB", "JudgePairedDelta", "JudgePairedP",
                         "JudgePairedBOnly", "JudgePairedEOnly", "HumanRaters",
                         "HumanKappaN"):
                todo(name, "too few matched annotated pairs")

    agreement = RESULTS / "judge_agreement.json"
    if agreement.exists():
        print("  %s" % agreement)
        a = json.loads(agreement.read_text(encoding="utf-8"))
        macro("KappaHumanHuman", "%.3f" % a.get("human_kappa", 0))
        macro("KappaJudgeLo", "%.3f" % a.get("judge_kappa_lo", 0))
        macro("KappaJudgeHi", "%.3f" % a.get("judge_kappa_hi", 0))
        macro("KappaN", a.get("human_n", 0))
        for cond, label in (("system_llm", "E"), ("full_context", "B"),
                            ("no_memory", "A"), ("oracle_cue", "C"),
                            ("oracle_constraint", "D")):
            rows = a.get("per_condition", {}).get(cond, {})
            if rows:
                hs = [v["human"] for v in rows.values()]
                macro("Human%sLo" % label, pct(min(hs)))
                macro("Human%sHi" % label, pct(max(hs)))
    else:
        for name in ("KappaHumanHuman", "KappaJudgeLo", "KappaJudgeHi", "KappaN"):
            todo(name, "run scripts/judge\\_agreement.py --export")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote %s  (%d macros)" % (OUT, len(lines)))
    if missing:
        print("\n  %d macro(s) still \\todo -- they render as [TODO: ...] in the PDF:" % len(missing))
        for name in missing[:10]:
            print("    \\%s" % name)
        if len(missing) > 10:
            print("    ... and %d more" % (len(missing) - 10))
    return 0


if __name__ == "__main__":
    sys.exit(main())
