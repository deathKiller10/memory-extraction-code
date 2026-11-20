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


# Table "interventions": each row against the row before it.
INTERVENTION_CHAIN = (("Permissive", "Forced"), ("Forced", "Hybrid"),
                      ("Hybrid", "NoTrig"), ("NoTrig", "Events"), ("Events", "Person"))


def matched_delta(rows, before):
    """Accuracy of `rows` minus `before`, in points, on the items both scored."""
    common = set(rows) & set(before)
    if not common:
        return 0.0, 0
    gain = sum((rows[k]["label"] == "correct") - (before[k]["label"] == "correct")
               for k in common)
    return 100.0 * gain / len(common), len(common)


def signed(points):
    """'+0.0', '-1.0', '+23.2' -- never '-0.0'."""
    text = "%+.1f" % points
    return "+0.0" if text == "-0.0" else text


DESIGNED_ORDER = ("no_memory", "full_context", "system_llm", "oracle_cue",
                  "oracle_constraint")


def judge_shape(a):
    """Judge leniency on the weakest and strongest condition, the spread of
    scores across conditions, and the one ordering exception (18 Nov: the
    draft said 'within a few points' and 'one adjacent pair tied'; the file
    says 0 and 8 points, and the two oracles swapped for one annotator)."""
    per = a.get("per_condition", {})
    names = ("JudgeOverALo", "JudgeOverAHi", "JudgeOverDLo", "JudgeOverDHi",
             "HumanSpreadLo", "HumanSpreadHi", "JudgeSpreadLo", "JudgeSpreadHi",
             "OrderSwapCue", "OrderSwapConstraint")
    if not all(c in per for c in DESIGNED_ORDER) or not a.get("spread"):
        for name in names:
            todo(name, "judge_agreement.json lacks per-condition rates")
        return
    for cond, label in (("no_memory", "A"), ("oracle_constraint", "D")):
        over = [v["judge"] - v["human"] for v in per[cond].values()]
        macro("JudgeOver%sLo" % label, pct(min(over)))
        macro("JudgeOver%sHi" % label, pct(max(over)))
    for who, key in (("Human", "human"), ("Judge", "judge")):
        vals = [v[key] for v in a["spread"].values()]
        macro(who + "SpreadLo", pct(min(vals)))
        macro(who + "SpreadHi", pct(max(vals)))
    swaps = []
    for annotator in per["no_memory"]:
        rates = [per[c][annotator]["human"] for c in DESIGNED_ORDER]
        bad = [i for i in range(len(rates) - 1) if rates[i] > rates[i + 1]]
        swaps += [(annotator, i, rates) for i in bad]
    if len(swaps) == 1 and swaps[0][1] == 3:          # oracle cue above oracle note
        rates = swaps[0][2]
        macro("OrderSwapCue", pct(rates[3]))
        macro("OrderSwapConstraint", pct(rates[4]))
    else:
        todo("OrderSwapCue", "ordering exception is not the single oracle swap")
        todo("OrderSwapConstraint", "ordering exception is not the single oracle swap")


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

    # The change each intervention made, against the row before it, on the
    # items BOTH runs scored. Not a difference of the two Score macros: the
    # NoTrig run has 99 scored items (#2374's judge reply is unparsed), and
    # 65/100 - 41/99 overstated the extraction gain (23.6) against the
    # matched 64/99 - 41/99 (23.2); likewise NoTrig vs Hybrid read -0.6 where
    # the matched change is zero (18 Nov).
    for prev, name in INTERVENTION_CHAIN:
        if prev in loaded and name in loaded:
            delta, common = matched_delta(loaded[name], loaded[prev])
            macro(name + "Delta", signed(delta))
            macro(name + "DeltaN", common)
        else:
            todo(name + "Delta", "needs the %s and %s runs" % (prev, name))
            todo(name + "DeltaN", "needs the %s and %s runs" % (prev, name))
    # The size of the one intervention that worked: the writing stage.
    if "Events" in loaded and "NoTrig" in loaded:
        delta, _ = matched_delta(loaded["Events"], loaded["NoTrig"])
        macro("DeltaExtraction", "%.1f" % delta)
    else:
        todo("DeltaExtraction", "needs the notrig and events runs")

    # Person-aware re-ranking (18 Nov): which notes it changed, and the items
    # it won and lost. "74 of 100" in earlier drafts counted reorderings too;
    # the set of carried notes changed on fewer items.
    if "Events" in loaded and "Person" in loaded:
        ev, pe = loaded["Events"], loaded["Person"]
        common = set(ev) & set(pe)
        macro("PersonSetChangedN", sum(set(ev[k].get("carried") or []) !=
                                       set(pe[k].get("carried") or []) for k in common))
        macro("PersonGainN", sum(pe[k]["label"] == "correct" and ev[k]["label"] == "wrong"
                                 for k in common))
        macro("PersonLossN", sum(pe[k]["label"] == "wrong" and ev[k]["label"] == "correct"
                                 for k in common))
    else:
        for name in ("PersonSetChangedN", "PersonGainN", "PersonLossN"):
            todo(name, "needs the events and person runs")

    # Salience decay: items on which it pruned any note, headline run (18 Nov).
    if "Full" in loaded:
        macro("FullPrunedItems", sum(1 for r in loaded["Full"].values()
                                     if r.get("notes_pruned")))
    else:
        todo("FullPrunedItems", "needs the full run")

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

    # ---- reconciling with LoCoMo-Plus's published scale (24 Oct) -----------
    #
    # LoCoMo-Plus (Li et al. 2026) Table 1 puts every system on the Cognitive
    # split between 9.57 and 26.06 (best: gemini-2.5-pro, full context), with
    # a judge that agrees with humans at kappa 0.80-0.82 (their Sec. 6.4).
    # These four are THEIR published numbers, kept here so the prose types
    # none of them. Ours come from scripts/reconcile_judge.py --export.
    macro("LiCogMin", "9.57")
    macro("LiCogMax", "26.06")
    macro("LiKappaLo", "0.80")
    macro("LiKappaHi", "0.82")
    # Later systems scored on the same 401 Cognitive items under the benchmark's
    # protocol, but with their own answer models and judges (19 Nov re-check):
    # T-Mem (Guo et al., arXiv 2606.15405, Table 3; GPT-4o answers,
    # Gemini-2.5-Flash judge) and JustMem (Chen et al., arXiv 2609.19877,
    # Table 6; GPT-4.1-mini builds, answers and judges). THEIR published
    # numbers, kept here so the prose types none of them.
    macro("TMemCog", "74.81")
    macro("JustMemCog", "62.59")
    rec_file = Path("results/reconcile_judge.json")
    if rec_file.exists():
        rec = json.loads(rec_file.read_text(encoding="utf-8"))["conditions"]
        def hat(cond):
            return rec[cond]["human_hat_mean"]
        b_ci = [a["ci"] for a in rec["full_context"]["annotators"].values()]
        macro("HumanEqA", "%.1f" % (100 * hat("no_memory")))
        macro("HumanEqB", "%.1f" % (100 * hat("full_context")))
        macro("HumanEqE", "%.1f" % (100 * hat("system_llm")))
        macro("HumanEqBLo", "%.0f" % (100 * min(c[0] for c in b_ci)))
        macro("HumanEqBHi", "%.0f" % (100 * max(c[1] for c in b_ci)))
        macro("HumanEqMargin", "%.1f" % (100 * (hat("system_llm") - hat("full_context"))))
        macro("HumanEqBN", rec["full_context"]["n"])
    else:
        for name in ("HumanEqA", "HumanEqB", "HumanEqE", "HumanEqBLo",
                     "HumanEqBHi", "HumanEqMargin", "HumanEqBN"):
            todo(name, "run scripts/reconcile_judge.py --export")

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
    # Bug 25 (24 Oct): this used the median of `notes_extracted` -- notes that
    # survived after NONE replies were dropped -- but extraction is one call
    # per WINDOW. The window count comes from scripts/extraction_cost.py,
    # which counts them from the data exactly as run_system.py cuts them.
    cost_file = Path("results/extraction_cost.json")
    windows = None
    if cost_file.exists():
        cost = json.loads(cost_file.read_text(encoding="utf-8"))
        if cost.get("n_items") == 401 and cost.get("strip_trigger"):
            windows = int(cost["median_windows_per_item"])
    if per_extraction and windows:
        macro("TokensPerExtraction", "{:,}".format(per_extraction))
        macro("WindowsPerItem", windows)
        macro("ExtractionTokensPerItem",
              "{:,}".format(windows * per_extraction))
    else:
        for name in ("TokensPerExtraction", "WindowsPerItem",
                     "ExtractionTokensPerItem"):
            todo(name, "needs scripts/run_system.py and "
                       "results/extraction_cost.json (extraction_cost.py --export)")

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
        judge_shape(a)
    else:
        for name in ("KappaHumanHuman", "KappaJudgeLo", "KappaJudgeHi", "KappaN"):
            todo(name, "run scripts/judge\\_agreement.py --export")

    # ---- does the extraction fix transfer? (HANDOFF 16.7) -----------------
    #
    # Two exports from scripts/written_compare.py --export, made in Colab (they
    # need the embedder). Same measure on both: a note within cosine 0.5 of the
    # gold evidence. "cognitive" = the two n=100 runs behind \DeltaExtraction;
    # "transfer" = extract_only.py over every answerable non-Cognitive question.
    # Which side is the events prompt is read from the export's prompt names,
    # never from file order (bug 17). (2 Oct)
    for name, prefix in (("cognitive", "CogWritten"), ("transfer", "Transfer")):
        path = RESULTS / ("written_%s.json" % name)
        keys = ("N", "VOne", "Events", "Delta", "P", "Convs", "ConvVOne",
                "ConvEvents", "ConvTies", "SignP")
        if not path.exists():
            for k in keys:
                todo(prefix + k, "run written\\_compare.py --export %s in Colab" % name)
            continue
        e = json.loads(path.read_text(encoding="utf-8"))
        print("  %s   (%s vs %s)" % (path, e["file_a"], e["file_b"]))
        if e["prompt_b"] == ["events"] and e["prompt_a"] != ["events"]:
            ev, v1 = "b", "a"
        elif e["prompt_a"] == ["events"] and e["prompt_b"] != ["events"]:
            ev, v1 = "a", "b"
        else:
            raise SystemExit("%s: cannot tell which side is the events prompt "
                             "(prompts %s / %s)" % (path, e["prompt_a"], e["prompt_b"]))
        if e["model_a"] != e["model_b"]:
            raise SystemExit("%s: the two sides were extracted with different "
                             "models %s / %s" % (path, e["model_a"], e["model_b"]))
        n = e["n"]
        r_ev, r_v1 = e["written_" + ev] / n, e["written_" + v1] / n
        macro(prefix + "N", "{:,}".format(n))
        macro(prefix + "VOne", pct(r_v1))
        macro(prefix + "Events", pct(r_ev))
        macro(prefix + "Delta", "%.1f" % (100 * r_ev - 100 * r_v1))
        macro(prefix + "P", tex_pvalue(e["p"]))
        macro(prefix + "Convs", e["conversations"])
        macro(prefix + "ConvVOne", e["conv_won_" + v1])
        macro(prefix + "ConvEvents", e["conv_won_" + ev])
        macro(prefix + "ConvTies", e["conv_ties"])
        macro(prefix + "SignP", tex_pvalue(e.get("sign_p", 1.0)))

    # ---- the write-policy x selection grid (HANDOFF 20.3-20.19) -------------
    #
    # n=100 subset, seed 42. Rows: write policies (ours: VOne, Events; Yuan et
    # al.'s: Facts, Summary, Raw). Columns: Cosine, BM (BM25), Hybrid (their
    # Hybrid+Rerank), Ours (our LLM re-ranker over the whole store), Ceiling
    # (oracle window: every note written from the cue's window). Baselines are
    # the pilot's no-memory and full-context rows on the same 100 items.
    GPOL = {"VOne": "", "Events": "_events", "Facts": "_facts",
            "Summary": "_summary", "Raw": "_raw"}
    GSEL = {"Cosine": "similar3", "BM": "bm253", "Hybrid": "pool3",
            "Ours": "llm3", "Ceiling": "oracle_window3"}
    pilot_path = RESULTS / "pilot_n100_seed42.json"
    pilot_rows = (json.loads(pilot_path.read_text(encoding="utf-8"))
                  if pilot_path.exists() else [])
    def pilot_cond(cond):
        return {r["sample_index"]: r["label"] == "correct" for r in pilot_rows
                if r["condition"] == cond and r.get("label") in SCORED}
    g_nomem, g_full = pilot_cond("no_memory"), pilot_cond("full_context")
    g = {}
    for pn, ps in GPOL.items():
        for sn, ss in GSEL.items():
            rows = load(f"system_n100_seed42_{ss}_d1_notrig{ps}.json")
            if not rows:
                if sn == "Ours" and pn in ("Facts", "Summary", "Raw"):
                    continue          # not run: facts exceeds Groq's per-request cap
                todo(f"Grid{pn}{sn}", "grid cell missing")
                continue
            g[(pn, sn)] = {i: r["label"] == "correct" for i, r in rows.items()}
            macro(f"Grid{pn}{sn}", pct(sum(g[(pn, sn)].values()) / len(g[(pn, sn)])))
            if sn == "Cosine":
                macro(f"GridNotes{pn}", "%d" % median([r["notes_extracted"] for r in rows.values()]))
    def paired(a, b):
        ks = sorted(set(a) & set(b))
        return mcnemar([a[k] for k in ks], [b[k] for k in ks])
    have_base = bool(g_nomem) and bool(g_full)
    if have_base:
        macro("GridNoMem", pct(sum(g_nomem.values()) / len(g_nomem)))
        macro("GridFull", pct(sum(g_full.values()) / len(g_full)))
    else:
        todo("GridNoMem", "needs results/pilot_n100_seed42.json")
        todo("GridFull", "needs results/pilot_n100_seed42.json")
    yuan = [(p, s) for p in ("Facts", "Summary", "Raw") for s in ("Cosine", "BM", "Hybrid")]
    if have_base and all(c in g for c in yuan):
        rates = [sum(g[c].values()) / len(g[c]) for c in yuan]
        macro("GridYuanMin", pct(min(rates)))
        macro("GridYuanMax", pct(max(rates)))
        vs_nomem = [paired(g[c], g_nomem) for c in yuan]
        vs_full = [paired(g[c], g_full) for c in yuan]
        macro("GridYuanNoMemPMin", tex_pvalue(min(t.p_value for t in vs_nomem)))
        macro("GridYuanBeatNoMem", "%d" % sum(t.p_value < 0.05 and t.only_a > t.only_b
                                             for t in vs_nomem))
        macro("GridYuanFullPMax", tex_pvalue(max(t.p_value for t in vs_full)))
        macro("GridYuanWorseThanFull", "%d" % sum(t.p_value < 0.05 and t.only_b > t.only_a
                                                 for t in vs_full))
        write_ranges = [max(sum(g[(p, s)].values()) for p in ("Facts", "Summary", "Raw"))
                        - min(sum(g[(p, s)].values()) for p in ("Facts", "Summary", "Raw"))
                        for s in ("Cosine", "BM", "Hybrid")]
        sel_ranges = [max(sum(g[(p, s)].values()) for s in ("Cosine", "BM", "Hybrid"))
                      - min(sum(g[(p, s)].values()) for s in ("Cosine", "BM", "Hybrid"))
                      for p in ("Facts", "Summary", "Raw")]
        n_g = len(g[yuan[0]])
        macro("GridYuanWriteRangeMax", pct(max(write_ranges) / n_g))
        macro("GridYuanSelectRangeMax", pct(max(sel_ranges) / n_g))
        ceil = [sum(g[(p, "Ceiling")].values()) / len(g[(p, "Ceiling")])
                for p in ("Facts", "Summary", "Raw")]
        macro("GridYuanCeilMin", pct(min(ceil)))
        macro("GridYuanCeilMax", pct(max(ceil)))
    else:
        for name in ("GridYuanMin", "GridYuanMax", "GridYuanNoMemPMin", "GridYuanBeatNoMem",
                     "GridYuanFullPMax", "GridYuanWorseThanFull", "GridYuanWriteRangeMax",
                     "GridYuanSelectRangeMax", "GridYuanCeilMin", "GridYuanCeilMax"):
            todo(name, "Yuan cells of the grid incomplete")
    if have_base and ("Events", "Ours") in g:
        for tag, base in (("NoMem", g_nomem), ("Full", g_full)):
            t = paired(g[("Events", "Ours")], base)
            macro(f"GridOurs{tag}P", tex_pvalue(t.p_value))
            macro(f"GridOurs{tag}Win", "%d" % t.only_a)
            macro(f"GridOurs{tag}Lose", "%d" % t.only_b)
    for name, a, b in (("GridFactsEventsCeilP", ("Facts", "Ceiling"), ("Events", "Ceiling")),
                       ("GridEventsFactsHybridP", ("Events", "Hybrid"), ("Facts", "Hybrid")),
                       ("GridEventsOursCosineP", ("Events", "Ours"), ("Events", "Cosine")),
                       ("GridEventsCosineBMP", ("Events", "Cosine"), ("Events", "BM"))):
        if a in g and b in g:
            macro(name, tex_pvalue(paired(g[a], g[b]).p_value))
        else:
            todo(name, "grid cells missing")
    # Facts extraction under Groq's per-request output cap (rule v3): totals
    # over the oracle-window cell, where each window is first extracted.
    facts_rows = load("system_n100_seed42_oracle_window3_d1_notrig_facts.json")
    if facts_rows:
        rows = list(facts_rows.values())
        macro("GridFactsCapped", "%d" % sum(r.get("facts_capped_windows") or 0 for r in rows))
        macro("GridFactsSplit", "%d" % sum(r.get("facts_split_windows") or 0 for r in rows))
        macro("GridFactsKeptCut", "%d" % sum(r.get("facts_truncated_kept") or 0 for r in rows))
        macro("GridFactsBudget", "%d" % rows[0].get("write_budget", 0))
    cost100 = RESULTS / "extraction_cost_n100.json"
    if cost100.exists():
        macro("GridWindows", "%d" % json.loads(cost100.read_text())["distinct_windows"])
    else:
        todo("GridWindows", "run extraction_cost.py --n 100 --export")
    # Grid configuration, read from the code that ran it (not typed).
    from bapca import pipeline as _pl, llm as _llm
    import importlib.util as _ilu
    _rg_spec = _ilu.spec_from_file_location(
        "run_grid", Path(__file__).resolve().parent / "run_grid.py")
    _rg = _ilu.module_from_spec(_rg_spec)
    _rg_spec.loader.exec_module(_rg)
    macro("GridTopK", "%d" % _rg.TOP_K)
    macro("GridRawTurns", "%d" % _pl.RAW_CHUNK_TURNS)
    macro("GridPoolWidth", "%d" % (_rg.TOP_K * _pl.POOL_MULTIPLIER))
    macro("GridFactsCappedBudget", "%d" % _pl.FACTS_CAPPED_BUDGET)
    macro("GridGroqOutputCap", "%d" % (_pl.FACTS_CAPPED_BUDGET + _llm.OTPM_MARGIN))
    macro("GridPolicies", "%d" % len(_pl.WRITE_POLICIES))
    # Word forms for the abstract ("five write policies ... four selectors"),
    # counted from grid_table's own lists: the rows, and the selectors that
    # choose notes (Yuan et al.'s three + ours; the oracle window is a
    # measurement, named separately as the ceiling).
    _gt_spec = _ilu.spec_from_file_location(
        "grid_table", Path(__file__).resolve().parent / "grid_table.py")
    _gt = _ilu.module_from_spec(_gt_spec)
    _gt_spec.loader.exec_module(_gt)
    _words = ("zero one two three four five six seven eight nine ten").split()
    _n_sel = len(_gt.SHARED) + len([s for s in _gt.EXTRA if s != "oracle_window"])
    assert len(_gt.POLICIES) == len(_pl.WRITE_POLICIES)
    macro("GridPoliciesWord", _words[len(_gt.POLICIES)])
    macro("GridSelectorsWord", _words[_n_sel])
    from bapca.store import RetrievalConfig as _RC
    macro("StoreMergeThreshold", "%.2f" % _RC().dedup_threshold)
    # Yuan et al.'s top-k (their configs/settings.py, TOP_K = 5) -- a
    # published constant, kept here so the prose types none of it.
    macro("YuanTopK", "5")
    # The store's near-duplicate merge, switched off for summaries.
    for sn, ss in (("Ceiling", "oracle_window3"), ("Cosine", "similar3"), ("Hybrid", "pool3")):
        rows = load(f"system_n100_seed42_{ss}_d1_notrig_summary_nomerge.json")
        if rows:
            macro(f"GridSummaryNoMerge{sn}",
                  pct(sum(r["label"] == "correct" for r in rows.values()) / len(rows)))
        else:
            todo(f"GridSummaryNoMerge{sn}", "nomerge cell missing")

    # ---- a second answering model and a second judge (HANDOFF 20.31-20.35) ---
    #
    # Second answering model: Cohere command-a-03-2025, same 100 items, same
    # notes, selections and judge (run_second_model.py; files *_gencohere).
    # Its own baselines come from its own pilot file. Facts x hybrid: Groq
    # accepted 25 window requests it had refused for the paper's run, so the
    # answering prompt differs on a few items -- reported on the matched items
    # too. Second judge: GPT-OSS over the paper's own 300 answers (rejudge.py).
    sec_pilot = RESULTS / "pilot_n100_seed42_full_context+no_memory_gencohere.json"
    SEC_CELLS = {"Ours": "llm3_d1_notrig_events", "EventsHybrid": "pool3_d1_notrig_events",
                 "Raw": "pool3_d1_notrig_raw", "Facts": "pool3_d1_notrig_facts",
                 "Summary": "pool3_d1_notrig_summary"}
    sec_rows = {k: load(f"system_n100_seed42_{v}_gencohere.json") for k, v in SEC_CELLS.items()}
    if sec_pilot.exists() and all(len(r) == 100 for r in sec_rows.values()):
        prow = json.loads(sec_pilot.read_text(encoding="utf-8"))
        s_nomem = {r["sample_index"]: r["label"] == "correct" for r in prow
                   if r["condition"] == "no_memory" and r.get("label") in SCORED}
        s_full = {r["sample_index"]: r["label"] == "correct" for r in prow
                  if r["condition"] == "full_context" and r.get("label") in SCORED}
        sec = {k: {i: r["label"] == "correct" for i, r in v.items()} for k, v in sec_rows.items()}
        models = {r.get("generator") for v in sec_rows.values() for r in v.values()}
        models |= {r.get("generator") for r in prow}
        assert models == {"command-a-03-2025"}, models
        def rate(d):
            return pct(sum(d.values()) / len(d))
        macro("SecNoMem", rate(s_nomem))
        macro("SecFull", rate(s_full))
        macro("SecOurs", rate(sec["Ours"]))
        macro("SecEventsHybrid", rate(sec["EventsHybrid"]))
        yuan = ("Raw", "Facts", "Summary")
        yr = [sum(sec[c].values()) / len(sec[c]) for c in yuan]
        macro("SecYuanMin", pct(min(yr)))
        macro("SecYuanMax", pct(max(yr)))
        vs_a = [paired(sec[c], s_nomem) for c in yuan]
        vs_b = [paired(sec[c], s_full) for c in yuan]
        macro("SecYuanBeatNoMem", "%d" % sum(t.p_value < 0.05 and t.only_a > t.only_b for t in vs_a))
        macro("SecYuanWorseThanFull", "%d" % sum(t.p_value < 0.05 and t.only_b > t.only_a for t in vs_b))
        macro("SecYuanFullPMax", tex_pvalue(max(t.p_value for t in vs_b)))
        macro("SecYuanNoMemPMin", tex_pvalue(min(t.p_value for t in vs_a)))
        t = paired(sec["Ours"], s_nomem)
        macro("SecOursNoMemP", tex_pvalue(t.p_value))
        t = paired(sec["Ours"], s_full)
        macro("SecOursFullP", tex_pvalue(t.p_value))
        macro("SecOursFullWin", "%d" % t.only_a)
        macro("SecOursFullLose", "%d" % t.only_b)
        macro("SecOursHybridP", tex_pvalue(paired(sec["Ours"], sec["EventsHybrid"]).p_value))
        macro("SecHybridFullP", tex_pvalue(paired(sec["EventsHybrid"], s_full).p_value))
        macro("SecHybridNoMemP", tex_pvalue(paired(sec["EventsHybrid"], s_nomem).p_value))
        paper_facts = load("system_n100_seed42_pool3_d1_notrig_facts.json")
        same = {i for i, r in sec_rows["Facts"].items()
                if i in paper_facts and paper_facts[i]["carried"] == r["carried"]}
        macro("SecFactsPromptDiffN", "%d" % (len(sec_rows["Facts"]) - len(same)))
        macro("SecFactsMatchedN", "%d" % len(same))
        fm = {i: v for i, v in sec["Facts"].items() if i in same}
        macro("SecFactsMatchedFullP", tex_pvalue(paired(fm, s_full).p_value))
        gh = g.get(("Events", "Hybrid"))
        if gh and have_base:
            macro("GridEventsHybridFullP", tex_pvalue(paired(gh, g_full).p_value))
    else:
        for name in ("SecNoMem", "SecFull", "SecOurs", "SecEventsHybrid", "SecYuanMin",
                     "SecYuanMax", "SecYuanBeatNoMem", "SecYuanWorseThanFull",
                     "SecYuanFullPMax", "SecYuanNoMemPMin", "SecOursNoMemP", "SecOursFullP",
                     "SecOursFullWin", "SecOursFullLose", "SecOursHybridP", "SecHybridFullP",
                     "SecHybridNoMemP", "SecFactsPromptDiffN", "SecFactsMatchedN",
                     "SecFactsMatchedFullP", "GridEventsHybridFullP"):
            todo(name, "second-model run incomplete (run_second_model.py)")
    j2_path = RESULTS / "judge2_n100.json"
    j2 = json.loads(j2_path.read_text(encoding="utf-8")) if j2_path.exists() else []
    j2 = [r for r in j2 if r.get("label2") in SCORED]
    if len(j2) == 300:
        assert {r["judge2"] for r in j2} == {"openai/gpt-oss-120b"}
        from bapca.stats import cohens_kappa as _kappa
        k = _kappa([r["label"] == "correct" for r in j2], [r["label2"] == "correct" for r in j2])
        macro("JudgeTwoN", "%d" % len(j2))
        macro("JudgeTwoKappa", "%.3f" % k.kappa)
        macro("JudgeTwoAgree", pct(k.raw))
        by = {c: {r["sample_index"]: r["label2"] == "correct" for r in j2 if r["condition"] == c}
              for c in ("no_memory", "full_context", "ours")}
        macro("JudgeTwoNoMem", pct(sum(by["no_memory"].values()) / len(by["no_memory"])))
        macro("JudgeTwoFull", pct(sum(by["full_context"].values()) / len(by["full_context"])))
        macro("JudgeTwoOurs", pct(sum(by["ours"].values()) / len(by["ours"])))
        macro("JudgeTwoOursNoMemP", tex_pvalue(paired(by["ours"], by["no_memory"]).p_value))
        macro("JudgeTwoOursFullP", tex_pvalue(paired(by["ours"], by["full_context"]).p_value))
    else:
        for name in ("JudgeTwoN", "JudgeTwoKappa", "JudgeTwoAgree", "JudgeTwoNoMem",
                     "JudgeTwoFull", "JudgeTwoOurs", "JudgeTwoOursNoMemP", "JudgeTwoOursFullP"):
            todo(name, "second judge incomplete (rejudge.py)")

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
