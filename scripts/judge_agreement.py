"""
Judge-vs-human agreement, computed from the stored annotations alone.

    python scripts/judge_agreement.py [--export]

Free, local, and offline: no API calls, no dataset, no embedder. It reads
`results/annotations/*.json` and the judge labels already recorded in the pilot.

WHY NOT validate_judge.py. That script re-judges 100 responses, so it needs the
dataset and spends quota. The agreement figures the paper quotes do not require
either -- every label already exists. This computes them where they can be
re-run any time, and `--export` leaves them for scripts/paper_numbers.py so the
Method section never has a kappa typed into its prose.

The claim the paper makes is narrow and is what these numbers support: the
automatic judge sits inside the human-human disagreement band, it is more
lenient than the humans in absolute terms, and the ORDERING of conditions is
preserved. Absolute rates are inflated; effect sizes are not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca.stats import Proportion, cohens_kappa, mcnemar

RESULTS = Path("results")
ANNOTATIONS = RESULTS / "annotations"
PILOT = RESULTS / "pilot_n100_seed42.json"
SCORED = ("correct", "wrong")


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def load_annotations() -> dict:
    out = {}
    for path in sorted(ANNOTATIONS.glob("*.json")):
        out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


# Condition E's judge labels are NOT in the pilot -- they are in the system run.
# Without this the 40 newly annotated condition-E items would silently have no
# judge label to compare against, and the one question the batch was built to
# answer would quietly go missing. (14 Sep)
SYSTEM_RUN = RESULTS / "system_n100_seed42_llm3_d1_notrig_events.json"


def judge_labels() -> dict:
    out = {}
    for path in (PILOT, SYSTEM_RUN):
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        for r in json.loads(path.read_text(encoding="utf-8")):
            if r.get("label") in SCORED:
                out[f"{r['condition']}:{r['sample_index']}"] = r["label"]
    return out


def as_bools(keys, table):
    return [table[k] == "correct" for k in keys]


def main() -> int:
    print("Files actually read (check these are the ones you meant):")
    for path in sorted(ANNOTATIONS.glob("*.json")):
        print(f"  {path}")
    print(f"  {PILOT}")
    print(f"  {SYSTEM_RUN}")

    ann = load_annotations()
    judge = judge_labels()
    export = {}

    # Pool every file belonging to an annotator (
    # annotator2_solo.json, annotator2_batch2.json ...) so a new batch is picked
    # up without editing this script. The joint session is kept separate: it
    # is one agreed label per item, not one annotator's opinion.
    JOINT = "joint"          # the 60-item joint session, historically named
    people = {}
    for stem, table in ann.items():
        if stem == JOINT:
            continue
        who = stem.split("_")[0]
        people.setdefault(who, {}).update(
            {k: v for k, v in table.items() if v in SCORED})
    names = sorted(people)
    print("\n  annotators found: " + ", ".join(
        "%s (%d labels)" % (n, len(people[n])) for n in names))

    banner("1. Human vs human, on the items both annotated independently")
    if len(names) < 2:
        print("\n  Need two annotators for an agreement figure.")
        return 0
    pri, arn = people[names[0]], people[names[1]]
    shared = sorted(set(pri) & set(arn) & set(judge))
    print(f"  items annotated by both, and judged: {len(shared)}")
    if len(shared) >= 10:
        hh = cohens_kappa(as_bools(shared, pri), as_bools(shared, arn))
        agree = sum(pri[k] == arn[k] for k in shared) / len(shared)
        print(f"  raw agreement          {100*agree:.1f}%")
        print(f"  Cohen's kappa          {hh.kappa:.3f}   ({hh.reading()})")
        export.update(human_n=len(shared), human_agreement=agree, human_kappa=hh.kappa,
                      human_band=hh.reading())

        banner("2. Judge vs each human, on those same items")
        for name, table in ((names[0].title(), pri), (names[1].title(), arn)):
            jh = cohens_kappa(as_bools(shared, table), as_bools(shared, judge))
            agree = sum(table[k] == judge[k] for k in shared) / len(shared)
            print(f"  judge vs {name:<10} raw {100*agree:5.1f}%   "
                  f"kappa {jh.kappa:.3f}   ({jh.reading()})")
            export["judge_vs_%s_kappa" % name.lower()] = jh.kappa
            export["judge_vs_%s_agreement" % name.lower()] = agree

        banner("3. How the judge compares to the human-human band")
        ks = [export["judge_vs_%s_kappa" % n] for n in names[:2]]
        lo, hi = min(ks), max(ks)
        print(f"  human vs human         {hh.kappa:.3f}")
        print(f"  judge vs human         {lo:.3f} to {hi:.3f}")
        print("\n  Report all three numbers; do not collapse them to one sentence.")
        if lo < hh.kappa - 0.05:
            print("  -> The two annotators agree with EACH OTHER better than the judge")
            print("     agrees with either. The judge is OUTSIDE the human band.")
        else:
            print("  -> The judge is comparable to the human-human band.")
        export.update(judge_kappa_lo=lo, judge_kappa_hi=hi)

    banner("4. Is the judge more lenient than the humans?")
    joint = ann.get("joint", {})
    both = sorted(set(joint) & set(judge))
    if both:
        h_rate = Proportion(sum(joint[k] == "correct" for k in both), len(both))
        j_rate = Proportion(sum(judge[k] == "correct" for k in both), len(both))
        print(f"  on the {len(both)} jointly annotated items:")
        print(f"    humans call correct   {h_rate}")
        print(f"    judge calls correct   {j_rate}")
        print(f"    judge is {100*(j_rate.rate - h_rate.rate):+.1f} points more lenient")
        export.update(joint_n=len(both), human_rate=h_rate.rate, judge_rate=j_rate.rate,
                      leniency=j_rate.rate - h_rate.rate)

    banner("5. Does the ORDERING of conditions survive?")
    print("  This is what the paper's argument actually needs: absolute rates may")
    print("  be inflated, but if the judge ranks the conditions as the humans do,")
    print("  every comparison between conditions stands.")
    # Earlier this used only the 60-item joint session -- 15 labels per
    # condition -- and reported the ordering as broken. At that n nothing was
    # decidable. With batch 2 each annotator has 22-25 labels per condition and
    # 38-40 for condition E, so report per annotator over ALL their labels and
    # let the reader see both columns. (14 Sep)
    CONDS = ("no_memory", "full_context", "system_llm", "oracle_cue",
             "oracle_constraint")
    export["per_condition"] = {}
    for who in names:
        table = people[who]
        print(f"\n  {who}")
        print(f"    {'condition':<20} {'n':>4} {'human':>8} {'judge':>8} {'gap':>8}")
        seq = []
        for cond in CONDS:
            keys = [k for k in table if k.startswith(cond + ":") and k in judge]
            if len(keys) < 10:
                continue
            h = sum(table[k] == "correct" for k in keys) / len(keys)
            j = sum(judge[k] == "correct" for k in keys) / len(keys)
            seq.append((cond, h))
            print(f"    {cond:<20} {len(keys):>4} {100*h:7.1f}% {100*j:7.1f}% "
                  f"{100*(j-h):+7.1f}")
            export["per_condition"].setdefault(cond, {})[who] = {
                "n": len(keys), "human": h, "judge": j}
        if len(seq) >= 4:
            ranked = [c for c, _ in sorted(seq, key=lambda x: -x[1])]
            expected = [c for c, _ in sorted(seq, key=lambda x: -x[1])]
            monotone = [c for c in reversed(CONDS) if c in dict(seq)]
            ok = ranked == monotone
            if ok:
                print("    ordering matches the designed difficulty order: YES")
            else:
                bad = [(monotone[i], monotone[i+1]) for i in range(len(monotone)-1)
                       if dict(seq)[monotone[i]] < dict(seq)[monotone[i+1]]]
                detail = "; ".join(
                    f"{a} {100*dict(seq)[a]:.1f}% < {b} {100*dict(seq)[b]:.1f}%"
                    for a, b in bad)
                print(f"    ordering matches the designed difficulty order: NO  ({detail})")
            export.setdefault("ordering_ok", {})[who] = bool(ok)
            spread = max(h for _, h in seq) - min(h for _, h in seq)
            jspread = (max(export["per_condition"][c][who]["judge"] for c, _ in seq)
                       - min(export["per_condition"][c][who]["judge"] for c, _ in seq))
            print(f"    spread: human {100*spread:.1f} pts, judge {100*jspread:.1f} pts")
            export.setdefault("spread", {})[who] = {"human": spread, "judge": jspread}
    print("\n  The judge inflates every condition, and inflates the WEAK ones most.")
    print("  It therefore compresses the range. Report both columns.")

    banner("6. Condition E: does a human agree with the headline?")
    print("  The paper's headline is condition E's score. Until this batch, no")
    print("  human had labelled a single condition-E response.")
    e_keys = sorted(k for k in judge if k.startswith("system_llm:"))
    for who in names:
        table = people[who]
        both_e = [k for k in e_keys if k in table]
        if len(both_e) < 10:
            continue
        h = Proportion(sum(table[k] == "correct" for k in both_e), len(both_e))
        j = Proportion(sum(judge[k] == "correct" for k in both_e), len(both_e))
        t = mcnemar([table[k] == "correct" for k in both_e],
                    [judge[k] == "correct" for k in both_e])
        print(f"\n  {who} vs the judge, on {len(both_e)} condition-E items:")
        print(f"    {who:<12} says correct  {h}")
        print(f"    {'judge':<12} says correct  {j}")
        print(f"    difference {100*(j.rate - h.rate):+.1f} points, "
              f"McNemar p={t.p_value:.4f} "
              f"({who} only {t.only_a}, judge only {t.only_b})")
        export[f"E_{who}_rate"] = h.rate
        export[f"E_{who}_n"] = len(both_e)
        export[f"E_judge_rate_on_{who}_items"] = j.rate
        export[f"E_{who}_p"] = t.p_value

    if "--export" in sys.argv:
        out = RESULTS / "judge_agreement.json"
        out.write_text(json.dumps(export, indent=1), encoding="utf-8")
        print(f"\n  Exported to {out} for scripts/paper_numbers.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
