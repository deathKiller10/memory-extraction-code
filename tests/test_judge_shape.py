"""
paper_numbers.judge_shape: the judge-validation paragraph's figures come from
judge_agreement.json, not from prose (18 Nov: the draft said the judge was
'within a few points' on the strongest condition and that one adjacent pair
was 'tied'; the file says 0 and 8 points and the two oracles swapped).
"""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "paper_numbers", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "paper_numbers.py")
pn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pn)


def agreement(cue_b=0.70, constraint_b=0.80):
    rates = {"no_memory": (0.25, 0.45, 0.20, 0.40), "full_context": (0.40, 0.60, 0.35, 0.55),
             "system_llm": (0.50, 0.60, 0.45, 0.62), "oracle_cue": (0.70, 0.75, cue_b, 0.80),
             "oracle_constraint": (0.80, 0.80, constraint_b, 0.88)}
    per = {c: {"a": {"human": h1, "judge": j1}, "b": {"human": h2, "judge": j2}}
           for c, (h1, j1, h2, j2) in rates.items()}
    return {"per_condition": per,
            "spread": {"a": {"human": 0.55, "judge": 0.35}, "b": {"human": 0.60, "judge": 0.48}}}


def emitted(a):
    pn.lines.clear()
    pn.missing.clear()
    pn.judge_shape(a)
    return dict(l[len("\\newcommand{\\"):-1].split("}{", 1) for l in pn.lines)


def test_leniency_and_spread_ranges():
    m = emitted(agreement())
    assert (m["JudgeOverALo"], m["JudgeOverAHi"]) == ("20.0", "20.0")
    assert (m["JudgeOverDLo"], m["JudgeOverDHi"]) == ("0.0", "8.0")
    assert (m["HumanSpreadLo"], m["HumanSpreadHi"]) == ("55.0", "60.0")
    assert (m["JudgeSpreadLo"], m["JudgeSpreadHi"]) == ("35.0", "48.0")


def test_single_oracle_swap_is_reported():
    m = emitted(agreement(cue_b=0.727, constraint_b=0.72))
    assert (m["OrderSwapCue"], m["OrderSwapConstraint"]) == ("72.7", "72.0")


def test_no_swap_or_another_swap_renders_todo():
    assert "todo" in emitted(agreement())["OrderSwapCue"]           # order intact
    a = agreement()
    a["per_condition"]["full_context"]["a"]["human"] = 0.10          # A above B instead
    assert "todo" in emitted(a)["OrderSwapCue"]
