"""
The interventions table's changes are computed on the items BOTH runs scored
(paper_numbers.matched_delta), not as a difference of two scores with different
denominators. The strip-the-query run has 99 scored items (#2374 unparsed):
65/100 - 41/99 read 23.6 where the matched gain is 23.2, and 41/99 - 42/100
read -0.6 where nothing changed (18 Nov).
"""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "paper_numbers", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "paper_numbers.py")
pn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pn)


def rows(correct, wrong, start=0):
    out = {i: {"label": "correct"} for i in range(start, start + correct)}
    out.update({i: {"label": "wrong"} for i in range(start + correct, start + correct + wrong)})
    return out


def test_unscored_item_is_dropped_from_both_sides():
    before = rows(42, 58)                    # 42/100, item 0 correct
    after = rows(42, 58)
    del after[0]                             # the unparsed item: 41/99
    delta, n = pn.matched_delta(after, before)
    assert n == 99 and delta == 0.0          # unmatched: 41.4 - 42.0 = -0.6


def test_gain_is_counted_on_common_items_only():
    before = {i: {"label": "wrong"} for i in range(1, 100)}          # 0/99, item 0 missing
    after = {i: {"label": "correct"} for i in range(0, 50)}
    after.update({i: {"label": "wrong"} for i in range(50, 100)})    # 50/100, item 0 correct
    delta, n = pn.matched_delta(after, before)
    assert n == 99 and round(delta, 1) == 49.5                      # 49/99, not 50/100


def test_signed_never_prints_negative_zero():
    assert pn.signed(0.0) == "+0.0"
    assert pn.signed(-0.0) == "+0.0"
    assert pn.signed(-0.04) == "+0.0"
    assert pn.signed(23.23) == "+23.2"
    assert pn.signed(-1.0) == "-1.0"


def test_chain_follows_the_table_order():
    assert pn.INTERVENTION_CHAIN == (("Permissive", "Forced"), ("Forced", "Hybrid"),
                                     ("Hybrid", "NoTrig"), ("NoTrig", "Events"),
                                     ("Events", "Person"))
