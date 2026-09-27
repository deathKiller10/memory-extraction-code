"""
Tests for the dataset loader, judge parsing, and statistics.

The statistics tests check against values computed independently (textbook
Wilson intervals, exact binomial tails worked by hand) rather than against
whatever my implementation happens to produce. A stats bug here would not
crash -- it would quietly produce a confident wrong conclusion, which is the
single most expensive kind of error this project can make.
"""

import json
import pathlib

import pytest

from bapca.dataset import LocomoPlus, Sample
from bapca.evaluation import (CONDITIONS, CONDITIONS_BY_KEY, JUDGE_SYSTEM,
                              parse_verdict, prompt_full_context,
                              prompt_no_memory, prompt_oracle_constraint,
                              prompt_oracle_cue)
from bapca.stats import McNemar, Proportion, mcnemar, min_detectable_gap


def make_sample(**kwargs) -> Sample:
    base = dict(
        index=0,
        input_prompt="Caroline said, \"hi\"\n" * 50 + "A: I'm overwhelmed.",
        trigger="A: I'm overwhelmed.",
        evidence="Caroline: After learning to say 'no', I've felt less stressed.",
        category="Cognitive",
        answer=None,
        time_gap="two weeks later",
    )
    base.update(kwargs)
    return Sample(**base)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

@pytest.fixture
def dataset_file(tmp_path):
    records = [
        {"input_prompt": f"conv {i}", "trigger": f"t{i}", "evidence": f"e{i}",
         "category": "Cognitive", "time_gap": "one month later"}
        for i in range(50)
    ] + [
        {"input_prompt": f"c{i}", "trigger": f"t{i}", "evidence": f"e{i}",
         "category": "single-hop", "answer": "yes"}
        for i in range(30)
    ]
    path = tmp_path / "unified.json"
    path.write_text(json.dumps(records))
    return path


def test_loads_and_separates_cognitive(dataset_file):
    data = LocomoPlus(dataset_file)
    assert len(data) == 80
    assert len(data.cognitive()) == 50


def test_cognitive_samples_have_no_reference_answer(dataset_file):
    """By design: correctness is evidence-consistency, judged, not string-matched."""
    assert all(s.answer is None for s in LocomoPlus(dataset_file).cognitive())


def test_subset_is_reproducible(dataset_file):
    data = LocomoPlus(dataset_file)
    first = [s.index for s in data.subset(10, seed=42)]
    second = [s.index for s in data.subset(10, seed=42)]
    assert first == second


def test_different_seeds_give_different_subsets(dataset_file):
    data = LocomoPlus(dataset_file)
    assert [s.index for s in data.subset(10, seed=1)] != \
           [s.index for s in data.subset(10, seed=2)]


def test_subset_larger_than_pool_returns_everything(dataset_file):
    assert len(LocomoPlus(dataset_file).subset(999)) == 50


def test_subset_only_draws_from_the_requested_category(dataset_file):
    assert all(s.is_cognitive for s in LocomoPlus(dataset_file).subset(20))


def test_missing_file_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_eval_set"):
        LocomoPlus(tmp_path / "nope.json")


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------

def test_no_memory_is_only_the_trigger():
    sample = make_sample()
    assert prompt_no_memory(sample) == sample.trigger


def test_full_context_is_the_benchmark_prompt_unchanged():
    """Reformatting it would break comparability with published numbers."""
    sample = make_sample()
    assert prompt_full_context(sample) == sample.input_prompt


def test_oracle_cue_contains_the_evidence_and_the_trigger():
    sample = make_sample()
    prompt = prompt_oracle_cue(sample)
    assert sample.evidence in prompt and sample.trigger in prompt


def test_oracle_conditions_are_far_smaller_than_full_context():
    """The efficiency claim depends on this being true by construction."""
    sample = make_sample()
    assert len(prompt_oracle_cue(sample)) < len(prompt_full_context(sample)) / 5


def test_oracle_constraint_uses_the_extracted_note_not_the_raw_cue():
    sample = make_sample()
    prompt = prompt_oracle_constraint(sample, "She protects her time by saying no.")
    assert "protects her time" in prompt
    assert sample.evidence not in prompt


def test_no_condition_discloses_that_memory_is_being_tested():
    """LoCoMo-Plus 5.3: task disclosure changes behaviour and breaks
    comparability. Our prompts must read as ordinary conversation."""
    sample = make_sample()
    banned = ("memory", "recall", "remember", "benchmark", "evaluat", "test")
    for condition in CONDITIONS:
        text = condition.build(sample, "a standing note").lower()
        # Strip the sample's own content; we are checking OUR wrapper text.
        wrapper = text.replace(sample.input_prompt.lower(), "")
        wrapper = wrapper.replace(sample.evidence.lower(), "")
        wrapper = wrapper.replace(sample.trigger.lower(), "")
        for word in banned:
            assert word not in wrapper, f"{condition.key} leaks {word!r}"


def test_only_the_constraint_condition_needs_extraction():
    assert [c.key for c in CONDITIONS if c.needs_extraction] == ["oracle_constraint"]


def test_judge_system_prompt_matches_the_benchmark_wording():
    assert JUDGE_SYSTEM.startswith("You are a Memory Awareness Judge")


# --------------------------------------------------------------------------
# Judge parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ('{"label": "correct", "reason": "reflects the cue"}', "correct"),
    ('{"label": "wrong", "reason": "no link"}', "wrong"),
    ('Sure!\n```json\n{"label":"correct","reason":"yes"}\n```', "correct"),
    ('  {"label":"CORRECT","reason":"x"}  ', "correct"),
    ("correct", "correct"),
    ("wrong", "wrong"),
])
def test_verdicts_parse(raw, expected):
    assert parse_verdict(raw).label == expected


@pytest.mark.parametrize("raw", ["", "I am not sure about this one.", "{oops"])
def test_unusable_judge_output_is_flagged_not_scored_wrong(raw):
    """Scoring a malformed verdict as 'wrong' would bias every result downward
    and look exactly like a real finding."""
    verdict = parse_verdict(raw)
    assert verdict.label == "unparsed"
    assert not verdict.parsed


def test_scores_map_to_one_and_zero():
    assert parse_verdict('{"label":"correct","reason":""}').score == 1.0
    assert parse_verdict('{"label":"wrong","reason":""}').score == 0.0
    assert parse_verdict("garbage").score == 0.0


# --------------------------------------------------------------------------
# Statistics -- checked against independently computed values
# --------------------------------------------------------------------------

def test_wilson_interval_matches_hand_computed_value():
    """
    p=0.5, n=40, z=1.96, worked longhand:
        denom  = 1 + 1.96^2/40                       = 1.09604
        margin = 1.96*sqrt(0.25/40 + 1.96^2/6400)/denom = 0.148007
    -> [0.351993, 0.648007]
    """
    low, high = Proportion(20, 40).wilson()
    assert low == pytest.approx(0.351993, abs=1e-5)
    assert high == pytest.approx(0.648007, abs=1e-5)


def test_wilson_stays_inside_zero_and_one_at_the_extremes():
    """Where the normal approximation famously fails."""
    low, high = Proportion(0, 20).wilson()
    assert low == 0.0 and 0 < high < 1
    low, high = Proportion(20, 20).wilson()
    assert high == 1.0 and 0 < low < 1


def test_wilson_narrows_as_n_grows():
    width = lambda n: (lambda lo_hi: lo_hi[1] - lo_hi[0])(Proportion(n // 2, n).wilson())  # noqa: E731
    assert width(400) < width(40) < width(10)


def test_empty_proportion_does_not_divide_by_zero():
    assert Proportion(0, 0).rate == 0.0
    assert Proportion(0, 0).wilson() == (0.0, 0.0)


def test_mcnemar_counts_the_four_cells():
    a = [True, True, False, False]
    b = [True, False, True, False]
    result = mcnemar(a, b)
    assert (result.both, result.only_a, result.only_b, result.neither) == (1, 1, 1, 1)


def test_mcnemar_with_no_disagreement_is_p_one():
    result = mcnemar([True, False, True], [True, False, True])
    assert result.p_value == 1.0
    assert "identical" in result.verdict()


def test_mcnemar_exact_p_matches_hand_computed_binomial():
    """
    10 discordant pairs, all favouring b. Two-sided exact binomial:
    2 * (1/2)^10 = 0.001953125.
    """
    a = [False] * 10
    b = [True] * 10
    result = mcnemar(a, b)
    assert result.discordant == 10
    assert result.p_value == pytest.approx(0.001953125, abs=1e-9)
    assert "B better" in result.verdict()


def test_mcnemar_small_discordance_is_not_significant():
    """3 discordant pairs cannot reach p<0.05 however they split -- the
    minimum two-sided p is 0.25. A 40-item pilot must not claim otherwise."""
    result = mcnemar([False, False, False], [True, True, True])
    assert result.p_value == pytest.approx(0.25, abs=1e-9)
    assert "no significant" in result.verdict()


def test_mcnemar_is_symmetric_in_p_value():
    a = [True] * 7 + [False] * 3
    b = [False] * 7 + [True] * 3
    assert mcnemar(a, b).p_value == pytest.approx(mcnemar(b, a).p_value)


def test_mcnemar_names_the_better_condition():
    """10 discordant pairs all favouring A: p = 2*(1/2)^10 = 0.00195."""
    result = mcnemar([True] * 10, [False] * 10)
    assert result.p_value == pytest.approx(0.001953125, abs=1e-9)
    assert "A better" in result.verdict()


def test_an_eight_to_two_split_is_still_not_significant():
    """
    Worth pinning down, because 8-2 *looks* decisive and is not:
    two-sided exact = 2 * P(X <= 2 | n=10) = 2*56/1024 = 0.109.
    A pilot that claimed a win here would be claiming noise.
    """
    result = mcnemar([True] * 8 + [False] * 2, [False] * 8 + [True] * 2)
    assert result.p_value == pytest.approx(0.109375, abs=1e-9)
    assert "no significant" in result.verdict()


def test_mismatched_pair_lengths_are_rejected():
    with pytest.raises(ValueError, match="paired"):
        mcnemar([True, False], [True])


def test_min_detectable_gap_shrinks_with_sample_size():
    assert min_detectable_gap(40) > min_detectable_gap(150) > min_detectable_gap(401)


def test_min_detectable_gap_at_forty_is_about_fifteen_points():
    """The number we print so nobody over-reads the pilot."""
    assert 0.13 < min_detectable_gap(40) < 0.16


def test_assistant_system_prompt_does_not_disclose_the_task():
    """It reaches the model on every generation, so it is part of the prompt
    surface that must not mention memory (LoCoMo-Plus 5.3)."""
    from bapca.evaluation import ASSISTANT_SYSTEM
    lowered = ASSISTANT_SYSTEM.lower()
    for word in ("memory", "recall", "remember", "benchmark", "evaluat", "constraint"):
        assert word not in lowered


def test_run_pilot_passes_the_system_prompt():
    """Regression: it was defined but never passed, so the generator got a raw
    conversation dump with no instruction."""
    source = (pathlib.Path(__file__).parent.parent / "scripts" / "run_pilot.py").read_text()
    assert "system=ASSISTANT_SYSTEM" in source
    assert "system=None" not in source


# --------------------------------------------------------------------------
# Judge agreement
# --------------------------------------------------------------------------

def test_kappa_is_one_for_perfect_agreement():
    from bapca.stats import cohens_kappa
    a = [True, False, True, False, True]
    assert cohens_kappa(a, a).kappa == pytest.approx(1.0)


def test_kappa_is_zero_when_agreement_is_only_chance():
    """
    Both raters say 'correct' half the time, and agree on exactly half the
    items -- which is what chance predicts. Observed 0.5, expected
    0.5*0.5 + 0.5*0.5 = 0.5, so kappa = 0.
    """
    from bapca.stats import cohens_kappa
    a = [True, True, False, False]
    b = [True, False, True, False]
    assert cohens_kappa(a, b).kappa == pytest.approx(0.0)


def test_kappa_punishes_a_lenient_judge_that_raw_agreement_flatters():
    """
    The case we actually care about: rater A marks everything correct, rater B
    marks 80% correct. Raw agreement is a comfortable 80%, but they agree no
    more than chance forces them to -- kappa exposes it as 0.
    """
    from bapca.stats import cohens_kappa
    a = [True] * 10
    b = [True] * 8 + [False] * 2
    result = cohens_kappa(a, b)
    assert result.raw == pytest.approx(0.8)
    assert result.kappa == pytest.approx(0.0)
    assert result.reading() == "slight"


def test_kappa_can_go_negative():
    from bapca.stats import cohens_kappa
    result = cohens_kappa([True, True, False, False], [False, False, True, True])
    assert result.kappa < 0
    assert result.reading() == "worse than chance"


def test_kappa_counts_directional_disagreement():
    from bapca.stats import cohens_kappa
    result = cohens_kappa([True, True, True], [True, False, False])
    assert (result.both_correct, result.a_only, result.b_only) == (1, 2, 0)


def test_kappa_rejects_mismatched_lengths():
    from bapca.stats import cohens_kappa
    with pytest.raises(ValueError, match="paired"):
        cohens_kappa([True], [True, False])


def test_truncated_judge_json_still_yields_its_label():
    """gemini-3.6-flash returned 9 unparsed verdicts out of 11: it was thinking
    first and running out of tokens mid-JSON. The label was always readable."""
    truncated = '{"label": "correct", "reason": "The prediction acknowledges her earlier'
    verdict = parse_verdict(truncated)
    assert verdict.label == "correct"
    assert verdict.parsed


def test_truncated_wrong_label_is_recovered_too():
    assert parse_verdict('{"label": "wrong", "reason": "No connection to the').label == "wrong"


def test_label_recovery_handles_single_quotes_and_spacing():
    assert parse_verdict("{'label' : 'correct'").label == "correct"
    assert parse_verdict('label="wrong"').label == "wrong"


def test_recovery_does_not_invent_a_label_from_prose():
    """'correct' appearing in a sentence must not be mistaken for a verdict."""
    assert parse_verdict("It is hard to say whether this is correct.").label == "unparsed"
