"""
Tests for the no-oracle pipeline.

The recurring danger in this file is an accidental oracle: any place the system
uses information a real deployment would not have. The tests below are mostly
there to catch that.
"""

from datetime import datetime

import pathlib

import pytest

from bapca.dataset import Sample
from bapca.embeddings import HashEmbedder
from bapca.memory import MemoryType
from bapca.pipeline import (DEFAULT_WINDOW_TURNS, Note, SystemConfig, Window,
                            build_store, estimate_unique_windows, extract_notes,
                            hit_rate, parse_note, prompt_with_memory, segment)

EMB = HashEmbedder()


def make_sample(prompt="", **kw):
    base = dict(index=0, input_prompt=prompt, trigger="A: I'm overwhelmed.",
                evidence="Caroline: I've learned to say no.", category="Cognitive",
                answer=None, time_gap="two weeks later")
    base.update(kw)
    return Sample(**base)


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------

def test_fixed_windows_when_there_are_no_date_markers():
    prompt = "\n".join(f"Speaker said, \"line {i}\"" for i in range(30))
    windows = segment(prompt, window_turns=10)
    assert len(windows) == 3
    assert all(not w.dated for w in windows)
    assert [w.index for w in windows] == [0, 1, 2]


def test_session_boundaries_are_used_when_present():
    prompt = ("DATE: 1 May 2023 CONVERSATION:\nA said x\nB said y\n"
              "DATE: 8 May 2023 CONVERSATION:\nA said z\n")
    windows = segment(prompt)
    assert len(windows) == 2
    assert all(w.dated for w in windows)
    assert "1 May" in windows[0].text and "8 May" in windows[1].text


def test_a_single_date_line_does_not_trigger_session_mode():
    """One marker is a header, not a boundary; two or more is a structure."""
    prompt = "DATE: 1 May 2023\n" + "\n".join(f"turn {i}" for i in range(30))
    assert all(not w.dated for w in segment(prompt, window_turns=10))


def test_blank_lines_do_not_become_windows():
    prompt = "a\n\n\n\nb\n\n\nc"
    assert len(segment(prompt, window_turns=2)) == 2


def test_empty_prompt_yields_nothing():
    assert segment("") == []


def test_windows_are_ordered_oldest_first():
    prompt = "\n".join(f"turn {i}" for i in range(24))
    windows = segment(prompt, window_turns=12)
    assert "turn 0" in windows[0].text
    assert "turn 23" in windows[-1].text


def test_unique_window_count_measures_the_caching_saving():
    """The ten base conversations repeat, and that is what makes this runnable."""
    shared = "\n".join(f'Speaker {i%2} said, "turn {i}"' for i in range(120))
    samples = [make_sample(shared), make_sample(shared), make_sample(shared)]
    total, unique = estimate_unique_windows(samples, window_turns=12)
    assert total == unique * 3     # three identical items collapse perfectly
    assert unique < total


def test_identical_items_cost_nothing_extra():
    shared = "\n".join(f'Speaker {i%2} said, "turn {i}"' for i in range(120))
    one = estimate_unique_windows([make_sample(shared)], 12)[1]
    fifty = estimate_unique_windows([make_sample(shared)] * 50, 12)[1]
    assert one == fifty


# --------------------------------------------------------------------------
# Note parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,kind,body", [
    ("goal: She is preparing for an exam.", MemoryType.GOAL, "She is preparing for an exam."),
    ("value - He does not eat meat.", MemoryType.VALUE, "He does not eat meat."),
    ("CONSTRAINT: She avoids late calls.", MemoryType.CONSTRAINT, "She avoids late calls."),
    ("state: He is anxious about money.", MemoryType.STATE, "He is anxious about money."),
])
def test_typed_notes_parse(raw, kind, body):
    note = parse_note(raw, 3)
    assert note.mem_type is kind
    assert note.text == body
    assert note.window_index == 3


@pytest.mark.parametrize("raw", ["NONE", "none", "  NONE  ", "", "   ", '"NONE"'])
def test_nothing_lasting_returns_no_note(raw):
    assert parse_note(raw, 0) is None


def test_untyped_note_defaults_to_the_fastest_decaying_type():
    """Defaulting to `value` would let junk notes outlive real ones."""
    note = parse_note("He bought a standing desk.", 0)
    assert note.mem_type is MemoryType.STATE


def test_only_the_first_line_of_a_rambling_answer_is_kept():
    note = parse_note("goal: She wants to run a marathon.\nAlso she likes cats.", 0)
    assert note.text == "She wants to run a marathon."


def test_a_typed_none_is_still_none():
    assert parse_note("goal: NONE", 0) is None


# --------------------------------------------------------------------------
# The store, and the absence of oracles
# --------------------------------------------------------------------------

def _notes(n=6):
    return [Note(f"She does thing {i}.", MemoryType.GOAL, i) for i in range(n)]


def test_all_notes_start_with_the_same_weight():
    """Weighting notes by anything we know about the gold cue would be an
    oracle smuggled in at write time."""
    store = build_store(_notes(), EMB, SystemConfig())
    assert len({n.weight for n in store.nodes}) == 1


def test_older_windows_are_aged_further_back():
    store = build_store(_notes(4), EMB, SystemConfig())
    by_window = {n.source_turn: n.last_accessed for n in store.nodes}
    assert by_window[0] < by_window[1] < by_window[2] < by_window[3]


def test_recent_notes_outrank_old_ones_when_the_budget_bites():
    config = SystemConfig(carry=2)
    store = build_store(_notes(8), EMB, config)
    prompt, _, carried = prompt_with_memory(make_sample(), store, EMB)
    assert len(carried) == 2
    # decay should favour the newest windows
    assert "thing 7" in " ".join(carried) or "thing 6" in " ".join(carried)


def test_carry_budget_is_respected():
    for carry in (1, 3, 5):
        store = build_store(_notes(20), EMB, SystemConfig(carry=carry))
        _, _, carried = prompt_with_memory(make_sample(), store, EMB)
        assert len(carried) <= carry


def test_no_notes_means_the_prompt_is_just_the_trigger():
    sample = make_sample()
    store = build_store([], EMB, SystemConfig())
    prompt, _, carried = prompt_with_memory(sample, store, EMB)
    assert prompt == sample.trigger
    assert carried == []


def test_the_prompt_never_contains_the_gold_evidence_verbatim():
    """The whole point of E: it must not receive the answer."""
    sample = make_sample()
    store = build_store(_notes(), EMB, SystemConfig())
    prompt, _, _ = prompt_with_memory(sample, store, EMB)
    assert sample.evidence not in prompt


def test_the_prompt_is_far_smaller_than_full_context():
    sample = make_sample("\n".join(f"turn {i}" for i in range(500)))
    store = build_store(_notes(), EMB, SystemConfig())
    _, tokens, _ = prompt_with_memory(sample, store, EMB)
    assert tokens < len(sample.input_prompt) // 40


def test_prompt_does_not_disclose_that_memory_is_being_tested():
    store = build_store(_notes(), EMB, SystemConfig())
    prompt, _, _ = prompt_with_memory(make_sample(), store, EMB)
    wrapper = prompt.replace(make_sample().trigger, "").lower()
    for word in ("memory", "recall", "remember", "evaluat", "benchmark"):
        assert word not in wrapper


# --------------------------------------------------------------------------
# The diagnostic
# --------------------------------------------------------------------------

def test_hit_rate_is_zero_when_nothing_was_carried():
    assert hit_rate([], "some evidence", EMB) == 0.0


def test_hit_rate_finds_an_exact_match():
    assert hit_rate(["She has learned to say no."], "She has learned to say no.", EMB)


def test_hit_rate_rejects_unrelated_notes():
    assert not hit_rate(["He plays badminton."], "She has learned to say no.", EMB)


def test_extraction_skips_windows_that_reveal_nothing():
    class FakeLLM:
        def __init__(self): self.calls = 0
        def complete(self, *a, **k):
            self.calls += 1
            return "NONE" if self.calls % 2 else "goal: She is training."

    llm = FakeLLM()
    notes = extract_notes(llm, [Window(i, f"w{i}", False) for i in range(4)])
    assert llm.calls == 4        # every window is examined
    assert len(notes) == 2       # only half yield anything
    assert all(n.mem_type is MemoryType.GOAL for n in notes)


# --------------------------------------------------------------------------
# Content-defined chunking
#
# The real dry run showed 5,047 windows collapsing to only 1,153 unique, when
# ten base conversations should have collapsed much further. Cause: fixed
# windows are anchored to position, so inserting the cue at a different point
# in each item shifts every downstream boundary and kills deduplication.
# --------------------------------------------------------------------------

def _conversation(n=240):
    return [f'Speaker {i % 2} said, "turn {i}"' for i in range(n)]


def test_an_insertion_disturbs_only_a_few_windows():
    """The property that makes the whole run affordable."""
    base = _conversation()
    original = {w.text for w in segment("\n".join(base), 12)}

    with_cue = base[:100] + ['Speaker 0 said, "THE CUE"'] + base[100:]
    after = {w.text for w in segment("\n".join(with_cue), 12)}

    survived = original & after
    assert len(survived) >= len(original) - 3   # at most a couple disturbed


def test_deduplication_stays_near_optimal_across_many_insertion_points():
    base = _conversation()
    alone = len(segment("\n".join(base), 12))

    seen = set()
    positions = list(range(0, 240, 6))          # 40 different insertion points
    for pos in positions:
        doc = "\n".join(base[:pos] + ['Speaker 0 said, "THE CUE"'] + base[pos:])
        seen.update(w.text for w in segment(doc, 12))

    ideal = alone + len(positions)              # every base window + one cue window each
    assert len(seen) <= ideal * 1.15            # fixed windows came in at ~2.5x


def test_fixed_position_windows_would_have_been_much_worse():
    """Documents the bug this replaced, so nobody reverts it."""
    base = _conversation()
    fixed = lambda lines, n=12: {"\n".join(lines[i:i + n]) for i in range(0, len(lines), n)}

    seen_fixed, seen_cdc = set(), set()
    for pos in range(0, 240, 24):
        doc = base[:pos] + ['Speaker 0 said, "THE CUE"'] + base[pos:]
        seen_fixed |= fixed(doc)
        seen_cdc |= {w.text for w in segment("\n".join(doc), 12)}
    assert len(seen_cdc) < len(seen_fixed)


def test_windows_stay_within_sane_size_bounds():
    windows = segment("\n".join(_conversation(600)), 12)
    sizes = [len(w.text.splitlines()) for w in windows]
    assert min(sizes[:-1]) >= 6        # target // 2
    assert max(sizes) <= 24            # target * 2


def test_chunking_is_deterministic():
    doc = "\n".join(_conversation())
    assert [w.text for w in segment(doc, 12)] == [w.text for w in segment(doc, 12)]


def test_every_line_survives_segmentation():
    """Losing turns would silently hide evidence from the extractor."""
    lines = _conversation(137)
    rejoined = "\n".join(w.text for w in segment("\n".join(lines), 12))
    assert rejoined.splitlines() == lines


def test_average_window_is_near_the_target():
    windows = segment("\n".join(_conversation(1200)), 12)
    average = sum(len(w.text.splitlines()) for w in windows) / len(windows)
    assert 7 <= average <= 18


# --------------------------------------------------------------------------
# The defaults that condition E's first run exposed
#
# E scored 36% (below the 52% full-context baseline) with the needed note
# reaching the prompt 14% of the time, against 17% for random choice. Two
# causes: decay tuned for a different timescale deleted most notes, and a
# budget of 5 out of 29 made the rest a lottery.
# --------------------------------------------------------------------------

def test_default_decay_no_longer_deletes_most_of_the_conversation():
    """At 7 days per window a 29-window conversation lost most of its notes
    to pruning before selection ever ran."""
    from datetime import datetime
    notes = [Note(f"She does thing {i}.", MemoryType.STATE, i) for i in range(29)]
    store = build_store(notes, EMB, SystemConfig())
    store.prune(datetime(2026, 1, 1))
    assert len(store.archive) == 0
    assert len(store.nodes) == 29


def test_the_old_settings_would_have_pruned_heavily():
    """Documents the failure, so the defaults do not quietly drift back."""
    from datetime import datetime
    from bapca.store import RetrievalConfig
    notes = [Note(f"She does thing {i}.", MemoryType.STATE, i) for i in range(29)]
    old = SystemConfig(days_per_window=7.0,
                       retrieval=RetrievalConfig(standing_budget=5,
                                                 standing_floor=0.05,
                                                 prune_threshold=0.20))
    store = build_store(notes, EMB, old)
    store.prune(datetime(2026, 1, 1))
    assert len(store.archive) > 15      # most of the memory, gone


def test_carrying_everything_is_the_default():
    """5 of 29 by recency is a lottery when the cue sits at a random point."""
    assert SystemConfig().carry == 0
    store = build_store([Note(f"n{i}.", MemoryType.GOAL, i) for i in range(29)],
                        EMB, SystemConfig())
    _, _, carried = prompt_with_memory(make_sample(), store, EMB)
    assert len(carried) == 29


def test_carrying_everything_is_still_far_cheaper_than_full_context():
    """The budget was buying nothing: 29 notes is ~350 tokens against 22,385."""
    sample = make_sample("\n".join(f"turn {i}" for i in range(4000)))
    store = build_store([Note(f"She does thing number {i}.", MemoryType.GOAL, i)
                         for i in range(29)], EMB, SystemConfig())
    _, tokens, _ = prompt_with_memory(sample, store, EMB)
    assert tokens < 700
    assert tokens < len(sample.input_prompt) // 4 // 20     # 20x cheaper at least


def test_an_explicit_budget_still_works_for_the_ablation():
    store = build_store([Note(f"n{i}.", MemoryType.GOAL, i) for i in range(29)],
                        EMB, SystemConfig(carry=5))
    _, _, carried = prompt_with_memory(make_sample(), store, EMB)
    assert len(carried) == 5


# --------------------------------------------------------------------------
# Locating the cue exactly, and the two selection modes
#
# Carrying all 27 notes scored 40% against a 52% full-context baseline, and
# having the right note was worth only +8 points -- where one correct note
# (condition D) was worth 87.6% against a 37% floor. The notes are not the
# problem; the twenty-six around them are. So: measure precisely, then test
# the retrieval we argued would not work.
# --------------------------------------------------------------------------

from bapca.pipeline import locate_evidence_window, render_prompt, select_notes


def _conversation_with_cue(cue_line, at=60, n=120):
    lines = [f'Speaker {i % 2} said, "turn {i}"' for i in range(n)]
    lines.insert(at, cue_line)
    return "\n".join(lines)


def test_cue_window_is_located_exactly():
    cue = 'Caroline said, "After learning to say no, I have felt a lot less stressed."'
    windows = segment(_conversation_with_cue(cue), 12)
    found = locate_evidence_window(
        windows,
        "Caroline：After learning to say no, I have felt a lot less stressed.\n"
        "Melanie：That is a great skill to develop.")
    assert found is not None
    assert "less stressed" in windows[found].text


def test_locator_survives_speaker_prefix_and_punctuation_differences():
    """Gold evidence uses a fullwidth colon and no 'said,' wrapper."""
    cue = 'Jon said, "My back pain has been so constant that I bought a standing desk."'
    windows = segment(_conversation_with_cue(cue), 12)
    assert locate_evidence_window(
        windows, "Jon：My back pain's been so constant that I bought a standing desk!") is not None


def test_locator_returns_none_when_the_cue_is_absent():
    windows = segment(_conversation_with_cue('Speaker 0 said, "hello"'), 12)
    assert locate_evidence_window(
        windows, "Bob：Something entirely unrelated that appears nowhere here.") is None


def test_locator_ignores_fragments_too_short_to_be_distinctive():
    """A five-word line would match half the conversation by accident."""
    windows = segment("\n".join(f"turn {i}" for i in range(60)), 12)
    assert locate_evidence_window(windows, "A：turn 5") is None


def test_carry_mode_returns_every_surviving_note():
    store = build_store([Note(f"n{i}.", MemoryType.GOAL, i) for i in range(20)],
                        EMB, SystemConfig())
    assert len(select_notes(store, make_sample(), EMB, mode="carry")) == 20


def test_similar_mode_returns_only_top_k():
    store = build_store([Note(f"n{i}.", MemoryType.GOAL, i) for i in range(20)],
                        EMB, SystemConfig())
    chosen = select_notes(store, make_sample(), EMB, mode="similar", top_k=3)
    assert len(chosen) == 3


def test_similar_mode_actually_ranks_by_similarity_to_the_trigger():
    sample = make_sample(trigger="A: I am completely overwhelmed by this project.")
    relevant = "He is overwhelmed by his workload on the project."
    notes = [Note("She enjoys gardening at weekends.", MemoryType.GOAL, 0),
             Note(relevant, MemoryType.STATE, 1),
             Note("He collects vintage stamps.", MemoryType.VALUE, 2)]
    store = build_store(notes, EMB, SystemConfig())
    from bapca.embeddings import Embedder  # real semantics needed here
    try:
        embedder = Embedder()
        embedder.dim
    except Exception:
        import pytest as _pytest
        _pytest.skip("sentence-transformers unavailable offline")
    store = build_store(notes, embedder, SystemConfig())
    chosen = select_notes(store, sample, embedder, mode="similar", top_k=1)
    assert chosen[0].text == relevant


def test_unknown_selection_mode_is_rejected():
    store = build_store([Note("n.", MemoryType.GOAL, 0)], EMB, SystemConfig())
    with pytest.raises(ValueError, match="unknown selection mode"):
        select_notes(store, make_sample(), EMB, mode="magic")


def test_similar_mode_is_far_cheaper_in_tokens_than_carrying_everything():
    notes = [Note(f"She does thing number {i} regularly.", MemoryType.GOAL, i)
             for i in range(29)]
    store = build_store(notes, EMB, SystemConfig())
    _, carried_tokens, _ = render_prompt(
        make_sample(), select_notes(store, make_sample(), EMB, mode="carry"))
    _, topk_tokens, _ = render_prompt(
        make_sample(), select_notes(store, make_sample(), EMB, mode="similar", top_k=3))
    assert topk_tokens < carried_tokens / 5


def test_render_prompt_with_no_notes_is_just_the_trigger():
    sample = make_sample()
    prompt, _, lines = render_prompt(sample, [])
    assert prompt == sample.trigger and lines == []


# --------------------------------------------------------------------------
# LLM re-ranking
#
# The measured picture: the extractor captures the cue 91% of the time; with
# three notes in the prompt and one of them right, the answer is good 71% of
# the time; but similarity retrieval finds the right note only 21% of the
# time. Recall is the sole remaining bottleneck, and the cue-trigger link is
# inferential rather than lexical -- so the selector has to reason.
# --------------------------------------------------------------------------

from bapca.pipeline import select_by_llm


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.reply


def _notes_numbered(n=5):
    return [Note(f"Note number {i}.", MemoryType.GOAL, i) for i in range(n)]


def test_reranker_picks_the_numbered_notes():
    notes = _notes_numbered()
    chosen = select_by_llm(FakeLLM("3, 1"), notes, make_sample(), top_k=3)
    assert [n.text for n in chosen] == ["Note number 2.", "Note number 0."]


def test_reranker_respects_the_budget():
    chosen = select_by_llm(FakeLLM("1,2,3,4,5"), _notes_numbered(), make_sample(), top_k=2)
    assert len(chosen) == 2


def test_reranker_refusal_is_no_longer_special_cased():
    """
    It used to be. The prompt offered "NONE" for the case where no note bears
    on the message, on the theory that a selector which always returns
    something re-creates the precision problem.

    Measured, that theory was wrong in the direction that mattered. The
    re-ranker's picks were right 87.5% of the time -- the oracle's own rate --
    but it returned a median of one note out of three and often none, so the
    cue reached the prompt 16 times in 100, below similarity's 21. Precision
    was never the constraint; recall was. "NONE" now parses to no numbers and
    therefore no notes, which `hybrid` then backfills.
    """
    assert select_by_llm(FakeLLM("NONE"), _notes_numbered(), make_sample()) == []


def test_reranker_prompt_demands_a_full_slate():
    llm = FakeLLM("1,2,3")
    select_by_llm(llm, _notes_numbered(), make_sample(), top_k=3)
    prompt = llm.prompts[0]
    assert "exactly 3" in prompt
    assert "If none apply" not in prompt


def test_backfill_tops_up_a_short_answer():
    notes = _notes_numbered(5)
    chosen = select_by_llm(FakeLLM("2"), notes, make_sample(), top_k=3,
                           backfill=list(reversed(notes)))
    assert [n.text for n in chosen] == ["Note number 1.", "Note number 4.",
                                        "Note number 3."]


def test_backfill_rescues_a_refusal():
    notes = _notes_numbered(4)
    chosen = select_by_llm(FakeLLM("NONE"), notes, make_sample(), top_k=2,
                           backfill=notes)
    assert [n.text for n in chosen] == ["Note number 0.", "Note number 1."]


def test_backfill_never_exceeds_the_budget():
    notes = _notes_numbered(6)
    chosen = select_by_llm(FakeLLM("1,2,3"), notes, make_sample(), top_k=3,
                           backfill=notes)
    assert len(chosen) == 3


def test_llm_mode_does_not_backfill():
    """The two modes have to differ, or the comparison measures nothing."""
    store = build_store(_notes_numbered(), EMB, SystemConfig())
    chosen = select_notes(store, make_sample(), EMB, mode="llm",
                          top_k=3, llm=FakeLLM("NONE"))
    assert chosen == []


def test_hybrid_mode_fills_what_the_model_leaves_empty():
    store = build_store(_notes_numbered(), EMB, SystemConfig())
    chosen = select_notes(store, make_sample(), EMB, mode="hybrid",
                          top_k=3, llm=FakeLLM("NONE"))
    assert len(chosen) == 3


def test_reranker_ignores_out_of_range_numbers():
    chosen = select_by_llm(FakeLLM("99, 2, 0"), _notes_numbered(3), make_sample())
    assert [n.text for n in chosen] == ["Note number 1."]


def test_reranker_deduplicates_repeated_numbers():
    chosen = select_by_llm(FakeLLM("2, 2, 2"), _notes_numbered(), make_sample())
    assert len(chosen) == 1


def test_reranker_handles_prose_around_the_numbers():
    chosen = select_by_llm(FakeLLM("I think notes 4 and 1 apply here."),
                           _notes_numbered(), make_sample(), top_k=3)
    assert [n.text for n in chosen] == ["Note number 3.", "Note number 0."]


def test_reranker_sees_the_trigger_and_every_note():
    llm = FakeLLM("1")
    sample = make_sample(trigger="A: I am completely overwhelmed.")
    select_by_llm(llm, _notes_numbered(4), sample)
    prompt = llm.prompts[0]
    assert sample.trigger in prompt
    assert all(f"Note number {i}." in prompt for i in range(4))


def test_reranker_with_no_notes_makes_no_call():
    llm = FakeLLM("1")
    assert select_by_llm(llm, [], make_sample()) == []
    assert llm.prompts == []


def test_llm_mode_requires_an_llm():
    store = build_store(_notes_numbered(), EMB, SystemConfig())
    with pytest.raises(ValueError, match="needs an llm"):
        select_notes(store, make_sample(), EMB, mode="llm")


def test_reranker_prompt_does_not_disclose_the_task():
    llm = FakeLLM("1")
    select_by_llm(llm, _notes_numbered(3), make_sample())
    wrapper = llm.prompts[0].lower()
    for word in ("memory", "recall", "benchmark", "evaluat", "cue", "evidence"):
        assert word not in wrapper


def test_every_selection_mode_gets_its_own_results_file():
    """
    Regression: the filename tag only special-cased "similar", so --select llm
    wrote to the carry run's file, found 100 rows already recorded and resumed
    past every single item. The run reported the carry results and spent
    nothing, which looked like success.
    """
    def tag_for(select, top_k, carry):
        return (f"carry{carry or 'all'}" if select == "carry"
                else f"{select}{top_k}")

    tags = {tag_for(*args) for args in
            [("carry", 3, 0), ("carry", 3, 5), ("similar", 3, 0),
             ("similar", 5, 0), ("llm", 3, 0), ("llm", 5, 0),
             ("hybrid", 3, 0), ("hybrid", 5, 0)]}
    assert len(tags) == 8


def test_hybrid_is_a_selectable_mode_in_the_driver():
    source = (pathlib.Path(__file__).parent.parent / "scripts" / "run_system.py").read_text()
    assert '"carry", "similar", "llm", "hybrid"' in source


def test_run_system_names_files_by_selection_mode():
    source = (pathlib.Path(__file__).parent.parent / "scripts" / "run_system.py").read_text()
    assert 'tag = f"{args.select}{args.top_k}"' in source


# ---------------------------------------------------------------------------
# strip_trigger -- bug 13: extraction was reading the query
# ---------------------------------------------------------------------------

from bapca.pipeline import strip_trigger


CONVO = (
    'Nate said, "Hey Joanna! Long time no see! Anything fun going on?"\n'
    'Joanna said, "I have been working on a project lately, pretty absorbing."\n'
    'Nate said, "No worries, Joanna. Hope you enjoy it!"\n'
)
TRIGGER = ("A: It is strange, I barely recognize the person who used to grab "
           "whatever sounded good without thinking about it")
TAIL = ('Joanna said, "It is strange, I barely recognize the person who used to '
        'grab whatever sounded good without thinking about it"')


def test_strip_trigger_removes_the_appended_query():
    kept = strip_trigger(CONVO + TAIL, TRIGGER)
    assert "barely recognize" not in kept
    assert "Long time no see" in kept
    assert "working on a project" in kept


def test_strip_trigger_is_a_noop_when_the_query_is_not_at_the_end():
    assert strip_trigger(CONVO, TRIGGER) == CONVO


def test_strip_trigger_is_a_noop_on_an_empty_or_tiny_trigger():
    assert strip_trigger(CONVO, "") == CONVO
    assert strip_trigger(CONVO, "A: ok thanks") == CONVO


def test_strip_trigger_never_eats_more_than_the_tail():
    """A pathological trigger must not be able to delete the conversation."""
    every_word = "A: " + " ".join(CONVO.split())
    kept = strip_trigger(CONVO + TAIL, every_word)
    assert len(kept.splitlines()) >= len(CONVO.splitlines()) - 4
    assert kept.strip()


def test_stripping_changes_only_the_final_window():
    """
    The cache argument for doing this cheaply: content-defined chunking picks
    boundaries from a line hash, so dropping trailing lines must leave every
    earlier window byte-identical and therefore still cached.
    """
    full = CONVO * 12 + TAIL
    before = segment(full, 12)
    after = segment(strip_trigger(full, TRIGGER), 12)
    assert [w.text for w in before[:-1]] == [w.text for w in after[:-1]]
    assert not any("barely recognize" in w.text for w in after)


def test_the_unstripped_prompt_really_does_contain_the_query():
    """Guards the premise: if this fails, strip_trigger is solving nothing."""
    assert "barely recognize" in (CONVO + TAIL)
    assert any("barely recognize" in w.text for w in segment(CONVO + TAIL, 12))


# ---------------------------------------------------------------------------
# EXTRACT_PROMPT v2 -- v1 discarded the exact category the benchmark tests
# ---------------------------------------------------------------------------

from bapca.pipeline import (EXTRACT_PROMPT, EXTRACT_PROMPTS,
                            EXTRACT_PROMPT_EVENTS, extract_notes)


def test_v1_really_does_tell_the_model_to_discard_events():
    """
    Guards the premise. Every gold cue is a one-off event plus the durable
    change it caused; v1 says to throw those away. If this assertion ever
    fails, v1 has been edited and the v2 rationale needs rewriting.
    """
    assert "one-off events" in EXTRACT_PROMPT
    assert "reveal nothing lasting" in EXTRACT_PROMPT


def test_events_prompt_asks_for_both_halves_and_a_name():
    p = EXTRACT_PROMPT_EVENTS
    assert "something happened" in p
    assert "differently because of it" in p
    assert "BOTH halves" in p
    assert "Name the person" in p
    assert "one-off events" not in p          # the v1 exclusion is gone
    assert "{window}" in p
    assert "NONE" in p


def test_both_prompts_keep_the_parseable_answer_format():
    for name, p in EXTRACT_PROMPTS.items():
        assert "state, goal, value" in p, name
        assert p.rstrip().endswith("Answer:"), name


def test_extract_notes_uses_the_prompt_it_is_given():
    seen = []

    class FakeLLM:
        def complete(self, prompt, **kw):
            seen.append(prompt)
            return "constraint: Jon switched to flameless candles after a fire scare."

    windows = [Window(0, "Jon said, \"there was nearly a fire\"", dated=False)]
    notes = extract_notes(FakeLLM(), windows, prompt=EXTRACT_PROMPT_EVENTS)
    assert len(notes) == 1
    assert "something happened" in seen[0]
    assert "one-off events" not in seen[0]


def test_extract_notes_defaults_to_v1_so_old_results_stay_reproducible():
    seen = []

    class FakeLLM:
        def complete(self, prompt, **kw):
            seen.append(prompt)
            return "NONE"

    extract_notes(FakeLLM(), [Window(0, "hello", dated=False)])
    assert "one-off events" in seen[0]


# ---------------------------------------------------------------------------
# --dry-run cost accounting -- it decides whether a run fits inside RPD 1,000
# ---------------------------------------------------------------------------

def test_dry_run_counts_the_rerank_calls_it_will_actually_make():
    """
    The re-ranker runs on the JUDGE provider (`select_notes(..., llm=jdg)`), so
    every item under `--select llm` or `hybrid` costs a third Groq call on top
    of extraction and judging. dry_run counted only two kinds and reported 669
    where the run makes 769 -- a 100-call error against a 1,000-call daily
    limit, i.e. the difference between "this fits today" and "it does not".
    """
    import importlib.util, io, contextlib, pathlib
    from bapca.dataset import Sample

    spec = importlib.util.spec_from_file_location(
        "run_system_under_test",
        pathlib.Path(__file__).parent.parent / "scripts" / "run_system.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    samples = [Sample(index=i, input_prompt="\n".join(f"A: line {i} {j}" for j in range(40)),
                      trigger="A: line 39", evidence="A: line 3", category="Cognitive")
               for i in range(5)]

    def groq_line(select):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            module.dry_run(samples, SystemConfig(), strip=False, select=select)
        return next(l for l in buf.getvalue().splitlines() if "Groq" in l)

    llm_line = groq_line("llm")
    assert "re-rank 5" in llm_line
    # extraction + 5 re-rank + 5 judge, and the total must exceed the
    # similarity mode's, which makes no re-rank call at all.
    assert "re-rank 0" in groq_line("similar")


# ---------------------------------------------------------------------------
# compare_runs.py -- paired reading of a PARTIAL run against a finished one
# ---------------------------------------------------------------------------

def _result_row(index, label, cue, notes=30):
    return dict(condition="system_llm", sample_index=index, prompt_tokens=130,
                prediction="p", label=label, reason="r", notes_extracted=notes,
                notes_carried=3, notes_pruned=0, cue_window=1, cue_extracted=True,
                cue_selected=cue, carried=["a", "b", "c"], evidence_found=cue,
                strip_trigger=True, extract="v1", all_notes=["[state] n"])


def _run_compare(tmp_path, rows_a, rows_b):
    import json as _json, subprocess, sys as _sys, pathlib as _p
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(_json.dumps(rows_a)); b.write_text(_json.dumps(rows_b))
    script = _p.Path(__file__).parent.parent / "scripts" / "compare_runs.py"
    done = subprocess.run([_sys.executable, str(script), str(a), str(b)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_compare_runs_pairs_only_the_shared_items(tmp_path):
    """
    A partial run must be readable against a finished one the day it stops.
    The comparison has to be on the intersection: this project already
    retracted a claim built on comparing different item sets (4.7 -> 4.8).
    """
    full = [_result_row(1989 + i, "correct" if i < 4 else "wrong", i < 4)
            for i in range(20)]
    partial = [_result_row(1989 + i, "correct", True) for i in range(5)]
    out = _run_compare(tmp_path, full, partial)
    assert "items in BOTH: 5" in out
    assert "only in A: 15" in out
    # scored on the 5 shared items only: A got 4 of them, B all 5.
    assert "(4/5)" in out and "(5/5)" in out


def test_compare_runs_ignores_unparsed_rows(tmp_path):
    """#2374 came back `unparsed` and every comparison must drop it silently."""
    a = [_result_row(1, "correct", True), _result_row(2, "unparsed", False)]
    b = [_result_row(1, "wrong", False), _result_row(2, "unparsed", False)]
    out = _run_compare(tmp_path, a, b)
    assert "items in BOTH: 1" in out


def test_dry_run_says_how_many_DAYS_when_the_token_budget_needs_more_than_one():
    """
    Bug 16. The run died at item 10 of 100 on Groq's tokens-per-day limit,
    which `dry_run` did not model -- it counted requests, and requests were
    never the binding constraint.
    """
    import importlib.util, io, contextlib, pathlib
    from bapca.dataset import Sample

    spec = importlib.util.spec_from_file_location(
        "run_system_days", pathlib.Path(__file__).parent.parent / "scripts" / "run_system.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.GROQ_TPD == 200_000
    assert module.TOKENS_PER_EXTRACTION > 700, "the 460 constant under-priced the day"

    many = [Sample(index=i, input_prompt="\n".join(f"A: turn {i} {j}" for j in range(400)),
                   trigger="A: turn 399", evidence="A: turn 3", category="Cognitive")
            for i in range(60)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        module.dry_run(many, SystemConfig(), strip=False, select="llm")
    text = buf.getvalue()
    assert "TOKENS PER DAY" in text
    assert "DAYS, not one" in text


def test_the_E_vs_baseline_line_names_the_winning_condition():
    """
    McNemar's verdict() labels its two lists "A" and "B". The experimental
    conditions are also named A..E. `mcnemar(other, ours)` therefore printed
    "B better" to mean OUR system won, which reads as "condition B, full
    context, won" -- the opposite. Caught on 4 Sep when E scored 65.0% against
    full context's 52.0% and the line said "B better".
    """
    import pathlib
    source = (pathlib.Path(__file__).parent.parent / "scripts" / "run_system.py").read_text()
    assert 'winner = "E (ours)" if test.only_b > test.only_a else label' in source
    assert "test.verdict()" not in source, "the ambiguous verdict string is back"


# ---------------------------------------------------------------------------
# distractor_cost.py -- the paired A-vs-E test, on any run, not a hardcoded one
# ---------------------------------------------------------------------------

def _run_script(name, tmp_path, run_rows, pilot_rows):
    import json as _json, subprocess, sys as _sys, pathlib as _p
    run, pilot = tmp_path / "run.json", tmp_path / "pilot.json"
    run.write_text(_json.dumps(run_rows)); pilot.write_text(_json.dumps(pilot_rows))
    script = _p.Path(__file__).parent.parent / "scripts" / name
    done = subprocess.run([_sys.executable, str(script), str(run), str(pilot)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_distractor_cost_pairs_only_the_undelivered_items(tmp_path):
    """
    HANDOFF 4.8 retracted a -12 point claim built on comparing condition A over
    all 100 items against E over the ~68 where selection failed. The failed
    items are the hard ones. Only the paired restriction is admissible.
    """
    run, pilot = [], []
    for i in range(40):
        delivered = i < 25                      # 15 undelivered items
        run.append(_result_row(i, "correct" if delivered else "wrong", delivered))
        pilot.append(dict(condition="no_memory", sample_index=i,
                          label="correct" if i % 3 == 0 else "wrong"))
        pilot.append(dict(condition="full_context", sample_index=i, label="correct"))
    out = _run_script("distractor_cost.py", tmp_path, run, pilot)
    assert "items in both: 40" in out
    assert "15 items where the cue was NOT delivered" in out
    assert "full_context" not in out.split("The real test")[1]  # only condition A


def test_distractor_cost_names_the_files_it_read(tmp_path):
    """Bug 12 and bug 18: a script that reads a file you did not intend must
    say which file it read, at the top, before any number."""
    run = [_result_row(i, "correct", True) for i in range(12)]
    pilot = [dict(condition="no_memory", sample_index=i, label="wrong") for i in range(12)]
    out = _run_script("distractor_cost.py", tmp_path, run, pilot)
    assert "Files actually read" in out
    assert out.index("Files actually read") < out.index("items in both")


def test_the_analysis_scripts_no_longer_print_hardcoded_v1_numbers():
    """
    extraction_ceiling.py printed the literals '91%' and '38%' as "reported by
    the run" beside a run that reported 99% and 72%. true_recall.py ignored
    sys.argv and always read three hardcoded files.
    """
    import pathlib
    base = pathlib.Path(__file__).parent.parent / "scripts"
    ceiling = (base / "extraction_ceiling.py").read_text()
    assert 'reported by the run as cue_extracted (window) 91%' not in ceiling
    assert "this run's cue_extracted" in ceiling
    recall = (base / "true_recall.py").read_text()
    assert "if len(sys.argv) > 1:" in recall
    assert "Files actually read" in recall


# ---------------------------------------------------------------------------
# target_speaker -- who is "A"?  (4 Sep)
# ---------------------------------------------------------------------------

from bapca.pipeline import target_speaker

_CONV = '\n'.join([
    'Nate said, "Yeah, turtles are like zen masters! They remind me to slow down."',
    'Joanna said, "Yea, no worries! It was great catching up. Take it easy!"',
    'Joanna said, "It is strange, I barely recognize the person who used to grab '
    'whatever sounded good without thinking about labels."',
])
_TRIGGER = ('A: It is strange, I barely recognize the person who used to grab '
            'whatever sounded good without thinking about labels.')


def test_the_anonymous_A_resolves_to_the_speaker_of_the_matching_final_line():
    """
    Four of the n=100 selection failures carried notes about the OTHER person.
    The conversation is written `Name said, "..."` (NOT `Name:`) and the
    trigger's own text is the final line, attributed by name -- so `A` is
    recoverable from input_prompt + trigger alone.
    """
    assert target_speaker(_CONV, _TRIGGER) == "Joanna"


def test_target_speaker_never_reads_the_gold_evidence_field():
    """
    The whole claim that this is not an oracle rests on the signature: it takes
    the conversation and the query, both given at inference. If it ever grows
    an `evidence` argument, the result stops being reportable.
    """
    import inspect
    params = list(inspect.signature(target_speaker).parameters)
    assert params == ["prompt", "trigger"], params


def test_target_speaker_declines_rather_than_guessing():
    assert target_speaker(_CONV, "A: entirely different words about penguins and ice") is None
    assert target_speaker(_CONV, "A: ok") is None          # too short to match safely
    assert target_speaker("", _TRIGGER) is None
    assert target_speaker(_CONV, "") is None


def test_target_speaker_handles_the_name_colon_format_too():
    """The evidence field uses `Name：` with a FULLWIDTH colon; the conversation
    uses `Name said,`. Guessing this data's format has gone wrong five times --
    the resolver must not match a line it does not understand."""
    colon_style = 'Joanna：It is strange, I barely recognize the person who used to grab.'
    assert target_speaker(colon_style, _TRIGGER) is None


def test_target_speaker_only_scans_the_tail():
    """A stray early line that happens to echo the query must not win."""
    long_conv = "\n".join(['Mallory said, "It is strange, I barely recognize the person '
                           'who used to grab whatever sounded good without thinking."']
                          + ['Nate said, "filler line here."'] * 10)
    assert target_speaker(long_conv, _TRIGGER) is None


# ---------------------------------------------------------------------------
# person-aware re-ranking (4 Sep)
# ---------------------------------------------------------------------------

from bapca.pipeline import (RERANK_PROMPT, RERANK_PROMPTS, RERANK_PROMPT_PERSON,
                            _rerank_prompt)


def _person_sample():
    return make_sample(input_prompt=_CONV, trigger=_TRIGGER)


def test_person_prompt_names_the_speaker_it_is_replying_to():
    built = _rerank_prompt("person", "1. [state] x", _person_sample(), 3)
    assert "Joanna" in built
    assert "replying to Joanna" in built


def test_person_prompt_falls_back_to_v1_when_the_speaker_is_unknown():
    """
    A resolver that returns None must not put the string 'None' into the
    prompt. Degrade to the old wording, which is a known quantity.
    """
    unknown = make_sample(input_prompt="no speaker here at all",
                          trigger="A: entirely different words about penguins and ice")
    built = _rerank_prompt("person", "1. [state] x", unknown, 3)
    assert "None" not in built
    assert built == RERANK_PROMPT.format(notes="1. [state] x",
                                         trigger=unknown.trigger, top_k=3)


def test_v1_rerank_wording_is_unchanged_so_old_results_stay_reproducible():
    built = _rerank_prompt("v1", "1. [state] x", _person_sample(), 3)
    assert "Joanna" not in built
    assert built.startswith("Here is what you know about this person:")


def test_person_prompt_prefers_rather_than_filters():
    """
    A hard filter would drop every note that names nobody and would be
    unrecoverable whenever the resolver is wrong. The wording must leave the
    model able to take another person's note when it genuinely bears.
    """
    p = RERANK_PROMPT_PERSON
    assert "prefer a note about {speaker}" in p
    assert "unless another" in p
    # no absolute language that would turn a preference into a filter
    body = p.lower().split("answer with exactly")[0]
    for forbidden in ("only notes about", "ignore every", "never choose a note about"):
        assert forbidden not in body, forbidden
    for name, text in RERANK_PROMPTS.items():
        assert "{top_k}" in text and "{notes}" in text, name


def test_select_by_llm_passes_the_variant_through():
    seen = []

    class FakeLLM:
        def complete(self, prompt, **kw):
            seen.append(prompt)
            return "1, 2, 3"

    select_by_llm(FakeLLM(), _notes_numbered(3), _person_sample(), 3, rerank="person")
    assert "Joanna" in seen[0]


def test_rerank_variant_gets_its_own_results_filename():
    """Bug 12: every configuration needs a distinct file."""
    import pathlib
    source = (pathlib.Path(__file__).parent.parent / "scripts" / "run_system.py").read_text()
    assert 'if args.rerank != "v1":' in source
    assert 'tag += f"_{args.rerank}"' in source
    assert "rerank=args.rerank" in source


def test_the_person_fix_never_reaches_for_the_gold_field():
    """The whole result is unreportable if this stops being true."""
    import inspect
    from bapca import pipeline
    src = inspect.getsource(pipeline._rerank_prompt)
    assert "evidence" not in src


# ---- 27 Sep: non-Cognitive runs (the extraction-transfer test) -------------

def _load_run_system():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "run_system_27sep", pathlib.Path(__file__).parent.parent / "scripts" / "run_system.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_command_keeps_every_flag_that_names_the_results_file():
    """The quota message used to drop --category and --seed, so the printed
    'identical command' resumed a different file."""
    import argparse
    rs = _load_run_system()
    args = argparse.Namespace(n=100, seed=7, carry=0, window_turns=12,
                              days_per_window=1.0, select="llm", top_k=3,
                              strip_trigger=True, extract="events", rerank="v1",
                              category="single-hop")
    cmd = rs.resume_command(args)
    for flag in ("--n 100", "--seed 7", "--select llm", "--top-k 3",
                 "--strip-trigger", "--extract events", "--category single-hop"):
        assert flag in cmd, flag
    assert "--fresh" not in cmd


def test_resume_command_is_unchanged_for_the_headline_configuration():
    import argparse
    rs = _load_run_system()
    args = argparse.Namespace(n=401, seed=42, carry=0, window_turns=12,
                              days_per_window=1.0, select="llm", top_k=3,
                              strip_trigger=True, extract="events", rerank="v1",
                              category="Cognitive")
    assert rs.resume_command(args) == (
        "!python scripts/run_system.py --n 401 --select llm --top-k 3 "
        "--strip-trigger --extract events")


def test_extraction_estimate_reproduces_the_measured_778_and_scales():
    rs = _load_run_system()
    cognitive_like = ["x" * (4 * rs.MEASURED_WINDOW_TOKENS)]
    assert rs.tokens_per_extraction(cognitive_like) == rs.TOKENS_PER_EXTRACTION
    longer = ["x" * (4 * 836)]
    assert rs.tokens_per_extraction(longer) > rs.TOKENS_PER_EXTRACTION + 250
    assert rs.tokens_per_extraction([]) == rs.TOKENS_PER_EXTRACTION


def test_non_cognitive_runs_never_print_the_cognitive_baselines():
    import pathlib
    source = (pathlib.Path(__file__).parent.parent / "scripts" / "run_system.py").read_text(
        encoding="utf-8")
    assert 'if args.category != "Cognitive":' in source
    assert "__no_pilot_for_this_category__" in source


def test_compare_runs_names_the_winning_file_not_a_letter():
    import importlib.util, pathlib
    from bapca.stats import mcnemar
    spec = importlib.util.spec_from_file_location(
        "compare_runs_27sep", pathlib.Path(__file__).parent.parent / "scripts" / "compare_runs.py")
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    a = [False] * 20 + [True] * 5
    b = [True] * 20 + [True] * 5
    words = cr.named_verdict(mcnemar(a, b), "v1.json", "events.json")
    assert "events.json" in words and " B " not in words and "A better" not in words


def test_dry_run_never_counts_other_runs_items_and_never_goes_negative(tmp_path):
    """27 Sep: a 200-item single-hop dry run counted the 401 Cognitive items as
    already done and printed -201 calls and -4,210 tokens."""
    import io, contextlib, json as _json
    from bapca.dataset import Sample
    from bapca.pipeline import SystemConfig
    rs = _load_run_system()
    rs.RESULTS_DIR = tmp_path
    other = [dict(sample_index=1000 + i, extract="events", strip_trigger=True)
             for i in range(401)]
    (tmp_path / "system_n401_seed42_llm3_d1_notrig_events.json").write_text(_json.dumps(other))
    samples = [Sample(index=i, input_prompt="\n".join(f"A: line {i} {j}" for j in range(40)),
                      trigger="A: line 39", evidence="A: line 3", category="single-hop")
               for i in range(5)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rs.dry_run(samples, SystemConfig(), strip=True, select="llm", extract="events",
                   out_path=tmp_path / "system_n5_seed42_llm3_d1_notrig_events_singlehop.json")
    text = buf.getvalue()
    groq = next(l for l in text.splitlines() if "Groq   (" in l)
    assert "re-rank 5" in groq and "judge 5" in groq
    import re as _re
    assert not _re.search(r"-\d", groq), groq
    assert "already run" not in text


def test_dry_run_counts_items_recorded_in_this_runs_own_file(tmp_path):
    import io, contextlib, json as _json
    from bapca.dataset import Sample
    from bapca.pipeline import SystemConfig
    rs = _load_run_system()
    rs.RESULTS_DIR = tmp_path
    out = tmp_path / "mine.json"
    out.write_text(_json.dumps([dict(sample_index=i, extract="events", strip_trigger=True)
                                for i in range(2)]))
    samples = [Sample(index=i, input_prompt="\n".join(f"A: line {i} {j}" for j in range(40)),
                      trigger="A: line 39", evidence="A: line 3", category="single-hop")
               for i in range(5)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rs.dry_run(samples, SystemConfig(), strip=True, select="llm", extract="events",
                   out_path=out)
    groq = next(l for l in buf.getvalue().splitlines() if "Groq   (" in l)
    assert "re-rank 3" in groq and "judge 3" in groq
