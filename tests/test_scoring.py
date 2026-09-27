"""
Regression tests.

The first three lock the Review 1 paper's published numbers to the code, so if
anyone changes the scoring we find out immediately rather than in a review.
The rest cover the new behaviour (type routing, non-destructive pruning,
ablation switches).

Run:  python -m pytest tests/ -q
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from bapca.embeddings import HashEmbedder
from bapca.memory import DEFAULT_LAMBDA, Episode, MemoryNode, MemoryType, cosine
from bapca.store import EpisodicStore, RetrievalConfig

START = datetime(2026, 1, 1)
EMB = HashEmbedder()


def _node(text, weight, mem_type=MemoryType.FACT, freq=1, last=START):
    return MemoryNode(
        id=1,
        text=text,
        embedding=EMB.encode_one(text),
        weight=weight,
        created=START,
        last_accessed=last,
        mem_type=mem_type,
        frequency=freq,
    )


# --------------------------------------------------------------------------
# Review 1 paper values. Section VI-B reports 0.23 / 0.18 / 0.62 at day 45
# with lambda = 0.03. These must not drift.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "weight,freq,last_day,expected",
    [
        (0.9, 1, 0, 0.2334),   # identity fact, 45 days old  -> survives tau=0.20
        (0.6, 1, 5, 0.1807),   # badminton, 40 days old      -> pruned
        (0.7, 2, 18, 0.6227),  # perfume, restated on day 18 -> strongly retained
    ],
)
def test_paper_scores_reproduce(weight, freq, last_day, expected):
    node = _node("x", weight, freq=freq, last=START + timedelta(days=last_day))
    node.mem_type = MemoryType.FACT
    score = node.score(START + timedelta(days=45), per_type_decay=False)
    assert score == pytest.approx(expected, abs=5e-4)


def test_fact_lambda_matches_paper():
    assert DEFAULT_LAMBDA == 0.03


# --------------------------------------------------------------------------
# Scoring behaviour
# --------------------------------------------------------------------------

def test_score_decays_monotonically():
    node = _node("x", 0.8)
    scores = [node.score(START + timedelta(days=d)) for d in range(0, 60, 10)]
    assert scores == sorted(scores, reverse=True)


def test_frequency_switch_removes_c_term():
    node = _node("x", 0.5, freq=4)
    now = START + timedelta(days=10)
    assert node.score(now, use_frequency=False) == pytest.approx(
        node.score(now, use_frequency=True) / 4
    )


def test_standing_types_decay_slower_than_facts():
    now = START + timedelta(days=60)
    fact = _node("x", 0.5, MemoryType.FACT)
    goal = _node("x", 0.5, MemoryType.GOAL)
    assert goal.score(now) > fact.score(now)


def test_reinforce_resets_clock_and_bumps_frequency():
    node = _node("x", 0.5)
    node.reinforce(START + timedelta(days=30))
    assert node.frequency == 2
    assert node.last_accessed == START + timedelta(days=30)


def test_weight_stays_in_range():
    node = _node("x", 0.98)
    node.reinforce(START, weight_bump=0.5)
    assert node.weight == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Store behaviour
# --------------------------------------------------------------------------

def _store(**kwargs):
    return EpisodicStore(RetrievalConfig(**kwargs))


def test_pattern_completion_reinforces_instead_of_duplicating():
    store = _store()
    vec = EMB.encode_one("I bought the Skinn Amalfi Bleu perfume.")
    store.write("I bought the Skinn Amalfi Bleu perfume.", vec, when=START, weight=0.7)
    store.write("I really like that perfume I bought.", vec, when=START + timedelta(days=4), weight=0.7)
    assert len(store.nodes) == 1
    assert store.nodes[0].frequency == 2
    # We keep the ORIGINAL text -- the Review 1 notebook logged the new one.
    assert store.nodes[0].text.startswith("I bought")


def test_pruning_archives_rather_than_deletes():
    store = _store()
    store.write("transient chatter", EMB.encode_one("transient chatter"),
                when=START, weight=0.3, mem_type=MemoryType.FACT)
    dropped = store.prune(START + timedelta(days=200))
    assert len(dropped) == 1
    assert store.nodes == []
    assert len(store.archive) == 1  # auditable, not gone


def test_no_decay_switch_disables_pruning():
    store = _store(use_decay=False)
    store.write("x", EMB.encode_one("x"), when=START, weight=0.2)
    assert store.prune(START + timedelta(days=10_000)) == []
    assert len(store.nodes) == 1


def test_standing_memory_is_retrieved_without_similarity():
    """The core claim: a constraint reaches the context even when the query
    shares nothing with it. This is what flat RAG structurally cannot do."""
    store = _store()
    store.write(
        "I'm preparing for an important exam and want to minimise distractions.",
        EMB.encode_one("I'm preparing for an important exam and want to minimise distractions."),
        when=START, weight=0.8, mem_type=MemoryType.GOAL,
    )
    query = "Should I start watching that new TV series everyone is talking about?"
    got = store.retrieve(EMB.encode_one(query), START + timedelta(days=20))

    assert cosine(EMB.encode_one(query), store.nodes[0].embedding) < 0.3  # no overlap
    assert len(got.standing_hits) == 1                                    # retrieved anyway
    assert "exam" in got.as_context()


def test_type_routing_off_makes_constraints_similarity_gated():
    """Ablation sanity check: with routing off we degrade to flat RAG, and the
    unrelated constraint is correctly NOT retrieved."""
    store = _store(use_type_routing=False)
    store.write(
        "I'm preparing for an important exam and want to minimise distractions.",
        EMB.encode_one("I'm preparing for an important exam and want to minimise distractions."),
        when=START, weight=0.8, mem_type=MemoryType.GOAL,
    )
    got = store.retrieve(
        EMB.encode_one("Should I start watching that new TV series?"),
        START + timedelta(days=20),
    )
    assert got.standing_hits == []
    assert got.similarity_hits == []


def test_standing_budget_is_respected():
    store = _store(standing_budget=3)
    for i in range(10):
        text = f"constraint number {i}"
        store.write(text, EMB.encode_one(text), when=START, weight=0.9,
                    mem_type=MemoryType.CONSTRAINT)
    got = store.retrieve(EMB.encode_one("anything"), START + timedelta(days=1))
    assert len(got.standing_hits) == 3


def test_correctness_gating_only_rewards_success():
    store = _store()
    node = store.write("x", EMB.encode_one("x"), when=START, weight=0.5)
    store.reinforce_used([node], START + timedelta(days=1), correct=False)
    assert node.frequency == 1
    store.reinforce_used([node], START + timedelta(days=1), correct=True)
    assert node.frequency == 2


def test_token_estimate_grows_with_context():
    store = _store()
    empty = store.retrieve(EMB.encode_one("q"), START).token_estimate()
    store.write("a fairly long standing constraint about diet and health",
                EMB.encode_one("a fairly long standing constraint about diet and health"),
                when=START, weight=0.9, mem_type=MemoryType.VALUE)
    filled = store.retrieve(EMB.encode_one("q"), START).token_estimate()
    assert filled > empty == 0


def test_episode_tuple_roundtrips():
    node = _node("x", 0.5)
    node.episode = Episode(time="Tuesday", space="Lisbon", entity="Alice",
                           content="badminton tournament", detail="lost in the semis")
    assert not node.episode.is_empty()
    assert node.to_dict()["episode"]["space"] == "Lisbon"
    assert "embedding" not in node.to_dict()
