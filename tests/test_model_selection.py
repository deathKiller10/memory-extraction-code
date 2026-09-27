"""
Tests for model ranking, reasoning suppression, and cache edge cases.

Every test here exists because of a real failure on a real Colab run:

  1. The judge picked `gemini-2.5-flash-lite`, which the list endpoint still
     advertised but which 404s for new API keys -- alphabetical sorting let the
     older version win.
  2. The generator picked a reasoning model, which returned an empty string, and
     the smoke test reported "cache HIT (good)" because two empties matched.
  3. The generator then returned `"<think>\\nHere's a thinking process:..."` as
     its answer -- the chain of thought leaked into the content, and judging it
     would have scored the monologue instead of the response.
  4. `qwen3.6-27b` outranked `qwen3.8-27b`, because the version parser only
     understood `gemini-3.5-` style ids and scored both glued names as 0.0.

None of these need network.
"""

import pytest

from bapca.llm import (AVOID, GEMINI, GROQ, ROLES, LLM, Limits, PromptTooLarge,
                       Throttle, _version_of, estimate_tokens, reasoning_kwargs,
                       strip_reasoning)


# --------------------------------------------------------------------------
# Version parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("gemini-3.5-flash-lite", 3.5),
        ("gemini-2.5-flash-lite", 2.5),
        ("models/gemini-3.7-flash", 3.7),
        ("qwen/qwen3.6-27b", 3.6),     # version glued to the family name
        ("qwen/qwen3.8-27b", 3.8),
        ("llama-3.3-70b-versatile", 3.3),
        ("llama-4-scout", 4.0),        # integer version, separated
        ("allam-2-7b", 2.0),
        ("openai/gpt-oss-120b", 0.0),  # 120 is a parameter count, not a version
        ("some-model", 0.0),
    ],
)
def test_version_extraction(model_id, expected):
    assert _version_of(model_id) == expected


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------

def _rank(role, available):
    """Rank without constructing a client (no network, no API key)."""
    client = LLM.__new__(LLM)
    client.role = role
    client._prefs = ROLES[role][1]
    return LLM.rank_candidates(client, available)


def test_newer_version_beats_older_in_the_same_family():
    ranked = _rank("judge", ["models/gemini-2.5-flash-lite", "models/gemini-3.5-flash-lite"])
    assert ranked[0] == "models/gemini-3.5-flash-lite"


def test_glued_version_names_are_ordered_correctly():
    """qwen3.8 must beat qwen3.6 -- both scored 0.0 before the parser was fixed."""
    ranked = _rank("judge", ["qwen/qwen3.6-27b", "qwen/qwen3.8-27b"])
    assert ranked[0] == "qwen/qwen3.8-27b"


def test_flash_preference_does_not_swallow_flash_lite():
    """Plain 'flash' is a substring of 'flash-lite'; exclusions keep them apart.
    The generator wants flash-lite for its RPD 1,000."""
    available = ["models/gemini-3.5-flash", "models/gemini-3.5-flash-lite"]
    assert _rank("generator", available)[0] == "models/gemini-3.5-flash-lite"


def test_secondary_judge_picks_the_larger_gpt_oss_first():
    """The agreement study is worthless if the second judge is much weaker."""
    ranked = _rank("judge_secondary", ["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
    assert ranked[0] == "openai/gpt-oss-120b"


def test_judge_prefers_instruct_over_reasoning_models():
    ranked = _rank("judge", ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"])
    assert ranked[0] == "llama-3.3-70b-versatile"


def test_reasoning_model_still_available_as_a_fallback():
    """Groq's free tier has no instruct models left -- we must not end up empty."""
    ranked = _rank("judge", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"])
    assert ranked[0] == "qwen/qwen3.8-27b"
    assert "openai/gpt-oss-120b" in ranked


def test_non_chat_models_are_excluded():
    available = [
        "whisper-large-v3",
        "text-embedding-004",
        "meta-llama/llama-guard-4-12b",
        "canopylabs/orpheus-v1-english",
        "qwen/qwen3.8-27b",
    ]
    assert _rank("judge", available) == ["qwen/qwen3.8-27b"]


def test_unknown_models_are_kept_last_not_dropped():
    ranked = _rank("judge", ["brand-new-model-9000", "qwen/qwen3.8-27b"])
    assert ranked[0] == "qwen/qwen3.8-27b"
    assert "brand-new-model-9000" in ranked


def test_avoid_list_and_preferences_are_lowercase():
    assert all(token == token.lower() for token in AVOID)
    for _provider, prefs, _limits in ROLES.values():
        for want, forbid in prefs:
            assert want == want.lower()
            assert all(f == f.lower() for f in forbid)


# --------------------------------------------------------------------------
# Reasoning suppression -- the bug that would have ruined the results
# --------------------------------------------------------------------------

def test_closed_think_block_is_removed():
    raw = "<think>\nThe user wants a greeting.\n</think>\nHello, how can I help?"
    assert strip_reasoning(raw) == "Hello, how can I help?"


def test_truncated_think_block_yields_nothing_rather_than_thoughts():
    """The exact failure: budget ran out mid-thought, so there IS no answer.
    Returning '' makes complete() retry with more room instead of scoring the
    monologue."""
    raw = "<think>\nHere's a thinking process:\n\n1.  **Analyze User Input"
    assert strip_reasoning(raw) == ""


def test_multiple_and_uppercase_think_blocks_are_removed():
    raw = "<THINK>a</THINK>Answer one. <think>b</think>Answer two."
    assert strip_reasoning(raw) == "Answer one. Answer two."


def test_ordinary_text_is_untouched():
    assert strip_reasoning("  A plain answer.  ") == "A plain answer."


def test_answer_mentioning_thinking_is_not_mangled():
    text = "I think you should rest."
    assert strip_reasoning(text) == text


@pytest.mark.parametrize(
    "model,expected_key",
    [
        ("qwen/qwen3.8-27b", "reasoning_format"),    # Qwen supports reasoning_format
        ("openai/gpt-oss-120b", "include_reasoning"),  # GPT-OSS does not; it needs this
    ],
)
def test_groq_reasoning_switches_match_the_model_family(model, expected_key):
    kwargs = reasoning_kwargs("groq", model)
    assert expected_key in kwargs


def test_gpt_oss_never_gets_reasoning_format():
    """Groq's docs: the two parameters are mutually exclusive, and GPT-OSS
    rejects reasoning_format outright."""
    assert "reasoning_format" not in reasoning_kwargs("groq", "openai/gpt-oss-20b")


def test_gemini_gets_no_groq_specific_parameters():
    assert reasoning_kwargs("gemini", "models/gemini-3.5-flash-lite") == {}


# --------------------------------------------------------------------------
# Rate limits and role assignment
#
# The reason this section exists: we originally put the generator on Groq. The
# LoCoMo-Plus Cognitive prompts have a median of 21,201 tokens and Groq's free
# tier caps at 8,000 tokens per minute, so no full-context call could ever have
# succeeded. Retries would have burned the day's quota discovering that.
# --------------------------------------------------------------------------

LOCOMO_MEDIAN_PROMPT_TOKENS = 21_201  # measured from unified_input_samples_v2.json


def test_generator_can_actually_swallow_a_locomo_prompt():
    _provider, _prefs, limits = ROLES["generator"]
    assert limits.tpm >= LOCOMO_MEDIAN_PROMPT_TOKENS * 1.2


def test_generator_and_judge_are_different_vendors():
    """Self-preference bias defence: judging must not be same-vendor."""
    assert ROLES["generator"][0].name != ROLES["judge"][0].name


def test_secondary_judge_differs_from_primary_judge():
    """
    They now share a vendor (both Groq) because Gemini's flash tier allows only
    20 requests a day. What must still differ is the MODEL FAMILY -- two Qwen
    models agreeing would measure nothing.
    """
    primary = [want for want, _ in ROLES["judge"][1]]
    secondary = [want for want, _ in ROLES["judge_secondary"][1]]
    assert secondary[0] not in primary[:2]


def test_generator_has_the_daily_headroom_for_the_study():
    """401 Cognitive items across several conditions needs real RPD."""
    assert ROLES["generator"][2].rpd >= 1_000


def test_oversized_prompt_fails_fast_instead_of_retrying():
    throttle = Throttle(Limits(rpm=30, tpm=8_000, rpd=1_000))
    with pytest.raises(PromptTooLarge) as excinfo:
        throttle.wait_for(LOCOMO_MEDIAN_PROMPT_TOKENS)
    assert "exceeds" in str(excinfo.value)
    assert throttle.day_count == 0     # nothing was spent finding out


def test_throttle_admits_requests_inside_the_budget():
    throttle = Throttle(Limits(rpm=30, tpm=250_000, rpd=1_000))
    for _ in range(5):
        assert throttle.wait_for(21_000) == 0.0   # no sleeping needed
    assert throttle.day_count == 5


def test_throttle_counts_tokens_not_just_requests():
    throttle = Throttle(Limits(rpm=100, tpm=10_000, rpd=1_000))
    throttle.wait_for(9_000)
    now = __import__("time").monotonic()
    used = sum(n for _, n in throttle._events)
    assert used == 9_000 and len(throttle._events) == 1


def test_fits_rejects_a_prompt_over_the_window():
    client = LLM.__new__(LLM)
    client.limits = Limits(rpm=30, tpm=8_000, rpd=1_000)
    assert not LLM.fits(client, "x" * (21_201 * 4))
    assert LLM.fits(client, "x" * 400)


def test_token_estimate_is_in_the_right_ballpark():
    assert estimate_tokens("x" * 4_000) == 1_000


# --------------------------------------------------------------------------
# Rate limiting, learned the hard way
#
# A 429 from Gemini reported quotaValue '5' for a model our configured limits
# called 10, and its stated 40-second retry window outlasted an exponential
# backoff that summed to ~30s. Both are now handled from the error itself.
# --------------------------------------------------------------------------

def test_retry_delay_is_read_from_the_provider_message():
    from bapca.llm import retry_delay_from
    assert retry_delay_from("Please retry in 40.570678389s.") == pytest.approx(40.5706, abs=1e-3)
    assert retry_delay_from("'retryDelay': '40s'") == 40.0
    assert retry_delay_from('"retryDelay": "7s"') == 7.0


def test_retry_delay_absent_returns_none():
    from bapca.llm import retry_delay_from
    assert retry_delay_from("some other error") is None
    assert retry_delay_from("") is None


def test_throttle_halves_its_rate_after_a_429():
    """Published limits have been wrong twice; the throttle learns instead."""
    t = Throttle(Limits(rpm=10, tpm=250_000, rpd=250))
    assert t.effective_rpm == 10
    t.observed_rate_limit()
    assert t.effective_rpm == 5
    t.observed_rate_limit()
    assert t.effective_rpm == 2
    assert t.throttle_events == 2


def test_throttle_never_drops_below_one_request_per_minute():
    t = Throttle(Limits(rpm=2, tpm=250_000, rpd=250))
    for _ in range(10):
        t.observed_rate_limit()
    assert t.effective_rpm == 1


def test_throttle_paces_on_the_learned_rate_not_the_configured_one():
    t = Throttle(Limits(rpm=4, tpm=250_000, rpd=250))
    t.observed_rate_limit()          # effective now 2
    assert t.wait_for(10) == 0.0
    assert t.wait_for(10) == 0.0
    assert len(t._events) == 2       # a third would have to wait


def test_secondary_judge_has_a_usable_daily_budget():
    """
    Gemini reported 20 requests per DAY for gemini-3.6-flash, which makes a
    100-item agreement study impossible. Whatever model hosts this role must
    have room for one.
    """
    assert ROLES["judge_secondary"][2].rpd >= 200


def test_secondary_judge_is_a_different_family_from_the_primary():
    """Both judges on Groq is fine; both being Qwen would not be a check at all."""
    primary_prefs = [want for want, _ in ROLES["judge"][1]]
    secondary_prefs = [want for want, _ in ROLES["judge_secondary"][1]]
    assert secondary_prefs[0] != primary_prefs[0]
    assert "gpt-oss" in secondary_prefs[0]


def test_secondary_judge_is_still_not_the_generator_vendor():
    """The self-preference defence depends on this."""
    assert ROLES["judge_secondary"][0].name != ROLES["generator"][0].name


def test_secondary_judge_has_the_daily_headroom_for_an_agreement_study():
    """gemini-3.6-flash allowed 20 requests a day, which made this impossible."""
    assert ROLES["judge_secondary"][2].rpd >= 200


def test_daily_quota_raises_rather_than_retrying():
    from bapca.llm import DailyQuotaExhausted
    t = Throttle(Limits(rpm=30, tpm=8_000, rpd=2))
    t.wait_for(10)
    t.wait_for(10)
    with pytest.raises(DailyQuotaExhausted, match="resets"):
        t.wait_for(10)


def test_throttle_learns_the_real_daily_cap_from_the_error():
    t = Throttle(Limits(rpm=5, tpm=250_000, rpd=250))
    t.observed_daily_limit(20)
    assert t.limits.rpd == 20
    assert t.limits.rpm == 5      # the per-minute limit is a separate quota


# ---------------------------------------------------------------------------
# Bug 16 -- tokens per day, the limit the cost model did not know existed
# ---------------------------------------------------------------------------

GROQ_TPD_429 = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` on "
    "tokens per day (TPD): Limit 200000, Used 199303, Requested 1048. Please "
    "try again in 2m31.632s.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)


def test_groq_publishes_a_token_per_day_limit_and_we_record_it():
    """
    The 1 Sep events run died at item 10 of 100 on TPD 200,000. `Limits` had
    rpm/tpm/rpd and no tpd at all, so every cost estimate in the project priced
    the wrong resource.
    """
    assert ROLES["judge"][2].tpd == 200_000
    assert ROLES["judge_secondary"][2].tpd == 200_000


def test_observed_daily_limit_keeps_the_token_budget():
    t = Throttle(Limits(rpm=5, tpm=250_000, rpd=250, tpd=200_000))
    t.observed_daily_limit(20)
    assert t.limits.rpd == 20
    assert t.limits.tpd == 200_000     # a request cap says nothing about tokens


def test_token_per_day_429_fails_fast_instead_of_retrying_seven_times():
    """
    Groq states TPD exhaustion in prose; the existing fast-fail regex only
    matched Gemini's quota JSON, so the run burned seven backoff attempts and
    then died with a bare RuntimeError. Nothing recovers inside the same day.
    """
    from bapca.llm import DailyQuotaExhausted, LLM

    calls = []

    class Boom(LLM):
        def __init__(self):                      # bypass the real constructor
            pass
        def _create(self, messages, temperature, max_tokens):
            calls.append(1)
            raise RuntimeError(GROQ_TPD_429)

    client = Boom()
    client.provider = ROLES["judge"][0]
    client.model = "qwen/qwen3.8-27b"
    client.limits = Limits(rpm=30, tpm=8_000, rpd=1_000, tpd=200_000)
    client.throttle = Throttle(client.limits)
    client.max_retries = 7
    client.calls_made = 0
    client.tokens_sent = 0
    client.verbose = False
    client._reasoning_ok = True

    with pytest.raises(DailyQuotaExhausted, match="200,000"):
        client._call_with_backoff([{"role": "user", "content": "hi"}], 0.0, 16)
    assert len(calls) == 1, f"should not retry a daily token cap, made {len(calls)} calls"


# ---------------------------------------------------------------------------
# 4 Sep: a Gemini 503 spike killed an unattended run at item 44 of 100
# ---------------------------------------------------------------------------

GEMINI_503 = (
    "Error code: 503 - [{'error': {'code': 503, 'message': 'This model is "
    "currently experiencing high demand. Spikes in demand are usually "
    "temporary. Please try again later.', 'status': 'UNAVAILABLE'}}]"
)


def _stub_client(model="models/gemini-3.5-flash-lite", role="generator", raises=None):
    from bapca.llm import LLM

    class Stub(LLM):
        def __init__(self):
            pass
        def _create(self, messages, temperature, max_tokens):
            Stub.calls += 1
            raise RuntimeError(raises)

    Stub.calls = 0
    c = Stub()
    c.provider, c.model = ROLES[role][0], model
    c.limits = ROLES[role][2]
    c.throttle = Throttle(c.limits)
    c.max_retries = 2
    c.calls_made = c.tokens_sent = 0
    c.verbose = False
    c._reasoning_ok = True
    return c, Stub


def test_a_transient_provider_failure_raises_its_own_type(monkeypatch):
    """
    It used to raise a bare RuntimeError, so run_system could not tell "the
    model is briefly overloaded, skip this item" apart from a real bug and
    died with a traceback mid-run.
    """
    import time as _time
    from bapca.llm import TransientProviderError
    monkeypatch.setattr(_time, "sleep", lambda *_: None)
    client, _ = _stub_client(raises=GEMINI_503)
    with pytest.raises(TransientProviderError):
        client._call_with_backoff([{"role": "user", "content": "hi"}], 0.0, 16)


def test_transient_is_not_confused_with_the_daily_quota_stop():
    """DailyQuotaExhausted must stay a separate stop: one waits, one skips."""
    from bapca.llm import DailyQuotaExhausted, TransientProviderError
    assert not issubclass(TransientProviderError, DailyQuotaExhausted)
    assert not issubclass(DailyQuotaExhausted, TransientProviderError)


def test_an_overloaded_model_is_waited_out_at_its_own_timescale(monkeypatch):
    """
    The 2,4,8,16,32,60 ladder gave up after about two minutes. Gemini's own
    message says the spike is temporary, so the first wait should be tens of
    seconds, not two.
    """
    import time as _time
    from bapca.llm import TransientProviderError
    slept = []
    monkeypatch.setattr(_time, "sleep", lambda s: slept.append(s))
    client, _ = _stub_client(raises=GEMINI_503)
    client.max_retries = 3
    with pytest.raises(TransientProviderError):
        client._call_with_backoff([{"role": "user", "content": "hi"}], 0.0, 16)
    assert slept, "should have waited at all"
    assert min(slept) >= 30.0, f"waits too short for a 503: {slept}"


def test_run_system_skips_a_stalled_item_and_stops_after_three():
    """One overloaded moment costs one item; a dead provider stops the run."""
    import pathlib
    source = (pathlib.Path(__file__).parent.parent / "scripts" / "run_system.py").read_text()
    assert "TransientProviderError" in source
    assert "MAX_CONSECUTIVE_STALLS = 3" in source
    assert "stalled = 0" in source          # reset after a good item
    assert "stalled >= MAX_CONSECUTIVE_STALLS" in source
