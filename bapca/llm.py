"""
Cached, rate-limit-aware LLM clients over free OpenAI-compatible endpoints.

ROLE ASSIGNMENT IS DRIVEN BY MEASURED LIMITS, NOT PREFERENCE.

  GENERATOR -> Gemini flash-lite   TPM 250,000 | RPM 15 | RPD 1,000
  JUDGE     -> Groq  (Qwen/GPT-OSS) TPM   8,000 | RPM 30 | RPD 1,000
  2nd JUDGE -> Gemini flash        TPM 250,000 | RPM 10 | RPD   250

We started with the opposite assignment and it was unrunnable. The LoCoMo-Plus
Cognitive prompts have a median of 21,201 tokens, and Groq's free tier caps at
8,000 tokens per minute -- so a single full-context call could never succeed
there, no matter how patiently we retried. Gemini's free tier allows 250,000
TPM, which fits a 21k prompt comfortably.

The judge, by contrast, only ever sees a short answer plus the evidence line
(~500 tokens), so it fits inside Groq's 8k window with room to spare. Putting
the judge on Groq also preserves the property we wanted from the start: the
model being judged and the model doing the judging come from different vendors
and different families, which is the one-line answer to "how do you know this
isn't self-preference bias?"

Everything else here is defensive because it bit us on a real run: model names
that the list endpoint serves but the API rejects, reasoning models emitting
their chain of thought as the answer, and empty responses being cached as if
they were fine.
"""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from typing import Optional

from .cache import DiskCache

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - import guard for a clearer message
    OpenAI = None  # type: ignore


GROQ_BASE = "https://api.groq.com/openai/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

CHARS_PER_TOKEN = 4  # rough, but we only need it to pace requests and warn early


@dataclass
class Provider:
    name: str
    base_url: str
    api_key_env: str

    def key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"No API key found. Set the {self.api_key_env} environment variable.\n"
                f"  Groq   -> https://console.groq.com/keys      (key starts 'gsk_')\n"
                f"  Gemini -> https://aistudio.google.com/apikey (key starts 'AIza')\n"
                f"In Colab: use the key icon in the left sidebar to add it as a secret."
            )
        return key


GROQ = Provider("groq", GROQ_BASE, "GROQ_API_KEY")
GEMINI = Provider("gemini", GEMINI_BASE, "GEMINI_API_KEY")


@dataclass
class Limits:
    """Free-tier limits, verified Aug 2026. Used to pace requests, not guess.

    `tpd` was added on 2 Sep after the `--extract events` run died at item 10 of
    100 on a limit this class did not model at all: Groq enforces **tokens per
    day**, 200,000 for the on-demand free tier, and a cold extraction pass over
    569 windows needs roughly three times that. Every cost estimate in this
    project counted requests, because RPD 1,000 was the only daily limit we
    knew about. Requests were never the binding constraint. See bug 16.
    """
    rpm: int
    tpm: int
    rpd: int
    tpd: Optional[int] = None      # None = not published / not observed


# (provider, model preferences, limits)
#
# Preferences are (substring we want, substrings that disqualify it). The
# exclusions matter: plain "flash" is a substring of "flash-lite", so without
# them we could never express "a full flash model, not the lite one".
ROLES: dict[str, tuple[Provider, list[tuple[str, tuple[str, ...]]], Limits]] = {
    "generator": (
        GEMINI,
        [("flash-lite", ("preview", "thinking")),
         ("flash", ("lite", "preview", "thinking", "image", "audio", "omni")),
         ("pro", ("preview", "vision"))],
        Limits(rpm=15, tpm=250_000, rpd=1_000, tpd=None),
    ),
    "judge": (
        GROQ,
        [("llama-3.3-70b", ()), ("llama-4", ()), ("qwen", ()),
         ("kimi", ()), ("gpt-oss-120b", ()), ("gpt-oss-20b", ()), ("llama-3.1-8b", ())],
        Limits(rpm=30, tpm=8_000, rpd=1_000, tpd=200_000),
    ),
    # A second family on Groq, not a second Gemini model. Gemini's free tier
    # reported "GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue:
    # '20'" for gemini-3.6-flash -- twenty calls a day, against the 250 our
    # config claimed and the "250 RPD" the documentation claimed. An agreement
    # study needs ~100 calls, so Gemini flash simply cannot host one.
    #
    # GPT-OSS is a different family from the Qwen primary judge (a real
    # stability check) and still a different vendor from the Gemini generator,
    # so the self-preference defence for the primary judge is unaffected.
    "judge_secondary": (
        GROQ,
        [("gpt-oss-120b", ()), ("gpt-oss-20b", ()), ("kimi", ()), ("llama", ())],
        Limits(rpm=30, tpm=8_000, rpd=1_000, tpd=200_000),
    ),
}

AVOID = ("whisper", "tts", "embed", "guard", "moderation", "image", "audio",
         "video", "orpheus", "compound")

PROBE_PROMPT = "Reply with exactly one word: ready"
PROBE_MAX_TOKENS = 512  # generous, so a reasoning model still reaches its answer

THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

# A 429 usually tells us exactly how long to wait. Guessing with exponential
# backoff wasted five attempts on a 40-second window and then gave up.
_RETRY_AFTER = re.compile(
    r"(?:retry in|retrydelay['\"]?\s*[:=]\s*['\"]?)\s*(\d+(?:\.\d+)?)\s*s",
    re.IGNORECASE)


def retry_delay_from(message: str) -> Optional[float]:
    """Seconds the provider asked us to wait, if it said."""
    match = _RETRY_AFTER.search(message or "")
    return float(match.group(1)) if match else None


class ModelUnavailable(RuntimeError):
    """The endpoint served the name but refuses to run it. Try the next one."""


class PromptTooLarge(RuntimeError):
    """No amount of retrying will help: one call exceeds the per-minute budget."""


class TransientProviderError(RuntimeError):
    """
    The provider was briefly unable to serve the call -- an overloaded model
    (503 UNAVAILABLE), a timeout, a rate limit we could not wait out.

    Distinct from RuntimeError so a caller can retry the ITEM later without
    also swallowing genuine bugs. Raised on 4 Sep after a Gemini 503 spike
    ("This model is currently experiencing high demand") killed a run at item
    44 of 100 while nobody was watching. Nothing was lost -- rows are written
    per item -- but a whole unattended session was.
    """


class DailyQuotaExhausted(RuntimeError):
    """Out of requests until the quota resets. Waiting minutes will not help."""


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def strip_reasoning(text: str) -> str:
    """
    Remove chain-of-thought that leaked into the content.

    A truncated response can contain an *unclosed* `<think>`, in which case
    everything from the tag on is thinking, not answer -- drop it. If that
    leaves nothing, the caller retries with a bigger budget rather than
    accepting a monologue as an answer.
    """
    text = THINK_BLOCK.sub("", text)
    lowered = text.lower()
    if "<think>" in lowered:
        text = text[: lowered.index("<think>")]
    return text.strip()


def _version_of(model_id: str) -> float:
    """
    Family version from a model id, both spellings:
        gemini-3.5-flash-lite -> 3.5      (separated)
        qwen/qwen3.6-27b      -> 3.6      (glued to the family name)
        openai/gpt-oss-120b   -> 0.0      (120 is a parameter count)
    """
    decimal = re.search(r"\d+\.\d+", model_id)
    if decimal:
        return float(decimal.group())
    integer = re.search(r"[-/](\d+)[-/]", model_id)
    if integer:
        value = float(integer.group(1))
        return value if value < 100 else 0.0
    return 0.0


def reasoning_kwargs(provider_name: str, model_id: str) -> dict:
    """
    Keep the chain of thought out of the answer. Per Groq's reasoning docs:
      * Qwen / MiniMax -> `reasoning_format: "hidden"` returns only the answer.
      * GPT-OSS        -> rejects reasoning_format; needs `include_reasoning`
                          instead (the two are mutually exclusive).
    Gemini needs none of this.
    """
    if provider_name != "groq":
        return {}
    model = model_id.lower()
    if "gpt-oss" in model:
        return {"include_reasoning": False, "reasoning_effort": "low"}
    if "qwen" in model or "minimax" in model:
        return {"reasoning_format": "hidden", "reasoning_effort": "none"}
    return {}


class Throttle:
    """
    Sliding-window pacer so we approach the free-tier limits instead of
    bouncing off them. Cheaper than a 429 storm and far more predictable:
    a paced run finishes, a hammered one spends its budget on retries.
    """

    def __init__(self, limits: Limits):
        self.limits = limits
        self._events: list[tuple[float, int]] = []  # (timestamp, tokens)
        self.day_count = 0
        self.effective_rpm = limits.rpm
        self.throttle_events = 0

    def observed_daily_limit(self, value: int) -> None:
        """The provider told us its real daily cap. Believe it over our config."""
        self.limits = Limits(self.limits.rpm, self.limits.tpm, value,
                             self.limits.tpd)

    def observed_rate_limit(self) -> None:
        """
        We hit a 429 despite pacing, so our configured RPM is wrong. Back off
        permanently rather than rediscovering it on every call -- published
        limits have been wrong twice now.
        """
        self.throttle_events += 1
        self.effective_rpm = max(1, self.effective_rpm // 2)

    def _prune(self, now: float) -> None:
        self._events = [(t, n) for t, n in self._events if now - t < 60.0]

    def wait_for(self, tokens: int) -> float:
        if tokens > self.limits.tpm:
            raise PromptTooLarge(
                f"A single request of ~{tokens:,} tokens exceeds this provider's "
                f"free-tier limit of {self.limits.tpm:,} tokens per minute. "
                "Retrying cannot help -- use a provider with a larger window, or "
                "shorten the prompt."
            )
        if self.day_count >= self.limits.rpd:
            raise DailyQuotaExhausted(
                f"Used {self.day_count} of {self.limits.rpd} requests allowed today "
                "for this model. This resets on the provider's schedule -- retrying "
                "now cannot help. Use a different model, or continue tomorrow "
                "(everything already done is cached)."
            )
        slept = 0.0
        while True:
            now = time.monotonic()
            self._prune(now)
            used_tokens = sum(n for _, n in self._events)
            if len(self._events) < self.effective_rpm and used_tokens + tokens <= self.limits.tpm:
                self._events.append((now, tokens))
                self.day_count += 1
                return slept
            oldest = min(t for t, _ in self._events)
            pause = max(0.5, 60.0 - (now - oldest))
            time.sleep(pause)
            slept += pause


class LLM:
    """One model on one provider, with a cache and a pacer in front of it."""

    def __init__(
        self,
        role: str = "generator",
        model: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 7,
        verbose: bool = False,
    ):
        if OpenAI is None:
            raise ImportError("pip install openai")
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}. Expected one of {list(ROLES)}")

        self.role = role
        self.provider, self._prefs, self.limits = ROLES[role]
        self.verbose = verbose
        self._client = OpenAI(api_key=self.provider.key(), base_url=self.provider.base_url)
        self.cache = DiskCache(namespace or f"llm/{role}")
        self._model_cache = DiskCache("model_choice")
        self.throttle = Throttle(self.limits)
        self.calls_made = 0        # real network calls, i.e. quota actually spent
        self.tokens_sent = 0
        self._reasoning_ok = True  # flips off if the provider rejects the params
        self.model = model or self.resolve_model()
        self.temperature = temperature
        self.max_retries = max_retries

    # ---- model discovery -------------------------------------------------

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list().data)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not list models on {self.provider.name}: {exc}\n"
                "Check the API key and your network connection."
            ) from exc

    def rank_candidates(self, available: Optional[list[str]] = None) -> list[str]:
        available = available if available is not None else self.list_models()
        usable = [m for m in available if not any(bad in m.lower() for bad in AVOID)]

        ranked: list[tuple[int, float, str]] = []
        for model_id in usable:
            lowered = model_id.lower()
            rank = len(self._prefs)  # unmatched sort last, but stay as fallbacks
            for i, (want, forbid) in enumerate(self._prefs):
                if want in lowered and not any(bad in lowered for bad in forbid):
                    rank = i
                    break
            ranked.append((rank, -_version_of(model_id), model_id))
        ranked.sort()
        return [model_id for _, _, model_id in ranked]

    def probe(self, model_id: str) -> bool:
        """Does this model return a real, non-reasoning answer? Costs one call."""
        try:
            self.throttle.wait_for(PROBE_MAX_TOKENS)
            response = self._client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": PROBE_PROMPT}],
                max_tokens=PROBE_MAX_TOKENS,
                temperature=0,
                extra_body=reasoning_kwargs(self.provider.name, model_id) or None,
            )
            self.calls_made += 1
            text = strip_reasoning(response.choices[0].message.content or "")
            if not text and self.verbose:
                print(f"    {model_id}: no answer outside its reasoning -- skipping")
            return bool(text)
        except Exception as exc:  # noqa: BLE001
            if self.verbose:
                print(f"    {model_id}: {str(exc)[:110]} -- skipping")
            return False

    def resolve_model(self) -> str:
        """
        Pick a model that demonstrably works, and remember the choice.

        Keyed on the full served list, so a change in the provider's lineup
        re-probes automatically instead of failing on a name that has died.
        """
        available = self.list_models()
        key = DiskCache.make_key(
            provider=self.provider.name, role=self.role, available=available, v=3
        )
        remembered = self._model_cache.get(key)
        if remembered:
            return remembered

        candidates = self.rank_candidates(available)
        if self.verbose:
            print(f"  probing {self.provider.name} models for role '{self.role}'...")
        for model_id in candidates[:6]:
            if self.probe(model_id):
                self._model_cache.put(key, model_id)
                return model_id

        raise RuntimeError(
            f"No working model found on {self.provider.name} for role '{self.role}'.\n"
            f"Tried: {candidates[:6]}\nAll served models: {available}"
        )

    # ---- completion ------------------------------------------------------

    def fits(self, prompt: str, system: str = "", max_tokens: int = 512) -> bool:
        """Check before building a whole experiment on a prompt that cannot run."""
        return estimate_tokens(prompt + system) + max_tokens <= self.limits.tpm

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 512,
    ) -> str:
        temp = self.temperature if temperature is None else temperature
        key = DiskCache.make_key(
            provider=self.provider.name, model=self.model, system=system,
            prompt=prompt, temperature=temp, max_tokens=max_tokens,
        )
        cached = self.cache.get(key)
        if cached:  # an empty cached string is a miss, not a hit
            return cached

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        text = self._call_with_backoff(messages, temp, max_tokens)

        # If the whole budget went on thinking, give it room once more, then
        # fail loudly. Caching an empty answer would corrupt every downstream
        # result while looking perfectly healthy in a table.
        if not text:
            text = self._call_with_backoff(messages, temp, max_tokens * 4)
        if not text:
            raise RuntimeError(
                f"{self.model} produced no answer outside its reasoning trace, even at "
                f"{max_tokens * 4} tokens. Raise max_tokens or pick another model."
            )

        self.cache.put(key, text, meta={"model": self.model, "provider": self.provider.name})
        return text

    def _create(self, messages, temperature: float, max_tokens: int):
        """One API call, degrading gracefully if reasoning params are rejected."""
        extra = reasoning_kwargs(self.provider.name, self.model) if self._reasoning_ok else {}
        try:
            return self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, extra_body=extra or None,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if extra and any(
                token in message
                for token in ("reasoning", "unknown", "unsupported", "unrecognized",
                              "invalid_request", "400")
            ):
                self._reasoning_ok = False
                if self.verbose:
                    print(f"    note: {self.model} rejected reasoning params, "
                          "stripping <think> client-side instead")
                return self._client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                )
            raise

    def _call_with_backoff(self, messages, temperature: float, max_tokens: int) -> str:
        """
        Pace first, then retry what is worth retrying.

        A dead model name and an oversized prompt will never succeed -- both
        fail immediately rather than sleeping through five pointless retries.
        """
        payload_tokens = estimate_tokens("".join(m["content"] for m in messages)) + max_tokens
        self.throttle.wait_for(payload_tokens)  # raises PromptTooLarge if hopeless

        delay = 2.0
        last_error: Optional[Exception] = None
        attempt = 0

        for attempt in range(self.max_retries):
            try:
                response = self._create(messages, temperature, max_tokens)
                self.calls_made += 1
                self.tokens_sent += payload_tokens
                return strip_reasoning(response.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001 - providers raise varied types
                last_error = exc
                message = str(exc).lower()
                if any(token in message for token in
                       ("404", "not_found", "no longer available", "does not exist")):
                    raise ModelUnavailable(
                        f"{self.model} is no longer served by {self.provider.name}: {exc}\n"
                        "Delete the 'model_choice' folder in your cache dir to re-probe."
                    ) from exc
                retryable = any(token in message for token in
                                ("rate", "429", "quota", "timeout", "503", "502", "overload"))
                if not retryable or attempt == self.max_retries - 1:
                    break

                # Groq states it in prose: "on tokens per day (TPD): Limit
                # 200000, Used 199303". No amount of backing off inside a day
                # recovers from this, so say so at once instead of sleeping
                # through six more attempts.
                tpd = re.search(r"tokens per day \(tpd\)\s*:\s*limit\s*(\d+)", message)
                if tpd:
                    raise DailyQuotaExhausted(
                        f"{self.model} has used its {int(tpd.group(1)):,} tokens for the day. "
                        f"Everything completed so far is cached and every finished item is "
                        f"already written to the results file. Re-run the SAME command after "
                        f"the daily reset and it resumes."
                    ) from exc

                daily = re.search(r"perday\w*-freetier[^}]*?quotavalue['\"]?\s*:\s*['\"]?(\d+)",
                                  message, re.IGNORECASE | re.DOTALL)
                if daily:
                    # Out of requests for the day. Backing off for a minute is
                    # pointless; say so immediately instead of burning retries.
                    self.throttle.observed_daily_limit(int(daily.group(1)))
                    raise DailyQuotaExhausted(
                        f"{self.model} allows only {daily.group(1)} requests per day on "
                        f"the free tier, and they are used up. Everything completed so "
                        f"far is cached. Switch models or resume after the reset."
                    ) from exc
                if "429" in message or "quota" in message or "rate" in message:
                    self.throttle.observed_rate_limit()

                # Prefer the wait the server asked for. Our exponential backoff
                # summed to ~30s against a stated 40s window, so every attempt
                # was doomed before it was made.
                asked = retry_delay_from(str(exc))
                pause = (asked + 1.0) if asked else delay
                # "high demand ... usually temporary" clears in tens of seconds,
                # but our 2,4,8,16,32,60 ladder gave up after ~2 minutes total.
                # Wait at the provider's own timescale for an overload.
                if not asked and any(t in message for t in
                                     ("503", "unavailable", "overload", "high demand")):
                    pause = max(pause, 30.0)
                if self.verbose and asked:
                    print(f"    rate limited; provider asked for {asked:.0f}s, waiting")
                time.sleep(pause + random.uniform(0, 1.0))
                delay = min(delay * 2, 60.0)

        raise TransientProviderError(
            f"{self.provider.name} call failed after {attempt + 1} attempt(s): {last_error}"
        )

    def stats(self) -> dict:
        return {
            "role": self.role,
            "provider": self.provider.name,
            "model": self.model,
            "network_calls": self.calls_made,
            "tokens_sent": self.tokens_sent,
            "requests_today": self.throttle.day_count,
            "effective_rpm": self.throttle.effective_rpm,
            "rate_limit_hits": self.throttle.throttle_events,
            "rpd_limit": self.limits.rpd,
            **self.cache.stats(),
        }


def generator(model: Optional[str] = None, verbose: bool = True) -> LLM:
    """The system under test. On Gemini because it must swallow 21k-token prompts."""
    return LLM("generator", model, verbose=verbose)


def judge(model: Optional[str] = None, verbose: bool = True) -> LLM:
    """
    The primary evaluator. On Groq: its prompts are small (an answer plus one
    evidence line), and a different vendor from the generator is the cheapest
    possible defence against judge self-preference bias.
    """
    return LLM("judge", model, verbose=verbose)


def judge_secondary(model: Optional[str] = None, verbose: bool = True) -> LLM:
    """
    A second judge from a different family, for the agreement study.

    Our primary judge is Qwen 27B, chosen because it fits the free tier. That
    is defensible only if the scores do not hinge on it -- so we re-judge a
    subset with GPT-OSS and report Cohen's kappa, as LoCoMo-Plus does in their
    Table 3. Note that agreement between two open models is weaker evidence
    than agreement with humans: scripts/human_annotate.py is the stronger
    check, and it costs no quota at all.
    """
    return LLM("judge_secondary", model, namespace="llm/judge2", verbose=verbose)
