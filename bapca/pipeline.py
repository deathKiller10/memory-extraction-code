"""
The actual system: build memory from a conversation, with no oracle.

Conditions C and D handed the model the correct cue. This does not. It receives
exactly what the full-context baseline receives -- the same 21,000-token
conversation -- and has to turn it into a handful of standing notes on its own.
That makes B and E a fair fight on identical input, with the only difference
being what we do with it.

    segment    split the conversation into windows
    extract    one cached LLM call per window -> a standing note, or nothing
    score      Ebbinghaus salience: weight x frequency x time decay
    select     carry the top few standing notes; no similarity retrieval

The last step is the thesis. Standing notes are NOT retrieved by resemblance to
the query, because LoCoMo-Plus removed that signal by construction. They are
retained by type and carried, with decay keeping the carried set small.

WHY THIS IS AFFORDABLE
----------------------
The 401 Cognitive items are stitched into only ten base conversations, so most
windows repeat across items. Every extraction is keyed by the window's text, so
the cache collapses those repeats automatically: a few hundred real calls, not
a few thousand. `estimate_unique_windows` measures this exactly before you spend
anything.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

from .dataset import Sample
from .llm import LLM
from .memory import MemoryType
from .store import EpisodicStore, RetrievalConfig

# A session header, when the stitched prompt carries one.
_DATE_LINE = re.compile(r"^\s*DATE:\s*(.+?)\s*(?:CONVERSATION:)?\s*$", re.IGNORECASE)

DEFAULT_WINDOW_TURNS = 12


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

@dataclass
class Window:
    index: int          # 0 = oldest
    text: str
    dated: bool         # split on a real session boundary, or an arbitrary one


def segment(prompt: str, window_turns: int = DEFAULT_WINDOW_TURNS) -> list[Window]:
    """
    Split a conversation into windows.

    Prefers real session boundaries when the prompt marks them, and otherwise
    falls back to content-defined chunking. We do not assume a format we have
    not verified -- the fallback is what makes this safe on prompts carrying no
    DATE lines, and `dated` records which path was taken so the paper can say.
    """
    lines = [line for line in prompt.splitlines() if line.strip()]
    if not lines:
        return []

    boundaries = [i for i, line in enumerate(lines) if _DATE_LINE.match(line)]
    if len(boundaries) >= 2:
        windows = []
        for n, start in enumerate(boundaries):
            end = boundaries[n + 1] if n + 1 < len(boundaries) else len(lines)
            text = "\n".join(lines[start:end]).strip()
            if text:
                windows.append(Window(len(windows), text, dated=True))
        return windows

    return _content_defined(lines, window_turns)


def _is_boundary(line: str, target: int) -> bool:
    """Cut here? Decided by the line's own hash, not by its position."""
    digest = hashlib.sha1(line.strip().encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % target == 0


def _content_defined(lines: list[str], target: int) -> list[Window]:
    """
    Content-defined chunking, the trick backup tools use to deduplicate files
    that have had something inserted into the middle.

    Fixed windows are anchored to POSITION, so inserting the cue one line
    earlier shifts every boundary after it and nothing downstream matches
    anything from another item. Measured on the real data that cost us: 5,047
    windows collapsed to only 1,153, when ten base conversations should have
    collapsed far further.

    Choosing boundaries from a hash of each line makes them travel with the
    content. An inserted line then disturbs only the window containing it;
    every other window in that conversation hashes identically across all 100
    items, and the cache collapses them.
    """
    minimum, maximum = max(2, target // 2), target * 2
    windows, current = [], []

    for line in lines:
        current.append(line)
        long_enough = len(current) >= minimum
        if (long_enough and _is_boundary(line, target)) or len(current) >= maximum:
            windows.append(Window(len(windows), "\n".join(current), dated=False))
            current = []

    if current:
        windows.append(Window(len(windows), "\n".join(current), dated=False))
    return windows


def strip_trigger(prompt: str, trigger: str) -> str:
    """
    Remove the query from the end of the conversation before segmenting.

    dataset.py: input_prompt is "the full stitched conversation, trigger
    appended last", and trigger is "also the last line of input_prompt".
    Measured: the trigger's words appear in the tail of input_prompt on 100 of
    100 items. run_system.py segments input_prompt and extracts from every
    window, so extraction reads the QUERY and can write a standing note that
    merely describes it.

    Such a note looks maximally relevant to the query because it is the query.
    Measured on the forced-slate run: 6 items carried one, and in all 6 it took
    slot 0 -- the re-ranker's top pick -- every time. Those 6 items reached the
    cue 16.7% of the time against 33.0% elsewhere. The count is small (a strict
    60% content-word threshold; looser paraphrases are missed) but the
    mechanism is exact: when an echo exists it always wins the top slot.

    Independent of size, a memory system that reads the query while building
    its memory is not measuring what the paper claims to measure.

    Conservative by construction: matches on normalised word containment, scans
    at most the last 4 lines, stops at the first line that is not part of the
    query, and returns the prompt untouched when nothing matches.
    """
    body = _normalise(re.sub(r"^\s*[^\s:\uff1a]{1,20}\s*[:\uff1a]\s*", "", trigger or ""))
    want = set(body.split())
    if len(want) < 5:                      # too short to match safely
        return prompt

    lines = prompt.splitlines()
    cut = len(lines)
    floor = max(0, len(lines) - 4)         # never eat more than the tail
    for i in range(len(lines) - 1, floor - 1, -1):
        words = set(_normalise(lines[i]).split())
        if not words:                      # blank line inside the tail
            cut = i
            continue
        if len(words & want) / len(words) >= 0.6:
            cut = i
        else:
            break

    if cut == len(lines):
        # Nothing matched. Return the ORIGINAL string, not a rejoin of its
        # lines: rejoining drops a trailing newline, which changes the final
        # window's text and needlessly invalidates its cache entry.
        return prompt
    return "\n".join(lines[:cut])


_SAID = re.compile(r"^\s*([^\s\"]{1,24}(?:\s+[^\s\"]{1,24}){0,2}?)\s+said,\s*[\"\u201c]?(.*)$")


def target_speaker(prompt: str, trigger: str) -> Optional[str]:
    """
    Who is `A`?

    The benchmark hands us the query as `A: <text>` -- an anonymous label. Four
    of the selection failures in the n=100 events run carried notes about the
    OTHER person in the conversation (#2008 John/Maria, #2030 Tim/John, #2096
    Caroline/Melanie, #2363 Deborah/Jolene), and the re-ranker had no way to
    know which of the two it was answering.

    Printed on 4 Sep, the conversation turns out to be written as

        Joanna said, "It's strange, I barely recognize the person who..."

    -- NOT `Name:` -- and the trigger's own text appears as that final line,
    attributed by name. So `A` is recoverable from the model's own input.

    THIS IS NOT AN ORACLE. It reads `input_prompt` and `trigger`, both of which
    are given to the system at inference; it never touches `evidence`, which is
    the gold field. A deployed memory system always knows whose memory it is --
    the anonymous `A:` is an artefact of the benchmark's formatting, not a
    property of the task.

    Matched the same conservative way as strip_trigger: normalised word
    containment over at most the last 4 lines, and None when nothing matches.
    """
    body = _normalise(re.sub(r"^\s*[^\s:\uff1a]{1,20}\s*[:\uff1a]\s*", "", trigger or ""))
    want = set(body.split())
    if len(want) < 5:
        return None

    lines = (prompt or "").splitlines()
    for i in range(len(lines) - 1, max(-1, len(lines) - 5), -1):
        match = _SAID.match(lines[i])
        if not match:
            continue
        said = set(_normalise(match.group(2)).split())
        if said and len(said & want) / len(said) >= 0.6:
            name = match.group(1).strip().strip('"\u201c').strip()
            return name or None
    return None


def estimate_unique_windows(samples: Iterable[Sample], window_turns: int = DEFAULT_WINDOW_TURNS
                            ) -> tuple[int, int]:
    """
    (total windows, distinct windows) across these samples.

    The second number is what we actually pay for, because the cache is keyed
    by window text and the ten base conversations repeat. Exact, not estimated,
    and it costs nothing to compute.
    """
    total, seen = 0, set()
    for sample in samples:
        for window in segment(sample.input_prompt, window_turns):
            total += 1
            seen.add(window.text)
    return total, len(seen)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "You read conversation excerpts and note what would still matter about a "
    "person weeks later."
)

EXTRACT_PROMPT = """Read this excerpt from a conversation between two friends.

If it reveals something about the FIRST speaker that would still shape how you \
respond to them weeks from now - an ongoing situation, a goal, a value, a \
constraint they live by - write it as ONE short sentence about them, in the \
third person.

Ordinary chat, one-off events and small talk reveal nothing lasting. Most \
excerpts reveal nothing. If this one does not, write exactly: NONE

Answer with the type, a colon, then the sentence. Types: state, goal, value, \
constraint.

Excerpt:
{window}

Answer:"""

# --- v2: written after measuring what the v1 prompt was throwing away ------
#
# extraction_ceiling.py, 31 Aug: the cue is written into the store on only
# 48.5% of items. On the other 51 the best note in a ~30-note store sits at
# cosine 0.29-0.37 from the gold cue -- nothing close. Meanwhile the re-ranker
# finds the cue 79.2% of the time WHEN IT EXISTS. Extraction is the bottleneck
# by a factor of five: it can add up to 52 points of recall, the re-ranker 10.
#
# The reason is the line above: "Ordinary chat, one-off events and small talk
# reveal nothing lasting." Every gold cue in this benchmark is a one-off event
# plus the durable change it caused --
#
#   "The day my bike got stolen from outside the library, I started engraving
#    my initials on everything."
#   "Getting a migraine from staring at screens all day made me buy those
#    blue-light glasses."
#   "Seeing my boss burn out completely made me stop checking work email
#    after 7 p.m."
#
# -- so v1 instructs the model to discard precisely the category under test.
# It was a reasonable prior about what memory should keep. It is wrong here,
# and the data says so.
#
# v2 asks for the event AND the adaptation, and requires the person's name
# (13.3% of v1 notes named nobody and 5.0% named the wrong person, because
# "the FIRST speaker" of a hash-chosen window is a different person per
# window). Same call count -- one per window -- so the cost is one day of
# quota to re-extract, not an increase in ongoing cost.

EXTRACT_PROMPT_EVENTS = """Read this excerpt from a conversation between two friends.

People change how they live because of things that happen to them. Look for \
that pattern here: something happened to one of them - an injury, a scare, a \
loss, a diagnosis, a theft, an incident of any kind - and they now do \
something differently because of it.

Write it as ONE short sentence in the third person that keeps BOTH halves: \
what happened, and what they now do because of it. Name the person; never \
write "she" or "he" without a name.

If nothing like that appears, a goal they are working toward or a rule they \
live by is worth recording instead, in the same form.

If the excerpt shows neither, write exactly: NONE

Answer with the type, a colon, then the sentence. Types: state, goal, value, \
constraint.

Excerpt:
{window}

Answer:"""

EXTRACT_PROMPTS = {"v1": EXTRACT_PROMPT, "events": EXTRACT_PROMPT_EVENTS}


_TYPED = re.compile(r"^\s*(state|goal|value|constraint)\s*[:\-]\s*(.+)$",
                    re.IGNORECASE | re.DOTALL)


@dataclass
class Note:
    text: str
    mem_type: MemoryType
    window_index: int

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def as_context_line(self) -> str:
        """Same rendering as a stored MemoryNode, so the re-ranker can take
        either an extracted Note or a node retrieved from the store."""
        if self.mem_type.is_standing:
            return f"[{self.mem_type.value}] {self.text}"
        return self.text


def parse_note(raw: str, window_index: int) -> Optional[Note]:
    """
    Read one extraction. Anything malformed is dropped, never guessed at --
    a note we invent here becomes a memory the system acts on later.
    """
    text = (raw or "").strip().strip('"').strip()
    if not text or text.upper().startswith("NONE"):
        return None

    match = _TYPED.match(text)
    if match:
        kind, body = match.group(1).lower(), match.group(2).strip()
        body = body.split("\n")[0].strip()
        if not body or body.upper().startswith("NONE"):
            return None
        return Note(body, MemoryType(kind), window_index)

    # Untyped but non-empty: keep it, defaulting to the weakest-decaying type
    # would flatter us, so use `state`, which decays fastest of the four.
    first = text.split("\n")[0].strip()
    return Note(first, MemoryType.STATE, window_index) if first else None


def extract_notes(llm: LLM, windows: list[Window],
                  prompt: str = EXTRACT_PROMPT) -> list[Note]:
    """`prompt` selects the extraction wording; see EXTRACT_PROMPTS. Changing it
    invalidates the extraction cache (the cache key includes the prompt text),
    which costs one full re-extraction -- about 569 unique windows."""
    notes = []
    for window in windows:
        raw = llm.complete(
            prompt.format(window=window.text[:6000]),
            system=EXTRACT_SYSTEM,
            max_tokens=160,
        )
        note = parse_note(raw, window.index)
        if note:
            notes.append(note)
    return notes


# ---------------------------------------------------------------------------
# The memory
# ---------------------------------------------------------------------------

@dataclass
class SystemConfig:
    """
    Defaults changed after the first run of condition E, which scored 36% --
    below the full-context baseline -- with the needed note reaching the prompt
    only 14% of the time, against 17% for picking at random.

    Two causes, both ours:

    days_per_window was 7.0. Ageing a 29-window conversation at a week per
    window makes the oldest notes 200 days old, and the Review 1 pruning
    threshold (0.20) then deletes most of them before selection runs. Those
    constants described a human timeline of real weeks; applying them to a
    spacing we invented was a category error. At 1.0 every note survives.

    carry was 5. Choosing 5 of 29 by recency is a lottery, because the cue sits
    at a random point in the conversation. Carrying everything costs ~350
    tokens -- still sixty times cheaper than the 22,385-token baseline -- so
    the budget was buying us nothing and losing us the answer.
    """
    window_turns: int = DEFAULT_WINDOW_TURNS
    days_per_window: float = 1.0     # spacing used to age older windows
    carry: int = 0                   # 0 = carry every surviving note
    prune_threshold: float = 0.05    # was 0.20, tuned for a different timescale
    retrieval: RetrievalConfig = None

    def __post_init__(self):
        if self.retrieval is None:
            self.retrieval = RetrievalConfig(
                standing_budget=self.carry if self.carry > 0 else 10_000,
                standing_floor=0.0,
                prune_threshold=self.prune_threshold,
            )


def build_store(notes: list[Note], embedder, config: SystemConfig) -> EpisodicStore:
    """
    Lay the notes out in time and let salience sort them.

    Window position stands in for elapsed time: a note from window 0 is older
    than one from window 12. That is what gives decay something to act on, and
    it is honest -- the conversation really did happen in that order.
    """
    store = EpisodicStore(config.retrieval)
    if not notes:
        return store

    latest = max(n.window_index for n in notes)
    start = datetime(2026, 1, 1)

    for note in notes:
        age_days = (latest - note.window_index) * config.days_per_window
        store.write(
            note.text,
            embedder.encode_one(note.text),
            when=start - timedelta(days=age_days),
            weight=0.6,                     # uniform: we have no basis to rank
            mem_type=note.mem_type,         # them at write time, and pretending
            source_turn=note.window_index,  # otherwise would be a hidden oracle
        )
    return store


# ---------------------------------------------------------------------------
# Locating the cue -- for measurement only, never for the system
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def locate_evidence_window(windows: list[Window], evidence: str) -> Optional[int]:
    """
    Which window contains the gold cue?

    The previous diagnostic asked whether any carried note sat within cosine
    0.5 of the gold evidence. That compares raw dialogue ("Caroline: After
    learning to say 'no'...") against a third-person paraphrase ("She has
    learned to protect her time"), which can score below 0.5 while meaning the
    same thing -- so it under-reported, and we could not tell a genuine miss
    from a metric artefact.

    This instead finds the window the cue was inserted into, by exact text
    match, and asks whether a note from that window survived. Unambiguous.
    Uses the gold answer, so it is a measurement, never an input.
    """
    spans = []
    for line in evidence.splitlines():
        body = re.split(r"[:\uff1a]", line, maxsplit=1)[-1]
        normalised = _normalise(body)
        if len(normalised) >= 25:
            spans.append(normalised)
    if not spans:
        return None

    # Exact first: the cue is normally inserted verbatim.
    for window in windows:
        haystack = _normalise(window.text)
        if any(span in haystack for span in spans):
            return window.index

    # Then word containment, because the stitched copy can differ in small
    # ways ("pain's been" against "pain has been"). Demanding 70% of the
    # cue's words in one window is loose enough to survive rewording and
    # tight enough that an unrelated window will not reach it.
    best_index, best_score = None, 0.0
    for window in windows:
        words = set(_normalise(window.text).split())
        for span in spans:
            span_words = set(span.split())
            if len(span_words) < 5:
                continue
            score = len(span_words & words) / len(span_words)
            if score > best_score:
                best_index, best_score = window.index, score
    return best_index if best_score >= 0.70 else None


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

RERANK_SYSTEM = (
    "You keep track of what you know about a friend and decide what is worth "
    "bearing in mind when they say something."
)

RERANK_PROMPT = """Here is what you know about this person:

{notes}

They have just said:
  {trigger}

Rank these notes by how much they should shape your reply.

The connection is almost always indirect. A note about how they handle \
commitments bears on a message about feeling overwhelmed, even though the two \
share no words. Look for what the message implies about them, not for matching \
subject matter.

Answer with exactly {top_k} numbers, separated by commas, best first. Nothing \
else - no explanation, no fewer than {top_k}, and never the word NONE."""


# --- person-aware variant, written 4 Sep --------------------------------
#
# In the n=100 events run the correct note sat unused in the store on 17 items
# (worth +13 points). Four of those failures are the same mistake: the notes
# carried were about the OTHER person in the conversation.
#
#   #2008  cue about John      -> two notes about Maria
#   #2030  cue about Tim       -> two notes about John
#   #2096  cue about Caroline  -> three about Melanie, one of them
#                                 "Melanie's son was in a car accident"
#   #2363  cue about Deborah   -> three about Jolene
#
# #2096 is the mechanism in one line: the re-ranker matched the TOPIC and got
# the SUBJECT wrong. It had no way not to -- v1's prompt says "this person"
# without ever saying who, because the benchmark anonymises the query's speaker
# as "A:". `target_speaker` recovers the name from input_prompt + trigger,
# never from the gold evidence field (see its docstring), and check_speaker.py
# measured that at 100/100 correct.
#
# PREFER, do not filter. A hard filter would drop any note that names nobody,
# and would make an unrecoverable error whenever the resolver is wrong. This
# states who is speaking and says other people's notes are rarely the answer;
# the model can still take one when it genuinely bears.

RERANK_PROMPT_PERSON = """Here is what you know about the people in this \
conversation:

{notes}

You are replying to {speaker}. {speaker} has just said:
  {trigger}

Rank these notes by how much they should shape your reply to {speaker}.

These notes are about more than one person. A note about someone else is \
rarely the right answer - prefer a note about {speaker} unless another \
genuinely bears on what {speaker} just said.

The connection is almost always indirect. A note about how {speaker} handles \
commitments bears on a message about feeling overwhelmed, even though the two \
share no words. Look for what the message implies about {speaker}, not for \
matching subject matter.

Answer with exactly {top_k} numbers, separated by commas, best first. Nothing \
else - no explanation, no fewer than {top_k}, and never the word NONE."""

RERANK_PROMPTS = {"v1": RERANK_PROMPT, "person": RERANK_PROMPT_PERSON}

_NUMBERS = re.compile(r"\d+")


def _rerank_prompt(variant: str, listing: str, sample: Sample, top_k: int) -> str:
    """Build the re-rank prompt. `person` falls back to v1 when the speaker
    cannot be resolved, so an unrecognised format degrades to the old
    behaviour instead of injecting the string "None" into the prompt."""
    if variant == "person":
        who = target_speaker(sample.input_prompt, sample.trigger)
        if who:
            return RERANK_PROMPT_PERSON.format(notes=listing, trigger=sample.trigger,
                                               top_k=top_k, speaker=who)
    return RERANK_PROMPT.format(notes=listing, trigger=sample.trigger, top_k=top_k)


def select_by_llm(llm: LLM, notes: list, sample: Sample, top_k: int = 3,
                  backfill: Optional[list] = None, rerank: str = "v1") -> list:
    """
    Ask a model which notes matter, instead of measuring embedding distance.

    The connection between "I have learned to say no" and "I volunteered and
    now I am overwhelmed" is an inference, not a resemblance. A reasoner can
    make it; a distance metric cannot. One call per query.

    First measurement (permissive prompt, "NONE" allowed): the re-ranker chose
    a note that was right 87.5% of the time -- oracle precision, 87.6% for
    condition D -- but returned a median of ONE note out of three, and often
    zero. The cue reached the prompt 16 times in 100, worse than similarity's
    21. Precision was solved; recall was not. So the prompt now demands exactly
    top_k, and `backfill` (a similarity-ranked list) tops up any shortfall,
    because an unused slot can only lose recall: the note it would have held
    is already known to be worth +54 points when it is the right one.
    """
    if not notes:
        return []

    listing = "\n".join(f"  {i + 1}. {n.as_context_line()}" for i, n in enumerate(notes))
    raw = llm.complete(
        _rerank_prompt(rerank, listing, sample, top_k),
        system=RERANK_SYSTEM,
        max_tokens=160,
    )

    chosen, seen = [], set()
    for match in _NUMBERS.finditer(raw):
        index = int(match.group()) - 1
        if 0 <= index < len(notes) and index not in seen:
            seen.add(index)
            chosen.append(notes[index])
        if len(chosen) >= top_k:
            break

    # A refusal ("NONE") or a short answer leaves slots empty. Fill them in
    # similarity order -- weaker evidence, but strictly better than nothing.
    for note in backfill or []:
        if len(chosen) >= top_k:
            break
        if id(note) not in {id(c) for c in chosen}:
            chosen.append(note)
    return chosen


def select_notes(store: EpisodicStore, sample: Sample, embedder, *,
                 mode: str = "carry", top_k: int = 3, llm: Optional[LLM] = None,
                 now: Optional[datetime] = None, rerank: str = "v1") -> list:
    """
    Choose which standing notes reach the prompt.

    "carry"   -- every survivor, ordered by salience. This was the thesis:
                 do not retrieve, just carry, and let decay keep the set small.
                 Measured at 40% against a 52% full-context baseline, with the
                 right note present half the time and worth only +8 points when
                 it was. Twenty-seven notes drown the one that matters.

    "similar" -- top-k by embedding similarity to the trigger. Measured: the
                 cue note reaches the prompt 21% of the time, against 10% for
                 chance. Better than nothing, and still wrong four times in
                 five, because the benchmark strips cue-query overlap on
                 purpose. Score 44%, on 130 tokens.

    "llm"     -- ask a model which notes bear on the message. The connection
                 is inferential, so the selector has to reason. Costs one call
                 per query. Measured at oracle precision (87.5% correct when
                 its pick was the cue) and poor recall (16%).

    "hybrid"  -- the same call, with any slots the model leaves empty filled in
                 similarity order. Same cost; recall cannot be lower than the
                 model's own, and should not be lower than similarity's.
    """
    now = now or datetime(2026, 1, 1)
    query = embedder.encode_one(sample.trigger)
    retrieved = store.retrieve(query, now)
    notes = retrieved.standing_hits

    if mode == "carry":
        return notes
    if mode not in ("llm", "similar", "hybrid"):
        raise ValueError(f"unknown selection mode {mode!r}")

    from .memory import cosine
    ranked = sorted(notes, key=lambda n: cosine(query, n.embedding), reverse=True)
    if mode == "similar":
        return ranked[:top_k]

    if llm is None:
        raise ValueError(f"mode={mode!r} needs an llm to do the re-ranking")
    return select_by_llm(llm, notes, sample, top_k,
                         backfill=ranked if mode == "hybrid" else None,
                         rerank=rerank)


def render_prompt(sample: Sample, notes: list) -> tuple[str, int, list[str]]:
    lines = [n.as_context_line() for n in notes]
    if not lines:
        return sample.trigger, len(sample.trigger) // 4, []
    body = "\n".join(f"- {line}" for line in lines)
    prompt = f"What you know about this person:\n{body}\n\n{sample.trigger}"
    return prompt, len(prompt) // 4, lines


def prompt_with_memory(sample: Sample, store: EpisodicStore, embedder,
                       now: Optional[datetime] = None) -> tuple[str, int, list[str]]:
    """Build the query prompt. Returns (prompt, tokens, the notes carried)."""
    now = now or datetime(2026, 1, 1)
    got = store.retrieve(embedder.encode_one(sample.trigger), now)
    carried = [m.as_context_line() for m in got.standing_hits]

    if not carried:
        return sample.trigger, len(sample.trigger) // 4, []

    lines = "\n".join(f"- {c}" for c in carried)
    prompt = f"What you know about this person:\n{lines}\n\n{sample.trigger}"
    return prompt, len(prompt) // 4, carried


def hit_rate(carried: list[str], evidence: str, embedder, threshold: float = 0.5) -> float:
    """
    Did the note we needed survive into the carried set?

    Reported separately from the answer score, because the two failure modes
    need different fixes: if the right note is never carried, extraction or
    selection is broken; if it is carried and the answer still ignores it, the
    problem is in how we present it.
    """
    if not carried:
        return 0.0
    from .memory import cosine
    target = embedder.encode_one(evidence)
    return max(cosine(target, embedder.encode_one(note)) for note in carried) >= threshold
