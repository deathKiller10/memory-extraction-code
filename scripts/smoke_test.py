"""
End-to-end check that the free stack works, before we build anything on it.

    python scripts/smoke_test.py            # offline: memory logic only
    python scripts/smoke_test.py --online   # also checks Groq + Gemini keys

The online mode spends about 4 requests total, which is nothing against a
combined free budget of ~16,000/day. Run it once after you create the keys.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Run from a plain checkout (or a Colab clone) without installing anything.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bapca import EpisodicStore, MemoryType, RetrievalConfig, cosine
from bapca.embeddings import Embedder, HashEmbedder

START = datetime(2026, 1, 1)


def banner(text: str) -> None:
    print(f"\n{'=' * 68}\n{text}\n{'=' * 68}")


def check_cache() -> bool:
    """
    Prove the cache actually persists BEFORE spending any quota on it.

    A cache that works in memory but never lands on disk turns a one-off cost
    into a cost per session, and you discover it on day four when the free
    quota runs dry mid-experiment.
    """
    banner("0. Cache health")
    from bapca.cache import DiskCache

    result = DiskCache("selftest").selftest()
    print(f"  directory: {result['path']}")
    print(f"  write:     {'ok' if result['written'] else 'FAILED'}")
    print(f"  read back: {'ok' if result['readable'] else 'FAILED'}")

    if result["readable"]:
        print("\n  PASS: results will survive a restart.")
        return True

    print(f"\n  FAIL: {result.get('error') or 'the file did not read back'}")
    print("  Everything still works this session (results are held in memory),")
    print("  but they are lost on restart and re-fetching costs API quota.")
    print("  Fix: point BAPCA_CACHE_DIR at local storage, e.g. /content/bapca_cache,")
    print("  and copy it to Drive at the end of the session.")
    return False


def check_memory_layer(embedder) -> bool:
    banner("1. Memory layer -- the Review 1 failure case, done properly")

    store = EpisodicStore(RetrievalConfig())

    # A standing goal, stated once, early. This is the LoCoMo-Plus setup.
    goal = "I'm preparing for an important exam next month and want to minimise distractions."
    store.write(goal, embedder.encode_one(goal), when=START, weight=0.8,
                mem_type=MemoryType.GOAL)

    # Ordinary chatter that should fade away on its own.
    for day, text in [(2, "I had dosa for breakfast."),
                      (4, "It rained heavily in Lisbon today."),
                      (6, "I watched a cricket match last night.")]:
        store.write(text, embedder.encode_one(text), when=START + timedelta(days=day),
                    weight=0.3, mem_type=MemoryType.FACT)

    # A trigger query 30 days later with no lexical overlap with the goal.
    query = "Should I start watching that new TV series everyone is talking about?"
    now = START + timedelta(days=30)
    q_vec = embedder.encode_one(query)

    similarity = cosine(q_vec, store.nodes[0].embedding) if store.nodes else 0.0
    got = store.retrieve(q_vec, now)

    print(f"\nQuery: {query}")
    print(f"Cosine(query, the goal memory) = {similarity:.3f}")
    print(f"  -> a similarity retriever with a 0.25 floor would "
          f"{'FIND' if similarity >= 0.25 else 'MISS'} it")
    print(f"\nStanding memories carried:   {len(got.standing_hits)}")
    print(f"Similarity hits returned:    {len(got.similarity_hits)}")
    print(f"Injected context (~tokens):  {got.token_estimate()}")
    print(f"\n--- context handed to the LLM ---\n{got.as_context() or '(empty)'}")

    stats = store.stats(now)
    print(f"\nStore: {stats['retained']} retained, {stats['archived']} archived "
          f"(of {stats['total_written']} written)")

    ok = len(got.standing_hits) == 1
    print(f"\n{'PASS' if ok else 'FAIL'}: the goal "
          f"{'reached' if ok else 'did NOT reach'} the context without similarity.")
    return ok


def check_apis() -> bool:
    banner("2. Free API stack")
    from bapca.llm import generator, judge

    ok = True
    for label, factory in (("generator (Gemini)", generator), ("judge (Groq)", judge)):
        print(f"\n{label}")
        try:
            client = factory()
            print(f"  model chosen: {client.model}")
            print(f"  limits: RPM {client.limits.rpm}, TPM {client.limits.tpm:,}, "
                  f"RPD {client.limits.rpd}")
            fits = client.fits("x" * (21_201 * 4))
            print(f"  can send a 21,201-token LoCoMo prompt: "
                  f"{'yes' if fits else 'no (judge only needs short prompts)'}")

            reply = client.complete(
                "In one short sentence, what is the capital of France?", max_tokens=256
            )
            if not reply.strip():
                # An empty answer must never count as success. Our first run
                # reported "cache HIT (good)" while caching '' -- that would
                # have silently produced a table of empty responses.
                print("  FAIL: model returned an empty response")
                ok = False
                continue
            print(f"  reply: {reply[:90]!r}")

            # The bug that would have ruined every result: a reasoning model
            # emitting its chain of thought as the answer. The judge would then
            # have scored the monologue instead of the response.
            if "<think" in reply.lower() or reply.lower().startswith("here's a thinking"):
                print("  FAIL: chain-of-thought leaked into the answer")
                ok = False
                continue
            print("  reasoning leak: none (good)")

            calls_before = client.calls_made
            second = client.complete(
                "In one short sentence, what is the capital of France?", max_tokens=256
            )
            spent = client.calls_made - calls_before
            stats = client.cache.stats()

            # The only question that matters: did the repeat cost us quota?
            free_repeat = spent == 0 and second == reply
            print(f"  repeat call: {'served from cache (good)' if free_repeat else 'HIT THE API AGAIN'}")
            print(f"  cache stats: hits={stats['hits']} (memo={stats['memo_hits']}) "
                  f"misses={stats['misses']} disk_ok={stats['disk_ok']} "
                  f"entries={stats['entries_on_disk']}")
            print(f"  network calls this session: {client.calls_made}")
            ok = ok and free_repeat
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}")
            ok = False
    return ok


def show_available_models() -> None:
    banner("3. What each provider is actually serving")
    from bapca.llm import ROLES, LLM

    for role, (provider, _prefs, limits) in ROLES.items():
        try:
            client = LLM(role, model="placeholder", verbose=False)  # no probing
            ranked = client.rank_candidates()
            print(f"\n{role} -> {provider.name}  "
                  f"(RPM {limits.rpm}, TPM {limits.tpm:,}, RPD {limits.rpd})")
            for model_id in ranked[:6]:
                print(f"    {model_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n{role} ({provider.name}): could not list models ({str(exc)[:80]})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true",
                        help="also test the Groq and Gemini keys")
    parser.add_argument("--fake-embeddings", action="store_true",
                        help="skip the 80MB model download (logic check only)")
    args = parser.parse_args()

    embedder = HashEmbedder() if args.fake_embeddings else Embedder()
    if args.fake_embeddings:
        print("NOTE: using hash embeddings -- structure only, no real semantics.")

    results = {"cache": check_cache(), "memory layer": check_memory_layer(embedder)}
    if args.online:
        results["api stack"] = check_apis()
        show_available_models()

    banner("Summary")
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    if not args.online:
        print("\n  (run with --online once you have both API keys)")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
