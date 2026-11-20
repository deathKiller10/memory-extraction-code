"""
19 Nov: the second answering model (Cohere Command A) and what it needed.

  * generator2 is Cohere with the model PINNED (no list/probe calls out of a
    1,000-call trial month, no drift), and Cohere gets its system prompt in
    the "developer" role.
  * Reasoning-capable models (Mistral, others) can return the message content as a
    LIST of chunks; only text chunks are the answer.
  * A Groq output-cap refusal is remembered, so re-running a facts cell does
    not pay ~1,000 refused requests again.
  * Every results file of a second-model run is distinct from the paper's files,
    and the printed resume command carries --generator.
"""
import argparse
import importlib.util
import pathlib

import pytest

from bapca.cache import DiskCache, clear_memo
from bapca.llm import (LLM, ROLES, Limits, OutputCapExceeded, content_text,
                       _date_suffix)

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rank(role, available):
    client = LLM.__new__(LLM)
    client.role = role
    client._prefs = ROLES[role][1]
    return LLM.rank_candidates(client, available)


def test_generator2_is_cohere_command_a_pinned(monkeypatch):
    from bapca import llm
    assert ROLES["generator2"][0].name == "cohere"
    assert ROLES["generator2"][0].base_url == "https://api.cohere.ai/compatibility/v1"
    assert ROLES["generator2"][2].rpm < 20                      # under the trial's 20/min
    made = {}
    monkeypatch.setattr(llm, "LLM", lambda role, model, verbose: made.update(role=role, model=model))
    llm.generator2(verbose=False)
    assert made == {"role": "generator2", "model": "command-a-03-2025"}
    served = ["command-a-03-2025", "command-a-plus-05-2026", "command-a-reasoning-08-2025",
              "command-r-plus", "embed-v4.0"]
    assert _rank("generator2", served)[0] == "command-a-03-2025"


def test_cohere_gets_the_system_prompt_in_the_developer_role(tmp_path, monkeypatch):
    client = _fake_llm(tmp_path, monkeypatch)
    seen = []
    client._call_with_backoff = lambda messages, *a, **k: seen.append(messages) or "ok"
    client.provider = ROLES["generator2"][0]
    LLM.complete(client, "q", system="be brief", max_tokens=10)
    client.provider = ROLES["judge"][0]
    LLM.complete(client, "q2", system="be brief", max_tokens=10)
    assert [m[0]["role"] for m in seen] == ["developer", "system"]


def test_date_tiebreak_leaves_the_existing_roles_alone():
    assert _rank("judge", ["qwen/qwen3.6-27b", "qwen/qwen3.8-27b"])[0] == "qwen/qwen3.8-27b"
    assert _date_suffix("models/gemini-3.5-flash-lite") == 0
    assert _date_suffix("mistral-small-2603") == 2603    # generic YYMM tiebreak


def test_content_text_keeps_only_text_chunks():
    assert content_text("plain") == "plain"
    assert content_text(None) == ""
    chunks = [{"type": "thinking", "thinking": [{"type": "text", "text": "hmm"}]},
              {"type": "text", "text": "The "}, {"type": "text", "text": "answer."}]
    assert content_text(chunks) == "The answer."

    class Chunk:
        def __init__(self, type, text=None):
            self.type, self.text = type, text
    assert content_text([Chunk("thinking"), Chunk("text", "ok")]) == "ok"


def _fake_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("BAPCA_CACHE_DIR", str(tmp_path))
    clear_memo()
    client = LLM.__new__(LLM)
    client.provider = ROLES["judge"][0]
    client.model = "qwen/test"
    client.temperature = 0.0
    client.cache = DiskCache("llm/judge-test")
    client.calls = 0

    def refuse(messages, temperature, max_tokens, on_output_cap="shrink"):
        client.calls += 1
        if on_output_cap == "raise":
            raise OutputCapExceeded(1000, 1544, max_tokens)
        return "short reply"
    client._call_with_backoff = refuse
    return client


def test_output_cap_refusal_is_remembered(tmp_path, monkeypatch):
    client = _fake_llm(tmp_path, monkeypatch)
    for _ in range(3):
        with pytest.raises(OutputCapExceeded) as info:
            LLM.complete(client, "window text", system="s", max_tokens=2048,
                         on_output_cap="raise")
        assert (info.value.limit, info.value.requested, info.value.budget) == (1000, 1544, 2048)
    assert client.calls == 1                          # refused once, then remembered
    clear_memo()                                      # and it survives a restart (disk)
    with pytest.raises(OutputCapExceeded):
        LLM.complete(client, "window text", system="s", max_tokens=2048, on_output_cap="raise")
    assert client.calls == 1
    # the fallback call is a different request and still runs and caches normally
    assert LLM.complete(client, "window text", system="s", max_tokens=936) == "short reply"
    assert client.calls == 2


def test_shrink_mode_calls_are_untouched(tmp_path, monkeypatch):
    client = _fake_llm(tmp_path, monkeypatch)
    assert LLM.complete(client, "p", max_tokens=300) == "short reply"
    assert LLM.complete(client, "p", max_tokens=300) == "short reply"
    assert client.calls == 1                          # ordinary cache hit


def _system_args(**kw):
    base = dict(n=100, seed=42, select="llm", top_k=3, carry=0, window_turns=12,
                days_per_window=1.0, strip_trigger=True, extract="events", rerank="v1",
                no_merge=False, category="Cognitive", generator="gemini")
    base.update(kw)
    return argparse.Namespace(**base)


def test_resume_command_carries_the_generator():
    rs = _load("run_system")
    assert "--generator" not in rs.resume_command(_system_args())
    assert rs.resume_command(_system_args(generator="cohere")).endswith("--generator cohere")


def test_second_model_files_never_collide_with_the_papers():
    rsm = _load("run_second_model")
    rg = _load("run_grid")
    paper = {rg.cell_file(p, s, v).name for p, s, v in rg.cells()}
    paper |= {"pilot_n100_seed42.json", "system_n100_seed42_llm3_d1_notrig_events.json"}
    ours = {rsm.target(step).name for step in rsm.QUEUE}
    assert len(ours) == len(rsm.QUEUE) and not ours & paper
    assert all("gencohere" in name for name in ours)
    assert rsm.target(("pilot", None)).name == \
        "pilot_n100_seed42_full_context+no_memory_gencohere.json"
    assert rsm.target(("events", "llm")).name == \
        "system_n100_seed42_llm3_d1_notrig_events_gencohere.json"
    assert rsm.target(("v1", "pool")).name == "system_n100_seed42_pool3_d1_notrig_gencohere.json"


def test_second_model_commands_match_the_grid_cells():
    rsm = _load("run_second_model")
    rg = _load("run_grid")
    cmd = rsm.command(("facts", "pool"))
    assert cmd[1:] == rg.command("facts", "pool")[1:] + ["--generator", "cohere"]
    pilot = rsm.command(("pilot", None))
    assert pilot[1] == "scripts/run_pilot.py" and pilot[-2:] == ["--generator", "cohere"]
    assert "no_memory" in pilot and "full_context" in pilot


# ---- the free cache check (run_second_model.py --check) --------------------

CONV = ("DATE: 1:56 pm on 8 May, 2023\nCONVERSATION:\n"
        + "\n".join(f'Caroline said, "line {i} about the charity race"' for i in range(30))
        + "\nDATE: 2:00 pm on 9 May, 2023\nCONVERSATION:\n"
        + "\n".join(f'Melanie said, "second session line {i}"' for i in range(30)))


def _writer(tmp_path, monkeypatch, refuse_first_window=True):
    """A judge whose 'network' answers every write prompt, refusing the first
    facts window at 2048 as Groq's output cap did (rule v3)."""
    client = _fake_llm(tmp_path, monkeypatch)
    seen = {"raise": 0}

    def answer(messages, temperature, max_tokens, on_output_cap="shrink"):
        client.calls += 1
        if on_output_cap == "raise":
            seen["raise"] += 1
            if refuse_first_window and seen["raise"] == 1:
                raise OutputCapExceeded(1000, 1544, max_tokens)
            return '{"facts": ["Caroline ran a charity race."]}'
        prompt = messages[-1]["content"]
        if '"facts"' in prompt or "facts" in prompt.lower()[:200]:
            return '{"facts": ["Caroline ran a charity race."]}'
        return "[event] Caroline ran a charity race and now trains weekly."
    client._call_with_backoff = answer
    return client


def test_check_cache_passes_when_warm_and_writes_nothing(tmp_path, monkeypatch):
    from bapca.dataset import Sample
    from bapca.pipeline import segment, write_notes
    rsm = _load("run_second_model")
    rs = _load("run_system")
    sample = Sample(index=7, input_prompt=CONV + "\n\nI am exhausted after the race.",
                    trigger="I am exhausted after the race.", evidence="x",
                    category="Cognitive")
    jdg = _writer(tmp_path, monkeypatch)
    windows = segment(rs._source(sample, True), 12)
    for policy in ("events", "facts", "summary"):
        write_notes(jdg, windows, policy)              # the original run fills the cache
    # The real Drive cache predates refusal markers (19 Nov): remove them, so
    # the check must recognise the capped window on its own.
    import json as _json
    markers = [f for f in tmp_path.rglob("*.json")
               if _json.loads(f.read_text()).get("meta", {}).get("refused") == "otpm"]
    assert len(markers) == 1
    for f in markers:
        f.unlink()
    clear_memo()
    files_before = sorted(p.name for p in tmp_path.rglob("*") if p.is_file())
    calls_before = jdg.calls
    assert rsm.check_cache(jdg=jdg, samples=[sample]) is True
    assert jdg.calls == calls_before                  # nothing reached the 'network'
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == files_before


def test_check_cache_fails_when_cold(tmp_path, monkeypatch):
    from bapca.dataset import Sample
    rsm = _load("run_second_model")
    sample = Sample(index=7, input_prompt=CONV + "\n\nI am exhausted after the race.",
                    trigger="I am exhausted after the race.", evidence="x",
                    category="Cognitive")
    jdg = _writer(tmp_path, monkeypatch)
    assert rsm.check_cache(jdg=jdg, samples=[sample]) is False
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]   # and wrote nothing


# ---- analysis scripts --------------------------------------------------------

def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    path.write_text(_json.dumps(rows))


def test_second_model_table_pairs_by_item_and_refuses_mixed_models(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    tab = _load("second_model_table")
    rsm = _load("run_second_model")
    monkeypatch.setitem(__import__("sys").modules, "run_second_model", rsm)
    R = tmp_path / "results"
    pilot = [{"condition": c, "sample_index": i, "label": lab}
             for c, lab in (("no_memory", "wrong"), ("full_context", "wrong")) for i in range(10)]
    _write(R / "pilot_n100_seed42.json", pilot)
    _write(rsm.target(("pilot", None)), [{**r, "generator": "command-a-03-2025"} for r in pilot])
    ours = [{"sample_index": i, "label": "correct", "generator": "command-a-03-2025"}
            for i in range(10)]
    _write(R / "system_n100_seed42_llm3_d1_notrig_events.json", ours)
    _write(rsm.target(("events", "llm")), ours)
    assert tab.main() == 0
    out = capsys.readouterr().out
    assert "ours_vs_no_memory      10 vs 0 discordant" in out
    _write(rsm.target(("raw", "pool")), [{"sample_index": 0, "label": "wrong",
                                          "generator": "other-model"}])
    assert tab.main() == 1                       # mixed answering models: refuse


def test_rejudge_report_counts_agreement_and_pairs(capsys):
    rj = _load("rejudge")
    rows = []
    for i in range(20):
        rows.append({"condition": "no_memory", "sample_index": i, "label": "wrong",
                     "label2": "wrong", "judge2": "gpt-oss"})
        rows.append({"condition": "ours", "sample_index": i, "label": "correct",
                     "label2": "correct" if i < 15 else "wrong", "judge2": "gpt-oss"})
    rows.append({"condition": "ours", "sample_index": 99, "label": "correct",
                 "label2": "unparsed", "judge2": "gpt-oss"})
    out = rj.report(rows)
    assert out["n_scored"] == 40 and out["conditions"]["ours"]["judge2"] == 75.0
    p = out["paired"]["ours_vs_no_memory_judge2"]
    assert (p["n"], p["only_a"], p["only_b"]) == (20, 15, 0)
