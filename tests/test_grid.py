"""
Tests for the write-policy x selection grid (24 Oct).

Offline: no dataset, no API. The two risks guarded here are (1) a copied
prompt or retriever silently differing from Yuan et al.'s, and (2) an
accidental oracle -- the gold evidence reaching anything the system uses.
"""
import json

import pytest

from bapca.dataset import Sample
from bapca.embeddings import HashEmbedder
from bapca.memory import MemoryType
from bapca.pipeline import (FACTS_PROMPT, POOL_PROMPT, SELECT_MODES, SUMMARY_PROMPT,
                            WRITE_POLICIES, Note, SystemConfig, Window, bm25_rank,
                            bm25_scores, build_store, notes_from_window,
                            parse_facts, parse_summary, raw_chunks,
                            render_prompt, select_notes, select_pooled,
                            write_notes)

EMB = HashEmbedder()
EVIDENCE = "Caroline: After the burnout I learned to say no to extra projects."


def sample(trigger="A: I volunteered again and now I'm swamped."):
    return Sample(index=1, input_prompt="", trigger=trigger, evidence=EVIDENCE,
                  category="Cognitive", answer=None, time_gap="two weeks later")


class Recorder:
    def __init__(self, reply):
        self.reply, self.prompts, self.systems, self.budgets = reply, [], [], []

    def complete(self, prompt, system=None, max_tokens=None, **kw):
        self.prompts.append(prompt)
        self.systems.append(system)
        self.budgets.append(max_tokens)
        return self.reply


# ---- write policies -------------------------------------------------------

def test_policy_list_is_the_grid():
    assert set(WRITE_POLICIES) == {"v1", "events", "facts", "summary", "raw"}
    assert {"similar", "bm25", "pool"} <= set(SELECT_MODES)


def test_parse_facts_reads_every_fact_in_order():
    raw = json.dumps({"facts": [
        {"fact": "Caroline learned to say no after burnout", "speakers": ["Caroline"],
         "type": "event"},
        {"fact": "Caroline volunteers at a shelter", "speakers": ["Caroline"],
         "type": "personal_detail"}]})
    notes = parse_facts(raw, 7)
    assert [n.text for n in notes] == ["Caroline learned to say no after burnout",
                                       "Caroline volunteers at a shelter"]
    assert all(n.window_index == 7 and not n.typed for n in notes)
    assert all(n.mem_type is MemoryType.STATE for n in notes)


def test_parse_facts_keeps_complete_facts_from_a_truncated_list():
    raw = ('{"facts": [{"fact": "Jon bought flameless candles", "speakers": ["Jon"]},'
           ' {"fact": "Jon nearly started a fi')          # cut by the token budget
    assert [n.text for n in parse_facts(raw, 0)] == ["Jon bought flameless candles"]


def test_parse_facts_unescapes_and_survives_garbage():
    raw = '```json\n{"facts": [{"fact": "Mia said \\"never again\\" to night shifts"}]}\n```'
    assert parse_facts(raw, 0)[0].text == 'Mia said "never again" to night shifts'
    assert parse_facts('{"facts": []}', 0) == []
    assert parse_facts("I could not find anything.", 0) == []


def test_parse_summary_is_one_untyped_paragraph():
    note = parse_summary("  Caroline told Mel about her burnout.\n\nShe now says no. ", 3)
    assert note.text == "Caroline told Mel about her burnout. She now says no."
    assert note.window_index == 3 and not note.typed
    assert parse_summary("", 0) is None
    assert parse_summary("NONE", 0) is None


def test_raw_chunks_are_three_turns_and_cost_no_call():
    text = "\n".join(f"P{i}: line {i}" for i in range(7))
    chunks = raw_chunks(Window(4, text, False))
    assert [c.text.count("\n") + 1 for c in chunks] == [3, 3, 1]
    assert all(c.window_index == 4 and not c.typed for c in chunks)

    class Exploding:
        def complete(self, *a, **k):
            raise AssertionError("raw must not call the model")
    assert len(write_notes(Exploding(), [Window(4, text, False)], "raw")) == 3


def test_facts_and_summary_use_their_prompts_and_budgets():
    windows = [Window(0, "Caroline: I burned out.\nMel: Oh no.", False)]
    rec = Recorder('{"facts": [{"fact": "Caroline burned out"}]}')
    assert [n.text for n in write_notes(rec, windows, "facts")] == ["Caroline burned out"]
    assert "Caroline: I burned out." in rec.prompts[0]
    assert "Be thorough" in rec.prompts[0] and "valid JSON" in rec.systems[0]
    assert rec.budgets[0] > 160

    rec = Recorder("Caroline burned out and told Mel.")
    assert len(write_notes(rec, windows, "summary")) == 1
    assert "(unknown date)" in rec.prompts[0]
    assert "Do NOT omit details" in rec.prompts[0]


def test_v1_and_events_still_write_typed_notes():
    rec = Recorder("goal: Caroline is training for a race.")
    notes = write_notes(rec, [Window(0, "x", False)], "events")
    assert notes[0].typed and notes[0].mem_type is MemoryType.GOAL
    with pytest.raises(ValueError):
        write_notes(rec, [], "nonsense")


def test_prompts_carry_their_published_wording():
    # Spot-checks against github.com/boqiny/memory-probe @ 9289a05. A silent
    # edit to a copied prompt would make the comparison with their grid unfair.
    assert FACTS_PROMPT.startswith("You are a memory extraction agent.")
    assert "[event, preference, relationship, plan, personal_detail, opinion]" in FACTS_PROMPT
    assert "missing a fact means it's lost forever." in FACTS_PROMPT
    assert SUMMARY_PROMPT.startswith("You are a memory summarization agent.")
    assert "5. Any references to previous conversations or shared history" in SUMMARY_PROMPT
    assert POOL_PROMPT.startswith("You are ranking memory entries by relevance to a question.")
    assert "—" in FACTS_PROMPT and "—" in SUMMARY_PROMPT


# ---- rendering ------------------------------------------------------------

def test_untyped_notes_render_without_a_tag_typed_ones_keep_it():
    notes = [Note("Caroline burned out.", MemoryType.STATE, 0, typed=False),
             Note("Caroline values rest.", MemoryType.VALUE, 1)]
    store = build_store(notes, EMB, SystemConfig())
    lines = sorted(n.as_context_line() for n in store.nodes)
    assert lines == ["Caroline burned out.", "[value] Caroline values rest."]
    _, _, carried = render_prompt(sample(), store.nodes)
    assert "Caroline burned out." in carried
    assert not any(l.startswith("[state]") for l in carried)


# ---- selection ------------------------------------------------------------

def test_bm25_matches_rank_bm25():
    rank_bm25 = pytest.importorskip("rank_bm25")
    docs = ["Caroline learned to say no", "Mel went hiking with her kids",
            "say no to extra projects at work", "the dog ate the cake", "no no no"]
    query = "I said yes to extra projects"
    ours = bm25_scores(docs, query)
    ref = rank_bm25.BM25Okapi([d.lower().split() for d in docs]).get_scores(
        query.lower().split())
    assert ours == pytest.approx(list(ref))


def test_bm25_rank_puts_the_overlapping_note_first():
    notes = [Note(t, MemoryType.STATE, i, typed=False) for i, t in
             enumerate(["Mel likes pottery", "Caroline volunteers at the shelter",
                        "Jon fixed his bike"])]
    assert bm25_rank(notes, "Did Caroline volunteer at the shelter?")[0].window_index == 1


def _numbered(n):
    return [Note(f"note {i}", MemoryType.STATE, i, typed=False) for i in range(n)]


def test_pool_merges_cosine_then_bm25_and_reranks():
    base = _numbered(12)                       # one set of objects, two orders
    cos, bm = base, list(reversed(base))
    rec = Recorder('{"ranked_ids": ["5", "2"]}')
    chosen = select_pooled(rec, cos, bm, sample(), top_k=3)
    # merged = cos[:9] then bm[:9] minus duplicates -> 12 candidates
    assert rec.prompts[0].count("[ID: ") == 12
    assert [c.text for c in chosen[:2]] == ["note 4", "note 1"]
    assert len(chosen) == 3 and chosen[2].text == "note 0"   # padded in merged order
    assert EVIDENCE not in rec.prompts[0]                      # no oracle


def test_pool_skips_the_call_when_nothing_to_rank():
    class Exploding:
        def complete(self, *a, **k):
            raise AssertionError("no call needed")
    notes = _numbered(2)
    assert select_pooled(Exploding(), notes, notes, sample(), top_k=3) == notes


def test_pool_candidates_are_cut_to_200_characters():
    long = [Note("x" * 500, MemoryType.STATE, i, typed=False) for i in range(5)]
    rec = Recorder("[1, 2, 3]")
    select_pooled(rec, long, [], sample(), top_k=3)
    assert "x" * 201 not in rec.prompts[0] and "x" * 200 in rec.prompts[0]


def test_select_notes_runs_bm25_and_pool_through_the_store():
    notes = [Note(t, MemoryType.STATE, i, typed=False) for i, t in enumerate(
        ["Mel likes pottery", "Caroline is swamped by volunteering", "Jon fixed his bike",
         "Caroline said no to extra projects", "Mel's kids love camping"])]
    store = build_store(notes, EMB, SystemConfig())
    picked = select_notes(store, sample(), EMB, mode="bm25", top_k=2)
    assert len(picked) == 2
    rec = Recorder('{"ranked_ids": ["1"]}')
    picked = select_notes(store, sample(), EMB, mode="pool", top_k=3, llm=rec)
    assert len(picked) == 3 and len(rec.prompts) == 1
    assert EVIDENCE not in rec.prompts[0]
    with pytest.raises(ValueError):
        select_notes(store, sample(), EMB, mode="pool", top_k=3)


def test_oracle_window_returns_exactly_that_windows_notes():
    notes = [Note(f"n{i}", MemoryType.STATE, i // 2, typed=False) for i in range(6)]
    assert [n.text for n in notes_from_window(notes, 1)] == ["n2", "n3"]
    assert notes_from_window(notes, None) == []
    store = build_store(notes, EMB, SystemConfig())
    assert sorted(n.text for n in notes_from_window(store.nodes, 2)) == ["n4", "n5"]


# ---- grid_table.py ---------------------------------------------------------

import pathlib
import subprocess
import sys

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "grid_table.py"


def _write_cell(results, policy, selector, correct_ids, n_items=20, model=("g", "j")):
    suffix = "" if policy == "v1" else f"_{policy}"
    rows = [dict(sample_index=i, label="correct" if i in correct_ids else "wrong",
                 generator=model[0], judge=model[1]) for i in range(n_items)]
    (results / f"system_n20_seed42_{selector}3_d1_notrig{suffix}.json").write_text(
        json.dumps(rows))


def _run(tmp_path, *args):
    return subprocess.run([sys.executable, str(SCRIPT), "--n", "20", *args],
                          cwd=tmp_path, capture_output=True, text=True)


def test_grid_table_ranges_and_named_winner(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    # events writes well, v1 badly; selectors barely matter
    for sel in ("similar", "bm25", "pool"):
        _write_cell(res, "events", sel, set(range(14)))
        _write_cell(res, "v1", sel, set(range(4)))
    _write_cell(res, "events", "oracle_window", set(range(18)))
    out = _run(tmp_path, "--export")
    assert out.returncode == 0, out.stderr
    assert "across WRITE policies, selector cosine  :  50.0 points" in out.stdout
    assert "across SELECTORS, write policy events  :   0.0 points" in out.stdout
    assert "more correct: events" in out.stdout
    assert "events   ceiling  90.0%" in out.stdout
    grid = json.loads((res / "grid.json").read_text())
    assert grid["cells"]["events|similar"] == {"correct": 14, "total": 20}
    assert grid["tests"]["write@similar"]["best"] == "events|similar"


def test_grid_table_refuses_mixed_models(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _write_cell(res, "events", "similar", {1, 2}, model=("gemini-a", "qwen"))
    _write_cell(res, "v1", "similar", {1}, model=("gemini-b", "qwen"))
    out = _run(tmp_path)
    assert out.returncode == 1 and "REFUSING" in out.stdout


# ---- run_grid.py -------------------------------------------------------------

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("run_grid", pathlib.Path(__file__).resolve().parent.parent
                                     / "scripts" / "run_grid.py")
run_grid = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(run_grid)


def test_grid_plan_covers_every_cell_once_in_phase_order():
    allc = run_grid.cells()
    assert len(allc) == len(set(allc)) == 23
    plan = [(p, s) for p, s, v in allc if not v]
    assert len(plan) == 20
    assert set(plan) == {(p, s) for p in WRITE_POLICIES
                         for s in ("oracle_window", "similar", "pool", "bm25")}
    assert ("facts", "llm") not in plan     # over Groq's per-request ceiling
    first_new = min(plan.index(c) for c in plan if c[0] in ("facts", "summary"))
    assert all(c[0] in ("v1", "events", "raw") for c in plan[:first_new])
    assert all(c[1] == "bm25" for c in plan[-5:])
    variants = [c for c in allc if c[2]]
    assert variants == [("summary", "oracle_window", "nomerge"),
                        ("summary", "similar", "nomerge"), ("summary", "pool", "nomerge")]
    # the cheap nomerge ceiling runs before facts; the other two last
    assert allc.index(variants[0]) < allc.index(("facts", "oracle_window", ""))
    assert allc[-2:] == variants[1:]


def test_grid_files_are_the_names_run_system_and_grid_table_use():
    gt_spec = _ilu.spec_from_file_location("grid_table", SCRIPT)
    gt = _ilu.module_from_spec(gt_spec)
    gt_spec.loader.exec_module(gt)
    for pol, sel, var in run_grid.cells():
        if not var:
            assert run_grid.cell_file(pol, sel) == gt.cell_path(100, pol, sel)
    assert run_grid.cell_file("summary", "pool", "nomerge").name == \
        "system_n100_seed42_pool3_d1_notrig_summary_nomerge.json"
    assert run_grid.command("summary", "pool", "nomerge")[-1] == "--no-merge"
    assert "--no-merge" not in run_grid.command("summary", "pool")
    assert run_grid.cell_file("v1", "pool").name == "system_n100_seed42_pool3_d1_notrig.json"
    assert run_grid.cell_file("facts", "oracle_window").name == \
        "system_n100_seed42_oracle_window3_d1_notrig_facts.json"
    cmd = run_grid.command("facts", "pool")
    assert "--strip-trigger" in cmd and "--fresh" not in cmd
    assert cmd[cmd.index("--extract") + 1] == "facts"


def test_grid_refuses_to_resume_rows_from_an_old_write_budget(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    path = run_grid.cell_file("facts", "oracle_window")
    path.write_text(json.dumps([{"sample_index": 1, "label": "correct"}]))  # no budget: the 26 Oct row
    assert run_grid.stale_rows(path, "facts") == 1
    path.write_text(json.dumps([{"sample_index": 1, "write_budget": 2048}]))
    assert run_grid.stale_rows(path, "facts") == 1      # 2048 but the pre-6-Nov rule
    path.write_text(json.dumps([{"sample_index": 1, "write_budget": 2048,
                                 "facts_protocol": 3}]))
    assert run_grid.stale_rows(path, "facts") == 0
    path.write_text(json.dumps([{"sample_index": 1, "write_budget": 2048, "facts_protocol": 2,
                                 "facts_split_windows": 0, "facts_truncated_kept": 0}]))
    assert run_grid.stale_rows(path, "facts") == 0       # v2 without splits = v3
    summary = run_grid.cell_file("summary", "pool")
    summary.write_text(json.dumps([{"sample_index": 1, "write_budget": 2048}]))
    assert run_grid.stale_rows(summary, "summary") == 0    # the rule is facts-only
    assert run_grid.stale_rows(run_grid.cell_file("v1", "pool"), "v1") == 0


def test_write_budgets_match_their_default():
    from bapca.pipeline import WRITE_BUDGETS
    assert WRITE_BUDGETS == {"facts": 2048, "summary": 2048}
    plan = [(p, s) for p, s, v in run_grid.cells() if not v]
    assert max(plan.index(c) for c in plan if c[0] == "summary" and c[1] != "bm25") < \
        min(plan.index(c) for c in plan if c[0] == "facts" and c[1] != "bm25")


def test_grid_stops_before_running_anything_when_a_file_is_stale(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    stale = res / run_grid.cell_file("facts", "oracle_window").name
    stale.write_text(json.dumps([{"sample_index": 1989, "label": "correct"}]))
    out = subprocess.run([sys.executable, str(pathlib.Path(run_grid.__file__))],
                         cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 1 and "Nothing was run" in out.stdout
    assert ">>> cell" not in out.stdout


def test_grid_table_never_pairs_a_tied_cell_with_itself(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _write_cell(res, "summary", "similar", {1, 2, 3})
    _write_cell(res, "summary", "pool", {1, 2, 4})          # same rate, different items
    out = _run(tmp_path)
    assert out.returncode == 0, out.stderr
    line = [l for l in out.stdout.splitlines() if "policy summary" in l and "p=" in l][0]
    assert "cosine vs cosine" not in line and "hybrid vs hybrid" not in line
    assert "cosine vs hybrid" in line or "hybrid vs cosine" in line
    assert "more correct: neither" in line


def test_no_merge_keeps_near_duplicate_notes_apart():
    twins = [Note("Andrew and Audrey talked about birds.", MemoryType.STATE, 0, typed=False),
             Note("Andrew and Audrey talked about birds.", MemoryType.STATE, 5, typed=False)]
    merged = build_store(twins, EMB, SystemConfig())
    apart = build_store(twins, EMB, SystemConfig(merge=False))
    assert len(merged.nodes) == 1                      # default: pattern completion
    assert sorted(n.source_turn for n in apart.nodes) == [0, 5]
    assert notes_from_window(apart.nodes, 5) and not notes_from_window(merged.nodes, 5)
    assert SystemConfig(merge=False).retrieval.dedup_threshold > 1.0
    assert SystemConfig().retrieval.dedup_threshold == 0.85


def test_run_system_names_and_resumes_nomerge_runs():
    source = (pathlib.Path(__file__).parent.parent / "scripts" / "run_system.py").read_text()
    assert 'tag += "_nomerge"' in source and 'flags += " --no-merge"' in source
    assert "merge=not args.no_merge" in source


# ---- facts under Groq's output cap: split, never truncate (6 Nov) -----------

from bapca.llm import OutputCapExceeded
from bapca import pipeline as _pl


class CapLLM:
    """Refuses (OTPM) any facts request whose window has more than `cap_lines`
    lines; otherwise returns one fact per line. Records what it was asked."""

    def __init__(self, cap_lines, truncate_over=None):
        self.cap_lines, self.truncate_over, self.calls = cap_lines, truncate_over, []

    def complete(self, prompt, system=None, max_tokens=None, on_output_cap="shrink", **k):
        convo = prompt.split("Conversation:\n", 1)[1].split("\n\nReturn a JSON", 1)[0]
        lines = [l for l in convo.splitlines() if l.strip()]
        self.calls.append((len(lines), on_output_cap, max_tokens))
        if len(lines) > self.cap_lines and on_output_cap == "raise":
            raise OutputCapExceeded(1000, 1544, max_tokens)
        facts = [{"fact": f"said: {l}"} for l in lines]
        reply = json.dumps({"facts": facts})
        if self.truncate_over is not None and len(lines) > self.truncate_over:
            reply = reply[: len(reply) // 2]                     # cut off mid-list
        return reply


def _window(n):
    return Window(3, "\n".join(f"P{i}: line {i}" for i in range(n)), False)


def test_a_window_under_the_cap_is_one_call_as_before():
    before = dict(_pl.FACTS_STATS)
    llm = CapLLM(cap_lines=100)
    notes = write_notes(llm, [_window(8)], "facts")
    assert len(notes) == 8 and len(llm.calls) == 1
    assert llm.calls[0] == (8, "raise", 2048)
    assert _pl.FACTS_STATS == before


def test_a_refused_window_is_asked_again_whole_at_the_cap_not_split():
    before = dict(_pl.FACTS_STATS)
    llm = CapLLM(cap_lines=5)                              # refuses 8 lines at 2048
    notes = write_notes(llm, [_window(8)], "facts")
    assert [n.text for n in notes] == [f"said: P{i}: line {i}" for i in range(8)]
    assert all(n.window_index == 3 for n in notes)
    assert llm.calls == [(8, "raise", 2048), (8, "shrink", _pl.FACTS_CAPPED_BUDGET)]
    assert _pl.FACTS_STATS["capped_windows"] == before["capped_windows"] + 1
    assert _pl.FACTS_STATS["split_windows"] == before["split_windows"]


def test_a_cut_off_reply_is_split_in_halves_at_the_capped_budget():
    before = _pl.FACTS_STATS["split_windows"]
    llm = CapLLM(cap_lines=100, truncate_over=5)
    notes = write_notes(llm, [_window(8)], "facts")
    assert len(notes) == 8
    assert llm.calls == [(8, "raise", 2048), (4, "shrink", 936), (4, "shrink", 936)]
    assert _pl.FACTS_STATS["split_windows"] == before + 1


def test_splitting_stops_at_quarters_and_keeps_what_parsed():
    kept0 = _pl.FACTS_STATS["truncated_kept"]
    llm = CapLLM(cap_lines=100, truncate_over=1)           # every list > 1 line is cut off
    write_notes(llm, [_window(8)], "facts")
    assert [c[0] for c in llm.calls] == [8, 4, 2, 2, 4, 2, 2]
    assert all(c[2] == 936 for c in llm.calls[1:])          # parts never re-try 2048
    assert _pl.FACTS_STATS["truncated_kept"] == kept0 + 4


def test_protocol_is_recorded_and_versioned():
    assert _pl.FACTS_PROTOCOL == 3 and _pl.FACTS_MAX_SPLIT_DEPTH == 2
    assert _pl.FACTS_CAPPED_BUDGET == 936
    source = (pathlib.Path(__file__).parent.parent / "scripts" / "run_system.py").read_text()
    assert 'facts_protocol=FACTS_PROTOCOL if args.extract == "facts" else None' in source


def test_v2_rows_that_split_are_dropped_with_a_backup_and_the_rest_kept(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    path = run_grid.cell_file("facts", "oracle_window")
    rows = [{"sample_index": i, "write_budget": 2048, "facts_protocol": 2,
             "facts_split_windows": s, "facts_truncated_kept": 0}
            for i, s in ((1989, 0), (2014, 7), (2021, 0), (2026, 15))]
    path.write_text(json.dumps(rows))
    assert run_grid.drop_superseded(path, "facts") == 2
    assert [r["sample_index"] for r in json.loads(path.read_text())] == [1989, 2021]
    backups = list((tmp_path / "results").glob("OLD_superseded_*.json"))
    assert len(backups) == 1 and len(json.loads(backups[0].read_text())) == 4
    assert run_grid.stale_rows(path, "facts") == 0
    assert run_grid.drop_superseded(path, "facts") == 0            # idempotent
    assert run_grid.drop_superseded(run_grid.cell_file("v1", "pool"), "v1") == 0


def test_rule1_rows_are_still_refused_not_dropped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    path = run_grid.cell_file("facts", "oracle_window")
    path.write_text(json.dumps([{"sample_index": 1, "write_budget": 2048}]))   # no protocol
    assert run_grid.drop_superseded(path, "facts") == 0
    assert run_grid.stale_rows(path, "facts") == 1


def test_the_plan_names_the_code_version_first(tmp_path):
    (tmp_path / "results").mkdir()
    out = subprocess.run([sys.executable, str(pathlib.Path(run_grid.__file__)), "--plan"],
                         cwd=tmp_path, capture_output=True, text=True)
    first = out.stdout.splitlines()[0]
    assert first.startswith(f"grid code of {run_grid.CODE_VERSION}: facts rule v3")
