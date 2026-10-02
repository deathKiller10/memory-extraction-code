"""27 Sep: the extraction-only transfer test (HANDOFF 16.4). Offline, no dataset."""
import importlib.util
import io
import contextlib
import json
import pathlib

from bapca.dataset import Sample
from bapca.embeddings import HashEmbedder

SCRIPTS = pathlib.Path(__file__).parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_t", SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CONV = ("DATE: 1:56 pm on 8 May, 2023\nCONVERSATION:\n"
        + "\n".join(f'Caroline said, "line {i} about the charity race and mental health"'
                    for i in range(30))
        + "\nDATE: 2:00 pm on 9 May, 2023\nCONVERSATION:\n"
        + "\n".join(f'Melanie said, "second session line {i}"' for i in range(30)))


def _sample(i, q, cat="single-hop", evidence="Caroline：line 3", answer="mental health"):
    return Sample(index=i, input_prompt=CONV + f"\n\nQuestion: {q}", trigger=q,
                  evidence=evidence, category=cat, answer=answer)


def test_only_the_literal_question_line_is_removed_even_when_short():
    eo = _load("extract_only")
    short = _sample(1, "How old is Max?")          # strip_trigger refuses < 5 words
    assert eo.conversation_text(short) == CONV.rstrip("\n")
    odd = Sample(index=2, input_prompt=CONV + "\nsomething else", trigger="How old is Max?",
                 evidence="x", category="single-hop")
    assert eo.conversation_text(odd).endswith("something else")   # never eats other lines


def test_questions_about_one_conversation_share_one_store():
    eo = _load("extract_only")
    groups = eo.group_by_conversation([_sample(i, q) for i, q in
                                       enumerate(["How old is Max?", "Who is Jill?",
                                                  "What did the charity race raise awareness for?"])])
    assert len(groups) == 1


def test_windows_are_cut_like_cognitive_with_no_date_headers():
    eo = _load("extract_only")
    ws = eo.windows_like_cognitive(CONV)
    assert len(ws) > 2                     # not one window per session
    for w in ws:
        assert "DATE:" not in w.text and "CONVERSATION:" not in w.text
        assert not w.dated


def _write(path, prompt, notes, model="m1"):
    rows = [dict(sample_index=i, category="single-hop", extract=prompt, conversation=0,
                 model=model, all_notes=notes) for i in range(12)]
    path.write_text(json.dumps(rows))


def test_written_compare_names_the_winning_file(tmp_path):
    wc = _load("written_compare")
    ev = "Caroline：line 3 about the charity race and mental health"
    samples = {i: _sample(i, "What did the race raise awareness for?", evidence=ev)
               for i in range(12)}
    _write(tmp_path / "extract_only_v1.json", "v1", ["[state] Caroline likes tea."])
    _write(tmp_path / "extract_only_events.json", "events",
           ["[state] Caroline said line 3 about the charity race and mental health"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        wc.main([str(tmp_path / "extract_only_v1.json"),
                 str(tmp_path / "extract_only_events.json")],
                embedder=HashEmbedder(), samples=samples)
    out = buf.getvalue()
    assert "better: extract_only_events.json" in out
    assert " A better" not in out and " B better" not in out
    assert "DIFFERENT MODELS" not in out
    assert "conversations won" in out


def test_written_compare_shouts_when_the_models_differ(tmp_path):
    wc = _load("written_compare")
    samples = {i: _sample(i, "q?") for i in range(12)}
    _write(tmp_path / "a.json", "v1", ["x"], model="m1")
    _write(tmp_path / "b.json", "events", ["x"], model="m2")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        wc.main([str(tmp_path / "a.json"), str(tmp_path / "b.json")],
                embedder=HashEmbedder(), samples=samples)
    assert "DIFFERENT MODELS" in buf.getvalue()


def test_answer_in_a_note_skips_yes_no_and_needs_one_note():
    wc = _load("written_compare")
    assert wc.answer_in_a_note("Yes", ["yes indeed"]) is None
    assert wc.answer_in_a_note("Bach and Mozart", ["She loves Bach", "and Mozart"]) is False
    assert wc.answer_in_a_note("Bach and Mozart", ["She listens to Bach and Mozart"]) is True


def test_written_compare_reads_run_system_files_without_conversation_ids(tmp_path):
    """2 Oct: the Cognitive v1-vs-events contrast uses run_system.py files,
    which carry no 'conversation' or 'model' field."""
    wc = _load("written_compare")
    ev = "Caroline：line 3 about the charity race and mental health"
    samples = {i: _sample(i, "q?", cat="Cognitive", evidence=ev) for i in range(12)}
    for name, note in (("v1.json", "[state] Caroline likes tea."),
                       ("ev.json", "[state] Caroline said line 3 about the charity race and mental health")):
        rows = [dict(sample_index=i, all_notes=[note], carried=[]) for i in range(12)]
        (tmp_path / name).write_text(json.dumps(rows))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        wc.main([str(tmp_path / "v1.json"), str(tmp_path / "ev.json")],
                embedder=HashEmbedder(), samples=samples)
    out = buf.getvalue()
    assert "model not recorded" in out
    assert "conversation  0" in out and "conversation  1" not in out   # one base conversation


def test_export_records_prompts_and_paper_numbers_orients_by_prompt_name(tmp_path, monkeypatch):
    """2 Oct: the transfer figures reach the paper as macros. Which side is the
    events prompt must come from the prompt names, never from argument order."""
    import runpy
    wc = _load("written_compare")
    monkeypatch.chdir(tmp_path)
    ev = "Caroline：line 3 about the charity race and mental health"
    samples = {i: _sample(i, "q?", evidence=ev) for i in range(12)}
    _write(tmp_path / "extract_only_events.json", "events",
           ["[state] Caroline said line 3 about the charity race and mental health"])
    _write(tmp_path / "extract_only_v1.json", "v1", ["[state] Caroline likes tea."])
    with contextlib.redirect_stdout(io.StringIO()):
        # events FIRST on purpose
        wc.main([str(tmp_path / "extract_only_events.json"),
                 str(tmp_path / "extract_only_v1.json"), "--export", "transfer"],
                embedder=HashEmbedder(), samples=samples)
    e = json.loads((tmp_path / "results" / "written_transfer.json").read_text())
    assert e["prompt_a"] == ["events"] and e["prompt_b"] == ["v1"] and e["n"] == 12
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(SCRIPTS / "paper_numbers.py"), run_name="__main__")
        except SystemExit:
            pass
    tex = (tmp_path / "paper" / "numbers.tex").read_text()
    events_rate = 100 * e["written_a"] / 12
    assert "\\newcommand{\\TransferEvents}{%.1f}" % events_rate in tex
    assert "\\newcommand{\\TransferVOne}{%.1f}" % (100 * e["written_b"] / 12) in tex
    assert "\\CogWrittenN}{\\todo" in tex          # missing export stays loud
