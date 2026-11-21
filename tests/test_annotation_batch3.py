"""
20 Nov: annotation batch 3. The calibration in reconcile_judge.py is valid only
if annotated items are drawn at random within a condition, independently of
the judge's verdict; batch 3 must also be PAIRED (each item under full context
and under ours) and lie OUTSIDE the n=100 subset. Uses the shipped result files.
"""
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_b3", ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def in_root(monkeypatch):
    if not (ROOT / "results" / "system_n401_seed42_llm3_d1_notrig_events.json").exists():
        pytest.skip("401-item result files not present")
    monkeypatch.chdir(ROOT)


def test_batch3_is_paired_outside_the_subset_and_label_blind(in_root, capsys):
    b = _load("build_annotation_batch")
    q = b.batch3_queue(done=set())
    keys = [f"{r['condition']}:{r['sample_index']}" for r in q]
    assert len(q) == 140 and len(set(keys)) == 140
    fc = {r["sample_index"] for r in q if r["condition"] == "full_context"}
    ours = {r["sample_index"] for r in q if r["condition"] == "system_llm"}
    nomem = [r for r in q if r["condition"] == "no_memory"]
    assert fc == ours and len(fc) == 60 and len(nomem) == 20
    subset = {r["sample_index"] for r in json.loads(
        (ROOT / "results" / "system_n100_seed42_llm3_d1_notrig_events.json").read_text(encoding="utf-8"))}
    assert not fc & subset
    assert [r["sample_index"] for r in b.batch3_queue(done=set())] == [r["sample_index"] for r in q]


def test_judge_labels_agree_across_the_100_and_401_files(in_root):
    rj = _load("reconcile_judge")
    labels = rj.judge_labels(rj.CALIBRATION_FILES)      # asserts on any disagreement
    assert sum(k.startswith("system_llm:") for k in labels) == 401
    assert sum(k.startswith("full_context:") for k in labels) == 401
