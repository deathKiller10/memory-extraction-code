"""Tests for scripts/reconcile_judge.py (offline, synthetic data)."""
import importlib.util
import json
import pathlib
import random

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "reconcile_judge",
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "reconcile_judge.py")
rj = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rj)


def test_calibrate_recovers_known_rate():
    # judge says correct on 60%; humans agree on half of those, and on 10% of the rest
    pairs = [(True, True)] * 30 + [(True, False)] * 30 + \
            [(False, True)] * 4 + [(False, False)] * 36
    ppv, fo, hat = rj.calibrate(pairs, judge_rate=0.6)
    assert ppv == pytest.approx(0.5)
    assert fo == pytest.approx(0.1)
    assert hat == pytest.approx(0.6 * 0.5 + 0.4 * 0.1)


def test_calibrate_is_not_the_raw_annotated_rate():
    # annotated items over-sample judge-correct ones; calibration must undo that
    pairs = [(True, True)] * 40 + [(False, False)] * 10
    raw = sum(h for _, h in pairs) / len(pairs)
    _, _, hat = rj.calibrate(pairs, judge_rate=0.2)
    assert raw == pytest.approx(0.8)
    assert hat == pytest.approx(0.2)


def test_calibrate_refuses_one_sided_pairs():
    with pytest.raises(ValueError):
        rj.calibrate([(True, True), (True, False)], 0.5)


def test_annotators_skip_joint_and_pool_by_prefix(tmp_path):
    (tmp_path / "annotator2.json").write_text(json.dumps({"a:1": "correct"}))
    (tmp_path / "annotator2_solo.json").write_text(json.dumps({"a:2": "wrong"}))
    (tmp_path / "annotator2_batch2.json").write_text(json.dumps({"a:3": "correct",
                                                                "a:4": "skip"}))
    (tmp_path / "annotator1_solo.json").write_text(json.dumps({"a:2": "correct"}))
    people = rj.annotators(tmp_path)
    assert sorted(people) == ["annotator1", "annotator2"]
    assert people["annotator2"] == {"a:2": False, "a:3": True}   # joint and 'skip' excluded
    assert people["annotator1"] == {"a:2": True}


def test_bootstrap_interval_contains_point_estimate():
    pairs = [(True, True)] * 20 + [(True, False)] * 10 + [(False, False)] * 15 + [(False, True)] * 5
    target = [True] * 50 + [False] * 50
    _, _, hat = rj.calibrate(pairs, 0.5)
    lo, hi = rj.bootstrap(pairs, target, random.Random(1), reps=500)
    assert lo <= hat <= hi
