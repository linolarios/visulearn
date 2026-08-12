"""Unit tests for the eval scorer (eval/run_eval.py). No LLM required.

These prove the OR-group / forbidden-facts logic against the RBT golden fixture, so the
eval scoring can't silently rot. The eval *runner* (which needs a live model) stays out
of pytest; only the pure scoring functions are tested here.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))

from run_eval import score_output, _normalize_groups   # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "rbt_script.json"
RAW = FIXTURE.read_text()


def test_normalize_groups_accepts_str_and_list():
    assert _normalize_groups(["a", ["b", "c"]]) == [["a"], ["b", "c"]]
    assert _normalize_groups(None) == []


def test_golden_passes_gate_and_facts():
    s = score_output(RAW, expected_facts=[["order log n"], ["order n"]])
    assert s.parsed and s.gate_pass and s.facts_pass and s.drift_pass


def test_or_group_any_member_satisfies():
    # narration says "order log n" but not "logarithmic"; the OR-group still passes.
    s = score_output(RAW, expected_facts=[["order log n", "logarithmic"]])
    assert s.facts_pass and not s.facts_missing


def test_or_groups_are_anded_across():
    # second group can't be satisfied -> facts fail, and the missing group is reported.
    s = score_output(RAW, expected_facts=[["order log n"], ["order n squared"]])
    assert not s.facts_pass
    assert ["order n squared"] in s.facts_missing


def test_facts_flag_is_independent_of_gate():
    # a wrong fact does NOT flip gate_pass -- facts is a review flag, not a gate.
    s = score_output(RAW, expected_facts=[["totally absent phrase"]])
    assert s.gate_pass is True and s.facts_pass is False


def test_forbidden_facts_flag_drift():
    # RBT narration doesn't mention "balance factor" -> drift clean.
    clean = score_output(RAW, forbidden_facts=["balance factor"])
    assert clean.drift_pass and not clean.drift_hits
    # a term that IS present ("rotate") trips the drift flag, without failing the gate.
    hit = score_output(RAW, forbidden_facts=["rotate"])
    assert hit.gate_pass is True and hit.drift_pass is False and "rotate" in hit.drift_hits


def test_malformed_output_scores_unparsed():
    s = score_output("Sure, here's your script!", expected_facts=[["x"]])
    assert not s.parsed and s.error and s.facts_pass  # facts default True when unparsed


def test_too_few_segments_fails_gate_only():
    d = json.loads(RAW)
    d["segments"] = d["segments"][:4]
    s = score_output(json.dumps(d))
    assert s.parsed and not s.gate_pass and s.facts_pass and s.drift_pass