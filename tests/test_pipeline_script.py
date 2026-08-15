"""Offline tests for the front-half orchestrator make_script (AGENT.md §6/§7).

No network, no real LLM, no heavy renderer/TTS deps — a stubbed Script Engine provider
plus an injected fact sheet drive the whole Topic -> validated Script JSON + Storyboard
path, so CI (which installs only pydantic/pytest/requests) stays green. Raw-draft
artifacts are redirected into tmp_path so the repo is never written to.
"""
import json
from pathlib import Path

import pytest

import pipeline
import script_engine
from models import Script
from input_gateway import TopicError
from script_engine import ScriptEngineError

FIXTURE = Path(__file__).parent / "fixtures" / "rbt_script.json"
VALID = FIXTURE.read_text()  # 11 segments; opens title_card, closes, passes the gate


@pytest.fixture(autouse=True)
def _no_repo_writes(monkeypatch, tmp_path):
    """Any repair/hard-fail raw artifacts land in tmp_path, never the repo."""
    monkeypatch.setattr(script_engine, "OUTPUT_SCRIPTS_DIR", tmp_path / "raw")


class _FakeScriptProvider:
    """Fulfils the Provider seam; never the network.

    generate_script calls adapt_schema() (AGENT.md §3 — the OUTBOUND schema is adapted
    per provider before the request) and then complete(). Both are provided here.
    """

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
        self.seen_schemas = []

    def adapt_schema(self, schema):
        """Pass the canonical schema through unchanged — a neutral stub."""
        self.seen_schemas.append(schema)
        return schema

    def complete(self, messages, schema):
        idx = self.calls
        self.calls += 1
        if idx >= len(self.outputs):
            raise AssertionError("provider called more times than outputs provided")
        return self.outputs[idx]


def test_make_script_writes_validated_json_and_storyboard(tmp_path):
    provider = _FakeScriptProvider([VALID])
    out = tmp_path / "rbt.json"

    path, storyboard = pipeline.make_script(
        "Red-Black Tree", out, script_provider=provider, fact_sheet={"definition": "x"},
    )

    assert path == out and out.exists()
    script = Script.model_validate_json(out.read_text())   # written artifact is valid Script
    assert len(script.segments) == 11
    assert provider.calls == 1                              # one generation, no repair
    assert len(storyboard.scenes) == len(script.segments)   # every segment -> one scene
    assert {s.segment_id for s in storyboard.scenes} == {s.id for s in script.segments}


def test_fact_sheet_callable_receives_canonical_topic(tmp_path):
    provider = _FakeScriptProvider([VALID])
    seen = {}

    def facts(topic):
        seen["topic"] = topic
        return {"definition": "ok"}

    pipeline.make_script(
        "red-black tree", tmp_path / "rbt.json", script_provider=provider, fact_sheet=facts,
                          )

    assert seen["topic"] == "Red-Black Tree"   # input normalized to the canonical name first


def test_fact_sheet_dict_used_verbatim(tmp_path):
    provider = _FakeScriptProvider([VALID])
    sheet = {"definition": "exact", "year": "1972"}

    path, _ = pipeline.make_script("Red-Black Tree", tmp_path / "x.json",
                                   script_provider=provider, fact_sheet=sheet)

    assert path.exists() and provider.calls == 1


def test_unknown_topic_rejected_no_file(tmp_path):
    out = tmp_path / "x.json"
    with pytest.raises(TopicError):
        pipeline.make_script("quantum computing", out,
                             script_provider=_FakeScriptProvider([VALID]))
    assert not out.exists()


def test_script_hard_fail_raises_no_file(tmp_path):
    # 4 segments (fails the 8-12 gate, no title/closing) -> engine hard-fails after repair.
    bad = dict(json.loads(VALID))
    bad["segments"] = bad["segments"][3:7]
    payload = json.dumps(bad)
    out = tmp_path / "x.json"

    with pytest.raises(ScriptEngineError):
        pipeline.make_script("Red-Black Tree", out,
                             script_provider=_FakeScriptProvider([payload, payload]),
                             fact_sheet={"definition": "x"})
    assert not out.exists()   # failed generation must not leave a half-baked artifact
