"""Script Engine tests using a STUBBED provider no network, no API key (AGENT.md 7).

Proves all three branches of Golden Rule 1:
  * valid first try            -> LLM called exactly ONCE
  * invalid -> repair -> valid -> LLM called exactly TWICE
  * invalid -> repair -> invalid -> hard-fail, LLM called exactly TWICE (never a third time),
    raw output logged.
Also unit-tests the per-provider OUTBOUND schema adaptation (Ollama / Gemini / Groq).
"""
import json
import logging
import os
from pathlib import Path

import pytest

from models import Script, canonical_schema
from script_engine import (
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    Provider,
    ProviderError,
    ProviderResponse,
    ScriptEngineError,
    _load_dotenv,
    _normalize_finish_reason,
    _truncation_hint,
    build_provider,
    generate_script,
)

FIXTURE = Path(__file__).parent / "fixtures" / "rbt_script.json"
VALID_SCRIPT = FIXTURE.read_text()  # 11 segments, opens title_card, closes, passes the gate


class FakeProvider(Provider):
    """Stubbed LLM: returns queued raw strings, counts calls, never touches the network."""

    name = "fake"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.call_count = 0
        self.seen_messages = []
        self.seen_schema = []

    def complete(self, messages, schema):
        self.call_count += 1
        self.seen_schema.append(schema)
        if self.call_count > len(self.outputs):
            # The engine must never issue a third call. Fail loudly if it does.
            raise AssertionError(
                f"fake provider called a {self.call_count}rd time repair loop ran away"
            )
        self.seen_messages.append(messages)
        return self.outputs[self.call_count - 1]


def _gate_invalid_copy():
    """A structurally-valid Script that fails the 8-12 segment gate (drops closing/intro)."""
    d = json.loads(VALID_SCRIPT)
    d["segments"] = d["segments"][3:7]  # 4 segments, no title_card, no closing
    return json.dumps(d)


@pytest.fixture
def schema() -> dict:
    return json.loads(
        (Path(__file__).parent.parent / "config" / "schemas" / "script_schema.json").read_text()
    )


def test_valid_first_try_calls_llm_once(schema):
    fake = FakeProvider([VALID_SCRIPT])
    script = generate_script("Red-Black Tree", {"definition": "x"}, provider=fake, schema=schema)
    assert isinstance(script, Script)
    assert len(script.segments) == 11
    assert fake.call_count == 1  # success path: exactly one call, never a repair


def test_invalid_then_repair_then_valid_calls_llm_twice(schema):
    fake = FakeProvider([_gate_invalid_copy(), VALID_SCRIPT])
    script = generate_script("Red-Black Tree", {"definition": "x"}, provider=fake, schema=schema)
    assert isinstance(script, Script)
    assert len(script.segments) == 11
    assert fake.call_count == 2  # exactly one repair -> second call
    # repair prompt must be a DIFFERENT context (assistant's bad draft + correction),
    # not the identical original prompt (Golden Rule 1).
    repair_user = fake.seen_messages[1][-1]["content"]
    assert "rejected" in repair_user and "8-12 segments" in repair_user


def test_invalid_then_repair_then_invalid_hard_fails_and_logs_raw(schema, caplog):
    fake = FakeProvider([_gate_invalid_copy(), _gate_invalid_copy()])
    with caplog.at_level(logging.ERROR, logger="visulearn.script_engine"):
        with pytest.raises(ScriptEngineError) as exc:
            generate_script("Red-Black Tree", {"definition": "x"}, provider=fake, schema=schema)
    assert fake.call_count == 2  # EXACTLY two calls, never a third
    # Raw output from the LAST attempt is logged so the failure is debuggable.
    assert "script engine hard-fail" in caplog.text
    assert _gate_invalid_copy()[:40] in caplog.text
    assert "8-12 segments" in str(exc.value)


def test_structural_failure_also_repairs_then_valid(schema):
    # First output is not even JSON -> structural failure; repair returns a valid script.
    fake = FakeProvider(["this is prose, not JSON", VALID_SCRIPT])
    script = generate_script("Stack", {}, provider=fake, schema=schema)
    assert isinstance(script, Script)
    assert fake.call_count == 2


def test_provider_error_bubbles_readably():
    # Simulate the upstream failure class: provider returns an error / no JSON object.
    class Exploding(FakeProvider):
        def complete(self, messages, schema):
            raise ProviderError("Ollama unreachable at http://127.0.0.1:11434 (daemon down)")

    with pytest.raises(ProviderError, match="daemon down"):
        generate_script("Stack", {}, provider=Exploding([]), schema={})


# ---------------- per-provider OUTBOUND schema adaptation (no network) ------- #

def test_ollama_adapt_schema_passthrough(schema):
    p = OllamaProvider()
    out = p.adapt_schema(schema)
    assert out is schema  # Ollama accepts the full draft-2020-12 schema incl. $defs/$ref
    assert "$defs" in out
    assert p.num_predict >= 4096  # output-length floor


def test_groq_adapt_schema_wraps_response_format(schema):
    p = GroqProvider(api_key="dummy")
    out = p.adapt_schema(schema)
    assert out["type"] == "json_schema"
    js = out["json_schema"]
    assert js["name"] == "VisuLearnScript"
    assert js["strict"] is False  # strict:true would 400 (our schema has optional fields)
    assert js["schema"] is schema
    assert p.max_tokens >= 4096


def test_gemini_adapt_schema_inlines_and_strips(schema):
    p = GeminiProvider(api_key="dummy")
    out = p.adapt_schema(schema)
    # No $schema/$defs/$ref/default/minLength: keywords Gemini's responseSchema rejects.
    assert "$schema" not in out and "$defs" not in out
    # Enums and shared defs are inlined into the objects that referenced them.
    seggen = out["properties"]["segments"]["items"]
    assert seggen["properties"]["visual_cue"]["enum"]  # was a $ref -> VisualCue, now inlined
    meta = out["properties"]["meta"]
    assert meta["properties"]["category"]["enum"]      # was a $ref -> Category, now inlined
    assert "required" in meta and "topic" in meta["required"]
    assert p.max_output_tokens >= 4096
    # The canonical model must still validate the fixture (source of truth unchanged).
    Script.model_validate_json(VALID_SCRIPT)


def test_default_ollama_model_is_llama31():
    assert OllamaProvider().model == "llama3.1:8b"


def test_build_provider_defaults_to_llama31(monkeypatch):
    monkeypatch.setattr("script_engine._load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("VISULEARN_LLM_MODEL", raising=False)
    monkeypatch.delenv("VISULEARN_OLLAMA_MODEL", raising=False)
    assert build_provider().model == "llama3.1:8b"


def test_build_provider_reads_llm_model_env(monkeypatch):
    monkeypatch.delenv("VISULEARN_OLLAMA_MODEL", raising=False)
    monkeypatch.setenv("VISULEARN_LLM_MODEL", "some:model")
    assert build_provider().model == "some:model"


# ---------------- truncation is read, not guessed (AGENT.md 9.9) ------------- #

def test_normalize_finish_reason_unifies_the_three_provider_spellings():
    assert _normalize_finish_reason("length") == "length"        # Ollama / Groq
    assert _normalize_finish_reason("MAX_TOKENS") == "length"    # Gemini
    assert _normalize_finish_reason("STOP") == "stop"
    assert _normalize_finish_reason(None) == ""


def test_truncated_finish_reason_drives_the_hint_and_the_repair(schema):
    """A cut-off draft must say so in the repair prompt AND in the final error."""
    cut = ProviderResponse(text='{"meta": {"topic": "Red-Black', finish_reason="length")
    fake = FakeProvider([cut, cut])

    with pytest.raises(ScriptEngineError) as exc:
        generate_script("Red-Black Tree", {}, provider=fake, schema=schema)

    assert "TRUNCATED" in str(exc.value) and "finish_reason='length'" in str(exc.value)
    # the repair attempt was told why the draft was malformed, not just that it was
    assert "cut off by the output-token limit" in fake.seen_messages[1][-1]["content"]


def test_finish_reason_stop_suppresses_the_misleading_truncation_guess(schema):
    """Text the shape heuristic misreads as cut off, but the provider says finished.

    The old regex-only hint sent users chasing a token budget that was already fine.
    """
    prose = ProviderResponse(text='I think a stack is a "LIFO" structure', finish_reason="stop")
    fake = FakeProvider([prose, prose])

    with pytest.raises(ScriptEngineError) as exc:
        generate_script("Stack", {}, provider=fake, schema=schema)

    assert "truncat" not in str(exc.value).lower()


def test_shape_heuristic_still_fires_when_provider_reports_nothing():
    """Fallback path: no finish reason available, so the shape guess is all we have."""
    assert "may be truncated" in _truncation_hint('{"segments": [{"narration": "abc', "")
    assert _truncation_hint('{"ok": 1}', "") == ""


# ---------------- canonical schema is the source of truth (Golden Rule 2) ---- #

def test_committed_schema_matches_the_model(schema):
    """AGENT.md 7.1 as a unit test: the committed JSON must be what the model emits."""
    assert canonical_schema() == schema, (
        "config/schemas/script_schema.json is stale - run scripts/generate_schema.py"
    )


def test_generate_script_defaults_to_model_derived_schema():
    """With no schema passed, the engine derives it from Pydantic, never from disk."""
    fake = FakeProvider([VALID_SCRIPT])
    generate_script("Red-Black Tree", {}, provider=fake)
    assert fake.seen_schema[0] == canonical_schema()


def test_load_dotenv_sets_model(monkeypatch, tmp_path):
    monkeypatch.delenv("VISULEARN_LLM_MODEL", raising=False)
    monkeypatch.delenv("VISULEARN_OLLAMA_HOST", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(chr(10).join([
        "export VISULEARN_LLM_MODEL=foo:bar",
        "# a comment",
        "export VISULEARN_OLLAMA_HOST=http://1.2.3.4:11434",
        "",
    ]))
    _load_dotenv(dotenv)
    assert os.environ["VISULEARN_LLM_MODEL"] == "foo:bar"
    assert os.environ["VISULEARN_OLLAMA_HOST"] == "http://1.2.3.4:11434"


def test_build_provider_honours_temperature_zero(monkeypatch):
    """temperature=0 must reach the provider; the old `or` swallowed it as 'unset'."""
    monkeypatch.setattr("script_engine._load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("VISULEARN_TEMPERATURE", "0")
    assert build_provider().temperature == 0.0


def test_build_provider_reads_timeout_env(monkeypatch):
    monkeypatch.setattr("script_engine._load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("VISULEARN_TIMEOUT", "120")
    assert build_provider().timeout == 120.0


def test_clamp_output_tokens_logs_the_override(caplog):
    with caplog.at_level(logging.WARNING, logger="visulearn.script_engine"):
        p = OllamaProvider(num_predict=100)
    assert p.num_predict == 4096
    assert "clamping" in caplog.text

