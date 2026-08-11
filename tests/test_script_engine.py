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
from pathlib import Path

import pytest

from models import Script
from script_engine import (
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    Provider,
    ProviderError,
    ScriptEngineError,
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

    def complete(self, messages, schema):
        self.call_count += 1
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


def test_default_ollama_model_is_qwen3():
    assert OllamaProvider().model == "qwen3:14b"
