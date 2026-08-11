"""Functional tests for the Ollama provider over real localhost HTTP (no external network).

Spins up an in-process ThreadingHTTPServer that mimics the Ollama /api/chat endpoint, so
OllamaProvider.complete() runs its real transport, payload serialization, and response
parsing end-to-end. These stay green offline (loopback only, no API key).
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import script_engine
from models import Script
from script_engine import OllamaProvider, ProviderError, ScriptEngineError, generate_script

FIXTURE = Path(__file__).parent / "fixtures" / "rbt_script.json"
VALID = json.loads(FIXTURE.read_text())  # 11 segments, opens title_card, closes, passes gate
SCHEMA = json.loads(
    (Path(__file__).parent.parent / "config" / "schemas" / "script_schema.json").read_text()
)


class _OllamaStub(BaseHTTPRequestHandler):
    """Returns queued (status, json_body) responses and records inbound requests."""

    server_version = ""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        self.server.requests.append(json.loads(raw))
        # queued entries are (status, payload) or (status, payload, extra_headers)
        status, payload, *rest = self.server.responses.pop(0)
        # bytes payload == send it verbatim, so a test can serve a non-JSON body.
        raw_body = isinstance(payload, bytes)
        body = payload if raw_body else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html" if raw_body else "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (rest[0] if rest else {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture

def stub_ollama():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _OllamaStub)
    srv.daemon_threads = True
    srv.requests = []
    srv.responses = []
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=2)


def _provider(stub_ollama, **kw) -> OllamaProvider:
    return OllamaProvider(
        host=f"http://127.0.0.1:{stub_ollama.server_port}",
        timeout=10,
        **kw,
    )


def test_ollama_functional_valid_first_try(stub_ollama):
    stub_ollama.responses.append((200, {"message": {"content": json.dumps(VALID)}}))

    script = generate_script(
        "Red-Black Tree", {"definition": "x"},
        provider=_provider(stub_ollama), schema=SCHEMA,
    )

    assert isinstance(script, Script)
    assert len(script.segments) == 11
    assert len(stub_ollama.requests) == 1  # success path: exactly one HTTP call
    req = stub_ollama.requests[0]
    assert req["model"] == "llama3.1:8b"
    assert req["format"] == SCHEMA          # structured-output schema is sent on the wire
    assert req["options"]["num_predict"] >= 4096


def test_ollama_functional_repair_round_trip(stub_ollama):
    bad = dict(VALID)
    bad["segments"] = bad["segments"][3:7]  # fails the 8-12 gate, no title/closing
    stub_ollama.responses.append((200, {"message": {"content": json.dumps(bad)}}))
    stub_ollama.responses.append((200, {"message": {"content": json.dumps(VALID)}}))

    script = generate_script(
        "Red-Black Tree", {"definition": "x"},
        provider=_provider(stub_ollama), schema=SCHEMA,
    )

    assert len(script.segments) == 11
    assert len(stub_ollama.requests) == 2   # exactly one repair -> second HTTP call
    last = stub_ollama.requests[1]
    roles = [m["role"] for m in last["messages"]]
    assert roles[-2:] == ["assistant", "user"]      # distinct repair context
    assert "rejected" in last["messages"][-1]["content"]


@pytest.fixture
def no_sleep(monkeypatch):
    """Run the backoff path at full speed and record what it would have waited."""
    waited = []
    monkeypatch.setattr(script_engine, "_sleep", waited.append)
    return waited


def test_transport_retries_429_then_succeeds(stub_ollama, no_sleep):
    """A throttle is a request that never happened - retrying it is not a repair."""
    stub_ollama.responses.append((429, {"error": "rate limited"}))
    stub_ollama.responses.append((200, {"message": {"content": json.dumps(VALID)}}))

    script = generate_script(
        "Red-Black Tree", {}, provider=_provider(stub_ollama), schema=SCHEMA,
    )

    assert len(script.segments) == 11
    assert len(stub_ollama.requests) == 2   # one throttled, one served
    assert no_sleep == [1.0]                # exponential backoff, first step


def test_transport_retry_honours_retry_after(stub_ollama, no_sleep):
    stub_ollama.responses.append((429, {"error": "slow down"}, {"Retry-After": "7"}))
    stub_ollama.responses.append((200, {"message": {"content": json.dumps(VALID)}}))

    generate_script("Red-Black Tree", {}, provider=_provider(stub_ollama), schema=SCHEMA)

    assert no_sleep == [7.0]  # the provider's number wins over our backoff


def test_transport_retry_gives_up_with_a_quota_message(stub_ollama, no_sleep):
    for _ in range(3):  # DEFAULT_TRANSPORT_ATTEMPTS
        stub_ollama.responses.append((429, {"error": "rate limited"}))

    with pytest.raises(ProviderError, match="rate-limited") as exc:
        generate_script("Red-Black Tree", {}, provider=_provider(stub_ollama), schema=SCHEMA)

    assert len(stub_ollama.requests) == 3       # bounded, not infinite
    assert no_sleep == [1.0, 2.0]               # exponential, no sleep after the last try
    assert "no quota" in str(exc.value)         # points at the local-Ollama escape hatch


def test_transport_does_not_retry_a_response_that_arrived(stub_ollama, no_sleep):
    """A 400 is an answer. Only the repair loop may re-ask, and only once."""
    stub_ollama.responses.append((400, {"error": "bad request"}))

    with pytest.raises(ProviderError, match="400"):
        generate_script("Red-Black Tree", {}, provider=_provider(stub_ollama), schema=SCHEMA)

    assert len(stub_ollama.requests) == 1
    assert no_sleep == []


def test_ollama_functional_reports_done_reason_length(stub_ollama):
    """Ollama's done_reason reaches the engine, so truncation is stated as fact."""
    cut = json.dumps(VALID)[:120]  # a valid JSON prefix: exactly the AGENT.md 9.9 gotcha
    for _ in range(2):
        stub_ollama.responses.append(
            (200, {"message": {"content": cut}, "done_reason": "length", "eval_count": 4096})
        )

    with pytest.raises(ScriptEngineError, match="TRUNCATED") as exc:
        generate_script(
            "Red-Black Tree", {}, provider=_provider(stub_ollama), schema=SCHEMA,
        )
    assert "VISULEARN_MAX_OUTPUT_TOKENS" in str(exc.value)


def test_ollama_functional_non_json_200_is_provider_error(stub_ollama):
    """A 200 carrying HTML (proxy interstitial, gateway page) is a ProviderError.

    Before the shared _post_json guard this escaped as a bare JSONDecodeError, which
    is the "no JSON object found" failure wearing a disguise.
    """
    stub_ollama.responses.append((200, b"<html><body>502 Bad Gateway</body></html>"))

    with pytest.raises(ProviderError, match="non-JSON body") as exc:
        generate_script("Stack", {}, provider=_provider(stub_ollama), schema={})
    assert "502 Bad Gateway" in str(exc.value)  # the actual body is quoted back


def test_ollama_functional_unexpected_shape_is_provider_error(stub_ollama):
    """A well-formed JSON 200 that is missing message.content must not KeyError."""
    stub_ollama.responses.append((200, {"unexpected": "shape"}))

    with pytest.raises(ProviderError, match="no usable text"):
        generate_script("Stack", {}, provider=_provider(stub_ollama), schema={})


def test_ollama_functional_model_not_found_readable(stub_ollama):
    stub_ollama.responses.append(
        (404, {"error": "model 'nope:0' not found, try pulling it first"})
    )

    with pytest.raises(ProviderError, match="not found") as exc:
        generate_script("Stack", {}, provider=_provider(stub_ollama, model="nope:0"), schema={})
    assert "ollama pull" in str(exc.value)
