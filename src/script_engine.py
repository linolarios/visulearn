"""Script Engine — LLM -> validated Script. The stage that must not emit unvalidated JSON.

Golden Rule 1 (AGENT.md): JSON comes from the provider's structured-output mode, then
Pydantic validates it, then engine_gate_errors() checks production rules. On failure,
feed the errors back for exactly ONE repair attempt, then hard-fail and log raw output.
Never re-run the identical prompt into the identical context.

Providers sit behind a SMALL interface — `Provider.complete(messages, schema) -> str` —
so the same loop drives Ollama `format=<schema>`, Gemini `responseSchema`, and Groq
`response_format` (AGENT.md §1/§3). The Pydantic model (src/models.py) is the single
source of truth; only the OUTBOUND JSON Schema is adapted per provider, never the model.
"""
from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import requests

from models import Script, canonical_schema, engine_gate_errors

log = logging.getLogger("visulearn.script_engine")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA_PATH = REPO_ROOT / "config" / "schemas" / "script_schema.json"
DEFAULT_SYSTEM_PROMPTS = {
    "dsa": REPO_ROOT / "config" / "prompts" / "dsa_system_prompt.txt",
    "design_pattern": REPO_ROOT / "config" / "prompts" / "design_pattern_system_prompt.txt",
}

# A compliant 8-12 segment script needs a generous output budget. Never go below 4096:
# truncation is exactly the failure mode that produced "no JSON object found" upstream.
MIN_MAX_OUTPUT_TOKENS = 4096
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.4
MAX_REPAIR_ATTEMPTS = 1  # Golden Rule 1: exactly one repair, then hard-fail.


class ProviderError(RuntimeError):
    """A provider call failed in a way the caller should see and act on.

    Kept distinct from RuntimeError so callers/tests can catch a backend failure
    (model not found, connection refused, auth, truncation) vs. a validation failure.
    """


class Provider(ABC):
    """Minimal structured-output provider interface."""

    #: Human-readable provider id ("ollama", "gemini", "groq").
    name: str

    def adapt_schema(self, schema: dict):
        """Transform the canonical Pydantic schema into this provider's OUTBOUND form.

        The canonical schema (src/models.py) is NEVER weakened; only reshaped so the
        provider's structured-output mode will accept it (AGENT.md §3).
        """
        return schema

    @abstractmethod
    def complete(self, messages: list[dict], schema) -> str:
        """Send `messages` in structured-output mode and return the assistant's raw text.

        `schema` is the value produced by `adapt_schema` and is passed as the provider's
        structured-output parameter (Ollama `format`, Gemini `responseSchema`, Groq
        `response_format`). Must raise `ProviderError` on transport/HTTP/parse failure.
        """
        raise NotImplementedError


def _post_json(
    session: requests.Session,
    url: str,
    *,
    payload: dict,
    timeout: float,
    label: str,
    headers: dict | None = None,
    unreachable_hint: str = "",
    explain_status: Callable[[int, str], str | None] | None = None,
) -> dict:
    """POST JSON and return the decoded body, or raise `ProviderError`.

    Every wire-level failure funnels through here so the `Provider.complete()`
    contract — ProviderError on transport/HTTP/parse failure — holds identically for
    all three providers instead of being re-derived (and half-forgotten) in each.

    `explain_status` lets a provider turn a status code into its own actionable
    message (Ollama's model-not-found, Groq's auth) while the plumbing stays shared;
    returning None falls back to the generic form.
    """
    try:
        resp = session.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        hint = f" ({unreachable_hint})" if unreachable_hint else ""
        raise ProviderError(f"{label} unreachable at {url}{hint}: {exc}") from exc

    if resp.status_code != 200:
        detail = (resp.text or "").strip()
        special = explain_status(resp.status_code, detail) if explain_status else None
        raise ProviderError(special or f"{label} API error {resp.status_code}: {detail}")

    try:
        return resp.json()
    except ValueError as exc:
        # A 200 carrying something other than JSON — proxy interstitial, captive
        # portal, gateway HTML — is a parse failure, not model output. Catching it
        # here is what stops it reaching the engine disguised as an unparseable
        # response and producing the "no JSON object found" class of bug report.
        content_type = resp.headers.get("Content-Type", "no content-type")
        raise ProviderError(
            f"{label} returned a non-JSON body (HTTP 200, {content_type}): "
            f"{(resp.text or '')[:200]!r}"
        ) from exc


# --------------------------------------------------------------------------- #
# Ollama — local, the batch default (AGENT.md §1.1 / task spec)                #
# --------------------------------------------------------------------------- #

class OllamaProvider(Provider):
    name = "ollama"

    def __init__(
        self,
        *,
        model: str = "llama3.1:8b",
        host: str = "http://127.0.0.1:11434",
        temperature: float = DEFAULT_TEMPERATURE,
        num_predict: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout: float = 300.0,
        session: requests.Session | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_predict = max(MIN_MAX_OUTPUT_TOKENS, int(num_predict))
        self.timeout = timeout
        self.session = session or requests.Session()

    def adapt_schema(self, schema: dict) -> dict:
        # Ollama's `format` accepts a full draft-2020-12 schema including $defs/$ref.
        return schema

    def _explain_status(self, status: int, detail: str) -> str | None:
        if status == 404 and "model" in detail.lower():
            return (
                f"Ollama model not found: {self.model!r} — pull it with "
                f"`ollama pull {self.model}` ({detail})"
            )
        return None

    def complete(self, messages: list[dict], schema) -> str:
        data = _post_json(
            self.session,
            f"{self.host}/api/chat",
            payload={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": schema,
                "options": {"temperature": self.temperature, "num_predict": self.num_predict},
            },
            timeout=self.timeout,
            label="Ollama",
            unreachable_hint="is the daemon running? `ollama serve`",
            explain_status=self._explain_status,
        )
        if data.get("error"):
            raise ProviderError(f"Ollama error: {data['error']}")
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"Ollama returned no usable text: {data}") from exc


# --------------------------------------------------------------------------- #
# Gemini — cloud fallback, native responseSchema (AGENT.md §1)                 #
# --------------------------------------------------------------------------- #

class GeminiProvider(Provider):
    name = "gemini"

    _SUPPORTED_KEYS = {
        "type", "title", "description", "properties", "required", "items",
        "enum", "format", "minimum", "maximum", "minItems", "maxItems",
        "additionalProperties", "prefixItems",
    }

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout: float = 300.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max(MIN_MAX_OUTPUT_TOKENS, int(max_output_tokens))
        self.timeout = timeout
        self.session = session or requests.Session()

    def adapt_schema(self, schema: dict) -> dict:
        """Inline $defs/$ref and drop keywords Gemini's responseSchema does not support."""
        defs = schema.get("$defs", {})
        return _inline_schema(schema, defs, supported=self._SUPPORTED_KEYS)

    @staticmethod
    def _explain_status(status: int, detail: str) -> str | None:
        if status in (400, 401, 403):
            return f"Gemini API error {status} (bad/invalid API key or schema): {detail}"
        return None

    def complete(self, messages: list[dict], schema) -> str:
        body: dict = {"contents": [], "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": self.temperature,
            "maxOutputTokens": self.max_output_tokens,
        }}
        for msg in messages:
            role = msg["role"]
            if role == "system":
                # Gemini takes the system instruction separately, not as a content role.
                body.setdefault("systemInstruction", {"parts": []})["parts"].append({"text": msg["content"]})
            else:
                body["contents"].append({
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": msg["content"]}],
                })

        data = _post_json(
            self.session,
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            payload=body,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            timeout=self.timeout,
            label="Gemini",
            explain_status=self._explain_status,
        )
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Gemini returned no usable text: {data}") from exc


# --------------------------------------------------------------------------- #
# Groq — cloud fallback, response_format (AGENT.md §1)                         #
# --------------------------------------------------------------------------- #

class GroqProvider(Provider):
    name = "groq"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout: float = 300.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max(MIN_MAX_OUTPUT_TOKENS, int(max_tokens))
        self.timeout = timeout
        self.session = session or requests.Session()

    def adapt_schema(self, schema: dict) -> dict:
        # Groq json_schema mode. strict:false because strict mode requires EVERY field to be
        # required (our schema has optional/defaulted fields) — strict:true would 400.
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.get("title", "VisuLearnScript"),
                "strict": False,
                "schema": schema,
            },
        }

    @staticmethod
    def _explain_status(status: int, detail: str) -> str | None:
        if status in (401, 403):
            return f"Groq auth failed ({status}) — check GROQ_API_KEY: {detail}"
        return None

    def complete(self, messages: list[dict], schema) -> str:
        data = _post_json(
            self.session,
            "https://api.groq.com/openai/v1/chat/completions",
            payload={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": schema,
            },
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=self.timeout,
            label="Groq",
            explain_status=self._explain_status,
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Groq returned no usable text: {data}") from exc


def _build_ollama(cfg: dict, _float, _int) -> Provider:
    return OllamaProvider(
        model=(cfg.get("model")
               or os.getenv("VISULEARN_LLM_MODEL")
               or os.getenv("VISULEARN_OLLAMA_MODEL")
               or "llama3.1:8b"),
        host=cfg.get("host") or os.getenv("VISULEARN_OLLAMA_HOST", "http://127.0.0.1:11434"),
        temperature=_float("VISULEARN_TEMPERATURE", DEFAULT_TEMPERATURE),
        num_predict=_int("VISULEARN_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS),
    )


def _build_gemini(cfg: dict, _float, _int) -> Provider:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ProviderError("Gemini selected but no key - set GEMINI_API_KEY or GOOGLE_API_KEY.")
    return GeminiProvider(
        api_key=key,
        model=cfg.get("model") or os.getenv("VISULEARN_GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=_float("VISULEARN_TEMPERATURE", DEFAULT_TEMPERATURE),
        max_output_tokens=_int("VISULEARN_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS),
    )


def _build_groq(cfg: dict, _float, _int) -> Provider:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ProviderError("Groq selected but no key - set GROQ_API_KEY.")
    return GroqProvider(
        api_key=key,
        model=cfg.get("model") or os.getenv("VISULEARN_GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=_float("VISULEARN_TEMPERATURE", DEFAULT_TEMPERATURE),
        max_tokens=_int("VISULEARN_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS),
    )


_PROVIDER_BUILDERS: dict[str, Callable[..., Provider]] = {
    "ollama": _build_ollama,
    "gemini": _build_gemini,
    "groq": _build_groq,
}


def _load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from a .env file into the environment (no dependency).

    Real environment variables always win: keys already present in os.environ are
    left untouched. Handles an `export` prefix, surrounding quotes, and # comments.
    """
    path = path or (REPO_ROOT / ".env")
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip(chr(39) + chr(34)).strip()
        if key and key not in os.environ:
            os.environ[key] = val


def build_provider(config: dict | None = None) -> Provider:
    """Factory selecting a provider from env/config. Ollama is the batch default.

    Uses a provider-name -> builder registry (dict dispatch) instead of an if/else
    ladder. Runs the dependency-free .env loader first so local configuration under
    `<REPO_ROOT>/.env` (e.g. VISULEARN_LLM_MODEL) takes effect. Real env vars win.
    """
    _load_dotenv()
    cfg = config or {}
    provider = (cfg.get("provider") or os.getenv("VISULEARN_LLM_PROVIDER", "ollama")).lower()

    def _float(name: str, default: float) -> float:
        raw = cfg.get(name) or os.getenv(name)
        return float(raw) if raw is not None else default

    def _int(name: str, default: int) -> int:
        raw = cfg.get(name) or os.getenv(name)
        return int(raw) if raw is not None else default

    builder = _PROVIDER_BUILDERS.get(provider)
    if builder is None:
        raise ProviderError(
            f"Unknown VISULEARN_LLM_PROVIDER={provider!r} (use ollama, gemini, or groq)."
        )
    return builder(cfg, _float, _int)


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #

class ScriptEngineError(RuntimeError):
    """The engine exhausted its one repair and could not produce a valid Script."""




def generate_script(
    topic: str,
    fact_sheet: dict,
    *,
    provider: Provider | None = None,
    category: str = "dsa",
    schema: dict | None = None,
    config: dict | None = None,
) -> Script:
    """Build the prompt + adapted schema, drive structured output, validate, one repair.

    Returns a validated `Script` or raises `ScriptEngineError` after exactly one repair
    attempt (raw output is logged on hard-fail). See AGENT.md Golden Rule 1.
    """
    prov = provider or build_provider(config=config)
    if schema is None:
        schema = canonical_schema()
    adapted = prov.adapt_schema(schema)
    messages = _build_messages(topic, fact_sheet, category, schema=adapted)

    attempts = 0
    raw = prov.complete(messages, adapted)
    attempts += 1
    script, problems = _validate(raw)

    if problems and attempts <= MAX_REPAIR_ATTEMPTS:
        log.warning("script invalid, attempting one repair (%d): %s", attempts, problems)
        repair = _repair_messages(messages, raw, problems)
        raw = prov.complete(repair, adapted)  # distinct context -> not a blind identical retry
        attempts += 1
        script, problems = _validate(raw)

    if problems or script is None:
        log.error("script engine hard-fail after %d attempt(s). raw output follows:\n%s", attempts, raw)
        raise ScriptEngineError(
            "script engine could not produce valid output after "
            f"{attempts} attempt(s): {problems}" + _truncation_hint(raw)
        )
    return script


def _validate(raw: str) -> tuple[Script | None, list[str]]:
    try:
        script = Script.model_validate_json(raw)
    except Exception as e:  # noqa: BLE001 - surface any parse/validation issue to the repair loop
        return None, [f"structural: {e}"]
    return script, engine_gate_errors(script)


def _repair_messages(messages: list[dict], raw: str, problems: list[str]) -> list[dict]:
    return messages + [
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": "Your output was rejected:\n- " + "\n- ".join(problems)
                       + "\nReturn ONLY corrected JSON matching the provided schema.",
        },
    ]


def _build_messages(
    topic: str, fact_sheet: dict, category: str, *, schema
) -> list[dict]:
    prompt_path = DEFAULT_SYSTEM_PROMPTS.get(category)
    if prompt_path is None:
        raise ProviderError(f"unknown category {category!r}; use 'dsa' or 'design_pattern'")
    system = prompt_path.read_text()
    user = json.dumps(
        {"topic": topic, "facts": fact_sheet, "schema": schema}, ensure_ascii=False, indent=2
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _truncation_hint(raw: str) -> str:
    """If output looks cut off, guide the user to raise the max-output-token budget."""
    if not raw or (re.search(r"[\"}\]]\s*$", raw.strip()) is None and '"' in raw):
        return (". Output may be truncated — raise VISULEARN_MAX_OUTPUT_TOKENS "
                f"(floor {MIN_MAX_OUTPUT_TOKENS}) and retry.")
    return ""


def _inline_schema(node, defs: dict, *, supported: set[str]) -> dict:
    """Recursively inline $ref -> $defs and drop keywords not in `supported` (Gemini)."""
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if ref and ref.startswith("#/$defs/"):
        name = ref[len("#/$defs/"):]
        frag = defs[name]
        return _inline_schema(frag, defs, supported=supported)
    out: dict = {}
    for key, value in node.items():
        if key == "$defs":
            continue
        if key not in supported:
            continue
        if key == "properties":
            out[key] = {k: _inline_schema(v, defs, supported=supported) for k, v in value.items()}
        elif key == "items":
            out[key] = _inline_schema(value, defs, supported=supported)
        elif key == "prefixItems" and isinstance(value, list):
            out[key] = [_inline_schema(v, defs, supported=supported) for v in value]
        else:
            out[key] = value
    return out
