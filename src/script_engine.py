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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
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

# Transport-level retry. NOT the repair loop — see the note on _post_json.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
DEFAULT_TRANSPORT_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 1.0

_sleep = time.sleep  # indirection so tests can run the backoff path instantly

# Presence sentinel: distinguishes "not configured" from a genuine falsy value like
# temperature=0. The old `cfg.get(name) or os.getenv(name)` could not represent an
# explicit 0, silently falling back to DEFAULT_TEMPERATURE (#20 / #22).
_UNSET = object()

OUTPUT_SCRIPTS_DIR = REPO_ROOT / "output" / "scripts" # raw drafts persisted here


def _safe_topic(topic: str) -> str:
    """Filesystem-safe slug for a topic (spaces/path separators -> underscores)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", topic.strip()).strip("_") or "topic"


def _write_raw(topic: str, raw: str, *, suffix: str) -> Path | None:
    """Best-effort persist of `raw` to output/scripts/<topic><suffix>.json.

    Returns the path, or None on any error so a write failure can never break
    generation (Golden Rule 4: fail-soft).
    """
    try:
        OUTPUT_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_SCRIPTS_DIR / f"{_safe_topic(topic)}{suffix}.json"
        path.write_text(raw)
        return path
    except OSError as exc:
        log.warning("could not persist raw script artifact for %r: %s", topic, exc)
        return None


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved provider settings; the single source of truth for `build_provider`.

    Populated in `build_provider` from its `config` dict (indexed by env-var name)
    or the equivalent `VISULEARN_*` environment variables. Documented keys
    (AGENT.md §6 / §7.6 — config, not hardcode):

      VISULEARN_LLM_PROVIDER     'ollama' | 'gemini' | 'groq'   (default 'ollama')
      VISULEARN_LLM_MODEL        shared model override for any provider
      VISULEARN_OLLAMA_MODEL     Ollama model       (default llama3.1:8b)
      VISULEARN_OLLAMA_HOST      Ollama endpoint    (default http://127.0.0.1:11434)
      VISULEARN_GEMINI_MODEL     Gemini model       (default gemini-2.5-flash)
      VISULEARN_GROQ_MODEL       Groq model         (default llama-3.3-70b-versatile)
      GEMINI_API_KEY / GOOGLE_API_KEY   Gemini secrets
      GROQ_API_KEY               Groq secret
      VISULEARN_TEMPERATURE      temperature (default 0.4); 0 is a real choice
      VISULEARN_MAX_OUTPUT_TOKENS   output budget, floored at 4096
      VISULEARN_TIMEOUT          HTTP timeout seconds (default 300.0)
    """

    provider: str = "ollama"
    model: str | None = None
    host: str | None = None
    api_key: str | None = None
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    timeout: float = 300.0

def _clamp_output_tokens(value, *, name: str = "VISULEARN_MAX_OUTPUT_TOKENS") -> int:
    """Return `value` raised to the output-token floor, logging the override.

    A budget below 4096 truncates a full script mid-JSON and mimics a model bug
    (AGENT.md §9.9), so the floor is non-negotiable. When we overrule the operator
    we say so rather than silently changing their intent (#22).
    """
    value = int(value)
    if value < MIN_MAX_OUTPUT_TOKENS:
        log.warning(
            "%s=%d is below the %d output-token floor - clamping up to prevent "
            "mid-JSON truncation (AGENT.md §9.9)",
            name, value, MIN_MAX_OUTPUT_TOKENS,
        )
        return MIN_MAX_OUTPUT_TOKENS
    return value


class ProviderError(RuntimeError):
    """A provider call failed in a way the caller should see and act on.

    Kept distinct from RuntimeError so callers/tests can catch a backend failure
    (model not found, connection refused, auth, truncation) vs. a validation failure.
    """


@dataclass(frozen=True)
class ProviderResponse:
    """A provider's raw text plus the metadata the engine needs to interpret it.

    `finish_reason` is normalized to "length" | "stop" | "" (unknown). "length" is
    authoritative truncation — the output-token budget cut the model off — which
    AGENT.md §9.9 calls out as the gotcha that mimics a model bug. Reading it beats
    inferring it: JSON truncated mid-string ends in a quote and looks complete.

    `usage` is whatever token accounting the provider returned, kept for the §6
    per-stage logging requirement.
    """

    text: str
    finish_reason: str = ""
    usage: dict | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _normalize_finish_reason(raw: str | None) -> str:
    """Map a provider's finish reason onto {"length", "stop", ""}.

    The same event has three names on the wire: Ollama `done_reason="length"`,
    Gemini `finishReason="MAX_TOKENS"`, Groq `finish_reason="length"`.
    """
    if not raw:
        return ""
    key = str(raw).strip().lower()
    if key in ("length", "max_tokens", "maxtokens"):
        return "length"
    if key in ("stop", "end_turn", "eos"):
        return "stop"
    return key


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
    def complete(self, messages: list[dict], schema) -> ProviderResponse | str:
        """Send `messages` in structured-output mode and return the assistant's output.

        `schema` is the value produced by `adapt_schema` and is passed as the provider's
        structured-output parameter (Ollama `format`, Gemini `responseSchema`, Groq
        `response_format`). Must raise `ProviderError` on transport/HTTP/parse failure.

        Return a `ProviderResponse` so the engine can see the finish reason and token
        usage. A bare `str` is still accepted — the interface AGENT.md §1 documents —
        and is read as text with an unknown finish reason.
        """


def _retry_delay(resp: requests.Response, *, backoff: float, attempt: int) -> float:
    """Seconds to wait before a retry: the provider's Retry-After, else exponential.

    Retry-After may also be an HTTP-date; that form is not parsed and falls through
    to the backoff, which is the safe direction to be wrong in.
    """
    try:
        return max(0.0, float(resp.headers.get("Retry-After", "")))
    except (TypeError, ValueError):
        return backoff * (2 ** (attempt - 1))


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
    attempts: int = DEFAULT_TRANSPORT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF_SECONDS,
) -> dict:
    """POST JSON and return the decoded body, or raise `ProviderError`.

    Every wire-level failure funnels through here so the `Provider.complete()`
    contract — ProviderError on transport/HTTP/parse failure — holds identically for
    all three providers instead of being re-derived (and half-forgotten) in each.

    `explain_status` lets a provider turn a status code into its own actionable
    message (Ollama's model-not-found, Groq's auth) while the plumbing stays shared;
    returning None falls back to the generic form.

    RETRY, AND WHY IT IS NOT THE THING GOLDEN RULE 1 FORBIDS
    -------------------------------------------------------
    Rule 1 bans re-running an identical prompt into an identical context *after a
    validation failure* — the model already answered, and asking again unchanged is
    superstition. A 429 or a 502 is the opposite situation: the model never answered
    at all, nothing was validated, and the request is not yet spent. Retrying it is
    the only way to survive Groq's 30 RPM / TPM ceiling (AGENT.md §6, §9.7) without a
    single throttle killing a whole batch.

    So: retry the statuses in RETRY_STATUSES with exponential backoff, honouring
    Retry-After when the provider sends it. Never retry a response that arrived —
    that is the repair loop's job, exactly once, in generate_script().

    Transport exceptions (connection refused, DNS, timeout) are NOT retried: a dead
    Ollama daemon stays dead, and three silent 300-second timeouts is a worse failure
    than one legible one (§7.7).
    """
    resp = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            resp = session.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            hint = f" ({unreachable_hint})" if unreachable_hint else ""
            raise ProviderError(f"{label} unreachable at {url}{hint}: {exc}") from exc

        if resp.status_code not in RETRY_STATUSES or attempt == max(1, attempts):
            break

        delay = _retry_delay(resp, backoff=backoff, attempt=attempt)
        log.warning(
            "%s returned %d (attempt %d/%d) — backing off %.1fs before retrying",
            label, resp.status_code, attempt, attempts, delay,
        )
        _sleep(delay)

    if resp.status_code != 200:
        detail = (resp.text or "").strip()
        special = explain_status(resp.status_code, detail) if explain_status else None
        if special is None and resp.status_code == 429:
            special = (
                f"{label} rate-limited (429), still throttled after {attempts} attempt(s) — "
                "this is the free-tier RPM/TPM/RPD ceiling. Back off and rerun, or switch to "
                f"local Ollama, which has no quota (AGENT.md §1.1): {detail}"
            )
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
        self.num_predict = _clamp_output_tokens(num_predict)
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

    def complete(self, messages: list[dict], schema) -> ProviderResponse:
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
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"Ollama returned no usable text: {data}") from exc
        return ProviderResponse(
            text=text,
            finish_reason=_normalize_finish_reason(data.get("done_reason")),
            usage={
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
            },
        )


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
        self.max_output_tokens = _clamp_output_tokens(max_output_tokens)
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

    def complete(self, messages: list[dict], schema) -> ProviderResponse:
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
            candidate = data["candidates"][0]
            text = candidate["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            # A MAX_TOKENS cutoff can land here with no parts at all, so say so rather
            # than reporting the generic shape error.
            reason = _normalize_finish_reason(
                (data.get("candidates") or [{}])[0].get("finishReason")
                if isinstance(data.get("candidates"), list) else None
            )
            if reason == "length":
                raise ProviderError(
                    "Gemini hit the output-token limit before emitting any text — raise "
                    f"VISULEARN_MAX_OUTPUT_TOKENS (floor {MIN_MAX_OUTPUT_TOKENS})."
                ) from exc
            raise ProviderError(f"Gemini returned no usable text: {data}") from exc
        return ProviderResponse(
            text=text,
            finish_reason=_normalize_finish_reason(candidate.get("finishReason")),
            usage=data.get("usageMetadata"),
        )


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
        self.max_tokens = _clamp_output_tokens(max_tokens)
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

    def complete(self, messages: list[dict], schema) -> ProviderResponse:
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
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Groq returned no usable text: {data}") from exc
        return ProviderResponse(
            text=text,
            finish_reason=_normalize_finish_reason(choice.get("finish_reason")),
            usage=data.get("usage"),
        )


def _build_ollama(cfg: ProviderConfig) -> Provider:
    return OllamaProvider(
        model=cfg.model or os.getenv("VISULEARN_OLLAMA_MODEL") or "llama3.1:8b",
        host=cfg.host or os.getenv("VISULEARN_OLLAMA_HOST", "http://127.0.0.1:11434"),
        temperature=cfg.temperature,
        num_predict=cfg.max_output_tokens,
        timeout=cfg.timeout,
    )


def _build_gemini(cfg: ProviderConfig) -> Provider:
    key = cfg.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ProviderError("Gemini selected but no key - set GEMINI_API_KEY or GOOGLE_API_KEY.")
    return GeminiProvider(
        api_key=key,
        model=cfg.model or os.getenv("VISULEARN_GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
        timeout=cfg.timeout,
    )


def _build_groq(cfg: ProviderConfig) -> Provider:
    key = cfg.api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ProviderError("Groq selected but no key - set GROQ_API_KEY.")
    return GroqProvider(
        api_key=key,
        model=cfg.model or os.getenv("VISULEARN_GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
        timeout=cfg.timeout,
    )


_PROVIDER_BUILDERS: dict[str, Callable[[ProviderConfig], Provider]] = {
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

    Reads a `config` dict (indexed by `VISULEARN_*` env-var name) then the environment,
    resolving into a `ProviderConfig` (its docstring documents every key). The `_UNSET`
    sentinel distinguishes "not configured" from a genuine falsy value, so an explicit
    `VISULEARN_TEMPERATURE=0` is honoured instead of falling back. Timeout comes from
    `VISULEARN_TIMEOUT` (default 300s). Runs the dependency-free .env loader first so
    local config under `<REPO_ROOT>/.env` takes effect; real env vars win.
    """
    _load_dotenv()
    cfg = config or {}

    def _raw(name: str):
        return cfg[name] if name in cfg else os.getenv(name, _UNSET)

    def _str(name: str, default: str) -> str:
        raw = _raw(name)
        return str(raw) if raw is not _UNSET else default

    def _float(name: str, default: float) -> float:
        raw = _raw(name)
        return float(raw) if raw is not _UNSET else default

    def _int(name: str, default: int) -> int:
        raw = _raw(name)
        return int(raw) if raw is not _UNSET else default

    model = cfg["model"] if "model" in cfg else _raw("VISULEARN_LLM_MODEL")
    host = cfg["host"] if "host" in cfg else _raw("VISULEARN_OLLAMA_HOST")
    api_key = _raw("GEMINI_API_KEY")

    pcfg = ProviderConfig(
        provider=_str("VISULEARN_LLM_PROVIDER", "ollama").lower(),
        model=str(model) if model is not _UNSET else None,
        host=str(host) if host is not _UNSET else None,
        api_key=str(api_key) if api_key is not _UNSET else None,
        temperature=_float("VISULEARN_TEMPERATURE", DEFAULT_TEMPERATURE),
        max_output_tokens=_int("VISULEARN_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS),
        timeout=_float("VISULEARN_TIMEOUT", 300.0),
    )

    builder = _PROVIDER_BUILDERS.get(pcfg.provider)
    if builder is None:
        raise ProviderError(
            f"Unknown VISULEARN_LLM_PROVIDER={pcfg.provider!r} (use ollama, gemini, or groq)."
        )
    return builder(pcfg)


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
    response = _as_response(prov.complete(messages, adapted))
    attempts += 1
    script, problems = _validate(
            response.text, expected_topic=topic, expected_category=category
        )
    problems = _with_truncation_problem(problems, response)

    # Real repair loop, bounded by MAX_REPAIR_ATTEMPTS (Golden Rule 1 = exactly one
    # repair). A `while` (not `if`) keeps the bound in one obvious place; the FakeProvider
    # test enforces we never exceed it no matter how many outputs are available.
    while problems and attempts <= MAX_REPAIR_ATTEMPTS:
        # Persist + log the rejected draft BEFORE re-asking, so a crash in the repair
        # path can't lose the only evidence of what the model produced (#6).
        log.warning("script invalid on attempt %d/%d; rejected draft:\n%s",
                    attempts, MAX_REPAIR_ATTEMPTS, response.text)
        draft_path = _write_raw(topic, response.text, suffix=f".repair{attempts}.raw")
        if draft_path:
            log.warning("persisted rejected draft to %s", draft_path)
        repair = _repair_messages(messages, response.text, problems)
        response = _as_response(prov.complete(repair, adapted))
        attempts += 1
        script, problems = _validate(response.text)
        problems = _with_truncation_problem(problems, response)

    if problems or script is None:
        raw_path = _write_raw(topic, response.text, suffix=".raw")
        log.error(
            "script engine hard-fail after %d attempt(s). raw output follows:\n%s",
            attempts, response.text,
        )
        if raw_path:
            log.error("raw output also written to %s", raw_path)
        raise ScriptEngineError(
            "script engine could not produce valid output after "
            f"{attempts} attempt(s): {problems}"
            + _truncation_hint(response.text, response.finish_reason)
        )
    return script


def _as_response(result: ProviderResponse | str) -> ProviderResponse:
    """Accept the documented `-> str` provider contract as well as ProviderResponse."""
    return result if isinstance(result, ProviderResponse) else ProviderResponse(text=result)


def _with_truncation_problem(problems: list[str], response: ProviderResponse) -> list[str]:
    """Tell the repair attempt *why* the draft was malformed when it was cut off.

    Only added alongside real problems: a truncated response that somehow still
    validates is a valid script, and inventing a failure for it would be wrong.
    """
    if problems and response.truncated:
        return problems + [
            "the response was cut off by the output-token limit before it finished — "
            "return a shorter script that fits"
        ]
    return problems


def _validate(
    raw: str,
    *,
    expected_topic: str | None = None,
    expected_category: str | None = None,
) -> tuple[Script | None, list[str]]:
    try:
        script = Script.model_validate_json(raw)
    except Exception as e:  # noqa: BLE001 - surface any parse/validation issue to the repair loop
        return None, [f"structural: {e}"]
    return script, engine_gate_errors(
        script,
        expected_topic=expected_topic,
        expected_category=expected_category,
    )


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


def _truncation_hint(raw: str, finish_reason: str = "") -> str:
    """Guide the user to raise the token budget when the output was cut off.

    The provider's finish reason is authoritative and decides first — every backend
    reports it (Ollama `done_reason`, Gemini `finishReason`, Groq `finish_reason`),
    and AGENT.md §9.9 makes ruling truncation out explicitly a requirement.

    The shape heuristic is only a fallback for providers that report nothing, and it
    is unreliable in the most common case: JSON cut mid-string ends in `"`, which
    reads as complete. When the provider affirmatively says it finished, we suppress
    the guess rather than send the user chasing a token budget that is already fine.
    """
    if finish_reason == "length":
        return (". The provider reported finish_reason='length' — the output was TRUNCATED. "
                f"Raise VISULEARN_MAX_OUTPUT_TOKENS (floor {MIN_MAX_OUTPUT_TOKENS}) and retry.")
    if finish_reason:
        return ""  # provider says it finished on its own; don't guess otherwise
    if not raw or (re.search(r"[\"}\]]\s*$", raw.strip()) is None and '"' in raw):
        return (". Output may be truncated — raise VISULEARN_MAX_OUTPUT_TOKENS "
                f"(floor {MIN_MAX_OUTPUT_TOKENS}) and retry.")
    return ""


def _inline_schema(node, defs: dict, *, supported: set[str],
                   _seen: frozenset[str] = frozenset()) -> dict:
    """Recursively inline $ref -> $defs and drop keywords not in `supported` (Gemini).

    Recurse through every schema-shaped keyword — `properties`, `items`,
    `additionalProperties`, and the `anyOf`/`oneOf`/`allOf`/`prefixItems` unions — so a
    $ref buried anywhere is flattened, not just at the top level (#11/#12). Recursion is
    orthogonal to the `supported` whitelist: a union keyword still requires both the
    recursion here AND membership in `_SUPPORTED_KEYS` to survive for Gemini.

    A `$ref` whose target is missing from `$defs` raises `ProviderError` (a stale schema,
    never a silent drop) and a `$ref` cycle (A -> B -> A) is cut with a shallow
    `{"$ref": ...}` marker instead of recursing forever (#14). The immutable `_seen` set
    tracks only the current expansion chain, so siblings may re-inline the same
    definition without a false cycle.
    """
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if ref and ref.startswith("#/$defs/"):
        name = ref[len("#/$defs/"):]
        if name not in defs:
            raise ProviderError(
                f"Gemini schema references undefined $defs[{name!r}] "
                f"(available: {sorted(defs)}) — schema is stale; rerun scripts/generate_schema.py"
            )
        if name in _seen:
            # Cycle reached: stop expanding. A shallow $ref keeps the structure finite
            # instead of RecursionError.
            return {"$ref": ref}
        return _inline_schema(
            defs[name], defs, supported=supported, _seen=_seen | {name}
        )
    out: dict = {}
    for key, value in node.items():
        if key == "$defs":
            continue
        if key not in supported:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {
                k: _inline_schema(v, defs, supported=supported, _seen=_seen)
                for k, v in value.items()
            }
        elif isinstance(value, list) and key in ("prefixItems", "anyOf", "oneOf", "allOf"):
            out[key] = [
                _inline_schema(v, defs, supported=supported, _seen=_seen) for v in value
            ]
        elif isinstance(value, dict) and key in ("items", "additionalProperties"):
            out[key] = _inline_schema(value, defs, supported=supported, _seen=_seen)
        else:
            out[key] = value
    return out

