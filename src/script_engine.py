"""Script Engine — LLM -> validated Script. The stage that must not emit unvalidated JSON.

Golden Rule 1 (AGENT.md): JSON comes from the provider's structured-output mode, then
Pydantic validates it, then engine_gate_errors() checks production rules. On failure,
feed the errors back for exactly ONE repair attempt, then hard-fail and log raw output.
Never re-run the identical prompt into the identical context.
"""
from __future__ import annotations

import json
import logging

from models import Script, engine_gate_errors

log = logging.getLogger("visulearn.script_engine")


def generate_script(topic: str, fact_sheet: dict, *, call_llm, schema: dict) -> Script:
    """`call_llm(messages, schema) -> str` must use the provider's JSON/schema mode.

    Returns a validated Script or raises RuntimeError after one failed repair.
    """
    messages = _build_messages(topic, fact_sheet)
    raw = call_llm(messages, schema)

    script, problems = _validate(raw)
    if problems:
        log.warning("script invalid, attempting one repair: %s", problems)
        repair = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Your output was rejected:\n- " + "\n- ".join(problems)
                                        + "\nReturn ONLY corrected JSON matching the schema."},
        ]
        raw = call_llm(repair, schema)
        script, problems = _validate(raw)

    if problems or script is None:
        log.error("script engine hard-fail. raw output follows:\n%s", raw)
        raise RuntimeError(f"script engine could not produce valid output: {problems}")
    return script


def _validate(raw: str) -> tuple[Script | None, list[str]]:
    try:
        script = Script.model_validate_json(raw)
    except Exception as e:  # noqa: BLE001 - surface any parse/validation issue to the repair loop
        return None, [f"structural: {e}"]
    return script, engine_gate_errors(script)


def _build_messages(topic: str, fact_sheet: dict) -> list[dict]:
    system = "You are ScriptSmith. Return ONLY JSON matching the provided schema."
    user = json.dumps({"topic": topic, "facts": fact_sheet}, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
