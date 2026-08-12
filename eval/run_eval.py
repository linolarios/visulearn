"""Score the Script Engine across models on a fixed topic set. See AGENT.md §10.

This is NOT part of pytest -- it's slow, model-dependent, and consumes local compute
(or a cloud quota). Run it when choosing a model or before a release:

    python eval/run_eval.py                       # default model, from OLLAMA_HOST
    python eval/run_eval.py --models qwen3:8b qwen3:14b gemma3:4b

Design choice: each (model, topic) gets ONE call with NO repair. That gives the *unmasked*
first-try quality -- what you want for model SELECTION, because the repair loop would
otherwise paper over a weak model. `gate%` (first-try) is the inverse of "how often the
production repair loop must fire," so a low number means the model leans on repair.

Two SCORING TIERS, deliberately separate:
  * gate   -- structural + production rules (engine_gate_errors). This is the real bar;
              a failing script would trigger the repair loop in production.
  * facts / drift -- REVIEW FLAGS, not gates. `facts` checks that required concepts appear
              (OR-groups: any phrasing within a group, every group required). `drift`
              flags terms from a confusable topic (AVL vs Red-Black, etc.). Both can
              false-flag on legitimate contrastive teaching, so they inform review; they
              never block. A script can pass gate and still raise a flag.

Scoring reuses the exact gate the pipeline runs (`engine_gate_errors`,
`Script.model_validate_json`) -- eval and production can never drift apart.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models import Script, engine_gate_errors  # noqa: E402

SCHEMA_PATH = ROOT / "config" / "schemas" / "script_schema.json"
PROMPTS = {
    "dsa": ROOT / "config" / "prompts" / "dsa_system_prompt.txt",
    "design_pattern": ROOT / "config" / "prompts" / "design_pattern_system_prompt.txt",
}
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("VISULEARN_LLM_MODEL", "qwen3:8b")

# Type of an expected_facts entry after normalization: list of OR-groups.
Group = list[str]


# --------------------------------------------------------------------------- #
# Scoring -- pure functions, unit-testable without any LLM                     #
# --------------------------------------------------------------------------- #

@dataclass
class TopicScore:
    topic: str
    parsed: bool = False
    gate_pass: bool = False          # the real bar (structural + production rules)
    facts_pass: bool = True          # REVIEW FLAG (True when no expected_facts given)
    drift_pass: bool = True          # REVIEW FLAG (True when no forbidden_facts given)
    gate_errors: list[str] = field(default_factory=list)
    facts_missing: list[Group] = field(default_factory=list)   # OR-groups none of whose members appeared
    drift_hits: list[str] = field(default_factory=list)        # forbidden terms that appeared
    latency_s: float = 0.0
    error: str | None = None


def _normalize_groups(expected_facts) -> list[Group]:
    """A bare string is a 1-member OR-group; a list is an OR-group as-is."""
    groups: list[Group] = []
    for entry in expected_facts or []:
        groups.append([entry] if isinstance(entry, str) else list(entry))
    return groups


def score_output(raw: str, expected_facts=None, forbidden_facts=None) -> TopicScore:
    """Score one raw model output: structural gate + fact/drift review flags."""
    s = TopicScore(topic="")
    try:
        script = Script.model_validate_json(raw)
    except Exception as e:  # noqa: BLE001 - any parse/validation failure is a data point
        s.error = f"parse: {type(e).__name__}"
        return s
    s.parsed = True

    # Tier 1: the real gate.
    s.gate_errors = engine_gate_errors(script)
    s.gate_pass = not s.gate_errors

    blob = " ".join(seg.narration.lower() for seg in script.segments)

    # Tier 2a: expected facts (OR within a group, AND across groups).
    for group in _normalize_groups(expected_facts):
        if not any(alt.lower() in blob for alt in group):
            s.facts_missing.append(group)
    s.facts_pass = not s.facts_missing

    # Tier 2b: forbidden drift terms.
    s.drift_hits = [bad for bad in (forbidden_facts or []) if bad.lower() in blob]
    s.drift_pass = not s.drift_hits

    return s


# --------------------------------------------------------------------------- #
# Provider call -- inline Ollama (requests). Swap/extend for cloud providers.  #
# --------------------------------------------------------------------------- #

def ollama_call(model: str, system: str, user: str, schema: dict) -> str:
    """One grammar-constrained Ollama chat call -> raw JSON string."""
    import requests  # local import so scoring stays importable without the dep

    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,                 # grammar-constrained JSON
            "stream": False,
            "think": False,                   # Qwen3: no thinking mode for JSON
            "options": {"temperature": 0.4, "num_predict": 4096},
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _build_user_prompt(topic: str, schema: dict) -> str:
    return json.dumps({"topic": topic, "facts": {}, "schema": schema}, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #

def run_model(model: str, topics: list[dict], schema: dict) -> list[TopicScore]:
    results: list[TopicScore] = []
    for t in topics:
        system = PROMPTS[t["category"]].read_text(encoding="utf-8")
        user = _build_user_prompt(t["topic"], schema)
        start = time.monotonic()
        try:
            raw = ollama_call(model, system, user, schema)
            score = score_output(raw, t.get("expected_facts"), t.get("forbidden_facts"))
        except Exception as e:  # noqa: BLE001 - connection/timeout is a per-topic failure, not a crash
            score = TopicScore(topic=t["topic"], error=f"call: {type(e).__name__}")
        score.topic = t["topic"]
        score.latency_s = time.monotonic() - start
        results.append(score)
        flag = "ok  " if score.gate_pass else ("parse" if score.parsed else "FAIL")
        note = ""
        if score.parsed and not score.facts_pass:
            note += "  facts?"
        if score.parsed and not score.drift_pass:
            note += f"  DRIFT:{','.join(score.drift_hits)}"
        if score.error:
            note += f"  {score.error}"
        print(f"  [{flag}] {model:<14} {t['topic']:<22} {score.latency_s:5.1f}s{note}")
    return results


def summarize(model: str, results: list[TopicScore]) -> dict:
    n = len(results)
    parsed = sum(r.parsed for r in results) or 1
    return {
        "model": model,
        "parse_%": 100 * sum(r.parsed for r in results) / n,
        "gate_%": 100 * sum(r.gate_pass for r in results) / n,
        "facts_%": 100 * sum(r.facts_pass for r in results if r.parsed) / parsed,
        "drift_ok_%": 100 * sum(r.drift_pass for r in results if r.parsed) / parsed,
        "mean_latency_s": sum(r.latency_s for r in results) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    ap.add_argument("--topics", type=Path, default=Path(__file__).parent / "topics.yaml")
    args = ap.parse_args()

    import yaml  # eval-only dep; add pyyaml to your dev requirements
    topics = yaml.safe_load(args.topics.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    rows = []
    for model in args.models:
        print(f"\n=== {model} ===")
        rows.append(summarize(model, run_model(model, topics, schema)))

    print("\n" + "=" * 74)
    print(f"{'model':<16}{'parse%':>9}{'gate%':>9}{'facts%':>9}{'drift-ok%':>11}{'latency':>10}")
    print("-" * 74)
    for r in rows:
        print(f"{r['model']:<16}{r['parse_%']:>8.0f}%{r['gate_%']:>8.0f}%"
              f"{r['facts_%']:>8.0f}%{r['drift_ok_%']:>10.0f}%{r['mean_latency_s']:>8.1f}s")
    print("=" * 74)
    print("gate%   = first-try, no repair: THE model-selection signal (higher = less repair load).")
    print("facts%/drift-ok% = REVIEW FLAGS, not gates: they inform review, they don't block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())