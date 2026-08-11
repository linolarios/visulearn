"""Opt-in live smoke test against a real local Ollama model (AGENT.md 7).

Skipped by default. Run with:  RUN_LLM_SMOKE=1 /usr/bin/python3 -m pytest -q -s
Uses llama3.1:8b explicitly (bypasses any .env override) and proves the full
Ollama structured-output -> Pydantic -> gate pipeline end-to-end.
"""
import os

import pytest

from models import Script
from script_engine import OllamaProvider, generate_script


@pytest.mark.skipif(
    os.getenv("RUN_LLM_SMOKE") != "1",
    reason="live smoke test; set RUN_LLM_SMOKE=1 to run",
)
def test_live_ollama_smoke_llama31():
    provider = OllamaProvider(
        model="llama3.1:8b",
        host="http://127.0.0.1:11434",
        temperature=0.4,
        timeout=300,
    )
    script = generate_script(
        "Red-Black Tree",
        {
            "definition": "self-balancing binary search tree",
            "inventor": "Rudolf Bayer",
            "year": 1972,
            "use_cases": ["Linux kernel rbtree", "C++ std::map", "Java TreeMap"],
        },
        provider=provider,
        category="dsa",
    )
    assert isinstance(script, Script)
    assert 8 <= len(script.segments) <= 12
    assert script.segments[0].visual_cue.value == "title_card"
    assert script.segments[-1].visual_cue.value == "closing"
    print("\nSMOKE OK: segments =", len(script.segments))
