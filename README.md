# VisuLearn

Automated educational-video generator for **Data Structures, Algorithms, and Design
Patterns**. One command turns a topic string into a narrated 1080p explainer:

> **Topic → Research → Script (JSON) → Storyboard → Assets → TTS → Assembly → MP4**

100% open source, built to run on permanent free tiers.

## Status

Implemented & tested: the Pydantic data contract (`src/models.py`), the rule-based
Storyboard mapping, and the **Script Engine** (`src/script_engine.py`) — the LLM stage
that emits validated Script JSON via Ollama / Gemini / Groq structured output, with a
bounded one-repair loop and hard-fail-with-logging on repeated failure (AGENT.md Golden
Rule 1). The model-selection harness (`eval/run_eval.py`, AGENT.md §10) scores provider
models on a fixed topic set. Still to build: Research Agent, Input Gateway, Asset
Factory, TTS, Assembly, and the CLI wiring.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_schema.py     # emit config/schemas/script_schema.json from models
pytest                                # contract tests, no LLM needed
