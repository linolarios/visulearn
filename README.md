# VisuLearn

Automated educational-video generator for **Data Structures, Algorithms, and Design
Patterns**. One command turns a topic string into a narrated 1080p explainer:

> **Topic → Research → Script (JSON) → Storyboard → Assets → TTS → Assembly → MP4**

100% open source, built to run on permanent free tiers.

## Status

Contract + scaffolding stage. `src/models.py` (the Pydantic schema) and the storyboard
mapping are implemented and tested; the stage modules are stubs to fill in.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_schema.py     # emit config/schemas/script_schema.json from models
pytest                                # contract tests, no LLM needed
```

## System dependencies (not pip)

`ffmpeg`, `graphviz`, a LaTeX distribution (for Manim), and Node.js + `@mermaid-js/mermaid-cli`.

## Layout & rules

See **[AGENT.md](./AGENT.md)** — it is the build contract. Highlights:

- JSON is produced by the provider's structured-output mode, then Pydantic-validated,
  then checked by `engine_gate_errors()`; one repair attempt on failure, never a blind retry.
- The Pydantic models are the single source of truth; `script_schema.json` is generated.
- Manim runs from 5 vetted scene templates only, with Pillow fail-soft.
- MoviePy is pinned to `<2` (the codebase uses the 1.x API).

## License

MIT (see LICENSE).
