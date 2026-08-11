# AGENT.md — VisuLearn

Automated educational-video generator for DSA & Design Patterns.
Pipeline: **Topic → Research → Script (JSON) → Storyboard → Assets → TTS → Assembly → MP4**.
Stack target: 100% open source, permanent free tiers.

This file is the contract for an autonomous coding agent building or extending VisuLearn.
Read it fully before writing code. The **Golden Rules** are non-negotiable.

---

## 0. Golden Rules (do not violate)

1. **JSON is produced by the provider's structured-output mode, never by prompt-and-pray.**
   Use Groq `response_format`, Ollama `format=<schema>`, or Gemini `responseSchema`.
   Every model output is validated by Pydantic before it moves downstream. On validation
   failure: feed the error back for **one** repair attempt, then hard-fail and log the raw
   output. Never silently retry the identical prompt into the identical context.
2. **The Pydantic schema is the single source of truth.** `config/schemas/script_schema.json`
   is generated *from* the Pydantic models, not hand-edited. Downstream stages consume
   validated objects, not raw dicts.
3. **Manim is never generated free-form.** The agent fills parameters into vetted scene
   templates only (see §5.3). A topic that has no matching template falls back to a Pillow
   slide. Do not let the agent write arbitrary Manim per video.
4. **Fail-soft, never fail-whole.** One bad scene degrades to a static Pillow slide with a
   warning overlay. A single renderer crash must not kill the run.
5. **Every stage is idempotent and cached.** Re-running a topic must resume from the last
   good artifact, not recompute research/script/audio that already succeeded.
6. **Pin versions that have breaking releases** (MoviePy, ddgs). See §3.

---

## 1. Corrected Tech Stack

| Layer | Tool | Notes / corrections vs. original design doc |
|---|---|---|
| Language | Python 3.10+ | `ddgs` requires ≥3.10 |
| Validation | Pydantic v2 | Source of truth for the script schema |
| Research | **`ddgs`** (NOT `duckduckgo-search`) + Wikipedia API | `duckduckgo-search` was frozen Jul 2025 → renamed `ddgs`. Import `from ddgs import DDGS`. DDG blocks cloud IPs — prefer Wikipedia API for canonical CS facts; DDG is fallback. Drop arxiv for this domain (GoF/CLRS aren't on arxiv). |
| LLM (local, **recommended**) | Ollama — **`qwen3:14b`** (instruct/non-thinking) | **No token limits — this is the durable answer to free-tier caps.** Apache 2.0. Grammar-constrained JSON via `format=<schema>`. Size by VRAM: `qwen3:8b` (8 GB), `qwen3:14b` (16 GB), `qwen3.6:27b`/`qwen3:30b-a3b` (24 GB). CPU-only is fine for a batch pipeline. See §1.1. |
| LLM (cloud, reliable JSON) | Google AI Studio — Gemini 2.5 Flash | Native `responseSchema`. Good quality, but **free-tier token/RPD caps are restrictive for a multi-call pipeline** — you *will* exhaust them generating many videos. Fine for low volume; not for batch. |
| LLM (cloud, fast) | Groq — **`llama-3.3-70b-versatile`** | Free tier: 30 RPM, ~1,000 RPD, TPM (~6–12K) is the *binding* constraint. Use `response_format`. Same free-tier-cap caveat as Gemini. |
| LLM (dev/CI only) | Ollama — `gemma3:4b` | Cheap smoke-test model. Emits schema-valid JSON (grammar-constrained) but content quality is weak — do NOT ship scripts from it. Also note: `llama3.3:8b` does NOT exist; the 8B is `llama3.1:8b`. |
| TTS | Kokoro-82M (ONNX, Apache 2.0) | **24 kHz** mono output — verify before hardcoding WAV rate. Piper (22.05 kHz) is the stable fallback. |
| Static graphics | Pillow (PIL) | Title/bullets/tables/code/cheat sheets |
| Code highlight | Pygments | |
| Graphs | Graphviz | Deterministic with `dot`; seed `neato`/`fdp` layouts |
| Flowcharts | Mermaid CLI | Needs Node.js + `@mermaid-js/mermaid-cli` |
| Animation | Manim **Community Edition** + LaTeX | Highest-risk dependency. Templates only (§5.3). |
| Assembly | **MoviePy — pin `moviepy<2`** OR migrate to 2.x API | Design doc uses 1.x API (`crossfadein`, `moviepy.editor`). 2.x renames to `.with_duration()`, `from moviepy import ...`, crossfade via `.with_effects([vfx.CrossFadeIn(0.3)])`. Pick one and pin it. |
| Encoding | FFmpeg | Required by MoviePy and Kokoro I/O |

Model choices should be re-checked against a live model list at build time — free-tier model
names rot in weeks. Verify the model exists on the account before hardcoding it (pull the
provider's live model list with the real key; the marketing page and the entitlement differ).

### 1.1 Model selection & token limits (read before wiring the Script Engine)

**Every hosted free tier has a daily cap** — Gemini, Groq, OpenRouter, Qwen Cloud alike.
A video pipeline makes several LLM calls per video (disambiguation + script + optional
planning), so at any real volume you *will* hit the ceiling. This was observed with
Gemini 2.5 Flash: quality is good, but the free-tier token/request budget runs out fast.

**The only setup with no token limit is local inference.** Running an open-weight model
in Ollama is bounded by hardware and time, not a quota — generate hundreds of scripts
overnight for the cost of electricity. This is why `qwen3:14b` (local) is the default,
not a cloud model. Keep a cloud provider (Gemini) wired behind the same interface as a
fallback / low-volume option, but do not make it the primary for batch runs.

Two non-obvious rules when the LLM is a local Qwen3:

- **Use the non-thinking (Instruct) variant.** Qwen3 has a toggleable thinking mode;
  for constrained JSON you want it OFF (pull the instruct checkpoint or disable thinking).
  Reasoning traces waste compute and add nothing when the grammar already forces JSON.
- **Raise the output-token limit.** Set Ollama's `num_predict` high enough (e.g. 4096)
  that a full 8–12-segment script isn't truncated mid-object. A truncated response is
  valid-JSON-prefix that fails validation and *looks* like a model bug — rule it out
  explicitly. (Same class of gotcha as Groq/Gemini `max_tokens`.)

`gemma3:4b` is for dev/CI smoke tests only — grammar-constrained decoding makes its JSON
*shape* valid, but a 4B model's *content* is too weak to ship. Prove quality against the
model you'll actually run.

---

## 2. Repo Layout

```
visulearn/
├── config/
│   ├── prompts/{dsa_system_prompt.txt, design_pattern_system_prompt.txt}
│   ├── schemas/script_schema.json        # GENERATED from Pydantic, do not hand-edit
│   └── templates/{markdown_template.j2, anki_template.j2}
├── src/
│   ├── input_gateway.py                  # validate + normalize topic → canonical + category
│   ├── research_agent.py                 # ddgs + Wikipedia → fact sheet
│   ├── script_engine.py                  # LLM → validated Script object (structured output)
│   ├── storyboard_generator.py           # rule-based segment → scene mapping
│   ├── asset_factory/
│   │   ├── slide_renderer.py             # Pillow
│   │   ├── graph_renderer.py             # Graphviz
│   │   ├── flowchart_renderer.py         # Mermaid
│   │   ├── animation_renderer.py         # Manim (templates only)
│   │   └── code_renderer.py              # Pygments
│   ├── tts_pipeline.py                   # segmented Kokoro
│   ├── assembler.py                      # MoviePy
│   └── models.py                         # Pydantic: Script, Segment, Storyboard, Scene
├── output/{scripts,scenes,audio,videos}/
├── tests/test_pipeline.py
├── requirements.txt
└── visulearn.py                          # CLI entry point
```

---

## 3. The Schema Contract

Define in `src/models.py` (Pydantic v2). Emit `script_schema.json` from it.

- `Script { meta: Meta, segments: list[Segment], code_template: dict[str,str] }`
- `Meta { topic, category: Literal["dsa","design_pattern"], difficulty, target_duration_seconds, language }`
- `Segment { id: int, type: str, narration: str, visual_cue: str, duration_hint: int }`
- `visual_cue` must be one of the scene tags in §5. Validate with an `Enum`.

The Script Engine returns a **validated `Script` instance**, never a raw string. Storyboard
consumes `Script`, emits a validated `Storyboard { scenes: list[Scene] }`.

**Providers do NOT accept the same schema.** The generated `script_schema.json` goes
straight into Ollama's `format=`, but Gemini's `responseSchema` is an OpenAPI subset and
Groq's `json_schema` mode has its own constraints — keywords like `additionalProperties:
false`, `$defs`/`$ref`, and some `enum`/`format` constructs may be rejected or ignored
depending on the provider. Verify the schema against each provider's *current* docs and
transform it **per-provider on the way out** (inline `$defs`, drop/rename unsupported
keywords). Never relax `src/models.py` to appease a provider — the Pydantic model stays
strict; only the outbound wire-schema is adapted. Something that works on Ollama can 500
on Gemini, and it's invisible until you switch providers.

---

## 4. Stages & Acceptance Criteria

Each stage has a "**Done when**" gate. Do not proceed to the next stage until it passes.

**Input Gateway** — Done when: a free-text topic resolves to `{canonical, category, family}`
against the taxonomy (LeetCode patterns, GoF, CLRS). Unknown topics route to a disambiguation
LLM call (this *is* an LLM call — budget it). Reject non-DSA/non-pattern topics cleanly.

**Research Agent** — Done when: returns a fact sheet `{definition, inventor, year, use_cases,
misconceptions}` from Wikipedia (primary) with `ddgs` fallback. Must not hang or crash on
network/rate-limit — wrap in timeout + graceful empty-result path. Grounding is *advisory*;
for canonical topics a capable model already knows the complexities.

**Script Engine** — Done when: output parses as a valid `Script` (Pydantic) with 8–12 segments,
every `visual_cue` in the enum, narration free of raw symbols (`O(n)` → "order n"). Uses
provider structured output + one repair retry. Log raw output on hard-fail.

**Storyboard Generator** — Done when: every segment maps to exactly one `Scene {tag, renderer,
duration}`. Rule-based (`code_template→code_walkthrough`, `analogy→analogy_slide`,
`complexity→complexity_table`). Optional LLM scene-planning is a nice-to-have, not required.

**Asset Factory** — Done when: every scene yields a 1920×1080 PNG or a 1080p MP4 segment.
Any renderer failure falls back to a Pillow slide (Golden Rule 4). Deterministic output:
seed all layout randomness so re-runs are byte-identical (needed for caching).

**TTS** — Done when: one WAV per segment at the correct Kokoro sample rate, non-empty,
duration > 0. Normalize technical terms before synthesis.

**Assembly** — Done when: H.264 MP4, 1920×1080, 30fps, AAC audio. Each scene duration =
`max(audio_duration, min_display_time)`. Crossfades 0.3s. Audio synced per segment.

---

## 5. Scene Catalog & Renderer Rules

### 5.1 Renderers
- **Pillow**: title_card, what_is, complexity_table, code_walkthrough, tradeoff_grid,
  mnemonic_card, cheat_sheet, when_to_apply, closing, intuition (with SVG overlay)
- **Graphviz**: ds_graph (trees, graphs, linked lists, hash maps, tries) — SVG/hi-res PNG
- **Mermaid**: flowchart (decision trees, "when to use X vs Y")
- **Pygments**: syntax highlighting feeding into code_walkthrough
- **Manim**: algorithm_anim only (see 5.3)

### 5.2 Fonts & assets
Ship OFL/redistributable fonts in `assets/fonts/`. Do not depend on system fonts. Category
icons and backgrounds live in `assets/`.

### 5.3 Manim — templates only
Build exactly these 5 parameterized scenes and no more in v1:
`bst_insert`, `tree_rotation`, `sort_partition`, `bfs_layers`, `stack_push_pop`.
The agent selects a template and passes parameters (values, node list, etc.). It must not
author new Manim `Scene` subclasses at generation time. Any `algorithm_anim` cue with no
matching template → Pillow fallback slide. Manim renders 1080p MP4 segments (5–15s).

---

## 6. Orchestration Requirements

- Model the pipeline as an explicit **DAG of idempotent stages**, keyed by
  `(canonical_topic, stage)`. Cache each stage's output artifact on disk.
- **Resume**: on re-run, skip any stage whose cached artifact exists and validates.
- **Structured logging** per stage: inputs hash, model+params, tokens, cost, duration, ok/err.
- **Cost/quota tracking**: count LLM calls *per video* (disambiguation + script + optional
  planning), not per video = 1 call. On a cloud provider, respect the free-tier cap (Groq TPM,
  Gemini RPD/tokens) and back off before it hard-fails a batch. On local Ollama there is no
  quota — skip quota logic and just log latency/tokens (§1.1).
- CLI: `python visulearn.py script "Red-Black Tree" --out output/scripts/rbt.json`
  and `python visulearn.py video output/scripts/rbt.json --out output/videos/rbt.mp4`.
  Batch: `--batch topics.txt`.

---

## 7. Testing

**Preflight — the agent runs and reports these BEFORE writing stage code:**

1. **Contract in sync**: run `python scripts/generate_schema.py`; confirm
   `config/schemas/script_schema.json` has no diff. If it changed, the committed schema was
   stale — flag it, don't hand-edit the JSON.
2. **Baseline green**: run `pytest`; expect 9 passing. This is the floor; never drop below it.
3. **Runtime reachable**: the target backend actually responds — `ollama list` shows the
   configured model and the daemon answers on `127.0.0.1:11434`, or the cloud key env var is
   set. Prove constrained JSON end-to-end with a one-call `format=schema` smoke check.
4. **Per-provider schema accepted**: for each provider you wire, confirm the outbound schema
   is accepted (see §3) — don't assume Ollama-valid == Gemini-valid.
5. **Output not truncated**: max-output tokens (`num_predict` / `max_tokens`) is high enough
   for a full script; a truncated response fails validation and mimics a model bug.
6. **Config, not hardcode**: model name, provider, temperature come from env/config.
7. **Failures are legible**: model-not-found / connection-refused / auth errors surface a
   readable message — this is the exact class that produced "no JSON object found" upstream.

**Test suite:**

- **Schema tests**: malformed LLM output is rejected; repair loop fires once; hard-fail logs raw.
- **Renderer tests**: each renderer produces correct dimensions; forced failure triggers Pillow fallback.
- **Determinism test**: same topic rendered twice → identical scene bytes.
- **End-to-end smoke**: 3 topics (1 easy/1 medium/1 hard) produce playable MP4s. Run in CI
  with Ollama or a stubbed LLM so tests don't depend on a live free-tier quota.
- **Golden topics**: keep the Red-Black Tree script from the design doc as a fixture.

---

## 8. Definition of Done (v1)

`python visulearn.py video "Red-Black Tree"` runs topic→MP4 unattended, produces a 5-min
1080p MP4 with synced narration, degrades gracefully if Manim/LaTeX is absent, resumes from
cache on re-run, and stays within the chosen provider's limits (or runs unlimited on local
Ollama). No stage emits unvalidated JSON.

---

## 9. Known Gotchas (the short list that will actually break you)

1. `duckduckgo-search` → `ddgs` (frozen package, silent degradation).
2. "Llama 3.3 8B" does not exist → use `llama-3.1-8b-instant` / `llama3.1:8b`.
3. MoviePy 2.x renamed the entire API → pin `<2` or migrate deliberately.
4. Kokoro is 24 kHz, not 22 kHz.
5. Structured JSON must come from the provider's JSON mode + Pydantic validation, not the
   system prompt. Prompt-and-pray is why models return prose and the parser finds no JSON.
6. Manim needs LaTeX; treat its absence as a first-class fallback path, not an error.
7. Groq's binding limit is TPM, and free-tier model names change frequently — verify live.
8. **Every hosted free tier has token/RPD caps** — a multi-call video pipeline exhausts them
   (Gemini included). Local Ollama is the only limit-free path; default to it for batch (§1.1).
9. **Local truncation**: Ollama's default `num_predict` can cut a full script mid-JSON.
   Raise it (~4096). Truncated valid-prefix JSON fails validation and looks like a model bug.
10. **Provider schema drift**: Ollama `format`, Gemini `responseSchema`, and Groq `json_schema`
    accept different schema subsets — adapt the outbound schema per provider, never the model (§3).
11. **Qwen3 thinking mode**: use the Instruct (non-thinking) variant for JSON; reasoning traces
    waste tokens and add nothing under grammar-constrained decoding.