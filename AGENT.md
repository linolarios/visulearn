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
| LLM (cloud, fast) | Groq — **`llama-3.3-70b-versatile`** | Free tier: 30 RPM, ~1,000 RPD, TPM (~6–12K) is the *binding* constraint. Use `response_format`. |
| LLM (cloud, reliable JSON) | Google AI Studio — Gemini 2.5 Flash | Most generous free tier; native `responseSchema`. Recommended default for the Script Engine. |
| LLM (local, small) | Ollama — **`llama3.1:8b`** (NOT "llama 3.3 8B" — that size does not exist) | Use Ollama structured outputs (`format=<json schema>`). |
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

---

## 4. Stages & Acceptance Criteria

Each stage has a "**Done when**" gate. Do not proceed to the next stage until it passes.

**Input Gateway** — Done when: a free-text topic resolves to `{canonical, category, family}`
against the taxonomy (LeetCode patterns, GoF, CLRS). Unknown topics route to a disambiguation
LLM call (this *is* an LLM call — budget it). Reject non-DSA/non-pattern topics cleanly.

**Research Agent** — Done when: returns a fact sheet `{definition, inventor, year, use_cases,
misconceptions}` from Wikipedia (primary) with `ddgs` fallback. Must not hang or crash on
network/rate-limit — wrap in timeout + graceful empty-result path. Grounding is *advisory*;
for canonical topics the 70B model already knows the complexities.

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
  planning), not per video = 1 call. Respect Groq TPM as the real limiter.
- CLI: `python visulearn.py script "Red-Black Tree" --out output/scripts/rbt.json`
  and `python visulearn.py video output/scripts/rbt.json --out output/videos/rbt.mp4`.
  Batch: `--batch topics.txt`.

---

## 7. Testing

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
cache on re-run, and stays within the chosen free tier. No stage emits unvalidated JSON.

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
