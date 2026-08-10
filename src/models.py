"""VisuLearn data contract — the single source of truth for the pipeline.

Everything downstream of the Script Engine consumes *validated* objects from this
module, never raw dicts. `config/schemas/script_schema.json` is generated FROM these
models (see scripts/generate_schema.py) and must not be hand-edited.

Two layers of checking, deliberately kept separate (see AGENT.md):

  * Structural validation (this module's models): types, enums, ranges, non-empty.
    A Script that parses is well-formed.
  * Stage gates (`engine_gate_errors`): business rules the Script Engine must pass
    before handing off — segment count, TTS-safe narration. These return a list of
    human-readable errors designed to be fed straight back into a repair prompt.

The split is why a valid `Script` type stays reusable (e.g. a short-form video with
five segments is still a valid Script) while the Script Engine can still enforce the
stricter 8–12-segment production gate.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --------------------------------------------------------------------------- #
# Enumerations                                                                 #
# --------------------------------------------------------------------------- #

class Category(str, Enum):
    DSA = "dsa"
    DESIGN_PATTERN = "design_pattern"


class Difficulty(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class VisualCue(str, Enum):
    """The closed set of scene tags a segment may request.

    A segment's ``visual_cue`` is the contract with the Storyboard Generator:
    it names *what to show*, decoupled from the segment's semantic ``type``.
    """
    TITLE_CARD = "title_card"
    WHAT_IS = "what_is"
    INTUITION = "intuition"
    ANALOGY_SLIDE = "analogy_slide"
    DS_GRAPH = "ds_graph"
    ALGORITHM_ANIM = "algorithm_anim"
    CODE_WALKTHROUGH = "code_walkthrough"
    COMPLEXITY_TABLE = "complexity_table"
    FLOWCHART = "flowchart"
    TRADEOFF_GRID = "tradeoff_grid"
    MNEMONIC_CARD = "mnemonic_card"
    CHEAT_SHEET = "cheat_sheet"
    WHEN_TO_APPLY = "when_to_apply"
    CLOSING = "closing"


class Renderer(str, Enum):
    PILLOW = "pillow"
    GRAPHVIZ = "graphviz"
    MERMAID = "mermaid"
    MANIM = "manim"
    PYGMENTS = "pygments"


# The rule-based mapping the Storyboard Generator applies. Kept here because it is
# part of the contract: exactly one renderer per scene tag, no ambiguity. Manim is
# reserved for algorithm_anim only (and even then falls back to Pillow when no scene
# template matches — see AGENT.md §5.3).
SCENE_RENDERER_MAP: dict[VisualCue, Renderer] = {
    VisualCue.TITLE_CARD: Renderer.PILLOW,
    VisualCue.WHAT_IS: Renderer.PILLOW,
    VisualCue.INTUITION: Renderer.PILLOW,
    VisualCue.ANALOGY_SLIDE: Renderer.PILLOW,
    VisualCue.DS_GRAPH: Renderer.GRAPHVIZ,
    VisualCue.ALGORITHM_ANIM: Renderer.MANIM,
    VisualCue.CODE_WALKTHROUGH: Renderer.PYGMENTS,
    VisualCue.COMPLEXITY_TABLE: Renderer.PILLOW,
    VisualCue.FLOWCHART: Renderer.MERMAID,
    VisualCue.TRADEOFF_GRID: Renderer.PILLOW,
    VisualCue.MNEMONIC_CARD: Renderer.PILLOW,
    VisualCue.CHEAT_SHEET: Renderer.PILLOW,
    VisualCue.WHEN_TO_APPLY: Renderer.PILLOW,
    VisualCue.CLOSING: Renderer.PILLOW,
}

# Vetted Manim scene templates (v1). An algorithm_anim segment must resolve to one of
# these or degrade to a Pillow slide. The agent parameterizes these — it never authors
# new Manim Scene subclasses at generation time.
MANIM_TEMPLATES: frozenset[str] = frozenset(
    {"bst_insert", "tree_rotation", "sort_partition", "bfs_layers", "stack_push_pop"}
)

# Minimum seconds any scene is shown, even if its narration is shorter.
MIN_DISPLAY_TIME_SECONDS = 3

# Patterns that indicate un-normalized math/symbols Kokoro will mispronounce. The
# Script Engine gate rejects these so narration stays TTS-safe ("order log n", not "O(log n)").
_UNSAFE_NARRATION_PATTERNS = (
    (re.compile(r"[OΘΩοO]\s*\([^)]*\)"), "Big-O/Theta/Omega notation like 'O(n)' — write 'order n'"),
    (re.compile(r"\bΘ\b|\bΩ\b"), "raw complexity symbols Θ/Ω"),
    (re.compile(r"(?<![A-Za-z])[≤≥≠±×÷→←⇒]"), "math symbols (≤ ≥ ≠ → …) — spell them out"),
)


# --------------------------------------------------------------------------- #
# Script models (Script Engine output)                                        #
# --------------------------------------------------------------------------- #

class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., min_length=1, description="Canonical topic name, e.g. 'Red-Black Tree'.")
    category: Category
    difficulty: Difficulty = Difficulty.INTERMEDIATE
    target_duration_seconds: int = Field(300, ge=60, le=1200)
    language: str = Field("en", min_length=2, max_length=8)


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(..., ge=1)
    type: str = Field(..., min_length=1, description="Semantic segment type, e.g. 'invariants'. Free-form.")
    narration: str = Field(..., min_length=1)
    visual_cue: VisualCue
    duration_hint: int = Field(..., ge=1, le=120, description="Seconds; the audio may override this.")

    @field_validator("narration")
    @classmethod
    def _narration_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("narration must not be blank")
        return v


class Script(BaseModel):
    """A validated educational script. Parsing == well-formed; see engine_gate_errors
    for the stricter production gate."""
    model_config = ConfigDict(extra="forbid")

    meta: Meta
    segments: list[Segment] = Field(..., min_length=1)
    # Keyed by language name ("python", "cpp", …). Values are source strings.
    code_template: dict[str, str] = Field(default_factory=dict)

    @field_validator("segments")
    @classmethod
    def _ids_unique_and_ordered(cls, segs: list[Segment]) -> list[Segment]:
        ids = [s.id for s in segs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"segment ids must be unique, got {ids}")
        if ids != sorted(ids):
            raise ValueError(f"segment ids must be in ascending order, got {ids}")
        return segs

    @field_validator("code_template")
    @classmethod
    def _no_blank_code(cls, tpl: dict[str, str]) -> dict[str, str]:
        for lang, src in tpl.items():
            if not lang.strip():
                raise ValueError("code_template has an empty language key")
            if not src.strip():
                raise ValueError(f"code_template['{lang}'] is empty")
        return tpl


# --------------------------------------------------------------------------- #
# Storyboard models (Storyboard Generator output)                             #
# --------------------------------------------------------------------------- #

class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: int = Field(..., ge=1)
    visual_cue: VisualCue
    renderer: Renderer
    duration_seconds: int = Field(..., ge=MIN_DISPLAY_TIME_SECONDS, le=120)
    # Present only for algorithm_anim scenes that matched a vetted template.
    manim_template: str | None = None

    @field_validator("manim_template")
    @classmethod
    def _known_template(cls, v: str | None) -> str | None:
        if v is not None and v not in MANIM_TEMPLATES:
            raise ValueError(f"unknown manim template {v!r}; must be one of {sorted(MANIM_TEMPLATES)}")
        return v


class Storyboard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., min_length=1)
    scenes: list[Scene] = Field(..., min_length=1)


# --------------------------------------------------------------------------- #
# Stage gates & helpers                                                        #
# --------------------------------------------------------------------------- #

def engine_gate_errors(
    script: Script, *, min_segments: int = 8, max_segments: int = 12
) -> list[str]:
    """Production gate for the Script Engine. Returns human-readable errors (empty == pass).

    These are the strings you feed back into a single repair attempt. They are business
    rules layered on top of structural validation, not type invariants — so the `Script`
    type itself stays reusable for short-form or test scripts.
    """
    errors: list[str] = []

    n = len(script.segments)
    if not (min_segments <= n <= max_segments):
        errors.append(
            f"expected {min_segments}-{max_segments} segments, got {n}"
        )

    for seg in script.segments:
        for pattern, why in _UNSAFE_NARRATION_PATTERNS:
            if pattern.search(seg.narration):
                errors.append(
                    f"segment {seg.id} narration contains {why}"
                )
                break  # one complaint per segment is enough for a repair prompt

    # A title and a closing bookend every finished explainer.
    cues = {s.visual_cue for s in script.segments}
    if VisualCue.TITLE_CARD not in cues:
        errors.append("no title_card segment — every explainer opens on one")
    if VisualCue.CLOSING not in cues:
        errors.append("no closing segment — every explainer ends on one")

    return errors


def build_storyboard(script: Script) -> Storyboard:
    """Rule-based Script → Storyboard mapping. Deterministic, no LLM.

    Each segment becomes exactly one Scene. algorithm_anim segments are matched to a
    vetted Manim template by inspecting the segment type; unmatched ones stay as
    algorithm_anim but carry no template, signalling the Asset Factory to fall back to
    a Pillow slide (fail-soft).
    """
    scenes: list[Scene] = []
    for seg in script.segments:
        renderer = SCENE_RENDERER_MAP[seg.visual_cue]
        template = None
        if seg.visual_cue is VisualCue.ALGORITHM_ANIM:
            template = _match_manim_template(seg)
            if template is None:
                renderer = Renderer.PILLOW  # fail-soft: no template → static slide
        scenes.append(
            Scene(
                segment_id=seg.id,
                visual_cue=seg.visual_cue,
                renderer=renderer,
                duration_seconds=max(seg.duration_hint, MIN_DISPLAY_TIME_SECONDS),
                manim_template=template,
            )
        )
    return Storyboard(topic=script.meta.topic, scenes=scenes)


def _match_manim_template(seg: Segment) -> str | None:
    """Heuristic map from a segment to a vetted Manim template, or None."""
    haystack = f"{seg.type} {seg.narration}".lower()
    keyword_to_template = {
        "rotat": "tree_rotation",
        "insert": "bst_insert",
        "partition": "sort_partition",
        "quicksort": "sort_partition",
        "bfs": "bfs_layers",
        "breadth": "bfs_layers",
        "stack": "stack_push_pop",
        "push": "stack_push_pop",
    }
    for keyword, template in keyword_to_template.items():
        if keyword in haystack and template in MANIM_TEMPLATES:
            return template
    return None
