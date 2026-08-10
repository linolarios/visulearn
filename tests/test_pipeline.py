"""Contract tests that run WITHOUT a live LLM (AGENT.md §7).

Covers: structural validation, the Script Engine production gate (repair-prompt strings),
the deterministic storyboard mapping, and Manim fail-soft.
"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from models import (
    Renderer,
    Script,
    VisualCue,
    build_storyboard,
    engine_gate_errors,
)

FIXTURE = Path(__file__).parent / "fixtures" / "rbt_script.json"


@pytest.fixture
def rbt() -> Script:
    return Script.model_validate_json(FIXTURE.read_text())


def test_fixture_parses_and_passes_gate(rbt: Script):
    assert rbt.meta.topic == "Red-Black Tree"
    assert len(rbt.segments) == 11
    assert engine_gate_errors(rbt) == []


def test_storyboard_maps_every_segment(rbt: Script):
    sb = build_storyboard(rbt)
    assert len(sb.scenes) == len(rbt.segments)
    assert {s.segment_id for s in sb.scenes} == {s.id for s in rbt.segments}


def test_manim_fail_soft_without_template(rbt: Script):
    # segment 4 is algorithm_anim (invariants) with no matching vetted template
    sb = build_storyboard(rbt)
    seg4 = next(s for s in sb.scenes if s.segment_id == 4)
    assert seg4.visual_cue is VisualCue.ALGORITHM_ANIM
    assert seg4.manim_template is None
    assert seg4.renderer is Renderer.PILLOW  # degraded to a static slide


@pytest.mark.parametrize("mutate", [
    lambda d: d["segments"][0].__setitem__("visual_cue", "fancy_animation"),
    lambda d: d["segments"][1].__setitem__("id", 1),          # duplicate id
    lambda d: d["meta"].__setitem__("category", "quantum"),   # bad enum
    lambda d: d.__setitem__("hallucinated_field", 42),        # extra field
    lambda d: d["segments"][2].__setitem__("narration", "   "),
])
def test_structural_rejections(mutate):
    d = json.loads(FIXTURE.read_text())
    mutate(d)
    with pytest.raises(ValidationError):
        Script.model_validate(d)


def test_gate_emits_repair_strings():
    d = json.loads(FIXTURE.read_text())
    d["segments"] = d["segments"][:4]                                 # too few, drops closing
    d["segments"][3]["narration"] = "Search runs in O(log n) time."   # raw Big-O
    errs = engine_gate_errors(Script.model_validate(d))
    joined = " ".join(errs)
    assert "8-12 segments" in joined
    assert "Big-O" in joined
    assert "closing" in joined
