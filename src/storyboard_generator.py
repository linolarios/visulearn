"""Storyboard Generator — rule-based Script -> Storyboard. See AGENT.md §4/§5.

The mapping lives in models.build_storyboard (deterministic, no LLM). This module is
the place to add optional LLM scene-planning later; for v1 it just delegates.
"""
from __future__ import annotations

from models import Script, Storyboard, build_storyboard


def plan(script: Script) -> Storyboard:
    return build_storyboard(script)
