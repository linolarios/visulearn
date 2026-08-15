"""Pygments code renderer — deterministic 1920x1080 PNG of syntax-highlighted code.

Takes the Script's code templates (dict keyed by language) through the `code_templates`
seam and draws the language's source token-by-token with Pygments colors using Pillow's
embedded font (no system-font dependency, AGENT.md §5.2). Deterministic by construction
(no randomness) for idempotent caching. Any failure falls back to a Pillow slide in the
dispatcher (Golden Rule 4).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from models import Scene
from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.styles import get_style_by_name

WIDTH, HEIGHT = 1920, 1080
BG = (18, 18, 28)
FG = (208, 214, 224)        # default token color when the style has none
PAD_X, PAD_Y = 64, 72
LINE_H = 28

_style = get_style_by_name("friendly")


def _token_color(token) -> tuple[int, int, int]:
    """Pygments style foreground for a token -> (r, g, b)."""
    hex6 = _style.style_for_token(token).get("color")
    if not hex6:
        return FG
    return tuple(int(hex6[i:i + 2], 16) for i in (0, 2, 4))


def _draw_code(draw: ImageDraw.ImageDraw, code: str, lang: str) -> None:
    try:
        lexer = get_lexer_by_name(lang)
    except Exception:  # noqa: BLE001 - unknown lang -> plain text passes through Pygments
        lexer = get_lexer_by_name("text")
    font = ImageFont.load_default()
    x, y = PAD_X, PAD_Y
    for tok, value in lex(code, lexer):
        color = _token_color(tok)
        lines = value.split("\n")
        for i, line in enumerate(lines):
            if line:
                draw.text((x, y), line, font=font, fill=color)
                x += draw.textlength(line, font=font)
            if i < len(lines) - 1:  # newline -> advance cursor
                x = PAD_X
                y += LINE_H
                if y > HEIGHT - PAD_Y:
                    return  # clip long code; still deterministic


def render(scene: Scene, code_templates: Optional[dict], out_path: Path) -> Path:
    """Draw the scene's highlighted code (or a placeholder) and save a 1920x1080 PNG."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    src, lang = "", "text"
    if isinstance(code_templates, dict):
        for name, value in code_templates.items():
            if value:
                src, lang = value, name
                break

    if src:
        _draw_code(draw, src, lang)
    else:
        draw.text((PAD_X, PAD_Y), "(no code template)", fill=FG)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return out_path
