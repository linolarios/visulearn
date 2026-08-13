"""Research Agent — ground the LLM with real facts. See AGENT.md §1/§4.

Primary source: Wikipedia API (deterministic, no cloud-IP blocking).
Fallback: ddgs (NOT duckduckgo-search, which was frozen Jul 2025).
Every network call is wrapped in a timeout; on any failure we return a graceful
empty fact sheet rather than crash — grounding is advisory (AGENT.md §4): a capable
model already knows canonical CS topics, so an empty sheet must never break the
Script Engine downstream.

Network dependencies are kept lazy so the module imports with only `requests`
(the only transport CI installs + uses); `ddgs` is imported only if and when the
fallback path actually runs.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Optional

import requests

log = logging.getLogger(__name__)

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# Keys every fact sheet carries. inventor/year/use_cases/misconceptions are advisory
# extras: Wikipedia infobox parsing is out of v1 scope, so v1 populates `definition`
# robustly and leaves the rest as declared empty fields the Script Engine prompt can
# still reference without a KeyError.
FACT_KEYS = ("definition", "inventor", "year", "use_cases", "misconceptions")

# Tokens that end in a period but do NOT end a sentence. Lowercase, no trailing dot.
# Extend as real Wikipedia intros surface new ones. Keeping "etc"/"e.g"/"i.e" here is
# what prevents a definition from being truncated to a fragment like "..., e.g." when a
# lowercase clause follows (which would drop the predicate).
_ABBREVIATIONS = frozenset({
    "e.g", "i.e", "a.k.a", "etc", "vs", "cf", "al", "et al", "fig", "no", "eq",
    "approx", "ca", "resp", "dr", "mr", "mrs", "ms", "prof", "st",
})

# Trailing run of letters/dots immediately before a candidate full stop (for abbrev check).
_WORD_TAIL = re.compile(r"[A-Za-z.]+$")


def _empty_sheet() -> dict:
    return {
        "definition": "",
        "inventor": "",
        "year": "",
        "use_cases": [],
        "misconceptions": [],
    }


def _first_sentence(text: str) -> str:
    """First sentence of a paragraph: whitespace-normalized, TTS-safe.

    Ends at the first end-punctuation that is BOTH outside parentheses AND a genuine
    sentence boundary. Three things are deliberately NOT treated as boundaries:
      * end-punctuation inside parentheses — so "O(n log n)" never truncates mid-phrase;
      * a period followed immediately by a non-space char — internal dots of "e.g",
        "Node.js", and decimals like "3.5" (a real full stop is followed by space/EOS);
      * a period after a known abbreviation ("e.g.", "i.e.", "Fig.") even when a space
        follows — otherwise a lowercase continuation would lose the rest of the sentence.
    '!' and '?' are always boundaries at paren-depth 0 (they don't appear in abbreviations).
    """
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    depth = 0
    n = len(text)
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and ch in "!?":
            return text[: i + 1].strip()
        elif depth == 0 and ch == ".":
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt and not nxt.isspace():
                continue  # internal dot: "e.g", "Node.js", "3.5" — not a boundary
            match = _WORD_TAIL.search(text[:i])
            token = (match.group(0) if match else "").strip(".").lower()
            if token in _ABBREVIATIONS:
                continue  # "e.g." / "Fig." followed by space — still not a boundary
            return text[: i + 1].strip()
    return text


# --------------------------------------------------------------------------- #
# Wikipedia (primary)                                                         #
# --------------------------------------------------------------------------- #

def _fetch_wikipedia_json(topic: str, timeout_s: float) -> Optional[dict]:
    """GET the Wikipedia intro extract. Returns None on ANY network/HTTP error."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "exlimit": "1",
        "redirects": "1",
        "titles": topic,
    }
    try:
        resp = requests.get(WIKIPEDIA_API_URL, params=params, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return None


def _wikipedia_to_sheet(topic: str, payload: Optional[dict]) -> dict:
    """Map a Wikipedia query payload to a fact sheet (definition filled best-effort)."""
    sheet = _empty_sheet()
    if not payload:
        return sheet
    pages = payload.get("query", {}).get("pages") or []
    extract = ""
    # formatversion=2 -> pages is a list; the legacy shape is a dict keyed by pageid.
    for page in (pages.values() if isinstance(pages, dict) else pages):
        extract = (page.get("extract") or "").strip()
        if extract:
            break
    sheet["definition"] = _first_sentence(extract)
    return sheet


# --------------------------------------------------------------------------- #
# ddgs (fallback)                                                             #
# --------------------------------------------------------------------------- #

def _ddgs_search(topic: str, timeout_s: float) -> list:
    """Distinct top DDG snippet bodies, SOFT-timed out so it can never hang."""

    def _run() -> list:
        try:
            from ddgs import DDGS  # renamed package (frozen prior name is fallback)
        except ImportError:  # pragma: no cover
            from duckduckgo_search import DDGS  # type: ignore
        rows = DDGS().text(topic, max_results=5) or []
        return [r.get("body", "") for r in rows if r.get("body")]

    holder: dict = {}

    def _target() -> None:
        try:
            holder["val"] = _run()
        except Exception as exc:  # noqa: BLE001 - any search failure degrades to empty
            holder["val"] = []
            log.debug("ddgs fallback failed for %r: %s", topic, exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        # Gave up on the budget. The orphaned daemon is bounded by fallback RARITY —
        # ddgs runs only when Wikipedia also failed, which for canonical CS topics is
        # uncommon — NOT by anything cancelling it. Do not call this in a hot loop.
        return []
    return holder.get("val", [])


def _ddgs_to_sheet(topic: str, snippets: list) -> dict:
    sheet = _empty_sheet()
    first_body = next((s for s in (snippets or []) if s), "")
    sheet["definition"] = _first_sentence(first_body)
    return sheet


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def build_fact_sheet(canonical_topic: str, timeout_s: float = 8.0) -> dict:
    """Return {definition, inventor, year, use_cases, misconceptions}. Never raises.

    Wikipedia is primary (deterministic, no cloud-IP blocking); ddgs is the fallback.
    If both fail we return a graceful empty sheet — grounding is advisory (AGENT.md §4)
    and must never crash a downstream stage.
    """
    topic = (canonical_topic or "").strip()
    if not topic:
        return _empty_sheet()  # nothing to research -> empty, no network calls

    sheet = _wikipedia_to_sheet(topic, _fetch_wikipedia_json(topic, timeout_s))
    # A truthy definition is the sole fallback trigger: empty extract, None payload, and
    # a whitespace-only first sentence all collapse to "" here and hand off to ddgs.
    if sheet["definition"]:
        return sheet  # Wikipedia answered -> never touch ddgs (primary wins)

    return _ddgs_to_sheet(topic, _ddgs_search(topic, timeout_s))