"""Input Gateway — validate + normalize a raw topic string. See AGENT.md §4.

Maps free text to {canonical, category, family} against a curated taxonomy
(LeetCode/CLRS families for DSA; GoF families for design patterns). Known topics
resolve deterministically via exact/alias match; UNKNOWN topics route to a
disambiguation provider (an LLM call — budget it, AGENT.md §4). Inputs that neither
the taxonomy nor the provider can place are REJECTED cleanly (raise, never crash).

The provider seam is a plain callable `provider(raw_text) -> dict | None` returning
{"canonical", "category", "family"}; production wraps the Script Engine's LLM
Provider, tests inject a stub so nothing touches the network (AGENT.md §7).
"""
from __future__ import annotations

import re
from typing import Callable, Optional

CATEGORY_DSA = "dsa"
CATEGORY_DESIGN_PATTERN = "design_pattern"
VALID_CATEGORIES = frozenset({CATEGORY_DSA, CATEGORY_DESIGN_PATTERN})


# canonical display name -> {"category", "family"}
_TAXONOMY: dict[str, dict[str, str]] = {
    # ---- DSA (LeetCode/CLRS families) ----
    "Red-Black Tree": {"category": CATEGORY_DSA, "family": "Trees"},
    "Binary Search Tree": {"category": CATEGORY_DSA, "family": "Trees"},
    "AVL Tree": {"category": CATEGORY_DSA, "family": "Trees"},
    "B-Tree": {"category": CATEGORY_DSA, "family": "Trees"},
    "Trie": {"category": CATEGORY_DSA, "family": "Trees"},
    "Segment Tree": {"category": CATEGORY_DSA, "family": "Trees"},
    "Binary Heap": {"category": CATEGORY_DSA, "family": "Trees"},
    "Merge Sort": {"category": CATEGORY_DSA, "family": "Sorting"},
    "Quick Sort": {"category": CATEGORY_DSA, "family": "Sorting"},
    "Heap Sort": {"category": CATEGORY_DSA, "family": "Sorting"},
    "Insertion Sort": {"category": CATEGORY_DSA, "family": "Sorting"},
    "Counting Sort": {"category": CATEGORY_DSA, "family": "Sorting"},
    "Binary Search": {"category": CATEGORY_DSA, "family": "Searching"},
    "Linear Search": {"category": CATEGORY_DSA, "family": "Searching"},
    "Breadth-First Search": {"category": CATEGORY_DSA, "family": "Graphs"},
    "Depth-First Search": {"category": CATEGORY_DSA, "family": "Graphs"},
    "Dijkstra's Algorithm": {"category": CATEGORY_DSA, "family": "Graphs"},
    "Bellman-Ford Algorithm": {"category": CATEGORY_DSA, "family": "Graphs"},
    "Topological Sort": {"category": CATEGORY_DSA, "family": "Graphs"},
    "Union-Find": {"category": CATEGORY_DSA, "family": "Graphs"},
    "Hash Table": {"category": CATEGORY_DSA, "family": "Hashing"},
    "Hash Map": {"category": CATEGORY_DSA, "family": "Hashing"},
    "Hash Set": {"category": CATEGORY_DSA, "family": "Hashing"},
    "Stack": {"category": CATEGORY_DSA, "family": "Stacks & Queues"},
    "Queue": {"category": CATEGORY_DSA, "family": "Stacks & Queues"},
    "Deque": {"category": CATEGORY_DSA, "family": "Stacks & Queues"},
    "Priority Queue": {"category": CATEGORY_DSA, "family": "Stacks & Queues"},
    "Singly Linked List": {"category": CATEGORY_DSA, "family": "Linked Lists"},
    "Doubly Linked List": {"category": CATEGORY_DSA, "family": "Linked Lists"},
    "0-1 Knapsack": {"category": CATEGORY_DSA, "family": "Dynamic Programming"},
    "Longest Common Subsequence": {"category": CATEGORY_DSA, "family": "Dynamic Programming"},
    "Edit Distance": {"category": CATEGORY_DSA, "family": "Dynamic Programming"},
    "N-Queens": {"category": CATEGORY_DSA, "family": "Backtracking"},
    # ---- GoF design patterns ----
    "Singleton": {"category": CATEGORY_DESIGN_PATTERN, "family": "Creational"},
    "Factory Method": {"category": CATEGORY_DESIGN_PATTERN, "family": "Creational"},
    "Abstract Factory": {"category": CATEGORY_DESIGN_PATTERN, "family": "Creational"},
    "Builder": {"category": CATEGORY_DESIGN_PATTERN, "family": "Creational"},
    "Prototype": {"category": CATEGORY_DESIGN_PATTERN, "family": "Creational"},
    "Adapter": {"category": CATEGORY_DESIGN_PATTERN, "family": "Structural"},
    "Composite": {"category": CATEGORY_DESIGN_PATTERN, "family": "Structural"},
    "Decorator": {"category": CATEGORY_DESIGN_PATTERN, "family": "Structural"},
    "Facade": {"category": CATEGORY_DESIGN_PATTERN, "family": "Structural"},
    "Proxy": {"category": CATEGORY_DESIGN_PATTERN, "family": "Structural"},
    "Flyweight": {"category": CATEGORY_DESIGN_PATTERN, "family": "Structural"},
    "Bridge": {"category": CATEGORY_DESIGN_PATTERN, "family": "Structural"},
    "Observer": {"category": CATEGORY_DESIGN_PATTERN, "family": "Behavioral"},
    "Strategy": {"category": CATEGORY_DESIGN_PATTERN, "family": "Behavioral"},
    "Command": {"category": CATEGORY_DESIGN_PATTERN, "family": "Behavioral"},
    "Iterator": {"category": CATEGORY_DESIGN_PATTERN, "family": "Behavioral"},
    "State": {"category": CATEGORY_DESIGN_PATTERN, "family": "Behavioral"},
    "Template Method": {"category": CATEGORY_DESIGN_PATTERN, "family": "Behavioral"},
    "Visitor": {"category": CATEGORY_DESIGN_PATTERN, "family": "Behavioral"},
    "Mediator": {"category": CATEGORY_DESIGN_PATTERN, "family": "Behavioral"},
    "Memento": {"category": CATEGORY_DESIGN_PATTERN, "family": "Behavioral"},
}

# Raw alias -> canonical display name (lookup normalizes the key).
_ALIASES: dict[str, str] = {
    "rbt": "Red-Black Tree",
    "red black": "Red-Black Tree",
    "bst": "Binary Search Tree",
    "binary tree": "Binary Search Tree",
    "avl": "AVL Tree",
    "dijkstra": "Dijkstra's Algorithm",
    "dijkstra's": "Dijkstra's Algorithm",
    "dijkstras": "Dijkstra's Algorithm",
    "bfs": "Breadth-First Search",
    "dfs": "Depth-First Search",
    "factory": "Factory Method",
    "factory pattern": "Factory Method",
    "singleton pattern": "Singleton",
}


class TopicError(ValueError):
    """Base: the gateway could not map the raw topic to a valid target."""


class UnknownTopicError(TopicError):
    """Not in the taxonomy and no provider (or the provider could not resolve it)."""


class TopicRejection(TopicError):
    """The provider answered, but the answer is out-of-domain or malformed."""


def _canonical_key(text: Optional[str]) -> str:
    """Normalize free text into a stable lookup key (case/whitespace/punctuation-insensitive)."""
    t = re.sub(r"\s+", " ", (text or "").strip()).lower()
    t = re.sub(r"[\-_]+", " ", t)          # "red-black"/"red_black" -> "red black"
    t = re.sub(r"[^a-z0-9 ]+", "", t)      # drop apostrophes/punct: "Dijkstra's"->"dijkstra s"
    return re.sub(r"\s+", " ", t).strip()


def _build_lookup() -> dict[str, str]:
    """canonical_key -> canonical display name, from taxonomy + aliases."""
    lookup: dict[str, str] = {_canonical_key(c): c for c in _TAXONOMY}
    for alias, canon in _ALIASES.items():
        lookup[_canonical_key(alias)] = canon
    return lookup


def _resolve_via_provider(raw: str, provider: Callable[[str], Optional[dict]]) -> dict:
    """Ask the disambiguation provider, then validate its answer.

    Any provider failure (raise, None, malformed, out-of-domain) degrades to a clean
    TopicError subclass — never a crash. This is the budgeted LLM call (AGENT.md §4).
    """
    try:
        result = provider(raw)
    except Exception as exc:  # noqa: BLE001 - provider outage must reject, not crash
        raise UnknownTopicError(raw) from exc
    if not result:
        raise UnknownTopicError(raw)

    canonical = (result.get("canonical") or "").strip()
    category = (result.get("category") or "").strip().lower()
    family = (result.get("family") or "").strip()
    if not canonical or not family or category not in VALID_CATEGORIES:
        raise TopicRejection(raw)
    return {"canonical": canonical, "category": category, "family": family}


def normalize_topic(raw: str, *, provider: Optional[Callable[[str], Optional[dict]]] = None) -> dict:
    """Return {'canonical': str, 'category': 'dsa'|'design_pattern', 'family': str}.

    Deterministic taxonomy/alias resolve first; unknown topics route to `provider`
    (a budgeted LLM disambiguation call); irreducible inputs raise cleanly.
    """
    key = _canonical_key(raw)
    if not key:
        raise TopicError("empty topic")

    canonical = _build_lookup().get(key)
    if canonical is not None:
        meta = _TAXONOMY[canonical]
        return {"canonical": canonical, "category": meta["category"], "family": meta["family"]}

    if provider is None:
        raise UnknownTopicError(raw)
    return _resolve_via_provider(raw, provider)
