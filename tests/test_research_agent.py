"""Research Agent tests — STUBBED fetchers; no network, no API key (AGENT.md §7).

Proves: (1) Wikipedia is primary and ddgs is untouched when it succeeds; (2) ddgs is
the fallback when Wikipedia is empty AND that it actually fires; (3) both-empty yields a
graceful empty sheet; (4) blank inputs never call any fetcher; (5) timeout_s is propagated
to both fetchers; and the pure mapping helpers (payload shapes, TTS-safe first sentence
incl. abbreviations/parentheses/decimals) are correct.
"""
import pytest

import research_agent
from research_agent import (
    FACT_KEYS,
    _ddgs_to_sheet,
    _empty_sheet,
    _first_sentence,
    _wikipedia_to_sheet,
)


class _FakeFetcher:
    """Records call count AND the timeout it received; never performs network I/O."""

    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.received_timeout = None

    def __call__(self, topic, timeout_s=None):
        self.calls += 1
        self.received_timeout = timeout_s
        return self.result


def _wiki(extract: str, title: str = "Stack (abstract data type)") -> dict:
    return {"query": {"pages": [{"pageid": 1, "title": title, "extract": extract}]}}


# --------------------------------------------------------------------------- #
# Orchestration: primary/fallback, graceful-empty, no-I/O, timeout propagation #
# --------------------------------------------------------------------------- #

def test_wikipedia_primary_and_ddgs_not_called(monkeypatch):
    extract = ("In computer science, a stack is an abstract data type. "
               "It serves two main operations: push and pop.")
    wiki = _FakeFetcher(_wiki(extract))
    ddgs = _FakeFetcher([])
    monkeypatch.setattr(research_agent, "_fetch_wikipedia_json", wiki)
    monkeypatch.setattr(research_agent, "_ddgs_search", ddgs)

    sheet = research_agent.build_fact_sheet("Stack", timeout_s=5)

    assert wiki.calls == 1                          # primary consulted
    assert ddgs.calls == 0                          # fallback untouched
    assert set(sheet) == set(FACT_KEYS)             # full declared structure returned
    assert sheet["definition"].startswith("In computer science, a stack")


def test_wikipedia_empty_falls_back_to_ddgs(monkeypatch):
    wiki = _FakeFetcher(_wiki(""))
    ddgs = _FakeFetcher(["A Red-Black tree is a self-balancing binary search tree "
                         "with one extra bit per node."])
    monkeypatch.setattr(research_agent, "_fetch_wikipedia_json", wiki)
    monkeypatch.setattr(research_agent, "_ddgs_search", ddgs)

    sheet = research_agent.build_fact_sheet("Red-Black Tree")

    assert wiki.calls == 1 and ddgs.calls == 1      # tried primary, THEN failed over
    assert sheet["definition"].startswith("A Red-Black tree")


def test_none_wikipedia_payload_also_falls_back(monkeypatch):
    # 'empty' has two representations (None payload, ""-extract); both must trigger ddgs.
    wiki = _FakeFetcher(None)
    ddgs = _FakeFetcher(["An array is a contiguous data structure."])
    monkeypatch.setattr(research_agent, "_fetch_wikipedia_json", wiki)
    monkeypatch.setattr(research_agent, "_ddgs_search", ddgs)

    sheet = research_agent.build_fact_sheet("Array")

    assert ddgs.calls == 1
    assert sheet["definition"].startswith("An array")


def test_both_empty_returns_graceful_sheet(monkeypatch):
    monkeypatch.setattr(research_agent, "_fetch_wikipedia_json", _FakeFetcher(None))
    monkeypatch.setattr(research_agent, "_ddgs_search", _FakeFetcher([]))
    assert research_agent.build_fact_sheet("Some Obscure Topic") == _empty_sheet()


def test_blank_topic_calls_no_fetcher(monkeypatch):
    wiki = _FakeFetcher(None)
    ddgs = _FakeFetcher([])
    monkeypatch.setattr(research_agent, "_fetch_wikipedia_json", wiki)
    monkeypatch.setattr(research_agent, "_ddgs_search", ddgs)

    assert research_agent.build_fact_sheet("   ") == _empty_sheet()
    assert wiki.calls == 0 and ddgs.calls == 0      # nothing to research -> no I/O


def test_timeout_is_propagated_to_both_fetchers(monkeypatch):
    # The 'never hang' guarantee (AGENT.md §4) depends on the budget reaching the fetchers.
    wiki = _FakeFetcher(_wiki(""))                  # empty -> forces the ddgs path too
    ddgs = _FakeFetcher([])
    monkeypatch.setattr(research_agent, "_fetch_wikipedia_json", wiki)
    monkeypatch.setattr(research_agent, "_ddgs_search", ddgs)

    research_agent.build_fact_sheet("Whatever", timeout_s=3.5)

    assert wiki.received_timeout == 3.5
    assert ddgs.received_timeout == 3.5


# --------------------------------------------------------------------------- #
# Payload mapping helpers                                                      #
# --------------------------------------------------------------------------- #

def test_wikipedia_accepts_legacy_dict_payload_shape():
    payload = {"query": {"pages": {1: {"extract": "In computing, an array is a data structure. More."}}}}
    assert _wikipedia_to_sheet("Array", payload)["definition"].startswith("In computing, an array")


def test_wikipedia_missing_extract_is_empty():
    payload = {"query": {"pages": [{"pageid": 2, "title": "X", "extract": ""}]}}
    assert _wikipedia_to_sheet("X", payload)["definition"] == ""


def test_ddgs_to_sheet_takes_first_nonempty_body():
    sheet = _ddgs_to_sheet("Heap", ["", "A binary heap is a complete binary tree. More.", "ignored"])
    assert sheet["definition"].startswith("A binary heap")


# --------------------------------------------------------------------------- #
# _first_sentence — the TTS-safety logic, as a readable edge-case table        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw, expected", [
    # whitespace normalization
    ("  Stack\t is  a LIFO\nstructure.  ", "Stack is a LIFO structure."),
    # period INSIDE parentheses must not cut (now with a real inner period)
    ("See fig (ref. 3) for the layout. Next.", "See fig (ref. 3) for the layout."),
    # inline big-O with parens
    ("Merge sort runs in O(n log n). It then merges.", "Merge sort runs in O(n log n)."),
    # abbreviation "e.g." followed by a lowercase clause must NOT truncate
    ("A hash maps keys to values, e.g. a phone book. It uses hashing.",
     "A hash maps keys to values, e.g. a phone book."),
    # abbreviation "i.e."
    ("A BST is ordered, i.e. left < root < right. Done.",
     "A BST is ordered, i.e. left < root < right."),
    # "Fig." abbreviation
    ("See Fig. 2 for details. Next.", "See Fig. 2 for details."),
    # decimals: dot followed by a digit is not a boundary
    ("It runs in 3.5 seconds on average. Fast.", "It runs in 3.5 seconds on average."),
    # dotted identifier stays intact
    ("Node.js is single-threaded. It scales.", "Node.js is single-threaded."),
    # '?' and '!' are boundaries
    ("What is a stack? A LIFO structure.", "What is a stack?"),
    # no terminal punctuation -> whole (normalized) string
    ("A trie stores strings", "A trie stores strings"),
    # empty / None
    ("", ""),
    (None, ""),
])
def test_first_sentence(raw, expected):
    assert _first_sentence(raw) == expected