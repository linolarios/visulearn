"""Input Gateway tests — taxonomy + a STUBBED provider; no network, no API key (AGENT.md §7).

Proves the normalize_topic seam: deterministic taxonomy/alias resolve (case,
whitespace, punctuation-insensitive), design-pattern category detection, and the
three clean-reject paths (empty, unknown-without-provider, provider-declined)
plus provider acceptance (valid dsa / design_pattern answers, None, bad category,
raised provider errors).
"""
import pytest

from input_gateway import (
    TopicError,
    TopicRejection,
    UnknownTopicError,
    normalize_topic,
)


# --- known-topic taxonomy resolve ------------------------------------------- #

def test_canonical_case_and_punctuation_insensitive():
    assert normalize_topic("red-black tree") == {
        "canonical": "Red-Black Tree", "category": "dsa", "family": "Trees",
    }
    assert normalize_topic("RED BLACK Tree")["canonical"] == "Red-Black Tree"


def test_whitespace_collapsed():
    assert normalize_topic("  merge   sort ")["canonical"] == "Merge Sort"


def test_aliases_resolve_to_canonical():
    assert normalize_topic("bst")["canonical"] == "Binary Search Tree"
    assert normalize_topic("dijkstra's")["canonical"] == "Dijkstra's Algorithm"


def test_design_pattern_category_and_family():
    r = normalize_topic("singleton")
    assert r["category"] == "design_pattern" and r["family"] == "Creational"
    assert normalize_topic("observer")["family"] == "Behavioral"


# --- clean rejects without any provider ------------------------------------- #

def test_empty_topic_rejected():
    with pytest.raises(TopicError):
        normalize_topic("")
    with pytest.raises(TopicError):
        normalize_topic("    ")


def test_unknown_no_provider_rejects():
    with pytest.raises(UnknownTopicError):
        normalize_topic("quantum computing")


# --- provider seam: unknown topic routing ------------------------------------ #

def test_unknown_with_provider_resolves():
    def provider(text):
        return {"canonical": "Fibonacci", "category": "dsa", "family": "Dynamic Programming"}

    assert normalize_topic("fib", provider=provider)["canonical"] == "Fibonacci"


def test_provider_returning_none_rejects():
    with pytest.raises(UnknownTopicError):
        normalize_topic("mystery", provider=lambda t: None)


def test_provider_resolves_design_pattern():
    def provider(text):
        return {"canonical": "Strategy", "category": "design_pattern", "family": "Behavioral"}

    assert normalize_topic("strategy pat", provider=provider)["category"] == "design_pattern"


def test_provider_invalid_category_rejected():
    def provider(text):
        return {"canonical": "X", "category": "quantum", "family": "F"}

    with pytest.raises(TopicRejection):
        normalize_topic("x", provider=provider)


def test_provider_error_rejects_cleanly():
    def provider(text):
        raise RuntimeError("llm down")

    with pytest.raises(UnknownTopicError):
        normalize_topic("x", provider=provider)
