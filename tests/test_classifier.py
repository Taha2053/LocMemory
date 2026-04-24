"""
Tests for domain classification.
"""

import pytest

from core.memory.classifier import MemoryClassifier


def test_health_keywords(classifier_with_mock):
    """Health keywords should classify as health."""
    result = classifier_with_mock.classify("went to the gym today")
    assert result.get("domain") == "health"


def test_engineering_keywords(classifier_with_mock):
    """Engineering keywords should classify as engineering."""
    result = classifier_with_mock.classify("debugging my pytorch model")
    assert result.get("domain") == "engineering"


def test_work_keywords(classifier_with_mock):
    """Work keywords should classify as work."""
    result = classifier_with_mock.classify("client meeting tomorrow")
    assert result.get("domain") == "work"


def test_learning_keywords(classifier_with_mock):
    """Learning keywords should classify as learning."""
    result = classifier_with_mock.classify("reading a book about RL")
    assert result.get("domain") == "learning"


def test_needs_memory_personal(classifier_with_mock):
    """Personal query with pronoun should need memory."""
    # "how am I doing" - personal
    result = classifier_with_mock.needs_memory("how am I doing")
    assert result is True


def test_needs_memory_factual(classifier_with_mock):
    """Factual query should not need memory."""
    # "what is the capital of France" - factual
    result = classifier_with_mock.needs_memory("what is the capital of France")
    assert result is False


def test_needs_memory_with_pronoun(classifier_with_mock):
    """Query with pronoun should need memory."""
    result = classifier_with_mock.needs_memory("my project is struggling")
    assert result is True


def test_concept_detection_no_match(classifier_with_mock):
    """Empty concepts should return None."""
    concepts = classifier_with_mock.extract_concepts("asdfjkl;")
    assert len(concepts) == 0 or concepts is None


def test_detect_domain_health(classifier_with_mock):
    """Direct domain detection for health."""
    domain, score = classifier_with_mock.detect_domain("exercise and fitness")
    assert domain == "health"


def test_detect_domain_work(classifier_with_mock):
    """Direct domain detection for work."""
    domain, score = classifier_with_mock.detect_domain("deadline and meeting")
    assert domain == "work"