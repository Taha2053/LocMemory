"""Tests for MemoryClassifier (domain detection, concept extraction)."""


def test_list_domains_includes_defaults(classifier):
    domains = classifier.list_domains()
    for expected in ("health", "work", "programming", "learning", "personal", "finance"):
        assert expected in domains


def test_detect_domain_health(classifier):
    domain, conf = classifier.detect_domain("I went to the gym and did a workout")
    assert domain == "health"
    assert 0.0 <= conf <= 1.0


def test_detect_domain_programming(classifier):
    domain, _ = classifier.detect_domain("debugging a python memory leak")
    assert domain == "programming"


def test_detect_domain_work(classifier):
    domain, _ = classifier.detect_domain("client meeting tomorrow about the deadline")
    assert domain == "work"


def test_detect_domain_learning(classifier):
    domain, _ = classifier.detect_domain("reading a book about reinforcement learning")
    assert domain == "learning"


def test_classify_returns_structured_dict(classifier):
    result = classifier.classify("I went running this morning")
    assert "domain" in result
    assert "confidence" in result
    assert "concepts" in result
    assert isinstance(result["concepts"], list)


def test_extract_concepts_filters_stopwords(classifier):
    concepts = classifier.extract_concepts("the quick brown fox jumps over the lazy dog")
    joined = " ".join(concepts).lower()
    assert "the" not in concepts
    assert "over" not in concepts
    assert any(w in joined for w in ("quick", "brown", "fox", "jumps", "lazy", "dog"))


def test_extract_concepts_respects_max(classifier):
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    concepts = classifier.extract_concepts(text, max_concepts=3)
    assert len(concepts) <= 3


def test_get_all_scores_sorted_desc(classifier):
    scores = classifier.get_all_scores("deploying production infrastructure")
    values = list(scores.values())
    assert values == sorted(values, reverse=True)
