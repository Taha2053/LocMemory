"""
Semantic classification module for memories.

Uses sentence-transformers to detect domains and extract concepts.
Supports dynamic domain/sub-domain creation via LLM.
"""

import json
import re
import requests
from pathlib import Path
from typing import Optional

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral:7b-instruct"

DEFAULT_DOMAINS = {
    "health": [
        "health, fitness, wellness, exercise, diet, nutrition, medical, healthcare, gym, body",
        "I went to the gym today and did a workout",
        "Feeling tired, need more sleep",
    ],
    "engineering": [
        "engineering, design, architecture, system design, technical specifications, infrastructure",
        "Building a new system architecture for the project",
        "Mechanical design and blueprints",
    ],
    "programming": [
        "software development, coding, algorithms, debugging, programming, code, software, development",
        "Writing Python code for the new feature",
        "Debugging a tricky memory leak",
    ],
    "work": [
        "work, job, office, meeting, deadline, project, client, task, business, career",
        "Meeting with the team tomorrow at 10am",
        "Need to finish this task by Friday",
    ],
    "personal": [
        "personal, family, friends, home, hobby, vacation, weekend, life, home, relationships",
        "Spent the weekend with family",
        "My dog needs a walk",
    ],
    "finance": [
        "finance, money, investment, savings, budget, income, expense, banking, stock, trading",
        "Invested in index funds for retirement",
        "Budget analysis for this month",
    ],
    "learning": [
        "learning, study, studying, course, lecture, reading, book, learn, progress, chapter, exam, revision, notes, class, subject, homework, tutorial, research, education, practice, training, knowledge, assignment, quiz, test, university, school, college, academic",
        "Studying machine learning algorithms",
        "Read a book about reinforcement learning",
        "How is my studying going",
        "What am I learning lately",
        "How is my course going",
        "I need to revise my notes for the exam",
        "Making progress on my reading assignments",
        "Attended an interesting lecture today",
    ],
}

DEFAULT_SUBDOMAINS = {
    "programming": ["web-development", "mobile-development", "data-science", "devops", "security"],
    "work": ["management", "consulting", "freelance", "remote-work", "on-site"],
    "health": ["fitness", "nutrition", "mental-health", "medical", "sleep"],
    "finance": ["investing", "budgeting", "taxes", "real-estate", "crypto"],
    "learning": ["courses", "books", "tutorials", "practice", "research"],
}

COMMON_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "and", "but", "or", "because", "as", "until", "while", "although",
    "this", "that", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "it", "they", "them", "their", "what",
    "which", "who", "whom", "about", "also", "like", "get", "got", "today",
    "yesterday", "tomorrow", "last", "next", "really", "thing", "things",
}


class MemoryClassifier:
    """Semantic classifier for memory text using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        ollama_url: str = OLLAMA_URL,
        ollama_model: str = DEFAULT_MODEL,
        use_fallback: bool = True,
        confidence_threshold: float = 0.3,
        domains_file: str = "data/domains.json",
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.use_fallback = use_fallback
        self.confidence_threshold = confidence_threshold
        self.domains_file = Path(domains_file)

        self._model = self._load_model()
        self._domains = {}
        self._subdomains = {}
        self._domain_vectors = {}
        self._subdomain_vectors = {}

        self._load_domains()
        self._embed_all()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name)
            print(f"Loaded embedding model: {self.model_name}")
            return model
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _load_domains(self):
        if self.domains_file.exists():
            data = json.loads(self.domains_file.read_text())
            self._domains = data.get("domains", DEFAULT_DOMAINS)
            self._subdomains = data.get("subdomains", DEFAULT_SUBDOMAINS)
            print(f"Loaded domains from {self.domains_file}")
        else:
            self._domains = dict(DEFAULT_DOMAINS)
            self._subdomains = dict(DEFAULT_SUBDOMAINS)
            self._save_domains()
            print("Initialized with default domains")

    def _save_domains(self):
        self.domains_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "domains": self._domains,
            "subdomains": self._subdomains,
        }
        self.domains_file.write_text(json.dumps(data, indent=2))

    def _embed_all(self):
        self._domain_vectors = {}
        for domain, prototypes in self._domains.items():
            self._domain_vectors[domain] = self._embed(prototypes)

        self._subdomain_vectors = {}
        for domain, subdomains in self._subdomains.items():
            if subdomains:
                self._subdomain_vectors[domain] = self._embed(subdomains)

    def add_domain(self, name: str, prototypes: list[str]):
        """Add a new domain with prototype sentences."""
        self._domains[name] = prototypes
        self._domain_vectors[name] = self._embed(prototypes)
        if name not in self._subdomains:
            self._subdomains[name] = []
        self._save_domains()
        print(f"Added domain: {name}")

    def add_subdomain(self, domain: str, name: str):
        """Add a sub-domain to an existing domain."""
        if domain not in self._domains:
            raise ValueError(f"Domain '{domain}' does not exist")

        if name not in self._subdomains[domain]:
            self._subdomains[domain].append(name)
            self._save_domains()
            print(f"Added sub-domain '{name}' to '{domain}'")

    def detect_domain(self, text: str) -> tuple[str, float]:
        """
        Detect the best matching domain for the input text.

        Returns (domain, confidence) tuple.
        """
        text_embedding = self._embed([text])[0]

        scores = {}
        for domain, prototypes in self._domain_vectors.items():
            similarities = [
                self._cosine_similarity(text_embedding, proto_vec)
                for proto_vec in prototypes
            ]
            scores[domain] = max(similarities)

        best_domain = max(scores, key=scores.get)
        confidence = scores[best_domain]

        if confidence < self.confidence_threshold and self.use_fallback:
            refined = self._ollama_suggest_domain(text, scores)
            if refined:
                return refined, confidence

        return best_domain, confidence

    def detect_subdomain(self, text: str, parent_domain: str) -> tuple[str, float]:
        """
        Detect the best matching sub-domain for the input text within a parent domain.

        Returns (subdomain, confidence) tuple.
        """
        if parent_domain not in self._subdomain_vectors:
            return "", 0.0

        subdomains = self._subdomains.get(parent_domain, [])

        if not subdomains:
            return "", 0.0

        text_embedding = self._embed([text])[0]
        subdomain_embeddings = self._subdomain_vectors[parent_domain]

        scores = {}
        for i, sub in enumerate(subdomains):
            if i < len(subdomain_embeddings):
                scores[sub] = self._cosine_similarity(text_embedding, subdomain_embeddings[i])

        if not scores:
            return "", 0.0

        best_subdomain = max(scores, key=scores.get)
        confidence = scores[best_subdomain]

        return best_subdomain, confidence

    def extract_concepts(self, text: str, max_concepts: int = 5) -> list[str]:
        """
        Extract simple noun phrases or keywords from the text.

        Returns list of unique concepts.
        """
        text_lower = text.lower()

        words = re.findall(r"\b[a-zA-Z]{3,}\b", text_lower)
        filtered = [w for w in words if w not in COMMON_WORDS]

        word_freq: dict[str, int] = {}
        for word in filtered:
            word_freq[word] = word_freq.get(word, 0) + 1

        bigrams = re.findall(r"\b([a-zA-Z]{3,}\s+[a-zA-Z]{3,})\b", text_lower)
        bigram_set = {
            " ".join(w for w in bg.split() if w not in COMMON_WORDS): True
            for bg in bigrams
        }
        bigram_set = {k: True for k in bigram_set if len([w for w in k.split() if w]) >= 2}

        concepts = []

        for bigram, _ in bigram_set.items():
            if bigram not in " ".join(concepts).lower():
                concepts.append(bigram)

        for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
            if len(concepts) >= max_concepts:
                break
            if word not in " ".join(concepts).lower():
                concepts.append(word)

        return concepts[:max_concepts]

    def classify(self, text: str, include_subdomain: bool = True) -> dict:
        """
        Classify text and extract concepts.

        Returns: {"domain": "...", "subdomain": "...", "concepts": [...], "confidence": float}
        """
        domain, confidence = self.detect_domain(text)

        result = {
            "domain": domain,
            "confidence": round(confidence, 4),
            "concepts": self.extract_concepts(text),
        }

        if include_subdomain:
            subdomain, sub_confidence = self.detect_subdomain(text, domain)
            if subdomain:
                result["subdomain"] = subdomain
                result["subdomain_confidence"] = round(sub_confidence, 4)

        return result

    def _ollama_suggest_domain(self, text: str, scores: dict[str, float]) -> Optional[str]:
        """
        Use Ollama to suggest a domain when confidence is low,
        or create a new domain if needed.
        """
        domains_str = ", ".join(self._domains.keys())
        prompt = f"""Analyze this text and classify it into exactly one domain from the list, or suggest a NEW domain if it doesn't fit.

Available domains: {domains_str}

Text: "{text}"

Current top scores: {', '.join(f'{d}: {s:.2f}' for d, s in sorted(scores.items(), key=lambda x: -x[1])[:3])}

Respond in this exact format:
DOMAIN: <domain_name>
OR
NEW_DOMAIN: <new_domain_name>
PROTOTYPES: <3 prototype sentences describing this domain>

Respond with ONLY this format, nothing else."""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                timeout=30,
            )
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                return self._parse_llm_response(result, text)
        except Exception as e:
            print(f"Ollama fallback failed: {e}")

        return None

    def _parse_llm_response(self, response: str, original_text: str) -> Optional[str]:
        """Parse LLM response to extract domain suggestion."""
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip().upper()
            if line.startswith("DOMAIN:"):
                domain = line.split(":", 1)[1].strip()
                if domain in self._domains:
                    return domain
            elif line.startswith("NEW_DOMAIN:"):
                new_domain = line.split(":", 1)[1].strip().lower()
                prototypes_line = ""
                for l in lines:
                    if l.strip().upper().startswith("PROTOTYPES:"):
                        prototypes_line = l.split(":", 1)[1].strip()

                if prototypes_line:
                    prototypes = [p.strip() for p in prototypes_line.split("|")]
                    if len(prototypes) >= 2:
                        prototypes.append(original_text)
                    self.add_domain(new_domain, prototypes)
                    return new_domain
                else:
                    self.add_domain(new_domain, [
                        original_text,
                        f"Related to {new_domain}",
                        f"{new_domain} activities and topics",
                    ])
                    return new_domain

        return None

    def get_all_scores(self, text: str) -> dict[str, float]:
        """
        Get similarity scores for all domains.

        Returns dict of domain -> score sorted by score descending.
        """
        text_embedding = self._embed([text])[0]

        scores = {}
        for domain, prototypes in self._domain_vectors.items():
            similarities = [
                self._cosine_similarity(text_embedding, proto_vec)
                for proto_vec in prototypes
            ]
            scores[domain] = round(max(similarities), 4)

        return dict(sorted(scores.items(), key=lambda x: -x[1]))

    def list_domains(self) -> list[str]:
        """Return list of all available domains."""
        return list(self._domains.keys())

    def list_subdomains(self, domain: str) -> list[str]:
        """Return list of sub-domains for a given domain."""
        return self._subdomains.get(domain, [])


if __name__ == "__main__":
    classifier = MemoryClassifier(use_fallback=False)

    print("Available domains:", classifier.list_domains())
    print()

    test_texts = [
        "I learned about neural networks and transformers today",
        "Meeting with the client about the new project requirements",
        "Went to the gym and did a cardio workout",
        "Fixed a tricky memory leak in the Python application",
        "Spent quality time with my family this weekend",
        "Analyzed stock market trends and updated my portfolio",
    ]

    print("=" * 60)
    for text in test_texts:
        result = classifier.classify(text)
        print(f"\nText: {text}")
        print(f"  Domain: {result['domain']} (confidence: {result['confidence']})")
        if "subdomain" in result:
            print(f"  Sub-domain: {result['subdomain']} (confidence: {result['subdomain_confidence']})")
        print(f"  Concepts: {result['concepts']}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Adding a new domain via LLM simulation:")
    print("Use classify with use_fallback=True and low confidence text")
    print("to trigger LLM-driven domain creation.")