"""
Retrieval Necessity Heuristic for LocMemory.

Determines whether a user message requires graph-based memory retrieval
or can be answered directly by the LLM without accessing the memory graph.

This optimization reduces latency for purely factual or computational queries
while ensuring personal/contextual queries get the full memory context.
"""

import re
from typing import Optional

from core.settings.config import get_config


PERSONAL_PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
}

POSSESSIVE_PRONOUNS = {
    "my", "mine", "our", "ours", "your", "yours",
    "his", "her", "hers", "its", "their", "theirs",
}

REFLEXIVE_PRONOUNS = {
    "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "themselves",
}

HISTORY_PATTERNS = [
    r"\bwhat (do|did|have|has|had).*(i|my|me|we|our)\b",
    r"\bwhen (did|do|have|has).*(i|my|me|we|our)\b",
    r"\bwhere (did|do|have|has).*(i|my|me|we|our)\b",
    r"\bwhy (did|do|have|has).*(i|my|me|we|our)\b",
    r"\bhow (did|do|have|has).*(i|my|me|we|our)\b",
    r"\btell me about.*my\b",
    r"\bmy.*(history|history|background|past|previous)\b",
    r"\bwhat.*(remember|know about|learned|studied|worked on)\b",
    r"\bwho am i\b",
    r"\bwhat am i\b",
    r"\bdo i (have|know|remember|like|prefer)\b",
    r"\bdid i (have|know|remember|like|prefer|do)\b",
    r"\bhave i (ever|told|said|mentioned|learned)\b",
    r"\b(remember|know).*(me|i|my)\b",
    r"\babout (me|my|i)\b",
    r"\bfor (me|my|i)\b",
]

GENERAL_PATTERNS = [
    r"^(what is|what's|what are) (the|a|an|that|this)",
    r"^(who is|who's|who are) (the|a|an|that|this)",
    r"^(when is|when's|when did)",
    r"^(where is|where's|where did)",
    r"^(why is|why's|why does|why did)",
    r"^(how (does|do|is|to))",
    r"^(explain|describe|define|tell me about) (the|a|an)",
    r"^(calculate|compute|solve) ",
    r"^(convert|translate) ",
    r"^(what time|what date|what year)",
    r"^(what country|what city|what language)",
    r"^(list|give me).*(facts?|examples?|steps?|instructions?)",
    r"^(what are|what is).*(rules?|guidelines?|principles?)",
    r"^(who|what|where|when|why|how)\s+does\s+(a|an|the|one)\b",
    r"^(write|create|generate).*(code|program|function|class|script)",
    r"^(translate|convert).*(to|from)\b",
    r"^(solve|calculate|compute|evaluate)",
    r"^(what is|what's) (\d+|one|two|three|four|five|six|seven|eight|nine|ten)",
]

QUESTION_STARTERS = {
    "what", "what's", "when", "where", "why", "how", "who", "which",
    "does", "do", "did", "is", "are", "was", "were", "can", "could",
    "should", "would", "will", "has", "have", "had", "may", "might",
}


def _extract_words(text: str) -> set[str]:
    """Extract lowercase words from text."""
    return set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))


def _check_personal_pronouns(text: str) -> tuple[bool, Optional[str]]:
    """
    Check if text contains personal pronouns indicating self-referential content.
    
    Returns: (requires_retrieval, reason)
    """
    words = _extract_words(text)
    
    has_reflexive = bool(words & REFLEXIVE_PRONOUNS)
    has_possessive = bool(words & POSSESSIVE_PRONOUNS)
    has_personal = bool(words & PERSONAL_PRONOUNS)
    
    if has_reflexive:
        return True, "reflexive pronoun"
    if has_possessive and has_personal:
        return True, "possessive pronoun"
    if has_personal and any(c in text for c in '?.'):
        return True, "personal pronoun in question"
    
    return False, None


def _check_history_patterns(text: str) -> tuple[bool, Optional[str]]:
    """Check if text asks about user's personal history or state."""
    text_lower = text.lower()
    
    for pattern in HISTORY_PATTERNS:
        if re.search(pattern, text_lower):
            return True, f"history pattern: {pattern}"
    
    return False, None


def _check_general_query_patterns(text: str) -> tuple[bool, Optional[str]]:
    """Check if text is a general/factual query that doesn't need retrieval."""
    text_lower = text.lower()
    first_word = text_lower.split()[0] if text_lower.split() else ""
    
    for pattern in GENERAL_PATTERNS:
        if re.search(pattern, text_lower):
            return True, f"general pattern: {pattern}"
    
    if first_word in QUESTION_STARTERS:
        for pattern in GENERAL_PATTERNS:
            if re.search(pattern, text_lower):
                return True, f"question pattern: {pattern}"
    
    return False, None


def _check_conditional_retrieval(text: str) -> bool:
    """
    Check if query uses conditional or comparison language that benefits from retrieval.
    """
    conditional_keywords = {
        "compare", "comparison", "versus", "vs", "instead", "rather",
        "alternatively", "other than", "differ from", "difference",
        "better", "worse", "best", "worst", "improve", "improve upon",
    }
    
    words = _extract_words(text)
    return bool(words & conditional_keywords)


class RetrievalNecessityHeuristic:
    """
    Determines if a query requires memory graph retrieval.
    """
    
    def __init__(self, config: Optional[dict] = None):
        self._config = config or get_config()
        
        retrieval_section = self._config.get_section("retrieval")
        
        self.enabled = retrieval_section.get("necessity_heuristic", True)
        self.min_confidence = retrieval_section.get("necessity_min_confidence", 0.5)
        
        self._load_excluded_patterns()
        self._load_required_patterns()
    
    def _load_excluded_patterns(self):
        """Load patterns that should skip retrieval."""
        retrieval_section = self._config.get_section("retrieval")
        self.excluded_patterns = retrieval_section.get("exclude_patterns", [
            r"^(hello|hi|hey|good morning|good afternoon|good evening)",
            r"^(thank|thanks|thank you)",
            r"^(ok|okay|yes|no|sure|yep|nope)",
            r"^(exit|quit|bye|goodbye)",
        ])
    
    def _load_required_patterns(self):
        """Load patterns that always require retrieval."""
        retrieval_section = self._config.get_section("retrieval")
        self.required_patterns = retrieval_section.get("require_patterns", [])
    
    def should_retrieve(self, text: str) -> tuple[bool, str]:
        """
        Determine if the query requires memory retrieval.
        
        Args:
            text: The user input text
            
        Returns:
            (requires_retrieval: bool, reason: str)
        """
        if not self.enabled:
            return True, "heuristic disabled"
        
        text = text.strip()
        if not text:
            return False, "empty input"
        
        text_lower = text.lower()
        
        for pattern in self.excluded_patterns:
            if re.search(pattern, text_lower):
                return False, f"excluded pattern: {pattern}"
        
        for pattern in self.required_patterns:
            if re.search(pattern, text_lower):
                return True, f"required pattern: {pattern}"
        
        requires_pronoun, pronoun_reason = _check_personal_pronouns(text)
        if requires_pronoun:
            return True, pronoun_reason
        
        requires_history, history_reason = _check_history_patterns(text)
        if requires_history:
            return True, history_reason
        
        is_general, general_reason = _check_general_query_patterns(text)
        if is_general:
            return False, general_reason
        
        if _check_conditional_retrieval(text):
            return True, "conditional comparison"
        
        first_word = text_lower.split()[0] if text_lower.split() else ""
        if first_word in {"what", "who", "where", "when", "why", "how"}:
            words = _extract_words(text)
            if not (words & PERSONAL_PRONOUNS):
                return False, "generic question without personal reference"
        
        return True, "default: retrieve"


def should_retrieve(text: str) -> bool:
    """
    Convenience function to check if a query requires retrieval.
    
    Args:
        text: The user input text
        
    Returns:
        True if retrieval is recommended, False otherwise
    """
    heuristic = RetrievalNecessityHeuristic()
    requires, _ = heuristic.should_retrieve(text)
    return requires


if __name__ == "__main__":
    test_queries = [
        "Hello, how are you?",
        "What is Python?",
        "What did I learn about yesterday?",
        "Calculate 2 + 2",
        "My favorite color is blue, what's yours?",
        "Tell me about machine learning",
        "Do I have any meetings today?",
        "Remember when we discussed the project?",
        "Thanks for the help!",
        "What are the rules of Python?",
        "I love coding in Rust",
        "Where did I work before?",
        "Convert 100 USD to EUR",
        "compare Python and JavaScript",
        "What time is it?",
    ]
    
    heuristic = RetrievalNecessityHeuristic()
    
    print("=" * 70)
    print("Retrieval Necessity Heuristic Test")
    print("=" * 70)
    
    for query in test_queries:
        requires, reason = heuristic.should_retrieve(query)
        status = "RETRIEVE" if requires else "SKIP"
        print(f"\n[{status:6}] {query}")
        print(f"         Reason: {reason}")