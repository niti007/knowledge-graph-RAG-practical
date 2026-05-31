"""
router.py — Query router that classifies a question as 'graph' or 'vector'.

Strategy:
  1. Fast keyword-heuristic pass → returns immediately if unambiguous.
  2. LLM fallback (GPT-4o-mini) for borderline / mixed queries.

Run directly to test routing decisions:
    python router.py
"""

from __future__ import annotations
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import config  # noqa: F401
from config import OPENAI_API_KEY

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

GRAPH_KEYWORDS = {
    "who", "which actor", "which director", "acted with", "co-star",
    "directed by", "appeared in", "movies by", "films by", "starred in",
    "worked with", "same movie", "co-act", "what actors", "who directed",
    "director of", "cast of", "who else", "list", "how many movies",
    "filmography", "what movies", "what films",
}

VECTOR_KEYWORDS = {
    "about", "similar to", "like", "recommend", "find me", "suggest",
    "describe", "what kind", "theme", "genre", "feel", "mood",
    "story about", "involving", "related to", "tell me about", "summarize",
    "involving", "based on", "sounds like", "i want to watch",
}

# ---------------------------------------------------------------------------
# Keyword-based heuristic
# ---------------------------------------------------------------------------


def _keyword_route(query: str) -> Optional[str]:
    """
    Returns 'graph' or 'vector' if keywords strongly indicate one path,
    or None if the query is ambiguous.
    """
    q = query.lower()
    graph_score  = sum(1 for kw in GRAPH_KEYWORDS  if kw in q)
    vector_score = sum(1 for kw in VECTOR_KEYWORDS if kw in q)

    if graph_score > vector_score:
        return "graph"
    if vector_score > graph_score:
        return "vector"
    return None  # ambiguous


# ---------------------------------------------------------------------------
# LLM-based fallback
# ---------------------------------------------------------------------------

_llm = None

_ROUTING_PROMPT = ChatPromptTemplate.from_template(
    """You are a query routing assistant for a movie knowledge system.

Classify the user's question into exactly one of two categories:

  "graph"  — The question asks about specific entities, relationships, or
             structured facts (e.g. "Who acted with Tom Hanks?",
             "What did Christopher Nolan direct?", "Who is in Inception?").

  "vector" — The question is semantic, thematic, or recommendation-based
             (e.g. "Recommend a sci-fi movie", "Find something romantic",
             "Movies about survival", "Something like The Matrix").

Question: {query}

Respond with ONLY the single word "graph" or "vector" — nothing else.
"""
)


def _llm_route(query: str) -> str:
    """Call GPT-4o-mini to classify the query when heuristics are inconclusive."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain  = _ROUTING_PROMPT | _llm
    result = chain.invoke({"query": query})
    answer = result.content.strip().lower()
    return "graph" if "graph" in answer else "vector"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route(query: str) -> str:
    """
    Classify a query as 'graph' or 'vector'.

    Uses keyword heuristics for speed; falls back to an LLM call
    when the query is ambiguous.

    Returns:
        'graph'  — use Neo4j Cypher retrieval
        'vector' — use FAISS semantic similarity search
    """
    decision = _keyword_route(query)
    if decision:
        return decision
    # Ambiguous — let the LLM decide
    return _llm_route(query)


# ---------------------------------------------------------------------------
# Entry point — routing demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        # Expected: graph
        ("Who acted in The Matrix?",                          "graph"),
        ("What movies did Tom Hanks appear in?",              "graph"),
        ("Who directed Inception?",                           "graph"),
        ("Who has Keanu Reeves worked with?",                 "graph"),
        ("What films did Christopher Nolan direct?",          "graph"),
        # Expected: vector
        ("Recommend a sci-fi movie involving space.",         "vector"),
        ("Find me a movie about survival in extreme conditions.", "vector"),
        ("I want to watch something romantic and emotional.", "vector"),
        ("Suggest a film involving crime and moral dilemmas.","vector"),
        ("Something with a mind-bending twist.",              "vector"),
    ]

    print("Query Routing Test\n" + "=" * 60)
    correct = 0
    for query, expected in test_queries:
        actual = route(query)
        ok     = "[OK]" if actual == expected else "[FAIL]"
        if actual == expected:
            correct += 1
        print(f"  {ok} [{actual.upper():6}] {query}")

    print(f"\nAccuracy: {correct}/{len(test_queries)}")
