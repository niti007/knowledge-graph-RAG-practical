"""
===========================================================================================
router.py — Intelligent Query Router (Graph vs. Vector Retrieval Classifier)
===========================================================================================

PURPOSE:
  This module classifies user queries as either "graph" (structural questions) or 
  "vector" (semantic/recommendation questions) to route them to the appropriate retriever.

WHAT IT DOES:
  1. Uses fast keyword heuristics to classify queries (99% of the time)
  2. Falls back to an LLM (GPT-4o-mini) for ambiguous queries
  3. Returns 'graph' or 'vector' to tell the system which retriever to use

SIMPLE EXAMPLE:
  Query: "Who acted in The Matrix?"
    → Keywords: "who", "acted" → GRAPH keywords
    → Result: "graph" (use Cypher queries)
  
  Query: "Recommend a romantic movie"
    → Keywords: "recommend" → VECTOR keyword
    → Result: "vector" (use semantic similarity search)
  
  Query: "Find me an action movie with Tom Hanks"
    → Keywords: Mix of both (ambiguous)
    → Fall back to LLM → LLM decides → "graph" (it's asking about a specific actor)

===========================================================================================
"""

from __future__ import annotations
from typing import Optional

# Import LangChain's LLM interface
from langchain_openai import ChatOpenAI

# Import LangChain's prompt template system
from langchain_core.prompts import ChatPromptTemplate

# Import configuration to ensure API keys are loaded
import config  # noqa: F401

# Import the OpenAI API key from config
from config import OPENAI_API_KEY

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 1: Keyword Sets for Heuristic Routing
# ─────────────────────────────────────────────────────────────────────────────────────────
# These keywords indicate whether a query is asking for graph or vector retrieval

# Keywords that indicate the user is asking for structured/relational information
# These queries are best answered by traversing the graph (Cypher queries)
GRAPH_KEYWORDS = {
    "who",                  # "Who acted in..."
    "which actor",          # "Which actor appeared..."
    "which director",       # "Which director made..."
    "acted with",           # "Who acted with..."
    "co-star",              # "Co-stars of..."
    "directed by",          # "Directed by..."
    "appeared in",          # "Appeared in..."
    "movies by",            # "Movies by..."
    "films by",             # "Films by..."
    "starred in",           # "Starred in..."
    "worked with",          # "Worked with..."
    "same movie",           # "In the same movie..."
    "co-act",               # "Co-acted..."
    "what actors",          # "What actors..."
    "who directed",         # "Who directed..."
    "director of",          # "Director of..."
    "cast of",              # "Cast of..."
    "who else",             # "Who else..."
    "list",                 # "List all..."
    "how many movies",      # "How many movies..."
    "filmography",          # "Filmography..."
    "what movies",          # "What movies..."
    "what films",           # "What films..."
}

# Keywords that indicate the user is asking for semantic/thematic information
# These queries are best answered by semantic similarity search (FAISS vectors)
VECTOR_KEYWORDS = {
    "about",                # "Movies about..."
    "similar to",           # "Find something similar to..."
    "like",                 # "Find me something like..."
    "recommend",            # "Recommend a movie..."
    "find me",              # "Find me..."
    "suggest",              # "Suggest a movie..."
    "describe",             # "Describe a movie about..."
    "what kind",            # "What kind of movie..."
    "theme",                # "A movie with a theme of..."
    "genre",                # "A romantic genre movie..."
    "feel",                 # "A movie that feels like..."
    "mood",                 # "A sad mood movie..."
    "story about",          # "A story about..."
    "involving",            # "Involving a plot..."
    "related to",           # "Related to..."
    "tell me about",        # "Tell me about movies..."
    "summarize",            # "Summarize a movie about..."
    "based on",             # "Based on a concept..."
    "sounds like",          # "Sounds like..."
    "i want to watch",      # "I want to watch something..."
}

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 2: Keyword-Based Heuristic Router
# ─────────────────────────────────────────────────────────────────────────────────────────
# Fast routing based on keyword matching (runs in < 1ms)

def _keyword_route(query: str) -> Optional[str]:
    """
    PURPOSE:
      Use keyword matching to quickly classify a query as 'graph' or 'vector'.
      This is fast and accurate for most queries.
    
    HOW IT WORKS:
      1. Convert query to lowercase for case-insensitive matching
      2. Count how many GRAPH_KEYWORDS appear in the query
      3. Count how many VECTOR_KEYWORDS appear in the query
      4. Return 'graph' if more graph keywords, 'vector' if more vector keywords
      5. Return None if it's a tie (ambiguous query)
    
    EXAMPLE:
      Query: "Who acted in The Matrix?"
        → Contains: "who" (graph), "acted" (graph)
        → Graph score: 2
        → Vector score: 0
        → Result: "graph"
      
      Query: "Recommend a romantic movie to watch"
        → Contains: "recommend" (vector), "watch" (vector)
        → Graph score: 0
        → Vector score: 2
        → Result: "vector"
      
      Query: "Find movies with Tom Hanks"
        → Contains: "find" (vector), "with" (could be either)
        → Graph score: 1?
        → Vector score: 1?
        → Result: None (ambiguous → use LLM)
    
    RETURNS:
      'graph'  — clearly a graph query
      'vector' — clearly a vector query
      None     — ambiguous, needs LLM decision
    """
    # Convert query to lowercase for case-insensitive matching
    q = query.lower()
    
    # Count how many GRAPH_KEYWORDS appear in the query
    # sum(1 for kw in GRAPH_KEYWORDS if kw in q) means:
    #   - For each keyword in GRAPH_KEYWORDS
    #   - If that keyword appears in the query string
    #   - Add 1 to the count
    graph_score = sum(1 for kw in GRAPH_KEYWORDS if kw in q)
    
    # Count how many VECTOR_KEYWORDS appear in the query
    vector_score = sum(1 for kw in VECTOR_KEYWORDS if kw in q)

    # If graph keywords outnumber vector keywords, it's a graph query
    if graph_score > vector_score:
        return "graph"
    
    # If vector keywords outnumber graph keywords, it's a vector query
    if vector_score > graph_score:
        return "vector"
    
    # If they're tied or both zero, we can't decide → return None (ambiguous)
    return None


# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 3: LLM-Based Fallback Router
# ─────────────────────────────────────────────────────────────────────────────────────────
# When keyword heuristics can't decide, use GPT-4o-mini to classify

# Global variable to cache the LLM instance
# We create it once and reuse it (singleton pattern)
_llm = None

# The prompt template that tells GPT-4o-mini how to classify queries
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
    """
    PURPOSE:
      Use GPT-4o-mini to classify ambiguous queries.
      This is slower than keyword matching but much more accurate for edge cases.
    
    HOW IT WORKS:
      1. Get or create the LLM instance (cached)
      2. Create a chain: prompt template → LLM → output parser
      3. Invoke the chain with the query
      4. Extract the LLM's response ("graph" or "vector")
      5. Return the classification
    
    EXAMPLE:
      Query: "Find action-packed movies with Keanu Reeves"
      
      LLM reads the prompt and query:
        - "Find" (vector keyword)
        - "with Keanu Reeves" (graph: specific actor)
        - Ambiguous!
      
      LLM decides: "This is asking about a specific actor, so it's graph"
      Returns: "graph"
    
    PARAMETERS:
      query: The user's question string
    
    RETURNS:
      "graph" or "vector"
    """
    # Declare we're using the global _llm variable
    global _llm
    
    # Create the LLM on first call
    if _llm is None:
        # ChatOpenAI with model "gpt-4o-mini"
        # temperature=0 means deterministic (always same answer for same query)
        # Higher temperature = more creative/random
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create a chain: prompt template → LLM
    # The | operator pipes the output of one component as input to the next
    chain = _ROUTING_PROMPT | _llm
    
    # Invoke the chain with the user's query
    # This calls GPT-4o-mini with the formatted prompt
    result = chain.invoke({"query": query})
    
    # Extract the text response from the LLM result object
    # result.content is the string response (e.g., "graph")
    answer = result.content.strip().lower()
    
    # Check which keyword is in the answer
    # If "graph" appears anywhere in the response, return "graph"
    # Otherwise return "vector"
    return "graph" if "graph" in answer else "vector"


# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 4: Public API — Main Routing Function
# ─────────────────────────────────────────────────────────────────────────────────────────

def route(query: str) -> str:
    """
    PURPOSE:
      Classify a query as 'graph' or 'vector' using a two-stage approach.
      Stage 1: Fast keyword heuristics
      Stage 2: LLM fallback (if heuristics are inconclusive)
    
    HOW IT WORKS:
      1. Try the keyword heuristic first (fast, < 1ms)
      2. If heuristic returns a decision, use it immediately
      3. If heuristic returns None (ambiguous), call the LLM (slower, ~500ms)
      4. Return the final decision
    
    EXAMPLE FLOWCHART:
      Query in ↓
        → Keyword route?
          → Yes, "graph" → Return "graph" ✓
          → Yes, "vector" → Return "vector" ✓
          → No (None/ambiguous) ↓
            → LLM route
              → Returns "graph" → Return "graph" ✓
              → Returns "vector" → Return "vector" ✓
    
    PARAMETERS:
      query: The user's natural-language question
    
    RETURNS:
      'graph'  — Use Neo4j Cypher retrieval (graph_retriever.py)
      'vector' — Use FAISS semantic search (vector_store.py)
    """
    # Try the fast keyword-based heuristic first
    decision = _keyword_route(query)
    
    # If heuristic made a decision, return it immediately
    if decision:
        return decision
    
    # If heuristic couldn't decide (ambiguous query), use the LLM
    # This is slower but more accurate for edge cases
    return _llm_route(query)


# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 5: Main Entry Point (runs when you execute: python router.py)
# ─────────────────────────────────────────────────────────────────────────────────────────
# Smoke test to verify routing accuracy

if __name__ == "__main__":
    # Test queries with expected classifications
    test_queries = [
        # GRAPH queries (structural/relational questions)
        ("Who acted in The Matrix?",                          "graph"),
        ("What movies did Tom Hanks appear in?",              "graph"),
        ("Who directed Inception?",                           "graph"),
        ("Who has Keanu Reeves worked with?",                 "graph"),
        ("What films did Christopher Nolan direct?",          "graph"),
        
        # VECTOR queries (semantic/recommendation questions)
        ("Recommend a sci-fi movie involving space.",         "vector"),
        ("Find me a movie about survival in extreme conditions.", "vector"),
        ("I want to watch something romantic and emotional.", "vector"),
        ("Suggest a film involving crime and moral dilemmas.","vector"),
        ("Something with a mind-bending twist.",              "vector"),
    ]

    print("Query Routing Test\n" + "=" * 60)
    
    # Track how many we got correct
    correct = 0
    
    # Test each query
    for query, expected in test_queries:
        # Get the routing decision
        actual = route(query)
        
        # Check if it matches expected
        ok = "[OK]" if actual == expected else "[FAIL]"
        if actual == expected:
            correct += 1
        
        # Print the result
        print(f"  {ok} [{actual.upper():6}] {query}")

    # Print accuracy summary
    print(f"\nAccuracy: {correct}/{len(test_queries)}")
