"""
main.py — CLI entry point for the Hybrid Graph RAG system.

Usage
-----
  python main.py           Interactive query mode
  python main.py --load    (Re)load the movie dataset into Neo4j
  python main.py --eval    Run the full evaluation / benchmark suite
  python main.py --demo    Run 5 pre-set demo queries and exit
"""

import sys
from rag_chain import run_hybrid_rag

BANNER = r"""
  +==========================================================+
  |        HYBRID GRAPH RAG -- MOVIE ASSISTANT               |
  |   Neo4j Graph Traversal  +  FAISS Vector Search          |
  +==========================================================+
"""

HELP_TEXT = """
  Tip -- try queries like:
    * Who acted in The Matrix?              (graph -> Cypher)
    * What movies did Keanu Reeves star in? (graph -> Cypher)
    * Who has Tom Hanks worked with?        (graph -> multi-hop)
    * Recommend a sci-fi movie              (vector -> FAISS)
    * Find me something romantic to watch   (vector -> FAISS)
    * A film about survival in the wild     (vector -> FAISS)

  Commands: quit | exit | help
"""


def _interactive():
    """REPL-style interactive query loop."""
    print(BANNER)
    print(HELP_TEXT)
    print("  " + "-" * 58)

    while True:
        try:
            raw = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not raw:
            continue

        if raw.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        if raw.lower() in ("help", "?"):
            print(HELP_TEXT)
            continue

        print()
        result = run_hybrid_rag(raw, verbose=True)
        print(f"\n  Answer : {result['answer']}")
        print(f"  Source : {result['retriever'].upper()} retrieval")
        print("  " + "-" * 58)


def _demo():
    """Run 5 pre-set queries to showcase both retrieval paths."""
    from rag_chain import run_hybrid_rag

    queries = [
        "Who acted in The Matrix?",
        "What movies did Christopher Nolan direct?",
        "Who has Tom Hanks worked with across his career?",
        "Recommend a romantic movie to watch tonight.",
        "Find me a sci-fi film about dreams or alternate realities.",
    ]

    print(BANNER)
    print("  DEMO MODE — running 5 sample queries\n")
    print("  " + "-" * 58)

    for q in queries:
        print(f"\n  Q: {q}")
        result = run_hybrid_rag(q, verbose=True)
        print(f"\n  A: {result['answer']}")
        print("  " + "-" * 58)


def _load():
    """(Re)load the movie dataset into Neo4j."""
    from graph_loader import get_driver, clear_database, load_movies, verify_data

    print("  Loading movie dataset into Neo4j ...")
    driver = get_driver()
    driver.verify_connectivity()
    print("  [OK] Connected to Neo4j.\n")
    clear_database(driver)
    load_movies(driver)
    print("\n  Graph statistics:")
    verify_data(driver)
    driver.close()
    print("\n  [OK] Data loaded successfully!")


def _eval():
    """Run the full evaluation benchmark suite."""
    from evaluate import run_evaluation
    run_evaluation()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if "--load" in args:
        _load()
    elif "--eval" in args:
        _eval()
    elif "--demo" in args:
        _demo()
    else:
        _interactive()


if __name__ == "__main__":
    main()
