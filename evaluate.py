"""
evaluate.py — Benchmark suite for the Hybrid Graph RAG system.

Runs 10 questions (5 graph + 5 vector), compares:
  - Hybrid RAG answer (routed automatically)
  - Forced alternative retriever answer

Reports routing accuracy and a side-by-side comparison.

Run:
    python evaluate.py
"""

import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config  # noqa: F401
from rag_chain import run_hybrid_rag
from graph_retriever import graph_retrieve
from vector_store import similarity_search, get_vector_store

# ---------------------------------------------------------------------------
# Shared LLM for forced-retriever answers
# ---------------------------------------------------------------------------

_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "Use the following context to answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

_answer_chain = _ANSWER_PROMPT | _LLM | StrOutputParser()


def _answer_with_retriever(question: str, retriever_type: str) -> str:
    """Force a specific retriever and generate an answer."""
    if retriever_type == "graph":
        context = graph_retrieve(question)
    else:
        docs    = similarity_search(question, k=4)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return _answer_chain.invoke({"context": context, "question": question})


# ---------------------------------------------------------------------------
# Benchmark questions
# ---------------------------------------------------------------------------

GRAPH_QUESTIONS = [
    {
        "question": "Who acted in The Matrix?",
        "expected": "graph",
        "note":     "Direct entity -> cast lookup",
    },
    {
        "question": "What movies has Tom Hanks appeared in?",
        "expected": "graph",
        "note":     "Actor filmography",
    },
    {
        "question": "Who directed Inception?",
        "expected": "graph",
        "note":     "Director lookup",
    },
    {
        "question": "Who has Keanu Reeves acted with?",
        "expected": "graph",
        "note":     "Multi-hop: co-actors",
    },
    {
        "question": "What films did Christopher Nolan direct?",
        "expected": "graph",
        "note":     "Director filmography",
    },
]

VECTOR_QUESTIONS = [
    {
        "question": "Recommend a sci-fi movie involving space or advanced technology.",
        "expected": "vector",
        "note":     "Semantic genre search",
    },
    {
        "question": "Find me a movie about survival in extreme isolation.",
        "expected": "vector",
        "note":     "Thematic search",
    },
    {
        "question": "I want to watch something romantic and emotional.",
        "expected": "vector",
        "note":     "Mood-based search",
    },
    {
        "question": "Tell me about a film involving crime and moral dilemmas.",
        "expected": "vector",
        "note":     "Abstract theme search",
    },
    {
        "question": "Suggest a mind-bending movie with an unusual narrative structure.",
        "expected": "vector",
        "note":     "Concept-based search",
    },
]

ALL_QUESTIONS = GRAPH_QUESTIONS + VECTOR_QUESTIONS


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation() -> list[dict]:
    """
    Execute the full benchmark and print a detailed comparison report.
    Returns the list of result dicts for programmatic use.
    """
    sep = "=" * 72

    print(sep)
    print("  HYBRID GRAPH RAG — EVALUATION SUITE")
    print(sep)
    print(f"  Graph questions : {len(GRAPH_QUESTIONS)}")
    print(f"  Vector questions: {len(VECTOR_QUESTIONS)}")
    print(f"  Total           : {len(ALL_QUESTIONS)}\n")

    # Pre-warm vector store so it doesn't count against first question timing
    print("  Pre-warming vector store ...")
    get_vector_store()
    print("  [OK] Ready.\n")
    print(sep)

    results      = []
    correct_count = 0

    for idx, item in enumerate(ALL_QUESTIONS, start=1):
        question = item["question"]
        expected = item["expected"]

        print(f"\n[{idx:02d}/{len(ALL_QUESTIONS)}] {question}")
        print(f"  Note          : {item['note']}")
        print(f"  Expected      : {expected.upper()}")

        # Run hybrid RAG (auto-routed)
        t0            = time.time()
        hybrid_result = run_hybrid_rag(question, verbose=False)
        elapsed       = time.time() - t0

        actual  = hybrid_result["retriever"]
        correct = actual == expected
        if correct:
            correct_count += 1

        tick = "[OK]" if correct else "[FAIL]"
        print(f"  Routed to     : {actual.upper()}  {tick}")
        print(f"  Time          : {elapsed:.2f}s")

        # Forced alternative for comparison
        alt = "vector" if actual == "graph" else "graph"
        alt_answer = _answer_with_retriever(question, alt)

        # Truncate answers for display
        def trunc(s: str, n: int = 140) -> str:
            return s[:n] + " …" if len(s) > n else s

        print(f"\n  HYBRID answer ({actual.upper()}):")
        print(f"    {trunc(hybrid_result['answer'])}")
        print(f"\n  ALT answer    ({alt.upper()}):")
        print(f"    {trunc(alt_answer)}")
        print()

        results.append({
            "question":      question,
            "expected":      expected,
            "actual":        actual,
            "correct":       correct,
            "hybrid_answer": hybrid_result["answer"],
            "alt_retriever": alt,
            "alt_answer":    alt_answer,
            "elapsed":       elapsed,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    accuracy = correct_count / len(ALL_QUESTIONS) * 100
    graph_ok  = sum(1 for r in results if r["expected"] == "graph"  and r["correct"])
    vector_ok = sum(1 for r in results if r["expected"] == "vector" and r["correct"])
    avg_time  = sum(r["elapsed"] for r in results) / len(results)

    print(sep)
    print("  ROUTING ACCURACY SUMMARY")
    print(sep)
    print(f"  Overall accuracy   : {correct_count}/{len(ALL_QUESTIONS)}  ({accuracy:.1f}%)")
    print(f"  Graph  questions   : {graph_ok}/{len(GRAPH_QUESTIONS)} correct")
    print(f"  Vector questions   : {vector_ok}/{len(VECTOR_QUESTIONS)} correct")
    print(f"  Avg response time  : {avg_time:.2f}s per query")
    print(sep)

    # Incorrect routes table
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\n  Incorrectly routed ({len(wrong)}):")
        for r in wrong:
            print(f"    [FAIL] Expected {r['expected'].upper()} -> got {r['actual'].upper()} | {r['question']}")
    else:
        print("\n  All queries routed correctly! (Success)")
    print()

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
