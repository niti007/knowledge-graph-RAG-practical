"""
rag_chain.py — Hybrid RAG orchestration pipeline.

Flow:
  1. Router classifies the query → 'graph' or 'vector'
  2. The appropriate retriever fetches context
  3. GPT-4o-mini generates a grounded answer from the context
  4. Returns a result dict with query / retriever / context / answer

Run directly for a quick demo:
    python rag_chain.py
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import config  # noqa: F401
from router import route
from graph_retriever import graph_retrieve
from vector_store import similarity_search

# ---------------------------------------------------------------------------
# LLM + prompt
# ---------------------------------------------------------------------------

_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a knowledgeable movie assistant.

Use ONLY the context below to answer the user's question. If the context
does not contain enough information to answer confidently, say so explicitly
rather than guessing.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}

Answer:"""
)

_chain = _ANSWER_PROMPT | _LLM | StrOutputParser()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_docs(docs: list[Document]) -> str:
    """Concatenate a list of LangChain Documents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_hybrid_rag(query: str, verbose: bool = True) -> dict:
    """
    Execute the full hybrid RAG pipeline for a single query.

    Args:
        query:   The user's natural-language question.
        verbose: If True, print routing and retrieval info to stdout.

    Returns:
        dict with keys:
          - query       : original question (str)
          - retriever   : 'graph' or 'vector' (str)
          - context     : raw retrieved text passed to the LLM (str)
          - answer      : LLM-generated answer (str)
    """
    retriever_type = route(query)
    if verbose:
        print(f"  [Router]  -> {retriever_type.upper()} retrieval")

    # ── Step 2: Retrieve context ─────────────────────────────────────────────
    if retriever_type == "graph":
        context = graph_retrieve(query)
    else:
        docs    = similarity_search(query, k=4)
        context = _format_docs(docs)

    if verbose:
        print(f"  [Context] {len(context)} chars retrieved")

    # ── Step 3: Generate answer ───────────────────────────────────────────────
    answer = _chain.invoke({"context": context, "question": query})

    return {
        "query":     query,
        "retriever": retriever_type,
        "context":   context,
        "answer":    answer,
    }


# ---------------------------------------------------------------------------
# Entry point — quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_queries = [
        "Who acted in The Matrix?",
        "Recommend a movie about survival on a deserted island.",
        "Who has Tom Hanks worked with across his films?",
        "Find me something romantic and emotional to watch.",
        "What did Christopher Nolan direct?",
    ]

    print("Hybrid RAG Demo\n" + "=" * 60)
    for q in sample_queries:
        print(f"\nQ: {q}")
        result = run_hybrid_rag(q, verbose=True)
        print(f"A: {result['answer']}")
        print("-" * 60)
