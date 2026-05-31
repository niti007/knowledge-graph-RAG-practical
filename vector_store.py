"""
vector_store.py — Builds a FAISS in-memory vector store from movie data
using OpenAI text-embedding-3-small.

Run directly to test:
    python vector_store.py
"""

from __future__ import annotations
from typing import Optional, List

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config  # noqa: F401 — ensures OPENAI_API_KEY is set in os.environ
from graph_loader import MOVIES_DATA

# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------

def build_documents() -> list[Document]:
    """
    Convert each movie in MOVIES_DATA into a LangChain Document.
    The page_content is a rich text block that embeddings will be built from.
    """
    docs = []
    for movie in MOVIES_DATA:
        content = (
            f"Title: {movie['title']}\n"
            f"Year: {movie['year']}\n"
            f"Genre: {movie['genre']}\n"
            f"Director: {movie['director']}\n"
            f"Cast: {', '.join(movie['actors'])}\n"
            f"Description: {movie['description']}"
        )
        metadata = {
            "title":    movie["title"],
            "year":     movie["year"],
            "genre":    movie["genre"],
            "director": movie["director"],
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


# ---------------------------------------------------------------------------
# Vector store factory
# ---------------------------------------------------------------------------

def create_vector_store() -> FAISS:
    """
    Embed all movie documents and return a FAISS vector store.
    Called once; subsequent calls use the cached singleton.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = build_documents()
    print(f"  Embedding {len(docs)} movie documents ...")
    vs = FAISS.from_documents(docs, embeddings)
    print("  [OK] FAISS vector store built.")
    return vs


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_vector_store: Optional[FAISS] = None


def get_vector_store() -> "FAISS":
    """Return the cached vector store, creating it on first call."""
    global _vector_store
    if _vector_store is None:
        _vector_store = create_vector_store()
    return _vector_store


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def similarity_search(query: str, k: int = 4) -> List[Document]:
    """
    Perform semantic similarity search against the FAISS vector store.

    Args:
        query: Natural language question or description.
        k:     Number of top-matching documents to return.

    Returns:
        List of LangChain Documents ranked by semantic similarity.
    """
    vs = get_vector_store()
    return vs.similarity_search(query, k=k)


# ---------------------------------------------------------------------------
# Entry point — quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building vector store ...")
    vs = create_vector_store()

    test_queries = [
        "space exploration and simulated reality",
        "romantic love story and disaster",
        "crime corruption and moral dilemmas",
    ]

    print()
    for q in test_queries:
        print(f"Query: '{q}'")
        results = vs.similarity_search(q, k=3)
        for r in results:
            print(f"  -> {r.metadata['title']} ({r.metadata['year']}) [{r.metadata['genre']}]")
        print()
