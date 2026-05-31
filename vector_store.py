"""
===========================================================================================
vector_store.py — FAISS Vector Store Builder for Semantic Search
===========================================================================================

PURPOSE:
  This module converts movie text into mathematical embeddings (vectors) and stores them
  in a FAISS index. This enables semantic similarity search (finding movies by meaning,
  not exact keywords).

WHAT IT DOES:
  1. Takes movie data and converts it to rich text documents
  2. Uses OpenAI's embedding model to convert text → vectors (lists of numbers)
  3. Stores all vectors in FAISS (a fast search database)
  4. Provides a search function to find similar movies by query meaning

SIMPLE EXAMPLE:
  Movie description: "A hacker discovers he lives in a simulated reality"
  ↓ (embedding model)
  Vector: [0.123, -0.456, 0.789, 0.321, ..., -0.654] (1536 numbers)
  
  User query: "Find me a mind-bending sci-fi movie"
  ↓ (same embedding model)
  Query vector: [0.119, -0.451, 0.792, 0.318, ..., -0.651]
  
  FAISS compares vectors and finds: "The Matrix" is very similar!
  (vectors with similar numbers = similar meaning)

===========================================================================================
"""

from __future__ import annotations
from typing import Optional, List

import os
# Import OpenAI's embedding model
# This model converts text strings into vectors (lists of 1536 numbers)
from langchain_openai import OpenAIEmbeddings

# Import FAISS (Facebook AI Similarity Search)
# This creates a searchable index of vectors for fast similarity matching
from langchain_community.vectorstores import FAISS

# Import LangChain's Document class
# A Document holds content and metadata together
from langchain_core.documents import Document

# Import configuration (ensures OPENAI_API_KEY is available)
import config  # noqa: F401 — ensures OPENAI_API_KEY is set in os.environ

# Import the movie dataset from graph_loader
from graph_loader import MOVIES_DATA

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 1: Document Builder
# ─────────────────────────────────────────────────────────────────────────────────────────
# Convert raw movie data into searchable documents

def build_documents() -> list[Document]:
    """
    PURPOSE:
      Convert each movie dictionary into a LangChain Document.
      A Document has:
        - page_content: the text that will be embedded
        - metadata: searchable information (title, year, genre, etc.)
    
    HOW IT WORKS:
      For each movie, we combine all its info into a single rich text block.
      This text will be converted to a vector (embedding).
    
    EXAMPLE DOCUMENT CREATED:
      page_content:
        "Title: The Matrix
         Year: 1999
         Genre: Sci-Fi
         Director: Lana Wachowski
         Cast: Keanu Reeves, Laurence Fishburne, ...
         Description: A computer hacker learns from mysterious rebels..."
      
      metadata:
        {
          "title": "The Matrix",
          "year": 1999,
          "genre": "Sci-Fi",
          "director": "Lana Wachowski"
        }
    
    RETURNS:
      list[Document] — one Document per movie
    """
    # Create an empty list to hold all documents
    docs = []
    
    # Loop through each movie in MOVIES_DATA
    for movie in MOVIES_DATA:
        
        # ─────────────────────────────────────────────────────────────────────────
        # Build the rich text content block
        # ─────────────────────────────────────────────────────────────────────────
        # Combine all movie info into a single formatted string
        # This string will be embedded (converted to a vector)
        content = (
            f"Title: {movie['title']}\n"
            f"Year: {movie['year']}\n"
            f"Genre: {movie['genre']}\n"
            f"Director: {movie['director']}\n"
            f"Cast: {', '.join(movie['actors'])}\n"
            f"Description: {movie['description']}"
        )
        # This creates a string like:
        # "Title: The Matrix\nYear: 1999\nGenre: Sci-Fi\n..."
        
        # ─────────────────────────────────────────────────────────────────────────
        # Prepare the metadata (searchable fields)
        # ─────────────────────────────────────────────────────────────────────────
        # Metadata is structured data about the movie
        # We keep it separate so we can search/filter by these fields later
        metadata = {
            "title":    movie["title"],
            "year":     movie["year"],
            "genre":    movie["genre"],
            "director": movie["director"],
        }
        
        # ─────────────────────────────────────────────────────────────────────────
        # Create a LangChain Document and add to list
        # ─────────────────────────────────────────────────────────────────────────
        # A Document is like a book page:
        #   - page_content: the full text that will be searched
        #   - metadata: tags/labels about the page
        docs.append(Document(page_content=content, metadata=metadata))
    
    # Return the list of all movie documents
    return docs


# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 2: Vector Store Factory
# ─────────────────────────────────────────────────────────────────────────────────────────

def create_vector_store() -> FAISS:
    """
    PURPOSE:
      Build the FAISS vector index from all movie documents.
      This involves:
        1. Loading the embedding model
        2. Converting all movie texts to vectors
        3. Building a searchable index
    
    HOW IT WORKS:
      Step 1: Create an OpenAIEmbeddings object
        - This loads the "text-embedding-3-small" model from OpenAI
        - Small model = faster, cheaper, but slightly less accurate
        - Large model = slower, more expensive, more accurate
      
      Step 2: Build documents
        - Calls build_documents() to create LangChain Documents
      
      Step 3: Call FAISS.from_documents()
        - Passes documents to FAISS
        - FAISS calls the embedding model to convert each text to a vector
        - FAISS builds an index optimized for fast similarity search
    
    EXAMPLE PROCESS:
      Movie 1: "The Matrix" + full text
        ↓ (embedding model)
      Vector: [0.123, -0.456, ..., -0.654]  (1536 values)
      
      Movie 2: "Forrest Gump" + full text
        ↓ (embedding model)
      Vector: [0.234, -0.567, ..., -0.765]  (1536 values)
      
      ... (repeat for all 12 movies)
      
      ↓ (FAISS builds search index)
      
      Ready to answer: "Find me a sci-fi movie" by comparing vectors
    
    RETURNS:
      FAISS object — a searchable vector index
    """
    # Initialize the OpenAI embedding model
    # "text-embedding-3-small" is fast and good for most use cases
    # This will call OpenAI's API whenever we embed text
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Build all movie documents from MOVIES_DATA
    docs = build_documents()
    
    # Print status message
    print(f"  Embedding {len(docs)} movie documents ...")
    
    # Create FAISS index and populate it with embeddings
    # This calls OpenAI's API to embed each document
    # FAISS.from_documents():
    #   1. Takes each Document's page_content
    #   2. Calls the embeddings model to convert text to vector
    #   3. Stores the vector in the FAISS index
    # (This step takes a few seconds due to API calls)
    vs = FAISS.from_documents(docs, embeddings)
    
    # Print success message
    print("  [OK] FAISS vector store built.")
    
    return vs


# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 3: Singleton Accessor
# ─────────────────────────────────────────────────────────────────────────────────────────
# Ensure the vector store is built only once (for efficiency)

# Global variable to cache the vector store
# It starts as None; gets populated on first access
_vector_store: Optional[FAISS] = None


def get_vector_store() -> "FAISS":
    """
    PURPOSE:
      Return the cached vector store, building it on first call.
      This implements the "Singleton Pattern" — only one instance exists.
    
    HOW IT WORKS:
      First call: _vector_store is None, so we call create_vector_store()
        - Builds embeddings for all 12 movies
        - Takes a few seconds
        - Caches the result in _vector_store
      
      Subsequent calls: _vector_store already exists
        - Return it immediately (much faster)
        - No need to rebuild
    
    EXAMPLE:
      # First call (slow, ~2-3 seconds)
      vs = get_vector_store()
      
      # Second call (instant)
      vs = get_vector_store()  # Returns cached version
    
    RETURNS:
      FAISS object — the cached vector store
    """
    # Declare that we're working with the global _vector_store variable
    global _vector_store
    
    # Check if we've already built the vector store
    if _vector_store is None:
        # First time: build it
        _vector_store = create_vector_store()
    
    # Return the (possibly newly built or cached) vector store
    return _vector_store


# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 4: Public API — Search Function
# ─────────────────────────────────────────────────────────────────────────────────────────

def similarity_search(query: str, k: int = 4) -> List[Document]:
    """
    PURPOSE:
      Find the top k movies that are semantically similar to the query.
      Uses vector similarity (not keyword matching).
    
    HOW IT WORKS:
      1. Get the cached vector store
      2. Convert the query to a vector using the same embedding model
      3. FAISS finds the k vectors most similar to the query vector
      4. Return the corresponding movie Documents
    
    SIMPLE EXAMPLE:
      Query: "I want a mind-bending sci-fi movie"
      ↓ (convert to vector)
      Query vector: [0.456, -0.789, 0.123, ...]
      ↓ (find similar vectors in FAISS index)
      Top matches:
        - "The Matrix" (vector: [0.451, -0.792, 0.119, ...])  ← very similar!
        - "Inception" (vector: [0.448, -0.785, 0.125, ...])   ← similar
        - "The Dark Knight" (vector: [...])                   ← somewhat similar
        - "Titanic" (vector: [...])                           ← least similar (but still returned)
    
    PARAMETERS:
      query: Natural language question or description string
             Example: "Find me a romantic movie about a ship"
      
      k:     Number of top matches to return (default: 4)
             Example: k=1 returns only the best match
    
    RETURNS:
      List[Document] — List of Document objects ranked by similarity
                       Each Document has:
                         - page_content: the full text
                         - metadata: title, year, genre, director
    
    EXAMPLE USAGE:
      docs = similarity_search("recommend a romantic movie", k=3)
      for doc in docs:
          print(doc.metadata["title"])  # Print movie title
    """
    # Get the vector store (builds it if needed)
    vs = get_vector_store()
    
    # Perform similarity search and return top k results
    # vs.similarity_search():
    #   1. Converts query to a vector (using the same embedding model)
    #   2. Finds the k vectors in the index most similar to the query
    #   3. Returns the original Documents corresponding to those vectors
    return vs.similarity_search(query, k=k)


# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 5: Main Entry Point (runs when you execute: python vector_store.py)
# ─────────────────────────────────────────────────────────────────────────────────────────
# Useful for testing the vector store in isolation

if __name__ == "__main__":
    print("Building vector store ...")
    # Create and cache the vector store
    vs = create_vector_store()

    # Test queries to demonstrate semantic search
    test_queries = [
        "space exploration and simulated reality",
        "romantic love story and disaster",
        "crime corruption and moral dilemmas",
    ]

    print()
    
    # Run each test query and show the top 3 results
    for q in test_queries:
        print(f"Query: '{q}'")
        
        # Perform semantic similarity search for top 3 matches
        results = vs.similarity_search(q, k=3)
        
        # Display the top matches with their metadata
        for r in results:
            # Extract metadata and display in a readable format
            print(f"  -> {r.metadata['title']} ({r.metadata['year']}) [{r.metadata['genre']}]")
        print()
