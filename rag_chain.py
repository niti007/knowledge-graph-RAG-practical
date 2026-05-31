"""
===========================================================================================
rag_chain.py — Hybrid RAG Pipeline Orchestrator
===========================================================================================

PURPOSE:
  This module orchestrates the complete RAG (Retrieval-Augmented Generation) pipeline.
  It combines the router, retrievers, and LLM to answer user questions with sourced context.

WHAT IT DOES:
  1. Routes the user query to the appropriate retriever (graph or vector)
  2. Retrieves relevant context using the selected retriever
  3. Generates a grounded answer using GPT-4o-mini
  4. Returns the answer along with retriever source information

SIMPLE EXAMPLE:
  User: "Who acted in The Matrix?"
  
  Step 1 (Route):    Detects graph query → Use graph retriever
  Step 2 (Retrieve): Executes Cypher → Gets "Keanu Reeves, Laurence Fishburne, ..."
  Step 3 (Generate): LLM reads context + question → Writes polished answer
  Step 4 (Return):   {
                      "query": "Who acted in The Matrix?",
                      "retriever": "graph",
                      "context": "... (raw retrieved data) ...",
                      "answer": "The main cast includes Keanu Reeves as Neo, ..."
                    }

===========================================================================================
"""

# Import LangChain's LLM interface
from langchain_openai import ChatOpenAI

# Import LangChain's prompt templating system
from langchain_core.prompts import ChatPromptTemplate

# Import LangChain's output parser (converts LLM output to strings)
from langchain_core.output_parsers import StrOutputParser

# Import LangChain's Document class (for type hints)
from langchain_core.documents import Document

# Import configuration (ensures API keys are loaded)
import config  # noqa: F401

# Import the query router
from router import route

# Import the graph-based retriever
from graph_retriever import graph_retrieve

# Import the vector-based retriever
from vector_store import similarity_search

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 1: LLM Configuration & Prompt Template
# ─────────────────────────────────────────────────────────────────────────────────────────
# Define the LLM model and the prompt it will use

# Initialize the language model
# model="gpt-4o-mini" is a fast, cheaper version of GPT-4 (good for routing/summarization)
# temperature=0.2 means low creativity, high factuality (deterministic responses)
#   - temperature=0.0 = always same response (most deterministic)
#   - temperature=0.5 = moderate variation (balanced)
#   - temperature=1.0 = high variation (creative, less reliable)
_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Define the prompt template for answer generation
# This tells the LLM exactly how to generate answers
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
# This prompt:
#   1. Sets the role: "knowledgeable movie assistant"
#   2. Gives instructions: "Use ONLY the context"
#   3. Defines placeholders: {context} and {question}
#   4. Tells it to avoid hallucination

# Create the RAG chain by piping components together
# prompt → LLM → string parser
# The | operator chains components left to right
_chain = _ANSWER_PROMPT | _LLM | StrOutputParser()
# This means:
#   1. ChatPromptTemplate formats the prompt with variables
#   2. ChatOpenAI generates text based on the prompt
#   3. StrOutputParser extracts the text as a clean string


# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 2: Helper Functions
# ─────────────────────────────────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    """
    PURPOSE:
      Convert a list of LangChain Documents into a single context string.
      This formats the retrieved documents for the LLM to read.
    
    HOW IT WORKS:
      Takes multiple Document objects (each with page_content and metadata).
      Joins them together with a separator ("---") to create one long string.
      The LLM will use this string as context for generating answers.
    
    EXAMPLE:
      Input:
        [
          Document(page_content="Title: The Matrix\nYear: 1999\n..."),
          Document(page_content="Title: Inception\nYear: 2010\n..."),
        ]
      
      Output:
        "Title: The Matrix\nYear: 1999\n...\n\n---\n\nTitle: Inception\nYear: 2010\n..."
    
    PARAMETERS:
      docs: List of LangChain Document objects from similarity_search()
    
    RETURNS:
      str — A single formatted context string ready for the LLM
    """
    # Join all documents' page_content with a separator
    # "\n\n---\n\n" creates a clear visual break between documents
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 3: Main RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────────────────

def run_hybrid_rag(query: str, verbose: bool = True) -> dict:
    """
    PURPOSE:
      Execute the complete hybrid RAG pipeline for a single user query.
      This is the main function that orchestrates everything.
    
    HOW IT WORKS:
      PHASE 1 (Route): Classify the query → decide which retriever to use
      PHASE 2 (Retrieve): Fetch relevant context using the chosen retriever
      PHASE 3 (Generate): Use LLM to generate a grounded answer
      PHASE 4 (Return): Package everything into a result dictionary
    
    COMPLETE FLOW:
      Input: "Who acted in The Matrix?"
        ↓
      PHASE 1: route(query) → "graph" (structural query)
        ↓
      PHASE 2: graph_retrieve(query) → "Keanu Reeves, Laurence Fishburne, ..."
        ↓
      PHASE 3: _chain.invoke({"context": context, "question": query})
        → LLM reads: "Who acted in The Matrix?"
        → LLM reads context: retrieved actor list
        → LLM generates: "The main cast includes Keanu Reeves as Neo, ..."
        ↓
      Output: Dictionary with query, retriever, context, answer
    
    PARAMETERS:
      query:   Natural-language question from the user
               Example: "Who directed Inception?"
      
      verbose: If True, print routing and context info to stdout
               Example: "[Router] -> GRAPH retrieval"
    
    RETURNS:
      dict with keys:
        - query:     Original question (str)
        - retriever: Which retriever was used: 'graph' or 'vector' (str)
        - context:   Raw context passed to the LLM (str)
        - answer:    Final generated answer (str)
      
      Example return:
        {
          "query": "Who acted in The Matrix?",
          "retriever": "graph",
          "context": "Movie: The Matrix\nActors: Keanu Reeves, ...",
          "answer": "The main cast of The Matrix includes Keanu Reeves, ..."
        }
    """
    
    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 1: Route the query to determine which retriever to use
    # ─────────────────────────────────────────────────────────────────────────────
    # route() returns either "graph" or "vector"
    retriever_type = route(query)
    
    # If verbose mode is on, print which retriever was chosen
    if verbose:
        print(f"  [Router]  -> {retriever_type.upper()} retrieval")

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 2: Retrieve context using the chosen retriever
    # ─────────────────────────────────────────────────────────────────────────────
    
    # Check which retriever type was chosen
    if retriever_type == "graph":
        # Use graph retriever (Neo4j Cypher queries)
        # graph_retrieve() returns a string with the query results
        context = graph_retrieve(query)
    else:
        # Use vector retriever (FAISS semantic search)
        # similarity_search() returns a list of LangChain Documents
        docs = similarity_search(query, k=4)
        # k=4 means return top 4 most similar movies
        
        # Convert the Documents to a single context string for the LLM
        context = _format_docs(docs)

    # If verbose mode is on, print how much context was retrieved
    if verbose:
        print(f"  [Context] {len(context)} chars retrieved")

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 3: Generate the final answer using the LLM
    # ─────────────────────────────────────────────────────────────────────────────
    
    # Invoke the RAG chain to generate the answer
    # The chain does: prompt formatting → LLM call → string parsing
    # We pass two variables:
    #   - context: the retrieved information
    #   - question: the original user query
    answer = _chain.invoke({"context": context, "question": query})

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 4: Return results in a structured dictionary
    # ─────────────────────────────────────────────────────────────────────────────
    
    # Package everything into a result dictionary
    return {
        "query":     query,                  # The original question
        "retriever": retriever_type,         # 'graph' or 'vector'
        "context":   context,                # Raw retrieved context
        "answer":    answer,                 # LLM-generated answer
    }


# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 4: Main Entry Point (runs when you execute: python rag_chain.py)
# ─────────────────────────────────────────────────────────────────────────────────────────
# Quick demo of the hybrid RAG pipeline

if __name__ == "__main__":
    # Sample queries that showcase both retrieval paths
    sample_queries = [
        # GRAPH queries (structural)
        "Who acted in The Matrix?",
        
        # VECTOR queries (semantic/recommendation)
        "Recommend a movie about survival on a deserted island.",
        
        # GRAPH queries (multi-hop)
        "Who has Tom Hanks worked with across his films?",
        
        # VECTOR queries (mood/theme-based)
        "Find me something romantic and emotional to watch.",
        
        # GRAPH queries (director filmography)
        "What did Christopher Nolan direct?",
    ]

    print("Hybrid RAG Demo\n" + "=" * 60)
    
    # Run each query through the pipeline
    for q in sample_queries:
        print(f"\nQ: {q}")
        
        # Execute the full RAG pipeline
        result = run_hybrid_rag(q, verbose=True)
        
        # Print the final answer
        print(f"A: {result['answer']}")
        print("-" * 60)
