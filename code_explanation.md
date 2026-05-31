# Line-by-Line Code Explanation Guide

This guide explains in very simple terms what each line of code in our Hybrid Graph RAG project does. 

---

## 1. config.py (Configuration Loader)
This file reads and validates credentials from our `.env` file so other files can use them securely.

```python
import os
```
* **What it does:** Imports Python's built-in `os` (operating system) module.
* **Why:** Lets us read environment variables (like passwords and API keys) and resolve file paths.

```python
from dotenv import load_dotenv
```
* **What it does:** Imports the `load_dotenv` function from the `python-dotenv` package.
* **Why:** Reads the text file named `.env` and loads its variables into Python's environment.

```python
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
```
* **What it does:** Tells `load_dotenv` exactly where to find the `.env` file (in the same directory as `config.py`) and loads it.

```python
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```
* **What it does:** Retrieves the values for our Neo4j database link, username, password, and OpenAI API key from the environment variables.

```python
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
```
* **What it does:** Sets `OPENAI_API_KEY` globally in Python's environment if it exists.
* **Why:** So other libraries (like LangChain) can find and use it automatically.

```python
if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
    raise EnvironmentError(
        "Neo4j connection variables (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) "
        "are not set. Add them to your .env file."
    )
```
* **What it does:** Checks if any Neo4j variables are empty. If they are, it stops the program immediately with an error message telling us to check `.env`.

```python
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. "
        "Add it to the .env file or export it as an environment variable."
    )
```
* **What it does:** Checks if the OpenAI API Key is missing. If it is, it stops the program immediately with an error message.

```python
if __name__ == "__main__":
    print("Configuration loaded:")
    print(f"  NEO4J_URI  : {NEO4J_URI}")
    print(f"  NEO4J_USER : {NEO4J_USER}")
    print(f"  OPENAI_KEY : {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}")
```
* **What it does:** If you run this file directly, it prints out the loaded configuration details (hiding most of the OpenAI Key for safety).

---

## 2. graph_loader.py (Neo4j Data Ingester)
This file connects to our Neo4j database, clears old data, and loads 12 movie entries, their actors, and directors.

### Movie Dataset Definition (Lines 14-159)
* **What it does:** A large list named `MOVIES_DATA` containing dictionaries.
* **Why:** Holds the movie titles, release years, genres, plot descriptions, actor lists, and directors we want to build our graph with.

### Neo4j Helper Functions
```python
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
```
* **What it does:** Creates and returns a Neo4j driver connection object using our URI, username, and password.

```python
def clear_database(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("  [OK] Database cleared.")
```
* **What it does:** Runs a Cypher query `MATCH (n) DETACH DELETE n` which deletes all nodes and connections in the database, ensuring we start with a clean slate.

```python
def load_movies(driver):
    with driver.session() as session:
        for movie in MOVIES_DATA:
```
* **What it does:** Opens a database session and loops through each movie in our dataset.

```python
            session.run(
                """
                MERGE (m:Movie {title: $title})
                SET m.year        = $year,
                    m.genre       = $genre,
                    m.description = $description
                """,
                title=movie["title"],
                year=movie["year"],
                genre=movie["genre"],
                description=movie["description"],
            )
```
* **What it does:** Creates or updates (merges) a node labeled `Movie` with the title. Sets its properties like release year, genre, and plot description.

```python
            for actor in movie["actors"]:
                session.run(
                    """
                    MERGE (p:Person {name: $name})
                    WITH p
                    MATCH (m:Movie {title: $title})
                    MERGE (p)-[:ACTED_IN]->(m)
                    """,
                    name=actor,
                    title=movie["title"],
                )
```
* **What it does:** Loops through the movie's actors. Creates a node labeled `Person` for the actor, matches the `Movie` node, and draws an arrow relationship `ACTED_IN` from the actor to the movie.

```python
            session.run(
                """
                MERGE (d:Person {name: $director})
                WITH d
                MATCH (m:Movie {title: $title})
                MERGE (d)-[:DIRECTED]->(m)
                """,
                director=movie["director"],
                title=movie["title"],
            )
```
* **What it does:** Creates a node labeled `Person` for the director, matches the `Movie` node, and draws a relationship arrow `DIRECTED` from the director to the movie.

```python
def verify_data(driver) -> tuple[int, int]:
    with driver.session() as session:
        movie_count    = session.run("MATCH (m:Movie)          RETURN count(m) AS c").single()["c"]
        person_count   = session.run("MATCH (p:Person)         RETURN count(p) AS c").single()["c"]
        acted_count    = session.run("MATCH ()-[r:ACTED_IN]->() RETURN count(r) AS c").single()["c"]
        directed_count = session.run("MATCH ()-[r:DIRECTED]->() RETURN count(r) AS c").single()["c"]
```
* **What it does:** Queries the counts of movie nodes, person nodes, acted-in connections, and directed connections, then prints them to verify that the graph was loaded correctly.

---

## 3. vector_store.py (FAISS Vector Indexer)
This file turns movie details into mathematical embeddings (vectors) and saves them in a local fast-search database (FAISS) so we can do semantic searches.

```python
def build_documents() -> list[Document]:
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
```
* **What it does:** Converts the structured dictionary information of each movie into a single rich text block.

```python
        metadata = {
            "title":    movie["title"],
            "year":     movie["year"],
            "genre":    movie["genre"],
            "director": movie["director"],
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs
```
* **What it does:** Creates a LangChain `Document` containing the text block and metadata (like title and year), returning the list of documents.

```python
def create_vector_store() -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```
* **What it does:** Sets up OpenAI's `text-embedding-3-small` model to transform text strings into lists of numbers (embeddings).

```python
    docs = build_documents()
    vs = FAISS.from_documents(docs, embeddings)
    return vs
```
* **What it does:** Passes the documents to FAISS, which calls OpenAI's API to calculate the embeddings and constructs a searchable vector index.

```python
_vector_store: Optional[FAISS] = None

def get_vector_store() -> "FAISS":
    global _vector_store
    if _vector_store is None:
        _vector_store = create_vector_store()
    return _vector_store
```
* **What it does:** Ensures we build the vector store only once. Subsequent calls reuse the already built vector store (Singleton Pattern).

```python
def similarity_search(query: str, k: int = 4) -> List[Document]:
    vs = get_vector_store()
    return vs.similarity_search(query, k=k)
```
* **What it does:** Searches the FAISS index to find the top `k` (default 4) movies whose descriptions match the meaning of the query text.

---

## 4. graph_retriever.py (Graph Query Dispatcher)
This file extracts entity names from user text, runs the correct structured Cypher queries in Neo4j, and returns clean results.

### Query Functions (Lines 43-165)
* **What they do:** Functions like `get_actors_in_movie`, `get_movies_by_actor`, `get_director_of_movie`, `get_co_actors`, etc.
* **How:** They take an entity name, insert it into a parameterized Cypher query (e.g. `MATCH (p)-[:ACTED_IN]->(m)`), run it using the driver, format the database rows, and return a clean text string.

### NLP Heuristics & Dispatcher
```python
def graph_retrieve(query: str) -> str:
    q = query.lower()
```
* **What it does:** Turns the user's query into lowercase to make matching easier.

```python
    asks_co_actors      = any(w in q for w in ["acted with", "worked with", "co-act", ...])
    asks_director_query = any(w in q for w in ["who directed", "director of", ...])
    asks_movies_director= any(w in q for w in ["movies by", "films by", ...]) or (...)
    asks_movies_actor   = any(w in q for w in ["movies", "films", "appeared", ...])
    asks_cast           = any(w in q for w in ["cast", "who acted", "who starred", ...])
```
* **What it does:** Analyzes the question text for specific trigger phrases to figure out what the user is asking (e.g. asking for co-stars vs. asking for a director).

```python
    # 1. Co-actor multi-hop
    if asks_co_actors:
        actor = _find_entity(query, KNOWN_ACTORS)
        if actor:
            return get_co_actors(actor)
```
* **What it does:** If the query is about co-stars, it checks if a known actor's name is mentioned in the query. If found, it runs the `get_co_actors` query and returns the results.

*(Lines 233-277 follow this same pattern: they check the classification flags, extract the movie, director, or actor entity, and execute the corresponding Cypher function. If nothing matches, it lists all movies as a fallback.)*

---

## 5. router.py (LLM + Heuristic Query Router)
This file determines whether a question is factual/relationship-oriented (sent to **Neo4j Graph**) or semantic/recommender-oriented (sent to **FAISS Vector**).

### Heuristic Pass (Lines 25-58)
* **What it does:** Looks for explicit keyword matches in the query.
* **Why:** If the query has words like "who", "directed by", or "cast", it instantly routes to `graph`. If it contains words like "recommend", "similar to", or "about", it routes to `vector`.

### LLM Fallback (Lines 65-95)
* **What it does:** If keywords are ambiguous or balanced (e.g., scoring same on both sets), it calls `gpt-4o-mini` with a structured prompt.
* **Why:** The LLM classifies the intent based on semantic understanding and responds with exactly the single word "graph" or "vector".

```python
def route(query: str) -> str:
    decision = _keyword_route(query)
    if decision:
        return decision
    return _llm_route(query)
```
* **What it does:** Runs the fast keyword check first. If it can't decide, it falls back to calling the LLM classification.

---

## 6. rag_chain.py (RAG Pipeline Orchestrator)
This file ties the router, retrievers, and answer generator together into a complete RAG workflow.

```python
_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
```
* **What it does:** Initializes the `gpt-4o-mini` LLM with a low temperature (0.2) to make the responses factual and steady.

```python
_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a knowledgeable movie assistant.
Use ONLY the context below to answer the user's question...
Context: {context}
Question: {question}"""
)
```
* **What it does:** Sets up the system prompt. It instructs the LLM to restrict its answer strictly to the provided context data and avoid guessing.

```python
_chain = _ANSWER_PROMPT | _LLM | StrOutputParser()
```
* **What it does:** Pipes the prompt, the model, and the string output parser together using LangChain's expression language.

```python
def run_hybrid_rag(query: str, verbose: bool = True) -> dict:
    retriever_type = route(query)
```
* **What it does:** Phase 1: Calls `router.py` to decide which retriever to use (`graph` or `vector`).

```python
    if retriever_type == "graph":
        context = graph_retrieve(query)
    else:
        docs    = similarity_search(query, k=4)
        context = _format_docs(docs)
```
* **What it does:** Phase 2: Fetches the context text. If routed to graph, it executes Cypher. If routed to vector, it does similarity search and flattens the documents.

```python
    answer = _chain.invoke({"context": context, "question": query})
```
* **What it does:** Phase 3: Sends the compiled context and original question to the LLM to generate the final response.

---

## 7. evaluate.py (Comparative Benchmarking Suite)
This file runs 10 evaluation queries, compares the outputs of both retrievers, and evaluates accuracy.

```python
def _answer_with_retriever(question: str, retriever_type: str) -> str:
    if retriever_type == "graph":
        context = graph_retrieve(question)
...
    return _answer_chain.invoke({"context": context, "question": question})
```
* **What it does:** Helper function that forces the query to use a specific retriever (bypassing the router) to generate a comparison answer.

```python
def run_evaluation() -> list[dict]:
```
* **What it does:** Main benchmark loop. It iterates through 5 graph questions and 5 vector questions.
* **Why:**
  1. Records routing correctness.
  2. Measures responsiveness (execution time).
  3. Displays the Hybrid RAG answer side-by-side with the forced alternative retriever answer.
  4. Calculates and prints an overall summary of routing accuracy.

---

## 8. main.py (CLI User Interface)
This file is the main command-line script that accepts user options.

```python
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
```
* **What it does:** Reads input arguments from the terminal and branches:
  * `--load`: Triggers `graph_loader.py` to overwrite the graph data.
  * `--eval`: Triggers `evaluate.py` to run the 10-query benchmark.
  * `--demo`: Runs 5 quick pre-set queries.
  * *No argument:* Starts an interactive terminal chat (REPL) loop where you can type queries and chat with the movie assistant.
