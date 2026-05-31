"""
===========================================================================================
graph_retriever.py — Cypher Query Executor & Graph-Based Retrieval Engine
===========================================================================================

PURPOSE:
  This module executes Cypher queries against the Neo4j graph database.
  It takes natural language questions and dispatches them to the right Cypher templates
  to find relationships and entities (actors, movies, directors, collaborations).

WHAT IT DOES:
  1. Maintains a Neo4j driver connection
  2. Defines pre-built Cypher query functions for common searches
  3. Analyzes natural language questions using simple keyword heuristics
  4. Routes questions to the appropriate Cypher function
  5. Returns results as human-readable strings

SIMPLE EXAMPLE:
  User question: "Who acted in The Matrix?"
  
  1. Analyze: Contains "acted", "matrix" → asks about cast
  2. Extract entity: Find "The Matrix" in known movies
  3. Route to: get_actors_in_movie("The Matrix")
  4. Execute Cypher:
       MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
       WHERE m.title = "The Matrix"
       RETURN p.name
  5. Return: "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving"

===========================================================================================
"""

from __future__ import annotations
from typing import Optional, List, Dict

# Import Neo4j driver for database operations
from neo4j import GraphDatabase

# Import configuration (loads environment variables)
import config  # noqa: F401 — ensures env vars are loaded

# Import Neo4j connection credentials from config
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 1: Neo4j Driver Management (Module-Level Singleton)
# ─────────────────────────────────────────────────────────────────────────────────────────
# We create the connection once and reuse it to avoid repeated connection overhead

# Global driver variable (initialized as None, created on first use)
_driver = None


def get_driver():
    """
    PURPOSE:
      Get or create the Neo4j database driver connection.
      Uses singleton pattern: only one driver instance exists across the entire program.
    
    HOW IT WORKS:
      First call: _driver is None, so create it
        - Connects to Neo4j using URI, username, password
        - Stores the connection in the global _driver variable
      
      Subsequent calls: _driver already exists
        - Return the existing connection
        - No need to create a new one
    
    RETURNS:
      GraphDatabase.driver object — ready to execute Cypher queries
    """
    # Declare we're using the global _driver variable
    global _driver
    
    # Check if driver already exists
    if _driver is None:
        # First time: create the driver connection
        # GraphDatabase.driver() takes:
        #   - URI: address of Neo4j server (e.g., bolt://localhost:7687)
        #   - auth: tuple of (username, password)
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Return the driver (either newly created or cached)
    return _driver


def run_query(cypher: str, params: Optional[Dict] = None) -> List[Dict]:
    """
    PURPOSE:
      Execute a read-only Cypher query against Neo4j and return results as dictionaries.
      This is a helper function used by all the query functions below.
    
    HOW IT WORKS:
      1. Get the database driver
      2. Open a session (connection)
      3. Execute the Cypher query with optional parameters
      4. Extract results as dictionaries
      5. Close the session
    
    EXAMPLE CYPHER QUERIES:
      MATCH (m:Movie) RETURN m.title AS title
      MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE m.title = $title RETURN p.name
    
    PARAMETERS:
      cypher: The Cypher query string (can have $param placeholders)
      params: Dictionary of parameter values (e.g., {"title": "The Matrix"})
    
    RETURNS:
      List[Dict] — Each row from the query result as a dictionary
                   Example: [{"title": "The Matrix"}, {"title": "Inception"}]
    """
    # Get the driver connection
    with get_driver().session() as session:
        # Open a session and execute the Cypher query
        # session.run() takes the query and optional parameters
        result = session.run(cypher, params or {})
        # params or {} means: use params if provided, else use empty dict
        
        # Convert results to list of dictionaries
        # result.data() extracts all rows as Python dictionaries
        return result.data()


# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 2: Cypher Query Functions
# ─────────────────────────────────────────────────────────────────────────────────────────
# Pre-built queries for common questions about movies, actors, and directors

def get_actors_in_movie(movie_title: str) -> str:
    """
    PURPOSE:
      Find all actors who appeared in a movie.
    
    EXAMPLE:
      Input: "The Matrix"
      Output: "Movie: The Matrix\nActors: Keanu Reeves, Laurence Fishburne, ..."
    
    CYPHER QUERY EXPLAINED:
      MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
        → Find all Person nodes connected to Movie nodes via ACTED_IN relationships
      WHERE toLower(m.title) CONTAINS toLower($title)
        → Filter to movies whose title (case-insensitive) contains the search term
      RETURN m.title AS movie, collect(p.name) AS actors
        → Return the movie title and collect all actor names into a list
      ORDER BY m.title
        → Sort results alphabetically by movie title
    """
    rows = run_query(
        """
        MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
        WHERE toLower(m.title) CONTAINS toLower($title)
        RETURN m.title AS movie, collect(p.name) AS actors
        ORDER BY m.title
        """,
        {"title": movie_title},
    )
    
    # If no movies found, return a helpful error message
    if not rows:
        return f"No actors found for a movie matching '{movie_title}'."
    
    # Format the results as a readable string
    # For each movie returned, format as "Movie: X\nActors: Y, Z, ..."
    return "\n\n".join(
        f"Movie: {r['movie']}\nActors: {', '.join(r['actors'])}" for r in rows
    )


def get_movies_by_actor(actor_name: str) -> str:
    """
    PURPOSE:
      Find all movies an actor has appeared in.
    
    EXAMPLE:
      Input: "Tom Hanks"
      Output: "Actor: Tom Hanks\nMovies: Forrest Gump (1994), Cast Away (2000), ..."
    
    CYPHER QUERY EXPLAINED:
      MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
        → Find all Person-to-Movie connections via ACTED_IN
      WHERE toLower(p.name) CONTAINS toLower($name)
        → Filter to people whose name contains the search term
      collect(m.title + ' (' + toString(m.year) + ')') AS movies
        → Create a list of movie titles with their release years
        → Example: ["The Matrix (1999)", "Inception (2010)"]
    """
    rows = run_query(
        """
        MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
        WHERE toLower(p.name) CONTAINS toLower($name)
        RETURN p.name AS actor,
               collect(m.title + ' (' + toString(m.year) + ')') AS movies
        """,
        {"name": actor_name},
    )
    
    # If no actor found, return error message
    if not rows:
        return f"No movies found for actor matching '{actor_name}'."
    
    # Format results
    return "\n\n".join(
        f"Actor: {r['actor']}\nMovies: {', '.join(r['movies'])}" for r in rows
    )


def get_director_of_movie(movie_title: str) -> str:
    """
    PURPOSE:
      Find who directed a specific movie.
    
    EXAMPLE:
      Input: "Inception"
      Output: "Movie: Inception\nDirector: Christopher Nolan"
    
    CYPHER QUERY EXPLAINED:
      MATCH (d:Person)-[:DIRECTED]->(m:Movie)
        → Find all Person-to-Movie connections via DIRECTED relationship
      WHERE toLower(m.title) CONTAINS toLower($title)
        → Filter to movies whose title contains the search term
      RETURN m.title AS movie, d.name AS director
        → Return the movie and its director's name
    """
    rows = run_query(
        """
        MATCH (d:Person)-[:DIRECTED]->(m:Movie)
        WHERE toLower(m.title) CONTAINS toLower($title)
        RETURN m.title AS movie, d.name AS director
        ORDER BY m.title
        """,
        {"title": movie_title},
    )
    
    if not rows:
        return f"No director found for movie matching '{movie_title}'."
    
    return "\n\n".join(
        f"Movie: {r['movie']}\nDirector: {r['director']}" for r in rows
    )


def get_movies_by_director(director_name: str) -> str:
    """
    PURPOSE:
      Find all movies directed by a specific person.
    
    EXAMPLE:
      Input: "Christopher Nolan"
      Output: "Director: Christopher Nolan\nMovies: The Dark Knight (2008), Inception (2010)"
    """
    rows = run_query(
        """
        MATCH (d:Person)-[:DIRECTED]->(m:Movie)
        WHERE toLower(d.name) CONTAINS toLower($name)
        RETURN d.name AS director,
               collect(m.title + ' (' + toString(m.year) + ')') AS movies
        """,
        {"name": director_name},
    )
    
    if not rows:
        return f"No movies found for director matching '{director_name}'."
    
    return "\n\n".join(
        f"Director: {r['director']}\nMovies: {', '.join(r['movies'])}" for r in rows
    )


def get_co_actors(actor_name: str) -> str:
    """
    PURPOSE:
      Find all actors who have worked with a specific actor (multi-hop query).
      This is a "multi-hop" query because it traverses: Actor → Movie → Other Actors
    
    EXAMPLE:
      Input: "Keanu Reeves"
      Output: "Actor: Keanu Reeves\nCo-actors: Laurence Fishburne, Sandra Bullock, ..."
    
    CYPHER QUERY EXPLAINED:
      MATCH (a:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(co:Person)
        → Find a chain: Person1 -ACTED_IN-> Movie <-ACTED_IN- Person2
        → Effectively: "Find all pairs of people in the same movie"
      WHERE toLower(a.name) CONTAINS toLower($name) AND a.name <> co.name
        → Filter to the target actor AND exclude self-matches
      collect(DISTINCT co.name) AS co_actors
        → Collect all co-actor names, removing duplicates (DISTINCT)
    """
    rows = run_query(
        """
        MATCH (a:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(co:Person)
        WHERE toLower(a.name) CONTAINS toLower($name)
          AND a.name <> co.name
        RETURN a.name AS actor, collect(DISTINCT co.name) AS co_actors
        """,
        {"name": actor_name},
    )
    
    if not rows:
        return f"No co-actors found for '{actor_name}'."
    
    return "\n\n".join(
        f"Actor: {r['actor']}\nCo-actors: {', '.join(r['co_actors'])}" for r in rows
    )


def get_directors_of_actor(actor_name: str) -> str:
    """
    PURPOSE:
      Find all directors who have directed a specific actor.
      Another multi-hop query: Actor → Movie → Director
    
    EXAMPLE:
      Input: "Leonardo DiCaprio"
      Output: "Actor: Leonardo DiCaprio\nDirectors: James Cameron (in Titanic), ..."
    """
    rows = run_query(
        """
        MATCH (a:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Person)
        WHERE toLower(a.name) CONTAINS toLower($name)
        RETURN a.name AS actor,
               collect(DISTINCT d.name + ' (in ' + m.title + ')') AS directors
        """,
        {"name": actor_name},
    )
    
    if not rows:
        return f"No directors found for actor '{actor_name}'."
    
    return "\n\n".join(
        f"Actor: {r['actor']}\nDirectors: {', '.join(r['directors'])}" for r in rows
    )


def get_all_movies() -> str:
    """
    PURPOSE:
      Return a formatted list of all movies in the graph.
      Used as a fallback when no specific query matches.
    
    EXAMPLE OUTPUT:
      The Matrix (1999) — Sci-Fi
      Forrest Gump (1994) — Drama
      ...
    """
    rows = run_query(
        "MATCH (m:Movie) RETURN m.title AS title, m.year AS year, m.genre AS genre "
        "ORDER BY m.year"
    )
    
    if not rows:
        return "No movies found in the graph."
    
    return "\n".join(f"{r['title']} ({r['year']}) — {r['genre']}" for r in rows)


# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 3: Known Entities Lists
# ─────────────────────────────────────────────────────────────────────────────────────────
# Lists of known actors, movies, and directors for entity extraction from user queries

# All known actors in our dataset
KNOWN_ACTORS = [
    "Keanu Reeves", "Tom Hanks", "Denzel Washington", "Leonardo DiCaprio",
    "Christian Bale", "Laurence Fishburne", "Carrie-Anne Moss", "Hugo Weaving",
    "Robin Wright", "Gary Sinise", "Helen Hunt", "Sandra Bullock", "Dennis Hopper",
    "Michael Nyqvist", "Jason Robards", "Ethan Hawke", "Heath Ledger",
    "Aaron Eckhart", "Joseph Gordon-Levitt", "Elliot Page", "Kate Winslet",
    "Catherine Zeta-Jones", "Matt Damon", "Mykelti Williamson",
]

# All known movies in our dataset
KNOWN_MOVIES = [
    "The Matrix", "Forrest Gump", "Cast Away", "Speed", "John Wick",
    "Philadelphia", "Training Day", "The Dark Knight", "Inception", "Titanic",
    "The Terminal", "Saving Private Ryan",
]

# All known directors in our dataset
KNOWN_DIRECTORS = [
    "Lana Wachowski", "Robert Zemeckis", "Jan de Bont", "Chad Stahelski",
    "Jonathan Demme", "Antoine Fuqua", "Christopher Nolan", "James Cameron",
    "Steven Spielberg",
]


def _find_entity(query: str, entity_list: List[str]) -> Optional[str]:
    """
    PURPOSE:
      Extract the first known entity from a user query (case-insensitive).
      This is lightweight NLP for entity recognition.
    
    EXAMPLE:
      query = "Who acted in The Matrix?"
      entity_list = KNOWN_MOVIES
      Returns: "The Matrix"
    
    HOW IT WORKS:
      1. Convert query to lowercase for case-insensitive matching
      2. Loop through each entity in the list
      3. Check if the entity's lowercase name appears in the query
      4. Return the first match (or None if no match)
    """
    # Convert query to lowercase for comparison
    q_lower = query.lower()
    
    # Loop through each known entity
    for entity in entity_list:
        # Check if this entity's name appears in the query (case-insensitive)
        if entity.lower() in q_lower:
            return entity
    
    # No entity found
    return None


# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 4: Main Dispatcher / Query Router
# ─────────────────────────────────────────────────────────────────────────────────────────
# Analyzes natural language and routes to the appropriate Cypher function

def graph_retrieve(query: str) -> str:
    """
    PURPOSE:
      Analyze a natural-language query and dispatch to the appropriate Cypher function.
      Returns a human-readable string with the graph query results.
    
    HOW IT WORKS:
      1. Detect user intent from keywords
      2. Extract entities (actor, movie, director names) from the query
      3. Route to the matching Cypher function
      4. Return formatted results
    
    EXAMPLE QUERIES AND ROUTING:
      "Who acted in The Matrix?"
        → Intent: cast lookup
        → Extract: movie="The Matrix"
        → Call: get_actors_in_movie("The Matrix")
      
      "Who has Keanu Reeves worked with?"
        → Intent: co-actors
        → Extract: actor="Keanu Reeves"
        → Call: get_co_actors("Keanu Reeves")
      
      "What movies did Christopher Nolan direct?"
        → Intent: director filmography
        → Extract: director="Christopher Nolan"
        → Call: get_movies_by_director("Christopher Nolan")
    """
    # Convert query to lowercase for easier keyword matching
    q = query.lower()

    # ─────────────────────────────────────────────────────────────────────────────
    # Intent Detection: Look for keyword signals
    # ─────────────────────────────────────────────────────────────────────────────
    # Each of these checks for specific keywords that indicate the user's intent
    
    # Check if asking about co-actors (people who worked together)
    asks_co_actors = any(w in q for w in ["acted with", "worked with", "co-act",
                                          "who else", "same movie as", "co-star"])
    
    # Check if asking about directors
    asks_director_query = any(w in q for w in ["who directed", "director of", "directed", "direct", "director"])
    
    # Check if asking for a director's filmography (movies by/directed by a director)
    asks_movies_director = any(w in q for w in ["movies by", "films by", "directed by", "movies did", "films did"]) or (any(w in q for w in ["direct", "director", "directed"]) and _find_entity(query, KNOWN_DIRECTORS))
    
    # Check if asking for an actor's filmography
    asks_movies_actor = any(w in q for w in ["movies", "films", "appeared", "starred",
                                            "acted in", "filmography"])
    
    # Check if asking for movie cast
    asks_cast = any(w in q for w in ["cast", "who acted", "who starred",
                                     "who is in", "actors in"])

    # ─────────────────────────────────────────────────────────────────────────────
    # Route to the appropriate Cypher function based on intent
    # ─────────────────────────────────────────────────────────────────────────────
    # The order matters: check more specific conditions first

    # 1. Multi-hop: Find co-actors
    if asks_co_actors:
        actor = _find_entity(query, KNOWN_ACTORS)
        if actor:
            return get_co_actors(actor)

    # 2. Multi-hop: Find directors who worked with an actor
    if asks_director_query and not asks_movies_director:
        actor = _find_entity(query, KNOWN_ACTORS)
        if actor:
            return get_directors_of_actor(actor)

    # 3. Director's filmography
    if asks_movies_director:
        director = _find_entity(query, KNOWN_DIRECTORS)
        if director:
            return get_movies_by_director(director)

    # 4. Director of a specific movie
    if asks_director_query:
        movie = _find_entity(query, KNOWN_MOVIES)
        if movie:
            return get_director_of_movie(movie)
        director = _find_entity(query, KNOWN_DIRECTORS)
        if director:
            return get_movies_by_director(director)

    # 5. Actor's filmography
    if asks_movies_actor:
        actor = _find_entity(query, KNOWN_ACTORS)
        if actor:
            return get_movies_by_actor(actor)

    # 6. Cast of a movie
    if asks_cast:
        movie = _find_entity(query, KNOWN_MOVIES)
        if movie:
            return get_actors_in_movie(movie)

    # 7. Generic actor lookup
    actor = _find_entity(query, KNOWN_ACTORS)
    if actor:
        return get_movies_by_actor(actor)

    # 8. Generic movie lookup
    movie = _find_entity(query, KNOWN_MOVIES)
    if movie:
        return get_actors_in_movie(movie)

    # 9. Fallback — list all movies
    return "Available movies in the graph:\n" + get_all_movies()


# ─────────────────────────────────────────────────────────────────────────────────────────
# SECTION 5: Main Entry Point (runs when you execute: python graph_retriever.py)
# ─────────────────────────────────────────────────────────────────────────────────────────
# Smoke test to verify graph retriever works

if __name__ == "__main__":
    # Test queries covering different types of questions
    tests = [
        "Who acted in The Matrix?",
        "What movies has Tom Hanks appeared in?",
        "Who directed Inception?",
        "Who has Keanu Reeves acted with?",
        "What movies did Christopher Nolan direct?",
        "Which directors has Denzel Washington worked with?",
    ]

    print("Graph retriever smoke test\n" + "=" * 50)
    
    # Run each test query
    for q in tests:
        print(f"\nQ: {q}")
        # Execute the query and print results
        print(graph_retrieve(q))
