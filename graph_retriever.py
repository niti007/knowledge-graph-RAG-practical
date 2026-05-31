"""
graph_retriever.py — Cypher-based retrieval for relationship and entity queries.

Routes natural-language questions to the correct Neo4j Cypher template and
returns a human-readable string of graph results.

Run directly for a quick smoke test:
    python graph_retriever.py
"""

from __future__ import annotations
from typing import Optional, List, Dict

from neo4j import GraphDatabase
import config  # noqa: F401 — ensures env vars are loaded
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# ---------------------------------------------------------------------------
# Neo4j driver (module-level singleton)
# ---------------------------------------------------------------------------

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def run_query(cypher: str, params: Optional[Dict] = None) -> List[Dict]:
    """Execute a read-only Cypher query and return results as a list of dicts."""
    with get_driver().session() as session:
        result = session.run(cypher, params or {})
        return result.data()


# ---------------------------------------------------------------------------
# Cypher query functions
# ---------------------------------------------------------------------------

def get_actors_in_movie(movie_title: str) -> str:
    """Return the cast of a movie (partial title match)."""
    rows = run_query(
        """
        MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
        WHERE toLower(m.title) CONTAINS toLower($title)
        RETURN m.title AS movie, collect(p.name) AS actors
        ORDER BY m.title
        """,
        {"title": movie_title},
    )
    if not rows:
        return f"No actors found for a movie matching '{movie_title}'."
    return "\n\n".join(
        f"Movie: {r['movie']}\nActors: {', '.join(r['actors'])}" for r in rows
    )


def get_movies_by_actor(actor_name: str) -> str:
    """Return all movies a given actor has appeared in."""
    rows = run_query(
        """
        MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
        WHERE toLower(p.name) CONTAINS toLower($name)
        RETURN p.name AS actor,
               collect(m.title + ' (' + toString(m.year) + ')') AS movies
        """,
        {"name": actor_name},
    )
    if not rows:
        return f"No movies found for actor matching '{actor_name}'."
    return "\n\n".join(
        f"Actor: {r['actor']}\nMovies: {', '.join(r['movies'])}" for r in rows
    )


def get_director_of_movie(movie_title: str) -> str:
    """Return the director of a movie."""
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
    """Return all movies directed by a given person."""
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
    Multi-hop query: find all people who appeared in the same movie
    as the given actor.
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
    Multi-hop query: find all directors who have directed a given actor,
    including which film they collaborated on.
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
    """Return a formatted list of all movies in the graph."""
    rows = run_query(
        "MATCH (m:Movie) RETURN m.title AS title, m.year AS year, m.genre AS genre "
        "ORDER BY m.year"
    )
    if not rows:
        return "No movies found in the graph."
    return "\n".join(f"{r['title']} ({r['year']}) — {r['genre']}" for r in rows)


# ---------------------------------------------------------------------------
# Known entities for lightweight NL → entity extraction
# ---------------------------------------------------------------------------

KNOWN_ACTORS = [
    "Keanu Reeves", "Tom Hanks", "Denzel Washington", "Leonardo DiCaprio",
    "Christian Bale", "Laurence Fishburne", "Carrie-Anne Moss", "Hugo Weaving",
    "Robin Wright", "Gary Sinise", "Helen Hunt", "Sandra Bullock", "Dennis Hopper",
    "Michael Nyqvist", "Jason Robards", "Ethan Hawke", "Heath Ledger",
    "Aaron Eckhart", "Joseph Gordon-Levitt", "Elliot Page", "Kate Winslet",
    "Catherine Zeta-Jones", "Matt Damon", "Mykelti Williamson",
]

KNOWN_MOVIES = [
    "The Matrix", "Forrest Gump", "Cast Away", "Speed", "John Wick",
    "Philadelphia", "Training Day", "The Dark Knight", "Inception", "Titanic",
    "The Terminal", "Saving Private Ryan",
]

KNOWN_DIRECTORS = [
    "Lana Wachowski", "Robert Zemeckis", "Jan de Bont", "Chad Stahelski",
    "Jonathan Demme", "Antoine Fuqua", "Christopher Nolan", "James Cameron",
    "Steven Spielberg",
]


def _find_entity(query: str, entity_list: List[str]) -> Optional[str]:
    """Return the first known entity whose name appears in the query (case-insensitive)."""
    q_lower = query.lower()
    for entity in entity_list:
        if entity.lower() in q_lower:
            return entity
    return None


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def graph_retrieve(query: str) -> str:
    """
    Analyse a natural-language query and dispatch to the appropriate
    Cypher retrieval function.  Returns a human-readable string
    containing the graph results.
    """
    q = query.lower()

    # ── Intent signals ──────────────────────────────────────────────────────
    asks_co_actors      = any(w in q for w in ["acted with", "worked with", "co-act",
                                                "who else", "same movie as", "co-star"])
    asks_director_query = any(w in q for w in ["who directed", "director of", "directed", "direct", "director"])
    asks_movies_director= any(w in q for w in ["movies by", "films by", "directed by", "movies did", "films did"]) or (any(w in q for w in ["direct", "director", "directed"]) and _find_entity(query, KNOWN_DIRECTORS) is not None)
    asks_movies_actor   = any(w in q for w in ["movies", "films", "appeared", "starred",
                                                "acted in", "filmography"])
    asks_cast           = any(w in q for w in ["cast", "who acted", "who starred",
                                                "who is in", "actors in"])

    # ── Route to Cypher function ────────────────────────────────────────────

    # 1. Co-actor multi-hop
    if asks_co_actors:
        actor = _find_entity(query, KNOWN_ACTORS)
        if actor:
            return get_co_actors(actor)

    # 2. Which directors directed a specific actor
    if asks_director_query and not asks_movies_director:
        actor = _find_entity(query, KNOWN_ACTORS)
        if actor:
            return get_directors_of_actor(actor)

    # 3. All movies by a director
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

    # 5. Movies by a specific actor
    if asks_movies_actor:
        actor = _find_entity(query, KNOWN_ACTORS)
        if actor:
            return get_movies_by_actor(actor)

    # 6. Cast of a specific movie
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


# ---------------------------------------------------------------------------
# Entry point — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        "Who acted in The Matrix?",
        "What movies has Tom Hanks appeared in?",
        "Who directed Inception?",
        "Who has Keanu Reeves acted with?",
        "What movies did Christopher Nolan direct?",
        "Which directors has Denzel Washington worked with?",
    ]

    print("Graph retriever smoke test\n" + "=" * 50)
    for q in tests:
        print(f"\nQ: {q}")
        print(graph_retrieve(q))
