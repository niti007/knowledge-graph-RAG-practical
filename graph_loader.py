"""
graph_loader.py — Populates Neo4j with a sample movies dataset.

Run directly to load data:
    python graph_loader.py
"""

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# ---------------------------------------------------------------------------
# Sample dataset: 12 movies with actors and directors
# ---------------------------------------------------------------------------
MOVIES_DATA = [
    {
        "title": "The Matrix",
        "year": 1999,
        "genre": "Sci-Fi",
        "description": (
            "A computer hacker learns from mysterious rebels about the true nature "
            "of his reality and his role in the war against its controllers. "
            "A mind-bending journey into simulated reality."
        ),
        "actors": ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss", "Hugo Weaving"],
        "director": "Lana Wachowski",
    },
    {
        "title": "Forrest Gump",
        "year": 1994,
        "genre": "Drama",
        "description": (
            "The presidencies of Kennedy and Johnson, the Vietnam War, and other "
            "historical events unfold from the perspective of an Alabama man with "
            "an unusually kind heart and a simple view of the world."
        ),
        "actors": ["Tom Hanks", "Robin Wright", "Gary Sinise", "Mykelti Williamson"],
        "director": "Robert Zemeckis",
    },
    {
        "title": "Cast Away",
        "year": 2000,
        "genre": "Drama",
        "description": (
            "A FedEx executive must transform himself physically and emotionally "
            "to survive a crash landing on a deserted island. A story of solitude, "
            "survival, and human resilience."
        ),
        "actors": ["Tom Hanks", "Helen Hunt", "Nick Searcy"],
        "director": "Robert Zemeckis",
    },
    {
        "title": "Speed",
        "year": 1994,
        "genre": "Action",
        "description": (
            "A young police officer must prevent a bomb exploding aboard a city bus "
            "by keeping its speed above 50 mph. A high-octane thriller with "
            "non-stop action."
        ),
        "actors": ["Keanu Reeves", "Sandra Bullock", "Dennis Hopper"],
        "director": "Jan de Bont",
    },
    {
        "title": "John Wick",
        "year": 2014,
        "genre": "Action",
        "description": (
            "An ex-hitman comes out of retirement to track down the gangsters who "
            "killed his dog, a final gift from his deceased wife. A stylish and "
            "relentless action film."
        ),
        "actors": ["Keanu Reeves", "Michael Nyqvist", "Alfie Allen"],
        "director": "Chad Stahelski",
    },
    {
        "title": "Philadelphia",
        "year": 1993,
        "genre": "Drama",
        "description": (
            "A man is wrongfully dismissed from his law firm because he has AIDS. "
            "He hires a homophobic small-time lawyer as his only willing advocate "
            "in a landmark civil rights case."
        ),
        "actors": ["Tom Hanks", "Denzel Washington", "Jason Robards"],
        "director": "Jonathan Demme",
    },
    {
        "title": "Training Day",
        "year": 2001,
        "genre": "Crime",
        "description": (
            "On his first day as a Los Angeles narcotics officer, a rookie cop "
            "goes beyond a full work day alongside a rogue detective who is not "
            "what he seems. A tense crime thriller about corruption."
        ),
        "actors": ["Denzel Washington", "Ethan Hawke", "Scott Glenn"],
        "director": "Antoine Fuqua",
    },
    {
        "title": "The Dark Knight",
        "year": 2008,
        "genre": "Action",
        "description": (
            "When the Joker wreaks havoc and chaos on Gotham, Batman must accept "
            "one of the greatest psychological tests of his ability to fight "
            "injustice. A superhero film exploring the nature of good and evil."
        ),
        "actors": ["Christian Bale", "Heath Ledger", "Aaron Eckhart", "Maggie Gyllenhaal"],
        "director": "Christopher Nolan",
    },
    {
        "title": "Inception",
        "year": 2010,
        "genre": "Sci-Fi",
        "description": (
            "A thief who steals corporate secrets through the use of dream-sharing "
            "technology is given the inverse task of planting an idea into the "
            "mind of a CEO. A complex, layered narrative about dreams within dreams."
        ),
        "actors": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page", "Ken Watanabe"],
        "director": "Christopher Nolan",
    },
    {
        "title": "Titanic",
        "year": 1997,
        "genre": "Romance",
        "description": (
            "A seventeen-year-old aristocrat falls in love with a kind but poor "
            "artist aboard the luxurious, ill-fated RMS Titanic. An epic romantic "
            "disaster film set against a historical backdrop."
        ),
        "actors": ["Leonardo DiCaprio", "Kate Winslet", "Billy Zane"],
        "director": "James Cameron",
    },
    {
        "title": "The Terminal",
        "year": 2004,
        "genre": "Drama",
        "description": (
            "An Eastern European tourist is stranded in John F. Kennedy Airport "
            "after his homeland erupts in a coup. He must live in the terminal "
            "while bureaucracy prevents him from entering the United States."
        ),
        "actors": ["Tom Hanks", "Catherine Zeta-Jones", "Stanley Tucci"],
        "director": "Steven Spielberg",
    },
    {
        "title": "Saving Private Ryan",
        "year": 1998,
        "genre": "War",
        "description": (
            "Following the Normandy Landings, a group of U.S. soldiers go behind "
            "enemy lines to retrieve a paratrooper whose brothers have been killed "
            "in action. A harrowing portrayal of World War II combat."
        ),
        "actors": ["Tom Hanks", "Matt Damon", "Tom Sizemore"],
        "director": "Steven Spielberg",
    },
]


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def get_driver():
    """Return an authenticated Neo4j driver."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def clear_database(driver):
    """Remove all nodes and relationships from the database."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("  [OK] Database cleared.")


def load_movies(driver):
    """
    Create Movie and Person nodes, then add ACTED_IN and DIRECTED
    relationships from MOVIES_DATA.
    """
    with driver.session() as session:
        for movie in MOVIES_DATA:
            # Upsert Movie node
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

            # Upsert Person nodes + ACTED_IN edges
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

            # Upsert Director node + DIRECTED edge
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

    print(f"  [OK] Loaded {len(MOVIES_DATA)} movies into Neo4j.")


def verify_data(driver) -> tuple[int, int]:
    """Print a summary of graph statistics and return (movies, persons)."""
    with driver.session() as session:
        movie_count    = session.run("MATCH (m:Movie)          RETURN count(m) AS c").single()["c"]
        person_count   = session.run("MATCH (p:Person)         RETURN count(p) AS c").single()["c"]
        acted_count    = session.run("MATCH ()-[r:ACTED_IN]->() RETURN count(r) AS c").single()["c"]
        directed_count = session.run("MATCH ()-[r:DIRECTED]->() RETURN count(r) AS c").single()["c"]

    print(f"  Movies   : {movie_count}")
    print(f"  Persons  : {person_count}")
    print(f"  ACTED_IN : {acted_count} relationships")
    print(f"  DIRECTED : {directed_count} relationships")
    return movie_count, person_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Connecting to Neo4j ...")
    driver = get_driver()
    driver.verify_connectivity()
    print("  [OK] Connected.\n")

    print("Clearing existing data ...")
    clear_database(driver)

    print("Loading movie dataset ...")
    load_movies(driver)

    print("\nGraph statistics after load:")
    verify_data(driver)

    driver.close()
    print("\nDone!")
