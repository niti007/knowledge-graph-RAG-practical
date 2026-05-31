"""
config.py — Loads environment variables for the Hybrid Graph RAG lab.
"""

import os
from dotenv import load_dotenv

# Load .env from the same directory as this file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "lab1")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "eduhubspot")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Set key globally so LangChain sub-libraries pick it up automatically
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. "
        "Add it to the .env file or export it as an environment variable."
    )

if __name__ == "__main__":
    print("Configuration loaded:")
    print(f"  NEO4J_URI  : {NEO4J_URI}")
    print(f"  NEO4J_USER : {NEO4J_USER}")
    print(f"  OPENAI_KEY : {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}")
