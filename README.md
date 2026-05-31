# Knowledge Graph RAG Practical

A hybrid Retrieval-Augmented Generation (RAG) system that combines **Neo4j Graph Database** and **FAISS Vector Search** to intelligently route queries between structured graph traversal and semantic vector search.

This project demonstrates how to build an intelligent movie assistant that understands both factual relationships (who acted in what?) and semantic queries (find me a romantic movie to watch).

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    USER QUERY INPUT                             │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│               QUERY ROUTER (router.py)                          │
│  • Analyzes query intent                                        │
│  • Classifies as GRAPH or VECTOR retrieval task                │
│  • Routes to appropriate retriever                             │
└────────────────────────────────────────────────────────────────┘
        │                                           │
        │ (Structured/Relational)                   │ (Semantic/Descriptive)
        ▼                                           ▼
    ┌─────────────────────┐           ┌──────────────────────────┐
    │  GRAPH RETRIEVER    │           │  VECTOR RETRIEVER        │
    │  (graph_retriever)  │           │  (vector_store.py)       │
    └─────────────────────┘           └──────────────────────────┘
        │                                       │
        │                                       │
        ▼                                       ▼
    ┌─────────────────────┐           ┌──────────────────────────┐
    │  NEO4J GRAPH DB     │           │  FAISS INDEX             │
    │  • Movie nodes      │           │  • Movie embeddings      │
    │  • Person nodes     │           │  • Vector search         │
    │  • ACTED_IN edges   │           │  • Similarity ranking    │
    │  • DIRECTED edges   │           └──────────────────────────┘
    │                     │
    │  Cypher Query ──►   │           
    │  Multi-hop traversal│
    └─────────────────────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────┐
        │   LLM CHAIN (rag_chain.py)          │
        │   • Generate final answer           │
        │   • Context from retriever          │
        │   • Response formatting             │
        └─────────────────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────┐
        │   FINAL ANSWER TO USER              │
        │   + Retriever source information    │
        └─────────────────────────────────────┘
```

---

## 📋 System Components

### **1. Configuration Management** (`config.py`)
- Loads environment variables from `.env` file
- Validates Neo4j and OpenAI API credentials
- Sets up global configuration for all modules

### **2. Graph Loader** (`graph_loader.py`)
- Populates Neo4j with sample movie dataset (12 movies)
- Creates Movie and Person nodes
- Establishes ACTED_IN and DIRECTED relationships
- Provides database verification utilities

### **3. Vector Store** (`vector_store.py`)
- Initializes FAISS index for vector search
- Generates embeddings using OpenAI's embedding model
- Performs similarity search on movie descriptions
- Returns ranked results based on semantic relevance

### **4. Graph Retriever** (`graph_retriever.py`)
- Executes Cypher queries against Neo4j
- Handles single-hop queries (direct relationships)
- Supports multi-hop traversal (find collaborators)
- Formats results for LLM consumption

### **5. Query Router** (`router.py`)
- Analyzes incoming queries
- Classifies intent (graph vs. vector)
- Routes to appropriate retriever
- Uses LLM for intelligent decision-making

### **6. RAG Chain** (`rag_chain.py`)
- Orchestrates the entire RAG pipeline
- Retrieves context from router
- Generates final answer using LLM
- Returns answer with source attribution

### **7. Main CLI** (`main.py`)
- Interactive query mode (default)
- Data loading mode (`--load`)
- Evaluation/benchmark mode (`--eval`)
- Demo mode (`--demo`)

### **8. Evaluation** (`evaluate.py`)
- Benchmarks system performance
- Tests accuracy and latency
- Provides metrics and statistics

---

## 🎯 Query Routing Logic

The system intelligently routes queries:

| Query Type | Example | Retriever | Method |
|-----------|---------|-----------|--------|
| **Factual/Relational** | "Who acted in The Matrix?" | Graph | Cypher query + Neo4j |
| **Multi-hop** | "Who has Tom Hanks worked with?" | Graph | Graph traversal |
| **Semantic** | "Recommend a romantic movie" | Vector | FAISS similarity search |
| **Descriptive** | "Find me a sci-fi film about dreams" | Vector | Semantic embedding match |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Neo4j instance (local or cloud)
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/niti007/knowledge-graph-RAG-practical.git
   cd knowledge-graph-RAG-practical
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create `.env` file**
   ```bash
   cp .env.example .env
   # or manually create .env with:
   ```
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   OPENAI_API_KEY=sk-your-api-key-here
   ```

   > **Neo4j Setup Guide**: If you don't have Neo4j installed:
   > - **Local**: Install from [neo4j.com/download](https://neo4j.com/download/)
   > - **Docker**: `docker run --publish=7687:7687 --publish=7474:7474 neo4j:latest`
   > - **Cloud**: Use [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/)

4. **Load the dataset**
   ```bash
   python main.py --load
   ```
   
   Expected output:
   ```
   Connecting to Neo4j ...
   [OK] Connected.
   
   Clearing existing data ...
   [OK] Database cleared.
   
   Loading movie dataset ...
   [OK] Loaded 12 movies into Neo4j.
   
   Graph statistics after load:
   Movies   : 12
   Persons  : 31
   ACTED_IN : 39 relationships
   DIRECTED : 12 relationships
   ```

---

## 💻 Usage

### Interactive Mode (Default)
```bash
python main.py
```

**Example queries:**
```
You: Who acted in The Matrix?
Answer: Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving starred in The Matrix.
Source: GRAPH retrieval

You: Recommend a romantic movie to watch tonight.
Answer: Titanic is a great romantic choice, featuring an epic romance story...
Source: VECTOR retrieval

You: Who has Tom Hanks worked with across his career?
Answer: Tom Hanks has worked with multiple directors and actors including...
Source: GRAPH retrieval (multi-hop)
```

Commands:
- `help` - Show available query examples
- `quit` / `exit` - Exit the program
- `q` - Shortcut for quit

### Demo Mode
Runs 5 pre-set sample queries showcasing both retrieval paths:
```bash
python main.py --demo
```

### Data Loading
Reload/refresh the movie dataset:
```bash
python main.py --load
```

### Evaluation/Benchmarking
Run performance benchmarks:
```bash
python main.py --eval
```

---

## 📊 Dataset Overview

The system comes with a curated dataset of **12 movies** featuring:

| Movie | Year | Genre | Director | Cast |
|-------|------|-------|----------|------|
| The Matrix | 1999 | Sci-Fi | Lana Wachowski | Keanu Reeves, Laurence Fishburne, ... |
| Forrest Gump | 1994 | Drama | Robert Zemeckis | Tom Hanks, Robin Wright, Gary Sinise |
| The Dark Knight | 2008 | Action | Christopher Nolan | Christian Bale, Heath Ledger, Aaron Eckhart |
| Inception | 2010 | Sci-Fi | Christopher Nolan | Leonardo DiCaprio, Joseph Gordon-Levitt |
| Titanic | 1997 | Romance | James Cameron | Leonardo DiCaprio, Kate Winslet |

**Graph Structure:**
- 12 Movie nodes
- 31 Person nodes
- 39 ACTED_IN relationships
- 12 DIRECTED relationships

---

## 📦 Dependencies

```
neo4j>=5.0.0              # Graph database driver
langchain>=0.3.0          # LLM orchestration framework
langchain-openai>=0.2.0   # OpenAI integration
langchain-community>=0.3.0
langchain-core>=0.3.0
openai>=1.0.0             # OpenAI API
pydantic>=2.0.0           # Data validation
python-dotenv>=1.0.0      # Environment variables
faiss-cpu>=1.7.0          # Vector search
```

---

## 🔧 Configuration

Edit these variables in your `.env` file:

```env
# Neo4j Connection
NEO4J_URI=bolt://localhost:7687        # Graph database URL
NEO4J_USER=neo4j                        # Database username
NEO4J_PASSWORD=password                 # Database password

# OpenAI API
OPENAI_API_KEY=sk-...                   # Your API key
```

---

## 📚 Project Structure

```
knowledge-graph-RAG-practical/
├── main.py                 # CLI entry point
├── config.py               # Configuration loader
├── graph_loader.py         # Dataset loader + Neo4j setup
├── graph_retriever.py      # Cypher query executor
├── vector_store.py         # FAISS vector search
├── router.py               # Query intent classifier
├── rag_chain.py            # RAG pipeline orchestrator
├── evaluate.py             # Benchmarking suite
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## 🔍 How It Works: Step-by-Step

### Example: "Who acted in The Matrix?"

1. **Query Input** → User asks about The Matrix
2. **Router Analysis** → Detects factual/relational query
3. **Route Decision** → Routes to GRAPH retriever
4. **Cypher Execution** → 
   ```cypher
   MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: "The Matrix"})
   RETURN p.name
   ```
5. **Result Processing** → Gets [Keanu Reeves, Laurence Fishburne, ...]
6. **LLM Generation** → Formats answer: "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, and Hugo Weaving starred in The Matrix."
7. **Response** → Returns answer with source="GRAPH"

### Example: "Recommend a romantic movie"

1. **Query Input** → User asks for romantic movie
2. **Router Analysis** → Detects semantic/preference query
3. **Route Decision** → Routes to VECTOR retriever
4. **Embedding Generation** → Converts query to embedding
5. **Similarity Search** → Searches FAISS index
6. **Top Results** → Returns most similar movies (e.g., Titanic, Philadelphia)
7. **LLM Generation** → Generates personalized recommendation
8. **Response** → Returns answer with source="VECTOR"

---

## 🛠️ Development

### Adding New Movies

Edit `MOVIES_DATA` in `graph_loader.py` and run:
```bash
python main.py --load
```

### Debugging

Enable verbose mode (already active in interactive mode):
```python
result = run_hybrid_rag("your query", verbose=True)
```

### Testing Custom Queries

In Python:
```python
from rag_chain import run_hybrid_rag

result = run_hybrid_rag("Who acted in The Matrix?")
print(result['answer'])        # Final answer
print(result['retriever'])     # GRAPH or VECTOR
print(result['context'])       # Raw retrieval results
```

---

## 📈 Performance Benchmarks

Run evaluation suite:
```bash
python main.py --eval
```

Metrics tracked:
- **Query latency** (ms)
- **Retriever routing accuracy**
- **Answer quality**
- **Cache hit rates**

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Neo4j connection failed" | Check NEO4J_URI, user, password in .env |
| "OPENAI_API_KEY not set" | Add your API key to .env |
| "Module not found" | Run `pip install -r requirements.txt` |
| "FAISS import error" | Run `pip install faiss-cpu` (CPU) or `faiss-gpu` (GPU) |
| "No results from graph" | Run `python main.py --load` to populate database |

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🎓 Learning Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [LangChain Guide](https://python.langchain.com/)
- [FAISS Tutorial](https://github.com/facebookresearch/faiss)
- [OpenAI API Docs](https://platform.openai.com/docs/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)

---

## 👨‍💻 Author

Created by **niti007** as a practical exploration of hybrid RAG systems.

---

## ⭐ Acknowledgments

- Neo4j team for the graph database
- OpenAI for language models
- Meta AI for FAISS vector search
- LangChain community for orchestration framework
