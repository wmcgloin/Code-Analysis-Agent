# Code Analysis Agent

A LangGraph + Streamlit application for interactive **codebase analysis** and **querying**.

This project builds a system that:
- **Analyzes** Python codebases into semantic graphs (using LLMs and Neo4j)
- **Visualizes** module and function relationships interactively
- **Queries** the codebase using an agentic Graph + Retrieval-Augmented Generation (RAG) workflow

## Project Structure

```
src/
├── agent_router/         # LangGraph agent router
│   ├── graph.py
│   ├── nodes.py
│   ├── state.py
├── app/                  # Streamlit app logic
│   ├── graph_builders/    # Codebase graph construction
│   │   └── micro_graph_builder.py
│   ├── handlers/         # Query handling and processing
│   │   └── query_handler.py
│   ├── setup/            # Database initialization utilities
│   │   └── initialize_database.py
│   ├── ui/               # Frontend (chat, visualization)
│   │   ├── chat.py
│   │   └── visualization.py
│   └── session_state.py  # Session state management
├── rag/                  # RAG system setup
│   └── vector_rag.py
├── tools/                # Tools for LangGraph agents
│   ├── micro/            # Micro agent tools (Cypher, RAG)
│   │   ├── cypher_query_builder.py
│   │   ├── cypher_visualizer.py
│   │   └── tools.py
│   ├── macro/            # (reserved for future macro agent tools)
│   │   └── tools.py
├── utils/                # Utilities
│   ├── filesystem.py
│   ├── logger.py
│   └── repo.py
├── streamlit_app.py       # Main Streamlit application entrypoint
```

## Key Features

- **Multi-stage code analysis:**
  - Natural language descriptions of Python code.
  - Conversion to graph structures stored in Neo4j.
  - RAG system for flexible natural language queries.

- **LangGraph Agent Router:**
  - Routes queries dynamically to micro agent tools.
  - Supports Cypher relationship retrieval, visualization, and free-text explanation.

- **Interactive Streamlit Interface:**
  - Clone GitHub repositories.
  - Select folders to initialize analysis.
  - Visualize code structures live.
  - Chat with your codebase.

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (e.g., Neo4j credentials) in a `.env` file:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_key
```

3. Run the Streamlit app:
```bash
streamlit run src/streamlit_app.py
```

## Notes
- Requires a running Neo4j instance.
- Assumes Python 3.11+.
- Optimized for small-to-medium Python codebases (<10k files).

## Future Extensions
- Add macro-level graph building.
- Improve multi-agent coordination.
- Support additional graph visualizations.

---

Built with ❤️ using LangChain, LangGraph, Neo4j, and Streamlit.