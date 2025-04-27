# Code Analysis Agent

A LangGraph + Streamlit application for interactive **codebase analysis** and **querying**.

This project builds a system that:
- **Analyzes** Python codebases into semantic graphs (using LLMs and Neo4j)
- **Visualizes** module and function relationships interactively
- **Queries** the codebase using an agentic Graph + Retrieval-Augmented Generation (RAG) workflow
- **Generates macro-level summaries** and **mermaid diagrams** of overall code structure

## Project Structure

```
src/
├── agent_router/         # LangGraph agent router (multi-agent setup)
│   ├── graph.py
│   ├── nodes.py
│   └── state.py
├── app/                  # Streamlit app and session management
│   ├── graph_builders/    # Codebase graph construction
│   │   └── micro_graph_builder.py
│   ├── handlers/          # Query handling
│   │   └── query_handler.py
│   ├── setup/             # Database initialization
│   │   └── initialize_database.py
│   ├── ui/                # Frontend (chat, visualization)
│   │   ├── chat.py
│   │   └── visualization.py
│   └── session_state.py   # Streamlit session management
├── rag/                  # Retrieval-augmented generation (RAG) setup
│   └── vector_rag.py
├── tools/                # Tool modules for LangGraph agents
│   ├── macro/             # Macro agent tools (summarization, mermaid diagram generation)
│   │   └── tools.py
│   ├── micro/             # Micro agent tools (Cypher query building, visualization)
│   │   ├── cypher_query_builder.py
│   │   ├── cypher_visualizer.py
│   │   └── tools.py
│   └── setup.py
├── utils/                # Utility functions
│   ├── filesystem.py
│   ├── logger.py
│   └── repo.py
├── streamlit_app.py       # Main Streamlit application entrypoint
```

## Key Features

- **Multi-stage code analysis:**
  - Extracts natural language descriptions of classes, functions, and modules.
  - Converts codebases into semantic graph structures stored in Neo4j.
  - Builds a RAG system for flexible natural language querying.

- **LangGraph Multi-Agent Router:**
  - Routes queries dynamically between *micro-level* (fine-grained) and *macro-level* (big picture) tools.
  - Micro agent supports Cypher relationship retrieval, visualization, and RAG querying.
  - Macro agent generates text summaries and mermaid diagrams for architecture overviews.

- **Interactive Streamlit Interface:**
  - Clone GitHub repositories.
  - Select and analyze source folders.
  - Visualize live code graphs.
  - Chat with your codebase using flexible agents.

## Getting Started

### Option 1: Local Installation

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

### Option 2: Docker Deployment

1. Make sure you have Docker and Docker Compose installed on your system.

2. Set up environment variables:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_key
```

3. Build and start the containers:
```bash
docker-compose up -d
```

4. Access the application at http://localhost:8501

## Notes
- Requires a running Neo4j instance.
- Assumes Python 3.10+.
- Optimized for small-to-medium Python codebases (<10k files).
- OpenAI API (GPT-4o) is used for LLM-based analysis.

## Future Extensions
- **Visualization Improvements:** Enhance styling, interactivity, and responsiveness of the graph visualizations.
- **Assorted Usability Improvements:** Better error handling, more flexible file selection, support for additional file types beyond .py.

---

Built with ❤️ using LangChain, LangGraph, Neo4j, and Streamlit.
