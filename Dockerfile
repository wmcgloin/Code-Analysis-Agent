FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . .

RUN uv venv
RUN uv sync --locked

# Add venv packages to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Streamlit port
EXPOSE 8501

# These should be overridden at runtime
ENV NEO4J_URI=bolt://neo4j:7687
ENV NEO4J_USERNAME=neo4j
ENV NEO4J_PASSWORD=password
ENV OPENAI_API_KEY=your_openai_key

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.address=0.0.0.0"]

