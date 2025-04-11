# Code Analysis Multi-Agent Workflow

This project implements a multi-agent workflow for analyzing code repositories. It consists of a supervisor agent and two specialized agents:

1. **Semantic Agent**: Creates detailed semantic relationship graphs showing how classes, methods, and functions correlate to each other.
2. **Logical Agent**: Creates high-level logical relationship graphs with minimal nodes (fewer than 5 for small codebases) for a quick overview.

The supervisor agent decides which specialized agent to use based on the user's query.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
python multi_agent_workflow.py /path/to/repository "I want a detailed analysis of the code structure"
```

### Web Interface

```bash
python app.py
```

Then open your browser and navigate to the URL displayed in the terminal (typically http://127.0.0.1:7860).

## Project Structure

```
src/
├── agents/
│   ├── __init__.py
│   ├── semantic_agent.py
│   ├── logical_agent.py
│   └── supervisor.py
├── multi_agent_workflow.py
├── app.py
requirements.txt
README.md
```

## Example

```python
from multi_agent_workflow import CodeAnalysisWorkflow

# Initialize the workflow
workflow = CodeAnalysisWorkflow(model_name="gpt-4o-mini")

# Analyze a repository
result = workflow.analyze(
    repo_path="/path/to/repository",
    query="I want a high-level overview of the codebase structure",
)

print(f"Analysis completed using {result['agent_type']}")
print(f"Output saved to: {result['output_path']}")
```

## Output

The workflow generates HTML files with interactive visualizations of the code structure:

- `./output/semantic_graph.html`: Detailed semantic relationship graph
- `./output/logical_graph.html`: High-level logical relationship graph
