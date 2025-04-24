"""
Tool configuration module for LangGraph agents.

Defines the toolsets available to the macro and micro agents.
These tools enable agents to perform code analysis, generate visualizations,
and provide textual explanations of code structure and relationships.
"""

from typing import Dict, List
from langchain_core.tools import BaseTool  # If using custom tool classes, adjust accordingly

# Import micro-level tool functions from your local tools module
# (Assumes these are defined with LangChain-compatible interfaces)
import tools.micro_tools as mt
# If macro tools are ever added, they can be imported here similarly:
# from tools.code_analysis_tools import generate_repo_tree, read_code_file, ...

def setup_analysis_tools() -> Dict[str, List[BaseTool]]:
    """
    Initializes and returns toolsets for LangGraph macro and micro agents.

    Macro tools operate at the repository level (e.g., tree structure, file listing).
    Micro tools operate at the file or node level (e.g., relationship querying, explanation).

    Returns:
        A dictionary with two keys:
        - 'macro_tools': list of tools used by the macro agent (can be empty for now)
        - 'micro_tools': list of tools used by the micro agent
    """

    # Placeholder macro tools – these can be populated when repo-level logic is needed
    macro_tools: List[BaseTool] = [
        # Example (commented out until implemented or needed):
        # generate_repo_tree,
        # read_code_file,
        # list_python_files,
        # explain_code_logic,
    ]

    # Micro tools handle specific tasks like querying relationships from a graph DB
    micro_tools: List[BaseTool] = [
        mt.retrieve_cypher_relationships,
        mt.visualize_relationships,
        mt.generate_text_response,
    ]

    return {
        "macro_tools": macro_tools,
        "micro_tools": micro_tools
    }
