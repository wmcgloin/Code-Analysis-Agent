"""
Macro Agent Tools

This module defines tools for the LangGraph macro agent to:
- Generate textual explanations for code base structure, logical relationships between modules.
- Generate mermaid visualizations from the textual explanations.
"""

import json
# Standard Library
import os
from typing import Annotated, Any, Dict, List

# Third-Party Libraries
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
# Internal Utilities
from utils import get_logger
from utils.filesystem import list_python_files, read_code_file
from utils.repo import generate_repo_tree

# Initialization
# Load environment variables from .env file
load_dotenv()

# Set up module-wide logger
logger = get_logger()

repo_path = ""


@tool
def generate_text_response(question: str):
    """
    Reads in the codebase and generate a short to medium length summary in natural language that captures the main logical modules of the codebase. Output text that is preferably good for translating to mermaid diagrams, and good for reading.

    Args:
        path (str): The path to the codebase

    Returns:
        str: Natural language answer to the predefined question.
    """
    logger.info(f"Generating text response for codebase at: {repo_path}")

    # Initialize the language model
    llm = init_chat_model("gpt-4o", model_provider="openai")

    # Get all Python files in the repository
    python_files = list_python_files(repo_path)

    if not python_files:
        return "No Python files found in the specified path."

    # Generate repository tree structure
    repo_tree = generate_repo_tree(repo_path)

    # Read the content of each Python file (limit to a reasonable number to avoid token limits)
    file_contents = {}
    max_files = 20  # Limit to avoid token limits
    for file_path in python_files[:max_files]:
        full_path = os.path.join(repo_path, file_path)
        content = read_code_file(full_path)
        file_contents[file_path] = content

    # Create a summary of file contents for the LLM
    file_summaries = []
    for file_path, content in file_contents.items():
        # Truncate very large files to avoid token limits
        if len(content) > 2000:
            content = content[:1000] + "\n...[content truncated]...\n" + content[-1000:]

        file_summaries.append(f"File: {file_path}\n```python\n{content}\n```\n")

    # Join file summaries with a limit to avoid token limits
    file_content_text = "\n\n".join(file_summaries)

    # Create the prompt for the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are a code analysis expert. Your task is to analyze a codebase and provide a comprehensive summary of its structure, 
        modules, and relationships. Focus on creating a summary that can be easily translated into a mermaid diagram. Output scale should also take into account on the user's question: {question}
        
        Your analysis should include:
        1. Main modules and their purposes
        2. Logical flow of the application
        3. Entry points and important components
        
        Format your response with clear sections and use bullet points where appropriate. Make sure your description 
        of relationships is clear enough to be parsed for creating a mermaid diagram.""",
            ),
            (
                "human",
                """I need you to analyze this codebase and provide a detailed summary.
        
        Repository structure:
        {repo_tree}
        
        Here are the contents of key files:
        {file_contents}
        
        Please provide a comprehensive analysis that describes the modules, their relationships, and the overall architecture.
        Make sure your description is suitable for generating a mermaid diagram.""",
            ),
        ]
    )

    # Generate the analysis using the LLM
    chain = prompt | llm | StrOutputParser()
    analysis = chain.invoke(
        {"repo_tree": repo_tree, "file_contents": file_content_text}
    )

    logger.info("Text response generated successfully using LLM")
    return analysis


@tool
def generate_mermaid_diagram(text: str):
    """
    Generate a mermaid diagram from the textual explanation generated in `generate_text_response` tool.

    Args:
        text (str): The descriptive text from `generate_text_response`

    Returns:
        dict: Summary statistics or metadata about the rendered visualization.
    """
    logger.info("Generating mermaid diagram from text response")

    # Initialize the language model
    llm = init_chat_model("gpt-4o", model_provider="openai")

    # Create the prompt for the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at converting textual descriptions of code architecture into mermaid diagrams.
        Your task is to analyze the provided text description and create a mermaid diagram that visualizes the described
        modules and their relationships.
        
        Follow these guidelines:
        1. Use the graph TD (top-down) format for the mermaid diagram
        2. Represent each module as a node with an appropriate shape
        3. Use arrows to show relationships between modules
        4. Include a legend if necessary
        5. Keep the diagram clean and readable
        
        Return ONLY the mermaid diagram code without any additional explanation.""",
            ),
            (
                "human",
                """Please convert this textual description of a codebase into a mermaid diagram:
        
        {text}
        
        Generate a mermaid diagram that accurately represents the modules and relationships described above.""",
            ),
        ]
    )

    # Generate the mermaid diagram using the LLM
    chain = prompt | llm | StrOutputParser()
    mermaid_code = chain.invoke({"text": text})

    # Clean up the response to extract just the mermaid code
    if "```mermaid" in mermaid_code:
        mermaid_code = mermaid_code.split("```mermaid")[1]
        if "```" in mermaid_code:
            mermaid_code = mermaid_code.split("```")[0]

    # Ensure the mermaid code starts with graph TD if not already present
    if not mermaid_code.strip().startswith("graph TD"):
        mermaid_code = "graph TD\n" + mermaid_code

    # Format the final mermaid code
    final_mermaid_code = f"```mermaid\n{mermaid_code.strip()}\n```"

    # Save the mermaid diagram to a file
    output_file = "codebase_structure.md"
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, "w") as f:
        f.write(final_mermaid_code)

    logger.info(f"Mermaid diagram saved to {output_file}")

    # Save the rendered HTML file
    output_html_file = "codebase_structure.html"
    if os.path.exists(output_html_file):
        os.remove(output_html_file)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Codebase Structure Diagram</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{ startOnLoad: true }});
    </script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <h1>Codebase Structure Diagram</h1>
    <div class="mermaid">
{mermaid_code.strip()}
    </div>
</body>
</html>
"""

    with open(output_html_file, "w") as f:
        f.write(html_content)

    logger.info(f"Mermaid diagram HTML saved to {output_html_file}")
    # Parse the mermaid code to extract statistics
    node_count = 0
    relationship_count = 0
    modules = []

    for line in mermaid_code.split("\n"):
        line = line.strip()
        # Count nodes (lines with node definitions)
        if (
            "-->" not in line
            and line
            and not line.startswith("graph")
            and not line.startswith("subgraph")
        ):
            node_count += 1
            # Try to extract module name
            if "[" in line and "]" in line:
                module_name = line.split("[")[1].split("]")[0].replace('"', "")
                modules.append(module_name)
        # Count relationships (lines with arrows)
        elif "-->" in line:
            relationship_count += 1

    # Return statistics about the diagram
    return {
        "diagram_type": "mermaid",
        "output_file": output_file,
        "node_count": node_count,
        "relationship_count": relationship_count,
        "modules": modules,
    }
