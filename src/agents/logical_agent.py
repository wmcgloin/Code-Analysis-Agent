"""
Logical Agent for analyzing code repositories and creating high-level logical relationship graphs.
This agent focuses on creating a simplified abstraction of the codebase with minimal nodes.
"""

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from pyvis.network import Network


class LogicalAgent:
    """
    Agent that analyzes code repositories and creates logical relationship graphs
    showing how files, components, modules, and concepts correlate to each other.
    This agent aims to create a high-level abstraction with minimal nodes.
    """

    def __init__(self, model_name="gpt-4o-mini"):
        """Initialize the logical agent with the specified LLM model."""
        self.llm = ChatOpenAI(model=model_name)
        self.system_prompt = """
        You are an expert in analyzing codebases and generating high-level abstractions of their logical structure.
        Your task is to create a simplified graph representation with AS FEW nodes as possible (aim for less than 10 nodes, but more than 4 nodes for small codebases).
        
        Focus on identifying the main components, modules, or concepts in the codebase and how they relate to each other logically.
        
        Only use the list of types provided below:
        - component: major functional units of the codebase (e.g., "authentication", "data_processing", "visualization")
        - module: logical groupings of functionality (can span multiple files)
        - concept: abstract ideas or patterns implemented in the code
        
        When generating the structured natural language description, follow these rules:    
        - Focus on high-level relationships and dependencies
        - Identify the core components and how they interact
        - Group related functionality into logical modules
        - Identify key concepts that span multiple files
        - Aim for a minimal representation that captures the essence of the codebase
        - Do not include implementation details
        - Do not describe individual functions or methods unless they represent a core concept
        - Prioritize clarity and simplicity over completeness
                
        Natural language should follow a similar format as below:
            {source.id} is a {source.type} that {relationship} {target.id} which is a {target.type}.
            
        Example:
        - "authentication is a component that depends_on database which is a component."
        - "data_processing is a module that implements MapReduce which is a concept."
        - "visualization is a component that uses data_processing which is a module."
        """

        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=["component", "module", "concept"],
        )

    def generate_repo_tree(self, repo_path: str) -> str:
        """Generate a tree representation of the repository structure."""
        tree_string = ""
        for root, dirs, files in os.walk(repo_path):
            # Filter out __pycache__ and hidden directories
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            files = [f for f in files if not f.startswith(".")]

            level = root.replace(repo_path, "").count(os.sep)
            indent = "│   " * level + "├── "  # Formatting the tree
            tree_string += f"{indent}{os.path.basename(root)}/\n"

            sub_indent = "│   " * (level + 1) + "├── "
            for file in files:
                tree_string += f"{sub_indent}{file}\n"

        return tree_string

    def analyze_codebase(self, repo_path: str) -> List[Dict[str, Any]]:
        """
        Analyze the codebase and extract logical relationships at a high level.

        Args:
            repo_path: Path to the repository to analyze

        Returns:
            List of graph documents representing the logical relationships
        """
        repo_tree_string = self.generate_repo_tree(repo_path)

        # For logical analysis, we'll analyze the entire codebase at once
        # to get a high-level overview rather than file-by-file

        # First, collect all Python files
        python_files = {}
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

            for file in files:
                if file.endswith(".py") and not file.startswith("."):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, repo_path)

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            python_files[relative_path] = f.read()
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        # Create a summary of the codebase
        file_summaries = []
        for file_path, content in python_files.items():
            file_summaries.append(f"File: {file_path}\n\n{content[:500]}...\n\n")

        codebase_summary = "\n".join(file_summaries)

        # Get logical description of the entire codebase
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=f"""
                Repository Tree:
                {repo_tree_string}
                
                Codebase Summary (first 500 chars of each file):
                {codebase_summary}
                
                Create a high-level logical representation of this codebase with minimal nodes (aim for 5 or fewer).
                Focus on the main components, modules, and concepts and how they relate to each other.
            """
            ),
        ]

        response = self.llm.invoke(messages)
        logical_description = response.content

        # Convert to document
        document = Document(
            page_content=logical_description, metadata={"source": "logical_overview"}
        )

        # Convert to graph documents
        graph_documents = self.llm_transformer.convert_to_graph_documents([document])

        return graph_documents

    def create_visualization(self, graph_documents, output_path="logical_graph.html"):
        """
        Create a visualization of the logical relationships.

        Args:
            graph_documents: List of graph documents
            output_path: Path to save the visualization
        """
        # Create Pyvis network
        net = Network(
            notebook=True, cdn_resources="in_line", height="800px", width="100%"
        )

        # Create a NetworkX graph
        G = nx.Graph()

        # Dictionary to track unique nodes and metadata
        node_metadata = {}

        # Helper to create a hashable key from type, id, and properties
        def make_node_key(node):
            return (node.type, node.id, tuple(sorted(node.properties.items())))

        # Add nodes and edges
        for graph in graph_documents:
            for rel in graph.relationships:
                # Get full, unique node keys
                source_key = make_node_key(rel.source)
                target_key = make_node_key(rel.target)
                rel_type = rel.type

                # Store node metadata
                node_metadata[source_key] = {
                    "id": rel.source.id,
                    "type": rel.source.type,
                    "properties": rel.source.properties,
                }
                node_metadata[target_key] = {
                    "id": rel.target.id,
                    "type": rel.target.type,
                    "properties": rel.target.properties,
                }

                # Add nodes and edges
                G.add_node(source_key)
                G.add_node(target_key)
                G.add_edge(source_key, target_key, label=rel_type)

        # Get unique types for coloring
        unique_types = list(set(meta["type"] for meta in node_metadata.values()))
        color_map = plt.get_cmap("tab10")
        type_colors = {
            t: color_map(i / len(unique_types)) for i, t in enumerate(unique_types)
        }
        type_colors_rgba = {
            t: f"rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 0.8)"
            for t, c in type_colors.items()
        }

        # Degree-based sizing
        degrees = dict(G.degree())
        min_size, max_size = 20, 60  # Larger nodes for the logical graph
        max_degree = max(degrees.values()) if degrees else 1
        size_scale = {
            node: min_size + (max_size - min_size) * (deg / max_degree)
            for node, deg in degrees.items()
        }

        # Add nodes to Pyvis
        for node_key in G.nodes():
            metadata = node_metadata[node_key]
            label = metadata["id"]
            node_type = metadata["type"]
            properties = metadata.get("properties", {})
            color = type_colors_rgba.get(node_type, "gray")

            # Property display
            props_html = (
                "<br>".join(f"{k}: {v}" for k, v in properties.items())
                if properties
                else "No properties"
            )

            net.add_node(
                str(node_key),  # string key for Pyvis
                label=label,
                size=size_scale[node_key],
                color=color,
                title=f"<b>{node_type}</b> ({label})<br>{props_html}",
            )

        # Add edges
        for source, target, attr in G.edges(data=True):
            rel_label = attr.get("label", "")
            net.add_edge(str(source), str(target), title=rel_label, label=rel_label)

        # Save graph
        net.save_graph(output_path)

        # Build legend
        legend_html = """
        <div id="legend" style="position: absolute; top: 10px; left: 10px; background: white; padding: 10px; border-radius: 8px; box-shadow: 0px 0px 5px rgba(0,0,0,0.2); font-family: Arial, sans-serif; z-index: 1000;">
            <h4 style="margin: 0; padding-bottom: 5px;">Node Legend</h4>
        """

        for node_type, color in type_colors_rgba.items():
            legend_html += f'<div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 15px; height: 15px; background:{color}; margin-right: 5px; border-radius: 50%;"></div> {node_type}</div>'

        legend_html += "</div>"

        # Inject legend
        with open(output_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        html_content = html_content.replace("</body>", legend_html + "</body>")

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(html_content)

        print(f"Logical graph saved as {output_path}")

        return output_path
