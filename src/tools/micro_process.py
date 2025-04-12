"""
Code Analysis Tools for the Micro Agent.

This module provides tools for analyzing code repositories and creating visualizations.
"""

import os
from typing import Annotated, Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_experimental.graph_transformers import LLMGraphTransformer
from pyvis.network import Network

from utils import get_logger
from langchain_neo4j import Neo4jGraph

import os
from dotenv import load_dotenv

# You can pass the path if the file isn't in the same directory
load_dotenv()


logger = get_logger()


@tool
def generate_repo_tree(
    repo_path: Annotated[str, "Path to the repository to analyze"],
) -> str:
    """
    Generate a tree representation of the repository structure.

    Args:
        repo_path: Path to the repository to analyze

    Returns:
        String representation of the repository tree
    """
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

        logger.debug(f"Tool Call - Repo tree generated:\n {tree_string}")
    return tree_string


@tool
def read_code_file(file_path: Annotated[str, "Path to the code file to read"]) -> str:
    """
    Read the content of a code file.

    Args:
        file_path: Path to the code file to read

    Returns:
        Content of the code file
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Tool Call - Error reading file: {file_path}")

        return f"Error reading file: {str(e)}"


@tool
def list_python_files(
    repo_path: Annotated[str, "Path to the repository to analyze"],
) -> List[str]:
    """
    List all Python files in a repository.

    Args:
        repo_path: Path to the repository to analyze

    Returns:
        List of Python file paths
    """
    python_files = []
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for file in files:
            if file.endswith(".py") and not file.startswith("."):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                python_files.append(relative_path)

    logger.debug(f"Tool Call - Python files listed: {python_files}")
    return python_files


class ProcessMicroData:
    """
    Tools for analyzing code repositories and creating visualizations.
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the code analysis tools.

        Args:
            llm: The language model to use for analysis
        """
        self.llm = llm
        self.system_prompt = """
        You are an expert in analyzing Python code and generating structured natural language descriptions for graph-based querying in Cypher. 
        Given a Python codebase, extract meaningful relationships between functions, classes, and imported modules. 
        
        Only use the list of types provided below:
        - class : classes defined in a module
        - method : methods defined in a class
        - function : functions defined in a module
        - module : python scripts defined within the repository. Exclude .py when mentioning the module name.
        - package : packages imported that are not modules.
        Do not include information on variables, parameters, arguments.
        Python scripts must be modules and pre defined packages such as numpy and pandas must be packages

        When generating the structured natural language description, follow these rules:    
        - Do not give explanations for the code logic or functionality.
        - Do not use adjectives and adverbs. 
        - Only describe the code and do not give an overall summary.
        - Do not use ambiguous pronouns and use exact names in every description.
        - Explain each class, function separately and do not include explanations such as 'as mentioned before' or anything that refers to a previous explanation.
        - make each description sufficient for a standalone statement for one relationship in the graph.    
        - Each class and funciton should be connected to the module where it was defined.
        - Each imported package should be connected to the function, method or class where it was used.
        - Always include an explanation on how the outermost class or method is connected to the module where it is defined.
        - If the outermost layer is an 'if __name__ == "__main__":' block, then the outermost layer is whatever is inside the block. Plus whatever is defined outside the block. Make sure to mention the connection between the module and the closses and functions.
        - When mentioning modules, take note of the current file path(relative repository) given in input, and change slashes to dots and remove the .py extension.
        - If a function or class is used in another function or class, make sure to mention the connection between them.
        - 
                
        Natural language should follow a similar format as below:
            {source.id} is a {source.type} with properties {source.properties} defined in {target.id} which is a {target.type}.        
        Example: 
        - When mentioning classes, always refer them as {relative_repository}.{module_name}.{class_name}
        - When mentioning methods, always refer to them as {relative_repository}.{module_name}.{class_name}.{method_name}
        - When mentioning functions, always refer them as {relative_repository}.{module_name}.{function_name}
        - If the file path is deduplication/LSH.py and there is a class LSH in it, the module is deduplication.LSH and the class is deduplication.LSH.LSH.

        Example:
        deduplication.LSHImproved.LSHImproved is a module that defines the class deduplication.LSH.lsh_base, which consists of  method deduplication.LSH.lsh_base.hash_function.
        deduplication.LSH.lsh_base is a class and inherits from the class utils.utils.BaseLSH.
        numpy is a package and is used in the method deduplication.LSH.lsh_base.hash_function.
        bitarray is a package and is used in the deduplication.bloom_filter.BloomFilter_KM_Opt.__init__ method of the class deduplication.bloom_filter.BloomFilter_KM_Opt.  
        deduplication.LSH.lsh_base is a class defined in the module deduplication.LSH.
        
        If a module from our repository is imported in another module in our repository, refer to it as the entire path of the module.
        Example:
        code within deduplication\\__main__.py : from deduplication.LSHImproved import LSHImproved
        Natural Language Description:
        The Class deduplication.LSHImproved.LSHImproved is imported into the module deduplication.__main__.
        """

        self.refine_prompt = """
        You are an expert text editor. Your goal is to modyfy the text in a way that it is consistent with the given repository tree and file path.
        Keep in mind this natural language is meant to be used for graph-based querying in Cypher.
        Only output the modified text, do not give any explanations or anything else.
        The only change you have to make is to modify potential node ids, so they are consistent and can be connected by a graph.
        
        Rules:
        - When mentioning modules, take note of the current file path(relative repository) given in input, and change slashes to dots and remove the .py extension.
        - This means that the current file path or module name should be the beginning of all classes/methods/functions defined in the module.    
        - When mentioning classes, always refer them as {module_location_name}.{class_name}
        - When mentioning methods, always refer to them as {module_location_name}.{class_name}.{method_name}
        - When mentioning functions, always refer them as {module_location_name}.{function_name}

        Example: 
        - if the relative path is deduplication/LSH.py and there is a class LSH in it, the module is deduplication.LSH and the class is deduplication.LSH.LSH.
        - Given the current file path deduplication\\dedup.py
            'deduplication.Baseline is a class defined in deduplication.dedup' is extremely incorrect. 
            The correct description is 'deduplication.dedup.Baseline is a class defined in deduplication.dedup'
        
        
        Example:
        - If the file location is deduplication/LSH.py and there is a class LSH in it, the module should be deduplication.LSH and the class should be deduplication.LSH.LSH. in the natural language description.
        - If the original text is bloom_filter.BloomFilter is a class, and the current file is deduplication\\bloom_filter.py, the modified text should be deduplication.bloom_filter.BloomFilter is a class.
        - If my current module imports a function from a module 'utils.utils import clean_document' the modified text should be utils.utils.clean_document is a function.
    """

        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=["class", "method", "function", "package", "module"],
        )

    def analyze_codebase(self, repo_path: str) -> List[Dict[str, Any]]:
        """
        Analyze the codebase and extract semantic relationships.

        Args:
            repo_path: Path to the repository to analyze

        Returns:
            List of graph documents representing the semantic relationships
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        repo_tree_string = generate_repo_tree(repo_path)
        descriptions = {}

        # Process each file in the repository
        python_files = list_python_files(repo_path)

        for relative_path in python_files:
            print(f"Processing file: {relative_path}")
            file_path = os.path.join(repo_path, relative_path)

            try:
                code_content = read_code_file(file_path)

                # Get initial description
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(
                        content=f"""
                        Tree:
                        {repo_tree_string}

                        Current File Path:
                        {relative_path}

                        Code:
                        {code_content}
                    """
                    ),
                ]

                response = self.llm.invoke(messages)
                descriptions[relative_path] = response.content

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Refine descriptions for consistency
        refined_descriptions = {}
        for file_path, description in descriptions.items():
            messages = [
                SystemMessage(content=self.refine_prompt),
                HumanMessage(
                    content=f"""
                    Repository Tree:
                    {repo_tree_string}

                    Current File Path:
                    {file_path}

                    Natural Language Description:
                    {description}
                """
                ),
            ]

            response = self.llm.invoke(messages)
            refined_descriptions[file_path] = response.content

        # Convert to documents
        description_documents = []
        for file_path, description in refined_descriptions.items():
            document = Document(
                page_content=description, metadata={"source": file_path}
            )
            description_documents.append(document)

        # Convert to graph documents
        print("Converting to graph documents...")
        graph_documents = self.llm_transformer.convert_to_graph_documents(description_documents)
        
        try:
            url = os.getenv('NEO4J_URI')
            username = os.getenv('NEO4J_USERNAME')
            password = os.getenv('NEO4J_PASSWORD')
            graph = Neo4jGraph(url=url, username=username, password=password, enhanced_schema=True)
        except Exception as e:
            return print(f"Error connecting to Neo4j: {e}")
        
        print("Adding graph documents to Neo4j...")
        graph.add_graph_documents(graph_documents)
        graph.refresh_schema()

        return graph, graph_documents

   
    def create_visualization(self, graph_documents, output_path="semantic_graph.html"):
        """
        Create a visualization of the semantic relationships.

        Args:
            graph_documents: List of graph documents
            output_path: Path to save the visualization

        Returns:
            Path to the saved visualization
        """
        # Create Pyvis network
        net = Network(notebook=True, cdn_resources='in_line', height="1000px", width="100%")

        # Create a NetworkX graph
        G = nx.Graph()

        # Dictionary to track unique nodes and metadata
        node_metadata = {}

        # Helper to create a hashable key from type, id, and properties
        def make_node_key(node):
            return (
                node.type,
                node.id,
                tuple(sorted(node.properties.items()))
            )

        # Add nodes and edges
        for graph in graph_documents:
            for rel in graph.relationships:
                # if rel.source.type == "Package" or rel.target.type == "Package":
                #     continue  # Skip packages entirely
                # Get full, unique node keys
                source_key = make_node_key(rel.source)
                target_key = make_node_key(rel.target)
                rel_type = rel.type

                # Store node metadata
                node_metadata[source_key] = {
                    "id": rel.source.id,
                    "type": rel.source.type,
                    "properties": rel.source.properties
                }
                node_metadata[target_key] = {
                    "id": rel.target.id,
                    "type": rel.target.type,
                    "properties": rel.target.properties
                }

                # Add nodes and edges
                G.add_node(source_key)
                G.add_node(target_key)
                G.add_edge(source_key, target_key, label=rel_type)

        # Get unique types for coloring
        unique_types = list(set(meta["type"] for meta in node_metadata.values()))
        color_map = plt.get_cmap("tab10")
        type_colors = {t: color_map(i / len(unique_types)) for i, t in enumerate(unique_types)}
        type_colors_rgba = {
            t: f'rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 0.8)' for t, c in type_colors.items()
        }

        # Get all unique relationship types
        relationship_types = list(set(attr.get("label", "") for _, _, attr in G.edges(data=True)))

        # Create colors for each relationship type
        rel_color_map = plt.get_cmap("Set2")  # Using a different colormap for relationships
        rel_colors = {
            rel_type: f'rgba({int(rel_color_map(i / len(relationship_types))[0]*255)}, {int(rel_color_map(i / len(relationship_types))[1]*255)}, {int(rel_color_map(i / len(relationship_types))[2]*255)}, 0.9)'
            for i, rel_type in enumerate(relationship_types)
        }

        # Degree-based sizing
        degrees = dict(G.degree())
        min_size, max_size = 10, 50
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
            props_html = "<br>".join(f"{k}: {v}" for k, v in properties.items()) if properties else "No properties"

            net.add_node(
                str(node_key),  # string key for Pyvis
                label=label,
                size=size_scale[node_key],
                color=color,
                title=f"<b>{node_type}</b> ({label})<br>{props_html}"
            )

        # Add edges with longer length, colors, and arrows
        for source, target, attr in G.edges(data=True):
            rel_label = attr.get("label", "")
            edge_color = rel_colors.get(rel_label, "#888888")  # Default gray if not found
            # Adding longer length to edges (300 instead of default), with arrows and colors
            net.add_edge(str(source), str(target), title=rel_label, label=rel_label, length=300, color=edge_color, arrows='to')

        # Save graph
        net.save_graph(output_path)

        # Build legend
        legend_html = """
        <div id="legend" style="position: absolute; top: 10px; left: 10px; background: white; padding: 10px; border-radius: 8px; box-shadow: 0px 0px 5px rgba(0,0,0,0.2); font-family: Arial, sans-serif; z-index: 1000;">
            <h4 style="margin: 0; padding-bottom: 5px;">Node Legend</h4>
        """

        for node_type, color in type_colors_rgba.items():
            legend_html += f'<div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 15px; height: 15px; background:{color}; margin-right: 5px; border-radius: 50%;"></div> {node_type}</div>'

        legend_html += """
            <h4 style="margin: 5px 0; padding-bottom: 5px;">Relationship Legend</h4>
        """

        for rel_type in sorted(relationship_types):
            rel_color = rel_colors.get(rel_type, "#888888")
            legend_html += f'<div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 20px; height: 3px; background: {rel_color}; margin-right: 5px;"></div> {rel_type}</div>'

        legend_html += "</div>"

        # Inject legend
        with open(output_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        html_content = html_content.replace("</body>", legend_html + "</body>")

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(html_content)

        print(f"Graph saved as {output_path}")

        return output_path
