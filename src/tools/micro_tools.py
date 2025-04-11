
import os
from typing import Annotated, Any, Dict, List
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from dotenv import load_dotenv

# You can pass the path if the file isn't in the same directory
load_dotenv()


@tool
def retrieve_cypher_relationships(question: str):
    """
    Connects to a local Neo4j instance and uses a language model to generate a Cypher-based graph
    representation of relationships between entities (e.g., code modules).

    Returns:
        dict: A structured representation of nodes and relationships inferred from the graph.
    """

    from langchain_neo4j import Neo4jGraph
    from tools.micro_cypher_chain import CypherGraphBuilder

    url = os.getenv('NEO4J_URI')
    username = os.getenv('NEO4J_USERNAME')
    password = os.getenv('NEO4J_PASSWORD')

    graph_db = Neo4jGraph(
        url=url,
        username=username,
        password=password,
        enhanced_schema=True
    )

    # Create the Cypher graph
    builder = CypherGraphBuilder(llm=BaseChatModel, graph_db=graph_db)
    # cypher_graph = builder.create_cypher_graph()
    cypher_graph = builder.create_cypher_graph()
    result = cypher_graph.invoke({"question": question})

    return result['answer']

    # Run the graph with a question
    # result = cypher_graph.invoke({"question": "How is the Utils.Utils module related to the Deduplicatioin.__Main__ module?"})
    # print(result["answer"])
    


@tool
def visualize_relationships(cypher_results):
    """
    Takes Cypher graph results and generates a local HTML visualization of the relationships.

    Args:
        cypher_results (dict): The Cypher graph data containing nodes and relationships.

    Returns:
        dict: Summary statistics or metadata about the rendered visualization.
    """
    from tools.micro_visual_response import visualize_cypher_results
    stats = visualize_cypher_results(cypher_results, "code_relationships_graph.html")
    return stats # Not really necessary


@tool
def generate_text_response(initial_response):
    """
    Builds a RAG (Retrieval-Augmented Generation) graph from the codebase, then runs a 
    natural language query against it to extract a human-readable explanation.

    Args:
        initial_response: the initial natural language text response generated from the GRAPH RAG.

    Returns:
        str: Natural language answer to the predefined question.
    """
    global query_engine
    # Query the system
    answer = query_engine.query(initial_response)

    return answer