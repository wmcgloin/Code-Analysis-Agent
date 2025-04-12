
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
    print("importing tools")
    from langchain_neo4j import Neo4jGraph
    from tools.micro_cypher_chain import CypherGraphBuilder

    url = os.getenv('NEO4J_URI')
    username = os.getenv('NEO4J_USERNAME')
    password = os.getenv('NEO4J_PASSWORD')

    print("initializing Neo4jGraph")
    graph_db = Neo4jGraph(
        url=url,
        username=username,
        password=password,
        enhanced_schema=True
    )
    from langchain.chat_models import init_chat_model
    llm = init_chat_model("gpt-4o", model_provider="openai")

    print("Building Cypher graph")
    # Create the Cypher graph
    builder = CypherGraphBuilder(llm=llm, graph_db=graph_db)
    # cypher_graph = builder.create_cypher_graph()
    cypher_graph = builder.create_cypher_graph()

    print("Invoking cypher graph")
    result = cypher_graph.invoke({"question": question})

    return result

    # Run the graph with a question
    # result = cypher_graph.invoke({"question": "How is the Utils.Utils module related to the Deduplicatioin.__Main__ module?"})
    # print(result["answer"])
    


@tool
def visualize_relationships(results):
    """
    Takes Cypher graph results and generates a local HTML visualization of the relationships.

    Args:
        results (str): The entire results from the retrieve_cypher_relationships tool with no edits

    Returns:
        dict: Summary statistics or metadata about the rendered visualization.
    """
    print("Visualizing relationships...")
    print("Results:", results)
    import json

    if isinstance(results, str):
        results = json.loads(results)

    cypher_results = results["cypher_results"]

    from tools.micro_visual_response import visualize_cypher_results
    stats = visualize_cypher_results(cypher_results, "code_relationships_graph.html")
    return stats # Not really necessary

query_engine = None

@tool
def generate_text_response(results):
    """
    Builds a RAG (Retrieval-Augmented Generation) graph from the codebase, then runs a 
    natural language query against it to extract a human-readable explanation.

    Args:
        results (str): The entire results from the retrieve_cypher_relationships tool with no edits

    Returns:
        str: Natural language answer to the predefined question.
    """
    print("Generating text response...")
    print("Results:", results)

    if query_engine is None:
        raise ValueError("query_engine is not initialized.")
    
    import json

    if isinstance(results, str):
        results = json.loads(results)

    initial_response = results["answer"]

    # Query the system
    answer = query_engine.query(initial_response)

    return answer