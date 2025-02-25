# NOTE: USING langchain_experimental.graph_transformers main.
from dotenv import load_dotenv
import logging

logging.basicConfig(
    format="[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

load_dotenv()

import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")


from typing_extensions import Annotated, TypedDict


# TypedDict
class Json(TypedDict):
    """Json to return."""

    setup: Annotated[dict, ..., "The setup of the dict"]
    depth: Annotated[int, ..., "How many layers deep the dict is."]


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
"""
from langchain_experimental.graph_transformers import LLMGraphTransformer

llm_transformer = LLMGraphTransformer(llm=llm)
documents = [Document(page_content=text)]
logger.info(f"documents:{documents}")
graph_documents = llm_transformer.convert_to_graph_documents(documents)
for i, doc in enumerate(graph_documents):
    logger.info(f"document #{i+1}")
    nodes = doc.nodes
    relationships = doc.relationships
    for n in nodes:
        logger.info(f"Nodes:{n}")
    for r in relationships:
        logger.info(f"Relationships:{r}")
from plot_graph import plot_network
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

nodes = [str(node) for node in graph_documents[0].nodes]
relationships = [
    (str(rel.source), str(rel.target)) for rel in graph_documents[0].relationships
]

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(relationships)

custom_colors = sns.color_palette("Set2", n_colors=len(G.nodes()))
node_sizes = [3000 if d > 5 else 1000 for v, d in G.degree()]

fig, ax = plot_network(
    G,
    node_size=node_sizes,
    node_color=custom_colors,
    edge_color="#cccccc",
    font_size=12,
    layout="spring",
    palette="Set2",
)

plt.tight_layout()
plt.show()

# NOTE: Using classic llm.with_structured_output(json) approach

# messages = [
#     SystemMessage(
#         "Return a json that describes the logic flow of the given codebase. Only track call stacks. Output in Cypher compliant format for graph representation."
#     ),
#     HumanMessage(
#         """
#             def greet(name):
#                 return f"Hello, {name}!"
#
#             def add(a, b):
#                 return a + b
#
#             def main():
#                 name = input("Enter your name: ")
#                 print(greet(name))
#
#                 try:
#                     x = float(input("Enter first number: "))
#                     y = float(input("Enter second number: "))
#                     print(f"The sum is: {add(x, y)}")
#                 except ValueError:
#                     print("Invalid input. Please enter numbers.")
#
#             if __name__ == "__main__":
#                 main()
#             """
#     ),
# ]
#
# structured_llm = llm.with_structured_output(Json)
# response = structured_llm.invoke(messages)
# print(response)
