"""
Graph Assembly for LangGraph Router

This module creates the full LangGraph router graph that delegates user queries
between macro and micro agents via a supervisor node.
"""

from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

from .nodes import create_supervisor_node, create_agent_node
from .state import RouterState
from utils import get_logger

# Logger setup
logger = get_logger()

# In-memory checkpointing
memory = MemorySaver()


def create_router_graph(
    llm: BaseChatModel,
    macro_tools: List[BaseTool] = None,
    micro_tools: List[BaseTool] = None,
) -> StateGraph:
    """
    Creates a complete LangGraph router with supervisor, macro, and micro agents.

    Args:
        llm: The language model to use for all agents.
        macro_tools: Tools available to the macro agent (high-level reasoning).
        micro_tools: Tools available to the micro agent (detailed code analysis).

    Returns:
        A compiled StateGraph that can process incoming queries.
    """
    # Create node functions for each agent role
    supervisor_node = create_supervisor_node(llm)
    macro_node = create_agent_node("macro", llm, macro_tools)
    micro_node = create_agent_node("micro", llm, micro_tools)

    # Build and compile the state graph
    logger.info("Building the router graph...")
    builder = StateGraph(RouterState)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("macro", macro_node)
    builder.add_node("micro", micro_node)

    logger.info("Compiling the router graph...")
    return builder.compile(checkpointer=memory)