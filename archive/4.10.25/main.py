"""
Main module for the LangGraph routing system.

This module demonstrates how to use the agent_router to create and run a graph
with a supervisor and specialized agents for code analysis.
"""

import argparse
import json
import os
from typing import Any, Dict, List

from langchain_anthropic import \
    ChatAnthropic  # You can replace with your preferred LLM
from langchain_core.messages import HumanMessage

from agent_router.graph import create_router_graph
from tools.code_analysis_tools import (explain_code_logic, generate_repo_tree,
                                       list_python_files, read_code_file)
from utils import LogLevel, get_logger, set_log_level

# Get the logger instance
logger = get_logger()


def setup_tools() -> Dict[str, List]:
    """
    Set up tools for the macro and micro agents.

    Returns:
        Dictionary containing tools for each agent
    """
    # Tools for the macro agent (code analysis)
    macro_tools = [
        generate_repo_tree,
        read_code_file,
        list_python_files,
        explain_code_logic,
    ]

    # For now, we'll return empty list for micro agent
    return {"macro_tools": macro_tools, "micro_tools": []}


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="LangGraph Agent Router for Code Analysis"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity level (can be used multiple times, e.g., -vv)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Set the log level explicitly",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="Analyze the structure of this codebase and explain the relationships between components.",
        help="The query to process",
    )
    parser.add_argument(
        "--repo-path",
        "-r",
        type=str,
        default=".",
        help="Path to the repository to analyze",
    )
    return parser.parse_args()


def set_verbosity(args):
    """
    Set the verbosity level based on command line arguments.

    Args:
        args: Parsed command line arguments
    """
    if args.log_level:
        # Explicit log level takes precedence
        set_log_level(args.log_level)
    else:
        # Map verbosity count to log levels
        verbosity_map = {
            0: "INFO",  # Default
            1: "DEBUG",  # -v
            2: "DEBUG",  # -vv (same as -v for now, but could be more detailed)
        }
        # Get the highest verbosity level if count exceeds map
        level = verbosity_map.get(args.verbose, "DEBUG")
        set_log_level(level)


def format_message_for_display(msg, detailed=False):
    """
    Format a message for display.

    Args:
        msg: The message to format
        detailed: Whether to include detailed information

    Returns:
        Formatted message string
    """
    role = getattr(msg, "type", getattr(msg, "role", "unknown"))
    name = getattr(msg, "name", "system")
    content = msg.content

    if detailed:
        return f"{role.upper()} ({name}):\n{content}"
    else:
        preview = content[:100] + "..." if len(content) > 100 else content
        return f"{role.upper()} ({name}): {preview}"


def main():
    """
    Main function to demonstrate the agent router for code analysis.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Set verbosity level
    set_verbosity(args)

    # Log the configuration
    logger.info(f"Starting agent router with verbosity level: {logger.logger.level}")
    logger.info(f"Repository path: {args.repo_path}")

    # Set up the language model
    # Replace with your preferred LLM and API key setup
    llm = ChatAnthropic(model="claude-3-sonnet-20240229")

    # Set up tools for each agent
    tools = setup_tools()

    # Create the router graph
    graph = create_router_graph(
        llm=llm, macro_tools=tools["macro_tools"], micro_tools=tools["micro_tools"]
    )

    # Get the user query from arguments
    user_query = f"{args.query} Repository path: {args.repo_path}"

    # Initialize the state with the user query
    initial_state = {"messages": [HumanMessage(content=user_query)]}

    # Run the graph and process the results
    logger.info(f"Processing query: {user_query}")

    # Store all events for analysis
    all_events = []

    for event_type, event_data in graph.stream(initial_state, subgraphs=True):
        all_events.append((event_type, event_data))

        # Log different types of events at appropriate levels
        if isinstance(event_type, tuple) and len(event_type) > 0:
            # Node execution event
            node_name = (
                event_type[0].split(":")[0] if ":" in event_type[0] else event_type[0]
            )
            logger.debug(f"Node execution: {node_name}")

            if event_data and logger.logger.level <= LogLevel.DEBUG.value:
                try:
                    logger.debug(f"Event data: {json.dumps(event_data, indent=2)}")
                except:
                    logger.debug(f"Event data: {event_data}")

        elif event_type == "state":
            # State update event
            state = event_data
            logger.info(f"State update: next={state.get('next', 'N/A')}")

            if "messages" in state and state["messages"]:
                latest_message = state["messages"][-1]
                logger.info(
                    f"Latest message: {format_message_for_display(latest_message)}"
                )

                # Log all messages at debug level
                if logger.logger.level <= LogLevel.DEBUG.value:
                    logger.debug("All messages:")
                    for i, msg in enumerate(state["messages"]):
                        logger.debug(f"  [{i}] {format_message_for_display(msg)}")

    # Print a summary of the final conversation
    logger.info("\nFINAL CONVERSATION SUMMARY:")

    final_state = (
        all_events[-1][1] if all_events and all_events[-1][0] == "state" else None
    )

    if final_state and "messages" in final_state:
        for i, msg in enumerate(final_state["messages"]):
            logger.info(f"\n[{i}] {format_message_for_display(msg, detailed=True)}")

    logger.info("Execution complete!")


if __name__ == "__main__":
    main()
