"""
Streamlit app for the LangGraph agent router.

This module provides a web interface for interacting with the agent router.
"""

import os
from typing import Any, Dict, List

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage

from agent_router import create_router_graph
from tools.code_analysis_tools import (explain_code_logic, generate_repo_tree,
                                       list_python_files, read_code_file)
from utils import LogLevel, get_logger, set_log_level

logger = get_logger()

#############
# LOG LEVEL #
#############
set_log_level("DEBUG")

# Set page configuration
st.set_page_config(
    page_title="Code Analysis Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


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

    # TODO: Microtools
    return {"macro_tools": macro_tools, "micro_tools": []}


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "graph" not in st.session_state:

        #########
        # MODEL #
        #########
        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

        #########
        # TOOLS #
        #########
        tools = setup_tools()

        st.session_state.graph = create_router_graph(
            llm=llm, macro_tools=tools["macro_tools"], micro_tools=tools["micro_tools"]
        )

    if "repo_path" not in st.session_state:
        st.session_state.repo_path = "."


def display_messages():
    """Display the conversation history."""
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)


def process_query(user_query: str, repo_path: str):
    """
    Process a user query using the agent router.

    Args:
        user_query: The user's query
        repo_path: Path to the repository to analyze
    """
    # Add repository path to the query
    full_query = f"{user_query} Repository path: {repo_path}"

    # Add the user message to the conversation history
    user_message = HumanMessage(content=full_query)
    st.session_state.messages.append(user_message)

    # Display the user message
    with st.chat_message("user"):
        st.write(user_query)

    # Initialize the state with the user query
    initial_state = {"messages": [user_message]}

    # Process the query with the agent router
    with st.spinner("Thinking..."):
        # Create a placeholder for the assistant's response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            # Debug message to track execution
            logger.debug("Starting to process query with agent router using invoke()")

            # Store all messages from the execution
            all_messages = []
            final_response = None

            # Process the query using invoke() instead of stream()
            try:
                result = st.session_state.graph.invoke(initial_state)

                # Log the result for debugging
                logger.debug(f"Graph execution completed. Result type: {type(result)}")

                # Extract messages from the result
                if "messages" in result:
                    all_messages = result["messages"]
                    logger.debug(f"Found {len(all_messages)} messages in result")

                    # Find the last non-supervisor message from an agent
                    for message in reversed(all_messages):
                        if (
                            hasattr(message, "name")
                            and message.name != "supervisor"
                            and isinstance(message, AIMessage)
                        ):
                            final_response = message.content
                            logger.debug(
                                f"Found response from {message.name}: {final_response[:100]}..."
                            )
                            break

                # Display the response
                if final_response:
                    response_placeholder.write(final_response)
                    # Add to conversation history
                    st.session_state.messages.append(AIMessage(content=final_response))
                else:
                    logger.warning("No agent response found")
                    response_placeholder.write(
                        "I'm sorry, I couldn't generate a response. Please try again."
                    )

                    # Debug the messages to understand why no response was found
                    if all_messages:
                        logger.debug("All messages in result:")
                        for i, msg in enumerate(all_messages):
                            msg_type = type(msg).__name__
                            msg_name = getattr(msg, "name", "unknown")
                            logger.debug(
                                f"  [{i}] {msg_type} ({msg_name}): {msg.content[:100]}..."
                                if len(msg.content) > 100
                                else f"  [{i}] {msg_type} ({msg_name}): {msg.content}"
                            )

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                response_placeholder.write(f"An error occurred: {str(e)}")


def main():
    """Main function for the Streamlit app."""
    # Initialize session state
    initialize_session_state()

    # Sidebar
    st.sidebar.title("Code Analysis Agent")
    st.sidebar.markdown(
        """
    This app uses a LangGraph-based agent router to analyze code repositories.
    """
    )

    # Repository path input
    repo_path = st.sidebar.text_input(
        "Repository Path",
        value=st.session_state.repo_path,
        help="Path to the repository to analyze",
    )
    st.session_state.repo_path = repo_path

    # Display repository structure if path exists
    if os.path.exists(repo_path):
        with st.sidebar.expander("Repository Structure"):
            try:
                repo_tree = generate_repo_tree(repo_path)
                st.code(repo_tree, language="bash")
            except Exception as e:
                st.error(f"Error generating repository tree: {str(e)}")

    # Main content
    st.title("Code Analysis Conversation")

    # Display conversation history
    display_messages()

    # User input
    user_query = st.chat_input("Ask a question about the codebase...")

    if user_query:
        process_query(user_query, repo_path)


if __name__ == "__main__":
    main()
