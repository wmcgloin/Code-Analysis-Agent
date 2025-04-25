"""
Streamlit application for interactive codebase analysis using a LangGraph agent router.

Features:
- GitHub repository cloning
- Visualization of file structure and code relationships
- Multi-folder code graph construction and database initialization
- Conversational agent that answers questions about the codebase

Main logic begins in the `main()` function.
"""

import os
import shutil
import subprocess
import atexit
import time
import uuid
from functools import lru_cache
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components

from langchain_anthropic import ChatAnthropic  # unused
from langchain.chat_models import init_chat_model # may not be needed in main loop
from langchain_core.messages import AIMessage, HumanMessage

from app.session_state import initialize_session_state

from utils.repo import (
    clone_repo_from_url,
    delete_cloned_repo,
    repo_exists,
    get_cached_repo_tree,
    REPO_DIR
)


from utils import LogLevel, get_logger, set_log_level

from app.setup.initialize_database import initialize_database


import tools.micro_tools as mt



logger = get_logger()
set_log_level("DEBUG")


# Set page configuration
st.set_page_config(
    page_title="Code Analysis Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


from app.session_state import cleanup_repo
# Register cleanup function to run on process exit
atexit.register(lambda: cleanup_repo())




















def main():
    """
    Main entrypoint for the Streamlit Code Analysis Agent app.

    This function sets up:
    - Sidebar UI for repo input, analysis, reset, and database init
    - Repository structure preview
    - LangGraph-based agent chat interface for answering codebase questions
    """
    
    # Ensure all required session state variables are initialized
    initialize_session_state()
    logger.info(f"Session thread_id: {st.session_state.thread_id}")
    # st.write(f"🧵 Session thread ID: `{st.session_state.thread_id}`")

    # Sidebar Header and Description
    st.sidebar.title("Code Analysis Agent")
    st.sidebar.markdown("This app uses a LangGraph-based agent router to analyze code repositories.")

    # Input: GitHub repo URL
    repo_url = st.sidebar.text_input(
        "GitHub Repository URL",
        value=st.session_state.repo_path,
        help="Enter a GitHub repo to analyze (e.g., https://github.com/user/repo)",
        placeholder="https://github.com/username/repo"
    )
    st.session_state.repo_path = repo_url

    # Two-button layout: Analyze and Reset
    col1, col2 = st.sidebar.columns(2)

    # Analyze button: Clone the repo
    with col1:
        if st.button("Analyze"):
            if repo_url:
                with st.spinner("Cloning repository..."):
                    if clone_repo_from_url(repo_url):
                        st.session_state.repo_cloned = True
                        st.success("Repository cloned successfully!")
                    else:
                        st.error("Failed to clone repository.")
            else:
                st.warning("Please enter a GitHub repository URL.")

    # Reset button: Clears graph DB, deletes repo, resets messages
    with col2:
        if st.button("Reset"):
            if repo_exists():
                with st.spinner("Deleting repository and database..."):
                    try:
                        if "database" in st.session_state and st.session_state.database:
                            graph_db = st.session_state.database.get("graph_db")
                            if graph_db:
                                graph_db.query("MATCH (n) DETACH DELETE n")
                                st.success("Graph database cleared!")
                                logger.debug("Neo4j DB cleared")
                    except Exception as e:
                        logger.error(f"Error clearing Neo4j DB: {e}")
                        st.error(f"Failed to clear Neo4j database: {e}")

                    if delete_cloned_repo():
                        st.session_state.repo_cloned = False
                        st.session_state.database = {}
                        st.session_state.messages = []
                        st.success("Repository and session state cleared.")
                    else:
                        st.error("Failed to delete repository contents.")
            else:
                st.info("No repository to delete.")

    # Reset conversation thread (new UUID, clears messages)
    if st.sidebar.button("Reset Conversation"):
        old_thread_id = st.session_state.thread_id if "thread_id" in st.session_state else "none"
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        logger.debug(f"Thread reset from {old_thread_id} to {st.session_state.thread_id}")
        st.sidebar.success("New conversation started.")

    # If repo is cloned, show repo structure & database initialization tools
    if repo_exists():
        with st.sidebar.expander("Repository Structure"):
            try:
                repo_last_modified = os.path.getmtime(REPO_DIR)
            except:
                repo_last_modified = time.time()
            try:
                repo_tree = get_cached_repo_tree(REPO_DIR, repo_last_modified)
                st.code(repo_tree, language="bash")
            except Exception as e:
                st.error(f"Error generating repository tree: {str(e)}")

        # UI for initializing database from user-selected folders
        st.sidebar.subheader("Database Initialization")
        src_folders = st.sidebar.text_input(
            "Source folders (comma-separated)",
            help="e.g., src,app/lib, or . for root",
            placeholder="src,app/src,lib"
        )
        if st.sidebar.button("Initialize Database"):
            if src_folders:
                folder_list = [f.strip() for f in src_folders.split(",")]
                st.session_state.database = initialize_database(REPO_DIR, folder_list)
            else:
                st.sidebar.warning("Please enter at least one source folder.")

        # If visualization was generated, show the toggle/download UI
        display_visualization()

        # Input: User enters a natural language question about the codebase
        user_query = st.chat_input("Ask a question about the codebase...")

        # Handle query depending on current state
        if user_query:
            if repo_exists():
                if "database" in st.session_state and st.session_state.database:
                    # All set – process query using LangGraph router
                    process_query(user_query, REPO_DIR)
                else:
                    # Database not initialized yet
                    st.session_state.messages.append(HumanMessage(content=user_query))
                    st.session_state.messages.append(AIMessage(content="Please initialize the database using the 'Initialize Database' button in the sidebar before asking questions."))
                    display_messages()
            else:
                # Repo not cloned yet
                st.session_state.messages.append(HumanMessage(content=user_query))
                st.session_state.messages.append(AIMessage(content="Please clone a repository first using the 'Analyze' button before asking questions."))
                display_messages()
        else:
            # No new input – show chat history
            display_messages()































def display_visualization():
    """Display visualization controls in the sidebar."""
    if os.path.exists("code_relationships_graph.html"):
        # Create a container for the visualization controls in the sidebar
        st.sidebar.subheader("Code Visualization")
        
        # Option to display the visualization inline
        if st.sidebar.button("Show/Hide Visualization"):
            if "show_visualization" not in st.session_state:
                st.session_state.show_visualization = True
            else:
                st.session_state.show_visualization = not st.session_state.show_visualization
            
            # Provide feedback on current state
            if st.session_state.show_visualization:
                st.sidebar.success("Visualization shown below")
            else:
                st.sidebar.info("Visualization hidden")
        
        # Download button for the HTML file
        with open("code_relationships_graph.html", "rb") as file:
            st.sidebar.download_button(
                label="Download Visualization",
                data=file,
                file_name="code_relationships_graph.html",
                mime="text/html"
            )
        
        # Display visualization in the sidebar if show_visualization is True
        if "show_visualization" in st.session_state and st.session_state.show_visualization:
            # Create a container for the visualization in the sidebar
            st.sidebar.markdown("### Visualization Preview")
            
            # Read and display the HTML content in an iframe within the sidebar
            with open("code_relationships_graph.html", "r", encoding="utf-8") as file:
                html_content = file.read()
                
                # Create a container in the sidebar for the visualization
                sidebar_container = st.sidebar.container()
                
                # Use st.components.v1.html within the sidebar container
                with sidebar_container:
                    import streamlit.components.v1 as components
                    components.html(
                        html_content, 
                        height=400,  # Adjust height to fit in sidebar
                        scrolling=True
                    )




def process_query(user_query: str, repo_path: str):
    """
    Process a user query using the agent router.
    """
    # Check if database is initialized
    if not "database" in st.session_state or not st.session_state.database:
        # Add messages to session state only, don't display them here
        st.session_state.messages.append(HumanMessage(content=user_query))
        st.session_state.messages.append(AIMessage(content="Please initialize the database first before asking questions."))
        return
    
    # Connect query engine to micro_tools before each query
    if "query_engines" in st.session_state.database and st.session_state.database["query_engines"]:
        try:
            from tools import micro_tools
            # Get the first available query engine
            first_engine_key = list(st.session_state.database["query_engines"].keys())[0]
            query_engine = st.session_state.database["query_engines"][first_engine_key]
            
            # Set it in the micro_tools module
            micro_tools.query_engine = query_engine
            logger.debug(f"Query engine from {first_engine_key} connected to micro_tools")
        except Exception as e:
            logger.error(f"Failed to connect query engine to tools: {e}")
    
    # Add repository path to the query
    full_query = f"{user_query}"

    # Add the user message to the conversation history
    user_message = HumanMessage(content=full_query)
    st.session_state.messages.append(user_message)
    
    # Display messages up to this point (including the new user message)
    display_messages()
    
    # Create a placeholder for the assistant's thinking message
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("*Thinking...*")

    # Initialize the state with the user query
    initial_state = {"messages": [user_message]}

    # # Make query engine available to the router if needed
    # if "query_engines" in st.session_state.database and st.session_state.database["query_engines"]:
    #     # Add the first query engine to the initial state (can be expanded to include all engines)
    #     first_engine_key = list(st.session_state.database["query_engines"].keys())[0]
    #     query_engine = st.session_state.database["query_engines"][first_engine_key]
        
    #     # Import and set up micro_tools to use our query engine
    #     try:
    #         from tools import micro_tools
    #         micro_tools.query_engine = query_engine
    #         logger.debug("RAG query engine made available to router")
    #     except Exception as e:
    #         logger.error(f"Failed to make query engine available to router: {e}")

    
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Process the query with the agent router
    # Debug message to track execution
    logger.debug("Starting to process query with agent router using invoke()")

    # Store all messages from the execution
    all_messages = []
    final_response = None

    # Process the query using invoke() instead of stream()
    try:
        logger.info(f"Configuration thread_id: {config}")
        result = st.session_state.graph.invoke(initial_state, config=config)

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

        # Add the response to session state
        if final_response:
            st.session_state.messages.append(AIMessage(content=final_response))
            # Update the placeholder with the final response
            thinking_placeholder.markdown(final_response)
        else:
            logger.warning("No agent response found")
            error_msg = "I'm sorry, I couldn't generate a response. Please try again."
            st.session_state.messages.append(AIMessage(content=error_msg))
            # Update the placeholder with the error message
            thinking_placeholder.markdown(error_msg)

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
        error_msg = f"An error occurred: {str(e)}"
        st.session_state.messages.append(AIMessage(content=error_msg))
        # Update the placeholder with the error message
        thinking_placeholder.markdown(error_msg)



def display_messages():
    """Display the conversation history."""
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)
































if __name__ == "__main__":
    # Run the main function
    main()