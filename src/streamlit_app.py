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

from app.session_state import initialize_session_state, cleanup_repo
from app.setup import initialize_database
from app.ui import display_messages, display_visualization
from app.handlers import process_query
from utils.repo import (clone_repo_from_url, delete_cloned_repo, repo_exists, get_cached_repo_tree, REPO_DIR)
from utils import LogLevel, get_logger, set_log_level



logger = get_logger()
set_log_level("DEBUG")

# Set page configuration
st.set_page_config(
    page_title="Code Analysis Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
                        # Clear Neo4j database if initialized
                        if "database" in st.session_state and st.session_state.database:
                            graph_db = st.session_state.database.get("graph_db")
                            if graph_db:
                                graph_db.query("MATCH (n) DETACH DELETE n")
                                st.success("Graph database cleared!")
                                logger.debug("Neo4j DB cleared")

                        # Delete HTML visualization file
                        if os.path.exists("code_relationships_graph.html"):
                            os.remove("code_relationships_graph.html")
                            logger.debug("Visualization file deleted")
                            
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
            "Source folder (relative path)",
            help="Enter a single folder to analyze, such as `src` or `.` for the repo root.",
            placeholder="e.g. src, backend/api, ."
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


if __name__ == "__main__":
    # Run the main function
    main()