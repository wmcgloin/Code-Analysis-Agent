"""
Session management for the Streamlit code analysis app.

Handles:
- Initializing session state variables (thread ID, messages, graph router, etc.)
- Setting up cleanup routines to delete cloned repos and clear the Neo4j database
"""
import os
import shutil
import time
import uuid
import streamlit as st
from utils import get_logger, set_log_level
from tools.setup import setup_analysis_tools
from agent_router.graph import create_router_graph

logger = get_logger()
set_log_level("DEBUG")  # Optional but helpful for debugging state/session logic


REPO_DIR = "cloned_repository"  # Directory where GitHub repos eare cloned


def initialize_session_state():
    """
    Initialize all required keys in Streamlit's session_state to maintain
    consistent app behavior across reruns and interactions.

    Sets up:
    - message history
    - a unique thread ID for LangGraph
    - the LangGraph agent router
    - repository state flags
    - the shared database object (code graph + RAG)
    - a cleanup routine that clears cloned files and the Neo4j DB at session end
    """

    # Initialize chat message history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Assign a unique thread ID to track this user’s conversation in LangGraph
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        logger.debug(f"Created new thread_id: {st.session_state.thread_id}") ### MAKE SURE THREADID IS CONSISTENT!!!!!

    # Initialize the LangGraph router if not already set up
    if "graph" not in st.session_state:
        from langchain.chat_models import init_chat_model
        
        # Create an LLM interface (OpenAI GPT-4o by default)
        llm = init_chat_model("gpt-4o", model_provider="openai")

        # Load tool functions for macro-level (repo) and micro-level (file/query) tasks
        tools = setup_analysis_tools()

        # If a database exists, attach the first available query engine to the micro tools
        if "database" in st.session_state and "query_engines" in st.session_state.database:
            try:
                first_engine_key = list(st.session_state.database["query_engines"].keys())[0]
                query_engine = st.session_state.database["query_engines"][first_engine_key]
                mt.query_engine = query_engine
            except Exception as e:
                logger.warning(f"Could not initialize micro tools with query engine: {e}")

        # Create the router graph (LangGraph agent network) and store in session
        st.session_state.graph = create_router_graph(
            llm=llm,
            macro_tools=tools["macro_tools"],
            micro_tools=tools["micro_tools"]
        )

    # Default values for repo path and clone status if not already set
    if "repo_path" not in st.session_state:
        st.session_state.repo_path = ""

    if "repo_cloned" not in st.session_state:
        st.session_state.repo_cloned = False

    # Empty database object to hold code graph + RAG system
    if "database" not in st.session_state:
        st.session_state.database = {}

    # Register a cleanup handler to clear resources when session ends
    if "cleanup_resource" not in st.session_state:
        st.session_state.cleanup_resource = get_session_cleanup()


def get_session_cleanup():
    """
    Creates a cleanup resource that deletes the cloned repo and clears
    the Neo4j database when the Streamlit session ends.
    """
    class SessionCleanup:
        def __init__(self):
            self.initialized = True
        
        def __del__(self):
            # Called when the session ends and the object is garbage collected
            try:
                logger.debug("Session ending, cleaning up repository...")

                # If a graph database exists, clear all its contents
                if "database" in st.session_state and st.session_state.database:
                    try:
                        graph_db = st.session_state.database.get("graph_db")
                        if graph_db:
                            graph_db.query("MATCH (n) DETACH DELETE n")
                            logger.debug("Neo4j database cleared on session end")
                    except Exception as e:
                        logger.error(f"Error clearing Neo4j database on session end: {e}")

                # Clean up cloned repository files
                cleanup_repo()

            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

    return SessionCleanup()


def cleanup_repo():
    """
    Deletes all files from the cloned repository directory.
    Handles edge cases like locked Git objects or Windows permission errors.
    """
    try:
        if os.path.exists(REPO_DIR):
            # Step 1: Try to remove .git directory (can cause lock issues)
            git_dir = os.path.join(REPO_DIR, ".git")
            if os.path.exists(git_dir):
                for root, _, files in os.walk(git_dir):
                    for file in files:
                        try:
                            os.chmod(os.path.join(root, file), 0o666)  # Ensure write permission
                        except:
                            pass
                shutil.rmtree(git_dir, ignore_errors=True)

            # Step 2: Remove all other files and folders inside REPO_DIR
            for item in os.listdir(REPO_DIR):
                item_path = os.path.join(REPO_DIR, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    else:
                        os.chmod(item_path, 0o666)
                        os.remove(item_path)
                except:
                    pass

            # Log the result of cleanup
            if not os.listdir(REPO_DIR):
                logger.debug("Repository successfully cleaned up")
            else:
                logger.warning("Some repository files could not be deleted")

    except Exception as e:
        logger.error(f"Error during repository cleanup: {e}")