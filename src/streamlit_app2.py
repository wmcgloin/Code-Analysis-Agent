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

from old_agent_router import create_router_graph

from tools.code_analysis_tools import (
    explain_code_logic,
    generate_repo_tree,
    list_python_files,
    read_code_file
)
import tools.micro_tools as mt

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

# Define the fixed repository directory
REPO_DIR = "cloned_repository"


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
    st.write(f"🧵 Session thread ID: `{st.session_state.thread_id}`")

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
                    if clone_github_repo(repo_url):
                        st.session_state.repo_cloned = True
                        st.success("Repository cloned successfully!")
                    else:
                        st.error("Failed to clone repository.")
            else:
                st.warning("Please enter a GitHub repository URL.")

    # Reset button: Clears graph DB, deletes repo, resets messages
    with col2:
        if st.button("Reset"):
            if is_repo_cloned():
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
    if is_repo_cloned():
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
            if is_repo_cloned():
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

def clone_github_repo(repo_url: str) -> bool:
    """
    Clone a GitHub repository to the fixed repository directory.
    
    Args:
        repo_url: The URL of the GitHub repository
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(REPO_DIR, exist_ok=True)
        
        # Clear the directory if it already exists
        if os.path.exists(REPO_DIR) and os.listdir(REPO_DIR):
            shutil.rmtree(REPO_DIR)
            os.makedirs(REPO_DIR, exist_ok=True)
        
        # Clone the repository
        subprocess.run(["git", "clone", repo_url, REPO_DIR], check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error cloning repository: {e}")
        return False
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return False


def delete_cloned_repo() -> bool:
    """
    Delete the contents of the cloned repository directory.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(REPO_DIR):
            # Force Git to clean up its own files first (helps with file locks)
            try:
                # Navigate to the repository
                original_dir = os.getcwd()
                os.chdir(REPO_DIR)
                
                # Tell Git to clean up
                subprocess.run(["git", "gc"], check=False, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                
                # Go back to original directory
                os.chdir(original_dir)
            except Exception as e:
                logger.debug(f"Error during git cleanup (non-critical): {e}")
            
            # On Windows, sometimes we need to retry deletion a few times
            max_retries = 3
            retry_delay = 1  # seconds
            
            for retry in range(max_retries):
                try:
                    # First try to remove the .git folder which often causes problems
                    git_dir = os.path.join(REPO_DIR, ".git")
                    if os.path.exists(git_dir):
                        # Make sure all files are writable
                        for root, dirs, files in os.walk(git_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    os.chmod(file_path, 0o666)  # Make file writable
                                except:
                                    pass  # Ignore errors
                                    
                        # Try to delete the .git directory
                        shutil.rmtree(git_dir, ignore_errors=True)
                    
                    # Now try to delete everything else
                    for item in os.listdir(REPO_DIR):
                        item_path = os.path.join(REPO_DIR, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path, ignore_errors=True)
                        else:
                            try:
                                os.chmod(item_path, 0o666)  # Make file writable
                                os.remove(item_path)
                            except:
                                pass  # Ignore errors
                    
                    # Check if we successfully cleaned up
                    remaining = [f for f in os.listdir(REPO_DIR) if f != '.git']
                    if not remaining:
                        return True
                    
                    # If we still have files, wait and retry
                    time.sleep(retry_delay)
                    
                except Exception as e:
                    logger.debug(f"Delete retry {retry+1} failed: {e}")
                    time.sleep(retry_delay)
            
            # If we get here, we couldn't clean everything, but return true anyway
            # and just log a warning
            logger.warning("Could not delete all repository files, but continuing anyway")
            return True
            
        return True
    except Exception as e:
        st.error(f"Error deleting repository contents: {e}")
        return False


def is_repo_cloned() -> bool:
    """
    Check if a repository is currently cloned.
    
    Returns:
        True if a repository is cloned, False otherwise
    """
    return os.path.exists(REPO_DIR) and len(os.listdir(REPO_DIR)) > 0



def initialize_database(repo_path: str, src_folders: List[str]) -> Dict:
    """
    Initialize graph database and RAG system for the specified source folders within the repository.
    
    Args:
        repo_path: Path to the cloned repository
        src_folders: List of source folders to analyze
        
    Returns:
        Dict containing unified graph_db, graph_documents, and query_engine
    """
    try:
        from tools.micro_process import ProcessMicroData
        import tools.code_analysis_tools
        import time
        
        # Create a status message and progress bar
        status = st.status("Initializing database...", expanded=True)
        progress_bar = status.progress(0)
        

        from langchain.chat_models import init_chat_model
        llm = init_chat_model("gpt-4o", model_provider="openai")
        status.update(label="Using OpenAI GPT-4o model for analysis", state="running")
        
        # Initialize variables to store unified results
        all_graph_documents = []
        last_graph_db = None
        processed_folders = []
        query_engines = {}
        
        # Global counter for tracking progress
        total_files_processed = 0
        total_files_count = 0
        
        # First pass: count all files to process
        for folder in src_folders:
            folder_path = os.path.join(repo_path, folder)
            if os.path.exists(folder_path):
                python_files = list_python_files(folder_path)
                total_files_count += len(python_files)
        
        status.update(label=f"Found {total_files_count} Python files to analyze across {len(src_folders)} folders", state="running")
        
        # Store original method 
        original_read_code_file = tools.code_analysis_tools.read_code_file
        
        # Create a progress-tracking wrapper for read_code_file
        def progress_tracking_read_code_file(file_path):
            nonlocal total_files_processed
            
            # Increment counter
            total_files_processed += 1
            
            # Get relative path for display
            try:
                rel_path = os.path.relpath(file_path, repo_path)
            except:
                rel_path = file_path
                
            # Update status and progress
            status.update(label=f"Processing file {total_files_processed}/{total_files_count}: {rel_path}", state="running")
            if total_files_count > 0:
                progress_bar.progress(min(total_files_processed / total_files_count, 1.0))
                
            # Call original function
            return original_read_code_file(file_path)
        
        # Process each folder with progress tracking
        total_folders = len(src_folders)
        
        for i, folder in enumerate(src_folders):
            folder_path = os.path.join(repo_path, folder)
            if os.path.exists(folder_path):
                status.update(label=f"Analyzing folder {i+1}/{total_folders}: {folder}", state="running")
                
                # Replace the read_code_file with our tracking version
                tools.code_analysis_tools.read_code_file = progress_tracking_read_code_file
                
                try:
                    # Create a new instance for each folder
                    gen_graphdb = ProcessMicroData(llm=llm)
                    graph_db, graph_documents = gen_graphdb.analyze_codebase(repo_path=folder_path)
                    
                    # Store results
                    last_graph_db = graph_db
                    all_graph_documents.extend(graph_documents)
                    processed_folders.append(folder)
                    
                    # Update status
                    status.update(label=f"Completed graph analysis of {folder}", state="running")
                finally:
                    # Make sure we restore the original method even if there's an error
                    pass
            else:
                status.update(label=f"Folder not found: {folder}", state="error")
                time.sleep(2)  # Give user time to see the error
        
        # Restore original method
        tools.code_analysis_tools.read_code_file = original_read_code_file
        
        # Initialize RAG system for each processed folder
        if processed_folders:
            status.update(label="Initializing RAG system for text-based queries...", state="running")
            
            try:
                from tools.micro_text_answer import create_rag_system
                
                for i, folder in enumerate(processed_folders):
                    folder_path = os.path.join(repo_path, folder)
                    status.update(label=f"Setting up RAG system for {folder} ({i+1}/{len(processed_folders)})", state="running")
                    
                    # Initialize RAG system
                    try:
                        db_setup, query_engine = create_rag_system(
                            repo_path=folder_path,
                            embedding_model="text-embedding-3-large",
                            llm_model="gpt-4o",
                        )
                        query_engines[folder] = query_engine
                        status.update(label=f"RAG system initialized for {folder}", state="running")
                    except Exception as e:
                        st.warning(f"Failed to initialize RAG system for {folder}: {e}")
                        status.update(label=f"Failed to initialize RAG system for {folder}: {e}", state="error")
                        time.sleep(2)
            except Exception as e:
                st.warning(f"Failed to initialize RAG systems: {e}")
                status.update(label=f"Failed to initialize RAG systems: {e}", state="error")
                time.sleep(2)
        
        # Update final status
        if processed_folders:
            status.update(
                label=(
                    f"Database initialized with {len(all_graph_documents)} code elements from {len(processed_folders)} folders. "
                    f"RAG systems created for {len(query_engines)} folders."
                ), 
                state="complete"
            )
        else:
            status.update(label="No folders were successfully processed", state="error")

        # After creating RAG systems and before returning results
        if query_engines and processed_folders:
            # Set up the query engine for micro_tools to use
            try:
                from tools import micro_tools
                # Use the first query engine by default
                first_folder = processed_folders[0]
                micro_tools.query_engine = query_engines[first_folder]
                status.update(label=f"Connected query engine to tools system", state="complete")
                logger.debug(f"Query engine from {first_folder} connected to micro_tools")
            except Exception as e:
                logger.error(f"Failed to connect query engine to tools: {e}")
                status.update(label=f"Warning: Failed to connect query engine to tools: {e}", state="warning")
        
        # Return unified results
        return {
            "graph_db": last_graph_db,
            "graph_documents": all_graph_documents,
            "processed_folders": processed_folders,
            "query_engines": query_engines
        }
    except Exception as e:
        # Restore original method in case of error
        if 'original_read_code_file' in locals() and 'tools' in locals():
            tools.code_analysis_tools.read_code_file = original_read_code_file
            
        logger.error(f"Error initializing database: {e}")
        if 'status' in locals():
            status.update(label=f"Error initializing database: {str(e)}", state="error")
        else:
            st.error(f"Failed to initialize database: {str(e)}")
        return {}


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



# Create a cached version of the repo tree generator
@st.cache_data
def get_cached_repo_tree(repo_path, last_modified=None):
    """
    Generate and cache the repository tree.
    
    Args:
        repo_path: Path to the repository
        last_modified: A value that changes when the repo is modified
        
    Returns:
        String representation of the repo tree
    """
    return generate_repo_tree(repo_path)








































if __name__ == "__main__":
    # Run the main function
    main()