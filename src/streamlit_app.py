"""
Streamlit app for the LangGraph agent router.

This module provides a web interface for interacting with the agent router.
"""

import os
import shutil
import subprocess
import atexit
from typing import Any, Dict, List

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage

from agent_router import create_router_graph

from tools.code_analysis_tools import (explain_code_logic, generate_repo_tree,
                                       list_python_files, read_code_file)

from utils import LogLevel, get_logger, set_log_level

import time

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

# Define the fixed repository directory
REPO_DIR = "cloned_repository"

# Register cleanup function to run on process exit
atexit.register(lambda: cleanup_repo())


@st.cache_resource
def get_session_cleanup():
    """Create a resource that will clean up when the session ends."""
    class SessionCleanup:
        def __init__(self):
            self.initialized = True
        
        def __del__(self):
            # This will be called when the session ends and the object is garbage collected
            try:
                logger.debug("Session ending, cleaning up repository...")
                cleanup_repo()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    return SessionCleanup()


def cleanup_repo():
    """Clean up the repository directory using a robust approach."""
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
                    if not os.path.exists(REPO_DIR) or not os.listdir(REPO_DIR):
                        logger.debug("Repository contents successfully cleaned up")
                        return
                    
                    # If we still have files, wait and retry
                    time.sleep(retry_delay)
                    
                except Exception as e:
                    logger.debug(f"Cleanup retry {retry+1} failed: {e}")
                    time.sleep(retry_delay)
            
            # If we get here, we couldn't clean everything, but continue anyway
            logger.warning("Could not delete all repository files during cleanup, but continuing anyway")
            
    except Exception as e:
        logger.error(f"Error during repository cleanup: {e}")


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
        st.session_state.repo_path = ""
    
    if "repo_cloned" not in st.session_state:
        st.session_state.repo_cloned = False
    
    if "database" not in st.session_state:
        st.session_state.database = {}
    
    # Get session cleanup resource (will clean up on session end)
    if "cleanup_resource" not in st.session_state:
        st.session_state.cleanup_resource = get_session_cleanup()

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
    Initialize graph database for the specified source folders within the repository.
    
    Args:
        repo_path: Path to the cloned repository
        src_folders: List of source folders to analyze
        
    Returns:
        Dict containing unified graph_db and a combined list of graph_documents
    """
    try:
        from tools.micro_process import ProcessMicroData
        
        # Create a status message and progress bar
        status = st.status("Initializing database...", expanded=True)
        progress_bar = status.progress(0)
        
        # Use OpenAI model if available, otherwise fallback to Anthropic
        try:
            from langchain.chat_models import init_chat_model
            llm = init_chat_model("gpt-4o", model_provider="openai")
            status.update(label="Using OpenAI GPT-4o model for analysis", state="running")
        except Exception as e:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
            status.update(label="Using Anthropic Claude model for analysis", state="running")
        
        # Initialize variables to store unified results
        all_graph_documents = []
        last_graph_db = None
        processed_folders = []
        
        # Create a custom ProcessMicroData class that reports progress
        class StreamlitProcessMicroData(ProcessMicroData):
            def analyze_codebase(self, repo_path: str):
                # Count total files to process
                from tools.code_analysis_tools import list_python_files
                python_files = list_python_files(repo_path)
                total_files = len(python_files)
                status.update(label=f"Found {total_files} Python files to analyze", state="running")
                
                # Track file progress
                file_counter = 0
                
                # Store original methods
                original_read_code_file = read_code_file
                
                # Override read_code_file to track progress
                def patched_read_code_file(file_path):
                    nonlocal file_counter
                    file_counter += 1
                    
                    # Get relative path for display
                    try:
                        rel_path = os.path.relpath(file_path, repo_path)
                    except:
                        rel_path = file_path
                        
                    # Update status and progress
                    status.update(label=f"Processing file {file_counter}/{total_files}: {rel_path}", state="running")
                    if total_files > 0:
                        progress_bar.progress(min(file_counter / total_files, 1.0))
                        
                    # Call original function
                    return original_read_code_file(file_path)
                
                # Apply the patch
                import tools.code_analysis_tools
                tools.code_analysis_tools.read_code_file = patched_read_code_file
                
                try:
                    # Run the original method
                    result = super().analyze_codebase(repo_path)
                    return result
                finally:
                    # Restore original method
                    tools.code_analysis_tools.read_code_file = original_read_code_file
        
        # Process each folder with progress tracking
        total_folders = len(src_folders)
        for i, folder in enumerate(src_folders):
            folder_path = os.path.join(repo_path, folder)
            if os.path.exists(folder_path):
                status.update(label=f"Analyzing folder {i+1}/{total_folders}: {folder}", state="running")
                
                # Use our custom class for analysis
                gen_graphdb = StreamlitProcessMicroData(llm=llm)
                graph_db, graph_documents = gen_graphdb.analyze_codebase(repo_path=folder_path)
                
                # Store results
                last_graph_db = graph_db
                all_graph_documents.extend(graph_documents)
                processed_folders.append(folder)
                
                # Update status
                status.update(label=f"Completed analysis of {folder}", state="complete")
            else:
                status.update(label=f"Folder not found: {folder}", state="error")
                time.sleep(2)  # Give user time to see the error
        
        # Update final status
        if processed_folders:
            status.update(label=f"Database initialized with {len(all_graph_documents)} code elements from {len(processed_folders)} folders", 
                         state="complete")
        else:
            status.update(label="No folders were successfully processed", state="error")
        
        # Return unified results
        return {
            "graph_db": last_graph_db,
            "graph_documents": all_graph_documents,
            "processed_folders": processed_folders
        }
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        if 'status' in locals():
            status.update(label=f"Error initializing database: {str(e)}", state="error")
        else:
            st.error(f"Failed to initialize database: {str(e)}")
        return {}

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

    # Repository URL input
    repo_url = st.sidebar.text_input(
        "GitHub Repository URL",
        value=st.session_state.repo_path,
        help="URL of the GitHub repository to analyze (e.g., https://github.com/username/repo)",
        placeholder="https://github.com/username/repo"
    )
    st.session_state.repo_path = repo_url

    # Analyze and Reset buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Analyze"):
            if repo_url:
                with st.spinner("Cloning repository..."):
                    # Clone the repository
                    if clone_github_repo(repo_url):
                        st.session_state.repo_cloned = True
                        st.success(f"Repository cloned successfully!")
                    else:
                        st.error("Failed to clone repository.")
            else:
                st.warning("Please enter a GitHub repository URL.")
    
    with col2:
        if st.button("Reset"):
            if is_repo_cloned():
                with st.spinner("Deleting repository..."):
                    # Delete the cloned repository
                    if delete_cloned_repo():
                        st.session_state.repo_cloned = False
                        # Also clear any database
                        if "database" in st.session_state:
                            del st.session_state.database
                        st.success("Repository deleted successfully!")
                    else:
                        st.error("Failed to delete repository contents.")
            else:
                st.info("No repository to delete.")

    # Display repository structure if cloned
    if is_repo_cloned():
        with st.sidebar.expander("Repository Structure"):
            try:
                repo_tree = generate_repo_tree(REPO_DIR)
                st.code(repo_tree, language="bash")
            except Exception as e:
                st.error(f"Error generating repository tree: {str(e)}")
        
        # Database initialization section
        st.sidebar.subheader("Database Initialization")
        
        # Input for source folders
        src_folders = st.sidebar.text_input(
            "Source folders (comma-separated)",
            help="Enter the paths to source folders relative to repository root (e.g., src,app/src,lib)",
            placeholder="src,app/src,lib"
        )
        
        # Initialize database button
        if st.sidebar.button("Initialize Database"):
            if src_folders:
                # Split and clean the folder paths
                folder_list = [folder.strip() for folder in src_folders.split(",")]
                
                # Initialize the database for each folder
                st.session_state.database = initialize_database(REPO_DIR, folder_list)

            else:
                st.sidebar.warning("Please enter at least one source folder")

    # Main content
    st.title("Code Analysis Conversation")

    # Display conversation history
    display_messages()

    # User input
    user_query = st.chat_input("Ask a question about the codebase...")

    if user_query:
        if is_repo_cloned():
            if "database" in st.session_state and st.session_state.database:
                # Database is initialized, process the query
                process_query(user_query, REPO_DIR)
            else:
                # Repository cloned but database not initialized
                with st.chat_message("assistant"):
                    st.warning("Please initialize the database using the 'Initialize Database' button in the sidebar before asking questions.")
                    st.session_state.messages.append(HumanMessage(content=user_query))
                    st.session_state.messages.append(AIMessage(content="Please initialize the database using the 'Initialize Database' button in the sidebar before asking questions."))
        else:
            with st.chat_message("assistant"):
                st.warning("Please clone a repository first using the 'Analyze' button before asking questions.")
                st.session_state.messages.append(HumanMessage(content=user_query))
                st.session_state.messages.append(AIMessage(content="Please clone a repository first using the 'Analyze' button before asking questions."))

if __name__ == "__main__":
    main()