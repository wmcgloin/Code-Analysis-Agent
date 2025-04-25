"""
Database Initialization Module

This script defines the `initialize_database` function, which processes a cloned
GitHub repository and prepares:
- A Neo4j graph database based on code structure
- A retrieval-augmented generation (RAG) system for semantic search
- A list of graph documents used for downstream query tools

It is meant to be run as part of the app setup pipeline.
"""

import os
import time
from typing import List, Dict

import streamlit as st
from langchain.chat_models import init_chat_model

from utils import get_logger
from utils.filesystem import list_python_files
import utils.filesystem as fs
from rag.vector_rag import create_rag_system
from app.graph_builders.micro_graph_builder import MicroCodeGraphBuilder
import tools.micro.tools as mt

# Logger setup
logger = get_logger()

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
        # Create a status message and progress bar
        status = st.status("Initializing database...", expanded=True)
        progress_bar = status.progress(0)
        
        # Initialize the LLM (language model) for code analysis
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
        original_read_code_file = fs.read_code_file
        
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
                fs.read_code_file = progress_tracking_read_code_file
                
                try:
                    # Create a new instance for each folder
                    gen_graphdb = MicroCodeGraphBuilder(llm=llm)
                    graph_db, graph_documents = gen_graphdb.build_graph_and_upload(repo_path=folder_path)
                    
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
        fs.read_code_file = original_read_code_file
        
        # Initialize RAG system for each processed folder
        if processed_folders:
            status.update(label="Initializing RAG system for text-based queries...", state="running")
            
            try:       
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
                # Use the first query engine by default
                first_folder = processed_folders[0]
                mt.query_engine = query_engines[first_folder]
                status.update(label=f"Connected query engine to tools system", state="complete")
                logger.debug(f"Query engine from {first_folder} connected to micro_tools")
            except Exception as e:
                logger.error(f"Failed to connect query engine to tools: {e}")
                status.update(label=f"Warning: Failed to connect query engine to tools: {e}", state="warning")
        
        # Return unified results
        return {
            "graph_db": graph_db,
            "graph_documents": all_graph_documents,
            "processed_folders": processed_folders,
            "query_engines": query_engines
        }
    except Exception as e:
        # Restore original method in case of error
        if 'original_read_code_file' in locals() and 'tools' in locals():
            read_code_file = original_read_code_file
            
        logger.error(f"Error initializing database: {e}")
        if 'status' in locals():
            status.update(label=f"Error initializing database: {str(e)}", state="error")
        else:
            st.error(f"Failed to initialize database: {str(e)}")
        return {}