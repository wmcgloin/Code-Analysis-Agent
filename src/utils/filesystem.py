"""
Filesystem Utilities

Provides helper functions for traversing and reading files from a repository.
Used during codebase parsing and analysis (e.g., before populating graphs or RAG systems).
"""

import os
from typing import List
from utils import get_logger

logger = get_logger()

EXCLUDED_DIRS = {
    "__pycache__", ".git", ".venv", "legacy", "archive", "archived", "old", ".mypy_cache"
}

def list_python_files(repo_path: str) -> List[str]:
    """
    Recursively list all Python files in a repository, skipping excluded folders and hidden files.

    Args:
        repo_path: Path to the root of the repository.

    Returns:
        List of relative paths to all Python files found.
    """
    python_files = []

    for root, dirs, files in os.walk(repo_path):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]

        for file in files:
            # Include only .py files, skip hidden files
            if file.endswith(".py") and not file.startswith("."):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                python_files.append(relative_path)

    logger.debug(f"Python files found in {repo_path}: {python_files}")
    return python_files


def read_code_file(file_path: str) -> str:
    """
    Read and return the contents of a code file.

    Args:
        file_path: Absolute or relative path to the code file.

    Returns:
        File content as a string, or an error message if reading fails.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Error reading file: {file_path} - {e}")
        return f"Error reading file: {str(e)}"
