"""
Utility functions for working with GitHub repositories in the code analysis app.

Includes:
- Cloning a GitHub repo to a local directory
- Deleting the cloned repo and cleaning up Git internals
- Checking if a repo has already been cloned
- Generating and caching a tree representation of the repo structure
"""

import os
import shutil
import subprocess
import time
import streamlit as st
from utils import get_logger

# Logger setup
logger = get_logger()

# Path where the repository is cloned
REPO_DIR = "cloned_repository"


def clone_repo_from_url(repo_url: str) -> bool:
    """
    Clone a GitHub repository from a given URL into the REPO_DIR.

    Args:
        repo_url (str): URL of the GitHub repository to clone.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        os.makedirs(REPO_DIR, exist_ok=True)

        # Clean the directory if it has old contents
        if os.path.exists(REPO_DIR) and os.listdir(REPO_DIR):
            shutil.rmtree(REPO_DIR)
            os.makedirs(REPO_DIR, exist_ok=True)

        # Run Git clone
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
    Delete the contents of the REPO_DIR (used to remove the cloned repo).

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        if os.path.exists(REPO_DIR):
            # Run Git cleanup to remove locks, temp files, etc.
            try:
                original_dir = os.getcwd()
                os.chdir(REPO_DIR)
                subprocess.run(["git", "gc"], check=False,
                               stderr=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL)
                os.chdir(original_dir)
            except Exception as e:
                logger.debug(f"Git cleanup warning: {e}")

            # Attempt multiple retries in case of file lock issues
            max_retries = 3
            retry_delay = 1  # seconds

            for retry in range(max_retries):
                try:
                    # First remove .git folder
                    git_dir = os.path.join(REPO_DIR, ".git")
                    if os.path.exists(git_dir):
                        for root, dirs, files in os.walk(git_dir):
                            for file in files:
                                try:
                                    os.chmod(os.path.join(root, file), 0o666)
                                except:
                                    pass  # Ignore permission errors
                        shutil.rmtree(git_dir, ignore_errors=True)

                    # Now delete the rest of the repo contents
                    for item in os.listdir(REPO_DIR):
                        item_path = os.path.join(REPO_DIR, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path, ignore_errors=True)
                        else:
                            try:
                                os.chmod(item_path, 0o666)
                                os.remove(item_path)
                            except:
                                pass

                    # Success condition: directory is now empty
                    remaining = [f for f in os.listdir(REPO_DIR) if f != ".git"]
                    if not remaining:
                        return True

                    time.sleep(retry_delay)

                except Exception as e:
                    logger.debug(f"Retry {retry+1} failed during deletion: {e}")
                    time.sleep(retry_delay)

            # Couldn’t clean everything, but returning True to continue app flow
            logger.warning("Repository cleanup incomplete, continuing anyway")
            return True

        return True  # Nothing to delete

    except Exception as e:
        st.error(f"Error deleting repository contents: {e}")
        return False


def repo_exists() -> bool:
    """
    Check if a repository has been successfully cloned to REPO_DIR.

    Returns:
        bool: True if REPO_DIR exists and is not empty, else False.
    """
    return os.path.exists(REPO_DIR) and len(os.listdir(REPO_DIR)) > 0


@st.cache_data
def get_cached_repo_tree(repo_path: str, last_modified=None) -> str:
    """
    Return a cached, tree-style representation of the repository structure.

    Args:
        repo_path (str): Path to the repository root.
        last_modified (optional): Timestamp for cache invalidation.

    Returns:
        str: Tree representation of the repository.
    """
    return generate_repo_tree(repo_path)


IGNORED_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".idea", ".vscode", ".mypy_cache", "dist", "build", ".pytest_cache",
    ".tox", ".coverage", ".cache", "archive", "archived", "legacy", "logs"
}

IGNORED_EXTENSIONS = {
    ".pyc", ".pyo", ".log", ".tmp", ".DS_Store", ".pth",
    ".so", ".dll", ".exe", ".bak"
}


def generate_repo_tree(repo_path: str) -> str:
    """
    Generate a tree representation of the repository, filtering out common clutter.

    Args:
        repo_path (str): Root path of the repo.

    Returns:
        str: Formatted string tree.
    """
    tree_string = ""

    for root, dirs, files in os.walk(repo_path):
        # Filter out noisy/system directories
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]

        # Filter files by extension
        files = [f for f in files if not f.startswith(".") and os.path.splitext(f)[1] not in IGNORED_EXTENSIONS]

        level = root.replace(repo_path, "").count(os.sep)
        indent = "│   " * level + "├── "
        tree_string += f"{indent}{os.path.basename(root)}/\n"

        sub_indent = "│   " * (level + 1) + "├── "
        for file in files:
            tree_string += f"{sub_indent}{file}\n"

    logger.debug(f"Repo tree generated:\n{tree_string}")
    return tree_string

