"""
Macro Agent Module

This module implements a macro-level strategic agent that handles high-level planning,
strategic thinking, and big picture analysis. It specializes in code analysis and visualization.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from utils import get_logger
from tools.code_analysis_tools import (
    CodeAnalysisTools,
    generate_repo_tree,
    read_code_file,
    list_python_files,
    explain_code_logic,
    identify_code_patterns
)

# Get the logger instance
logger = get_logger()


class MacroAgent:
    """
    A macro-level strategic agent that handles high-level planning, strategic thinking,
    and code analysis.
    """
    
    def __init__(self, llm: BaseChatModel, tools: List[BaseTool] = None):
        """
        Initialize the MacroAgent.
        
        Args:
            llm: The language model to use
            tools: Optional list of tools the agent can use
        """
        self.llm = llm
        self.tools = tools or []
        self.system_prompt = """
        You are a macro-level strategic agent specializing in code analysis and visualization. You excel at:
        - Understanding the big picture of codebases
        - Creating high-level plans and strategies for code analysis
        - Breaking down complex codebases into manageable parts
        - Identifying key components and their relationships
        - Explaining code architecture and design patterns
        
        Your primary goal is to help users understand codebases by:
        1. Analyzing the structure and relationships between components
        2. Creating visual representations of code relationships
        3. Explaining the high-level architecture and design decisions
        4. Identifying patterns and anti-patterns in the code
        
        Focus on providing clear, structured responses with your analysis and recommendations.
        When asked about specific implementation details, defer to the micro agent.
        """
        
        # Initialize code analysis tools
        self.code_analysis_tools = CodeAnalysisTools(llm)
        
        # Create output directory if it doesn't exist
        os.makedirs("./output", exist_ok=True)
    
    def analyze_codebase(self, repo_path: str) -> Dict[str, Any]:
        """
        Analyze a codebase and create a visualization of the relationships.
        
        Args:
            repo_path: Path to the repository to analyze
            
        Returns:
            Dictionary with the analysis results
        """
        logger.info(f"Analyzing codebase at {repo_path}")
        
        # Generate repository tree
        repo_tree = generate_repo_tree(repo_path)
        logger.debug(f"Repository tree:\n{repo_tree}")
        
        # Analyze the codebase
        graph_documents = self.code_analysis_tools.analyze_codebase(repo_path)
        logger.info(f"Generated {len(graph_documents)} graph documents")
        
        # Create visualization
        output_path = self.code_analysis_tools.create_visualization(
            graph_documents, "./output/code_graph.html"
        )
        logger.info(f"Visualization saved to {output_path}")
        
        return {
            "repo_path": repo_path,
            "output_path": output_path,
            "graph_documents": graph_documents,
        }
    
    def explain_architecture(self, repo_path: str) -> str:
        """
        Explain the architecture of a codebase.
        
        Args:
            repo_path: Path to the repository to analyze
            
        Returns:
            Explanation of the codebase architecture
        """
        logger.info(f"Explaining architecture of {repo_path}")
        
        # Generate repository tree
        repo_tree = generate_repo_tree(repo_path)
        
        # List Python files
        python_files = list_python_files(repo_path)
        
        # Sample a few files to understand the architecture
        sample_files = python_files[:min(5, len(python_files))]
        file_contents = {}
        
        for file_path in sample_files:
            full_path = os.path.join(repo_path, file_path)
            file_contents[file_path] = read_code_file(full_path)
        
        # Generate architecture explanation
        messages = [
            SystemMessage(content=f"""
                You are an expert in software architecture and code analysis.
                Given a repository structure and sample code files, provide a high-level explanation
                of the codebase architecture, including:
                
                1. The overall architectural pattern (if identifiable)
                2. Main components and their responsibilities
                3. How components interact with each other
                4. Key design patterns used
                5. Potential strengths and weaknesses of the architecture
                
                Focus on the big picture rather than implementation details.
            """),
            HumanMessage(content=f"""
                Repository structure:
                {repo_tree}
                
                Sample files:
                {', '.join(sample_files)}
                
                File contents:
                
                {''.join([f'--- {path} ---\n{content}\n\n' for path, content in file_contents.items()])}
                
                Please explain the architecture of this codebase.
            """)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def identify_patterns(self, repo_path: str) -> str:
        """
        Identify design patterns and anti-patterns in a codebase.
        
        Args:
            repo_path: Path to the repository to analyze
            
        Returns:
            Description of identified patterns
        """
        logger.info(f"Identifying patterns in {repo_path}")
        
        # Use the identify_code_patterns tool
        return identify_code_patterns(repo_path)
    
    def create_plan(self, task: str) -> Dict[str, Any]:
        """
        Create a high-level plan for the given task.
        
        Args:
            task: The task to create a plan for
            
        Returns:
            A dictionary containing the plan details
        """
        logger.info(f"Creating plan for task: {task}")
        
        messages = [
            SystemMessage(content=f"""
                You are a strategic planning expert. Given a task related to code analysis,
                create a detailed plan with steps to accomplish the task.
                
                Your plan should include:
                1. Clear, actionable steps
                2. Required resources or tools for each step
                3. Expected outcomes
                4. Potential challenges and mitigation strategies
                
                Format your response as a structured plan with numbered steps.
            """),
            HumanMessage(content=f"Task: {task}")
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            "task": task,
            "plan": response.content,
        }
    
    def summarize_codebase(self, repo_path: str) -> str:
        """
        Generate a high-level summary of a codebase.
        
        Args:
            repo_path: Path to the repository to analyze
            
        Returns:
            Summary of the codebase
        """
        logger.info(f"Summarizing codebase at {repo_path}")
        
        # Generate repository tree
        repo_tree = generate_repo_tree(repo_path)
        
        # List Python files
        python_files = list_python_files(repo_path)
        
        messages = [
            SystemMessage(content=f"""
                You are an expert in code analysis and summarization.
                Given a repository structure and the list of files, provide a concise summary
                of what this codebase likely does, its main components, and its purpose.
                
                Focus on creating a clear, high-level overview that would help someone
                quickly understand what this project is about.
            """),
            HumanMessage(content=f"""
                Repository structure:
                {repo_tree}
                
                Python files:
                {', '.join(python_files)}
                
                Please provide a high-level summary of this codebase.
            """)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
