"""
Supervisor Agent for managing the multi-agent workflow.
This agent decides which specialized agent to use based on the user's query.
"""

from typing import Literal, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

class Router(TypedDict):
    """Agent to route to next."""
    next: Literal["semantic_agent", "logical_agent"]
    reason: str

class SupervisorAgent:
    """
    Supervisor agent that decides which specialized agent to use based on the user's query.
    """
    
    def __init__(self, model_name="gpt-4o-mini"):
        """Initialize the supervisor agent with the specified LLM model."""
        self.llm = ChatOpenAI(model=model_name)
        self.system_prompt = """
        You are a supervisor agent responsible for routing user queries to the appropriate specialized agent.
        You have two agents available:
        
        1. semantic_agent: This agent analyzes code repositories and creates detailed semantic relationship graphs
           showing how classes, methods, and functions correlate to each other. It provides a comprehensive view
           of the codebase's structure and relationships.
           
        2. logical_agent: This agent analyzes code repositories and creates high-level logical relationship graphs
           showing how files, components, modules, and concepts correlate to each other. It aims to create a simplified
           abstraction with minimal nodes (fewer than 5 for small codebases) for a quick overview.
        
        Based on the user's query, decide which agent would be most appropriate to handle the request.
        
        If the user wants detailed information about code structure, relationships between specific classes or methods,
        or a comprehensive view of the codebase, choose the semantic_agent.
        
        If the user wants a high-level overview, abstraction, or simplified representation of the codebase's logical
        structure with minimal nodes, choose the logical_agent.
        """
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Route the user's query to the appropriate agent.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with the selected agent and the reason for selection
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"User query: {query}\n\nWhich agent should handle this query? Choose either 'semantic_agent' or 'logical_agent' and provide a brief reason.")
        ]
        
        response = self.llm.with_structured_output(Router).invoke(messages)
        
        return {
            "agent": response["next"],
            "reason": response["reason"]
        }
