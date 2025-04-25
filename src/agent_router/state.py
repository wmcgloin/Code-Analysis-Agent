"""
State Definitions for LangGraph Router

Defines the data structures used to manage routing decisions
and message history across nodes in the LangGraph workflow.
"""

from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


class RouterState(MessagesState):
    """
    Extended message state with a `next` pointer for tracking routing decisions.
    """
    next: str


class RouterOutput(TypedDict):
    """
    Structured output from the supervisor, used to determine next routing step.
    """
    next: Literal["macro", "micro", "FINISH"]
