"""
Agent Router Module

This module implements a LangGraph-based routing system with a supervisor agent
that delegates tasks between two specialized agents: macro and micro.
"""

from typing import Any, Dict, List, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict

from utils import get_logger

# Get the logger instance
logger = get_logger()


class RouterState(MessagesState):
    """State for the router graph, extending MessagesState with next field."""

    next: str


class RouterOutput(TypedDict):
    """Structured output for the supervisor to determine the next agent."""

    next: Literal["macro", "micro", "FINISH"]


def create_supervisor_node(llm: BaseChatModel) -> callable:
    """
    Creates a supervisor node that decides which agent to route to next.

    Args:
        llm: The language model to use for the supervisor

    Returns:
        A function that processes the state and returns a command
    """
    system_prompt = """
        You are a supervisor agent responsible for routing user queries to the appropriate specialized agent.
        You have two agents available:
        
        1. micro: This agent analyzes code repositories and creates detailed semantic relationship graphs
           showing how classes, methods, and functions are related to each other. It provides a comprehensive view
           of the codebase's structure and relationships. It is capable of generating text responses with visualizations
           when the user requests them.
           
        2. macro: This agent analyzes code repositories and creates high-level logical relationship graphs
           showing how files, components, modules, and concepts correlate to each other. It aims to create a simplified
           abstraction with minimal nodes (fewer than 5 for small codebases) for a quick overview.
        
        If the user wants detailed information about code structure, relationships between specific classes or methods,
        or a comprehensive visualization of the codebase, choose the micro.
        
        If the user wants a high-level overview, abstraction, or simplified representation of the codebase's logical
        structure with minimal nodes, choose the macro.
    
        Given the user request and conversation history, decide which agent should act next.
        When the task is complete, respond with FINISH.
        
        Always explain your reasoning for choosing a particular agent or deciding to finish.
        """

    def supervisor_node(
        state: RouterState,
    ) -> Command[Literal["macro", "micro", "__end__"]]:
        messages = [
            SystemMessage(content=system_prompt),
        ] + state["messages"]

        logger.debug(f"Supervisor is processing with {len(messages)} messages")

        response = llm.with_structured_output(RouterOutput).invoke(messages)
        goto = response["next"]

        logger.info(f"Supervisor decided to route to: {goto}")

        if goto == "FINISH":
            goto = END

        # Add the supervisor's decision as a message for better tracking
        supervisor_message = AIMessage(
            content=f"I've decided to route to {goto if goto != END else 'FINISH'} because this is the appropriate next step.",
            name="supervisor",
        )

        return Command(
            goto=goto,
            update={"next": goto, "messages": state["messages"] + [supervisor_message]},
        )

    return supervisor_node


def create_agent_node(
    agent_name: str, llm: BaseChatModel, tools: List[BaseTool] = None
) -> callable:
    """
    Creates an agent node that processes tasks and returns results.

    Args:
        agent_name: Name of the agent ("macro" or "micro")
        llm: The language model to use for the agent
        tools: Optional list of tools the agent can use

    Returns:
        A function that processes the state and returns a command
    """
    from langgraph.prebuilt import create_react_agent

    if tools is None:
        tools = []

    if agent_name == "macro":
        system_prompt = """
        You are a macro-level strategic agent. You excel at:
        - Understanding the big picture
        - Creating high-level plans and strategies
        - Breaking down complex problems into manageable parts
        - Identifying key objectives and priorities
        
        Focus on planning and strategic thinking. For detailed implementation, 
        defer to the micro agent.
        
        Always provide clear, structured responses with your analysis and recommendations. Also respond complete once the generation is completed.
        """
    else:  # micro agent
        system_prompt = f"""
        You are a suupervisor that answers the questions given the following tools.
        {tools}

        You will use each tool only once and the guidelines of the tools are as follows:
        - You will always use the generate_text_response tool at least once to get a final answer. Use the generate_text_response tool to begin with if the user is not asking for a relationship but asking for a definition.
        - You will use the retrieve_cypher_relationships tool if the user is asking about relationships.
        - Only use the visualize_relationships tool to visualize the relationships when the user asks for a visualization.

        Operate from one of the three Scenarios below:
            - Scenario 1 : generate_text_response
            - Scenario 2 : retrieve_cypher_relationships --> generate_text_response --> visualize_relationships
            - Scenario 3 : retrieve_cypher_relationships --> generate_text_response

        USE EACH TOOL ONLY ONCE.

        Always provide clear, actionable responses with specific details and implementation steps.
        """

    # Create a ReAct agent with the appropriate tools and system prompt
    agent = create_react_agent(llm, tools=tools, prompt=system_prompt)

    def agent_node(state: RouterState) -> Command[Literal["supervisor"]]:
        logger.info(f"{agent_name.capitalize()} agent is processing...")

        # Invoke the agent with the current state
        result = agent.invoke(state)

        # Get the agent's response
        agent_response = result["messages"][-1].content
        logger.debug(
            f"{agent_name.capitalize()} agent response: {agent_response[:100]}..."
            if len(agent_response) > 100
            else agent_response
        )

        # Log the full response for debugging
        logger.debug(f"Full {agent_name} response: {agent_response}")

        # Create a new AIMessage with the agent's response and explicit name
        agent_message = AIMessage(content=agent_response, name=agent_name)
        logger.debug(f"Created AIMessage with name: {agent_message.name}")

        # Return a command to update the messages and go back to the supervisor
        return Command(
            update={"messages": state["messages"] + [agent_message]},
            goto="supervisor",
        )

    return agent_node


from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

def create_router_graph(
    llm: BaseChatModel,
    macro_tools: List[BaseTool] = None,
    micro_tools: List[BaseTool] = None,
) -> StateGraph:
    """
    Creates a complete router graph with supervisor, macro, and micro agents.

    Args:
        llm: The language model to use for all agents
        macro_tools: List of tools for the macro agent
        micro_tools: List of tools for the micro agent

    Returns:
        A compiled StateGraph ready for execution
    """
    ################
    # IMPORT NODES #
    ################
    supervisor_node = create_supervisor_node(llm)
    macro_node = create_agent_node("macro", llm, macro_tools)
    micro_node = create_agent_node("micro", llm, micro_tools)

    # Build the graph
    logger.info("Building the router graph...")
    builder = StateGraph(RouterState)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("macro", macro_node)
    builder.add_node("micro", micro_node)

    # Compile the graph
    logger.info("Compiling the router graph...")
    return builder.compile(checkpointer=memory)
