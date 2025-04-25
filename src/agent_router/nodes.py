"""
LangGraph Agent Nodes

Defines supervisor and agent nodes used within the LangGraph router system.
Each node handles different roles in the task delegation workflow.
"""

from typing import List, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent

from .state import RouterState, RouterOutput
from utils import get_logger

logger = get_logger()


def create_supervisor_node(llm: BaseChatModel) -> callable:
    """
    Builds the supervisor node responsible for routing queries to either
    the macro or micro agent.

    Args:
        llm: The language model used for decision-making.

    Returns:
        A callable node function used by LangGraph.
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

    def supervisor_node(state: RouterState) -> Command[Literal["macro", "micro", "__end__"]]:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        logger.debug(f"Supervisor processing {len(messages)} messages")
        response = llm.with_structured_output(RouterOutput).invoke(messages)
        goto = response["next"]
        logger.info(f"Supervisor routed to: {goto}")

        if goto == "FINISH":
            goto = END

        supervisor_message = AIMessage(
            content=f"I've decided to route to {goto if goto != END else 'FINISH'} because it's the most suitable next step.",
            name="supervisor"
        )

        return Command(
            goto=goto,
            update={
                "next": goto,
                "messages": state["messages"] + [supervisor_message]
            },
        )

    return supervisor_node


def create_agent_node(agent_name: str, llm: BaseChatModel, tools: List[BaseTool] = None) -> callable:
    """
    Builds an agent node (macro or micro) that performs a specific analysis task.

    Args:
        agent_name: "macro" or "micro"
        llm: The language model used by the agent.
        tools: Optional list of tools this agent has access to.

    Returns:
        A callable node function used by LangGraph.
    """
    tools = tools or []

    # Role-specific system prompt
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
    else:
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

    agent = create_react_agent(llm, tools=tools, prompt=system_prompt)

    def agent_node(state: RouterState) -> Command[Literal["supervisor"]]:
        logger.info(f"{agent_name.capitalize()} agent is processing...")
        result = agent.invoke(state)

        agent_response = result["messages"][-1].content
        logger.debug(f"{agent_name.capitalize()} agent response: {agent_response[:100]}...")

        agent_message = AIMessage(content=agent_response, name=agent_name)
        return Command(
            update={"messages": state["messages"] + [agent_message]},
            goto="supervisor",
        )

    return agent_node
