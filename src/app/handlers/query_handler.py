from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
import logging
from tools import micro_tools
from app.ui import display_messages

logger = logging.getLogger(__name__)

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