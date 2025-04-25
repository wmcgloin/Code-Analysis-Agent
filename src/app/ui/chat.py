# app/ui/chat.py

import streamlit as st
from langchain_core.messages import HumanMessage

def display_messages():
    """Render the conversation history in the chat interface."""
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)
