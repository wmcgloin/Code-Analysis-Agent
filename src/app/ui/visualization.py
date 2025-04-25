# app/ui/visualization.py

import os
import streamlit as st
import streamlit.components.v1 as components


def display_visualization():
    """Display visualization controls and preview in the sidebar."""
    if os.path.exists("code_relationships_graph.html"):
        st.sidebar.subheader("Code Visualization")

        # Show/hide toggle
        if st.sidebar.button("Show/Hide Visualization"):
            if "show_visualization" not in st.session_state:
                st.session_state.show_visualization = True
            else:
                st.session_state.show_visualization = not st.session_state.show_visualization

            if st.session_state.show_visualization:
                st.sidebar.success("Visualization shown below")
            else:
                st.sidebar.info("Visualization hidden")

        # Download button
        with open("code_relationships_graph.html", "rb") as file:
            st.sidebar.download_button(
                label="Download Visualization",
                data=file,
                file_name="code_relationships_graph.html",
                mime="text/html"
            )

        # Show preview if enabled
        if st.session_state.get("show_visualization"):
            st.sidebar.markdown("### Visualization Preview")

            with open("code_relationships_graph.html", "r", encoding="utf-8") as file:
                html_content = file.read()

            with st.sidebar.container():
                components.html(
                    html_content,
                    height=400,
                    scrolling=True
                )
