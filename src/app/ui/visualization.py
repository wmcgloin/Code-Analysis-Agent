# app/ui/visualization.py

import os

import streamlit as st
import streamlit.components.v1 as components


def display_visualization():
    """Display visualization controls and preview in the sidebar."""

    graph_path = "code_relationships_graph.html"
    structure_path = "codebase_structure.html"

    # Available visualizations
    available_options = {
        "Code Relationships Graph": graph_path,
        "Codebase Structure": structure_path,
    }

    st.sidebar.subheader("Code Visualization")

    # Select which visualization to view
    selected_name = st.sidebar.selectbox(
        "Select Visualization", list(available_options.keys())
    )

    selected_file = available_options[selected_name]

    try:
        # Download button
        with open(selected_file, "rb") as file:
            st.sidebar.download_button(
                label="Download Visualization",
                data=file,
                file_name=os.path.basename(selected_file),
                mime="text/html",
            )

        # Display the HTML preview
        st.sidebar.markdown("### Visualization Preview")
        with open(selected_file, "r", encoding="utf-8") as file:
            html_content = file.read()

        with st.sidebar.container():
            components.html(html_content, height=400, scrolling=True)
    except Exception as e:
        st.sidebar.error(f"Error displaying visualization: {e}")
