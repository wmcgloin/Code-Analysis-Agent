"""
Simple web application for the code analysis multi-agent workflow.
"""

import os

import gradio as gr

from multi_agent_workflow import CodeAnalysisWorkflow

# Initialize the workflow
workflow = CodeAnalysisWorkflow(model_name="gpt-4o-mini")


def analyze_repo(repo_path, query):
    """
    Analyze a repository based on the user's query.

    Args:
        repo_path: Path to the repository to analyze
        query: The user's query

    Returns:
        HTML output and explanation
    """
    if not os.path.exists(repo_path):
        return None, f"Repository path does not exist: {repo_path}"

    try:
        result = workflow.analyze(repo_path, query)

        # Read the HTML file
        with open(result["output_path"], "r", encoding="utf-8") as f:
            html_content = f.read()

        explanation = f"""
        Analysis completed using {result['agent_type']}
        
        Reason for agent selection: {result['reason']}
        
        Output saved to: {result['output_path']}
        """

        return html_content, explanation

    except Exception as e:
        return None, f"Error analyzing repository: {str(e)}"


# Create the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Code Analysis Multi-Agent Workflow")
    gr.Markdown(
        """
    This application uses a multi-agent workflow to analyze code repositories.
    
    - **Semantic Agent**: Creates detailed semantic relationship graphs showing how classes, methods, and functions correlate to each other.
    - **Logical Agent**: Creates high-level logical relationship graphs with minimal nodes for a quick overview.
    
    The supervisor agent will decide which specialized agent to use based on your query.
    """
    )

    with gr.Row():
        with gr.Column():
            repo_path = gr.Textbox(
                label="Repository Path", placeholder="/path/to/repository"
            )
            query = gr.Textbox(
                label="Query",
                placeholder="I want a high-level overview of the codebase structure",
            )
            analyze_btn = gr.Button("Analyze Repository")

        with gr.Column():
            explanation = gr.Textbox(label="Explanation", lines=10)

    html_output = gr.HTML(label="Graph Visualization")

    analyze_btn.click(
        fn=analyze_repo, inputs=[repo_path, query], outputs=[html_output, explanation]
    )


def main():
    app.launch()


if __name__ == "__main__":
    main()
