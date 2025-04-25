"""
Multi-agent workflow for code analysis.
This module implements a workflow with a supervisor agent and two specialized agents:
1. Semantic Agent: Creates detailed semantic relationship graphs
2. Logical Agent: Creates high-level logical relationship graphs with minimal nodes
"""

import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from agents.logical_agent import LogicalAgent
from agents.semantic_agent import SemanticAgent
from agents.supervisor import SupervisorAgent

load_dotenv()


class CodeAnalysisWorkflow:
    """
    Multi-agent workflow for analyzing code repositories.
    """

    def __init__(self, model_name="gpt-4o-mini"):
        """
        Initialize the workflow with a supervisor and specialized agents.

        Args:
            model_name: The name of the OpenAI model to use
        """
        self.supervisor = SupervisorAgent(model_name=model_name)
        self.semantic_agent = SemanticAgent(model_name=model_name)
        self.logical_agent = LogicalAgent(model_name=model_name)

        # Create output directory if it doesn't exist
        os.makedirs("./output", exist_ok=True)

    def analyze(self, repo_path: str, query: str) -> Dict[str, Any]:
        """
        Analyze a code repository based on the user's query.

        Args:
            repo_path: Path to the repository to analyze
            query: The user's query

        Returns:
            Dictionary with the analysis results
        """
        # Route the query to the appropriate agent
        routing = self.supervisor.route_query(query)
        selected_agent = routing["agent"]
        reason = routing["reason"]

        print(f"Selected agent: {selected_agent}")
        print(f"Reason: {reason}")

        # Analyze the codebase with the selected agent
        if selected_agent == "semantic_agent":
            graph_documents = self.semantic_agent.analyze_codebase(repo_path)
            output_path = self.semantic_agent.create_visualization(
                graph_documents, "./output/semantic_graph.html"
            )
            agent_type = "Semantic Agent"
        else:  # logical_agent
            graph_documents = self.logical_agent.analyze_codebase(repo_path)
            output_path = self.logical_agent.create_visualization(
                graph_documents, "./output/logical_graph.html"
            )
            agent_type = "Logical Agent"

        return {
            "agent_type": agent_type,
            "output_path": output_path,
            "reason": reason,
            "graph_documents": graph_documents,
        }


def main():
    """
    Example usage of the CodeAnalysisWorkflow.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze a code repository using a multi-agent workflow"
    )
    parser.add_argument("repo_path", help="Path to the repository to analyze")
    parser.add_argument("query", help="Query describing what kind of analysis you want")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")

    args = parser.parse_args()

    workflow = CodeAnalysisWorkflow(model_name=args.model)
    result = workflow.analyze(args.repo_path, args.query)

    print(f"\nAnalysis completed using {result['agent_type']}")
    print(f"Output saved to: {result['output_path']}")


if __name__ == "__main__":
    main()
