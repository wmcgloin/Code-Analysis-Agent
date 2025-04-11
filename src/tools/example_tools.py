"""
Example tools for the macro and micro agents.

This module provides example tool implementations that can be used by the agents.
"""

from typing import Annotated, Dict, Any
from langchain_core.tools import tool


@tool
def search_tool(query: Annotated[str, "The search query to look up information"]) -> str:
    """
    Search for information on the web.
    
    Args:
        query: The search query
        
    Returns:
        Search results as a string
    """
    # This is a placeholder - in a real implementation, you would integrate with a search API
    return f"Search results for: {query}\n1. Result 1\n2. Result 2\n3. Result 3"


@tool
def analyze_data_tool(
    data: Annotated[str, "The data to analyze, as a JSON string or description"]
) -> str:
    """
    Analyze data and provide insights.
    
    Args:
        data: The data to analyze
        
    Returns:
        Analysis results as a string
    """
    # This is a placeholder - in a real implementation, you would perform actual data analysis
    return f"Analysis of data: {data}\n- Key insight 1\n- Key insight 2\n- Key insight 3"


@tool
def calculate_metrics_tool(
    metrics: Annotated[str, "The metrics to calculate, as a JSON string or description"]
) -> str:
    """
    Calculate specific metrics based on provided data.
    
    Args:
        metrics: Description of metrics to calculate
        
    Returns:
        Calculation results as a string
    """
    # This is a placeholder - in a real implementation, you would perform actual calculations
    return f"Calculation results for: {metrics}\n- Metric 1: 0.85\n- Metric 2: 0.72\n- Metric 3: 0.91"


@tool
def generate_code_tool(
    specification: Annotated[str, "The specification for the code to generate"]
) -> str:
    """
    Generate code based on a specification.
    
    Args:
        specification: The specification for the code
        
    Returns:
        Generated code as a string
    """
    # This is a placeholder - in a real implementation, you would use the LLM to generate code
    return f"""
    Generated code for: {specification}
    
    ```python
    def example_function():
        # Implementation based on specification
        print("Hello, world!")
        return True
    
    # Additional code would be generated here
    ```
    """
