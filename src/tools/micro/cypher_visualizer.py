"""
Visualize a graph from Cypher query results using Pyvis and NetworkX.
"""
from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Any

from typing import List, Dict, Any

def visualize_cypher_results(cypher_results: List[Dict[str, Any]], output_file: str = "cypher_graph.html"):
    from pyvis.network import Network
    import networkx as nx
    import matplotlib.pyplot as plt

    net = Network(notebook=False, cdn_resources='in_line', height="800px", width="100%")
    G = nx.DiGraph()

    unique_nodes = set()
    unique_edges = set()
    node_metadata = {}

    for result in cypher_results:
        path = result.get('path', [])
        node_types = result.get('nodeTypes', [])

        node_idx = 0  # Index to track nodeTypes separately
        
        for i in range(0, len(path), 2):
            if i < len(path):
                node = path[i]
                node_id = node.get('id')

                if node_id and node_id not in unique_nodes:
                    unique_nodes.add(node_id)

                    # Grab the nodeType from nodeTypes list
                    if node_idx < len(node_types):
                        node_meta_type = node_types[node_idx][0]  # nodeTypes is a list of list
                    else:
                        node_meta_type = "Unknown"

                    node_metadata[node_id] = {
                        "id": node_id,
                        "meta": node_meta_type,
                        "properties": node.get('properties', {})
                    }

                    G.add_node(node_id)

                node_idx += 1  # Move to the next nodeType

            if i + 1 < len(path) and i + 2 < len(path):
                source_id = path[i].get('id')
                relationship = path[i + 1]
                target_id = path[i + 2].get('id')

                if source_id and target_id:
                    edge_key = (source_id, target_id, relationship)
                    if edge_key not in unique_edges:
                        unique_edges.add(edge_key)
                        G.add_edge(source_id, target_id, label=relationship)

    # Group nodes by their meta type
    unique_metas = list(set(meta["meta"] for meta in node_metadata.values()))
    color_map = plt.get_cmap("tab10")
    meta_colors = {m: color_map(i / len(unique_metas)) for i, m in enumerate(unique_metas)}
    meta_colors_rgba = {
        m: f'rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 0.8)'
        for m, c in meta_colors.items()
    }

    degrees = dict(G.degree())
    min_size, max_size = 15, 50
    max_degree = max(degrees.values()) if degrees else 1
    size_scale = {
        node: min_size + (max_size - min_size) * (deg / max_degree)
        for node, deg in degrees.items()
    }

    for node_id in G.nodes():
        metadata = node_metadata.get(node_id, {})
        node_meta = metadata.get("meta", "Unknown")
        color = meta_colors_rgba.get(node_meta, "gray")

        net.add_node(
            node_id,
            label=node_id,
            size=size_scale[node_id],
            color=color,
            title=f"<b>{node_meta}</b><br>{node_id}"
        )

    relationship_types = list(set(attr.get("label", "") for _, _, attr in G.edges(data=True)))
    rel_color_map = plt.get_cmap("Set2")
    rel_colors = {
        rel_type: f'rgba({int(rel_color_map(i / len(relationship_types))[0]*255)}, {int(rel_color_map(i / len(relationship_types))[1]*255)}, {int(rel_color_map(i / len(relationship_types))[2]*255)}, 0.9)'
        for i, rel_type in enumerate(relationship_types)
    }

    for source, target, attr in G.edges(data=True):
        rel_label = attr.get("label", "")
        edge_color = rel_colors.get(rel_label, "#888888")
        net.add_edge(source, target, title=rel_label, label=rel_label, arrows='to', length=300, color=edge_color)

    net.save_graph(output_file)

    # Build the correct legend
    legend_html = """
    <div id="legend" style="position: absolute; top: 10px; left: 10px; background: white; padding: 10px; border-radius: 8px; box-shadow: 0px 0px 5px rgba(0,0,0,0.2); font-family: Arial, sans-serif; z-index: 1000;">
        <h4 style="margin: 0; padding-bottom: 5px;">Node Legend (by Node Type)</h4>
    """
    for meta_type, color in meta_colors_rgba.items():
        legend_html += f'<div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 15px; height: 15px; background:{color}; margin-right: 5px; border-radius: 50%;"></div> {meta_type}</div>'

    legend_html += """
        <h4 style="margin: 5px 0; padding-bottom: 5px;">Relationship Legend</h4>
    """
    for rel_type in sorted(relationship_types):
        rel_color = rel_colors.get(rel_type, "#888888")
        legend_html += f'<div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 20px; height: 3px; background: {rel_color}; margin-right: 5px;"></div> {rel_type}</div>'

    legend_html += "</div>"

    with open(output_file, "r", encoding="utf-8") as file:
        html_content = file.read()

    html_content = html_content.replace("</body>", legend_html + "</body>")

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"Graph visualization saved as {output_file}")

    return {
        "nodes": len(G.nodes()),
        "edges": len(G.edges()),
        "node_types": unique_metas,
        "relationship_types": list(set(attr.get("label", "") for _, _, attr in G.edges(data=True)))
    }



# def visualize_cypher_results(cypher_results: List[Dict[str, Any]], output_file: str = "cypher_graph.html"):
#     """
#     Visualize Cypher query results using Pyvis.
    
#     Args:
#         cypher_results: List of path results from Cypher query
#         output_file: Output HTML file name
#     """
#     # Create Pyvis network
#     net = Network(notebook=False, cdn_resources='in_line', height="800px", width="100%")
    
#     # Create a NetworkX graph
#     G = nx.DiGraph()  # Using DiGraph for directed relationships
    
#     # Track unique nodes and edges to avoid duplicates
#     unique_nodes = set()
#     unique_edges = set()
    
#     # Extract node metadata (type from ID prefix)
#     node_metadata = {}
    
#     # Process each path in the results
#     for result in cypher_results:
#         path = result.get('path', [])
        
#         # Paths alternate between nodes and relationships
#         for i in range(0, len(path), 2):
#             # Add node
#             if i < len(path):
#                 node = path[i]
#                 node_id = node.get('id')
#                 if node_id and node_id not in unique_nodes:
#                     unique_nodes.add(node_id)
#                     # Extract node type from ID (assuming format like "Type.Name")
#                     node_parts = node_id.split('.')
#                     node_type = node_parts[0] if len(node_parts) > 0 else "Unknown"
                    
#                     # Store node metadata
#                     node_metadata[node_id] = {
#                         "id": node_id,
#                         "type": node_type,
#                         "properties": {}
#                     }
                    
#                     # Add node to NetworkX graph
#                     G.add_node(node_id)
            
#             # Add relationship/edge
#             if i + 1 < len(path) and i + 2 < len(path):
#                 source_id = path[i].get('id')
#                 relationship = path[i + 1]  # This is a string like "USES"
#                 target_id = path[i + 2].get('id')
                
#                 if source_id and target_id:
#                     edge_key = (source_id, target_id, relationship)
#                     if edge_key not in unique_edges:
#                         unique_edges.add(edge_key)
#                         # Add edge to NetworkX graph
#                         G.add_edge(source_id, target_id, label=relationship)
    
#     # Get unique types for coloring
#     unique_types = list(set(meta["type"] for meta in node_metadata.values()))
#     color_map = plt.get_cmap("tab10")
#     type_colors = {t: color_map(i / len(unique_types)) for i, t in enumerate(unique_types)}
#     type_colors_rgba = {
#         t: f'rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 0.8)' 
#         for t, c in type_colors.items()
#     }
    
#     # Degree-based sizing
#     degrees = dict(G.degree())
#     min_size, max_size = 15, 50
#     max_degree = max(degrees.values()) if degrees else 1
#     size_scale = {
#         node: min_size + (max_size - min_size) * (deg / max_degree)
#         for node, deg in degrees.items()
#     }
    
#     # Add nodes to Pyvis with proper styling
#     for node_id in G.nodes():
#         metadata = node_metadata.get(node_id, {})
#         node_type = metadata.get("type", "Unknown")
#         color = type_colors_rgba.get(node_type, "gray")
        
#         # Show full node ID as the label
#         # Add node to Pyvis network
#         net.add_node(
#             node_id,
#             label=node_id,  # Using full node_id as the label
#             size=size_scale[node_id],
#             color=color,
#             title=f"<b>{node_type}</b><br>{node_id}"
#         )
    
#     # Get all unique relationship types
#     relationship_types = list(set(attr.get("label", "") for _, _, attr in G.edges(data=True)))
    
#     # Create colors for each relationship type
#     rel_color_map = plt.get_cmap("Set2")  # Using a different colormap for relationships
#     rel_colors = {
#         rel_type: f'rgba({int(rel_color_map(i / len(relationship_types))[0]*255)}, {int(rel_color_map(i / len(relationship_types))[1]*255)}, {int(rel_color_map(i / len(relationship_types))[2]*255)}, 0.9)'
#         for i, rel_type in enumerate(relationship_types)
#     }
    
#     # Add edges to Pyvis with longer length and different colors
#     for source, target, attr in G.edges(data=True):
#         rel_label = attr.get("label", "")
#         edge_color = rel_colors.get(rel_label, "#888888")  # Default gray if not found
#         # Adding longer length to edges (300 instead of default)
#         net.add_edge(source, target, title=rel_label, label=rel_label, arrows='to', length=300, color=edge_color)
    
#     # Save graph
#     net.save_graph(output_file)
    
#     # Build legend
#     legend_html = """
#     <div id="legend" style="position: absolute; top: 10px; left: 10px; background: white; padding: 10px; border-radius: 8px; box-shadow: 0px 0px 5px rgba(0,0,0,0.2); font-family: Arial, sans-serif; z-index: 1000;">
#         <h4 style="margin: 0; padding-bottom: 5px;">Node Legend</h4>
#     """
    
#     for node_type, color in type_colors_rgba.items():
#         legend_html += f'<div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 15px; height: 15px; background:{color}; margin-right: 5px; border-radius: 50%;"></div> {node_type}</div>'
    
#     legend_html += """
#         <h4 style="margin: 5px 0; padding-bottom: 5px;">Relationship Legend</h4>
#     """
    
#     for rel_type in sorted(relationship_types):
#         rel_color = rel_colors.get(rel_type, "#888888")
#         legend_html += f'<div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 20px; height: 3px; background: {rel_color}; margin-right: 5px;"></div> {rel_type}</div>'
    
#     legend_html += "</div>"
    
#     # Inject legend into the HTML
#     with open(output_file, "r", encoding="utf-8") as file:
#         html_content = file.read()
    
#     html_content = html_content.replace("</body>", legend_html + "</body>")
    
#     with open(output_file, "w", encoding="utf-8") as file:
#         file.write(html_content)
    
#     print(f"Graph visualization saved as {output_file}")
    
#     # Return graph stats
#     return {
#         "nodes": len(G.nodes()),
#         "edges": len(G.edges()),
#         "node_types": unique_types,
#         "relationship_types": list(set(attr.get("label", "") for _, _, attr in G.edges(data=True)))
#     }