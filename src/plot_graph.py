import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt


def plot_network(
    G,
    node_size=1000,
    node_color="skyblue",
    edge_color="gray",
    font_size=10,
    title="Network Graph",
    figsize=(12, 8),
    with_labels=True,
    layout="spring",
    palette="husl",
):
    """
    Plot a network graph with seaborn-style aesthetics.

    Parameters:
    -----------
    G : networkx.Graph
        The network graph to visualize
    node_size : int or list
        Size of nodes (can be a single value or list for different sizes)
    node_color : str or list
        Color of nodes (can be a single value or list for different colors)
    edge_color : str
        Color of edges
    font_size : int
        Size of node labels
    title : str
        Title of the plot
    figsize : tuple
        Figure size (width, height)
    with_labels : bool
        Whether to show node labels
    layout : str
        Type of layout ('spring', 'circular', 'random', 'shell')
    palette : str
        Seaborn color palette to use if node_color is not specified

    Returns:
    --------
    fig, ax : tuple
        Matplotlib figure and axis objects
    """
    # Set the style
    sns.set_style("whitegrid")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Choose layout
    layouts = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "random": nx.random_layout,
        "shell": nx.shell_layout,
    }
    pos = layouts.get(layout, nx.spring_layout)(G)

    # If node_color is not specified, use seaborn palette
    if isinstance(node_color, str) and node_color == "skyblue":
        colors = sns.color_palette(palette, n_colors=len(G.nodes()))
    else:
        colors = node_color

    # Draw the network
    nx.draw(
        G,
        pos,
        node_color=colors,
        node_size=node_size,
        edge_color=edge_color,
        with_labels=with_labels,
        font_size=font_size,
        font_weight="bold",
        ax=ax,
    )

    # Add title
    plt.title(title, fontsize=font_size + 4, pad=20)

    return fig, ax
