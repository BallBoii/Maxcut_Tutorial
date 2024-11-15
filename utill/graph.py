import networkx as nx
import matplotlib.pyplot as plt
import random

def draw_graph(graph):
    # Get positions for all nodes
    pos = nx.spring_layout(graph)

    # Draw the nodes and edges
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')

    # Extract edge weights
    edge_labels = nx.get_edge_attributes(graph, 'weight')

    # Draw edge labels
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)

    # Show the plot
    plt.show()

def getRandomGraph(n, p=0.5, weighed=True, seed=None):
    G = nx.Graph()
    V = range(n)
    G.add_nodes_from(V)
    
    if seed is not None:
        random.seed(seed)
    
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() > 0.5:
                w = int(random.uniform(1, 10))
                G.add_edge(i, j, weight = w)
    return G


def draw_partition_graph(graph: nx.Graph, bitstring: list[int]):
    nodes = list(graph.nodes)
    if len(bitstring) != len(nodes):
        raise ValueError("Bitstring length must match the number of nodes.")

    # Partition nodes based on the bitstring
    S = [nodes[i] for i in range(len(nodes)) if bitstring[i] == 0]
    T = [nodes[i] for i in range(len(nodes)) if bitstring[i] == 1]

    # Create a bipartite graph
    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from(S, bipartite=0)  # Set S
    bipartite_graph.add_nodes_from(T, bipartite=1)  # Set T

    # Separate cut edges for highlighting
    cut_edges = []
    for u, v, data in graph.edges(data=True):
        if (u in S and v in T) or (u in T and v in S):
            bipartite_graph.add_edge(u, v, weight=data.get('weight', 1))
            cut_edges.append((u, v))

    # Layout for bipartite graph
    pos = nx.drawing.layout.bipartite_layout(bipartite_graph, S)

    # Draw nodes and edges
    nx.draw(bipartite_graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)

    # Highlight cut edges in a different color
    nx.draw_networkx_edges(bipartite_graph, pos, edgelist=cut_edges, edge_color='red', width=2)

    # Draw edge labels for weights
    edge_labels = nx.get_edge_attributes(bipartite_graph, 'weight')
    nx.draw_networkx_edge_labels(bipartite_graph, pos, edge_labels=edge_labels, font_size=8)

    # Display the graph
    plt.title("Bipartite Graph Representation for MaxCut with Highlighted Edges")
    plt.show()
    