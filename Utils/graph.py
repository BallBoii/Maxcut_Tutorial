import networkx as nx
import matplotlib.pyplot as plt
from qiskit_optimization.applications import Maxcut
import numpy as np
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
    # Ensure the bitstring length matches the graph's nodes
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

    # Separate cut edges and non-cut edges
    cut_edges = []
    non_cut_edges = []
    for u, v, data in graph.edges(data=True):
        if (u in S and v in T) or (u in T and v in S):
            cut_edges.append((u, v))
        else:
            non_cut_edges.append((u, v))

    # Add all edges to the bipartite graph for layout consistency
    bipartite_graph.add_edges_from(cut_edges)
    bipartite_graph.add_edges_from(non_cut_edges)

    # Layout for bipartite graph
    pos = nx.drawing.layout.bipartite_layout(bipartite_graph, S)

    # Draw nodes
    nx.draw(bipartite_graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)

    # Draw edges with different colors
    nx.draw_networkx_edges(bipartite_graph, pos, edgelist=cut_edges, edge_color='red', width=2, label="Cut Edges")
    nx.draw_networkx_edges(bipartite_graph, pos, edgelist=non_cut_edges, edge_color='gray', style="dashed", label="Non-Cut Edges")

    # Draw edge labels for weights
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(bipartite_graph, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Bipartite Graph Representation for MaxCut with Highlighted Edges")
    plt.show()

def get_QUBO_Matrix(graph: nx.Graph):
    w = nx.adjacency_matrix(graph).todense()
    max_cut = Maxcut(w)
    qp = max_cut.to_quadratic_program()
    Quadratic = qp.objective.quadratic.to_array()
    linear = qp.objective.linear.to_array()
    np.fill_diagonal(Quadratic, linear)
    return Quadratic


    