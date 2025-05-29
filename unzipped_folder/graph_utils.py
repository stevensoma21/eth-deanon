# graph_utils.py
# === graph_utils.py ===
import networkx as nx
import numpy as np
import pandas as pd
from collections import Counter


def build_transaction_graph(df, filter_min_degree=None):
    """
    Build a directed graph from transaction data.

    Args:
        df: DataFrame with 'from_address' and 'to_address'
        filter_min_degree: Optional minimum out-degree filter

    Returns:
        G_complete: full transaction graph
        G_filtered: filtered graph if threshold provided
    """
    print("Building transaction graph...")
    G_complete = nx.DiGraph()
    for _, row in df.iterrows():
        G_complete.add_edge(row['from_address'], row['to_address'])

    print(f"Graph: {G_complete.number_of_nodes()} nodes, {G_complete.number_of_edges()} edges")

    if filter_min_degree is not None:
        filtered_nodes = [n for n in G_complete.nodes() if G_complete.out_degree(n) > filter_min_degree]
        G_filtered = G_complete.subgraph(filtered_nodes).copy()
        return G_complete, G_filtered

    return G_complete, None


def calculate_entropy(graph, address):
    """
    Compute entropy of outgoing transactions from a given address.
    """
    if graph.out_degree(address) == 0:
        return 0.0

    targets = [t for _, t in graph.out_edges(address)]
    freq = Counter(targets)
    total = sum(freq.values())
    probs = [count / total for count in freq.values()]
    return -sum(p * np.log2(p) for p in probs)


def compute_address_metrics(graph):
    """
    Compute entropy and degree metrics for all nodes in a graph.

    Returns:
        DataFrame with address, degree, entropy
    """
    print("Computing address metrics...")
    degrees = dict(graph.degree())
    filtered_nodes = [node for node, deg in degrees.items() if deg > 1]

    entropy = {node: calculate_entropy(graph, node) for node in filtered_nodes}
    filtered_degrees = {node: degrees[node] for node in filtered_nodes}

    summary_df = pd.DataFrame({
        'address': list(filtered_degrees.keys()),
        'degree': list(filtered_degrees.values()),
        'entropy': [entropy.get(a, 0) for a in filtered_degrees.keys()]
    })

    print(f"Filtered to {len(summary_df)} active addresses")
    return summary_df