from typing import Any
import os
import sys

# Add the pmc directory to sys.path so pmc_core.so can be found
# This is necessary when running through PM2 or other process managers
_pmc_dir = os.path.dirname(os.path.abspath(__file__))
if _pmc_dir not in sys.path:
    sys.path.insert(0, _pmc_dir)

import numpy as np

try:
    from pmc_core import max_clique as _max_clique_core
except ImportError:
    raise ImportError(
        "pmc_core module not found. Please build the Python extension by running: make python"
    )

def pmc(ei, ej, index_offset=0, algorithm=0, threads=2, time_limit=20,
        heu_strat="rand", vertex_search_order="rand", neigh_edge_order="rand"):
    """
    Find maximum clique in a graph.
    
    Parameters:
    -----------
    ei : array-like
        Source vertex indices of edges (0-indexed by default)
    ej : array-like
        Target vertex indices of edges (0-indexed by default)
    index_offset : int, optional
        Index offset (0 for 0-indexed vertices, 1 for 1-indexed vertices). Default is 0.
    algorithm : int, optional
        Algorithm to use: 0 = full, 1 = no neighborhood cores, 2 = only basic k-core pruning. Default is 0.
    threads : int, optional
        Number of threads to use. If 0, uses all available cores. Default is 0.
    time_limit : float, optional
        Time limit in seconds. Default is 3600.0 (1 hour).
    heu_strat : str, optional
        Heuristic strategy. Default is "kcore".
    vertex_search_order : str, optional
        Vertex search ordering. Default is "deg".
    neigh_edge_order : str, optional
        Neighborhood edge ordering. Default is "".
    
    Returns:
    --------
    numpy.ndarray
        Array of vertex indices in the maximum clique (using the same indexing as input)
    
    Notes:
    ------
    The C library expects edges where ei > ej (upper triangular part).
    The library will filter edges internally, so you can pass all edges.
    """
    # Convert to numpy arrays if needed
    ei = np.asarray(ei, dtype=np.int32)
    ej = np.asarray(ej, dtype=np.int32)
    
    # Ensure arrays are contiguous
    ei = np.ascontiguousarray(ei, dtype=np.int32)
    ej = np.ascontiguousarray(ej, dtype=np.int32)
    
    # Call the pybind11 wrapped function
    return _max_clique_core(ei, ej, index_offset, algorithm, threads, time_limit,
                           heu_strat, vertex_search_order, neigh_edge_order)

def pmc_algorithm(number_of_nodes: int, adjacency_list: list[list[int]]) -> list[int]:
    """
    Find maximum clique in a graph.
    """
    ei = []
    ej = []
    for i, neighbors in enumerate(adjacency_list):
        for j in neighbors:
            if i > j:
                ei.append(i)
                ej.append(j)
    ei = np.array(ei, dtype=np.int32)
    ej = np.array(ej, dtype=np.int32)

    return list(pmc(ei, ej))

def is_valid_maximum_clique(number_of_nodes: int, adjacency_list: list[list[int]], nodes: list[int]) -> bool:
    """
    Returns True if the given nodes form a clique in the graph.
    """
    node_set = set(nodes)
    # 0. Check if the node set is empty
    if len(node_set) == 0:
        return False

    # 1. Check for duplicates or out-of-range nodes
    if len(node_set) != len(nodes):
        print(f"Duplicate nodes: {nodes}")
        return False
    if not node_set.issubset(range(number_of_nodes)):
        return False

    # 2. Check if all pairs of nodes are connected (i.e., form a clique)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if nodes[j] not in adjacency_list[nodes[i]]:
                return False

    # 3. Check if any other node can be added to form a larger clique
    all_nodes = set(range(number_of_nodes))
    remaining_nodes = all_nodes - node_set
    for candidate in remaining_nodes:
        # Candidate must be connected to all nodes in the current clique
        if node_set.issubset(adjacency_list[candidate]):
            return False  # Clique can be extended, so it's not maximum

    return True

if __name__ == "__main__":
    # Example usage
    # Create a simple triangle graph: edges must have ei > ej for the library
    # Triangle: 0-1, 1-2, 2-0 becomes: (1,0), (2,0), (2,1)
    import json
    import os

    # Load sample.json adjacency list
    data_dir = "test_data"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' does not exist.")

    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r") as f:
            data = json.load(f)
        print(f"\n===== Processing: {filename} =====")
        number_of_nodes = data.get("number_of_nodes")
        adjacency_list = data.get("adjacency_list")
        if number_of_nodes is None or adjacency_list is None:
            print(f"Skipping {filename}: missing 'number_of_nodes' or 'adjacency_list'")
            continue

        # Build edge list from adjacency list (undirected, collect only i > j to match library expectation)
        ei = []
        ej = []
        for i, neighbors in enumerate(adjacency_list):
            for j in neighbors:
                if i > j:
                    ei.append(i)
                    ej.append(j)
        ei = np.array(ei, dtype=np.int32)
        ej = np.array(ej, dtype=np.int32)

        # Call pmc
        computed_clique = list[int](pmc(ei, ej))

        # Validate result
        valid = is_valid_maximum_clique(number_of_nodes, adjacency_list, computed_clique)
        print(f"Result is {'✅ valid maximum clique' if valid else '❌ invalid clique'}")