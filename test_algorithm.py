import json
from collections import Counter
import numpy as np
import time
from CliqueAI.clique_algorithms import (networkx_algorithm,
                                        scattering_clique_algorithm)
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph

data_paths = [
    # "test_data/general_0.1.json",
    # "test_data/general_0.2.json",
    "test_data/general_0.4.json",
]


def get_test_data(data_path: str) -> MaximumCliqueOfLambdaGraph:
    with open(data_path, "r") as f:
        data = json.load(f)
    synapse = MaximumCliqueOfLambdaGraph.model_validate(data)
    return synapse


def check_clique(adjacency_list: list[list[int]], clique: list[int]) -> bool:
    clique_set = set(clique)
    for i in range(len(clique)):
        node = clique[i]
        neighbors = set(adjacency_list[node])
        if not clique_set.issubset(neighbors.union({node})):
            return False
    for v in range(len(adjacency_list)):
        if v in clique_set:
            continue
        if all(v in adjacency_list[node] for node in clique):
            return False
    return True

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

def run(algorithm, synapse: MaximumCliqueOfLambdaGraph):
    codec = GraphCodec()
    adjacency_matrix = codec.decode_matrix(synapse.encoded_matrix)
    adjacency_list = codec.matrix_to_list(adjacency_matrix)
    maximum_clique = algorithm(synapse.number_of_nodes, adjacency_list)
    clique_check = check_clique(adjacency_list, maximum_clique)
    if not clique_check:
        print("Invalid clique found by algorithm!")
    else:
        print(f"Clique size: {len(maximum_clique)}")

def optimality(number_of_nodes: int, adjacency_list: list[list[int]], clique: list[int]) -> tuple[float, float, float, float]:
    """
    Calculate the optimality scores for a single response.
    """
    val = 1 if is_valid_maximum_clique(number_of_nodes, adjacency_list, clique) else 0
    size = len(clique) * val
    
    if size == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # For single response, max_size = size, rel = 1.0, pr = 0.0
    max_size = size
    rel = 1.0 if max_size > 0 else 0.0
    pr = 0.0  # No other responses to compare
    
    omega = np.exp(-pr / rel) if val == 1 and rel > 0 else 0.0
    max_omega = omega
    omega_normalized = 1.0 if omega > 0 else 0.0
    
    return rel, pr, omega, omega_normalized

def diversity_score(number_of_nodes: int, adjacency_list: list[list[int]], clique: list[int], all_responses: list[list[int]] = None) -> float:
    """
    Calculate the diversity score for a single response.
    If all_responses is None, assumes this is the only response (uniqueness = 1.0).
    """
    val = 1 if is_valid_maximum_clique(number_of_nodes, adjacency_list, clique) else 0
    
    if all_responses is None:
        # Single response is always unique
        unq = 1.0
    else:
        canonical_response = tuple(sorted(clique))
        canonical_responses = [tuple(sorted(r)) for r in all_responses]
        counts = Counter(canonical_responses)
        unq = 1.0 / counts[canonical_response]
    
    delta = val * unq
    # For single response, normalization is just the value itself
    delta_normalized = delta
    return delta_normalized

def get_score(
    number_of_nodes: int, adjacency_list: list[list[int]], clique: list[int], difficulty: float,
    all_responses: list[list[int]] = None
) -> tuple[float, float, float, float, float, float]:
    """
    Compute normalized scores for a single response.
    """
    rel, pr, omega, optimality_score = optimality(number_of_nodes, adjacency_list, clique)
    diversity_val = diversity_score(number_of_nodes, adjacency_list, clique, all_responses)
    
    rewards = optimality_score * (1 + difficulty) + diversity_val
    return rel, pr, omega, optimality_score, diversity_val, rewards

def main():
    for data_path in data_paths:
        synapse = get_test_data(data_path)
        print(f"Testing data from {data_path} with {synapse.number_of_nodes} nodes")
        # put your algorithm here

        def timed_run(algorithm, synapse):
            start = time.perf_counter()
            results = algorithm(synapse.number_of_nodes, synapse.adjacency_list)
            elapsed = time.perf_counter() - start
            for result in results:
                clique_check = is_valid_maximum_clique(synapse.number_of_nodes, synapse.adjacency_list, result)
                if not clique_check:
                    print(f"❌ Invalid clique found by {algorithm.__name__}!")
                else:
                    print(f"🎉 {algorithm.__name__}: Clique size: {len(result)} | Elapsed time: {elapsed:.4f} seconds")
                    print(result)
                    rel, pr, omega, optimality_score, diversity_val, rewards = get_score(synapse.number_of_nodes, synapse.adjacency_list, result, 0.1)
                    print(f"Rel: {rel}")
                    print(f"PR: {pr}")
                    print(f"Omega: {omega}")
                    print(f"Optimality score: {optimality_score}")
                    print(f"Diversity val: {diversity_val}")
                    print(f"Rewards: {rewards}")

        # timed_run(networkx_algorithm, synapse)
        # timed_run(scattering_clique_algorithm, synapse)

def test_clique_removal():
    for data_path in data_paths:
        synapse = get_test_data(data_path)
        print(f"Testing data from {data_path} with {synapse.number_of_nodes} nodes")
        # put your algorithm here
        import networkx as nx
        dict_of_lists = {i: synapse.adjacency_list[i] for i in range(synapse.number_of_nodes)}
        graph = nx.from_dict_of_lists(dict_of_lists)
        # max_clique = nx.approximation.max_clique(graph)
        # i_set, cliques = nx.approximation.clique_removal(graph)
        # max_independent_set = nx.approximation.maximum_independent_set(graph)
        # print("Max independent set:", max_independent_set)
        # print("Cliques:", cliques)
        # print("Size:", size)
        # print("Max clique:", max_clique)
        # print("I set:", i_set)
        
        # Enumerate cliques, find true maximum size, then keep cliques >= 90% of max size (up to 100)
        cliques = nx.find_cliques(graph)
        max_len = nx.approximation.large_clique_size(graph)
        print("Max clique size found:", max_len)

        candidates: list[list[int]] = []
        if max_len > 0:
            threshold = int(max_len * 1)
            cliques = nx.find_cliques(graph)
            for clique in cliques:
                if not check_clique(synapse.adjacency_list, clique):
                    continue
                l = len(clique)
                if l >= threshold:
                    print(f"Clique size: {l}, clique: {clique}")
                    candidates.append(list(clique))
                    if len(candidates) >= 100:
                        break

        print("Maximum clique size found:", max_len)
        print(f"Candidates (len >= 90% of max, limit 100, threshold={int(max_len * 0.9)}):", candidates)

if __name__ == "__main__":
    test_clique_removal()
