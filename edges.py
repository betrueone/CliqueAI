import json
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph

def edges():
    data_paths = [
        "test_data/general_0.1.json",
        "test_data/general_0.2.json",
        "test_data/general_0.4.json",
    ]
    def get_test_data(data_path: str) -> MaximumCliqueOfLambdaGraph:
        with open(data_path, "r") as f:
            data = json.load(f)
        synapse = MaximumCliqueOfLambdaGraph.model_validate(data)
        return synapse

    for data_path in data_paths:
        synapse = get_test_data(data_path)
        number_of_nodes = synapse.number_of_nodes
        adjacency_list = synapse.adjacency_list
        edges = sum(len(neighbors) for neighbors in adjacency_list) // 2
        print(number_of_nodes, edges)
        
if __name__ == "__main__":
    edges()