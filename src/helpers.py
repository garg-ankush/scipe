import json

def read_graph(path_to_json: str):
    with open(path_to_json, "r") as json_file:
        graph_json = json.load(json_file)
    return graph_json["edges"]