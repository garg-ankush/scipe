from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict


class Node:
    def __init__(self, name):
        self.name: str = name
        self.parents: List = []
        self.starting_node: bool = None
        self.ending_node: bool = None

    def add_parent(self, parent):
        self.parents.append(parent)

    def __repr__(self):
        return self.name


@dataclass
class Edge:
    source: str
    target: str
    data: None
    conditional: None


def convert_edges_to_dag(edges: List[Dict]) -> defaultdict:
    edges = [
        Edge(
            source=edge["source"],
            target=edge["target"],
            data=edge.get("data", None),
            conditional=edge.get("conditional", None),
        )
        for edge in edges
    ]
    # Take the edges from Langgraph and create a parent/child relationship
    nodes = defaultdict(list)

    for edge in edges:
        if edge.source == "__start__":
            continue
        nodes[edge.source] = Node(edge.source)

    for edge in edges:
        if edge.target == "__end__":
            node = nodes.get(edge.source, None)
            node.ending_node = True
            continue

        if edge.source == "__start__":
            node = nodes.get(edge.target, None)
            node.starting_node = True
            continue

        node = nodes.get(edge.target, None)
        parent_node = nodes.get(edge.source, None)
        node.add_parent(parent_node)

    return nodes
