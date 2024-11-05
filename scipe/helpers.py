from collections import defaultdict
from dataclasses import dataclass
from typing import List
from langgraph import graph


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


def convert_edges_to_dag(graph: graph) -> defaultdict:
    edges = graph.to_json()['graph']['edges'] # Extract edges for source/target mapping from langgraph
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

def construct_prompt(input_: str, output_: str, special_instructions=None):
    prompt = f"""You are a validator who is diligent and careful. When things are incorrect, you call it out and nothing gets past you.
        Given a input and ouput, your goal is to check if the output followed the directions in the input.

        Special instructions: 
        1. If the input task was to format something as a python list of strings, you can ignore that. 
        2. If the task is to extract two words, and more words have been extracted, ignore that. That output is correct.
        3. If there is a word-limit, you can ignore that as well as long as the output is close to the requested word limit.

        Analyze and output in JSON format with keys: "reason" (the reason why this is correct or incorrect), "validation" (1 for correct and 0 for incorrect)
        
        Please, absolutely no preamble in the response, just a json output. You'll be penalized otherwise.

        Input: {input_}
        Output: {output_} """
    
    if special_instructions is not None:
        instructions = f"""

        Please follow special instructions:\n{special_instructions}
        """
        prompt = prompt + instructions
    
    return prompt