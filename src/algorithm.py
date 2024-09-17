from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from .middleware import Node


def calculate_probabilities(
    node: str, data: pd.DataFrame, dependencies: List[Node]
) -> Tuple[float, Dict[str, float]]:
    """Calculate failure probabilities for the node and its upstream dependencies."""
    node_fails = data[node] is False
    p_node_fails = node_fails.mean()

    # Calculate independent failure probability
    if not dependencies:
        p_independent_fail = node_fails.mean()
    else:
        deps_pass = data[[dep.name for dep in dependencies]].all(axis=1)
        p_independent_fail = (node_fails & deps_pass).sum() / deps_pass.sum()

    # Calculate conditional failure probabilities for dependencies
    p_node_fails_given_dep_fails = {}
    for dep in dependencies:
        dep_fails = data[dep.name] is False
        result = (node_fails & dep_fails).sum() / dep_fails.sum()
        p_node_fails_given_dep_fails[dep.name] = 0 if np.isnan(result) else result

    return p_node_fails, p_independent_fail, p_node_fails_given_dep_fails


def find_root_cause(
    node: str, data: pd.DataFrame, graph: Dict[str, Node]
) -> Tuple[List[str], float, Dict[str, float]]:
    """Recursively find the root cause of failures, tracing from downstream to upstream."""

    dependencies = graph[node].parents  # These are upstream nodes

    p_node_fails, p_independent_fail, p_node_fails_given_dep_fails = (
        calculate_probabilities(node, data, dependencies)
    )

    print(f"Analyzing node: {node}")
    print(f"Overall failure probability for this node: {p_node_fails:.4f}")
    print(f"Independent failure probability: {p_independent_fail:.4f}")
    print(
        f"Node failure because dep fails: {max([v for _, v in p_node_fails_given_dep_fails.items()], default=0)}"
    )
    print("Conditional failure probabilities given upstream dependency failures:")
    for dep, prob in p_node_fails_given_dep_fails.items():
        print(f"  P({node} fails | {dep} fails): {prob:.4f}")
    print()

    # Check if independent failure is more likely than any upstream dependency failure
    if p_independent_fail > max(p_node_fails_given_dep_fails.values(), default=0):
        return [node], p_independent_fail, p_node_fails_given_dep_fails

    if not dependencies:
        return [node], p_independent_fail, p_node_fails_given_dep_fails

    max_dep = max(p_node_fails_given_dep_fails, key=p_node_fails_given_dep_fails.get)
    upstream_path, upstream_independent_prob, upstream_final_probs = find_root_cause(
        max_dep, data, graph
    )

    return [node] + upstream_path, upstream_independent_prob, upstream_final_probs


def improve_system(
    data: pd.DataFrame, dag: Dict[str, Node]
) -> Tuple[List[str], float, Dict[str, float]]:
    downstream_node = [node for node in dag.values() if node.ending_node is True][0]
    downstream_node_name = downstream_node.name
    """Entry point for the root cause analysis, starting from the most downstream node."""
    path, independent_prob, final_probs = find_root_cause(
        downstream_node_name, data, dag
    )

    print("\nRoot cause analysis complete.")
    print(f"Debug path (from downstream to upstream): {' -> '.join(path)}")
    print(f"Most likely root cause (most upstream issue): {path[-1]}")
    print(f"Independent failure probability of root cause: {independent_prob:.4f}")
    print("Conditional failure probabilities given root cause's dependency failures:")
    for dep, prob in final_probs.items():
        print(f"  P({path[-1]} fails | {dep} fails): {prob:.4f}")

    if independent_prob > max(final_probs.values(), default=0):
        print(f"The most likely cause is an independent failure in node {path[-1]}")
    else:
        most_likely_dep = max(final_probs, key=final_probs.get)
        print(f"The most likely cause is a failure in dependency: {most_likely_dep}")

    return path, independent_prob, final_probs
