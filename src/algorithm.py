import logging
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from .helpers import Node


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
EPSILON = 1e-10

@dataclass
class NodeResult:
    name: str
    failure_prob: float # Overall failure probability
    independent_failure_prob: float
    dependent_failure_prob: Dict[str, float] # Failure probablity due to dependency
    dependecies: List[str] # All dependencies
    is_root_cause: bool = False
    
@dataclass
class EvaluationResult:
    root_cause: str
    debug_path: List[str]
    node_results: Dict[str, NodeResult]
    
    def to_json(self) -> dict:
        return {
            "root_cause": self.root_cause,
            "debug_path": self.debug_path,
            "node_results": {
                name: {
                    "overall_failure_probability": result.failure_prob,
                    "independent_failure_probability": result.independent_failure_prob,
                    "conditional_failure_probabilities": result.dependent_failure_prob,
                    "dependencies": result.dependecies,
                    "is_root_cause": result.is_root_cause
                }
                for name, result in self.node_results.items()
            }
        }
    
def calculate_probabilities(
    node: str, data: pd.DataFrame, dependencies: List[Node]
) -> Tuple[float, float, Dict[str, float]]:   
    """Generate probabilities of the following:
    1. Overall failure
    2. Independent failure of a node
    3. Dependent failure of a node

    Args:
        node (str): Name of the node
        data (pd.DataFrame): LLM evaluations
        dependencies (List[Node]): All dependencies of the node

    Returns:
        Tuple[float, float, Dict[str, float]]
    """
    node_fails = data[node] == False  # noqa: E712
    p_node_fails = node_fails.mean()

    # Calculate independent failure probability
    if not dependencies:
        p_independent_fail = node_fails.mean()
    else:
        deps_pass = data[[dep.name for dep in dependencies]].all(axis=1)
        p_independent_fail = (node_fails & deps_pass).sum() / (deps_pass.sum() + EPSILON)

    # Calculate conditional failure probabilities for dependencies
    p_node_fails_given_dep_fails = {}
    for dep in dependencies:
        dep_fails = data[dep.name] == False  # noqa: E712
        result = (node_fails & dep_fails).sum() / (dep_fails.sum() + EPSILON)
        p_node_fails_given_dep_fails[dep.name] = round(result, 3)

    return p_node_fails, p_independent_fail, p_node_fails_given_dep_fails

def find_root_cause(
    node: str,
    data: pd.DataFrame,
    graph: Dict[str, Node],
    node_results: Dict,
    debug_path: List,
    verbose: bool = True
) -> Tuple[List[str], float, Dict[str, float]]:
    """Recursively find the root cause of failures, tracing from downstream to upstream.

    Args:
        node (str): Downstream Node Name
        data (pd.DataFrame): LLM evaluations
        graph (Dict[str, Node]): Parent/child relationships of nodes
        node_results (Dict): Failure rate of a node
        debug_path (List): Node path from downstream to upstream
        verbose (bool, optional): Log info. Defaults to True.
    Returns:
        Tuple[List[str], float, Dict[str, float]]: _description_
    """
    dependencies = graph[node].parents  # These are upstream nodes

    p_node_fails, p_independent_fail, p_node_fails_given_dep_fails = (
        calculate_probabilities(node, data, dependencies)
    )
    # Standardize results
    node_result = NodeResult(
        name=node,
        failure_prob=round(p_node_fails, 3),
        independent_failure_prob=round(p_independent_fail, 3),
        dependent_failure_prob={key: round(value, 3) for key, value in p_node_fails_given_dep_fails.items()},
        dependecies=[dep.name for dep in dependencies]
    )
    
    node_results[node] = node_result
    debug_path.append(node)

    if verbose:
        logger.info(f"Analyzing node: {node}")
        logger.info(f"Overall failure probability: {p_node_fails:.4f}")
        logger.info(f"Independent failure probability: {p_independent_fail:.4f}")
        logger.info("Conditional failure probabilities:")
        for dep, prob in p_node_fails_given_dep_fails.items():
            logger.info(f"  P({node} fails | {dep} fails): {prob:.4f}")

    # Check if independent failure is more likely than any upstream dependency failure
    if p_independent_fail > max(p_node_fails_given_dep_fails.values(), default=0):
        node_result.is_root_cause = True
        return node_result
    
    if not dependencies:
        node_result.is_root_cause = True
        return node_result

    # Get the largest depenedency that also fails
    max_dep = max(p_node_fails_given_dep_fails, key=p_node_fails_given_dep_fails.get)

    upstream_result = find_root_cause(
        node=max_dep,
        data=data,
        graph=graph,
        node_results=node_results,
        debug_path=debug_path,
        verbose=verbose
        )
    
    return upstream_result if upstream_result.is_root_cause else node_result

def find_problematic_node(
        data: pd.DataFrame,
        graph: Dict[str, Node],
        verbose: bool = True
):
    """Find the problematic node in the graph based on failure probabilities.

    Args:
        data (pd.DataFrame): LLM Evaluations
        graph (Dict[str, Node]): Parent/child relationships of nodes
        verbose (bool, optional): Get logs. Defaults to True.

    Returns:
        EvaluationResult (EvaluationResult): Result of the evalutaion
    """
    node_results = {}
    debug_path = []

    downstream_node = next(node for node in graph.values() if node.ending_node is True)
    downstream_node_name = downstream_node.name

    root_cause_result = find_root_cause(
        node=downstream_node_name,
        data=data,
        graph=graph,
        node_results=node_results,
        debug_path=debug_path,
        verbose=verbose
        )
    
    # Standardadize the result
    evaluation_result = EvaluationResult(
        root_cause=root_cause_result.name,
        debug_path=debug_path,
        node_results=node_results
    )

    if verbose:
        logger.info("\n\nRoot cause analysis complete.")
        logger.info(f"Debug path (from downstream to upstream): {' -> '.join(debug_path)}")
        logger.info(f"Most likely root cause: {root_cause_result.name}")
        logger.info(f"Independent failure probability of root cause: {root_cause_result.independent_failure_prob:.4f}")

    return evaluation_result
