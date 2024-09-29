# SCIPE - Systematic Chain Improvement and Problem Evaluation
### It helps you find bad nodes in LLM chains.

SCIPE is a powerful tool for evaluating and diagnosing LLM (Large Language Model) graphs or chains. It assesses LLM responses and employs a custom algorithm to identify problematic nodes within the LLM chain.

## Features

- Evaluates LLM responses within simple LLM Graphs (mainly [LangGraph](https://langchain-ai.github.io/langgraph/))
- Diagnoses problematic nodes in LLM graphs
- Provides failure rates of various nodes that make up the LLM chain/graph
- Supports various LLM frameworks (uses [LiteLLM](https://github.com/BerriAI/litellm) underneath the hood)

## Why Use SCIPE?

As AI application developers, we often overlook the critical step of evaluating LLM chains during the building phase. SCIPE simplifies this process by allowing developers to run their minimum set of prompts and responses (we recommend atleast 10 examples) through the tool. Within minutes, SCIPE reports back the problematic node in the LLM graph, enabling rapid identification and resolution of issues.

## Installation

```python
pip install scipe
```

## Getting Started

You should have a compiled graph (from [Langgraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/)) that you've been using for your LLM application. We'll use the nodes and edges of this graph soon. We also have a couple of examples in the `examples_data` folder for you to try out.

We'll read the saved (and compiled) Langgraph using the following and convert the format to a simpler DAG which we'll feed into SCIPE.

```python
from scipe.middleware import convert_edges_to_dag

with open("graph-healthcare.json", 'r') as f:
    example_graph = json.load(f)['edges'] # We only need the edges

example_graph = convert_edges_to_dag(example_graph)
```

```python
from scipe import LLMEvaluator

evaluator = LLMEvaluator(
  config_path="config.yml",
  responses=data,
  graph=example_graph
)

results = evaluator.run_validation().find_problematic_node()
```

The `run_validation()` runs LLM-as-judge on input/output pairs and `find_problematic_node()` method traverses through the graph to figure out which node has the highest failure rate. Once it finds the problematic node, the algorithm stops and returns the result. 

You can look at the results of the algorithm.

```python
results.to_json()
```
```python
Output: 

{'root_cause': 'pii_insurance',
 'debug_path': ['summarizer', 'extractor', 'pii_insurance'],
 'node_results': {'summarizer': {'overall_failure_probability': 0.361,
   'independent_failure_probability': 0.329,
   'conditional_failure_probabilities': {'extractor': 0.476},
   'dependencies': ['extractor'],
   'is_root_cause': False},
  'extractor': {'overall_failure_probability': 0.219,
   'independent_failure_probability': 0.191,
   'conditional_failure_probabilities': {'pii_insurance': 0.259},
   'dependencies': ['pii_insurance'],
   'is_root_cause': False},
  'pii_insurance': {'overall_failure_probability': 0.27,
   'independent_failure_probability': 0.285,
   'conditional_failure_probabilities': {'pii_medications': 0.233},
   'dependencies': ['pii_medications'],
   'is_root_cause': True}}}
```

## Configuration

SCIPE uses a YAML configuration file to set up your LLM graph evaluation. Here's an example of what your config.yaml might look like:

```yaml
# Example config.yaml

# Where to save the LLM as judge validations for further analysis
PATH_TO_SAVE_VALIDATIONS: "validations.csv"

# Mode name to use for LLM validations
MODEL_NAME: claude-3-haiku-20240307

# Each node name, input and output columns must match the application responses
node_input_output_mappings:
  pii_name_number_email:
    - prompt-1
    - response-1
  pii_id:
    - prompt-1
    - response-2
  pii_birthdate:
    - prompt-1
    - response-3
  pii_medications:
    - prompt-1
    - response-4
  pii_insurance:
    - prompt-1
    - response-5
  extractor:
    - prompt-1
    - response-6
  summarizer:
    - prompt-1
    - response-7
```

## How it works

SCIPE works by analyzing the failure probabilities of nodes in your application graph to identify the most impactful source of failures. The core problem it addresses is:

**What node's failures have the biggest impact on the most downstream node's failures?**

Here's a breakdown of how SCIPE approaches this problem:

1. **LLM as Judge**: SCIPE first uses an LLM as a judge to evaluate each node in the application graph:

   - For each node, it constructs a prompt using the node's input and output.
   - The LLM judge then evaluates whether the node's output is valid given its input.
   - This process generates a dataset of node evaluations across a sample of inputs.

2. **Failure Analysis**: For every node, SCIPE recognizes that failures can occur due to two main reasons:

   - Independent failures: The node itself (or the LLM processing it) is the primary cause of the failure.
   - Dependent failures: The node fails because one or more of its dependencies have failed, causing a ripple effect.

3. **Root Cause Analysis**: SCIPE then employs an algorithm to identify the root cause of failures. Here's a high-level pseudocode of the algorithm:

   ```
   function find_root_cause(node, data, graph):
       calculate probabilities for node (overall, independent, and dependent)
       if node has no dependencies or independent failure probability is highest:
           mark node as root cause
           return node
       else:
           find dependency with highest conditional failure probability
           recursively call find_root_cause on that dependency

   function find_problematic_node(data, graph):
       identify the most downstream node in the graph
       root_cause = find_root_cause(downstream_node, data, graph)
       calculate probabilities for all nodes in the graph
       construct debug trace from downstream node to root cause
       return EvaluationResult(root_cause, debug_path, node_results)
   ```

4. **Tracing**: As the algorithm traverses the graph from downstream to upstream, it maintains a debug path, providing insights into the flow of failures through the system. The analysis culminates in an `EvaluationResult` object, which includes the identified root cause, the debug path, and detailed results for each node. The results can be easily converted to a JSON format for further analysis or visualization.

Overall, SCIPE analyzes independent and dependent failure probabilities to identify the most impactful problematic node in the system. This helps developers pinpoint and fix issues in their LLM-based application graph, improving overall performance and reliability.

## Try it out
Here's a colab notebook try out SCIPE on sample data - [demo.ipynb](https://colab.research.google.com/drive/1sM0rpxlMVAauJk6wGB-27WSlTyDnygag?usp=sharing)