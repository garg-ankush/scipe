# SCIPE - Systematic Chain Improvement and Problem Evaluation
SCIPE is a powerful tool for evaluating and diagnosing LLM (Large Language Model) graphs or chains. It assesses LLM responses and employs a custom algorithm to identify problematic nodes within the LLM chain.

## Features
- Evaluates LLM responses within simple LLM Graphs (mainly [LangGraph](https://langchain-ai.github.io/langgraph/))
- Diagnoses problematic nodes in LLM graphs
- Provides failure rates of different nodes
- Supports various LLM frameworks (uses [LiteLLM](https://github.com/BerriAI/litellm) underneath the hood)

### Why Use SCIPE?
As AI application developers, we often overlook the critical step of evaluating LLM chains during the building phase. SCIPE simplifies this process by allowing developers to run their minimum set of prompts and responses (we recommend atleast 10 examples) through the tool. Within minutes, SCIPE reports back the problematic node in the LLM graph, enabling rapid identification and resolution of issues.

### Installation
```python
pip install scipe
```
### Getting Started
You should have a compiled graph (from [Langgraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/)) that you've been using for your LLM application. You should also, save the graph down as json by running the following. We'll use the nodes and edges of this graph soon. We have a couple of examples in the `examples_data` folder for you to try out.

```python
import json
with open("graph.json", "r") as file:
    json.dumps(compiled_graph.to_json(), file)
```

```python
from scipe import LanggraphImprover
improver = LanggraphImprover(f"{PATH_TO_CONFIG_YAML}")
improver.improve()
```
The `improve` method traverses through the graph to figure out which node has the highest failure rate. Once it finds the failure node, it prints it's failure probability to the console.

### Configuration
SCIPE uses a YAML configuration file to set up your LLM graph evaluation. Here's an example of what your config.yaml might look like:

```yaml
# Example config.yaml

PATH_TO_APPLICATION_GRAPH_JSON: example_data/graph-healthcare.json
PATH_TO_APPLICATION_RESPONSES: example_data/healthcare-responses.xlsx

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
