import yaml
import pandas as pd
from helpers import read_graph
from middleware import convert_edges_to_dag
from llm_as_judge import run_validation_using_LLM
from algorithm import improve_system
config = yaml.safe_load(open('config.yml'))

app_edges = read_graph(path_to_json=config["PATH_TO_APPLICATION_GRAPH_JSON"])

application_dag = convert_edges_to_dag(edges=app_edges)

# Read in dataset to evaluate
app_responses = pd.read_excel(config["PATH_TO_APPLICATION_RESPONSES"])

# LLM as a judge
if config["LLM_AS_A_JUDGE_MODE"]:
    llm_response_df = run_validation_using_LLM(
        dataframe=app_responses,
        node_input_output_mappings=config["node_input_output_mappings"]
    )
else:
    llm_response_df = pd.read_excel(config["PATH_TO_LLM_RESPONSES"])

improve_system(data=llm_response_df, dag=application_dag)