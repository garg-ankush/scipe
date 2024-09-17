import pandas as pd
from middleware import convert_edges_to_dag
from algorithm import improve_system
from llm_as_judge import construct_validator_graph, State


edges = []

# Get LLM responses
validator_graph = construct_validator_graph(State)

converted_graph = convert_edges_to_dag(edges=edges)

# mappings = [
#     ('prompt-1', 'output-keywords'),
#     ('prompt-2', 'output-article_search'),
#     ('prompt-3', 'output-replies'),
#     ('prompt-4', 'output-reply')
#     ]

# validations_dict = run_validation_using_LLM(
#         graph=validator_graph,
#         dataframe=input_output_df,
#         prompt_response_mappings=mappings
#     )

data_llm_outputs = pd.read_excel(
    "/Users/ankushgarg/Desktop/MIDS/epic-data-lab/validate-LLM-graphs/reddit-comments/llm-evals.xlsx"
)
data_llm_outputs.rename(
    columns={
        "check-1": "extract_keywords_from_title",
        "check-2": "tool_search",
        "check-3": "summarize_tone_sentiment_of_replies",
        "check-4": "generate_reply",
    },
    inplace=True,
)

improve_system("generate_reply", data=data_llm_outputs, dag=converted_graph)
