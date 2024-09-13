import pandas as pd
from middleware import convert_edges_to_dag
from middleware import Edge
from algorithm import improve_system
from llm_as_judge import construct_validator_graph, State, run_validation_using_LLM

####### EXAMPLE ##########
# Don't need this. Can work with Langgraph itself
edges = [Edge(source='__start__', target='extract_keywords_from_title', data=None, conditional=False),
 Edge(source='extract_keywords_from_title', target='tool_search', data=None, conditional=False),
 Edge(source='generate_reply', target='__end__', data=None, conditional=False),
 Edge(source='summarize_tone_sentiment_of_replies', target='generate_reply', data=None, conditional=False),
 Edge(source='tool_search', target='summarize_tone_sentiment_of_replies', data=None, conditional=False)]
####### EXAMPLE ##########

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

data_llm_outputs = pd.read_excel("/Users/ankushgarg/Desktop/MIDS/epic-data-lab/validate-LLM-graphs/reddit-comments/llm-evals.xlsx")
data_llm_outputs.rename(columns={
    "check-1": "extract_keywords_from_title",
    "check-2": "tool_search",
    "check-3": "summarize_tone_sentiment_of_replies",
    "check-4": "generate_reply"
}, inplace=True)

improve_system('generate_reply', data=data_llm_outputs, dag=converted_graph)










