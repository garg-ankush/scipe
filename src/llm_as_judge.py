import os
import json
import uuid
import operator
from dotenv import load_dotenv  # type: ignore
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from .prompt import construct_prompt


load_dotenv()


class State(TypedDict):
    batch_id: str
    input: Annotated[Sequence[BaseMessage], operator.add]
    output: Annotated[Sequence[BaseMessage], operator.add]
    node_name: Annotated[Sequence[BaseMessage], operator.add]
    validation: Annotated[Sequence[BaseMessage], operator.add]  # Binary 1 or 0
    reason: Annotated[Sequence[BaseMessage], operator.add]

class JudgeLLM:
    def __init__(self):
        self.llm = None

    def validate_response(self, state: State):
        input = state["input"][-1]
        output = state["output"][-1]

        # Input: Input to the LLM
        # Output: LLM response
        constructed_prompt = construct_prompt(
            input, output, validation_key="validation", reason_key="reason"
        )

        response = self.llm.invoke(constructed_prompt)
        content = json.loads(response.content)

        return {"validation": [content.get("validation")], "reason": [content["reason"]]}

    def construct_validator_graph(self, State):
        graph = StateGraph(State)
        graph.add_node("validate_response", self.validate_response)

        graph.add_edge(START, "validate_response")
        graph.add_edge("validate_response", END)

        graph = graph.compile()
        return graph
    
    def convert_llm_responses_to_dataframe(self, llm_evals: dict) -> pd.DataFrame:
        dataframe = pd.DataFrame()

        for batch_id, evaluation in llm_evals.items():
            validations = {f"{r['node_name'][0]}": r["validation"][0] for r in evaluation}
            reasons = {f"{r['node_name'][0]}_reason": r["reason"] for r in evaluation}
            inputs = {f"input_{idx+1}": r["input"] for idx, r in enumerate(evaluation)}
            outputs = {
                f"output_{idx+1}": r["output"][0] for idx, r in enumerate(evaluation)
            }

            row_data = {**inputs, **outputs, **validations, **reasons}
            row_data["batch_id"] = batch_id

            validations_df = pd.DataFrame([row_data])
            dataframe = pd.concat([dataframe, validations_df], ignore_index=True)

        return dataframe
    
    def run(
            self,
            llm_name: str,
            dataframe: pd.DataFrame, node_input_output_mappings: dict[tuple]
        ) -> pd.DataFrame:
        self.llm = ChatAnthropic(model=llm_name, api_key=os.getenv("ANTHROPIC_API_KEY"))
        graph = self.construct_validator_graph(State)

        # TODO implement Async code for this
        llm_evals = {}
        for _, row in dataframe.iterrows():
            json_data = row.to_dict()
            batch_id = json_data.get("batch_id", uuid.uuid4().hex)

            input_output_evaluation = []
            for node_name, mappings in node_input_output_mappings.items():
                prompt_key, response_key = mappings
                prompt_text = json_data.get(prompt_key, None)
                output_text = json_data.get(response_key, None)

                evaluation_response = graph.invoke(
                    {
                        "input": [prompt_text],
                        "output": [output_text],
                        "node_name": [node_name],
                    }
                )
                
                input_output_evaluation.append(evaluation_response)

            llm_evals[batch_id] = input_output_evaluation

        dataframe = self.convert_llm_responses_to_dataframe(llm_evals=llm_evals)
        return dataframe

def run_validations_using_llm(llm_name: str, dataframe=pd.DataFrame, node_input_output_mappings=dict):
    llm_judge = JudgeLLM()
    return llm_judge.run(
        llm_name=llm_name,
        dataframe=dataframe, 
        node_input_output_mappings=node_input_output_mappings
        )