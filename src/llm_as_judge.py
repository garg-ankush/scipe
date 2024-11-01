import os
import json
import uuid
import operator
from dotenv import load_dotenv  # type: ignore
import pandas as pd
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from litellm import completion
from tqdm.auto import tqdm
from .helpers import construct_prompt

load_dotenv()


class State(TypedDict):
    batch_id: str
    input: Annotated[Sequence[BaseMessage], operator.add]
    output: Annotated[Sequence[BaseMessage], operator.add]
    node_name: Annotated[Sequence[BaseMessage], operator.add]
    validation: Annotated[Sequence[BaseMessage], operator.add]  # Binary 1 or 0
    reason: Annotated[Sequence[BaseMessage], operator.add]

class JudgeLLM:
    def __init__(self, model_name: str, prompt: str):
        self.model_name = model_name
        self.prompt = prompt

    def validate_response(self, state: State):
        input = state["input"][-1]
        output = state["output"][-1]

        # Input: Input to the LLM
        # Output: LLM response
        if self.prompt is not None:
            # Using user's prompt
            constructed_prompt = f"{self.prompt}\n\nInput: {input}\nOutput: {output}"
        else:
            # Using prebuilt prompt
            constructed_prompt = construct_prompt(
                input_=input, 
                output_=output
                )

        # Make a call to any LLM that LiteLLM supports
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": constructed_prompt}],
            api_key=os.getenv("LLM_MODEL_API_KEY")
            )
        content = json.loads(response.choices[0].message.content)
        return {"output": [output], "validation": [content.get("validation")], "reason": [content.get("reason")]}

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
    
    def judge(
            self,
            dataframe: pd.DataFrame,
            node_input_output_mappings: dict[tuple]
        ) -> pd.DataFrame:
        graph = self.construct_validator_graph(State)

        def process_row(row):
            json_data = row.to_dict()
            batch_id = json_data.get("batch_id", uuid.uuid4().hex)
            evaluations = []
            for node_name, (prompt_key, response_key) in node_input_output_mappings.items():
                prompt_text = json_data.get(prompt_key, None)
                output_text = json_data.get(response_key, None)
        
                evaluation = graph.invoke({
                    "input": [prompt_text],
                    "output": [output_text],
                    "node_name": [node_name],
                })
                evaluations.append(evaluation)

            return batch_id, evaluations

        results = []
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="LLM Evals"):
            result = process_row(row)
            results.append(result)

        return dict(results)

def run_validations(model_name: str, dataframe: pd.DataFrame, node_input_output_mappings: dict, prompt: str):
    llm_judge = JudgeLLM(model_name=model_name, prompt=prompt)
    
    evals = llm_judge.judge(
        dataframe=dataframe, 
        node_input_output_mappings=node_input_output_mappings
        )
    return llm_judge.convert_llm_responses_to_dataframe(llm_evals=evals)
