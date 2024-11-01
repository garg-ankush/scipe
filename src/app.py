import pandas as pd
from typing import Any, Dict
from collections import defaultdict
from .llm_as_judge import run_validations
from .algorithm import find_problematic_node


class LLMEvaluator:
    def __init__(self, config: Dict, responses: pd.DataFrame, graph: defaultdict):
        """LLM Evaluator for running LLM-based evaluations on application data

        Args:
            config (dict): Configuration dictionary 
            responses (pd.DataFrame): Dataframe of responses from an application
            application_dag (defaultdict): Application dag with parent/child relationship
        """
        self.config = config
        self.responses = responses
        self.graph = graph
        self._validations = None

    def run_validation(self, prompt=None) -> None:
        """
        Run LLM validation on the application responses

        Args:
            use_llm (Optional[bool], optional): Where to use LLM for validation of application responses.
            Defaults to None.
        """
        if self.config["node_input_output_mappings"] is None:
            raise ValueError("You must update node input and output names in the config.")
        
        llm_validations = run_validations(
                model_name=self.config["MODEL_NAME"],
                dataframe=self.responses,
                node_input_output_mappings=self.config["node_input_output_mappings"],
                prompt=prompt
                )

        self.llm_validations = llm_validations

        # Save these down in case the user wants to use them again
        llm_validations.to_csv(f"{self.config['PATH_TO_SAVE_VALIDATIONS']}", index=None)

        return self
    
    def find_problematic_node(self) -> Any:
        """
        Improve the system by running an algorithm on top of LLM or Human responses

        Returns:
            Any: Prints out the result about which node to work on.
        """
        if self.graph is None:
            raise ValueError("Graph must be loaded before looking for the problematic node.")

        if self.llm_validations is None:
            raise ValueError("LLM Validations must be run looking for the problematic node.")
        
        return find_problematic_node(data=self.llm_validations, graph=self.graph)
        

