import yaml
import json
import pandas as pd
from typing import Any
import asyncio
from src.middleware import convert_edges_to_dag
from src.llm_as_judge import run_validations_using_llm
from src.algorithm import find_problematic_node

class LanggraphImprover:
    def __init__(self, config_path: str):
        """
        Initialize Langgraph Improver with a config file

        Args:
            config_path (str): Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.application_dag = None
        self.app_responses = None
        self.llm_validations = None

    def _load_config(self, config_path: str):
        """
        Load and return configs from a YAML file

        Args:
            config_path (str): Path to YAML configuration file
        """
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def load_application_graph(self) -> None:
        """
        Load application graph (converted to json)
        """
        with open(self.config["PATH_TO_APPLICATION_GRAPH_JSON"], "r") as json_file:
            graph_json = json.load(json_file)
        edges = graph_json["edges"]
        self.application_dag = convert_edges_to_dag(edges=edges)

    def load_application_responses(self) -> None:
        """
        Load application input/outputs to validate using an LLM
        """
        if self.config["PATH_TO_APPLICATION_RESPONSES"].split(".")[-1] == "csv":
            self.app_responses = pd.read_csv(self.config["PATH_TO_APPLICATION_RESPONSES"])
            return

        data = pd.read_excel(self.config["PATH_TO_APPLICATION_RESPONSES"])
        self.app_responses = data.tail(1).reset_index(drop=True)

    def run_llm_validation(self) -> None:
        """
        Run LLM validation on the application responses

        Args:
            use_llm (Optional[bool], optional): Where to use LLM for validation of application responses.
            Defaults to None.
        """
        if self.config["node_input_output_mappings"] is None:
            raise ValueError("You must update node input and output names in the Config file.")
        
        llm_validations = asyncio.run(
            run_validations_using_llm(
                model_name=self.config["MODEL_NAME"],
                dataframe=self.app_responses,
                node_input_output_mappings=self.config["node_input_output_mappings"]
                )
            )
        self.llm_validations = llm_validations
    
    def improve_system(self) -> Any:
        """
        Improve the system by running an algorithm on top of LLM or Human responses

        Returns:
            Any: Prints out the result about which node to work on.
        """
        if self.llm_validations is None or self.application_dag is None:
            raise ValueError("Validations and application graph must be loaded before finding the problematic node.")
        
        return find_problematic_node(data=self.llm_validations, dag=self.application_dag)
    

    def improve(self):
        """
        Put all the functions together
        """
        self.load_application_graph()
        self.load_application_responses()
        self.run_llm_validation()
        self.improve_system()
        

