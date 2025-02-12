"""
Author: Prabhav Singh
Implementation of: LLMRubric (https://aclanthology.org/2024.acl-long.745v2.pdf)

LLM_CALL: API to call OpenAI LLM for evaluating conversations using a rubric.
"""

from openai import OpenAI
import time
import json
import yaml
import logging
from typing import List, Optional, Dict, Union
import numpy as np
from src.config import Config
from src.utils import CacheManager, EvaluationResult
from tenacity import retry, stop_after_attempt, wait_exponential

class OpenAILLMCaller:
    """
    Handles API calls to OpenAI's LLM and retrieves probability distributions over options.

    Attributes:
        client (OpenAI): OpenAI API client.
        config (Config): Configuration settings for API usage.
        cache (CacheManager): Caching mechanism to avoid redundant API calls.
        logger (logging.Logger): Logger for tracking API calls and errors.
    """

    def __init__(self, config: Config, rubric: Dict[str, Dict], logger):
        """
        Initializes the OpenAI LLM Caller with API credentials and caching.

        Args:
            config (Config): Configuration object with model and API settings.
            rubric (Dict[str, Dict]): The parsed rubric dictionary.
            logger (logging.Logger): Logger for tracking API calls and errors.
        """
        self.client = OpenAI(api_key=config.openai_api_key)
        self.config = config
        self.cache = CacheManager(config.cache_dir, config, rubric)
        self.rubric = rubric
        self.logger = logger

    def _scale_logits_with_temp(self, logits: np.ndarray, old_temp: float, new_temp: float) -> np.ndarray:
        """
        Scale logits from one temperature to another.
        
        Args:
            logits: Original logits array
            old_temp: Temperature used for original sampling
            new_temp: Target temperature
            
        Returns:
            np.ndarray: Scaled logits
        """
        return logits * (old_temp/new_temp)


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_api(self, prompt: str, question_id: str, model: str, temperature: float) -> List[float]:
        """
        Calls OpenAI API and retrieves log probabilities for options.

        Args:
            prompt (str): The input prompt to send to OpenAI.
            question_id (str): The question ID from the rubric.
            model (str): OpenAI Model Version
            temperature (float): Temperature to sample at

        Returns:
            List[float]: A list of unnormalized probabilities corresponding to options.

        If an error occurs, returns a uniform probability distribution.
        """

        try:
            # Get the number of options for the given question from the rubric
            if question_id not in self.rubric:
                raise ValueError(f"Question ID {question_id} not found in rubric.")

            question_data = self.rubric[question_id]
            options = question_data["options"]
            n_options = len(options)

            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=temperature,
                max_tokens=n_options,
                logprobs=True,
                top_logprobs=n_options
            )

            # Extract log-probabilities for valid tokens (option indices)
            token_logprobs: Dict[str, float] = {}
            for i in response.choices[0].logprobs.content[0].top_logprobs:
                token = i.token.strip()
                logprob = i.logprob
                if int(token) in options.keys():  # Ensure we only get valid option keys
                    token_logprobs[token] = float(logprob)

            # Convert log-probs to unnormalized probabilities
            logprobs_array = np.array([
                token_logprobs.get(str(i + 1), float('-inf')) for i in range(n_options)
            ])

            # Exponentiate log-probs (no softmax, unnormalized probabilities)
            probs = np.exp(logprobs_array)
            return probs.tolist()

        except Exception as e:
            self.logger.error(f"Error in API call for {question_id}: {e}")
            # Return uniform probability in case of failure
            return [1.0 / n_options] * n_options

    def __call__(self, prompt: str, question_id: str) -> List[float]:
        """
        Retrieves probabilities for each option given a conversation prompt.

        Uses caching to avoid redundant API calls.

        Args:
            prompt (str): The input prompt.
            question_id (str): The question ID from the rubric.

        Returns:
            Returns: Dict[str, Dict[str, List[float]]]: Nested dict with structure: {model_name: {temperature: probabilities}}
        """

        results = {}

        # Loop over all possible model configuration
        for model_name, model_config in self.config.models.items():

            results[model_name] = {}

            for temp in model_config.temperatures:

                # Change Cache Key!
                cache_key = f"{question_id}-{prompt}-{model_name}-{temp}"
                cached_result = self.cache.get(cache_key, model_name, temp)

                if cached_result is not None:
                    results[model_name][temp] = cached_result
                    continue

                try:
                    probabilities = self._call_api(prompt, question_id, model_name, temp)
                    self.cache.set(cache_key, model_name, temp, probabilities)
                    results[model_name][temp] = probabilities

                except Exception as e:
                    self.logger.error(f"Error in LLM call for {question_id} with {model_name}, temp {temp}: {e}")
                    n_options = len(self.rubric[question_id]["options"])
                    results[model_name][temp] = [1.0 / n_options] * n_options

        return results

class LLMEvaluator:
    """
    Evaluates conversations using a rubric by generating prompts and querying an LLM.

    Attributes:
        rubric (Dict): A dictionary containing evaluation questions and options.
        prompt_template (str): The pre-defined prompt format for evaluation.
    """

    def __init__(self, system_path: str):
        """
        Initializes the evaluator by loading the rubric from a YAML or JSON file.

        Args:
            system_path (str): Path to the system evaluation file (AML).
        """
        self.prompt_template = ''
        rubric_path = self._load_system(system_path)
        self.rubric = self._load_rubric(rubric_path)

    def _load_system(self, path: str) -> str:
        """
        Loads the system prompt and get rubric path from a YAML file.

        Args:
            path (str): Path to the system file.

        Returns:
            path (str): Rubric file path.
        """

        with open(path, "r") as f:
            system_data = yaml.safe_load(f)
            self.prompt_template = system_data["system_prompt"]["prompt"]
            rubric_path = system_data["system_prompt"]["rubric_path"]

        return rubric_path

    def _load_rubric(self, path: str) -> Dict[str, Dict]:
        """
        Loads the rubric from a YAML or JSON file.

        Args:
            path (str): Path to the rubric file.

        Returns:
            Dict[str, Dict]: Parsed rubric content.
        """
        try:
            with open(path, "r") as f:
                if path.endswith((".yaml", ".yml")):
                    rubric_data = yaml.safe_load(f)
                elif path.endswith(".json"):
                    rubric_data = json.load(f)
                else:
                    raise ValueError("Unsupported file format. Use JSON or YAML.")

            return rubric_data["rubric"]

        except Exception as e:
            logging.error(f"Failed to load rubric: {e}")
            raise

    def _format_options(self, options: Dict[int, str]) -> str:
        """
        Formats options into a numbered list.

        Args:
            options (Dict[int, str]): The list of answer options.

        Returns:
            str: A formatted string with numbered answer choices.
        """
        return "\n".join(f"{key}. {value}" for key, value in options.items())

    def generate_prompt(self, text_unit: str, question_id: str) -> str:
        """
        Generates a formatted prompt using the conversation and rubric question.

        Args:
            text_unit (str): The conversation text.
            question_id (str): The question ID from the rubric.

        Returns:
            str: A fully formatted prompt.
        """
        if question_id not in self.rubric.keys():
            raise ValueError(f"Invalid question ID: {question_id}")

        question_data = self.rubric[question_id]
        formatted_options = self._format_options(question_data["options"])

        return self.prompt_template.format(
            TEXT=text_unit,
            QUESTION=question_data["prompt"],
            OPTIONS=formatted_options
        )
    
    def evaluate_conversation(self, text_unit: str, llm_caller) -> List[EvaluationResult]:
        """
        Create object of type EvaluationResult.
        For each question in Rubric, evaluate using prompt and get probability distribution!

        Args:
            text_unit (str): Conversation Text
            llm_caller (_type_): OpenAI LLM Caller API Class

        Returns:
            List[EvaluationResult]: List of objects with results stored within.
        """

        results = []
        
        for model_name, model_config in llm_caller.config.models.items():
            for temp in model_config.temperatures:
                result = EvaluationResult()
                result.model = model_name
                result.temperature = temp
                
                for q_id in self.rubric.keys():
                    prompt = self.generate_prompt(text_unit, q_id)
                    probs = llm_caller(prompt, q_id)[model_name][temp]
                    result.add_result(q_id, probs)
                
                results.append(result)

        return results
