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
from config import Config
from utils import CacheManager, EvaluationResult
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

    def __init__(self, config: Config, rubric: Dict[str, Dict]):
        """
        Initializes the OpenAI LLM Caller with API credentials and caching.

        Args:
            config (Config): Configuration object with model and API settings.
            rubric (Dict[str, Dict]): The parsed rubric dictionary.
        """
        self.client = OpenAI(api_key=config.openai_api_key)
        self.config = config
        self.cache = CacheManager(config.cache_dir, config, rubric)
        self.rubric = rubric

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_api(self, prompt: str, question_id: str) -> List[float]:
        """
        Calls OpenAI API and retrieves log probabilities for options.

        Args:
            prompt (str): The input prompt to send to OpenAI.
            question_id (str): The question ID from the rubric.

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
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=n_options,  # Dynamically set based on options count
                logprobs=True,
                top_logprobs=n_options  # Dynamically set based on options count
            )

            # Extract log-probabilities for valid tokens (option indices)
            token_logprobs: Dict[str, float] = {}
            for i in response.choices[0].logprobs.content[0].top_logprobs:
                token = i.token.strip()
                logprob = i.logprob
                if token in options.keys():  # Ensure we only get valid option keys
                    token_logprobs[token] = float(logprob)

            # Convert log-probs to unnormalized probabilities
            logprobs_array = np.array([
                token_logprobs.get(str(i + 1), float('-inf')) for i in range(n_options)
            ])

            # Exponentiate log-probs (no softmax, unnormalized probabilities)
            probs = np.exp(logprobs_array)

            self.logger.info(f"Generated probabilities for {question_id}: {probs}")
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
            List[float]: A probability distribution over valid options.
        """
        cache_key = f"{question_id}-{prompt}"

        # Check cache before calling the API
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        try:
            probabilities = self._call_api(prompt, question_id)

            # Store the result in cache
            self.cache.set(cache_key, probabilities)

            return probabilities

        except Exception as e:
            self.logger.error(f"Error in LLM call for {question_id}: {e}")
            # Return uniform probability as fallback
            n_options = len(self.rubric[question_id]["options"])
            return [1.0 / n_options] * n_options


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
            self.prompt_template = system_data["prompt"]
            rubric_path = system_data["rubric_path"]

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

    def generate_prompt(self, conversation: str, question_id: str) -> str:
        """
        Generates a formatted prompt using the conversation and rubric question.

        Args:
            conversation (str): The conversation text.
            question_id (str): The question ID from the rubric.

        Returns:
            str: A fully formatted prompt.
        """
        if question_id not in self.rubric:
            raise ValueError(f"Invalid question ID: {question_id}")

        question_data = self.rubric[question_id]
        formatted_options = self._format_options(question_data["options"])

        return self.prompt_template.format(
            conversation=conversation,
            question=question_data["prompt"],
            formatted_options=formatted_options
        )