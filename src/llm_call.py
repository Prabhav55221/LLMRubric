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

    def __init__(self, config: Config):
        """
        Initializes the OpenAI LLM Caller with API credentials and caching.
        """

        self.client = OpenAI(api_key=config.openai_api_key)
        self.config = config
        self.cache = CacheManager(config.cache_dir)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_api(self, prompt: str) -> List[float]:
        """
        Calls OpenAI API and retrieves log probabilities for options "1" to "4".

        Args:
            prompt (str): The input prompt to send to OpenAI.

        Returns:
            List[float]: A list of probabilities corresponding to options 1-4.

        If an error occurs, returns a uniform probability distribution [0.25, 0.25, 0.25, 0.25].
        """

        try:

            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=4,
                logprobs=True,
                top_logprobs=4
            )

            # Extract logprobs for valid tokens
            token_logprobs: Dict[str, float] = {}
            for i in response.choices[0].logprobs.content[0].top_logprobs:
                token = i.token.strip()
                logprob = i.logprob
                if token in {"1", "2", "3", "4"}:
                    token_logprobs[token] = float(logprob)

            # Convert logprobs to probabilities using softmax
            logprobs_array = np.array([
                token_logprobs.get(str(i + 1), float('-inf')) for i in range(4)
            ])

            # Softmax normalization to get probabilities
            exp_logprobs = np.exp(logprobs_array - np.max(logprobs_array))
            probs = exp_logprobs / exp_logprobs.sum()

            # Ensure no zero probabilities (add small smoothing factor)
            probs = np.maximum(probs, 1e-7)
            probs = probs / probs.sum()

            self.logger.info(f"Generated probabilities: {probs}")
            return probs.tolist()

        except Exception as e:
            self.logger.error(f"Error in API call: {e}")
            # Fallback uniform probability
            return [0.25, 0.25, 0.25, 0.25]

    def __call__(self, prompt: str) -> List[float]:
        """
        Retrieves probabilities for each option (1-4) given a conversation prompt.
        Uses caching to avoid redundant API calls.

        Args:
            prompt (str): The input prompt containing a conversation and evaluation question.

        Returns:
            List[float]: A probability distribution over options 1-4.
        """

        # Check cache before calling the API
        cached_result = self.cache.get(prompt)

        if cached_result is not None:
            return cached_result

        try:
            probabilities = self._call_api(prompt)

            # Store the result in cache
            self.cache.set(prompt, probabilities)

            return probabilities

        except Exception as e:
            self.logger.error(f"Error in LLM call: {e}")
            # Fallback uniform probability
            return [0.25, 0.25, 0.25, 0.25]

class LLMEvaluator:
    """
    Evaluates conversations using a rubric by generating prompts and querying an LLM.

    Attributes:
        rubric (Dict): A dictionary containing evaluation questions and options.
        prompt_template (str): The pre-defined prompt format for evaluation.
    """

    def __init__(self, rubric_path: str):
        """
        Initializes the evaluator by loading the rubric from a YAML or JSON file.

        Args:
            rubric_path (str): Path to the rubric file (JSON or YAML).
        """
        self.rubric = self._load_rubric(rubric_path)
        self.prompt_template = """
You are given a conversation between a user and an intelligent assistant for an enterprise chat scenario. In some cases, some references and citations are provided to back up the claims made by the intelligent assistant. Your primary job is to evaluate the quality of the conversation based on a criterion. To do so, read the conversation and references, and answer the followed question, by selecting only one of the choices.

Conversation: {conversation}

Question: **{question}**

Options:
{formatted_options}

Only print '1', '2', '3', or '4'.
"""

    def _load_rubric(self, path: str) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        """
        Loads the rubric from a YAML or JSON file.

        Args:
            path (str): Path to the rubric file.

        Returns:
            Dict[str, Dict[str, Union[str, List[str]]]]: Parsed rubric content.

        Raises:
            ValueError: If the file format is not supported or if required fields are missing.
        """

        try:
            with open(path, 'r') as f:
                if path.endswith((".yaml", ".yml")):
                    rubric_data = yaml.safe_load(f)
                elif path.endswith(".json"):
                    rubric_data = json.load(f)
                else:
                    raise ValueError("Unsupported file format. Use JSON or YAML.")

            self._validate_rubric(rubric_data["rubric"])
            return rubric_data["rubric"]

        except Exception as e:
            logging.error(f"Failed to load rubric: {e}")
            raise

    def _validate_rubric(self, rubric: Dict[str, Dict[str, Union[str, List[str]]]]) -> None:
        """
        Validates the structure of the rubric.

        Args:
            rubric (Dict): The parsed rubric dictionary.

        Raises:
            ValueError: If required fields are missing.
        """

        required_keys = [f"Q{i}" for i in range(9)]
        for key in required_keys:
            if key not in rubric:
                raise ValueError(f"Missing required question {key} in rubric")
            if "PromptDesc" not in rubric[key] or "Options" not in rubric[key]:
                raise ValueError(f"Missing required fields in {key}")

    def _format_options(self, options: List[Union[str, int]]) -> str:
        """
        Formats options into a numbered list.

        Args:
            options (List[Union[str, int]]): The list of answer options.

        Returns:
            str: A formatted string with numbered answer choices.
        """

        return "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))

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
        formatted_options = self._format_options(question_data["Options"])

        return self.prompt_template.format(
            conversation=conversation,
            question=question_data["PromptDesc"],
            formatted_options=formatted_options
        )