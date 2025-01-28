'''
Author: Prabhav Singh
Implentation Of: Eisner et al. (https://aclanthology.org/2024.acl-long.745v2.pdf)
LLM_CALL: API to call LLM!
'''

from openai import OpenAI
import time
import json
import logging
from typing import List, Optional, Dict, Union
import numpy as np
from config import Config
from utils import CacheManager, EvaluationResult
from tenacity import retry, stop_after_attempt, wait_exponential

class OpenAILLMCaller:
    def __init__(self, config: Config):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.config = config
        self.cache = CacheManager(config.cache_dir)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_api(self, prompt: str) -> List[float]:
        """Call OpenAI API and return probabilities for options 1-4."""
        try:
            # First get completion to set up context
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], 
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=4,
                logprobs=True,
                top_logprobs=4
             )  

            
            # Extract logprobs for the tokens "1", "2", "3", "4"
            token_logprobs = {}
            for i in response.choices[0].logprobs.content[0].top_logprobs:
                token = i.token
                logprob = i.logprob
                if str(token.strip()) in ["1", "2", "3", "4"]:
                    token_logprobs[token.strip()] = float(logprob)

            # Convert logprobs to probabilities
            probs = np.zeros(4)
            logprobs_array = np.array([token_logprobs.get(str(i+1), float('-inf')) 
                                     for i in range(4)])
            
            # Softmax calculation
            exp_logprobs = np.exp(logprobs_array - np.max(logprobs_array))
            probs = exp_logprobs / exp_logprobs.sum()
            
            # Ensure no zero probabilities (smooth)
            probs = np.maximum(probs, 1e-7)
            probs = probs / probs.sum()
            
            self.logger.info(f"Generated probabilities: {probs}")
            return probs.tolist()

        except Exception as e:
            self.logger.error(f"Error in API call: {e}")
            # Return uniform distribution in case of error
            return [0.25, 0.25, 0.25, 0.25]

    def __call__(self, prompt: str) -> List[float]:
        """
        Get probabilities for each option (1-4) given a prompt.
        Uses caching to avoid redundant API calls.
        """
        # Check cache first
        cached_result = self.cache.get(prompt)
        if cached_result is not None:
            return cached_result

        try:
            probabilities = self._call_api(prompt)
            
            # Cache the result
            self.cache.set(prompt, probabilities)
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Error in LLM call: {e}")
            return [0.25, 0.25, 0.25, 0.25]

class LLMEvaluator:
    def __init__(self, rubric_path: str):
        self.rubric = self._load_rubric(rubric_path)
        self.prompt_template = """
You are given a conversation between a user and an intelligent assistant for an enterprise chat scenario. In some cases, some references and citations are provided to back up the claims made by the intelligent assistant. Your primary job is to evaluate the quality of the conversation based on a criterion. To do so, read the conversation and references, and answer the followed question, by selecting only one of the choices.

Conversation: {conversation}

Question: **{question}**

Options:
{formatted_options}

Only print '1', '2', '3', or '4'.
"""

    def _load_rubric(self, path: str) -> Dict:
        try:
            with open(path, 'r') as f:
                rubric = json.load(f)
            self._validate_rubric(rubric)
            return rubric
        except Exception as e:
            logging.error(f"Failed to load rubric: {e}")
            raise

    def _validate_rubric(self, rubric: Dict) -> None:
        required_keys = [f"Q{i}" for i in range(9)]
        for key in required_keys:
            if key not in rubric:
                raise ValueError(f"Missing required question {key} in rubric")
            if "PromptDesc" not in rubric[key] or "Options" not in rubric[key]:
                raise ValueError(f"Missing required fields in {key}")

    def _format_options(self, options: List[Union[str, int]]) -> str:
        return "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))

    def generate_prompt(self, conversation: str, question_id: str) -> str:
        if question_id not in self.rubric:
            raise ValueError(f"Invalid question ID: {question_id}")
        
        question_data = self.rubric[question_id]
        formatted_options = self._format_options(question_data["Options"])
        
        return self.prompt_template.format(
            conversation=conversation,
            question=question_data["PromptDesc"],
            formatted_options=formatted_options
        )

    def evaluate_conversation(self, conversation: str, llm_caller) -> EvaluationResult:
        result = EvaluationResult()
        for q_id in self.rubric.keys():
            prompt = self.generate_prompt(conversation, q_id)
            probs = llm_caller(prompt)
            result.add_result(q_id, probs)
        return result