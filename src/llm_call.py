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
    def _call_api(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "You are an expert at evaluating conversations. Respond only with a number 1-4 corresponding to your choice from the given options."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=10
        )
        return response.choices[0].message.content

    def _extract_number_from_response(self, response: str) -> Optional[int]:
        clean_response = response.strip().split()[0]
        try:
            number = int(clean_response)
            if 1 <= number <= 4:
                return number
            self.logger.warning(f"Invalid number in response: {number}")
            return None
        except ValueError:
            self.logger.warning(f"Could not extract number from response: {response}")
            return None

    def _convert_to_probabilities(self, choice: Optional[int], num_options: int = 4) -> List[float]:
        if choice is None:
            return [1.0 / num_options] * num_options
        
        probs = [0.1 / (num_options - 1)] * num_options
        probs[choice - 1] = 0.9
        return probs

    def __call__(self, prompt: str) -> List[float]:
        # Check cache first
        cached_result = self.cache.get(prompt)
        if cached_result is not None:
            return cached_result

        try:
            response_text = self._call_api(prompt)
            self.logger.info(f"Raw response: {response_text}")
            
            choice = self._extract_number_from_response(response_text)
            probabilities = self._convert_to_probabilities(choice)
            
            # Cache the result
            self.cache.set(prompt, probabilities)
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            return [0.25, 0.25, 0.25, 0.25]
        
# evaluator.py
class LLMEvaluator:
    def __init__(self, rubric_path: str):
        self.rubric = self._load_rubric(rubric_path)
        self.prompt_template = """
        You are given a conversation between a user and an intelligent assistant for an enterprise chat scenario. 
        In some cases, some references and citations are provided to back up the claims made by the intelligent assistant. 
        Your primary job is to evaluate the quality of the conversation based on a criterion. To do so, read the conversation and references, 
        and answer the followed question, by selecting only one of the choices.

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