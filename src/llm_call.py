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
from src.template_parser import TemplateParser
from src.utils import CacheManager, EvaluationResult
from openai.types import ResponseFormatJSONSchema
from structured_logprobs import add_logprobs
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
    
    def _prepare_openai_schema(self, schema: Dict) -> Dict:
        """
        Transforms our rubric JSON schema into OpenAI's expected format.
        
        Args:
            schema: Original schema from rubric
            
        Returns:
            Dict formatted to match OpenAI's JSON schema requirement exactly
        """

        # Get all property names
        all_properties = list(schema["properties"].keys())
        
        openai_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "json_schema",
                "description": "schema_to_follow_for_output",
                "schema": {
                    "type": "object",
                    "properties": {},
                    "required": all_properties,
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        
        for field, field_schema in schema["properties"].items():
            min_val = field_schema.get("minimum", 1)
            max_val = field_schema.get("maximum", self.config.num_option)
            allowed_values = list(range(min_val, max_val + 1))
            
            openai_format["json_schema"]["schema"]["properties"][field] = {
                "type": "number",
                "enum": allowed_values,
                "description": field_schema.get("description", "")
            }
        
        return openai_format

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_api(self, prompt: str, question_id: str, model: str, temperature: float) -> Dict:
        """
        Calls OpenAI API and retrieves log probabilities for options.

        Args:
            prompt (str): The input prompt to send to OpenAI.
            question_id (str): The question ID from the rubric.
            model (str): OpenAI Model Version
            temperature (float): Temperature to sample at

        Returns:
            Dict: For regular questions:
                {
                    "probabilities": List[float],
                    "logits": List[float]
                }
                For JSON questions:
                {
                    field_name: {
                        "probabilities": List[float],
                        "logits": List[float]
                    }
                }
        """
        try:
            
            if question_id not in self.rubric:
                raise ValueError(f"Question ID {question_id} not found in rubric.")

            question_data = self.rubric[question_id]

            # Base API parameters
            api_params = {
                "messages": [{"role": "user", "content": prompt}],
                "model": model,
                "temperature": temperature,
                "logprobs": True,
                "top_logprobs": self.config.num_options
            }

            if question_data.get("type") == "json_question":

                # JSON response handling
                schema = question_data["json_schema"]
                open_ai_schema = self._prepare_openai_schema(schema)
                
                api_params.update({
                    "max_tokens": 1000,
                    "response_format": open_ai_schema.model_dump(by_alias=True)
                })

                response = self.client.chat.completions.create(**api_params)
                completion_with_probs = add_logprobs(response)
                parsed_response = json.loads(response.choices[0].message.content)
                
                result = {}
        
                for field, field_schema in schema["properties"].items():

                    logits = [float('-inf')] * self.config.num_options
                    value = parsed_response.get(field)
                    
                    # Find logprob for this value from completion_with_probs
                    if value is not None and 1 <= value <= self.config.num_options:

                        value_str = str(value)
                        for token_info in completion_with_probs.value.choices[0].logprobs.content:

                            if token_info.token.strip() == value_str:

                                logits[value - 1] = token_info.logprob
                                for top_logprob in token_info.top_logprobs:
                                    try:
                                        other_value = int(top_logprob.token.strip())
                                        if 1 <= other_value <= 5:
                                            logits[other_value - 1] = max(logits[other_value - 1], 
                                                                        top_logprob.logprob)
                                    except ValueError:
                                        continue
                                break
                    
                    probs = np.exp(logits)
                    total_prob = np.sum(probs)
                    
                    if total_prob > 0:
                        probs = probs / total_prob
                    else:
                        probs = np.ones(5) / 5
                        logits = [np.log(0.2)] * 5
                    
                    result[field] = {
                        "probabilities": probs.tolist(),
                        "logits": logits
                    }
                
            else:

                n_options = len(question_data["options"])
                api_params.update({
                    "max_tokens": n_options,
                    "top_logprobs": n_options
                })

                response = self.client.chat.completions.create(**api_params)
                
                token_logprobs = {}
                valid_options = set(str(k) for k in question_data["options"].keys())
                
                for logprob_info in response.choices[0].logprobs.content[0].top_logprobs:
                    token = logprob_info.token.strip()
                    if token in valid_options:
                        token_logprobs[token] = float(logprob_info.logprob)
                
                # Create logits array preserving option order
                logits = [token_logprobs.get(str(i), float('-inf')) 
                        for i in range(1, n_options + 1)]
                
                # Convert to probabilities
                logits_array = np.array(logits)
                probs = np.exp(logits_array)
                total_prob = np.sum(probs)
                
                if total_prob > 0:
                    probs = probs / total_prob
                else:
                    # If no valid probabilities, use uniform distribution
                    probs = np.ones(n_options) / n_options
                    logits = [np.log(1.0 / n_options)] * n_options
                
                return {
                    "probabilities": probs.tolist(),
                    "logits": logits
                }

        except Exception as e:
            self.logger.error(f"Error in API call for {question_id}: {e}")
            
            if question_data.get("type") == "json_question":
                # Return uniform distributions for all JSON fields
                return {
                    field: {
                        "probabilities": [1.0 / field_schema.get("maximum", self.config.num_option)] * field_schema.get("maximum", self.config.num_option),
                        "logits": [np.log(1.0 / field_schema.get("maximum", self.config.num_option))] * field_schema.get("maximum", self.config.num_option)
                    }
                    for field, field_schema in question_data["json_schema"]["properties"].items()
                }
            else:
                # Return uniform distribution for regular question
                n_options = len(question_data["options"])
                return {
                    "probabilities": [1.0 / n_options] * n_options,
                    "logits": [np.log(1.0 / n_options)] * n_options
                }

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

        for model_name, model_config in self.config.models.items():
            results[model_name] = {}
            
            # Sort temperatures and use the last one as base
            temps = sorted(model_config.temperatures)
            base_temp = temps[-1]
            
            # Try to get from cache first
            cache_key = f"{question_id}-{prompt}-{model_name}-{base_temp}"
            cached_result = self.cache.get(cache_key, model_name, base_temp)
            cached_logits = self.cache.get(f"{cache_key}_logits", model_name, base_temp)

            if cached_result is not None and cached_logits is not None:
                base_probs = cached_result
                base_logits = cached_logits
            else:
                try:
                    base_probs, base_logits = self._call_api(prompt, question_id, model_name, base_temp)
                    self.cache.set(cache_key, model_name, base_temp, base_probs)
                    self.cache.set(f"{cache_key}_logits", model_name, base_temp, base_logits)
                except Exception as e:
                    self.logger.error(f"Error in LLM call: {e}")
                    n_options = len(self.rubric[question_id]["options"])
                    base_probs = [1.0 / n_options] * n_options
                    base_logits = [np.log(1.0 / n_options)] * n_options

            # Store base temperature result
            results[model_name][base_temp] = base_probs
            
            # Scale for other temperatures
            base_logits_array = np.array(base_logits)
            for temp in temps[:-1]:
                scaled_logits = self._scale_logits_with_temp(base_logits_array, base_temp, temp)
                scaled_probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
                results[model_name][temp] = scaled_probs.tolist()
                
                # Cache the scaled results too
                self.cache.set(f"{question_id}-{prompt}-{model_name}-{temp}", 
                             model_name, temp, scaled_probs.tolist())

        return results

class LLMEvaluator:
    """
    Evaluates conversations using a rubric by generating prompts and querying an LLM.

    Attributes:
        rubric (Dict): A dictionary containing evaluation questions and options.
        prompt_template (str): The pre-defined prompt format for evaluation.
    """

    def __init__(self, system_path: str, logger):
        """
        Initializes the evaluator by loading the rubric from a YAML or JSON file.

        Args:
            system_path (str): Path to the system evaluation file (AML).
        """
        self.prompt_template = ''
        rubric_path = self._load_system(system_path)
        self.rubric = self._load_rubric(rubric_path)
        self.template_parser = TemplateParser()
        self.logger = logger

    def _check_question_condition(self, question_data: Dict, previous_answers: Dict[str, int]) -> bool:
        """
        Checks if conditions are met to evaluate this question.
        
        Args:
            question_data: Question definition from rubric
            previous_answers: Dict of question_id to score for previous questions
            
        Returns:
            bool: Whether the question should be evaluated
        """
        if "conditional" not in question_data:
            return True
            
        condition = self.template_parser.extract_condition(question_data["conditional"])
        return self.template_parser.evaluate_condition(condition, previous_answers)

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

        if question_data.get("type") == "json_question":
            
            # Format JSON question prompt
            schema = question_data["json_schema"]
            rating_scales = question_data.get("rating_scales", {})
            
            prompt_parts = []
            prompt_parts.append(question_data["prompt"])
            
            # Add schema requirements
            prompt_parts.append("\nPlease provide ratings for the following:")
            for field, props in schema["properties"].items():
                description = props.get("description", "")
                prompt_parts.append(f"\n{field}: {description}")
                
                # Add rating scale if provided
                if field in rating_scales:
                    prompt_parts.append("Scale:")
                    for score, desc in rating_scales[field].items():
                        prompt_parts.append(f"  {score}: {desc}")
            
            # Add input text
            prompt_parts.append(f"\nText to evaluate:\n{text_unit}")
            
            # Add format instructions
            prompt_parts.append("\nProvide response in valid JSON format with numerical ratings.")
            if schema.get("required"):
                required_fields = ", ".join(schema["required"])
                prompt_parts.append(f"\nRequired fields: {required_fields}")
                
            return "\n".join(prompt_parts)
        
        else:

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
                
                # Track answers to evaluate conditions
                previous_answers = {}
                
                question_order = [f"Q{i}" for i in range(1, Config.num_questions)] + ["Q0"]
                
                for q_id in question_order:
                    if q_id not in self.rubric:
                        continue
                        
                    q_data = self.rubric[q_id]
                    conditional = self._check_question_condition(q_data, previous_answers)
                    self.logger.info(str(conditional))
                    
                    if conditional == True:

                        # Get LLM probabilities
                        prompt = self.generate_prompt(text_unit, q_id)
                        probs = llm_caller(prompt, q_id)[model_name][temp]
                        result.add_result(q_id, probs)
                        
                        most_likely_score = max(range(len(probs)), 
                                              key=lambda i: probs[i]) + 1
                        previous_answers[q_id] = most_likely_score

                    else:

                        num_options = len(q_data["options"])
                        uniform_probs = [1.0 / num_options] * num_options
                        result.add_result(q_id, uniform_probs)
                
                results.append(result)
                
        return results
    