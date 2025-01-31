"""
Author: Prabhav Singh
Implementation of: LLMRubric (https://aclanthology.org/2024.acl-long.745v2.pdf)

Configuration class for setting up API interactions and caching.

This module defines a configuration class using Python's `dataclass`
to store settings for API usage, including model parameters, caching,
and retry policies.
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """
    Configuration settings for API interaction.

    Attributes:
        openai_api_key (str): API key for authentication with OpenAI's API. We recommend stroring as an env variable!
        model (str): The model name to use. Since we use log-probs, "gpt-3.5-turbo-16k" is probably the best option.
        temperature (float): Sampling temperature (0.8 is what OpenAI is rumored to use).
        cache_dir (str): Directory to store cached API responses.
        batch_size (int): Number of requests processed in parallel.
        max_retries (int): Maximum number of retries in case of API failure.
        retry_delay (int): Time in seconds between retries.
    """

    load_dotenv()
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    model: str = "gpt-3.5-turbo-16k"
    temperature: float = 0.8
    cache_dir: str = "cache"
    batch_size: int = 5 
    max_retries: int = 3
    retry_delay: int = 1
