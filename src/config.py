"""
Author: Prabhav Singh
Implementation of: LLMRubric (https://aclanthology.org/2024.acl-long.745v2.pdf)

Configuration class for setting up API interactions and caching.

This module defines a configuration class using Python's `dataclass`
to store settings for API usage, including model parameters, caching,
and retry policies.
"""

import os
import torch
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
        cache_expiry_days (int): Time in days post which, cache will be removed!
    """

    # LLM Logits Config
    load_dotenv()
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    model: str = "gpt-3.5-turbo-16k"
    temperature: float = 0.8
    cache_dir: str = "cache"
    batch_size: int = 5 
    max_retries: int = 3
    retry_delay: int = 1
    cache_expiry_days: int = 7

    # Calibration Configs
    num_questions: int = 7
    num_judges: int = 12
    num_options: int = 5

    # Network Training Configs
    num_folds: int = 5
    seed: int = 42
    batch_size: int = 32
    param_grid: dict = {
        "h1": [10, 25, 50, 100],
        "h2": [10, 25, 50, 100],
        "batch_size": [32, 64, 128, 256],
        "lr": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
        "num_epochs": [5, 10, 20, 30, 40, 50]
    }
    model_save_dir: str = "/export/fs06/psingh54/LLMRubric/models/calibration_model.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
