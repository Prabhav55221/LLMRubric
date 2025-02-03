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
from typing import Dict
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Config:
    """
    Configuration settings for API interaction and model training.

    Attributes:
        openai_api_key (str): API key for authentication with OpenAI's API.
        model (str): LLM model to use.
        temperature (float): Sampling temperature.
        cache_dir (str): Directory for API response caching.
        batch_size (int): Number of requests processed in parallel.
        max_retries (int): Max retry attempts for API failures.
        retry_delay (int): Delay (seconds) between retries.
        cache_expiry_days (int): Cache expiration duration in days.
        num_questions (int): Number of evaluation questions.
        num_judges (int): Number of distinct judges.
        num_options (int): Number of answer choices per question.
        num_folds (int): Number of folds for cross-validation.
        seed (int): Random seed for reproducibility.
        batch_size (int): Training batch size.
        param_grid (dict): Hyperparameter grid for tuning.
        model_save_dir (str): Path to save trained model.
        device (str): Computation device (CPU/GPU).
    """

    # Load API Key from Environment Variables
    load_dotenv()
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    
    # LLM Logits Config
    model: str = "gpt-3.5-turbo-16k"
    temperature: float = 0.8
    cache_dir: str = "cache"
    batch_size: int = 5
    max_retries: int = 3
    retry_delay: int = 1
    cache_expiry_days: int = 7

    # Calibration Configs
    num_questions: int = 7
    num_judges: int = 15
    num_options: int = 5

    # Network Training Configs
    num_folds: int = 5
    seed: int = 42
    batch_size: int = 32  # Training batch size

    model_save_dir: str = "/export/fs06/psingh54/LLMRubric/models/calibration_model.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
