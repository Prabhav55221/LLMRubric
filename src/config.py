'''
Author: Prabhav Singh
Implentation Of: Eisner et al. (https://aclanthology.org/2024.acl-long.745v2.pdf)
CONFIGS!
'''

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = "gpt-4-0125-preview"
    temperature: float = 0.0
    cache_dir: str = "cache"
    batch_size: int = 5
    max_retries: int = 3
    retry_delay: int = 1