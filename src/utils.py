'''
Author: Prabhav Singh
Implentation Of: Eisner et al. (https://aclanthology.org/2024.acl-long.745v2.pdf)
Utils: Cache Manager and Eval Results Format
'''

from typing import Dict, List, Union, Optional
import json
import hashlib
import pickle
import os
from config import Config
from datetime import datetime

class CacheManager:
    def __init__(self, cache_dir: str, config: Config, rubric: dict):
        self.cache_dir = cache_dir
        self.config = config
        self.rubric_hash = self._get_rubric_hash(rubric)
        os.makedirs(cache_dir, exist_ok=True)

    def _get_rubric_hash(self, rubric: dict) -> str:
        """Generate a hash of the rubric to detect changes in rubric version."""
        rubric_json = json.dumps(rubric, sort_keys=True)  # Ensure consistent ordering
        return hashlib.md5(rubric_json.encode()).hexdigest()

    def _get_cache_key(self, prompt: str) -> str:
        """Generate a unique cache key using prompt + model settings + rubric."""
        config_str = f"{self.config.model}-{self.config.temperature}-{self.config.max_retries}"
        key_string = f"{prompt}-{config_str}-{self.rubric_hash}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, prompt: str) -> Optional[List[float]]:
        """Retrieve cached result if available."""
        cache_key = self._get_cache_key(prompt)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def set(self, prompt: str, result: List[float]):
        """Store result in cache using unique key."""
        cache_key = self._get_cache_key(prompt)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

class EvaluationResult:
    def __init__(self):
        self.results: Dict[str, List[float]] = {}
        self.timestamps: Dict[str, datetime] = {}
        
    def add_result(self, question_id: str, probabilities: List[float]):
        self.results[question_id] = probabilities
        self.timestamps[question_id] = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "results": self.results,
            "timestamps": {k: v.isoformat() for k, v in self.timestamps.items()}
        }
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)