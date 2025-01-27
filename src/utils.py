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
from datetime import datetime

class CacheManager:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[List[float]]:
        cache_key = self._get_cache_key(prompt)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, prompt: str, result: List[float]):
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