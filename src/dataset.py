"""
Author: Prabhav Singh
Implementation of: LLMRubric (https://aclanthology.org/2024.acl-long.745v2.pdf)

Rubric Dataset Implementation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class RubricDataset(Dataset):
    """
    Custom dataset for handling rubric evaluation data
    
    Args:
        llm_probs (list): List of LLM probability distributions (JSON format)
        human_scores (np.array): Array of human scores (n_samples x n_questions)
        judge_ids (np.array): Array of judge identifiers for each row (n_samples,)
    """

    def __init__(self, llm_probs, human_scores, judge_ids):

        self.llm_probs = torch.tensor([
            [p for q in sample.values() for p in q] 
            for sample in llm_probs
        ], dtype=torch.float32)
        
        self.human_scores = torch.tensor(human_scores, dtype=torch.long)
        self.judge_ids = torch.tensor(judge_ids, dtype=torch.long)
        
    def __len__(self):
        """
        Return Length of Dataset

        Returns:
            len (int): Length
        """
        return len(self.llm_probs)
    
    def __getitem__(self, idx):
        return (
            self.llm_probs[idx],    # LLM probabilities (n_questions * n_options floats)
            self.human_scores[idx], # Human ratings (n_questions ints)
            self.judge_ids[idx]     # Judge ID (int)
        )