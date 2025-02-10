"""
Author: Prabhav Singh
Implementation of: LLMRubric (https://aclanthology.org/2024.acl-long.745v2.pdf)

Rubric Dataset Implementation
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
    
class RubricDataset(Dataset):
    """
    Custom dataset for handling rubric evaluation data with multiple LLM outputs per text
    
    Args:
        llm_probs (list): List of LLM probability distributions for all model-temp combinations
        human_scores (np.array): Array of human scores (n_samples x n_questions)
        judge_ids (np.array): Array of judge identifiers 
        config (Config): Configuration object for model info
    """
    def __init__(self, llm_probs, human_scores, judge_ids, config):
        self.config = config
        expansion_factor = config.get_dataset_expansion_factor()
        
        # Expand human scores and judge IDs to match LLM combinations
        self.human_scores = torch.tensor(np.repeat(human_scores, expansion_factor, axis=0) - 1, dtype=torch.long)
        self.judge_ids = torch.tensor(np.repeat(judge_ids, expansion_factor) - 1, dtype=torch.long)
        
        # Process LLM probabilities for all model-temp combinations
        self.llm_probs = []
        
        # For each text example
        for probs in llm_probs:

            model_name = probs['model']
            temp = probs['temperature']
            
            # Extract probabilities from results
            prob_tensor = torch.tensor([
                p for q in probs['results'].values() for p in q
            ], dtype=torch.float32)
            
            self.llm_probs.append(prob_tensor)
        
        self.llm_probs = torch.stack(self.llm_probs)
        
    def __len__(self):
        return len(self.llm_probs)
    
    def __getitem__(self, idx):
        return (
            self.llm_probs[idx],  # LLM probabilities
            self.human_scores[idx],  # Human ratings
            self.judge_ids[idx] # Judge ID
        )

    def get_split_indices(self, train_ratio=0.8):
        """
        Get train/val split indices that keep same-text entries together
        """
        unique_texts = len(self.llm_probs) // self.config.get_dataset_expansion_factor()
        indices = np.random.permutation(unique_texts)
        
        train_size = int(train_ratio * unique_texts)
        train_texts = indices[:train_size]
        val_texts = indices[train_size:]
        
        # Expand indices to account for model-temp combinations
        train_indices = []
        val_indices = []
        
        for i in range(len(train_texts)):
            start_idx = i * self.config.get_dataset_expansion_factor()
            train_indices.extend(range(
                start_idx, 
                start_idx + self.config.get_dataset_expansion_factor()
            ))
            
        for i in range(len(val_texts)):
            start_idx = i * self.config.get_dataset_expansion_factor()
            val_indices.extend(range(
                start_idx, 
                start_idx + self.config.get_dataset_expansion_factor()
            ))
            
        return train_indices, val_indices