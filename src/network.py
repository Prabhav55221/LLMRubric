"""
Author: Prabhav Singh
Implementation of: LLMRubric (https://aclanthology.org/2024.acl-long.745v2.pdf)

Calibration Network: Implements judge-specific calibration network model.
"""

import torch
import torch.nn as nn

class CalibrationNetwork(nn.Module):
    """
    Judge-specific calibration network with multi-task learning.
    Optimized implementation using tensor operations.

    Args:
        num_judges (int): Number of distinct judges
        num_questions (int): Number of questions in rubric
        options_per_q (int): Number of options per question
        h1 (int): First hidden layer size
        h2 (int): Second hidden layer size
    """
    def __init__(self, num_judges=12, num_questions=7, options_per_q=5, h1=50, h2=50):
        super().__init__()
        self.num_judges = num_judges
        self.num_questions = num_questions
        self.options_per_q = options_per_q
        
        # First layer parameters
        self.W1 = nn.Parameter(
            torch.randn((num_questions, options_per_q + 1, h1))
        )
        self.W1_a = nn.Parameter(
            torch.randn((num_judges, num_questions, options_per_q + 1, h1))
        )
        
        # Second layer parameters
        self.W2 = nn.Parameter(
            torch.randn((h1 + 1, h2))
        )
        self.W2_a = nn.Parameter(
            torch.randn((num_judges, h1 + 1, h2))
        )
        
        # Output layer parameters
        self.V = nn.Parameter(
            torch.randn((num_questions, h2 + 1, options_per_q))
        )
        self.V_a = nn.Parameter(
            torch.randn((num_judges, num_questions, h2 + 1, options_per_q))
        )

        # Initialize parameters
        for param in [self.W1, self.W1_a, self.W2, self.W2_a, self.V, self.V_a]:
            nn.init.xavier_uniform_(param)

    def forward(self, x, judge_ids):
        """
        Forward pass using efficient tensor operations
        
        Args:
            x (Tensor): Input probabilities (batch_size x num_questions x options_per_q)
            judge_ids (Tensor): Judge IDs (batch_size,)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Add bias to input
        x = x.view(batch_size, self.num_questions, -1)
        x_bias = torch.cat([
            torch.ones(batch_size, self.num_questions, 1, device=device),
            x
        ], dim=2)
        
        # First layer
        W1_combined = self.W1 + self.W1_a[judge_ids]
        z1 = torch.einsum('bqi,bqih->bqh', x_bias, W1_combined)
        z1 = torch.sigmoid(z1)
        
        # Add bias to first layer output
        z1_bias = torch.cat([
            torch.ones(batch_size, self.num_questions, 1, device=device),
            z1
        ], dim=2)
        
        # Second layer - expand W2 for all questions
        W2_combined = (self.W2[None, None, :, :] + 
                      self.W2_a[judge_ids, None, :, :])
        z2 = torch.einsum('bqi,bqih->bqh', z1_bias, 
                         W2_combined.expand(-1, self.num_questions, -1, -1))
        z2 = torch.sigmoid(z2)
        
        # Add bias to second layer output
        z2_bias = torch.cat([
            torch.ones(batch_size, self.num_questions, 1, device=device),
            z2
        ], dim=2)
        
        # Output layer
        V_combined = self.V + self.V_a[judge_ids]
        logits = torch.einsum('bqi,bqih->bqh', z2_bias, V_combined)
        
        # Split by questions and apply softmax
        outputs = []
        for q in range(self.num_questions):
            out_q = torch.softmax(logits[:, q], dim=1)
            outputs.append(out_q)
            
        return outputs
