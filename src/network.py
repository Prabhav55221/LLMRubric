"""
Author: Prabhav Singh
Implementation of: LLMRubric (https://aclanthology.org/2024.acl-long.745v2.pdf)

Calibration Network: Implements judge-specific calibration network model.
"""

import torch
import torch.nn as nn

class CalibrationNetwork(nn.Module):
    """
    Judge-specific calibration network with multi-task learning
    
    Architecture:
    - Two hidden layers with shared + judge-specific parameters
    - Multi-task output heads for each rubric question
    - Two-phase training (pre-training + fine-tuning)
    
    Args:
        num_judges (int): Number of distinct judges in data
        input_dim (int): Input dimension (questions * options)
        h1 (int): First hidden layer size
        h2 (int): Second hidden layer size
    """
    def __init__(self, num_judges=12, input_dim=35, h1=50, h2=50):
        super().__init__()
        self.num_judges = num_judges
        self.num_questions = 7
        
        # Shared weights
        self.W1 = nn.Linear(input_dim + 1, h1)  # Input + bias
        self.W2 = nn.Linear(h1 + 1, h2)
        
        # Judge-specific weights
        self.W1_a = nn.ModuleList([nn.Linear(input_dim + 1, h1) for _ in range(num_judges)])
        self.W2_a = nn.ModuleList([nn.Linear(h1 + 1, h2) for _ in range(num_judges)])
        
        # Question-specific output heads
        self.V = nn.ModuleList([nn.Linear(h2 + 1, 5) for _ in range(self.num_questions)])
        self.V_a = nn.ModuleList([
            nn.ModuleList([nn.Linear(h2 + 1, 5) for _ in range(self.num_questions)])
            for _ in range(num_judges)
        ])

    def forward(self, x, judge_ids):
        """
        Forward pass with judge-specific computations
        
        Args:
            x (Tensor): Input probabilities (batch_size x 35)
            judge_ids (Tensor): Judge IDs (batch_size,)
            
        Returns:
            list: List of output distributions for each question (batch_size x 5)
        """
        batch_size = x.shape[0]
        
        # Add bias and compute first hidden layer
        x = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)
        z1 = torch.relu(
            self.W1(x) + 
            torch.stack([self.W1_a[jid](x[i]) for i, jid in enumerate(judge_ids)])
        )
        
        # Second hidden layer
        z1 = torch.cat([z1, torch.ones(batch_size, 1, device=x.device)], dim=1)
        z2 = torch.relu(
            self.W2(z1) + 
            torch.stack([self.W2_a[jid](z1[i]) for i, jid in enumerate(judge_ids)])
        )
        
        # Compute question outputs
        z2 = torch.cat([z2, torch.ones(batch_size, 1, device=x.device)], dim=1)
        outputs = []
        for q in range(self.num_questions):
            out_q = self.V[q](z2) + torch.stack([
                self.V_a[jid][q](z2[i]) for i, jid in enumerate(judge_ids)
            ])
            outputs.append(torch.softmax(out_q, dim=1))
            
        return outputs