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
    Implements equations (3)-(5) from the paper.
    
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
        
        # Shared weights for each question
        self.W1 = nn.ModuleList([
            nn.Linear(options_per_q + 1, h1, bias=False) 
            for _ in range(num_questions)
        ])
        
        # Judge-specific weights (same input dimension)
        self.W1_a = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(options_per_q + 1, h1, bias=False)
                for _ in range(num_questions)
            ])
            for _ in range(num_judges)
        ])

        # Second layer takes h1 + 1 as input (adding bias)
        self.W2 = nn.Linear(h1 + 1, h2, bias=False)
        self.W2_a = nn.ModuleList([
            nn.Linear(h1 + 1, h2, bias=False) 
            for _ in range(num_judges)
        ])
        
        # Output layer takes h2 + 1 as input
        self.V = nn.ModuleList([
            nn.Linear(h2 + 1, options_per_q, bias=False)
            for _ in range(num_questions)
        ])
        
        self.V_a = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(h2 + 1, options_per_q, bias=False)
                for _ in range(num_questions)
            ])
            for _ in range(num_judges)
        ])

    def forward(self, x, judge_ids):
        """
        Forward pass implementing equations (3)-(5) from paper
        
        Args:
            x (Tensor): Input probabilities (batch_size x num_questions x options_per_q)
            judge_ids (Tensor): Judge IDs (batch_size,)
            
        Returns:
            list: List of output distributions for each question
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Split input by question
        x_by_q = x.view(batch_size, self.num_questions, -1)
        
        outputs = []
        for q in range(self.num_questions):

            x_q = torch.cat([
                torch.ones(batch_size, 1, device=device),
                x_by_q[:, q]
            ], dim=1)
            
            z1 = []
            for i, jid in enumerate(judge_ids):
                shared = self.W1[q](x_q[i])
                judge_specific = self.W1_a[jid][q](x_q[i])
                z1.append(shared + judge_specific)
            z1 = torch.stack(z1)
            z1 = torch.relu(z1)
            
            z1 = torch.cat([
                torch.ones(batch_size, 1, device=device),
                z1
            ], dim=1)
            
            z2 = []
            for i, jid in enumerate(judge_ids):
                shared = self.W2(z1[i])
                judge_specific = self.W2_a[jid](z1[i])
                z2.append(shared + judge_specific)
            z2 = torch.stack(z2)
            z2 = torch.relu(z2)
            
            z2 = torch.cat([
                torch.ones(batch_size, 1, device=device),
                z2
            ], dim=1)
            
            out_q = []
            for i, jid in enumerate(judge_ids):
                shared = self.V[q](z2[i])
                judge_specific = self.V_a[jid][q](z2[i])
                out_q.append(shared + judge_specific)
            out_q = torch.stack(out_q)
            
            # Apply softmax for probability distribution
            outputs.append(torch.softmax(out_q, dim=1))
            
        return outputs