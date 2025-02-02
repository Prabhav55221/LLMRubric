"""
Author: Prabhav Singh
Implementation of: LLMRubric (https://aclanthology.org/2024.acl-long.745v2.pdf)

LLM-Rubric Calibration Network Implementation: Uses judge-specific calibration network for aligning 
LLM rubric scores with human evaluations.
"""

import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from itertools import product
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import pearsonr, spearmanr, kendalltau

# SRC
from src.network import CalibrationNetwork
from src.config import Config
from src.dataset import RubricDataset

def train_model(model, dataloader, epochs, optimizer, phase='pre-train'):
    """
    Training loop with phase-specific objective
    
    Args:
        model: CalibrationNetwork instance
        dataloader: Training data loader
        epochs: Number of training epochs
        optimizer: Configured optimizer
        phase: 'pre-train' (all questions) or 'fine-tune' (Q0 only)
    """
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch, j_batch in dataloader:
            outputs = model(x_batch, j_batch)
            loss = 0
            
            if phase == 'pre-train':
                # Multi-task loss for all questions
                for q in range(7):
                    loss += criterion(outputs[q], y_batch[:, q])
            else:
                # Focus on main task (Q0)
                loss = criterion(outputs[6], y_batch[:, 6])
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"{phase} Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

def evaluate_model(model, dataloader):
    """
    Evaluate model performance on validation/test set
    
    Returns:
        dict: Dictionary of evaluation metrics (RMSE, correlations)
    """
    model.eval()
    predictions = []
    true_scores = []
    
    with torch.no_grad():
        for x_batch, y_batch, j_batch in dataloader:
            outputs = model(x_batch, j_batch)
            # Get expected value for Q0
            pred_probs = outputs[6].cpu().numpy()
            pred_scores = np.sum(pred_probs * np.arange(1,6), axis=1)
            predictions.extend(pred_scores)
            true_scores.extend(y_batch[:,6].cpu().numpy())
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((np.array(predictions) - np.array(true_scores))**2))
    pearson = pearsonr(predictions, true_scores)[0]
    spearman = spearmanr(predictions, true_scores)[0]
    kendall = kendalltau(predictions, true_scores)[0]
    
    return {
        'rmse': rmse,
        'pearson': pearson,
        'spearman': spearman,
        'kendall': kendall
    }

def grid_search_cv(dataset, param_grid):
    """
    Perform Grid Search with 5-fold Cross-Validation to find the best hyperparameters.
    
    Args:
        dataset (Dataset): The dataset used for training.
        param_grid (dict): Dictionary of hyperparameters to search over.

    Returns:
        dict: Best hyperparameters found.
    """

    best_params = None
    best_loss = float("inf")
    kfold = KFold(n_splits=5, shuffle=True, random_state=Config.seed)

    for h1, h2, batch_size, lr, num_epochs in product(*param_grid.values()):
        losses = []
        for train_idx, val_idx in kfold.split(dataset):
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            model = CalibrationNetwork(num_judges=12, input_dim=35, h1=h1, h2=h2).to(Config.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            # Training Loop
            for epoch in range(num_epochs):
                model.train()
                for x_batch, y_batch, j_batch in train_loader:
                    outputs = model(x_batch, j_batch)
                    loss = sum(criterion(outputs[q], y_batch[:, q]) for q in range(7))  # Multi-task loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Validation Loop
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x_batch, y_batch, j_batch in val_loader:
                    outputs = model(x_batch, j_batch)
                    loss = sum(criterion(outputs[q], y_batch[:, q]) for q in range(7))
                    val_losses.append(loss.item())

            avg_loss = np.mean(val_losses)
            losses.append(avg_loss)

        final_loss = np.mean(losses)
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = {"h1": h1, "h2": h2, "batch_size": batch_size, "lr": lr, "num_epochs": num_epochs}

    return best_params

def main(csv_path, json_path, num_epochs_pretrain=20, num_epochs_finetune=10):
    """
    Main function for training and evaluating the calibration network.

    Args:
        csv_path (str): Path to CSV containing human scores and judge IDs.
        json_path (str): Path to JSON file with LLM probability outputs.
        num_epochs_pretrain (int): Number of epochs for pre-training.
        num_epochs_finetune (int): Number of epochs for fine-tuning.
    """

    # Load Data
    df = pd.read_csv(csv_path)
    with open(json_path, "r") as f:
        llm_outputs = json.load(f)

    # Extract judge IDs and human scores
    judge_ids = df["Judge_IDS"].values
    human_scores = df.iloc[:, :-2].values

    # Create dataset
    dataset = RubricDataset(llm_outputs, human_scores, judge_ids)
    
    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Define hyperparameter grid for Grid Search CV
    param_grid = {
        "h1": [10, 25, 50, 100],
        "h2": [10, 25, 50, 100],
        "batch_size": [32, 64, 128, 256],
        "lr": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
        "num_epochs": [5, 10, 20, 30, 40, 50]
    }

    # Perform Grid Search Cross-Validation
    best_params = grid_search_cv(dataset, param_grid)

    # Initialize final model with best parameters
    model = CalibrationNetwork(
        num_judges=len(set(judge_ids)), 
        input_dim=35, 
        h1=best_params["h1"], 
        h2=best_params["h2"]
    ).to(Config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

    # **Pre-training Phase**
    train_model(model, train_loader, best_params["num_epochs"], optimizer, phase="pre-train")

    # **Fine-tuning Phase**
    train_model(model, train_loader, best_params["num_epochs"], optimizer, phase="fine-tune")

    # Evaluate the model
    metrics = evaluate_model(model, val_loader)
    print("Evaluation Metrics:", metrics)

    # Save the model
    torch.save(model.state_dict(), "calibration_model.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file with human scores.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file with LLM-generated probabilities.")

    args = parser.parse_args()
    
    main(args.csv_path, args.json_path)