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
from tqdm import tqdm
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
                for q in range(model.num_questions):
                    loss += criterion(outputs[q], y_batch[:, q])
            else:
                # Focus on main task (Q0)
                loss = criterion(outputs[model.num_questions - 1], y_batch[:, model.num_questions - 1])
                
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
            pred_probs = outputs[model.num_questions - 1].cpu().numpy()
            pred_scores = np.sum(pred_probs * np.arange(1,model.num_questions - 1), axis=1)
            predictions.extend(pred_scores)
            true_scores.extend(y_batch[:,model.num_questions - 1].cpu().numpy())
    
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

def grid_search_cv(dataset, param_grid, input_dim):
    """
    Perform Grid Search with 5-fold Cross-Validation to find the best hyperparameters.
    
    Args:
        dataset (Dataset): The dataset used for training.
        param_grid (dict): Dictionary of hyperparameters to search over.
        input_dim (int): Input dimension of probabilities

    Returns:
        dict: Best hyperparameters found.
    """

    best_params = None
    best_loss = float("inf")
    kfold = KFold(n_splits=5, shuffle=True, random_state=Config.seed)

    for h1, h2, batch_size, lr, num_epochs in tqdm(product(*param_grid.values())):
        losses = []
        for train_idx, val_idx in kfold.split(dataset):
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            model = CalibrationNetwork(num_judges=Config.num_judges, input_dim=input_dim, h1=h1, h2=h2).to(Config.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            # Training Loop
            for epoch in range(num_epochs):
                model.train()
                for x_batch, y_batch, j_batch in train_loader:
                    outputs = model(x_batch, j_batch)
                    loss = sum(criterion(outputs[q], y_batch[:, q]) for q in range(Config.num_questions))  # Multi-task loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Validation Loop
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x_batch, y_batch, j_batch in val_loader:
                    outputs = model(x_batch, j_batch)
                    loss = sum(criterion(outputs[q], y_batch[:, q]) for q in range(Config.num_questions))
                    val_losses.append(loss.item())

            avg_loss = np.mean(val_losses)
            losses.append(avg_loss)

        final_loss = np.mean(losses)
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = {"h1": h1, "h2": h2, "batch_size": batch_size, "lr": lr, "num_epochs": num_epochs}

    return best_params

def calibrate(csv_path, json_path, logger):
    """
    Calibrate function for training and evaluating the calibration network.

    Args:
        csv_path (str): Path to CSV containing human scores and judge IDs.
        json_path (str): Path to JSON file with LLM probability outputs.
        logger (logging.Logger): Logger for tracking API calls and errors.
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

    # Define hyperparameter grid for Grid Search CV
    input_dim = Config.num_questions * Config.num_options
    param_grid = Config.param_grid

    logger.info('Searching for best parameters with Grid CV!')

    # Perform Grid Search Cross-Validation
    best_params = grid_search_cv(dataset, param_grid, input_dim)

    logger.info(f"Best Parameters found: {best_params}")

    # Initialize final model with best parameters
    model = CalibrationNetwork(
        num_judges=len(set(judge_ids)), 
        input_dim=input_dim, 
        h1=best_params["h1"], 
        h2=best_params["h2"],
        num_question=Config.num_questions
    ).to(Config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])

    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)

    # **Pre-training Phase**
    logger.info('\n\n---PRETRAINING STEP---')
    train_model(model, train_loader, best_params["num_epochs"], optimizer, phase="pre-train")

    # **Fine-tuning Phase**
    logger.info('\n\n---FINETUNING STEP---')
    train_model(model, train_loader, best_params["num_epochs"], optimizer, phase="fine-tune")

    # Evaluate the model
    metrics = evaluate_model(model, val_loader)
    print("\nEvaluation Metrics:", metrics)

    # Save the model
    torch.save(model.state_dict(), Config.model_save_dir)
    print("Model saved successfully!")

    return metrics