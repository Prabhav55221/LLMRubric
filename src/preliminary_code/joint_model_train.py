import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from preliminary_code.joint_dataset import JointDataset
from joint_model import JointModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
from collections import defaultdict

def train():
    judge_list = [2, 3, 5, 7, 10, 11, 12, 13, 15, 16, 18, 22]

    for judge in judge_list:
        dataset = JointDataset(f"../data/joint_model_data/in_domain_train{judge}.json")
        eval_dataset = JointDataset(f"../data/joint_model_data/in_domain_dev{judge}.json")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = JointModel(9, 5, 3, 5, 512)
        device = "cpu"
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        log_interval = 100

        train_losses = []  # Stores training loss per epoch
        dev_losses = []    # Stores dev loss per epoch
        dev_accuracies = [] # Stores dev accuracy per epoch
        dev_metrics = []   # Stores RMSE, Pearson, Spearman, Kendall

        prev_dev_loss = float('inf')

        count = 0
        while True:
            count += 1
            model.train() 
            running_loss = 0.0

            for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"Training")):
                inputs, labels = inputs.to(device), labels.to(device) 
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if (batch_idx + 1) % log_interval == 0:
                    print(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

            avg_loss = running_loss / len(dataloader)
            train_losses.append(avg_loss)
            print(f"Training Loss: {avg_loss:.4f}")

            # Evaluate on dev set
            accuracy, dev_loss, metrics = evaluate(model, eval_dataset, criterion, device)
            dev_losses.append(dev_loss)
            dev_accuracies.append(accuracy)
            dev_metrics.append(metrics)
            print(f"Dev Accuracy: {accuracy:.4f}, Dev Loss: {dev_loss:.4f}")
            print(f"Metrics - RMSE: {metrics['rmse']:.4f}, Pearson: {metrics['pearson']:.4f}, Spearman: {metrics['spearman']:.4f}, Kendall: {metrics['kendall']:.4f}")
            
            scheduler.step(dev_loss)

            if prev_dev_loss - dev_loss < 1e-4 and count > 40:
                break
            prev_dev_loss = dev_loss

        # Save model after training
        torch.save(model.state_dict(), f"joint_model{judge}.pth")
        print("Training complete. Model saved.")

        # Plot Training Loss vs. Dev Loss
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, count+1), train_losses, label="Training Loss", marker="o")
        plt.plot(range(1, count+1), dev_losses, label="Dev Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve (Judge {judge})")
        plt.legend()
        plt.grid()
        plt.savefig(f"loss_curve_{judge}.png")
        plt.close()

        # Plot Dev Accuracy
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, count+1), dev_accuracies, label="Dev Accuracy", marker="d", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Dev Accuracy Curve (Judge {judge})")
        plt.legend()
        plt.grid()
        plt.savefig(f"accuracy_curve_{judge}.png")
        plt.close()


def evaluate(model, dataset, criterion, device="cpu"):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    correct = 0
    total_loss = 0.0
    count = 0

    all_predictions = []
    all_true_scores = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Evaluating Accuracy")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = torch.argmax(outputs).item()
            true_label = torch.argmax(labels).item()
            all_predictions.append(predicted)
            all_true_scores.append(true_label)

            if predicted == true_label:
                correct += 1

            count += 1

    avg_loss = total_loss / count if count > 0 else 0
    accuracy = correct / count if count > 0 else 0

    # Compute additional metrics
    metrics = {
        'rmse': np.sqrt(np.mean((np.array(all_predictions) - np.array(all_true_scores))**2)),
        'pearson': pearsonr(all_predictions, all_true_scores)[0] if len(set(all_predictions)) > 1 else 0.0,
        'spearman': spearmanr(all_predictions, all_true_scores)[0] if len(set(all_predictions)) > 1 else 0.0,
        'kendall': kendalltau(all_predictions, all_true_scores)[0] if len(set(all_predictions)) > 1 else 0.0
    }

    return accuracy, avg_loss, metrics

def evaluate_with_known_size():
    judge_list = [2, 3, 5, 7, 10, 11, 12, 13, 15, 16, 18, 22]
    device = "cpu"
    
    
    
    for judge in judge_list:
        metrics_by_known_questions = defaultdict(lambda: {'all_predictions': [], 'all_true_scores': []})
        model = JointModel(9, 5, 3, 5, 512)
        model.load_state_dict(torch.load(f"joint_model{judge}.pth"))
        model.to(device)
        model.eval()
        dataset = JointDataset(f"../data/joint_model_data/in_domain_dev{judge}.json")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        total_loss = 0.0
        count = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Evaluating Accuracy")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            known_questions = (inputs[:, :, 0] == 0).sum().item()
            
            predicted = torch.argmax(outputs).item()
            true_label = torch.argmax(labels).item()
            
            metrics_by_known_questions[known_questions]['all_predictions'].append(predicted)
            metrics_by_known_questions[known_questions]['all_true_scores'].append(true_label)
            
            if predicted == true_label:
                correct += 1
            
            count += 1
        
    # Compute separate metrics for each known_questions category
        final_metrics = {}
        for known_q, data in metrics_by_known_questions.items():
            all_predictions = np.array(data['all_predictions'])
            all_true_scores = np.array(data['all_true_scores'])
            
            if len(set(all_predictions)) > 1:
                pearson = pearsonr(all_predictions, all_true_scores)[0]
                spearman = spearmanr(all_predictions, all_true_scores)[0]
                kendall = kendalltau(all_predictions, all_true_scores)[0]
            else:
                pearson = spearman = kendall = 0.0
            
            final_metrics[known_q] = {
                'rmse': np.sqrt(np.mean((all_predictions - all_true_scores) ** 2)),
                'pearson': pearson,
                'spearman': spearman,
                'kendall': kendall
            }
        
        #print("Metrics by known questions:", final_metrics)
        sorted_metrics = dict(sorted(final_metrics.items()))
        
        # Print metrics in sorted order
        print("Metrics by known questions (sorted):")
        for known_q, metrics in sorted_metrics.items():
            print(f"Known Questions {known_q}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        known_questions = list(sorted_metrics.keys())
        metrics_names = ['rmse', 'pearson', 'spearman', 'kendall']
        
        # Create figure and axis with a larger size
        plt.figure(figsize=(12, 8))
        
        # Plot each metric
        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
        markers = ['o', 's', '^', 'D']
        
        for idx, metric in enumerate(metrics_names):
            values = [sorted_metrics[q][metric] for q in known_questions]
            plt.plot(known_questions, values, 
                    label=metric.upper(),
                    color=colors[idx],
                    marker=markers[idx],
                    linewidth=2,
                    markersize=8)

        # Customize the plot
        plt.title('Evaluation Metrics by Number of Known Questions', 
                fontsize=14, pad=20)
        plt.xlabel('Number of Known Questions', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Customize x-axis
        plt.xticks(known_questions)
        
        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plt.savefig(f"metrics_judge{judge}.png")

if __name__ == "__main__":
    #train()
    evaluate_with_known_size()