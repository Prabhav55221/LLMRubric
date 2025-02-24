import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from joint_dataset import JointDataset
from joint_model_annotator_embedding import JointModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
from collections import defaultdict
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

def train():
    dataset = JointDataset(f"../data/in_domain_train.json")
    eval_dataset = JointDataset(f"../data/in_domain_dev.json")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = JointModel(18, 5, 5, 6, 512, 25, 8)
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
    no_improvement_count = 0  # Tracks consecutive epochs without improvement
    max_no_improvement = 2    # Stop after 3 epochs of no improvement

    epoch = 0
    while True:
        epoch += 1
        model.train() 
        running_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            known_questions = batch[0]
            inputs = batch[1]
            labels = batch[2]
            annotators = batch[3]
            inputs, labels, annotators = inputs.to(device), labels.to(device), annotators.to(device) 
            optimizer.zero_grad()
            outputs = model(inputs, annotators)
            loss = torch.tensor([0.0])
            for i in range(annotators.shape[1]):
                loss += criterion(outputs[:, i, :], labels[:, i, :])
            loss /= annotators.shape[1]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0:
                print(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch} - Training Loss: {avg_loss:.4f}")

        accuracy, dev_loss, metrics = evaluate(model, eval_dataset, criterion, device)
        dev_losses.append(dev_loss)
        dev_accuracies.append(accuracy)
        dev_metrics.append(metrics)
        print(f"Epoch {epoch} - Dev Accuracy: {accuracy:.4f}, Dev Loss: {dev_loss:.4f}")
        print(f"Metrics - RMSE: {metrics['rmse']:.4f}, Pearson: {metrics['pearson']:.4f}, Spearman: {metrics['spearman']:.4f}, Kendall: {metrics['kendall']:.4f}")
        
        scheduler.step(dev_loss)

        if dev_loss < prev_dev_loss:
            no_improvement_count = 0 
        else:
            no_improvement_count += 1

        prev_dev_loss = dev_loss

        if no_improvement_count >= max_no_improvement or epoch >= 40:
            print(f"Stopping early after {epoch} epochs. No improvement in dev loss for {max_no_improvement} consecutive epochs.")
            break

    torch.save(model.state_dict(), f"joint_model_judges1.pth")
    print("Training complete. Model saved.")

    # Plot Training Loss vs. Dev Loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epoch+1), train_losses, label="Training Loss", marker="o")
    plt.plot(range(1, epoch+1), dev_losses, label="Dev Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig(f"loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epoch+1), dev_accuracies, label="Dev Accuracy", marker="d", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Dev Accuracy Curve")
    plt.legend()
    plt.grid()
    plt.savefig(f"accuracy_curve.png")
    plt.close()

def plot_calibration(all_pred_probs, all_true_scores, num_classes=4, n_bins=10):
    """
    Generates calibration plots for multiple classes.
    
    Parameters:
    - all_pred_probs: List of probability distributions per sample.
    - all_true_scores: List of actual labels (ground truth).
    - num_classes: Number of classes (e.g., 4 in this case).
    - n_bins: Number of bins for the calibration curve.
    """
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, num_classes, figsize=(16, 4), sharey=True)

    all_pred_probs = np.array(all_pred_probs)  # Convert list to array
    all_true_scores = np.array(all_true_scores)

    for class_idx in range(num_classes):
        # Convert labels to binary (1 if true label == class_idx, else 0)
        y_bin = (all_true_scores == class_idx).astype(int)
        
        # Get the predicted probabilities for the current class
        prob_true, prob_pred = calibration_curve(y_bin, all_pred_probs[:, class_idx], n_bins=n_bins)

        ax = axes[class_idx]
        ax.plot(prob_pred, prob_true, marker="o", label=f"Class {class_idx + 1}", color="red")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

        ax.set_xlabel("Predicted Probability")
        if class_idx == 0:
            ax.set_ylabel("Actual Frequency")
        ax.set_title(f"Calibration Plot (y0={class_idx + 1})")
        ax.legend()

    plt.tight_layout()
    plt.savefig("calibration_plot.png")

#evaluation during training
def evaluate(model, dataset, criterion, device="cpu", plot=False):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    correct = 0
    total_loss = 0.0
    count = 0

    all_predictions = []
    all_true_scores = []
    all_pred_probs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating on dev")):
            known_questions = batch[0]
            inputs = batch[1]
            labels = batch[2]
            annotators = batch[3]
            inputs, labels, annotators = inputs.to(device), labels.to(device), annotators.to(device) 
            outputs = model(inputs, annotators)
            loss = torch.tensor([0.0])
            for i in range(annotators.shape[1]):
                loss += criterion(outputs[:, i, :], labels[:, i, :])
                probs = torch.softmax(outputs[:, i, :], dim=-1).squeeze(0).cpu().numpy()
                all_pred_probs.append(probs) 
                predicted = torch.argmax(outputs[:, i]).item()
                true_label = torch.argmax(labels[:, i]).item()
                all_predictions.append(predicted)
                all_true_scores.append(true_label)

                if predicted == true_label:
                    correct += 1
                count += 1
            total_loss += loss.item()

            

            

    avg_loss = total_loss / count if count > 0 else 0
    accuracy = correct / count if count > 0 else 0

    # Compute additional metrics
    metrics = {
        'rmse': np.sqrt(np.mean((np.array(all_predictions) - np.array(all_true_scores))**2)),
        'pearson': pearsonr(all_predictions, all_true_scores)[0] if len(set(all_predictions)) > 1 else 0.0,
        'spearman': spearmanr(all_predictions, all_true_scores)[0] if len(set(all_predictions)) > 1 else 0.0,
        'kendall': kendalltau(all_predictions, all_true_scores)[0] if len(set(all_predictions)) > 1 else 0.0
    }

    if plot:
        plot_calibration(all_pred_probs, all_true_scores)

    return accuracy, avg_loss, metrics

#evaluation by 
def evaluate_with_known_size():
    device = "cpu"
    judge_list = list(range(24))
    metrics_by_known_questions = defaultdict(lambda: {'all_predictions': [], 'all_true_scores': []})
    model = JointModel(18, 5, 5, 6, 512, 25, 8)
    model.load_state_dict(torch.load(f"joint_model_judges.pth"))
    model.to(device)
    model.eval()
    dataset = JointDataset(f"../data/in_domain_dev.json")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    evaluate(model, dataset, criterion, plot=True)
    
    total_loss = 0.0
    count = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating on dev")):
        known_questions = batch[0]
        inputs = batch[1]
        labels = batch[2]
        annotators = batch[3]
        inputs, labels, annotators = inputs.to(device), labels.to(device), annotators.to(device) 
        outputs = model(inputs, annotators)
        loss = torch.tensor([0.0])
        for i in range(annotators.shape[1]):
            if known_questions[0][i].item() == 1: #add this to only evaluate the masked questions
                continue
            '''if annotators[0][i].item() == -1: #add this to only evaluate human questions
                continue'''
            loss += criterion(outputs[:, i, :], labels[:, i, :])
            predicted = torch.argmax(outputs[:, i]).item()
            true_label = torch.argmax(labels[:, i]).item()

            known_question = (inputs[:, :, 0] == 0).sum().item()      
            
            metrics_by_known_questions[known_question]['all_predictions'].append(predicted)
            metrics_by_known_questions[known_question]['all_true_scores'].append(true_label)

            if predicted == true_label:
                correct += 1
            count += 1
        total_loss += loss.item()
    
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
    plt.savefig(f"metrics_judges_unknown_all.png") #change to appropriate name

if __name__ == "__main__":
    #train()
    evaluate_with_known_size()