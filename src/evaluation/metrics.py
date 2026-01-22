"""
Evaluation Metrics for Anomaly Detection
Comprehensive metrics including precision, recall, F1, AUC-ROC, and AUC-PR.
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute comprehensive anomaly detection metrics.
    
    Args:
        y_true: Ground truth labels (1 = anomaly, 0 = normal)
        y_pred: Predicted labels
        y_scores: Anomaly scores (optional, for AUC metrics)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    
    # False positive rate (important for anomaly detection)
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Detection rate (same as recall)
    metrics['detection_rate'] = metrics['recall']
    
    # AUC metrics if scores provided
    if y_scores is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
            metrics['auc_pr'] = average_precision_score(y_true, y_scores)
        except ValueError:
            # Handle case where only one class is present
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """Print metrics in a formatted way."""
    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation Results")
    print(f"{'='*50}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  F1 Score:       {metrics['f1']:.4f}")
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:        {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:         {metrics['auc_pr']:.4f}")
    print(f"  FPR:            {metrics['fpr']:.4f}")
    print(f"{'='*50}")
    print(f"  TP: {metrics['true_positives']:5d}  |  FP: {metrics['false_positives']:5d}")
    print(f"  FN: {metrics['false_negatives']:5d}  |  TN: {metrics['true_negatives']:5d}")
    print(f"{'='*50}\n")


def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    save_path: str = None
):
    """
    Plot ROC curves for multiple models.
    
    Args:
        results: Dict of {model_name: (y_true, y_pred, y_scores)}
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    for name, (y_true, y_pred, y_scores) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")
    plt.close()


def plot_precision_recall_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    save_path: str = None
):
    """
    Plot Precision-Recall curves for multiple models.
    
    Args:
        results: Dict of {model_name: (y_true, y_pred, y_scores)}
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    for name, (y_true, y_pred, y_scores) in results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})', linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PR curves to {save_path}")
    plt.close()


def compare_models(
    results: Dict[str, Dict[str, float]]
) -> str:
    """
    Create a comparison table of model metrics.
    
    Args:
        results: Dict of {model_name: metrics_dict}
        
    Returns:
        Formatted comparison table string
    """
    headers = ['Model', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'AUC-PR', 'FPR']
    
    lines = [
        f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10} {'AUC-PR':>10} {'FPR':>10}",
        "-" * 90
    ]
    
    for name, metrics in results.items():
        line = f"{name:<20} {metrics.get('precision', 0):>10.4f} {metrics.get('recall', 0):>10.4f} "
        line += f"{metrics.get('f1', 0):>10.4f} {metrics.get('auc_roc', 0):>10.4f} "
        line += f"{metrics.get('auc_pr', 0):>10.4f} {metrics.get('fpr', 0):>10.4f}"
        lines.append(line)
    
    return "\n".join(lines)

