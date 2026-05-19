"""
Evaluation metrics for polyp size classification

Primary metric: Macro-F1
Additional: Per-class F1, Confusion Matrix, AUROC, Balanced Accuracy, ECE
"""

import torch
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score
)
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray = None,
                    class_names: List[str] = None) -> Dict[str, float]:
    """
    Compute all evaluation metrics

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        y_prob: Predicted probabilities (N, num_classes) - optional for AUROC
        class_names: List of class names

    Returns:
        Dictionary of metrics
    """
    if class_names is None:
        class_names = ["le_5mm", "gt_5mm"]

    metrics = {}

    # Macro F1 (primary metric)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', labels=list(range(len(class_names))), zero_division=0)

    # Per-class metrics (force all classes to be included even if missing)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_names))), zero_division=0
    )

    for i, name in enumerate(class_names):
        metrics[f'{name}_f1'] = f1[i]
        metrics[f'{name}_precision'] = precision[i]
        metrics[f'{name}_recall'] = recall[i]
        metrics[f'{name}_support'] = support[i]

    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    per_class_acc = []
    for i, name in enumerate(class_names):
        denom = cm[i].sum()
        acc_i = (cm[i, i] / denom) if denom > 0 else 0.0
        per_class_acc.append(acc_i)
        metrics[f'{name}_acc'] = acc_i

    # Weighted accuracy (support-weighted per-class accuracy)
    total = cm.sum()
    if total > 0:
        weights = cm.sum(axis=1) / total
        metrics['weighted_accuracy'] = float((weights * np.array(per_class_acc)).sum())
    else:
        metrics['weighted_accuracy'] = 0.0

    # Balanced accuracy
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # Overall accuracy
    metrics['accuracy'] = (y_pred == y_true).mean()

    # AUROC per class (if probabilities provided)
    if y_prob is not None:
        try:
            num_classes = len(class_names)
            # One-vs-rest AUROC
            for i, name in enumerate(class_names):
                y_true_binary = (y_true == i).astype(int)
                if len(np.unique(y_true_binary)) > 1:  # Need both classes present
                    metrics[f'{name}_auroc'] = roc_auc_score(y_true_binary, y_prob[:, i])
                else:
                    metrics[f'{name}_auroc'] = 0.0

            # Macro AUROC
            aurocs = [metrics[f'{name}_auroc'] for name in class_names
                      if metrics[f'{name}_auroc'] > 0]
            metrics['macro_auroc'] = np.mean(aurocs) if aurocs else 0.0

        except Exception as e:
            print(f"Warning: Could not compute AUROC: {e}")
            metrics['macro_auroc'] = 0.0

    return metrics


def compute_confusion_matrix(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: List[str] = None) -> np.ndarray:
    """
    Compute confusion matrix

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    if class_names is None:
        class_names = ["le_5mm", "gt_5mm"]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return cm


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: List[str],
                          save_path: str = None,
                          normalize: bool = False) -> plt.Figure:
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize by row

    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count' if not normalize else 'Proportion'})

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def compute_ece(y_true: np.ndarray,
                y_prob: np.ndarray,
                n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE)

    Args:
        y_true: Ground truth labels (N,)
        y_prob: Predicted probabilities (N, num_classes)
        n_bins: Number of bins for calibration

    Returns:
        ECE value
    """
    # Get predicted class and confidence
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def plot_reliability_diagram(y_true: np.ndarray,
                             y_prob: np.ndarray,
                             n_bins: int = 15,
                             save_path: str = None) -> plt.Figure:
    """
    Plot reliability diagram for calibration

    Args:
        y_true: Ground truth labels (N,)
        y_prob: Predicted probabilities (N, num_classes)
        n_bins: Number of bins
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.sum()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(prop_in_bin)
        else:
            bin_accs.append(0)
            bin_confs.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)

    # Compute ECE
    ece = compute_ece(y_true, y_prob, n_bins)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Reliability diagram
    ax.bar(bin_confs, bin_accs, width=1.0/n_bins,
           alpha=0.7, edgecolor='black', label='Outputs')

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')

    # Gap bars
    for conf, acc, count in zip(bin_confs, bin_accs, bin_counts):
        if count > 0:
            ax.plot([conf, conf], [conf, acc], 'gray', linewidth=1, alpha=0.5)

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Reliability Diagram (ECE = {ece:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def format_metrics_for_report(metrics: Dict[str, float],
                               cm: np.ndarray = None,
                               class_names: List[str] = None) -> str:
    """
    Format metrics as markdown table for report

    Args:
        metrics: Dictionary of metrics
        cm: Confusion matrix (optional)
        class_names: List of class names

    Returns:
        Markdown formatted string
    """
    if class_names is None:
        class_names = ["le_5mm", "gt_5mm"]

    report = "# Evaluation Metrics\n\n"

    # Overall metrics
    report += "## Overall Metrics\n\n"
    report += "| Metric | Value |\n"
    report += "|--------|-------|\n"
    report += f"| **Macro F1** (Primary) | **{metrics['macro_f1']:.4f}** |\n"
    report += f"| Balanced Accuracy | {metrics['balanced_accuracy']:.4f} |\n"
    report += f"| Accuracy | {metrics['accuracy']:.4f} |\n"

    if 'macro_auroc' in metrics:
        report += f"| Macro AUROC | {metrics['macro_auroc']:.4f} |\n"

    if 'ece' in metrics:
        report += f"| ECE (Calibration) | {metrics.get('ece', 0):.4f} |\n"

    # Per-class metrics
    report += "\n## Per-Class Metrics\n\n"
    report += "| Class | F1 | Precision | Recall | Support | AUROC |\n"
    report += "|-------|-----|-----------|--------|---------|-------|\n"

    for name in class_names:
        f1 = metrics.get(f'{name}_f1', 0)
        prec = metrics.get(f'{name}_precision', 0)
        rec = metrics.get(f'{name}_recall', 0)
        supp = int(metrics.get(f'{name}_support', 0))
        auroc = metrics.get(f'{name}_auroc', 0)

        report += f"| {name} | {f1:.4f} | {prec:.4f} | {rec:.4f} | {supp} | {auroc:.4f} |\n"

    # Confusion matrix
    if cm is not None:
        report += "\n## Confusion Matrix\n\n"
        report += "```\n"
        # Normalize CM
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

        # Header
        header = "          " + "  ".join([f"{name:>10}" for name in class_names])
        report += header + "\n"

        for i, name in enumerate(class_names):
            row = f"{name:>10}" + "  ".join([f"{cm_norm[i, j]:>10.3f}" for j in range(len(class_names))])
            report += row + "\n"

        report += "```\n"

    return report


# Example usage
if __name__ == "__main__":
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    num_classes = 2

    y_true = np.random.randint(0, num_classes, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    errors = np.random.rand(n_samples) < 0.2
    y_pred[errors] = np.random.randint(0, num_classes, errors.sum())

    # Generate probabilities
    y_prob = np.random.rand(n_samples, num_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print("Metrics:")
    for k, v in metrics.items():
        if not k.endswith('_support'):
            print(f"  {k}: {v:.4f}")

    # Compute and plot confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    fig = plot_confusion_matrix(cm, ["le_5mm", "gt_5mm"],
                                save_path="test_confusion.png")
    plt.close(fig)

    # Compute ECE
    ece = compute_ece(y_true, y_prob)
    print(f"\nECE: {ece:.4f}")

    # Plot reliability diagram
    fig = plot_reliability_diagram(y_true, y_prob, save_path="test_reliability.png")
    plt.close(fig)

    # Generate report
    report = format_metrics_for_report(metrics, cm)
    print("\n" + "=" * 60)
    print(report)
