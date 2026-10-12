"""Framework-independent evaluation of 5-class Kellgren-Lawrence CNN predictions.

Dependencies: numpy, pandas, scikit-learn, matplotlib.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve, cohen_kappa_score, mean_absolute_error)


def evaluate_predictions(y_true, y_prob=None, y_pred=None, *, model_name='cnn',
                         output_dir='metrics_output', class_names=None, logits=None):
    """Evaluate predictions; provide y_true and either y_prob, logits, or y_pred.

    y_true: shape (N,), integer labels 0..4.
    y_prob: shape (N,5), class probabilities in KL0..KL4 order.
    logits: shape (N,5), raw class scores (converted via softmax).
    y_pred: shape (N,), hard class predictions (AUROC unavailable without scores).
    """
    labels = np.arange(5)
    names = list(class_names) if class_names is not None else [f'KL{i}' for i in labels]
    if len(names) != 5:
        raise ValueError('class_names must contain exactly five names.')
    yt = np.asarray(y_true).reshape(-1)
    if yt.size == 0 or not np.all(np.isin(yt, labels)):
        raise ValueError('y_true must be a nonempty vector of KL labels 0..4.')
    if sum(v is not None for v in (y_prob, logits, y_pred)) != 1:
        raise ValueError('Supply exactly one of y_prob, logits, or y_pred.')
    probs = None
    if logits is not None:
        scores = np.asarray(logits, dtype=float)
        if scores.shape != (len(yt), 5) or not np.all(np.isfinite(scores)):
            raise ValueError('logits must be a finite N x 5 array.')
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
    elif y_prob is not None:
        probs = np.asarray(y_prob, dtype=float)
        if probs.shape != (len(yt), 5) or not np.all(np.isfinite(probs)):
            raise ValueError('y_prob must be a finite N x 5 array.')
        if np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(axis=1), 1, atol=1e-4):
            raise ValueError('Each probability row must sum to 1 and lie within [0,1].')
    yp = np.argmax(probs, axis=1) if probs is not None else np.asarray(y_pred).reshape(-1)
    if len(yp) != len(yt) or not np.all(np.isin(yp, labels)):
        raise ValueError('y_pred must match y_true length and contain labels 0..4.')

    out = Path(output_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)
    precision, recall, f1, support = precision_recall_fscore_support(
        yt, yp, labels=labels, zero_division=0)
    per_class = pd.DataFrame({'class': names, 'precision': precision, 'recall': recall,
                              'f1': f1, 'support': support})
    aucs = []
    for k in labels:
        binary = (yt == k).astype(int)
        auc = float(roc_auc_score(binary, probs[:, k])) if (probs is not None and len(np.unique(binary)) == 2) else np.nan
        aucs.append(auc)
    per_class['auroc_ovr'] = aucs
    per_class.to_csv(out / 'per_class_metrics.csv', index=False)

    cm = confusion_matrix(yt, yp, labels=labels)
    def plot_cm(normalized):
        data = cm.astype(float)
        if normalized:
            denom = data.sum(axis=1, keepdims=True)
            data = np.divide(data, denom, out=np.zeros_like(data), where=denom != 0)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(data, cmap='Blues', vmin=0, vmax=1 if normalized else None)
        ax.set(xticks=range(5), yticks=range(5), xticklabels=names, yticklabels=names,
               xlabel='Predicted KL grade', ylabel='True KL grade',
               title='Normalized confusion matrix' if normalized else 'Confusion matrix')
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f'{data[i,j]:.2f}' if normalized else str(cm[i,j]),
                        ha='center', va='center', color='white' if data[i,j] > data.max()/2 else 'black')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / ('normalized_confusion_matrix.png' if normalized else 'confusion_matrix.png'), dpi=180)
        plt.close(fig)
    plot_cm(False)
    plot_cm(True)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(5)
    for i, metric in enumerate(['precision', 'recall', 'f1']):
        ax.bar(x + (i-1)*0.25, per_class[metric], 0.25, label=metric)
    ax.set(xticks=x, xticklabels=names, ylim=(0, 1.05), ylabel='Score', title='Per-class classification metrics')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'per_class_metrics.png', dpi=180)
    plt.close(fig)

    if probs is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        plotted = False
        for k, auc in enumerate(aucs):
            if np.isfinite(auc):
                fpr, tpr, _ = roc_curve((yt == k).astype(int), probs[:, k])
                ax.plot(fpr, tpr, label=f'{names[k]} (AUC={auc:.3f})')
                plotted = True
        if plotted:
            ax.plot([0, 1], [0, 1], '--', color='gray')
            ax.set(xlabel='False positive rate', ylabel='True positive rate', title='One-vs-rest ROC curves')
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / 'roc_curves.png', dpi=180)
        plt.close(fig)

    _, _, macro_f1, _ = precision_recall_fscore_support(yt, yp, labels=labels, average='macro', zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(yt, yp, labels=labels, average='weighted', zero_division=0)
    present_aucs = [a for a in aucs if np.isfinite(a)]
    results = {
        'model_name': model_name, 'n_samples': int(len(yt)),
        'accuracy': float(accuracy_score(yt, yp)),
        'balanced_accuracy': float(balanced_accuracy_score(yt, yp)),
        'macro_f1': float(macro_f1), 'weighted_f1': float(weighted_f1),
        'quadratic_weighted_kappa': float(cohen_kappa_score(yt, yp, labels=labels, weights='quadratic')),
        'mae': float(mean_absolute_error(yt, yp)),
        'macro_auroc_ovr': float(np.mean(present_aucs)) if len(present_aucs) == 5 else None,
    }
    with open(out / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2, allow_nan=False)
    return results


def plot_training_history(history, *, model_name='cnn', output_dir='metrics_output'):
    """Save PyTorch training/validation accuracy and loss curves.

    history is a nonempty list of per-epoch dictionaries from the shared
    PyTorch notebook, with epoch, train_accuracy, val_accuracy,
    train_loss, and val_loss.
    """
    if not isinstance(history, list) or not history:
        raise ValueError('history must be a nonempty list of per-epoch dictionaries.')
    keys = ('train_accuracy', 'val_accuracy', 'train_loss', 'val_loss')
    try:
        data = {k: np.asarray([row[k] for row in history], dtype=float) for k in keys}
        epochs = np.asarray([row['epoch'] for row in history], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Each epoch must contain epoch, train_accuracy, val_accuracy, train_loss, val_loss.') from exc
    if not np.all(np.isfinite(epochs)) or any(not np.all(np.isfinite(v)) for v in data.values()):
        raise ValueError('Epochs, accuracy and loss values must be finite.')
    out = Path(output_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)
    saved = {}
    for metric, ylabel in [('accuracy', 'Accuracy'), ('loss', 'Loss')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, data[f'train_{metric}'], label=f'Training {metric}')
        ax.plot(epochs, data[f'val_{metric}'], label=f'Validation {metric}')
        ax.set(xlabel='Epoch', ylabel=ylabel, title=f'{model_name}: training and validation {metric}')
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        filename = out / f'training_validation_{metric}.png'
        fig.savefig(filename, dpi=180)
        plt.close(fig)
        saved[metric] = str(filename)
    return saved
