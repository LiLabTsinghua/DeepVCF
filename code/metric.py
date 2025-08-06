import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


@torch.no_grad()
def compute_binary_metrics(predictions: torch.Tensor, labels: torch.Tensor):
    """
    Compute standard binary classification metrics including AUROC, AUPRC, Accuracy,
    Precision, Recall, F1-Score, and MCC (Matthews Correlation Coefficient).

    Args:
        predictions (torch.Tensor): Model output logits or probabilities of shape (N,).
        labels (torch.Tensor): Ground truth binary labels (0 or 1) of shape (N,).

    Returns:
        dict: Dictionary containing computed metrics.
    """
    # Apply sigmoid if predictions are not in [0, 1]
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)

    # Ensure labels contain only 0s and 1s
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must only contain 0 and 1.")

    # Check that both positive and negative samples exist
    num_pos = (labels == 1).sum().item()
    num_neg = (labels == 0).sum().item()
    if num_pos == 0 or num_neg == 0:
        raise ValueError("Labels must contain both positive and negative samples.")

    # Move to CPU and convert to numpy for scikit-learn compatibility
    preds_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Binarize predictions using threshold 0.5
    binary_preds = (preds_np >= 0.5).astype(int)

    # Calculate metrics
    metrics = {
        "auroc": roc_auc_score(labels_np, preds_np),
        "auprc": average_precision_score(labels_np, preds_np),
        "accuracy": accuracy_score(labels_np, binary_preds),
        "precision": precision_score(labels_np, binary_preds, zero_division=0),
        "recall": recall_score(labels_np, binary_preds, zero_division=0),
        "f1_score": f1_score(labels_np, binary_preds, zero_division=0),
        "mcc": matthews_corrcoef(labels_np, binary_preds),  # 新增 MCC 指标
    }

    return metrics


@torch.no_grad()
def compute_rank(num: int, all_out: torch.Tensor, task_rel: torch.Tensor, filter_mask=None):
    """
    Compute ranks for positive samples among all candidates based on prediction scores.

    Args:
        num (int): Number of positive samples (batch size).
        all_out (torch.Tensor): Predicted scores for all candidate edges, shape (num * num_candidates,).
        task_rel (torch.Tensor): Relation indices corresponding to the task relations.
        filter_mask (tuple, optional): Tuple of masks for filtering valid candidates.

    Returns:
        tuple: List of ranks and list of top-100 indices per sample.
    """
    # Reshape output to (num, num_candidates)
    all_out_reshaped = all_out.view(num, -1)

    top_mod = []
    ranks = []

    for i in range(num):
        x_out = all_out_reshaped[i]

        # Get top 100 indices
        top_mod.append(torch.sort(x_out, descending=True)[1][:100].cpu().tolist())

        # Extract positive and negative scores using filter mask
        pos_scores = torch.masked_select(x_out, filter_mask[0][i].bool())
        neg_scores = torch.masked_select(x_out, filter_mask[1][i].bool())

        for pos_score in pos_scores:
            # Count how many negatives have score >= pos_score to determine rank
            rank = (neg_scores >= pos_score).sum().item() + 1  # 1-based ranking
            ranks.append(rank)

    # Compute intersection size between top-100 lists
    inter_size = []
    for i in range(len(top_mod)):
        for j in range(i + 1, len(top_mod)):
            inter_size.append(len(set(top_mod[i]).intersection(set(top_mod[j]))))
    print(f"Top-100 intersection size: {np.mean(inter_size):.2f}")

    return ranks, top_mod


def compute_mr(ranks: list):
    """
    Compute Mean Rank (MR).

    Args:
        ranks (list): List of ranks for positive samples.

    Returns:
        float: Mean rank.
    """
    return np.mean(ranks)


def compute_mrr(ranks: list):
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        ranks (list): List of ranks for positive samples.

    Returns:
        float: Mean reciprocal rank.
    """
    reciprocal_ranks = [1.0 / r for r in ranks]
    return np.mean(reciprocal_ranks)


def compute_hit(ranks: list, k: int):
    """
    Compute Hits@K metric.

    Args:
        ranks (list): List of ranks for positive samples.
        k (int): Threshold rank for hits.

    Returns:
        float: Proportion of ranks <= k.
    """
    hit_count = sum(1 for r in ranks if r <= k)
    return hit_count / len(ranks)


def compute_ranking_metrics(ranks: list):
    """
    Compute multiple ranking metrics including MR, MRR, and Hits@K (for K=10, 50, 100).

    Args:
        ranks (list): List of ranks for positive samples.

    Returns:
        dict: Dictionary containing 'mr', 'mrr', 'Hits@10', 'Hits@50', 'Hits@100'.
    """
    metrics = {
        "mr": compute_mr(ranks),
        "mrr": compute_mrr(ranks),
        "Hits@10": compute_hit(ranks, 10),
        "Hits@50": compute_hit(ranks, 50),
        "Hits@100": compute_hit(ranks, 100),
    }
    return metrics