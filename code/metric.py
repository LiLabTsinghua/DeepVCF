"""Binary-classification and ranking metrics for DeepVCF."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_RANKING_KS = (1, 5, 10, 20, 30, 50, 100)


# =============================================================================
# Binary Classification Metrics
# =============================================================================

@torch.no_grad()
def compute_binary_metrics(predictions: torch.Tensor, labels: torch.Tensor):
    """Compute standard binary classification metrics.

    The ``aupr`` value is the trapezoidal area under the precision-recall
    curve.

    Args:
        predictions (torch.Tensor): Probabilities with shape ``[N]`` or
            ``[N, 1]``.
        labels (torch.Tensor): Binary labels with shape ``[N]`` or ``[N, 1]``.

    Returns:
        dict: Dictionary containing computed metrics.

    Raises:
        ValueError: If either input is empty, has an unsupported shape, contains
            non-finite values, has a different number of elements, or violates
            the probability/binary-label constraints; also if labels do not
            contain both classes.
    """
    def _as_vector(tensor, name):
        if tensor.ndim == 1:
            vector = tensor
        elif tensor.ndim == 2 and tensor.size(1) == 1:
            vector = tensor.squeeze(1)
        else:
            raise ValueError(f'{name} must have shape [N] or [N, 1].')
        if vector.numel() == 0:
            raise ValueError(f'{name} tensor is empty.')
        return vector

    predictions = _as_vector(predictions, 'predictions')
    labels = _as_vector(labels, 'labels')
    if predictions.numel() != labels.numel():
        raise ValueError(
            'predictions and labels must contain the same number of values.'
        )
    if not torch.isfinite(predictions).all():
        raise ValueError('predictions must contain only finite values.')
    if not torch.isfinite(labels).all():
        raise ValueError('labels must contain only finite values.')
    if not torch.all((predictions >= 0) & (predictions <= 1)):
        raise ValueError('predictions must be probabilities in [0, 1].')

    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('labels must contain only 0 and 1.')

    num_pos = (labels == 1).sum().item()
    num_neg = (labels == 0).sum().item()
    if num_pos == 0 or num_neg == 0:
        raise ValueError('labels must contain both positive and negative samples.')

    preds_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()
    binary_preds = (preds_np >= 0.5).astype(int)
    pr_precision, pr_recall, _ = precision_recall_curve(labels_np, preds_np)
    aupr = float(auc(pr_recall[::-1], pr_precision[::-1]))

    metrics = {
        'auroc': roc_auc_score(labels_np, preds_np),
        'aupr': aupr,
        'accuracy': accuracy_score(labels_np, binary_preds),
        'precision': precision_score(labels_np, binary_preds, zero_division=0),
        'recall': recall_score(labels_np, binary_preds, zero_division=0),
        'f1_score': f1_score(labels_np, binary_preds, zero_division=0),
        'mcc': matthews_corrcoef(labels_np, binary_preds),
    }

    return metrics


@torch.no_grad()
def compute_binary_metrics_by_gene_group(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    src_indices: torch.Tensor,
    candidate_pools: dict,
):
    """Compute overall, GEM, non-GEM, and heterologous binary metrics.

    Samples are grouped by membership in the supplied gene candidate pools.
    Empty groups have ``None`` metrics.

    Args:
        predictions (torch.Tensor): Model probabilities, shape ``[N]`` or ``[N, 1]``.
        labels (torch.Tensor): Binary labels, shape ``[N]`` or ``[N, 1]``.
        src_indices (torch.Tensor): Source-node index per sample, shape ``[N]``.
        candidate_pools (dict): GEM, non-GEM, and heterologous gene ID tensors.

    Returns:
        tuple: Overall, GEM, non-GEM, and heterologous metric dictionaries.
            A group dictionary is ``None`` when that group has no samples.

    Raises:
        ValueError: If the binary metric inputs are invalid.
    """
    overall = compute_binary_metrics(predictions, labels)
    group_metrics = []
    for group in ('gem_genes', 'non_gem_genes', 'heterologous_genes'):
        sample_mask = torch.isin(src_indices, candidate_pools[group].to(src_indices.device))
        group_metrics.append(
            compute_binary_metrics(predictions[sample_mask], labels[sample_mask])
            if sample_mask.any() else None
        )
    return overall, *group_metrics


# =============================================================================
# Ranking Helpers
# =============================================================================

def _validate_ranks(ranks):
    """Validate and normalize a rank sequence.

    Args:
        ranks (Iterable[int]): Rank values to validate.

    Returns:
        np.ndarray: One-dimensional positive integer ranks as floats.

    Raises:
        ValueError: If ranks is not a non-empty one-dimensional sequence of
            finite positive integers.
    """
    try:
        rank_values = list(ranks)
    except TypeError as exc:
        raise ValueError('ranks must be an iterable of positive integers.') from exc
    if not rank_values:
        raise ValueError('ranks must contain at least one positive integer.')
    if any(isinstance(rank, (bool, np.bool_)) for rank in rank_values):
        raise ValueError('ranks must contain only positive integers.')
    try:
        rank_array = np.asarray(rank_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('ranks must contain only positive integers.') from exc
    if rank_array.ndim != 1:
        raise ValueError('ranks must be one-dimensional.')
    if (
            not np.isfinite(rank_array).all()
            or np.any(rank_array <= 0)
            or np.any(rank_array != np.floor(rank_array))):
        raise ValueError('ranks must contain only finite positive integers.')
    return rank_array


def compute_mr(ranks: list):
    """Compute Mean Rank (MR).

    Args:
        ranks (list[int]): Positive integer ranks.

    Returns:
        float: Mean rank.

    Raises:
        ValueError: If ``ranks`` is invalid.
    """
    return float(np.mean(_validate_ranks(ranks)))


def compute_mrr(ranks: list):
    """Compute Mean Reciprocal Rank (MRR).

    Args:
        ranks (list[int]): Positive integer ranks.

    Returns:
        float: Mean reciprocal rank.

    Raises:
        ValueError: If ``ranks`` is invalid.
    """
    rank_array = _validate_ranks(ranks)
    return float(np.mean(1.0 / rank_array))


def compute_hit(ranks: list, k: int):
    """Compute Hits@K.

    Args:
        ranks (list[int]): Positive integer ranks.
        k (int): Positive rank threshold.

    Returns:
        float: Proportion of ranks <= k.

    Raises:
        ValueError: If ``ranks`` or ``k`` is invalid.
    """
    if (
            isinstance(k, (bool, np.bool_))
            or not isinstance(k, (int, np.integer))
            or int(k) <= 0):
        raise ValueError('k must be a positive integer.')
    rank_array = _validate_ranks(ranks)
    return float(np.mean(rank_array <= int(k)))


# =============================================================================
# DRN Typed Raw Ranking
# =============================================================================

@torch.no_grad()
def compute_typed_raw_mrr(
    edge_batches,
    score_fn,
    device,
    node_type_id: torch.Tensor,
    type_pool_keys: list,
    type_prefix_to_id: dict,
    node_ids_by_prefix: dict,
):
    """Compute typed raw MRR over head and tail replacements.

    Each endpoint is ranked against all nodes of the same entity type. Other
    known positive triples are intentionally not filtered because this metric
    is used for cost-efficient DRN early stopping.

    Args:
        edge_batches (Iterable): Batches of positive ``(edge_index,
            edge_type)`` tensors.
        score_fn (Callable): Function mapping edge indices and edge types to
            one DRN score per edge.
        device (torch.device): Device used for scoring.
        node_type_id (torch.Tensor): Per-node entity-type IDs.
        type_pool_keys (list[str]): Ordered entity-type prefixes.
        type_prefix_to_id (dict[str, int]): Entity-type prefix to type ID.
        node_ids_by_prefix (dict[str, torch.Tensor]): Candidate nodes per entity type.

    Returns:
        float: Head/tail-averaged typed raw MRR, or ``0.0`` when no validation
            edges are available.

    Raises:
        ValueError: If the node-type lookup is unavailable.
    """
    if node_type_id is None:
        raise ValueError(
            'compute_typed_raw_mrr requires a node-type lookup.'
        )

    all_tail_ranks = []
    all_head_ranks = []
    for batch_edges, batch_types in edge_batches:
        batch_edges = batch_edges.to(device)
        batch_types = batch_types.to(device)
        batch_size = batch_edges.size(1)
        positive_scores = score_fn(batch_edges, batch_types)

        def typed_ranks(fixed_index, corrupt_index):
            """Rank a batch after replacing one endpoint with typed nodes."""
            corrupt_types = node_type_id[batch_edges[corrupt_index]]
            ranks = torch.zeros(batch_size, device=device)
            for prefix in type_pool_keys:
                type_id = type_prefix_to_id[prefix]
                pool = node_ids_by_prefix[prefix].to(device)
                mask = corrupt_types == type_id
                if not mask.any():
                    continue
                group_size = mask.sum().item()
                pool_size = len(pool)
                fixed_nodes = batch_edges[fixed_index, mask].repeat_interleave(
                    pool_size,
                )
                candidate_nodes = pool.repeat(group_size)
                relation_types = batch_types[mask].repeat_interleave(pool_size)
                if corrupt_index == 1:
                    candidate_edges = torch.stack([fixed_nodes, candidate_nodes])
                else:
                    candidate_edges = torch.stack([candidate_nodes, fixed_nodes])
                candidate_scores = score_fn(
                    candidate_edges,
                    relation_types,
                ).view(group_size, pool_size)
                ranks[mask] = (
                    candidate_scores >= positive_scores[mask].unsqueeze(1)
                ).sum(dim=1).float()
            return ranks

        all_tail_ranks.extend(
            typed_ranks(fixed_index=0, corrupt_index=1).clamp(min=1).tolist()
        )
        all_head_ranks.extend(
            typed_ranks(fixed_index=1, corrupt_index=0).clamp(min=1).tolist()
        )

    if not all_tail_ranks:
        return 0.0
    return (compute_mrr(all_tail_ranks) + compute_mrr(all_head_ranks)) / 2


# =============================================================================
# Filtered Target-Gene Ranking
# =============================================================================

@torch.no_grad()
def compute_gene_rank_records(
    ei: torch.Tensor,
    et: torch.Tensor,
    score_fn,
    gem_gene_pool: torch.Tensor,
    non_gem_gene_pool: torch.Tensor,
    full_pos_set: set,
    task_rel: torch.Tensor = None,
    heterologous_gene_pool: torch.Tensor = None,
):
    """Compute filtered gene-rank records for positive edges in input order.

    Args:
        ei (torch.Tensor): Positive edge indices with shape ``[2, T]``.
        et (torch.Tensor): Positive edge types with shape ``[T]``.
        score_fn (Callable): Function mapping edge indices and edge types to
            one score per edge.
        gem_gene_pool (torch.Tensor): Candidate GEM-gene indices.
        non_gem_gene_pool (torch.Tensor): Candidate non-GEM-gene indices.
        full_pos_set (set[tuple[int, int, int]]): Caller-supplied known-positive
            ``(gene_idx, relation_idx, metabolite_idx)`` triples to filter.
        task_rel (torch.Tensor or None): Two task-relation indices. When
            provided, records also compare the positive-relation score with
            its flipped-relation score.
        heterologous_gene_pool (torch.Tensor or None): Independent heterologous
            candidates. Defaults to an empty group.

    Returns:
        list[dict]: One record per positive edge in input order, including its
            ``gene_rank``.

    Raises:
        ValueError: If edge shapes, task relations, or the
            positive-gene candidate-pool assignments are inconsistent.
    """
    if ei.ndim != 2 or ei.size(0) != 2:
        raise ValueError('ei must have shape [2, T].')
    if et.ndim != 1 or et.numel() != ei.size(1):
        raise ValueError('et must have one entry per edge.')

    device = ei.device
    task_rel = task_rel.to(device) if task_rel is not None else None
    if task_rel is not None:
        if task_rel.ndim != 1 or task_rel.numel() != 2:
            raise ValueError('task_rel must contain exactly two relation indices.')
        if torch.unique(task_rel).numel() != 2:
            raise ValueError('task_rel must contain two distinct relation indices.')
        if et.numel() and not torch.isin(et.to(device), task_rel).all():
            invalid_relations = sorted(
                set(et.detach().cpu().tolist())
                - set(task_rel.detach().cpu().tolist())
            )
            raise ValueError(
                f'et contains non-task relation indices: {invalid_relations}.'
            )

    gem_gene_set = set(gem_gene_pool.detach().cpu().tolist())
    non_gem_gene_set = set(non_gem_gene_pool.detach().cpu().tolist())
    if heterologous_gene_pool is None:
        heterologous_gene_pool = torch.empty(0, dtype=torch.long)
    heterologous_gene_set = set(heterologous_gene_pool.detach().cpu().tolist())
    if (gem_gene_set & non_gem_gene_set
            or gem_gene_set & heterologous_gene_set
            or non_gem_gene_set & heterologous_gene_set):
        raise ValueError('GEM, non-GEM, and heterologous candidate pools must be disjoint.')
    pools = {
        'gem': gem_gene_pool,
        'non_gem': non_gem_gene_pool,
        'heterologous': heterologous_gene_pool,
    }
    candidate_sets = {
        'gem': gem_gene_set,
        'non_gem': non_gem_gene_set,
        'heterologous': heterologous_gene_set,
    }

    def group_for_gene(gene_index):
        if gene_index in heterologous_gene_set:
            return 'heterologous'
        return 'gem' if gene_index in gem_gene_set else 'non_gem'

    invalid_gene_assignments = []
    for i in range(ei.size(1)):
        gene_index = int(ei[0, i])
        gene_group = group_for_gene(gene_index)
        candidate_set = candidate_sets[gene_group]
        if gene_index not in candidate_set:
            invalid_gene_assignments.append((gene_index, gene_group))
    if invalid_gene_assignments:
        examples = invalid_gene_assignments[:10]
        raise ValueError(
            f'{len(invalid_gene_assignments)} positive genes are absent from '
            f'their candidate pools; examples: {examples}.'
        )

    cache = {}
    records = []

    for i in range(ei.shape[1]):
        g_i = int(ei[0, i])
        m_i = int(ei[1, i])
        r_i = int(et[i])
        gene_group = group_for_gene(g_i)
        is_gem = gene_group == 'gem'
        cache_key = (r_i, m_i, gene_group)

        if cache_key not in cache:
            pool = pools[gene_group].to(device)
            pool_list = [int(g) for g in pool.tolist()]
            mets_t = torch.full(
                (len(pool_list),), m_i, dtype=torch.long, device=device,
            )
            rels_t = torch.full(
                (len(pool_list),), r_i, dtype=torch.long, device=device,
            )
            pool_scores = score_fn(torch.stack([pool, mets_t]), rels_t).reshape(-1)
            scores_list = [float(s) for s in pool_scores.tolist()]
            base_mask = [(g, r_i, m_i) not in full_pos_set for g in pool_list]
            cache[cache_key] = {
                'pool_list': pool_list,
                'scores_list': scores_list,
                'score_by_gene': dict(zip(pool_list, scores_list)),
                'base_mask': base_mask,
            }

        item = cache[cache_key]
        pos_score = item['score_by_gene'][g_i]
        flipped_score = None
        directed_win = None
        if task_rel is not None:
            r0, r1 = int(task_rel[0]), int(task_rel[1])
            flipped_rel = r1 if r_i == r0 else r0
            flipped_score = float(score_fn(
                torch.tensor([[g_i], [m_i]], dtype=torch.long, device=device),
                torch.tensor([flipped_rel], dtype=torch.long, device=device),
            ))
            directed_win = pos_score > flipped_score
        filt = [
            s for g, ok, s in zip(item['pool_list'], item['base_mask'], item['scores_list'])
            if ok or g == g_i
        ]
        records.append({
            'head_idx': g_i,
            'rel_idx': r_i,
            'tail_idx': m_i,
            'gene_rank': max(sum(1 for s in filt if s >= pos_score), 1),
            'is_gem_gene': is_gem,
            'gene_group': gene_group,
            'score': pos_score,
            'flipped_score': flipped_score,
            'directed_win': directed_win,
        })

    return records


def summarize_rank_records(records, target, ks, *, directed):
    """Summarize gene or metabolite ranks for validated thresholds in ``ks``.

    With ``directed=True``, directed Hits@K counts a hit only when the
    record's ``directed_win`` flag is true. Empty records return ``None``.
    """
    if not records:
        return None

    ranks = _validate_ranks(record[f'{target}_rank'] for record in records)
    result = {f'{target}_mrr': float(np.mean(1.0 / ranks))}
    if directed:
        directed_wins = np.asarray([record['directed_win'] for record in records], dtype=bool)

    for k in ks:
        hits = ranks <= k
        result[f'{target}_hit@{k}'] = float(np.mean(hits))
        if directed:
            result[f'{target}_directed_hit@{k}'] = float(np.mean(hits & directed_wins))
    return result


@torch.no_grad()
def compute_filtered_target_gene_ranking(
    ei: torch.Tensor,
    et: torch.Tensor,
    score_fn,
    gem_gene_pool: torch.Tensor,
    non_gem_gene_pool: torch.Tensor,
    full_pos_set: set,
    ks: tuple = DEFAULT_RANKING_KS,
    task_rel: torch.Tensor = None,
    return_records: bool = False,
    heterologous_gene_pool: torch.Tensor = None,
):
    """Compute filtered target-gene MRR and Hit@k for positive edges.

    Filters only the caller-supplied ``full_pos_set`` from candidate rankings.
    Gene ranking groups by relation, metabolite, and gene group. The
    metabolite is fixed and only the candidate gene is replaced.

    Args:
        ei (torch.Tensor): Positive edge indices with shape ``[2, T]``.
        et (torch.Tensor): Positive edge types with shape ``[T]``.
        score_fn (Callable): Function mapping edge indices and edge types to
            one score per edge.
        gem_gene_pool (torch.Tensor): Candidate GEM-gene indices.
        non_gem_gene_pool (torch.Tensor): Candidate non-GEM-gene indices.
        full_pos_set (set[tuple[int, int, int]]): Known-positive
            ``(gene_idx, relation_idx, metabolite_idx)`` triples to filter.
        ks (tuple[int, ...]): Top-k values for Hit@k.
        task_rel (torch.Tensor or None): Two task-relation indices. When
            provided, directed hits additionally require the positive-relation
            score to exceed the flipped-relation score.
        return_records (bool): Whether to return raw gene-rank records.
        heterologous_gene_pool (torch.Tensor or None): Independent heterologous
            candidates. Defaults to an empty group.

    Returns:
        dict or tuple: Gene-ranking metrics, optionally paired with raw
            gene-rank records.

    Raises:
        ValueError: If ``ks`` is empty or contains duplicate/non-positive/
            non-integer values, no positive edges are available, or candidate
            pools and task relations are inconsistent with the positive edges.
    """
    try:
        ks = tuple(ks)
    except TypeError as exc:
        raise ValueError('ks must be an iterable of positive integers.') from exc
    if not ks:
        raise ValueError('ks must contain at least one positive integer.')
    if any(
            isinstance(k, (bool, np.bool_))
            or not isinstance(k, (int, np.integer))
            or int(k) <= 0
            for k in ks):
        raise ValueError('ks must contain only positive integers.')
    ks = tuple(int(k) for k in ks)
    if len(set(ks)) != len(ks):
        raise ValueError('ks must not contain duplicate values.')

    gene_rank_records = compute_gene_rank_records(
        ei, et, score_fn, gem_gene_pool, non_gem_gene_pool,
        full_pos_set,
        task_rel=task_rel,
        heterologous_gene_pool=heterologous_gene_pool,
    )
    if not gene_rank_records:
        raise ValueError('Target-gene ranking requires at least one positive edge.')

    result = summarize_rank_records(
        gene_rank_records, 'gene', ks, directed=task_rel is not None,
    )
    if return_records:
        return result, gene_rank_records
    return result
