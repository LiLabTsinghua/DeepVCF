"""DRN and TPN training functions for existing model instances."""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cf_data import CFEdgeDataLoader
from knowledge import KGEdgeDataLoader
from metric import (
    compute_binary_metrics_by_gene_group,
    compute_gene_rank_records,
    compute_typed_raw_mrr,
)


LOGGER = logging.getLogger(__name__)


# =============================================================================
# DRN Negative Sampling
# =============================================================================

def _sample_unfiltered_negative_edges(
    num_neg_samples,
    edge_index,
    edge_type,
    num_nodes,
):
    """Generate negatives by unfiltered uniform head/tail corruption.

    Args:
        num_neg_samples (int): Number of negatives per positive edge.
        edge_index (Tensor): Positive edge indices with shape ``[2, E]``.
        edge_type (Tensor): Relation types with shape ``[E]``.
        num_nodes (int): Total node count for uniform sampling.

    Returns:
        tuple[Tensor, Tensor]: Negative edges and their relation types.
    """
    edge_index = edge_index.repeat_interleave(num_neg_samples, dim=1)
    edge_type = edge_type.repeat_interleave(num_neg_samples)
    negative_edge_index = edge_index.clone()
    corrupt_head = torch.rand(
        negative_edge_index.size(1),
        device=edge_index.device,
    ) < 0.5
    candidates = torch.arange(num_nodes, device=edge_index.device)
    negative_edge_index[0, corrupt_head] = candidates[
        torch.randint(
            num_nodes,
            (int(corrupt_head.sum()),),
            device=edge_index.device,
        )
    ]
    negative_edge_index[1, ~corrupt_head] = candidates[
        torch.randint(
            num_nodes,
            (int((~corrupt_head).sum()),),
            device=edge_index.device,
        )
    ]
    return negative_edge_index, edge_type


# =============================================================================
# DRN Training
# =============================================================================

def train_drn(
    drn, drn_split, config, node_type_id, node_ids_by_prefix, fold_index=1,
):
    """Train an existing DRN on one KG training/validation split.

    Args:
        drn (DeepRepresentationNetwork): Instantiated DRN to optimize in place.
        drn_split (tuple): KG training graph and forward validation edges/types.
        config (dict): DRN optimization, negative-sampling, and early-stop settings.
        node_type_id (Tensor): Per-node prefix-group IDs for validation ranking.
        node_ids_by_prefix (dict): Candidate node IDs grouped by full prefix.
        fold_index (int): One-based unit number used in training logs.

    Each epoch trains on KG batches. Validation starts after epoch 10 at
    the configured interval; patience counts validation evaluations without
    improvement. The best DRN weights are restored after training.
    """
    kg_train_graph, valid_edge_index, valid_edge_type = drn_split
    device = next(drn.parameters()).device
    neg_strategy = config.get('drn_neg_sampling', 'adversarial')
    optimizer = torch.optim.AdamW(
        drn.parameters(),
        lr=config['drn_lr'],
        weight_decay=config['drn_wd'],
    )
    train_loader = KGEdgeDataLoader(
        kg_train_graph.kg_train_edge_index.to(device),
        kg_train_graph.kg_train_edge_type.to(device),
        config['drn_batch_size'],
    )
    valid_loader = KGEdgeDataLoader(
        valid_edge_index.to(device),
        valid_edge_type.to(device),
        128,
        shuffle=False,
    )
    prefixes = sorted(node_ids_by_prefix)
    prefix_to_id = {prefix: idx for idx, prefix in enumerate(prefixes)}

    LOGGER.info('DRN training started for fold #%d.', fold_index)
    best_mrr, patience_counter, best_state = 0, 0, None
    for epoch in range(1, config['drn_num_epochs'] + 1):
        batch_losses = []
        for batch_edges, batch_types in tqdm(train_loader):
            # Train on one KG batch: edges [2, B], relation indices [B].
            drn.train()
            optimizer.zero_grad()
            positive_scores = drn.kge_forward(batch_edges, batch_types)
            num_neg = config['drn_num_neg']
            negative_edges, negative_types = _sample_unfiltered_negative_edges(
                num_neg,
                batch_edges,
                batch_types,
                drn.num_nodes,
            )
            negative_scores = drn.kge_forward(
                negative_edges,
                negative_types,
            )
            negative_matrix = negative_scores.view(positive_scores.size(0), num_neg)
            margin = config.get('drn_margin', 1.0)
            if neg_strategy == 'adversarial':
                temperature = config.get('drn_adv_temperature', 1.0)
                weights = F.softmax(
                    negative_matrix * temperature,
                    dim=1,
                ).detach()
                loss = (
                    weights
                    * F.relu(
                        margin
                        - positive_scores.unsqueeze(1)
                        + negative_matrix
                    )
                ).sum(dim=1).mean()
            else:
                repeated_positive_scores = positive_scores.repeat_interleave(num_neg)
                loss = F.margin_ranking_loss(
                    repeated_positive_scores,
                    negative_scores,
                    torch.ones_like(repeated_positive_scores),
                    margin=margin,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drn.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
        LOGGER.info(
            'DRN epoch: %05d, loss: %.4f',
            epoch,
            np.mean(batch_losses),
        )

        if (
            epoch > 10
            and epoch % config['drn_eval_interval'] == 0
        ):
            # Rank validation heads/tails against nodes with the same prefix.
            mrr = compute_typed_raw_mrr(
                tqdm(valid_loader), drn.kge_forward, device,
                node_type_id, prefixes, prefix_to_id, node_ids_by_prefix,
            )
            if mrr > best_mrr:
                best_mrr = mrr
                patience_counter = 0
                best_state = {
                    key: value.clone()
                    for key, value in drn.state_dict().items()
                }
                LOGGER.info(
                    'Improved DRN validation typed raw MRR: %.4f.',
                    mrr,
                )
            else:
                patience_counter += 1
                if patience_counter >= config['drn_patience']:
                    LOGGER.info('DRN early stopping triggered.')
                    break

    if best_state is not None:
        drn.load_state_dict(best_state)
        LOGGER.info(
            'Best fold #%d DRN restored '
            '(validation typed raw MRR=%.4f).',
            fold_index, best_mrr,
        )


# =============================================================================
# TPN Training
# =============================================================================

def train_tpn(
    tpn,
    drn_embedding,
    data,
    config,
    candidate_pools,
    fixed_epochs=None,
    batch_size=None,
):
    """Train the TPN with configurable validation-metric early stopping.

    Args:
        tpn (TargetPredictionNetwork): Instantiated TPN to optimize in place.
        drn_embedding (Tensor): Normalized DRN node embeddings. Detached from
            the DRN during TPN training and stored in the TPN inference cache.
        data (Data): Training edges and labels, plus validation data when
            early stopping is enabled.
        config (dict): Hyperparameters using ``tpn_*`` keys.
        candidate_pools (dict): Current-species gene pools used to group
            validation samples for MRR or AUPR early stopping.
        fixed_epochs (int or None): If set, train exactly this many epochs
            and disable early stopping.
        batch_size (int or None): Optional batch size override. Defaults to
            ``config['tpn_batch_size']``; lets mechanistic pretraining use a
            larger batch than downstream fine-tuning.
    """
    loss_fn = config.get('tpn_loss', 'CE')
    early_stop_metric = config.get(
        'tpn_early_stop_metric', 'unfiltered_gene_mrr',
    )
    LOGGER.info('TPN training started.')

    device = next(tpn.parameters()).device
    tpn.kge_emb = drn_embedding.detach().to(device)
    data = data.to(device)

    optimizer = torch.optim.AdamW(
        tpn.parameters(),
        lr=config['tpn_lr'],
        weight_decay=config['tpn_wd'],
    )
    effective_batch_size = (
        batch_size if batch_size is not None else config['tpn_batch_size']
    )
    loader = CFEdgeDataLoader(
        data.train_edge_index,
        data.train_edge_type,
        data.train_label,
        effective_batch_size,
    )

    best_score, patience_counter, best_state = float('-inf'), 0, None

    num_epochs = (
        int(fixed_epochs)
        if fixed_epochs is not None
        else config['tpn_num_epochs']
    )
    use_early_stopping = fixed_epochs is None
    if fixed_epochs is not None:
        LOGGER.info(
            'Fixed-epoch TPN training enabled: %d epochs without early stopping.',
            num_epochs,
        )

    for epoch in range(1, num_epochs + 1):
        batch_losses = []
        for batch_edges, batch_types, batch_label in tqdm(loader):
            # Edges [2, B], relation indices [B], and binary labels [B].
            # BCE uses the queried-relation probability; CE maps positive
            # labels to relation heads and negative labels to background.
            tpn.train()
            optimizer.zero_grad()

            if loss_fn == 'BCE':
                out = tpn.predict_edge_probability(batch_edges, batch_types)
                loss = F.binary_cross_entropy(
                    out,
                    batch_label.to(out.device).unsqueeze(1),
                )
            else:  # CE
                # Positives → target = rel head index; negatives → target = background class
                logits = tpn.forward_logits(batch_edges)
                label = batch_label.to(logits.device).bool()
                # Vectorised relation→output-index lookup via LUT buffer.
                rel_indices = tpn.rel_to_output_idx[batch_types.to(logits.device)]
                bg_class = logits.size(1) - 1
                targets = torch.where(
                    label,
                    rel_indices,
                    torch.full_like(rel_indices, bg_class),
                )
                loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tpn.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
        LOGGER.info(
            'TPN epoch: %05d, loss: %.4f',
            epoch,
            np.mean(batch_losses),
        )

        # Fixed-epoch pretraining performs no validation or model selection.
        if not use_early_stopping:
            continue

        # Compute the configured early-stop score on this fold's validation data.
        tpn.eval()
        with torch.no_grad():
            if early_stop_metric == 'unfiltered_gene_mrr':
                # Keep the metabolite/relation fixed and rank positive genes
                # within their own group, without filtering known positives.
                positive_mask = data.valid_label.bool()
                records = compute_gene_rank_records(
                    data.valid_edge_index[:, positive_mask],
                    data.valid_edge_type[positive_mask],
                    lambda edges, types: tpn.predict_edge_probability(edges, types).squeeze(1),
                    candidate_pools['gem_genes'],
                    candidate_pools['non_gem_genes'],
                    full_pos_set=set(),
                    heterologous_gene_pool=candidate_pools.get('heterologous_genes'),
                )
                group_scores = []
                for group in ('gem', 'non_gem', 'heterologous'):
                    reciprocal_ranks = [
                        1.0 / record['gene_rank'] for record in records
                        if record['gene_group'] == group
                    ]
                    group_scores.append(
                        float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None
                    )
                available = [value for value in group_scores if value is not None]
                if not available:
                    raise ValueError(
                        'Unfiltered gene-MRR early stopping requires validation '
                        'positive edges.'
                    )
                score = float(np.mean(available))
                metric_label = 'unfiltered gene MRR'
            else:
                # Compute per-group AUPR from validation probabilities.
                valid_loader = KGEdgeDataLoader(
                    data.valid_edge_index, data.valid_edge_type, 128, shuffle=False,
                )
                predictions = torch.cat([
                    tpn.predict_edge_probability(edges, types) for edges, types in valid_loader
                ], dim=0)
                overall, gem, non_gem, heterologous = compute_binary_metrics_by_gene_group(
                    predictions, data.valid_label, data.valid_edge_index[0],
                    candidate_pools,
                )
                group_scores = [
                    metrics['aupr'] if metrics is not None else None
                    for metrics in (gem, non_gem, heterologous)
                ]
                available = [value for value in group_scores if value is not None]
                score = float(np.mean(available)) if available else overall['aupr']
                metric_label = 'mean-group AUPR'

        group_text = [f'{value:.4f}' if value is not None else 'n/a'
                      for value in group_scores]
        LOGGER.info(
            'TPN validation %s — GEM: %s, non-GEM: %s, heterologous: %s, mean: %.4f.',
            metric_label, *group_text, score,
        )
        if score > best_score:
            best_score = score
            patience_counter = 0
            best_state = {
                key: value.clone()
                for key, value in tpn.state_dict().items()
            }
            LOGGER.info(
                'Improved TPN validation %s: %.4f.',
                early_stop_metric,
                score,
            )
        else:
            patience_counter += 1
            if patience_counter >= config['tpn_patience']:
                LOGGER.info('TPN early stopping triggered.')
                break

    if best_state is not None:
        tpn.load_state_dict(best_state)
        LOGGER.info(
            'Best TPN restored (validation %s=%.4f).',
            early_stop_metric,
            best_score,
        )
