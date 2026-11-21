"""Grouped DeepVCF binary and ranking evaluation for each fold and ensemble."""

import torch

from knowledge import KGEdgeDataLoader
from metric import (
    DEFAULT_RANKING_KS,
    compute_binary_metrics,
    compute_filtered_target_gene_ranking,
    summarize_rank_records,
)


@torch.no_grad()
def evaluate_deepvcf_binary(tpns, test_data, labels, candidate_pools, batch_size=128):
    """Evaluate binary classification separately for the three gene groups.

    Args:
        tpns (list): The trained TPN from each DeepVCF fold.
        test_data (Data): Indexed ``test_edge_index`` and ``test_edge_type``.
        labels (Tensor): Binary labels aligned with the test edges.
        candidate_pools (dict): Gene IDs in ``gem_genes``, ``non_gem_genes``,
            and ``heterologous_genes`` for the requested species.
        batch_size (int): Number of test edges scored per batch.

    Returns:
        dict: ``fold_results`` and ``ensemble``, each containing binary metrics
            for GEM, non-GEM, and heterologous genes, plus overall metrics.
            Empty groups have ``None`` metrics. Raw predictions and labels
            follow GEM/non-GEM/heterologous group order.
    """
    device = next(tpns[0].parameters()).device
    edge_index = test_data.test_edge_index.to(device)
    edge_type = test_data.test_edge_type.to(device)
    labels = labels.reshape(-1).to(device)
    for tpn in tpns:
        tpn.eval()

    fold_results = [
        {'fold': index + 1, 'binary': {}} for index in range(len(tpns))
    ]
    ensemble = {'binary': {}}
    predictions_by_fold = [[] for _ in tpns]
    ensemble_predictions, grouped_labels = [], []

    # Split the test set by gene membership before scoring each group.
    for group, pool_name in (
        ('gem', 'gem_genes'),
        ('non_gem', 'non_gem_genes'),
        ('heterologous', 'heterologous_genes'),
    ):
        mask = torch.isin(edge_index[0], candidate_pools[pool_name].to(device))
        if not mask.any():
            for result in fold_results:
                result['binary'][group] = None
            ensemble['binary'][group] = None
            continue

        group_labels = labels[mask]
        grouped_labels.append(group_labels)
        loader = KGEdgeDataLoader(
            edge_index[:, mask], edge_type[mask], batch_size, shuffle=False,
        )
        group_predictions = []
        for index, tpn in enumerate(tpns):
            predictions = torch.cat([
                tpn.predict_edge_probability(edges, types) for edges, types in loader
            ], dim=0)
            predictions_by_fold[index].append(predictions)
            group_predictions.append(predictions)
            fold_results[index]['binary'][group] = compute_binary_metrics(
                predictions, group_labels,
            )

        # Ensemble metrics are computed from mean probabilities.
        mean_predictions = torch.stack(group_predictions).mean(dim=0)
        ensemble_predictions.append(mean_predictions)
        ensemble['binary'][group] = compute_binary_metrics(mean_predictions, group_labels)

    all_labels = torch.cat(grouped_labels)
    for result, predictions in zip(fold_results, predictions_by_fold):
        predictions = torch.cat(predictions)
        result['binary'].update({
            'predictions': predictions,
            'labels': all_labels,
            'overall': compute_binary_metrics(predictions, all_labels),
        })
    mean_predictions = torch.cat(ensemble_predictions)
    ensemble['binary'].update({
        'predictions': mean_predictions,
        'labels': all_labels,
        'overall': compute_binary_metrics(mean_predictions, all_labels),
    })
    return {'fold_results': fold_results, 'ensemble': ensemble}


@torch.no_grad()
def evaluate_deepvcf_ranking(
    tpns,
    test_data,
    labels,
    candidate_pools,
    task_rel,
    filter_pos_set,
    ks=DEFAULT_RANKING_KS,
    return_rank_records=False,
):
    """Evaluate filtered gene and metabolite ranking for each gene group.

    Args:
        tpns (list): The trained TPN from each DeepVCF fold.
        test_data (Data): Indexed ``test_edge_index`` and ``test_edge_type``.
        labels (Tensor): Binary labels aligned with the test edges. Only
            positive test edges enter ranking evaluation.
        candidate_pools (dict): GEM, non-GEM, and heterologous candidate gene IDs,
            plus same-species cytosolic metabolite IDs.
        task_rel (Tensor): The two task-relation indices.
        filter_pos_set (set): Known positive ``(gene, relation, metabolite)``
            triples to remove from candidate rankings, retaining the query itself.
        ks (sequence[int]): Thresholds used for Hit@k and directed Hit@k.
        return_rank_records (bool): Include the per-positive rank records.

    Returns:
        dict: ``fold_results`` and ``ensemble``. Each reports gene and metabolite
            ranking metrics overall and by gene group. Groups with no positive
            test edges have ``None`` metrics. Optional records contain both ranks
            and follow GEM/non-GEM/heterologous group order.
    """
    device = next(tpns[0].parameters()).device
    edge_index = test_data.test_edge_index.to(device)
    edge_type = test_data.test_edge_type.to(device)
    labels = labels.reshape(-1).to(device)
    task_rel = task_rel.to(device)
    ks = tuple(ks)
    metabolite_pool = candidate_pools['cytosolic_metabolites'].to(device)
    metabolite_ids = metabolite_pool.tolist()
    for tpn in tpns:
        tpn.eval()

    # Select each group's positive queries before evaluating any model.
    group_queries = {}
    for group, pool_name in (
        ('gem', 'gem_genes'),
        ('non_gem', 'non_gem_genes'),
        ('heterologous', 'heterologous_genes'),
    ):
        mask = labels.bool() & torch.isin(
            edge_index[0], candidate_pools[pool_name].to(device),
        )
        group_queries[group] = (edge_index[:, mask], edge_type[mask])

    fold_results = []
    ensemble = None
    for index in range(len(tpns) + 1):
        if index < len(tpns):
            tpn = tpns[index]
            score_fn = lambda edges, types: tpn.predict_edge_probability(
                edges, types,
            ).squeeze(1)
        else:
            # Average candidate probabilities before computing ensemble ranks.
            score_fn = lambda edges, types: torch.stack([
                tpn.predict_edge_probability(edges, types).squeeze(1) for tpn in tpns
            ]).mean(dim=0)

        group_metrics, metabolite_group_metrics, all_records = {}, {}, []
        metabolite_score_cache = {}
        for group, (query_edges, query_types) in group_queries.items():
            if query_types.numel() == 0:
                group_metrics[group] = None
                metabolite_group_metrics[group] = None
                continue
            metrics, records = compute_filtered_target_gene_ranking(
                query_edges, query_types, score_fn,
                candidate_pools['gem_genes'], candidate_pools['non_gem_genes'],
                filter_pos_set, ks=ks, task_rel=task_rel, return_records=True,
                heterologous_gene_pool=candidate_pools['heterologous_genes'],
            )
            group_metrics[group] = metrics

            # Replace only the metabolite, using this species' cytosolic pool.
            for record in records:
                gene = record['head_idx']
                relation = record['rel_idx']
                metabolite = record['tail_idx']
                cache_key = (gene, relation)
                if cache_key not in metabolite_score_cache:
                    candidate_edges = torch.stack([
                        torch.full_like(metabolite_pool, gene), metabolite_pool,
                    ])
                    candidate_types = torch.full_like(metabolite_pool, relation)
                    metabolite_score_cache[cache_key] = score_fn(
                        candidate_edges, candidate_types,
                    ).reshape(-1)
                scores = metabolite_score_cache[cache_key]
                positive_score = scores[metabolite_pool == metabolite].item()
                # Filter other known positives while retaining the query itself.
                keep = torch.tensor([
                    candidate == metabolite
                    or (gene, relation, candidate) not in filter_pos_set
                    for candidate in metabolite_ids
                ], dtype=torch.bool, device=device)
                record['metabolite_rank'] = int(
                    ((scores >= positive_score) & keep).sum().item()
                )

            metabolite_group_metrics[group] = summarize_rank_records(
                records, 'metabolite', ks, directed=True,
            )
            all_records.extend(records)

        # Overall metrics summarize all positive queries, while each group
        # keeps its own metrics above. Ranks always use that group's candidates.
        result = {
            'target_gene_ranking': summarize_rank_records(
                all_records, 'gene', ks, directed=True,
            ),
            'target_gene_ranking_by_gene_group': group_metrics,
            'target_metabolite_ranking': summarize_rank_records(
                all_records, 'metabolite', ks, directed=True,
            ),
            'target_metabolite_ranking_by_gene_group': metabolite_group_metrics,
            'target_gene_rank_records': all_records if return_rank_records else None,
        }
        if index < len(tpns):
            fold_results.append({'fold': index + 1, **result})
        else:
            ensemble = result

    return {'fold_results': fold_results, 'ensemble': ensemble}
