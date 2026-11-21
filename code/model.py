"""DeepVCF model orchestration, evaluation, checkpointing, and inference."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ModuleList

from evaluate import (
    evaluate_deepvcf_binary,
    evaluate_deepvcf_ranking,
)
from knowledge import KGEdgeDataLoader
from metric import DEFAULT_RANKING_KS
from networks import DeepRepresentationNetwork, TargetPredictionNetwork
from train import train_drn, train_tpn


LOGGER = logging.getLogger(__name__)

# =============================================================================
# DeepVCF Model Orchestration
# =============================================================================


class DeepVCFModel(torch.nn.Module):
    """Create and manage the ensemble's DRNs and TPNs.

    Each fold contains one independent DRN and TPN. This class constructs
    these networks and coordinates training, evaluation, and prediction.
    """

    def __init__(
        self,
        config,
        num_nodes: int,
        num_relations: int,
        task_rel: Tensor,
        kg_metadata: dict,
        species: str,
    ):
        """Initialize the DeepVCF fold ensemble.

        Args:
            config (dict): Model configuration, including network dimensions
                and ensemble settings.
            num_nodes (int): Total number of entities in the knowledge graph.
            num_relations (int): Total number of knowledge-graph relations.
            task_rel (Tensor): Relation types specific to the prediction task.
            kg_metadata (dict): Knowledge-graph node groups and index mappings.
            species (str): Species used for TPN candidate selection. Must be
                supplied explicitly and present in the KG metadata.

        """

        super().__init__()
        self.config = config
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.task_rel = task_rel
        self.kg_metadata = kg_metadata
        self.species = species
        # config.get() keeps DRN-only scripts from supplying TPN defaults.
        k = config.get('k', 5) if config.get('ensemble', True) else 1
        self.folds = ModuleList()
        for _ in range(k):
            drn = DeepRepresentationNetwork(
                num_nodes, num_relations, config.get('hidden_dim', 300),
            )
            tpn = TargetPredictionNetwork(
                drn.emb_dim, config.get('hidden_channels', 300),
                task_rel=task_rel, dropout_prob=config.get('dropout', 0.1),
            )
            self.folds.append(torch.nn.ModuleDict({'drn': drn, 'tpn': tpn}))

    # =========================================================================
    # Knowledge-Graph Metadata
    # =========================================================================

    def _resolve_species(self, species=None):
        """Return node groups and candidate pools for one KG species.

        Args:
            species (str or None): Requested species; ``None`` uses ``self.species``.

        Returns:
            tuple: Species node-group dictionary and candidate-pool dictionary.

        """
        selected_species = self.species if species is None else species
        return (
            self.kg_metadata['node_groups_by_species'][selected_species],
            self.kg_metadata['node_group_ids_by_species'][selected_species],
        )

    def set_species(self, species: str):
        """Switch the active species used for TPN operations.

        The model can be trained with one species' CF data and evaluated with a
        different species in a transfer experiment. DRN parameters and entity
        indices are unchanged; candidate pools come from the selected species.

        Args:
            species (str): Species key present in ``kg_metadata`` (for example ``ECO``
                or a target species name).

        """
        self.species = species
        LOGGER.info('Active KG species switched to %s.', species)

    # =========================================================================
    # Embedding Management
    # =========================================================================

    def get_drn_embeddings(self, fold_index=0):
        """Return normalized, detached embeddings from one fold DRN.

        This accessor defaults to fold 0 for single-model consumers such
        as sklearn baselines. DeepVCF TPNs use their own unit's DRN directly.

        Args:
            fold_index (int): fold index.

        Returns:
            Tensor: Node embeddings with shape ``[num_nodes, hidden_dim]``.
        """
        return F.normalize(
            self.folds[fold_index]['drn'].node_emb.detach(), p=2, dim=1,
        )

    def _refresh_tpn_embeddings(self):
        """Refresh every TPN from its fold's own DRN embedding."""
        for fold_index, fold in enumerate(self.folds):
            fold['tpn'].kge_emb = self.get_drn_embeddings(fold_index)

    # =========================================================================
    # DRN Training
    # =========================================================================

    def train_drn(self, drn_splits):
        """Train every fold DRN with an independent KG holdout split.

        Args:
            drn_splits (list[tuple]): One KG training graph and forward-only
                validation edge/type pair per fold.

        Raises:
            ValueError: If split count or negative-sampling configuration is invalid.
        """
        if len(drn_splits) != len(self.folds):
            raise ValueError(
                f'drn_splits length {len(drn_splits)} does not match number '
                f'of folds {len(self.folds)}.'
            )
        neg_strategy = self.config.get('drn_neg_sampling', 'adversarial')
        if neg_strategy not in {'adversarial', 'uniform'}:
            raise ValueError(
                "drn_neg_sampling must be 'adversarial' or 'uniform'; "
                f'got {neg_strategy!r}.'
            )
        # Complete-prefix candidates for DRN validation ranking.
        node_ids_by_prefix = {
            prefix: ids.to(self.device)
            for prefix, ids in self.kg_metadata['node_ids_by_prefix'].items()
        }
        node_type_id = torch.zeros(
            self.num_nodes, dtype=torch.long, device=self.device,
        )
        for type_id, prefix in enumerate(sorted(node_ids_by_prefix)):
            node_type_id[node_ids_by_prefix[prefix]] = type_id
        for fold_index, (fold, drn_split) in enumerate(
            zip(self.folds, drn_splits),
            start=1,
        ):
            train_drn(
                fold['drn'], drn_split, self.config, node_type_id,
                node_ids_by_prefix, fold_index=fold_index,
            )
        self._refresh_tpn_embeddings()

    # =========================================================================
    # TPN Training
    # =========================================================================

    def reset_tpn_cls_layer(self):
        """Reset each fold TPN classification layer independently."""
        for fold in self.folds:
            fold['tpn'].reset_cls_layer()
        LOGGER.info(
            'Reset every fold TPN classification layer after mechanistic '
            'pretraining.'
        )

    def train_tpn(self, train_folds, fixed_epochs=None, batch_size=None):
        """Train each fold TPN using its own frozen DRN embedding.

        Args:
            train_folds (list[Data]): Training data aligned with ``self.folds``.
                Pretraining can reuse the same data object for every fold.
            fixed_epochs (int or None): Train for a fixed number of epochs
                without validation or early stopping; used for pretraining.
                Otherwise use validation early stopping for each fold.
            batch_size (int or None): Optional training batch size override.

        Raises:
            ValueError: If the number of data objects differs from the number
                of model folds.
        """
        if len(train_folds) != len(self.folds):
            raise ValueError(
                f'train_folds length {len(train_folds)} does not match '
                f'number of folds {len(self.folds)}.'
            )
        LOGGER.info('Target prediction network training started.')
        _, candidate_pools = self._resolve_species()
        for fold_index, (fold, fold_data) in enumerate(
            zip(self.folds, train_folds),
        ):
            LOGGER.info('Training TPN for fold #%d.', fold_index + 1)
            train_tpn(
                fold['tpn'], self.get_drn_embeddings(fold_index),
                fold_data.to(self.device), self.config,
                candidate_pools=candidate_pools,
                fixed_epochs=fixed_epochs, batch_size=batch_size,
            )

    # =========================================================================
    # DeepVCF Evaluation
    # =========================================================================

    def evaluate_deepvcf(
        self,
        species,
        test_data,
        labels,
        test_type,
        filter_pos_set=None,
        ks=DEFAULT_RANKING_KS,
        return_rank_records=False,
    ):
        """Evaluate the DeepVCF folds and ensemble for a requested species.

        Args:
            species (str): Species whose gene candidate pools are used.
            test_data (Data): Indexed test edges and relations.
            labels (Tensor): Binary labels aligned with the test edges.
            test_type (str): ``binary`` or ``ranking``.
            filter_pos_set (set or None): Known positive triples supplied for
                ranking, including an empty set when no filtering is desired.
            ks (sequence[int]): Hit@k thresholds for ranking.
            return_rank_records (bool): Include per-positive ranking records.

        Returns:
            dict: Species, test type, per-fold results, and ensemble results.
                Both evaluation methods report the three gene groups separately.
                Ranking reports both gene and cytosolic-metabolite corruption.
        """
        _, candidate_pools = self._resolve_species(species)
        tpns = [fold['tpn'] for fold in self.folds]
        if test_type == 'binary':
            result = evaluate_deepvcf_binary(tpns, test_data, labels, candidate_pools)
        elif test_type == 'ranking':
            result = evaluate_deepvcf_ranking(
                tpns, test_data, labels, candidate_pools, self.task_rel,
                filter_pos_set, ks=ks, return_rank_records=return_rank_records,
            )
        else:
            raise ValueError("test_type must be 'binary' or 'ranking'.")
        return {'species': species, 'test_type': test_type, **result}

    # =========================================================================
    # Checkpoint Input and Output
    # =========================================================================

    def save_model(self, path: str = None):
        """Save full model state to disk.

        Persists config, graph metadata, and all fold-specific DRN/TPN states.
        TPN embedding caches are derived from their corresponding DRNs.

        Args:
            path (str or Path or None): Optional explicit destination file
                path. When omitted, falls back to
                ``<config['model_saved_dir']>/<config['model_name']>.pt``.
        """
        if path is not None:
            save_path = Path(path)
        else:
            save_dir = Path(self.config.get('model_saved_dir', './'))
            model_name = self.config.get('model_name', 'DeepVCF')
            save_path = save_dir / f'{model_name}.pt'
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'config': self.config,
            'num_nodes': self.num_nodes,
            'num_relations': self.num_relations,
            'task_rel': self.task_rel.cpu(),
            'kg_metadata': self.kg_metadata,
            'species': self.species,
            'folds_state_dict': self.folds.state_dict(),
        }
        torch.save(save_dict, save_path)
        LOGGER.info('Model saved to %s.', save_path)

    @staticmethod
    def load_model(path, device='cpu'):
        """Load DeepVCF model from a checkpoint file.

        Args:
            path (str or Path): Path to the ``.pt`` checkpoint file saved by
                ``save_model()``.
            device (str): Target device, e.g. 'cpu' or 'cuda'.

        Returns:
            DeepVCFModel: Restored model on the specified device.
        """
        path = Path(path)
        ckpt = torch.load(path, map_location=device)
        model = DeepVCFModel(
            config=ckpt['config'],
            num_nodes=ckpt['num_nodes'],
            num_relations=ckpt['num_relations'],
            task_rel=ckpt['task_rel'],
            kg_metadata=ckpt['kg_metadata'],
            species=ckpt['species'],
        ).to(device)
        model.folds.load_state_dict(ckpt['folds_state_dict'], strict=True)
        model._refresh_tpn_embeddings()
        LOGGER.info('Model loaded from %s.', path)
        return model

    # =========================================================================
    # Genome-Scale Inference
    # =========================================================================

    def _build_exhaustive_inference_edges(
        self,
        dst_index,
        task_rel,
        eval_range,
    ):
        """Build exhaustive triples for genome-scale inference.

        Holds each destination metabolite and task relation fixed while
        iterating over all head candidates.

        Args:
            dst_index (Tensor): One-dimensional metabolite node indices.
            task_rel (Tensor): One-dimensional task-relation indices.
            eval_range (Tensor): One-dimensional candidate gene/protein node
                indices.

        Returns:
            Tuple (all_edge_index, edge_type) of shape
            (2, num_dst * num_eval * num_task_rel) and (same,).
        """
        # dst is metabolite
        num_dst = len(dst_index)
        num_valid_nodes = len(eval_range)
        num_task_rel = len(task_rel)

        # Pair every query metabolite and task relation with every candidate gene.
        all_edge_src_index = eval_range.repeat(
            num_dst * num_task_rel,
        ).to(dst_index.device)
        all_edge_dst_index = dst_index.repeat_interleave(
            num_valid_nodes * num_task_rel,
        )
        all_edge_index = [all_edge_src_index, all_edge_dst_index]
        all_edge_index = torch.stack(all_edge_index, dim=0)

        # Expand edge_type to match the expanded edge_index
        edge_type = task_rel.repeat_interleave(num_valid_nodes).repeat(
            num_dst,
        ).to(dst_index.device)
        return all_edge_index, edge_type

    @torch.no_grad()
    def genome_scale_prediction(
        self,
        species,
        query_met,
        gene_group,
        output_dir=None,
    ):
        """Rank all candidate genes against each query metabolite.

        Generates all ``(gene, task_rel, metabolite)`` triples, runs ensemble
        TPN inference, and optionally saves per-metabolite rankings.

        Args:
            species (str): Species whose candidate gene pool is used.
            query_met (list[str]): Cytosolic metabolite names to query as tails.
            gene_group (str): Gene pool to use: ``'all'``, ``'gem'``,
                ``'non_gem'``, or ``'heterologous'``.
            output_dir (str or Path or None): Optional directory for result files.

        Returns:
            dict[str, pd.DataFrame]: Prediction tables keyed by query metabolite.
        """
        node_to_id = self.kg_metadata['node_to_id']
        species_groups, candidate_pools = self._resolve_species(species)

        # Query metabolites are tails; candidate genes come from the species pool.
        query_met_tensor = torch.tensor(
            [node_to_id[met] for met in query_met],
            dtype=torch.long, device=self.device,
        )
        gene_tensor = candidate_pools[f'{gene_group}_genes'].to(self.device)
        id_to_node = {index: node for node, index in node_to_id.items()}
        gene = [id_to_node[index] for index in gene_tensor.tolist()]

        all_edge_index, all_edge_type = self._build_exhaustive_inference_edges(
            dst_index=query_met_tensor,
            task_rel=self.task_rel,
            eval_range=gene_tensor
        )

        edge_loader = KGEdgeDataLoader(
            all_edge_index, all_edge_type, 10000, shuffle=False,
        )

        indiv_outputs = []

        for fold in self.folds:
            fold['tpn'].eval()
            model_pred = torch.cat(
                [fold['tpn'].predict_edge_probability(e, t).squeeze(1) for e, t in edge_loader], dim=0
            )
            indiv_outputs.append(model_pred)

        indiv_stack = torch.stack(indiv_outputs, dim=0)
        ensemble_mean = indiv_stack.mean(dim=0).cpu().numpy()
        ensemble_var = indiv_stack.var(dim=0, unbiased=False).cpu().numpy()
        indiv_np = indiv_stack.cpu().numpy()

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        num_mod = len(self.kg_metadata['modification'])
        num_candidate = len(gene) * num_mod

        gene_short = [g.split(':')[1] if ':' in g else g for g in gene]
        gene_group_col = [
            'heterologous' if g in species_groups['heterologous_genes']
            else 'GEM' if g in species_groups['gem_genes'] else 'non-GEM'
            for g in gene
        ]

        predictions = {}
        for idx, met in enumerate(query_met):
            sl = slice(idx * num_candidate, (idx + 1) * num_candidate)
            df = pd.DataFrame({
                'query_metabolite': met,
                'gene': gene_short * num_mod,
                'gene_group': gene_group_col * num_mod,
                'modification': [m for m in self.kg_metadata['modification']
                                 for _ in range(len(gene))],
                'deepvcf_score': ensemble_mean[sl],
                'ensemble_variance': ensemble_var[sl],
                # Per-fold raw scores so the user can inspect ensemble disagreement.
                **{f'fold_{f + 1}_score': indiv_np[f, sl]
                   for f in range(indiv_np.shape[0])},
            })
            df = df.sort_values(by='deepvcf_score', ascending=False)
            predictions[met] = df
            if output_dir is not None:
                df.to_csv(output_dir / f'{met}.tsv', sep='\t', index=False)
        return predictions

    @property
    def device(self):
        """Return the device of the model's parameters.

        Returns:
            torch.device: Device containing the model parameters.
        """
        return next(self.parameters()).device
