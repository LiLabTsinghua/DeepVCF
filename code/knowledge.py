"""Knowledge-graph loading, indexing, splitting, and batching."""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

KG_COLUMNS = ('h', 'r', 't')
TASK_RELATIONS = ('knock out', 'overexpress')
INVERSE_RELATION_SUFFIX = '_inverse'


def index_mapping_paths(index_root):
    """Return the node and relation mapping paths under index_mappings/."""
    root = Path(index_root) / 'index_mappings'
    return root / 'node_to_id.npy', root / 'relation_to_id.npy'


def load_index_mappings(index_root):
    """Load the node and relation dictionaries from one mapping directory.

    Args:
        index_root: Directory containing saved KG index mappings.

    Returns:
        tuple[dict, dict]: Node-to-index and relation-to-index dictionaries.

    Raises:
        FileNotFoundError: If either mapping file is missing.
    """
    node_path, relation_path = index_mapping_paths(index_root)
    return (
        np.load(node_path, allow_pickle=True).item(),
        np.load(relation_path, allow_pickle=True).item(),
    )


# =============================================================================
# Knowledge Graph Processing
# =============================================================================

class KGProcessor:
    """Build a deterministic knowledge graph for DeepVCF.

    A single-species KG is the primary use case. Multiple species KGs can be
    supplied as an optional extension. Identifiers are retained as supplied;
    equal identifiers are shared across species in a multi-species graph.
    """

    def __init__(self, kg_path, index_root, no_inverse_relations=None, seed=42):
        """Initialize the knowledge-graph processor.

        Args:
            kg_path (dict): Species-tagged KG paths as ``{species: path}``.
                Usually contains one species; multiple entries enable the
                optional multi-species graph construction.
            index_root (str or Path): Directory containing index mappings.
            no_inverse_relations (set): Relations that must not receive inverse edges.
            seed (int): Random seed for reproducibility.
        """
        self.kg_path = kg_path
        self.index_root = Path(index_root)
        self.no_inverse_relations = set(no_inverse_relations or ())
        self.seed = seed
        self.node_to_id = {}
        self.relation_to_id = {}

    @staticmethod
    def _read_species_kg(species, path):
        """Read and validate one species KG.

        Args:
            species: Species namespace assigned to the input file.
            path: Path to a tab-separated KG file.

        Returns:
            A deduplicated dataframe with ``h``, ``r``, and ``t`` columns.

        Raises:
            ValueError: If the file is empty, malformed, or has empty values.
        """
        kg = pd.read_csv(path, sep='\t', header=None, dtype=str, keep_default_na=False)
        if kg.shape[1] != 3 or kg.eq('').any().any():
            raise ValueError(f'KG for {species} must contain three nonempty columns: {path}')
        kg.columns = KG_COLUMNS
        return kg.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def _node_groups_from_kg(kg):
        """Build disjoint endogenous GEM, non-GEM, and heterologous groups.

        ``gem_genes`` contains genes represented in the genome-scale metabolic
        model, identified by their ``Catalyzes`` edges. Heterologous proteins
        have their own group regardless of their Catalyzes edges. GEM-group
        genes are metabolic genes. ``non_gem_genes`` contains genes not represented in
        the GEM; it consists mainly of non-metabolic genes, while potentially
        including metabolic genes that have not yet been incorporated into the
        GEM.

        Args:
            kg: DataFrame with h, r, t columns.

        Returns:
            dict: Coverage sets for metabolites and genes.
        """
        entities = set(kg['h']) | set(kg['t'])
        # Match the supplied TYPE:id or SPECIES_TYPE:id node names directly.
        metabolites = {
            node for node in entities
            if (node.startswith('Metabolite:')
                or node.split(':', 1)[0].endswith('_Metabolite'))
            and node.endswith('_c')
        }
        genes = {
            node for node in entities
            if node.startswith('Protein:')
            or node.split(':', 1)[0].endswith(('_Protein', '_Protein_heterologous'))
        }
        heterologous = {
            node for node in genes
            if node.split(':', 1)[0].endswith('_Protein_heterologous')
        }
        endogenous = genes - heterologous
        gem = set(kg.loc[kg['r'] == 'Catalyzes', 'h']) & endogenous
        return {
            'cytosolic_metabolites': metabolites,
            'all_genes': genes,
            'gem_genes': gem,
            'non_gem_genes': endogenous - gem,
            'heterologous_genes': heterologous,
        }

    def _load_knowledge_graphs(self):
        """Read and merge the configured species KGs.

        With multiple inputs, equal identifiers are shared across species
        exactly as supplied by the files.

        Returns:
            tuple: ``(kg, node_groups_by_species)``. The merged KG defines the
                training triples and entity/relation vocabulary. Each supplied
                species has its own node-group dictionary.
        """
        frames, groups = [], {}
        for species, path in sorted(self.kg_path.items()):
            kg = self._read_species_kg(species, path)
            frames.append(kg)
            groups[species] = self._node_groups_from_kg(kg)
            LOGGER.info('KG %s: %d triples.', species, len(kg))
        kg = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
        return kg, groups

    def _load_or_create_index_mappings(self, kg):
        """Load or create deterministic entity and relation mappings.

        Existing mappings must match the current KG vocabulary and inverse-relation
        configuration. Saved index assignments are retained when loading a cache.

        Args:
            kg: Merged raw KG dataframe defining the entity/relation vocabulary.

        Returns:
            LongTensor: Task-relation indices in ``TASK_RELATIONS`` order.

        Raises:
            ValueError: If relations or cached mappings are inconsistent.
            FileNotFoundError: If either cached mapping file is missing.
        """
        entities = sorted(set(kg['h']) | set(kg['t']))
        relations = sorted(set(kg['r']))
        unknown = self.no_inverse_relations - set(relations)
        if unknown:
            raise ValueError(f'Unknown no-inverse relations: {sorted(unknown)}')
        relations += [
            relation + INVERSE_RELATION_SUFFIX
            for relation in relations if relation not in self.no_inverse_relations
        ] + list(TASK_RELATIONS)
        if len(relations) != len(set(relations)):
            raise ValueError('KG relations collide with inverse or task relation names.')

        node_path, relation_path = index_mapping_paths(self.index_root)
        if node_path.exists() or relation_path.exists():
            self.node_to_id, self.relation_to_id = load_index_mappings(self.index_root)
            self._validate_index_mapping(self.node_to_id, entities, 'node_to_id')
            self._validate_index_mapping(self.relation_to_id, relations, 'relation_to_id')
        else:
            self.node_to_id = {name: idx for idx, name in enumerate(entities)}
            self.relation_to_id = {name: idx for idx, name in enumerate(relations)}
            node_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(node_path, self.node_to_id)
            np.save(relation_path, self.relation_to_id)
        return torch.tensor([self.relation_to_id[name] for name in TASK_RELATIONS])

    @staticmethod
    def _validate_index_mapping(mapping, names, label):
        """Ensure a cached mapping exactly represents the current KG.

        Args:
            mapping: Cached name-to-index dictionary.
            names: Names required by the current graph.
            label: Mapping label used in error messages.

        Raises:
            ValueError: If the mapping has wrong names or invalid indices.
        """
        if (not isinstance(mapping, dict) or set(mapping) != set(names)
                or set(mapping.values()) != set(range(len(names)))):
            raise ValueError(f'Cached {label} does not match the KG; use a new index_root.')

    def _index_kg(self, kg):
        """Map raw triples to integer edge tensors.

        Args:
            kg: Dataframe with ``h``, ``r``, and ``t`` columns.

        Returns:
            tuple[LongTensor, LongTensor]: Edge indices with shape ``[2, E]``
                and relation indices with shape ``[E]``.
        """
        edge_index = torch.tensor(np.vstack([
            kg['h'].map(self.node_to_id).to_numpy(dtype=np.int64),
            kg['t'].map(self.node_to_id).to_numpy(dtype=np.int64),
        ]), dtype=torch.long)
        edge_type = torch.tensor(kg['r'].map(self.relation_to_id).to_numpy(dtype=np.int64))
        return edge_index, edge_type

    def _index_kg_with_inverses(self, kg):
        """Index training triples and add inverse edges for eligible relations.

        Args:
            kg: Raw KG dataframe whose entities and relations are mapped.

        Returns:
            A tuple containing ``edge_index`` and ``edge_type`` tensors.
        """
        # Map forward edges once, then reuse their node indices for inverse edges.
        edge_index, edge_type = self._index_kg(kg)
        # Add inverse edges only for the configured relations.
        mask = ~kg['r'].isin(self.no_inverse_relations)
        inverse_types = torch.tensor(
            (kg.loc[mask, 'r'] + INVERSE_RELATION_SUFFIX)
            .map(self.relation_to_id).to_numpy(dtype=np.int64),
        )
        return (
            torch.cat([edge_index, edge_index.flip(0)[:, mask.to_numpy()]], dim=1),
            torch.cat([edge_type, inverse_types]),
        )

    def _build_drn_split(self, kg, task_rel, seed):
        """Build one DRN training/validation split from a raw KG.

        The split is performed before inverse edges are generated. Only
        training triples are expanded with generated inverse edges.

        Args:
            kg (DataFrame): Raw KG triples before inverse edges are added.
            task_rel (LongTensor): Indexed task-relation identifiers.
            seed (int): Random seed used for the KG split.

        Returns:
            tuple: ``(kg_train_graph, drn_valid_edge_index,
                drn_valid_edge_type)`` suitable for ``DeepVCFModel.train_drn``.

        Raises:
            ValueError: If the KG is too small.
        """
        if len(kg) < 2:
            raise ValueError('At least two KG triples are required for a DRN split.')
        valid_size = min(max(1, round(len(kg) * 0.01)), len(kg) - 1)
        valid = kg.sample(n=valid_size, random_state=seed)
        train_edges, train_types = self._index_kg_with_inverses(kg.drop(valid.index))
        valid_edges, valid_types = self._index_kg(valid)
        train_graph = Data(
            kg_train_edge_index=train_edges,
            kg_train_edge_type=train_types,
            num_nodes=len(self.node_to_id),
            num_edge_type=len(self.relation_to_id),
            task_rel=task_rel,
        )
        return train_graph, valid_edges, valid_types

    def process(self, num_drn_folds=None):
        """Main pipeline for background knowledge processing.

        Args:
            num_drn_folds (int or None): Number of independent FoldUnit DRN
                training/validation splits to build. None skips splitting.

        Returns:
            tuple: ``(knowledge_graph, kg_metadata)`` or, when requested,
                ``(knowledge_graph, kg_metadata, drn_splits)``.

        Raises:
            ValueError: If KG content, mappings, or split settings are invalid.
            FileNotFoundError: If an input KG or partial mapping cache is missing.
        """
        kg, groups = self._load_knowledge_graphs()
        task_rel = self._load_or_create_index_mappings(kg)
        # Entity names follow "TYPE:id" or "SPECIES_TYPE:id". Keep their
        # complete prefix for DRN validation ranking. With multiple KGs,
        # equal identifiers remain shared across species.
        prefix_ids = defaultdict(list)
        for node, idx in self.node_to_id.items():
            prefix_ids[node.split(':', 1)[0]].append(idx)
        kg_metadata = {
            # Node names grouped by species: cytosolic metabolites, all genes,
            # GEM genes, non-GEM genes, and heterologous genes. Used to select
            # named candidates for prediction and label their gene groups.
            'node_groups_by_species': groups,
            # Integer node IDs for the same species/group sets. Used as
            # candidate tensors for TPN ranking and gene-group evaluation.
            'node_group_ids_by_species': {
                species: {
                    group: torch.tensor(sorted(self.node_to_id[node] for node in nodes),
                                        dtype=torch.long)
                    for group, nodes in species_groups.items()
                }
                for species, species_groups in groups.items()
            },
            # Task-relation names in order: knock out and overexpress.
            # Used to label the modification associated with each prediction.
            'modification': list(TASK_RELATIONS),
            # Node IDs grouped by the full identifier prefix, e.g. ECO_Protein.
            # DRN validation replaces each endpoint with nodes from its prefix
            # group when computing typed raw MRR.
            'node_ids_by_prefix': {
                prefix: torch.tensor(sorted(ids), dtype=torch.long)
                for prefix, ids in prefix_ids.items()
            },
            # Node name -> integer ID. Preserves training-time indices in
            # checkpoints and maps prediction queries to model node indices.
            'node_to_id': dict(self.node_to_id),
            # Relation name -> integer ID. Preserves training-time relation
            # indices and supports checking mappings when loading checkpoints.
            'relation_to_id': dict(self.relation_to_id),
        }
        # Package graph metadata into a PyG Data object.
        graph = Data(
            num_nodes=len(self.node_to_id),
            num_edge_type=len(self.relation_to_id),
            task_rel=task_rel,
        )
        if num_drn_folds is None:
            return graph, kg_metadata
        if num_drn_folds <= 0:
            raise ValueError('num_drn_folds must be positive.')
        splits = [
            self._build_drn_split(kg, task_rel, self.seed + fold)
            for fold in range(num_drn_folds)
        ]
        return graph, kg_metadata, splits


# =============================================================================
# Unlabeled Edge Batching
# =============================================================================

class KGEdgeDataset(Dataset):
    """Dataset of unlabeled knowledge-graph edges."""

    def __init__(self, edge_index, edge_type):
        """Initialize the dataset.

        Args:
            edge_index (LongTensor): [2, E] tensor where row 0 is head and row 1 is tail.
            edge_type (LongTensor): [E] tensor representing relation IDs.

        Raises:
            ValueError: If tensor shapes are inconsistent.
        """
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError('edge_index must have shape [2, num_edges].')
        if edge_type.ndim != 1 or edge_type.numel() != edge_index.size(1):
            raise ValueError('edge_type must have shape [num_edges].')
        self.edge_index = edge_index
        self.edge_type = edge_type

    def __len__(self):
        """Return the total number of edges.

        Returns:
            Number of indexed KG edges.
        """
        return self.edge_index.size(1)

    def __getitem__(self, idx):
        """Return one indexed KG edge.

        Args:
            idx: Dataset row index.

        Returns:
            A ``(head_tail_indices, relation_id)`` tuple.
        """
        return self.edge_index[:, idx], self.edge_type[idx]


class KGEdgeDataLoader(DataLoader):
    """DataLoader that preserves the ``[2, batch_size]`` edge layout."""

    def __init__(self, edge_index, edge_type, batch_size, shuffle=True):
        """Initialize the DataLoader for unlabeled edges.

        Args:
            edge_index (LongTensor): Shape [2, E], head and tail node indices.
            edge_type (LongTensor): Shape [E], relation type for each edge.
            batch_size (int): Number of edges per batch.
            shuffle (bool): Whether to shuffle edges each epoch.
        """
        super().__init__(
            KGEdgeDataset(edge_index, edge_type),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        """Stack edge samples into batch tensors.

        Args:
            batch (list): List of tuples from ``KGEdgeDataset.__getitem__``.

        Returns:
            batch_edges (LongTensor): [2, BatchSize]
            batch_edge_types (LongTensor): [BatchSize]
        """
        edges, types = zip(*batch)
        return torch.stack(edges, dim=1), torch.stack(types)
