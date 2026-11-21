"""Cell-factory data loading, indexing, splitting, and batching."""

import logging
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from knowledge import load_index_mappings


LOGGER = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

CF_COLUMNS = ('h', 'r', 't', 'label')
TRIPLE_COLUMNS = ('h', 'r', 't')


# =============================================================================
# Shared CF Data Processing
# =============================================================================

class _BaseCFDataProcessor:
    """Shared file loading and indexing for labeled cell-factory data."""

    def __init__(self, index_root):
        """Initialize shared CF data state.

        Args:
            index_root: Directory containing KG index mappings.
        """
        self.index_root = Path(index_root)
        self.node_to_id = {}
        self.relation_to_id = {}

    @staticmethod
    def _read_labeled_file(path):
        """Read a four-column, tab-separated CF file without a header.

        Args:
            path: Input file containing head, relation, tail, and binary label.

        Returns:
            DataFrame with ``h``, ``r``, ``t``, and integer ``label`` columns.
        """
        data = pd.read_csv(path, sep='\t', header=None, dtype=str, keep_default_na=False)
        data.columns = CF_COLUMNS
        for column in CF_COLUMNS:
            data[column] = data[column].str.strip()
        data['label'] = pd.to_numeric(data['label']).astype(np.int64)
        return data

    @classmethod
    def _read_labeled_files(cls, paths):
        """Read and merge one or more labeled CF files, removing duplicate rows.

        Args:
            paths: One path or a list/tuple of paths.

        Returns:
            DataFrame containing the combined input rows in their original order.
        """
        if isinstance(paths, (str, PathLike)):
            paths = [paths]
        return pd.concat(
            [cls._read_labeled_file(path) for path in paths], ignore_index=True,
        ).drop_duplicates().reset_index(drop=True)

    def _load_index_mappings(self):
        """Load the node and relation indices saved by KG processing."""
        self.node_to_id, self.relation_to_id = load_index_mappings(self.index_root)

    def _coverage_mask(self, data):
        """Select triples represented by the KG mappings.

        Args:
            data: Labeled CF dataframe.

        Returns:
            Boolean mask aligned with the dataframe rows.
        """
        return (
            data['h'].isin(self.node_to_id)
            & data['t'].isin(self.node_to_id)
            & data['r'].isin(self.relation_to_id)
        )

    def _filter_to_coverage(self, data, source_name):
        """Remove training rows not represented by the KG mappings.

        Args:
            data: Labeled training dataframe.
            source_name: Human-readable source label used for logging.

        Returns:
            The coverage-filtered dataframe with a reset index.
        """
        mask = self._coverage_mask(data)
        num_dropped = int((~mask).sum())
        if num_dropped:
            LOGGER.warning(
                'Dropped %d/%d rows from %s because they are outside KG coverage.',
                num_dropped,
                len(data),
                source_name,
            )
        return data.loc[mask].reset_index(drop=True)

    def _index_cf_data(self, data):
        """Map CF triples to integer edge tensors.

        Args:
            data: Labeled CF dataframe using the saved KG vocabulary.

        Returns:
            A tuple containing ``edge_index`` and ``edge_type`` tensors.
        """
        edge_index = torch.from_numpy(
            np.vstack([
                data['h'].map(self.node_to_id).to_numpy(dtype=np.int64),
                data['t'].map(self.node_to_id).to_numpy(dtype=np.int64),
            ])
        ).long()
        edge_type = torch.from_numpy(
            data['r'].map(self.relation_to_id).to_numpy(dtype=np.int64)
        ).long()
        return edge_index, edge_type


# =============================================================================
# Task Training and Validation Data
# =============================================================================

class CFTrainDataProcessor(_BaseCFDataProcessor):
    """Construct train/validation folds for the cell-factory task."""

    def __init__(self, train_path, index_root, ensemble=True, k=5, seed=42,
                 filter_train_missing=False, always_train_path=None,
                 drop_always_train_overlap=False):
        """Initialize the task training-data processor.

        ``always_train_path`` is intended for adaptation data that must be in
        every training fold. By default it is required to be disjoint from
        ``train_path``. When ``drop_always_train_overlap`` is enabled,
        overlapping triples are removed from ``always_train_path`` instead.
        The random K-fold splits are used for early stopping and ensemble
        construction, not as held-out test sets.

        Args:
            train_path: Path or paths used for train/validation splitting.
            index_root: Directory containing KG index mappings.
            ensemble: Whether to construct K folds instead of one split.
            k: Number of folds used when ``ensemble`` is true.
            seed: Random seed used by the splitters.
            filter_train_missing: Whether to remove training rows outside KG coverage.
            always_train_path: Optional path or paths included in every training fold.
            drop_always_train_overlap: Whether to remove rows from
                ``always_train_path`` when their ``(h, r, t)`` triple is already
                present in ``train_path``.
        """
        super().__init__(index_root)
        self.train_path = train_path
        self.ensemble = ensemble
        self.k = k
        self.seed = seed
        self.filter_train_missing = filter_train_missing
        self.always_train_path = always_train_path
        self.drop_always_train_overlap = drop_always_train_overlap

    def _load_always_train_data(self):
        """Load rows that must remain in every training fold.

        Returns:
            Loaded always-train rows, or an empty dataframe when unset.
        """
        if self.always_train_path is None:
            return pd.DataFrame(columns=CF_COLUMNS)
        return self._read_labeled_files(self.always_train_path)

    @staticmethod
    def _validate_disjoint_sources(train, always_train):
        """Require support triples to be separate from the split-eligible data.

        Args:
            train: Rows used for train/validation splitting.
            always_train: Support rows intended for every training fold.

        Raises:
            ValueError: If a support triple also appears in the split data.
        """
        if CFTrainDataProcessor._always_train_overlap(train, always_train).any():
            raise ValueError('always_train_path overlaps train_path; use separate support triples.')

    @staticmethod
    def _always_train_overlap(train, always_train):
        """Return a mask selecting support triples already present in split data."""
        train_triples = pd.MultiIndex.from_frame(train[list(TRIPLE_COLUMNS)])
        support_triples = pd.MultiIndex.from_frame(always_train[list(TRIPLE_COLUMNS)])
        return support_triples.isin(train_triples)

    @staticmethod
    def _drop_always_train_overlap(train, always_train):
        """Remove support triples that already appear in the split-eligible data.

        Args:
            train: Rows used for train/validation splitting.
            always_train: Support rows intended for every training fold.

        Returns:
            Support rows with overlapping triples removed.
        """
        mask = CFTrainDataProcessor._always_train_overlap(train, always_train)
        if mask.any():
            LOGGER.info('Dropped %d overlapping support rows.', int(mask.sum()))
        return always_train.loc[~mask].reset_index(drop=True)

    def _split_train_valid(self, train):
        """Create random K-fold splits, or one 80/20 train/validation split.

        Args:
            train: DataFrame of rows eligible for splitting.

        Returns:
            List of ``(train_fold, valid_fold)`` DataFrame pairs.
        """
        if self.ensemble:
            splitter = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)
            return [
                (train.iloc[train_idx], train.iloc[valid_idx])
                for train_idx, valid_idx in splitter.split(train)
            ]
        return [train_test_split(train, test_size=0.2, random_state=self.seed)]

    def _prepare_training_data(self):
        """Load main/support rows and apply the requested overlap and KG filters.

        Returns:
            Pair of DataFrames: split-eligible rows and always-train support rows.
        """
        train = self._read_labeled_files(self.train_path)
        always_train = self._load_always_train_data()
        if self.drop_always_train_overlap:
            always_train = self._drop_always_train_overlap(train, always_train)
        else:
            self._validate_disjoint_sources(train, always_train)
        self._load_index_mappings()
        if self.filter_train_missing:
            train = self._filter_to_coverage(train, 'train_path')
            always_train = self._filter_to_coverage(always_train, 'always_train_path')
        return train, always_train

    def process_full(self):
        """Build one indexed dataset from every training row.

        Unlike ``process()``, this method does not create a validation split.
        It is intended for estimators such as sklearn baselines that train on
        the complete dataset without validation-based early stopping.

        Returns:
            ``(full_train_data, full_train_df)``. The Data object contains
            ``train_edge_index``, ``train_edge_type``, and ``train_label``;
            the dataframe retains the corresponding source rows.
        """
        train, always_train = self._prepare_training_data()
        if not always_train.empty:
            train = pd.concat([train, always_train], ignore_index=True)
        train = train.reset_index(drop=True)
        train_edge_index, train_edge_type = self._index_cf_data(train)
        LOGGER.info('Complete CF training rows: %d.', len(train_edge_type))
        return Data(
            train_edge_index=train_edge_index,
            train_edge_type=train_edge_type,
            train_label=torch.tensor(
                train['label'].to_numpy(),
                dtype=torch.float32,
            ),
        ), train

    def process(self):
        """Build indexed folds and retain source rows for auditing/export.

        Returns:
            ``(cf_train_folds, fold_dfs)``. ``cf_train_folds`` contains PyG Data
            objects used by the model. ``fold_dfs`` preserves the corresponding
            labeled rows for traceability and downstream result reporting.
        """
        train, always_train = self._prepare_training_data()

        cf_train_folds = []
        fold_dfs = []
        for fold_id, (train_fold, valid_fold) in enumerate(
            self._split_train_valid(train),
            start=1,
        ):
            if not always_train.empty:
                train_fold = pd.concat(
                    [train_fold, always_train],
                    ignore_index=True,
                )
            train_edge_index, train_edge_type = self._index_cf_data(train_fold)
            valid_edge_index, valid_edge_type = self._index_cf_data(valid_fold)
            LOGGER.info(
                'CF fold %d: train=%d, valid=%d.',
                fold_id,
                len(train_edge_type),
                len(valid_edge_type),
            )
            cf_train_folds.append(Data(
                train_edge_index=train_edge_index,
                train_edge_type=train_edge_type,
                train_label=torch.tensor(
                    train_fold['label'].to_numpy(),
                    dtype=torch.float32,
                ),
                valid_edge_index=valid_edge_index,
                valid_edge_type=valid_edge_type,
                valid_label=torch.tensor(
                    valid_fold['label'].to_numpy(),
                    dtype=torch.float32,
                ),
            ))
            fold_dfs.append((
                train_fold.reset_index(drop=True),
                valid_fold.reset_index(drop=True),
            ))

        return cf_train_folds, fold_dfs


# =============================================================================
# Labeled Data Without Fold Splitting
# =============================================================================

class CFDataProcessor(_BaseCFDataProcessor):
    """Load labeled CF data with existing KG indices, without creating folds.

    Used for fixed-epoch pretraining and independent test sets. Mechanistic
    pretraining data may come from FSEOF applied to genome-scale metabolic
    models (GEMs), or another method using the same labeled CF schema.
    """

    def __init__(self, data_path, index_root, split):
        """Initialize the labeled-data processor.

        Args:
            data_path: Path or paths containing four-column labeled CF rows.
            index_root: Directory containing KG index mappings.
            split: Output field prefix, ``'train'`` for pretraining or
                ``'test'`` for independent evaluation.
        """
        super().__init__(index_root)
        self.data_path = data_path
        self.split = split

    def process(self):
        """Read and index all labeled rows without splitting.

        Returns:
            A PyG Data object with ``<split>_edge_index``, ``<split>_edge_type``,
            and ``<split>_label`` tensors.
        """
        data = self._read_labeled_files(self.data_path)
        self._load_index_mappings()
        edge_index, edge_type = self._index_cf_data(data)
        LOGGER.info('CF %s rows: %d.', self.split, len(edge_type))
        return Data(**{
            f'{self.split}_edge_index': edge_index,
            f'{self.split}_edge_type': edge_type,
            f'{self.split}_label': torch.tensor(
                data['label'].to_numpy(), dtype=torch.float32,
            ),
        })


# =============================================================================
# Labeled Edge Batching
# =============================================================================

class CFEdgeDataset(Dataset):
    """Dataset of indexed, labeled cell-factory edges."""

    def __init__(self, edge_index, edge_type, label):
        """Initialize the labeled-edge dataset.

        Args:
            edge_index: Head/tail indices with shape ``[2, num_edges]``.
            edge_type: Relation indices with shape ``[num_edges]``.
            label: Binary labels with shape ``[num_edges]``.
        """
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.label = label
        self.num_edges = edge_index.size(1)

    def __len__(self):
        """Return the total number of edges.

        Returns:
            Number of indexed CF edges.
        """
        return self.num_edges

    def __getitem__(self, idx):
        """Return one indexed, labeled CF edge.

        Args:
            idx: Dataset row index.

        Returns:
            A ``(head_tail_indices, relation_id, label)`` tuple.
        """
        return self.edge_index[:, idx], self.edge_type[idx], self.label[idx]


class CFEdgeDataLoader(DataLoader):
    """Batch labeled edges while preserving the ``[2, batch_size]`` layout."""

    def __init__(self, edge_index, edge_type, label, batch_size, shuffle=True):
        """Initialize the labeled-edge DataLoader.

        Args:
            edge_index: Head/tail indices with shape ``[2, num_edges]``.
            edge_type: Relation indices with shape ``[num_edges]``.
            label: Binary labels with shape ``[num_edges]``.
            batch_size: Number of edges per batch.
            shuffle: Whether to shuffle edges before each epoch.
        """
        dataset = CFEdgeDataset(edge_index, edge_type, label)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        """Stack labeled edge samples into batch tensors.

        Args:
            batch: List of samples returned by ``CFEdgeDataset``.

        Returns:
            A tuple containing batched edges, relation indices, and labels.
        """
        batch_edges = torch.stack([item[0] for item in batch], dim=1)
        batch_edge_types = torch.stack([item[1] for item in batch])
        batch_labels = torch.stack([item[2] for item in batch])
        return batch_edges, batch_edge_types, batch_labels
