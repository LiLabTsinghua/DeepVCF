"""Shared embedding normalization, projections, and coordinate tables.

PCA uses two components. t-SNE and UMAP first apply PCA with up to 50
components; their parameters and random state are supplied by the figure
entry points. All projections are fitted independently per checkpoint.
"""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


PROJECTION_METHODS = ('pca', 'tsne', 'umap')
PROJECTION_LABELS = {'pca': 'PCA', 'tsne': 't-SNE', 'umap': 'UMAP'}


def load_kg_categories(kg_path: Path, species: str):
    """Collect metabolites and split species proteins by Catalyzes coverage."""
    prefix = f'{species}_Protein:'
    proteins, metabolites, covered = set(), set(), set()
    with Path(kg_path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 3:
                raise ValueError(f'Expected three fields at {kg_path}:{line_number}')
            head, relation, tail = fields
            for entity in (head, tail):
                if entity.startswith(prefix):
                    proteins.add(entity)
                elif entity.startswith('Metabolite:'):
                    metabolites.add(entity)
            if relation == 'Catalyzes' and head.startswith(prefix):
                covered.add(head)
    return {
        'metabolite': metabolites,
        'gem_covered_gene': covered,
        'gem_uncovered_gene': proteins - covered,
    }


def load_coordinate_groups(path, group_field, group_labels, category_labels,
                           projection_label):
    """Read exported coordinates with complete groups and valid point records."""
    group_keys = {label: key for key, label in group_labels.items()}
    category_keys = {label: key for key, label in category_labels.items()}
    grouped = {key: [] for key in group_labels}
    seen = set()
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {'projection', group_field, 'entity', 'category', 'x', 'y'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f'Missing coordinate columns in {path}')
        for line_number, row in enumerate(reader, 2):
            if (row['projection'] != projection_label
                    or row[group_field] not in group_keys
                    or row['category'] not in category_keys or not row['entity']):
                raise ValueError(f'Invalid coordinate labels at {path}:{line_number}')
            group = group_keys[row[group_field]]
            point_key = (group, row['entity'])
            if point_key in seen:
                raise ValueError(f'Duplicate entity at {path}:{line_number}')
            seen.add(point_key)
            coordinate = (float(row['x']), float(row['y']))
            if not np.all(np.isfinite(coordinate)):
                raise ValueError(f'Non-finite coordinate at {path}:{line_number}')
            grouped[group].append((row['entity'], category_keys[row['category']], coordinate))
    result = {}
    for group, rows in grouped.items():
        if not rows:
            raise ValueError(f'No saved coordinates for {group_labels[group]} in {path}')
        result[group] = (
            np.asarray([row[0] for row in rows], dtype=str),
            np.asarray([row[1] for row in rows], dtype=str),
            np.asarray([row[2] for row in rows], dtype=float),
        )
    return result

def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each entity vector after validating numeric content."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or not np.all(np.isfinite(embeddings)):
        raise ValueError("Embeddings must be a finite two-dimensional array.")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero-norm entity embedding encountered.")
    return embeddings / norms


def project_embeddings(
    embeddings: np.ndarray,
    method: str,
    random_state: int,
) -> np.ndarray:
    """Project normalized embeddings into two dimensions."""
    if method == "pca":
        projected = PCA(n_components=2).fit_transform(embeddings)
    else:
        pre_components = min(50, embeddings.shape[1], embeddings.shape[0] - 1)
        preprojected = PCA(n_components=pre_components).fit_transform(embeddings)
        if method == "tsne":
            projected = TSNE(
                n_components=2,
                perplexity=30,
                init="pca",
                learning_rate="auto",
                max_iter=1000,
                random_state=random_state,
            ).fit_transform(preprojected)
        elif method == "umap":
            from umap import UMAP

            projected = UMAP(
                n_components=2,
                n_neighbors=30,
                min_dist=0.1,
                metric="euclidean",
                random_state=random_state,
            ).fit_transform(preprojected)
        else:
            raise ValueError(f"Unsupported projection method: {method}")
    projected = np.asarray(projected, dtype=float)
    if projected.shape != (embeddings.shape[0], 2):
        raise ValueError(f"Unexpected {method} projection shape: {projected.shape}")
    if not np.all(np.isfinite(projected)):
        raise ValueError(f"Non-finite values in {method} projection.")
    return projected


