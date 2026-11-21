"""Serialization helpers for per-positive gene and metabolite rank records."""

from __future__ import annotations

import csv
from pathlib import Path


RANK_RECORD_COLUMNS = (
    'model',
    'seed',
    'test_set',
    'gene_group',
    'corruption',
    'positive_index',
    'gene',
    'relation',
    'metabolite',
    'gene_rank',
    'metabolite_rank',
    'score',
    'flipped_score',
    'directed_win',
)


def _inverse_mapping(mapping):
    """Build an integer-index to entity-name mapping.

    Args:
        mapping: Entity/relation mapping with names as keys and integer indices
            as values.

    Returns:
        dict[int, str]: Reverse mapping.

    Raises:
        ValueError: If the mapping contains duplicate indices.
    """
    inverse = {}
    for name, index in mapping.items():
        index = int(index)
        if index in inverse and inverse[index] != name:
            raise ValueError(
                f'Mapping contains duplicate index {index}: '
                f'{inverse[index]!r} and {name!r}.'
            )
        inverse[index] = str(name)
    return inverse


def _format_optional(value):
    """Format an optional record field for tab-separated output.

    Args:
        value: Optional scalar value from a rank record.

    Returns:
        object: Empty text for ``None``, integer text for booleans, or the
            original scalar value.
    """
    if value is None:
        return ''
    if isinstance(value, bool):
        return int(value)
    return value


def build_rank_rows(records_by_model, node_to_id, relation_to_id, metadata):
    """Build serialized rows for one test set and requested models.

    Args:
        records_by_model: Iterable of ``(model_name, records)`` pairs. Each
            record is returned by ``compute_gene_rank_records`` and contains
            indexed head/tail/relation fields and a one-based ``gene_rank``.
            DeepVCF ranking also supplies ``metabolite_rank``.
        node_to_id: Entity-name to integer-index mapping.
        relation_to_id: Relation-name to integer-index mapping.
        metadata: Common output fields such as ``seed``, ``test_set``,
            ``gene_group``, and ``corruption``.

    Returns:
        list[dict]: Rank rows following ``RANK_RECORD_COLUMNS``.

    Raises:
        KeyError: If a rank record references an unknown entity or relation.
        ValueError: If a required record field is missing or metadata contains
            a column name reserved for generated fields.
    """
    reserved = {'model', 'positive_index', 'gene', 'relation', 'metabolite'}
    invalid_metadata = sorted(set(metadata) & reserved)
    if invalid_metadata:
        raise ValueError(
            f'metadata cannot override generated rank fields: {invalid_metadata}'
        )

    index_to_entity = _inverse_mapping(node_to_id)
    index_to_relation = _inverse_mapping(relation_to_id)
    rows = []
    for model_name, records in records_by_model:
        for positive_index, record in enumerate(records, start=1):
            try:
                head_idx = int(record['head_idx'])
                rel_idx = int(record['rel_idx'])
                tail_idx = int(record['tail_idx'])
                gene_rank = int(record['gene_rank'])
            except KeyError as exc:
                raise ValueError(
                    f'Rank record is missing required field {exc.args[0]!r}.'
                ) from exc
            row = dict(metadata)
            row.update({
                'model': model_name,
                'positive_index': positive_index,
                'gene': index_to_entity[head_idx],
                'relation': index_to_relation[rel_idx],
                'metabolite': index_to_entity[tail_idx],
                'gene_rank': gene_rank,
                'metabolite_rank': _format_optional(record.get('metabolite_rank')),
                'score': _format_optional(record.get('score')),
                'flipped_score': _format_optional(record.get('flipped_score')),
                'directed_win': _format_optional(record.get('directed_win')),
            })
            rows.append(row)

    return rows


def write_rank_rows(path, rows):
    """Write prebuilt rank rows to one tab-separated file.

    Args:
        path: Destination TSV path.
        rows: Iterable of dictionaries following ``RANK_RECORD_COLUMNS``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(RANK_RECORD_COLUMNS),
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows(rows)


def write_rank_records(
    path,
    records_by_model,
    node_to_id,
    relation_to_id,
    metadata,
):
    """Build and write rank rows for one test set.

    Args:
        path: Destination TSV path.
        records_by_model: Iterable of ``(model_name, records)`` pairs.
        node_to_id: Entity-name to integer-index mapping.
        relation_to_id: Relation-name to integer-index mapping.
        metadata: Common output fields for this test set.
    """
    rows = build_rank_rows(
        records_by_model,
        node_to_id,
        relation_to_id,
        metadata,
    )
    write_rank_rows(path, rows)
