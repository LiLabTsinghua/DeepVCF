"""Shared result reading and metric validation for transfer figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DEFAULT_SEEDS = (1, 2, 3, 4, 5)
HIT_METRIC_KEYS = {
    'Hits@10': 'gene_hit@10',
    'Hits@20': 'gene_hit@20',
    'Hits@30': 'gene_hit@30',
}


def read_json(path):
    """Read one metrics file with an explicit path in any decoding error."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Metrics file not found: {path}')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON in {path}: {exc}') from exc


def validate_metric(value, context):
    """Return a finite probability-valued metric in the unit interval."""
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'Invalid metric at {context}: {value}')
    return value


def validate_seeds(seeds):
    """Require a nonempty sequence of distinct experiment seeds."""
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('--seeds must be non-empty and contain no duplicates.')
