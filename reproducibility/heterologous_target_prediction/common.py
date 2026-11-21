"""Input paths and summary output for heterologous target prediction."""

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / 'results' / 'heterologous_target_prediction'
KG_ROOT = REPO_ROOT / 'data' / 'processed' / 'kg' / 'ECO'
TRAIN_PATH = REPO_ROOT / 'data' / 'processed' / 'cf' / 'train' / 'train.txt'
VARIANT_KG_FILES = {
    'basic_extended': 'kg_basic_plus_extend.txt',
    'enhanced_extended': 'kg_enhanced_plus_extend.txt',
}
DEFAULT_VARIANTS = tuple(VARIANT_KG_FILES)
DEFAULT_SEEDS = tuple(range(1, 11))
QUERY_METABOLITES = ('Metabolite:ptrc_c',)


def write_json(path, payload):
    """Save the prediction's model, species, queries, and fold count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
