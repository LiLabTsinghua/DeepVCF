"""Shared external metric loaders, model selectors, and summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
import numpy as np
import pandas as pd
from figure.figure_style import METHOD_COLORS as SHARED_METHOD_COLORS

EXTERNAL_METHOD_COLORS = {
    **SHARED_METHOD_COLORS,
    'FSEOF': '#F3D9A6',
    'FVSEOF': '#E6AD87',
    'iBridge': '#C98A3D',
}

SEEDS = (1, 2, 3, 4, 5)


METRICS = ("AUPR", "AUROC", "Hits@10", "Hits@20", "Hits@30")


TEST_KEYS = {
    "GEM": "gem_gene/head_corrupt",
    "non-GEM": "non_gem_gene/head_corrupt",
}


@dataclass(frozen=True)
class MethodSpec:
    """Metadata needed to load one method's per-seed metrics."""

    name: str
    family: str
    kind: str
    root: str


METHOD_SPECS = (
    MethodSpec(
        "FSEOF", "Mechanistic", "mechanistic",
        "fseof",
    ),
    MethodSpec(
        "FVSEOF", "Mechanistic", "mechanistic",
        "fvseof",
    ),
    MethodSpec(
        "iBridge", "Mechanistic", "mechanistic",
        "ibridge",
    ),
    MethodSpec("LR_sklearn", "ML baseline", "ml", "lr"),
    MethodSpec("MLP_sklearn", "ML baseline", "ml", "mlp"),
    MethodSpec("RF_sklearn", "ML baseline", "ml", "rf"),
    MethodSpec(
        "DeepVCF", "DeepVCF", "deepvcf_fold1",
        "deepvcf",
    ),
    MethodSpec(
        "DeepVCF+BE", "DeepVCF", "deepvcf_ensemble",
        "deepvcf",
    ),
    MethodSpec(
        "DeepVCF+BE+MP", "DeepVCF", "deepvcf_ensemble",
        "deepvcf_mp",
    ),
)


def read_json(path):
    """Read one metrics JSON file.

    Args:
        path: JSON path to read.

    Returns:
        Parsed JSON object.

    Raises:
        FileNotFoundError: If the metrics file is absent.
        ValueError: If the JSON cannot be decoded.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _evaluation_block(payload, spec, test_key):
    """Select the metric block for a method and test set."""
    if spec.kind == "mechanistic":
        return payload["metrics"]
    test = payload["test_sets"][test_key]
    if spec.kind == "deepvcf_fold1":
        folds = test.get("folds", [])
        if not folds or folds[0].get("fold") != 1:
            raise ValueError(
                f"The first DeepVCF result is not fold 1 for seed {payload.get('seed')}."
            )
        return folds[0]
    if spec.kind == "deepvcf_ensemble":
        return test["ensemble"]
    return test


def metric_values(payload, spec, test_key):
    """Extract binary and filtered gene-ranking metrics from one payload.

    Args:
        payload: Parsed metrics JSON object.
        spec: Method metadata.
        test_key: External test-set key.

    Returns:
        Mapping from plot metric names to values.

    Raises:
        KeyError: If the payload does not contain the expected metric fields.
    """
    block = _evaluation_block(payload, spec, test_key)
    binary = block["binary"]
    gene_group = "gem" if test_key.startswith("gem_gene/") else "non_gem"
    binary_group = (
        binary if spec.kind == "mechanistic"
        else binary.get(gene_group) or binary.get("overall")
    )
    if binary_group is None:
        raise KeyError(f"No binary metrics for {gene_group} in {test_key}.")
    ranking = block["target_gene_ranking"]
    return {
        "AUPR": float(binary_group["aupr"]),
        "AUROC": float(binary_group["auroc"]),
        "Hits@10": float(ranking["gene_hit@10"]),
        "Hits@20": float(ranking["gene_hit@20"]),
        "Hits@30": float(ranking["gene_hit@30"]),
    }


def load_source_data(repo_root, specs=METHOD_SPECS, seeds=SEEDS, test_key=None):
    """Load per-seed external benchmark values into long-form data.

    Args:
        repo_root: Repository root containing result directories.
        specs: Method specifications to load.
        seeds: Experiment seeds.
        test_key: External test-set key.

    Returns:
        Long-form dataframe with one row per method, seed, and metric.

    Raises:
        ValueError: If seeds, values, or row counts are inconsistent.
    """
    if test_key is None:
        raise ValueError("test_key must be supplied explicitly.")
    records = []
    for spec in specs:
        for seed in seeds:
            # The external-evaluation results are organized by method first,
            # then seed: ``<method>/seed_<seed>/metrics.json``.  Keep the
            # loader independent of the model family because all benchmark
            # families now use the same layout.
            path = (
                repo_root / "results" / "external_evaluation"
                / spec.root / f"seed_{seed}" / "metrics.json"
            )
            payload = read_json(path)
            if int(payload.get("seed", -1)) != seed:
                raise ValueError(f"Seed mismatch in {path}.")
            values = metric_values(payload, spec, test_key)
            for metric, value in values.items():
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid {metric} value in {path}: {value}")
                records.append({
                    "method": spec.name,
                    "family": spec.family,
                    "seed": seed,
                    "metric": metric,
                    "value": value,
                })
    data = pd.DataFrame.from_records(records)
    expected_rows = len(specs) * len(seeds) * len(METRICS)
    if len(data) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(data)}.")
    return data


def summarize(data, method_order):
    """Calculate mean, sample SD, and replicate count for plotting."""
    data = data.copy()
    data["method"] = pd.Categorical(
        data["method"], categories=method_order, ordered=True,
    )
    return (
        data.groupby(["method", "family", "metric"], observed=True)["value"]
        .agg(mean="mean", std="std", n="size")
        .reset_index()
    )


ABLATION_TEST_KEYS = {"set1": "gem_gene/head_corrupt", "set2": "non_gem_gene/head_corrupt"}

METRIC_SPECS = {
    "AUPR": {
        "field": "aupr",
        "label": "Gene-corruption AUPR",
        "stem": "aupr",
        "tick_step": 0.2,
    },
    "AUROC": {
        "field": "auroc",
        "label": "Gene-corruption AUROC",
        "stem": "auroc",
        "tick_step": 0.2,
    },
    "Hits@30": {
        "field": "gene_hit@30",
        "label": "Filtered target-gene Hits@30",
        "stem": "hits30",
        "tick_step": 0.1,
    },
}

@dataclass(frozen=True)
class AblationSpec:
    """Location and metric-block selector for one plotted condition."""

    name: str
    family: str
    result_dir: str
    model: str


ABLATION_SPECS = (
    AblationSpec("LR + Random Emb.", "Random Emb.", "lr_random_embedding", "single"),
    AblationSpec("RF + Random Emb.", "Random Emb.", "rf_random_embedding", "single"),
    AblationSpec("MLP + Random Emb.", "Random Emb.", "mlp_random_embedding", "single"),
    AblationSpec("LR + DRN Emb.", "DRN Emb.", "lr", "single"),
    AblationSpec("RF + DRN Emb.", "DRN Emb.", "rf", "single"),
    AblationSpec("MLP + DRN Emb.", "DRN Emb.", "mlp", "single"),
    AblationSpec("DeepVCF", "DeepVCF", "deepvcf", "fold_1"),
    AblationSpec("DeepVCF+BE", "DeepVCF", "deepvcf", "ensemble"),
    AblationSpec(
        "DeepVCF+BE+MP",
        "DeepVCF",
        "deepvcf_mp",
        "ensemble",
    ),
)


ABLATION_ORDER = tuple(spec.name for spec in ABLATION_SPECS)


ABLATION_TICK_LABELS = {
    "LR + Random Emb.": "LR-Rand.",
    "RF + Random Emb.": "RF-Rand.",
    "MLP + Random Emb.": "MLP-Rand.",
    "LR + DRN Emb.": "LR-DRN",
    "RF + DRN Emb.": "RF-DRN",
    "MLP + DRN Emb.": "MLP-DRN",
    "DeepVCF": "DeepVCF",
    "DeepVCF+BE": "+BE",
    "DeepVCF+BE+MP": "+BE+MP",
}


ABLATION_COLORS = {
    "LR + Random Emb.": "#D1D5DB",
    "RF + Random Emb.": "#9CA3AF",
    "MLP + Random Emb.": "#6B7280",
    "LR + DRN Emb.": "#93C5FD",
    "RF + DRN Emb.": "#3B82F6",
    "MLP + DRN Emb.": "#1D4ED8",
    "DeepVCF": SHARED_METHOD_COLORS["DeepVCF"],
    "DeepVCF+BE": SHARED_METHOD_COLORS["DeepVCF+BE"],
    "DeepVCF+BE+MP": SHARED_METHOD_COLORS["DeepVCF+BE+MP"],
}


def select_metric_block(payload, spec, test_key):
    """Select the single, fold-1, or ensemble evaluation block."""
    test_result = payload["test_sets"][test_key]
    if spec.model == "single":
        return test_result
    if spec.model == "ensemble":
        return test_result["ensemble"]
    if spec.model == "fold_1":
        folds = test_result.get("folds", [])
        matches = [fold for fold in folds if fold.get("fold") == 1]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one fold 1 result for {spec.name}/{test_key}."
            )
        return matches[0]
    raise ValueError(f"Unsupported model selector: {spec.model}")


def metric_value(block, metric):
    """Extract one binary or target-gene ranking metric."""
    field = METRIC_SPECS[metric]["field"]
    if metric in {"AUPR", "AUROC"}:
        return float(block["binary"]["overall"][field])
    return float(block["target_gene_ranking"][field])


def load_ablation_data(repo_root, seeds):
    """Load seed-level AUPR, AUROC, and Hits@30 for both gene sets."""
    records = []
    for spec in ABLATION_SPECS:
        for seed in seeds:
            path = (
                repo_root / "results" / "external_evaluation"
                / spec.result_dir / f"seed_{seed}" / "metrics.json"
            )
            payload = read_json(path)
            if int(payload.get("seed", -1)) != seed:
                raise ValueError(f"Seed mismatch in {path}.")
            for gene_set, test_key in ABLATION_TEST_KEYS.items():
                block = select_metric_block(payload, spec, test_key)
                for metric in METRIC_SPECS:
                    value = metric_value(block, metric)
                    if not np.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"Invalid {metric} in {path}: {value}")
                    records.append({
                        "gene_set": gene_set,
                        "method": spec.name,
                        "family": spec.family,
                        "seed": seed,
                        "metric": metric,
                        "value": value,
                    })
    data = pd.DataFrame.from_records(records)
    expected = (
        len(ABLATION_SPECS) * len(seeds) * len(TEST_KEYS) * len(METRIC_SPECS)
    )
    if len(data) != expected:
        raise ValueError(f"Expected {expected} source rows, found {len(data)}.")
    return data


def summarize_ablation(data):
    """Calculate mean, sample SD, and replicate count."""
    data = data.copy()
    data["method"] = pd.Categorical(
        data["method"], categories=ABLATION_ORDER, ordered=True,
    )
    return (
        data.groupby(
            ["gene_set", "method", "family", "metric"], observed=True,
        )["value"]
        .agg(mean="mean", std="std", n="size")
        .reset_index()
    )
