#!/usr/bin/env python
"""Add coverage-matched gene and/or cytoplasmic-metabolite negatives to CF positives.

Sampling positives and filtering positives are separate inputs. Sampling
positives decide which rows receive generated negatives; filtering positives
are the known positives that generated negatives must not reproduce. Input rows
are tab-separated ``gene, relation, metabolite`` triples, optionally with a
fourth label column that must be 1. Output rows always have four columns. The
intervention relation is copied unchanged when either endpoint is corrupted.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


COLUMNS = ("h", "r", "t", "label")
TRIPLE_COLUMNS = ("h", "r", "t")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-path", type=Path, required=True,
        help="Positive rows for which negatives should be sampled.",
    )
    parser.add_argument(
        "--filter-positive-path", type=Path, nargs="+",
        help=(
            "Known positive rows used to filter candidate negatives. "
            "Defaults to --positive-path."
        ),
    )
    parser.add_argument("--kg-path", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--negative-level", choices=("gene", "metabolite", "both"),
        default="both",
    )
    parser.add_argument("--gene-ratio", type=int, default=3)
    parser.add_argument("--metabolite-ratio", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def entity_prefix(entity):
    """Return the exact typed prefix before the first colon."""
    return str(entity).split(":", 1)[0]


def entity_type(entity):
    """Return the terminal biological type encoded in an entity identifier."""
    return entity_prefix(entity).rsplit("_", 1)[-1]


def load_positives(paths):
    """Load one or more positive-only files and deduplicate their triples."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = list(paths)
    if not paths:
        raise ValueError("At least one positive file is required.")

    frames = []
    for path in paths:
        frame = pd.read_csv(
            path, sep="\t", header=None, dtype=str, keep_default_na=False,
        )
        if frame.shape[1] == 3:
            frame.columns = TRIPLE_COLUMNS
            frame["label"] = 1
        elif frame.shape[1] == 4:
            frame.columns = COLUMNS
            labels = pd.to_numeric(frame["label"], errors="coerce")
            if labels.isna().any() or not labels.eq(1).all():
                raise ValueError(
                    "The input must contain positive rows only (label=1)."
                )
            frame["label"] = 1
        else:
            raise ValueError("Input must have three or four tab-separated columns.")
        for column in TRIPLE_COLUMNS:
            frame[column] = frame[column].str.strip()
        empty_fields = frame[list(TRIPLE_COLUMNS)].eq("")
        if empty_fields.any(axis=None):
            fields = [column for column in TRIPLE_COLUMNS if empty_fields[column].any()]
            rows = (frame.index[empty_fields.any(axis=1)] + 1).tolist()[:10]
            raise ValueError(
                f"Positive triples contain empty fields {fields} "
                f"in {path} at rows {rows}."
            )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True).drop_duplicates(
        list(TRIPLE_COLUMNS)
    ).reset_index(drop=True)


def load_kg(paths):
    """Load and deduplicate one or more three-column KG files."""
    if isinstance(paths, (str, Path)):
        paths = [paths]

    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(
                path,
                sep="\t",
                header=None,
                dtype=str,
                keep_default_na=False,
            )
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"KG file is empty: {path}") from exc
        if frame.shape[1] != len(TRIPLE_COLUMNS):
            raise ValueError(
                f"KG file must have exactly three tab-separated columns; "
                f"found {frame.shape[1]} in {path}."
            )
        frame.columns = TRIPLE_COLUMNS
        for column in TRIPLE_COLUMNS:
            frame[column] = frame[column].str.strip()
        if frame[list(TRIPLE_COLUMNS)].eq("").any(axis=None):
            raise ValueError(f"KG file contains empty fields: {path}")
        frames.append(frame)

    if not frames:
        raise ValueError("At least one KG file is required.")
    return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)


def _kg_entities(kg):
    """Return all entities that occur in the KG."""
    return set(kg["h"]) | set(kg["t"])


def _positive_triples(positives):
    """Return the positive triples as a set for fast membership checks."""
    return set(
        positives[list(TRIPLE_COLUMNS)].itertuples(
            index=False, name=None,
        )
    )


def _validate_ratio(ratio):
    if ratio <= 0 or int(ratio) != ratio:
        raise ValueError("Negative-sampling ratios must be positive integers.")
    return int(ratio)


def sample_gene_negatives(
    kg, filter_positives, sample_positives, ratio, rng=None,
):
    """Sample gene negatives by replacing the head gene.

    ``filter_positives`` and ``sample_positives`` have different roles:

    * ``filter_positives`` supplies known positive ``(gene, relation,
      metabolite)`` triples that candidates must not reproduce;
    * ``sample_positives`` supplies the rows for which negatives are generated.

    For each row in ``sample_positives``:

    * keep ``relation`` and ``metabolite`` unchanged;
    * use genes with the same typed prefix as the positive gene;
    * keep the same GEM-covered or GEM-uncovered status;
    * remove genes that are already positive for this relation and metabolite.

    Each positive is sampled independently.  The returned set removes
    duplicates across positives, so the final ratio does not have to be exact.
    """
    ratio = _validate_ratio(ratio)
    rng = random.Random() if rng is None else rng

    # A gene is GEM-covered when it appears as the head of a Catalyzes edge.
    entities = _kg_entities(kg)
    genes = {
        entity for entity in entities if entity_type(entity) == "Protein"
    }
    gem_genes = set(kg.loc[kg["r"] == "Catalyzes", "h"]) & genes

    # Build pools keyed by (typed prefix, GEM coverage status).
    gene_pools = {}
    for gene in sorted(genes):
        pool_key = (entity_prefix(gene), gene in gem_genes)
        if pool_key not in gene_pools:
            gene_pools[pool_key] = []
        gene_pools[pool_key].append(gene)

    # For one (relation, metabolite), all filtering-positive genes are excluded.
    positive_genes_by_rt = {}
    for gene, relation, metabolite in filter_positives[
        list(TRIPLE_COLUMNS)
    ].itertuples(index=False, name=None):
        rt_key = (relation, metabolite)
        if rt_key not in positive_genes_by_rt:
            positive_genes_by_rt[rt_key] = set()
        positive_genes_by_rt[rt_key].add(gene)

    negative_triples = set()
    for gene, relation, metabolite in sample_positives[
        list(TRIPLE_COLUMNS)
    ].itertuples(index=False, name=None):
        pool_key = (entity_prefix(gene), gene in gem_genes)
        positive_genes = positive_genes_by_rt.get(
            (relation, metabolite),
            set(),
        )
        candidates = [
            candidate
            for candidate in gene_pools.get(pool_key, [])
            if candidate != gene and candidate not in positive_genes
        ]

        # If the pool is small, use all available candidates.
        sampled_genes = rng.sample(candidates, min(ratio, len(candidates)))
        negative_triples.update(
            (negative_gene, relation, metabolite)
            for negative_gene in sampled_genes
        )

    return negative_triples


def sample_metabolite_negatives(
    kg, filter_positives, sample_positives, ratio, rng=None,
):
    """Sample metabolite negatives by replacing the tail metabolite.

    ``filter_positives`` supplies known positive triples that candidates must
    not reproduce. ``sample_positives`` supplies the rows for which negatives
    are generated.

    For each row in ``sample_positives``:

    * keep ``gene`` and ``relation`` unchanged;
    * choose a candidate from all KG metabolites ending in ``_c``;
    * exclude the original metabolite and any known positive triple.

    Each positive is sampled independently and the returned set removes
    duplicates across positives.
    """
    ratio = _validate_ratio(ratio)
    rng = random.Random() if rng is None else rng

    # The metabolite candidate pool contains only cytoplasmic metabolites.
    entities = _kg_entities(kg)
    metabolites = sorted(
        entity for entity in entities
        if entity_type(entity) == "Metabolite" and entity.endswith("_c")
    )

    positive_triples = _positive_triples(filter_positives)
    negative_triples = set()
    for gene, relation, metabolite in sample_positives[
        list(TRIPLE_COLUMNS)
    ].itertuples(index=False, name=None):
        candidates = [
            candidate for candidate in metabolites
            if candidate != metabolite
            and (gene, relation, candidate) not in positive_triples
        ]
        sampled_metabolites = rng.sample(
            candidates, min(ratio, len(candidates)),
        )
        negative_triples.update(
            (gene, relation, candidate)
            for candidate in sampled_metabolites
        )
    return negative_triples


def main():
    args = parse_args()
    if args.negative_level in {"gene", "both"}:
        _validate_ratio(args.gene_ratio)
    if args.negative_level in {"metabolite", "both"}:
        _validate_ratio(args.metabolite_ratio)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}; pass --overwrite to replace it."
        )

    sample_positives = load_positives(args.positive_path)
    filter_positive_paths = args.filter_positive_path or [args.positive_path]
    filter_positives = load_positives(filter_positive_paths)
    kg = load_kg(args.kg_path)
    rng = random.Random(args.seed)

    gene_negatives = set()
    if args.negative_level in {"gene", "both"}:
        gene_negatives = sample_gene_negatives(
            kg, filter_positives, sample_positives, args.gene_ratio, rng,
        )

    metabolite_negatives = set()
    if args.negative_level in {"metabolite", "both"}:
        metabolite_negatives = sample_metabolite_negatives(
            kg,
            filter_positives,
            sample_positives,
            args.metabolite_ratio,
            rng,
        )

    negative_triples = gene_negatives | metabolite_negatives
    negative_frame = pd.DataFrame(
        [(*triple, 0) for triple in sorted(negative_triples)], columns=COLUMNS,
    )
    output = pd.concat([sample_positives, negative_frame], ignore_index=True)
    output = output.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", header=False, index=False)

    manifest = {
        "positive_path": str(args.positive_path.resolve()),
        "filter_positive_paths": [
            str(path.resolve()) for path in filter_positive_paths
        ],
        "kg_paths": [str(path.resolve()) for path in args.kg_path],
        "output": str(args.output.resolve()),
        "negative_level": args.negative_level,
        "gene_ratio": args.gene_ratio,
        "metabolite_ratio": args.metabolite_ratio,
        "seed": args.seed,
        "num_positive": len(sample_positives),
        "num_gene_negative": len(gene_negatives),
        "num_metabolite_negative": len(metabolite_negatives),
        "num_negative": len(negative_frame),
        "num_rows": len(output),
        "relation_is_preserved": True,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
