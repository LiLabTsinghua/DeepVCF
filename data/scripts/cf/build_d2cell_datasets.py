"""Build species-specific CF datasets from D2Cell DB.

The unified workflow supports ECO, BSU, CGL, PPU, and SCE. ECO is split into
before-2023 and 2023 datasets. Other species are restricted to metabolites
covered by both their own KG and the ECO before-2023 dataset.

Generated files are written to a reproduction directory and do not overwrite
the formal paper datasets. ``ECO_2024.txt`` and ``laser.txt`` are external
evaluation datasets and are not generated here.

The default ``oe-priority`` conflict policy keeps overexpression when a
metabolite–protein pair has both interventions. ``drop-conflicts`` removes
both interventions for such pairs across all ECO year splits. In that mode,
the before-2023 file is named ``ECO_BEFORE_2023.txt``.
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import tqdm


# =============================================================================
# Paths and dataset policy
# =============================================================================

DATA_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = DATA_DIR / "raw"
KG_DIR = DATA_DIR / "processed" / "kg"
CF_DIR = DATA_DIR / "processed" / "cf"
DATASET_DIR = CF_DIR / "datasets"
DEFAULT_OUTPUT_DIR = CF_DIR / "reproduced"

D2CELL_COLUMNS = [
    "strain type",
    "product",
    "knock out gene",
    "overexpress gene",
    "SMILES",
    "InChIKey",
    "main product",
    "Publication Year",
]
SUPPORTED_SPECIES = ("ECO", "BSU", "CGL", "PPU", "SCE")
YEAR_CUTOFFS = (2020, 2021, 2022, 2023, 2024)
CONFLICT_POLICIES = ("oe-priority", "drop-conflicts")

STRAIN_NAMES = {
    "ECO": ["Escherichia coli", "Escherichia", "E. coli", "E.coli"],
    "BSU": [
        "Bacillus subtilis",
        "Bacillus Subtilis",
        "Bacillus subtilis natto",
        "Bacillus subtilis subsp. inaquosorum",
        "B. subtilis",
    ],
    "CGL": [
        "Corynebacterium glutamicum",
        "Corynebacterium Glutamicum",
        "C. glutamicum",
        "Corynebacterium glutamicum ssp. lactofermentum",
    ],
    "PPU": ["Pseudomonas putida", "Pseudomonas putida KT2440"],
    "SCE": [
        "Saccharomyces cerevisiae",
        "Saccharomyces Cerevisiae",
        "Saccaromyces cerevisiae",
        "S. cerevisiae",
        "Recombinant Saccharomyces cerevisiae",
    ],
}

PROTEIN_PREFIX = {
    "ECO": "ECO_Protein",
    "BSU": "BSU_Protein",
    "CGL": "CGL_Protein",
    "PPU": "PPU_Protein",
    "SCE": "SCE_Protein",
}

# GEM genes are metabolic genes. Non-GEM genes are mainly non-metabolic genes
# and may include a small number of metabolic genes not incorporated into GEM.


# =============================================================================
# Shared input loading
# =============================================================================

def load_metabolite_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Load BiGG aliases and InChIKey-prefix mappings.

    Returns:
        The BiGG alias map and InChIKey-prefix-to-cytosolic-BiGG-ID map.
    """
    metabolite_df = pd.read_csv(
        RAW_DIR / "shared" / "metabolite_info.txt", sep="\t"
    )
    bigg_aliases = {}
    inchikey_to_bigg = {}

    for _, row in metabolite_df.iterrows():
        canonical_id = row["bigg_id"]
        old_ids = row.get("old_bigg_ids")
        aliases = [] if pd.isna(old_ids) else str(old_ids).split("; ")
        for alias in [*aliases, canonical_id]:
            bigg_aliases[alias] = canonical_id

        inchikey = row.get("inchikey")
        if canonical_id.endswith("_c") and pd.notna(inchikey):
            inchikey_prefix = "-".join(str(inchikey).split("-")[:-1])
            inchikey_to_bigg[inchikey_prefix] = canonical_id

    return bigg_aliases, inchikey_to_bigg


def load_common_metabolites(bigg_aliases: dict[str, str]) -> set[str]:
    """Load common metabolites excluded from CF datasets.

    Args:
        bigg_aliases: Historical-to-canonical BiGG ID mapping.

    Returns:
        Canonical IDs of common metabolites.
    """
    common_df = pd.read_csv(
        RAW_DIR / "shared" / "common_metabolite.txt", sep="\t"
    )
    return {
        bigg_aliases.get(raw_id, raw_id)
        for raw_id in common_df["met_id_in_1515"].unique()
    }


def load_d2cell() -> pd.DataFrame:
    """Load and minimally clean D2Cell records used by all species.

    Returns:
        Deduplicated rows with non-missing InChIKeys.
    """
    d2cell_df = pd.read_csv(RAW_DIR / "shared" / "D2Cell_DB.txt", sep="\t")
    return (
        d2cell_df[D2CELL_COLUMNS]
        .dropna(how="all")
        .dropna(subset=["InChIKey"])
        .drop_duplicates()
    )


def load_kg_entities(species: str, entity_prefix: str) -> set[str]:
    """Load one entity type from a species basic KG.

    Args:
        species: Species abbreviation.
        entity_prefix: Node prefix including its trailing colon.

    Returns:
        Entity identifiers with the prefix removed.
    """
    kg_df = pd.read_csv(
        KG_DIR / species / "kg_basic.txt",
        sep="\t",
        header=None,
        names=["head", "relation", "tail"],
    )
    nodes = set(kg_df["head"]) | set(kg_df["tail"])
    return {
        node[len(entity_prefix):]
        for node in nodes
        if node.startswith(entity_prefix)
    }


def load_eco_reference_metabolites() -> set[str]:
    """Load metabolites from the formal ECO before-2023 dataset.

    Returns:
        BiGG IDs represented in the formal ECO before-2023 CF dataset.
    """
    metabolites = set()
    reference_path = DATASET_DIR / "ECO_before_2023.txt"
    with reference_path.open("r", encoding="utf-8") as file:
        for line in file:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 3 and fields[2].startswith("Metabolite:"):
                metabolites.add(fields[2].removeprefix("Metabolite:"))
    return metabolites


# =============================================================================
# Species-specific gene mapping
# =============================================================================

def load_eco_gene_map() -> dict[str, str]:
    """Map lowercase ECO gene names to UniProt IDs.

    Returns:
        Gene-name-to-UniProt mapping.
    """
    gene_df = pd.read_csv(
        RAW_DIR / "ECO" / "eco_gene_info.txt",
        sep="\t",
        names=["uniprot_id", "name", "locus_tag", "full_name"],
    )
    return _gene_name_map(gene_df)


def load_bsu_gene_map() -> dict[str, str]:
    """Map lowercase BSU gene names to UniProt IDs.

    Returns:
        Gene-name-to-UniProt mapping.
    """
    gene_df = pd.read_csv(
        RAW_DIR / "BSU" / "uniprot_B_subtilis.tsv", sep="\t", header=None
    )
    return {
        str(row.iloc[1]).lower(): row.iloc[0]
        for _, row in gene_df.iterrows()
        if pd.notna(row.iloc[1])
    }


def load_cgl_gene_map() -> dict[str, str]:
    """Map lowercase CGL gene names to UniProt IDs.

    Returns:
        Gene-name-to-UniProt mapping.
    """
    gene_df = pd.read_csv(
        RAW_DIR / "CGL" / "cgl_gene_info.txt",
        sep="\t",
        names=["uniprot_id", "name", "cg_id", "cgl_id", "full_name"],
    )
    return _gene_name_map(gene_df)


def load_ppu_gene_map() -> dict[str, str]:
    """Map lowercase PPU gene annotations to UniProt IDs.

    Returns:
        Gene-annotation-to-UniProt mapping.
    """
    gene_df = pd.read_csv(RAW_DIR / "PPU" / "uniprot_mapping.tsv", sep="\t")
    gene_map = {}
    for _, row in gene_df.iterrows():
        uniprot_id = row["Entry"]
        if pd.notna(row["Gene Names"]):
            gene_map[str(row["Gene Names"]).lower()] = uniprot_id
        for value in row.iloc[1:]:
            if pd.notna(value) and not re.match(r"^PP_", str(value)):
                gene_map[str(value).lower()] = uniprot_id
    return gene_map


def load_sce_gene_map() -> dict[str, str]:
    """Map lowercase SCE gene names to UniProt IDs.

    Returns:
        Gene-name-to-UniProt mapping.
    """
    gene_df = pd.read_csv(
        RAW_DIR / "SCE" / "sce_gene_info.txt",
        sep="\t",
        names=["uniprot_id", "name", "locus_tag", "full_name"],
    )
    return _gene_name_map(gene_df)


def _gene_name_map(gene_df: pd.DataFrame) -> dict[str, str]:
    """Build a lowercase gene-name-to-UniProt map from standard columns.

    Args:
        gene_df: Table containing ``name`` and ``uniprot_id`` columns.

    Returns:
        Gene-name-to-UniProt mapping.
    """
    return {
        str(row["name"]).lower(): row["uniprot_id"]
        for _, row in gene_df.iterrows()
        if pd.notna(row["name"])
    }


GENE_LOADERS = {
    "ECO": load_eco_gene_map,
    "BSU": load_bsu_gene_map,
    "CGL": load_cgl_gene_map,
    "PPU": load_ppu_gene_map,
    "SCE": load_sce_gene_map,
}


# =============================================================================
# Pair extraction and output
# =============================================================================

def select_species_rows(d2cell_df: pd.DataFrame, species: str) -> pd.DataFrame:
    """Select D2Cell rows assigned to one species.

    Args:
        d2cell_df: Cleaned D2Cell table.
        species: Species abbreviation.

    Returns:
        Rows matching the configured strain aliases.
    """
    return d2cell_df[d2cell_df["strain type"].isin(STRAIN_NAMES[species])]


def map_d2cell_pairs(
    d2cell_df: pd.DataFrame,
    gene_column: str,
    inchikey_to_bigg: dict[str, str],
    gene_map: dict[str, str],
    common_metabolites: set[str],
) -> set[tuple[str, str]]:
    """Map D2Cell rows to unique metabolite–UniProt pairs.

    Args:
        d2cell_df: Rows with a non-missing intervention column.
        gene_column: Column containing semicolon-separated genes.
        inchikey_to_bigg: InChIKey-prefix-to-BiGG mapping.
        gene_map: Lowercase gene-name-to-UniProt mapping.
        common_metabolites: Metabolites excluded from CF datasets.

    Returns:
        Unique ``(metabolite_id, uniprot_id)`` pairs.
    """
    pairs = set()
    iterator = tqdm.tqdm(
        d2cell_df.iterrows(), total=len(d2cell_df), leave=False
    )
    for _, row in iterator:
        inchikey_prefix = "-".join(str(row["InChIKey"]).split("-")[:-1])
        metabolite_id = inchikey_to_bigg.get(inchikey_prefix)
        if metabolite_id is None or metabolite_id in common_metabolites:
            continue
        for gene_name in str(row[gene_column]).split(";"):
            uniprot_id = gene_map.get(gene_name.strip().lower())
            if uniprot_id is not None:
                pairs.add((metabolite_id, uniprot_id))
    return pairs


def write_triples(path: Path, triples: list[tuple[str, str, str]]) -> None:
    """Write deterministic tab-separated CF triples.

    Args:
        path: Destination file.
        triples: Ordered head-relation-tail rows.
    """
    with path.open("w", encoding="utf-8") as file:
        for head, relation, tail in triples:
            file.write(f"{head}\t{relation}\t{tail}\n")


def pairs_to_triples(
    overexpression_pairs: set[tuple[str, str]],
    knockout_pairs: set[tuple[str, str]],
    allowed_metabolites: set[str],
    protein_prefix: str,
    conflict_policy: str = "oe-priority",
) -> list[tuple[str, str, str]]:
    """Convert pairs to triples using the selected conflict policy.

    Args:
        overexpression_pairs: Mapped overexpression pairs.
        knockout_pairs: Mapped knockout pairs.
        allowed_metabolites: Metabolites retained in the output.
        protein_prefix: Species-specific protein node prefix.
        conflict_policy: Keep OE or discard both relations for conflicting pairs.

    Returns:
        Deterministically ordered CF triples.
    """
    if conflict_policy not in CONFLICT_POLICIES:
        raise ValueError(f"Unknown conflict policy: {conflict_policy}")
    conflicts = knockout_pairs & overexpression_pairs
    knockout_only = knockout_pairs - conflicts
    retained_overexpression = (
        overexpression_pairs - conflicts
        if conflict_policy == "drop-conflicts"
        else overexpression_pairs
    )
    return _pairs_for_relation_to_triples(
        knockout_only,
        "knock out",
        allowed_metabolites,
        protein_prefix,
    ) + _pairs_for_relation_to_triples(
        retained_overexpression,
        "overexpress",
        allowed_metabolites,
        protein_prefix,
    )


def _pairs_for_relation_to_triples(
    pairs: set[tuple[str, str]],
    relation: str,
    allowed_metabolites: set[str],
    protein_prefix: str,
) -> list[tuple[str, str, str]]:
    """Convert one intervention type to deterministic triples.

    Args:
        pairs: Mapped metabolite–UniProt pairs.
        relation: CF intervention relation.
        allowed_metabolites: Metabolites retained in the output.
        protein_prefix: Species-specific protein node prefix.

    Returns:
        Deterministically ordered CF triples.
    """
    return [
        (
            f"{protein_prefix}:{uniprot_id}",
            relation,
            f"Metabolite:{metabolite_id}",
        )
        for metabolite_id, uniprot_id in sorted(pairs)
        if metabolite_id in allowed_metabolites
    ]


# =============================================================================
# ECO time-split processing
# =============================================================================

def collect_pairs_by_cutoff(
    d2cell_df: pd.DataFrame,
    gene_column: str,
    inchikey_to_bigg: dict[str, str],
    gene_map: dict[str, str],
    common_metabolites: set[str],
) -> dict[int, set[tuple[str, str]]]:
    """Collect cumulative pairs before each publication-year cutoff.

    Args:
        d2cell_df: Species-specific D2Cell rows.
        gene_column: Intervention gene column.
        inchikey_to_bigg: InChIKey-prefix-to-BiGG mapping.
        gene_map: Lowercase gene-name-to-UniProt mapping.
        common_metabolites: Metabolites excluded from CF datasets.

    Returns:
        Exclusive year cutoffs mapped to cumulative pairs.
    """
    cumulative_pairs = {}
    for cutoff in YEAR_CUTOFFS:
        rows_before_cutoff = d2cell_df[
            d2cell_df["Publication Year"] < cutoff
        ]
        cumulative_pairs[cutoff] = map_d2cell_pairs(
            rows_before_cutoff,
            gene_column,
            inchikey_to_bigg,
            gene_map,
            common_metabolites,
        )
    return cumulative_pairs


def process_eco(
    d2cell_df: pd.DataFrame,
    inchikey_to_bigg: dict[str, str],
    common_metabolites: set[str],
    output_dir: Path,
    conflict_policy: str = "oe-priority",
) -> set[str]:
    """Build ECO before-2023 and 2023 CF datasets.

    Args:
        d2cell_df: Cleaned D2Cell table.
        inchikey_to_bigg: InChIKey-prefix-to-BiGG mapping.
        common_metabolites: Metabolites excluded from CF datasets.
        output_dir: Destination for reproduced files.
        conflict_policy: Keep OE or discard both relations for conflicting pairs.

    Returns:
        Metabolites represented in the reproduced ECO before-2023 dataset.
    """
    if conflict_policy not in CONFLICT_POLICIES:
        raise ValueError(f"Unknown conflict policy: {conflict_policy}")
    species_rows = select_species_rows(d2cell_df, "ECO")
    knockout_rows = species_rows.dropna(subset=["knock out gene"])
    overexpression_rows = species_rows.dropna(subset=["overexpress gene"])
    gene_map = load_eco_gene_map()
    kg_metabolites = load_kg_entities("ECO", "Metabolite:")

    print("\n" + "=" * 60)
    print("Processing ECO")
    print("=" * 60)
    print(
        f"D2Cell rows: {len(species_rows)}, KO: {len(knockout_rows)}, "
        f"OXP: {len(overexpression_rows)}"
    )

    overexpression_by_cutoff = collect_pairs_by_cutoff(
        overexpression_rows,
        "overexpress gene",
        inchikey_to_bigg,
        gene_map,
        common_metabolites,
    )
    knockout_by_cutoff = collect_pairs_by_cutoff(
        knockout_rows,
        "knock out gene",
        inchikey_to_bigg,
        gene_map,
        common_metabolites,
    )
    all_overexpression_pairs = set().union(*overexpression_by_cutoff.values())
    all_knockout_pairs = set().union(*knockout_by_cutoff.values())
    conflicts = all_knockout_pairs & all_overexpression_pairs

    triples_by_year = {}
    previous_cutoff = None
    for cutoff in YEAR_CUTOFFS:
        current_knockout = knockout_by_cutoff[cutoff].copy()
        current_overexpression = overexpression_by_cutoff[cutoff].copy()
        if previous_cutoff is not None:
            current_knockout -= knockout_by_cutoff[previous_cutoff]
            current_overexpression -= overexpression_by_cutoff[previous_cutoff]

        year_label = f"<{cutoff}" if previous_cutoff is None else str(cutoff - 1)
        knockout_triples = _pairs_for_relation_to_triples(
            current_knockout - conflicts,
            "knock out",
            kg_metabolites,
            PROTEIN_PREFIX["ECO"],
        )
        overexpression_triples = _pairs_for_relation_to_triples(
            current_overexpression - conflicts
            if conflict_policy == "drop-conflicts"
            else current_overexpression,
            "overexpress",
            kg_metabolites,
            PROTEIN_PREFIX["ECO"],
        )
        triples_by_year[year_label] = knockout_triples + overexpression_triples
        previous_cutoff = cutoff

    before_2023 = [
        triple
        for year, triples in triples_by_year.items()
        if year != "2023"
        for triple in triples
    ]
    in_2023 = triples_by_year["2023"]

    before_2023_filename = (
        "ECO_BEFORE_2023.txt"
        if conflict_policy == "drop-conflicts"
        else "ECO_before_2023.txt"
    )
    write_triples(output_dir / before_2023_filename, before_2023)
    write_triples(output_dir / "ECO_2023.txt", in_2023)
    print(f"{before_2023_filename}: {len(before_2023)} entries")
    print(f"ECO_2023.txt: {len(in_2023)} entries")

    return {tail.removeprefix("Metabolite:") for _, _, tail in before_2023}


# =============================================================================
# Non-ECO species processing
# =============================================================================

def process_non_eco_species(
    species: str,
    d2cell_df: pd.DataFrame,
    inchikey_to_bigg: dict[str, str],
    common_metabolites: set[str],
    eco_reference_metabolites: set[str],
    output_dir: Path,
    conflict_policy: str = "oe-priority",
) -> None:
    """Build one non-ECO species CF dataset.

    Args:
        species: Species abbreviation.
        d2cell_df: Cleaned D2Cell table.
        inchikey_to_bigg: InChIKey-prefix-to-BiGG mapping.
        common_metabolites: Metabolites excluded from CF datasets.
        eco_reference_metabolites: Metabolites in ECO before-2023 data.
        output_dir: Destination for reproduced files.
        conflict_policy: Keep OE or discard both relations for conflicting pairs.
    """
    species_rows = select_species_rows(d2cell_df, species)
    knockout_rows = species_rows.dropna(subset=["knock out gene"])
    overexpression_rows = species_rows.dropna(subset=["overexpress gene"])
    gene_map = GENE_LOADERS[species]()

    print("\n" + "=" * 60)
    print(f"Processing {species}")
    print("=" * 60)
    print(
        f"D2Cell rows: {len(species_rows)}, KO: {len(knockout_rows)}, "
        f"OXP: {len(overexpression_rows)}"
    )

    overexpression_pairs = map_d2cell_pairs(
        overexpression_rows,
        "overexpress gene",
        inchikey_to_bigg,
        gene_map,
        common_metabolites,
    )
    knockout_pairs = map_d2cell_pairs(
        knockout_rows,
        "knock out gene",
        inchikey_to_bigg,
        gene_map,
        common_metabolites,
    )
    print(
        f"Collected pairs — OXP: {len(overexpression_pairs)}, "
        f"KO: {len(knockout_pairs)}"
    )

    kg_metabolites = load_kg_entities(species, "Metabolite:")
    allowed_metabolites = kg_metabolites & eco_reference_metabolites
    triples = pairs_to_triples(
        overexpression_pairs,
        knockout_pairs,
        allowed_metabolites,
        PROTEIN_PREFIX[species],
        conflict_policy,
    )
    output_path = output_dir / f"{species}_dataset.txt"
    write_triples(output_path, triples)
    print(f"{output_path.name}: {len(triples)} entries")


# =============================================================================
# Command-line interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build species-specific CF datasets from D2Cell DB."
    )
    parser.add_argument(
        "--species",
        nargs="+",
        choices=["all", *SUPPORTED_SPECIES],
        default=["all"],
        help="Species to process; default: all.",
    )
    parser.add_argument(
        "--conflict-policy",
        choices=CONFLICT_POLICIES,
        default="oe-priority",
        help=(
            "How to handle pairs with both OE and KO: keep OE (default) "
            "or discard both."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for reproduced datasets.",
    )
    return parser.parse_args()


def resolve_species(requested_species: list[str]) -> list[str]:
    """Resolve and validate requested species.

    Args:
        requested_species: Values supplied to ``--species``.

    Returns:
        Species in canonical processing order.

    Raises:
        ValueError: If ``all`` is combined with an explicit species.
    """
    if "all" in requested_species:
        if len(requested_species) != 1:
            raise ValueError("'all' cannot be combined with explicit species.")
        return list(SUPPORTED_SPECIES)
    requested_set = set(requested_species)
    return [species for species in SUPPORTED_SPECIES if species in requested_set]


def main() -> None:
    """Run requested D2Cell-to-CF construction workflows."""
    args = parse_args()
    species_to_process = resolve_species(args.species)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading shared reference data...")
    bigg_aliases, inchikey_to_bigg = load_metabolite_maps()
    common_metabolites = load_common_metabolites(bigg_aliases)
    d2cell_df = load_d2cell()

    if "ECO" in species_to_process:
        eco_reference_metabolites = process_eco(
            d2cell_df,
            inchikey_to_bigg,
            common_metabolites,
            args.output_dir,
            args.conflict_policy,
        )
    else:
        eco_reference_metabolites = load_eco_reference_metabolites()

    for species in species_to_process:
        if species == "ECO":
            continue
        process_non_eco_species(
            species,
            d2cell_df,
            inchikey_to_bigg,
            common_metabolites,
            eco_reference_metabolites,
            args.output_dir,
            args.conflict_policy,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
