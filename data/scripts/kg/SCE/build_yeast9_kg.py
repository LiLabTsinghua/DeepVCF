"""Extract yeast9 triples using the SCE iMM904 KG conventions.

Example:
    python data/scripts/kg/SCE/build_yeast9_kg.py

The source SBML is yeast-GEM 9.1.1 (model/yeast-GEM.xml). Outputs are kept
separate from the existing iMM904 KG. Only the three GEM-derived relations
(Catalyzes, Consumes, Produces) are changed in the enhanced KG.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import unquote
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed/kg/SCE"
MODEL = RAW / "SCE/yeast-GEM_v9.1.1.xml"

SBML = "{http://www.sbml.org/sbml/level3/version1/core}"
FBC = "{http://www.sbml.org/sbml/level3/version1/fbc/version2}"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
GEM_RELATIONS = {"Catalyzes", "Consumes", "Produces"}
COMMON_RELATIONS = {"Belongs_to", "Related to", "in"}


def read_tsv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.reader(handle, delimiter="\t")


def read_triples(path: Path) -> set[tuple[str, str, str]]:
    return {tuple(row) for row in read_tsv(path) if len(row) == 3}


def write_triples(path: Path, triples: set[tuple[str, str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(sorted(triples))


def annotation_values(element: ET.Element, namespace: str) -> list[str]:
    prefix = f"https://identifiers.org/{namespace}/"
    return [
        unquote(item.attrib[RDF + "resource"][len(prefix):])
        for item in element.iter(RDF + "li")
        if item.attrib.get(RDF + "resource", "").startswith(prefix)
    ]


def load_aliases(path: Path, canonical_column: str) -> dict[str, str]:
    aliases = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            canonical = row[canonical_column]
            for alias in (row.get("old_bigg_ids") or "").split("; ") + [canonical]:
                if alias:
                    aliases[alias] = canonical
    return aliases


def model_mappings(model: ET.Element, counters: Counter):
    gene_ids = {}
    for uniprot, _, locus, _ in read_tsv(RAW / "SCE/sce_gene_info.txt"):
        if locus:
            gene_ids[locus] = uniprot
    known_uniprot = set(gene_ids.values())

    gene_products = model.find(FBC + "listOfGeneProducts")
    for gene in gene_products:
        gene_id = gene.get(FBC + "id")
        if gene_id not in gene_ids:
            uni = annotation_values(gene, "uniprot")
            if len(uni) == 1 and uni[0] in known_uniprot:
                gene_ids[gene_id] = uni[0]
                counters["genes_mapped_by_uniprot_annotation"] += 1

    met_aliases = load_aliases(RAW / "shared/metabolite_info.txt", "bigg_id")
    species_ids = {}
    for species in model.find(SBML + "listOfSpecies"):
        species_id = species.get("id")
        compartment = species.get("compartment")
        bigg = annotation_values(species, "bigg.metabolite")
        if bigg:
            candidate = bigg[0]
            if not candidate.endswith("_" + compartment):
                candidate += "_" + compartment
            mapped = met_aliases.get(candidate, candidate)
            counters["metabolites_with_bigg_annotation"] += 1
        else:
            mapped = f"yeast9_{species_id}_{compartment}"
            counters["metabolites_without_bigg_annotation"] += 1
        species_ids[species_id] = mapped

    rxn_aliases = load_aliases(RAW / "shared/reaction_info.txt", "bigg_id")
    candidates = {}
    for reaction in model.find(SBML + "listOfReactions"):
        rid = reaction.get("id")
        bigg = annotation_values(reaction, "bigg.reaction")
        if bigg:
            candidates[rid] = rxn_aliases.get(bigg[0], bigg[0])
            counters["reactions_with_bigg_annotation"] += 1
        else:
            candidates[rid] = "yeast9_" + rid
            counters["reactions_without_bigg_annotation"] += 1
    duplicates = Counter(candidates.values())
    reaction_ids = {}
    for rid, candidate in candidates.items():
        if duplicates[candidate] > 1:
            reaction_ids[rid] = f"{candidate}__yeast9_{rid}"
            counters["reactions_with_ambiguous_bigg_id"] += 1
        else:
            reaction_ids[rid] = candidate
    return gene_ids, species_ids, reaction_ids


def extract_basic(model: ET.Element, counters: Counter) -> set[tuple[str, str, str]]:
    gene_ids, species_ids, reaction_ids = model_mappings(model, counters)
    common = {row[2] for row in list(read_tsv(RAW / "shared/common_metabolite.txt"))[1:]
              if len(row) >= 3 and row[2]}
    met_aliases = load_aliases(RAW / "shared/metabolite_info.txt", "bigg_id")
    common = {met_aliases.get(m, m) for m in common}
    common_bases = {m.rsplit("_", 1)[0] for m in common}

    parameters = {
        item.get("id"): float(item.get("value"))
        for item in model.find(SBML + "listOfParameters")
    }
    triples = set()
    for reaction in model.find(SBML + "listOfReactions"):
        rid = reaction.get("id")
        if "biomass" in reaction.get("name", "").lower():
            counters["biomass_reactions_skipped"] += 1
            continue
        name = reaction_ids[rid]
        lower = parameters[reaction.get(FBC + "lowerFluxBound")]
        reversible = lower < 0
        participants = []
        for list_tag, relation in (("listOfReactants", "Consumes"),
                                   ("listOfProducts", "Produces")):
            listing = reaction.find(SBML + list_tag)
            if listing is None:
                continue
            for ref in listing:
                met = species_ids[ref.get("species")]
                if met in common or met.rsplit("_", 1)[0] in common_bases:
                    continue
                participants.append((relation, met))
                triples.add(("Reaction:" + name, relation, "Metabolite:" + met))
                if reversible:
                    reverse = "Produces" if relation == "Consumes" else "Consumes"
                    triples.add(("Reaction:" + name + "_-1", reverse, "Metabolite:" + met))
        if not participants:
            counters["reactions_all_metabolites_filtered"] += 1
            continue
        association = reaction.find(FBC + "geneProductAssociation")
        if association is None:
            continue
        for ref in association.iter(FBC + "geneProductRef"):
            gene_id = ref.get(FBC + "geneProduct")
            uni = gene_ids.get(gene_id)
            if not uni:
                counters["unmapped_gene_references"] += 1
                continue
            triples.add(("SCE_Protein:" + uni, "Catalyzes", "Reaction:" + name))
            if reversible:
                triples.add(("SCE_Protein:" + uni, "Catalyzes", "Reaction:" + name + "_-1"))
    return triples


def enhanced_from_basic(basic: set[tuple[str, str, str]]) -> set[tuple[str, str, str]]:
    original = read_triples(PROCESSED / "kg_enhanced.txt")
    enhanced = {triple for triple in original
                if triple[1] not in GEM_RELATIONS | COMMON_RELATIONS}
    enhanced.update(basic)
    metabolites = {node for h, _, t in basic for node in (h, t)
                   if node.startswith("Metabolite:")}
    for triple in read_triples(RAW / "shared/common_triples.txt"):
        if all(not node.startswith("Metabolite:") or node in metabolites
               for node in (triple[0], triple[2])):
            enhanced.add(triple)
    return enhanced


def write_comparison(basic: set[tuple[str, str, str]]) -> None:
    original = read_triples(PROCESSED / "kg_basic.txt")
    path = PROCESSED / "comparison_iMM904_vs_yeast9.tsv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("relation", "iMM904", "yeast9", "shared", "iMM904_only", "yeast9_only"))
        for relation in ("Catalyzes", "Consumes", "Produces", "ALL"):
            old = original if relation == "ALL" else {t for t in original if t[1] == relation}
            new = basic if relation == "ALL" else {t for t in basic if t[1] == relation}
            writer.writerow((relation, len(old), len(new), len(old & new),
                             len(old - new), len(new - old)))


def main() -> None:
    tree = ET.parse(MODEL)
    model = tree.getroot().find(SBML + "model")
    if model.get("metaid") != "yeastGEM_v9.1.1":
        raise ValueError(f"Unexpected yeast-GEM version: {model.get('metaid')}")
    counters = Counter()
    basic = extract_basic(model, counters)
    enhanced = enhanced_from_basic(basic)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    write_triples(PROCESSED / "kg_yeast9_basic.txt", basic)
    write_triples(PROCESSED / "kg_yeast9_enhanced.txt", enhanced)
    write_comparison(basic)
    summary = {
        "model": model.get("metaid"),
        "source": "https://github.com/SysBioChalmers/yeast-GEM/blob/main/model/yeast-GEM.xml",
        "source_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "basic_triples": len(basic),
        "enhanced_triples": len(enhanced),
        "basic_relations": dict(sorted(Counter(t[1] for t in basic).items())),
        "mapping_and_filter_counts": dict(sorted(counters.items())),
    }
    (PROCESSED / "kg_yeast9_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
