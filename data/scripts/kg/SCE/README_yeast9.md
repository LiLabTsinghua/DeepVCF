# yeast9 KG extraction

Run from the repository root:

```bash
python data/scripts/kg/SCE/build_yeast9_kg.py
```

Input: `data/raw/SCE/yeast-GEM_v9.1.1.xml`, downloaded from the
[yeast-GEM model directory](https://github.com/SysBioChalmers/yeast-GEM/tree/main/model).
The script checks the model version and records the input SHA-256 in the summary.
It does not change the iMM904 notebook or existing `kg_basic.txt` /
`kg_enhanced.txt` files.

Outputs in `data/processed/kg/SCE/`:

- `kg_yeast9_basic.txt`: yeast9 `Catalyzes`, `Consumes`, `Produces` triples.
- `kg_yeast9_enhanced.txt`: yeast9 GEM triples plus the same non-GEM SCE
  relations used by the iMM904 enhanced KG. Shared metabolite/pathway triples
  are re-filtered against metabolites present in yeast9.
- `comparison_iMM904_vs_yeast9.tsv`: counts and exact triple overlap by relation.
- `kg_yeast9_summary.json`: version, source checksum, and mapping diagnostics.

The extraction follows the iMM904 notebook: skip biomass and common metabolites;
expand reactions with negative lower flux bound into forward/reverse directions;
expand gene associations to individual protein–reaction triples; map genes to
SCE UniProt IDs, and map BiGG reaction/metabolite IDs through the local shared
alias tables. A gene is also resolved via its single UniProt annotation if the
systematic ID is missing locally. Model entities without a BiGG annotation keep
a `yeast9_`-prefixed model ID; they are **not** matched by name. Consequently,
the exact-overlap table measures ID-aligned triples, not complete biological
equivalence between the two models.
