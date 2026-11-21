# DeepVCF Data

Raw inputs, processed datasets, and data-building scripts for ECO, CGL, SCE,
BSU, and PPU.

## Layout

| Directory | Contents |
| --- | --- |
| `raw/shared/` | D2Cell DB and shared metabolite, reaction, and pathway references |
| `raw/<SPECIES>/` | GEM models, gene mappings, PPI, GO, and complex annotations |
| `processed/kg/<SPECIES>/` | Basic/enhanced KGs, ECO extensions and ablations, SCE Yeast9 KGs |
| `processed/cf/datasets/` | Positive CF triples from D2Cell, external datasets, and FSEOF |
| `processed/cf/train/` | Labeled training data and GEM/non-GEM subsets |
| `processed/cf/evaluation/` | Internal holdouts, external tests, and cross-species splits |
| `processed/cf/mechanistic_pretraining/` | Labeled FSEOF data for TPN pretraining |
| `processed/cf/negative_sampling_effect/` | Five negative-sampling variants |
| `processed/cf/conflict_sensitivity/` | Alternative CF data for conflict-sensitivity experiments |
| `processed/cf/case/` | FFA and tau case-study data and supporting tables |
| `scripts/kg/` | Species-specific KG notebooks and the SCE Yeast9 builder |
| `scripts/cf/` | Dataset extraction and negative sampling |

KGs use three tab-separated columns: `head`, `relation`, `tail`.
CF positives use `gene`, `relation`, `metabolite`; labeled CF files add a
fourth column, `label` (`1` for positive, `0` for negative). Case-study tables
have their own formats. `gem_gene` means genes represented in the GEM;
`non_gem_gene` means genes outside it.

## Building data

Use the `deepvcf` environment. Run Python scripts from the project root:

```bash
conda run -n deepvcf python data/scripts/cf/build_d2cell_datasets.py --species all
```

The default output directory is `data/processed/cf/reproduced/`.
`ECO_2024.txt` and `laser.txt` are supplied external datasets.

Add gene and metabolite negatives to a positive dataset:

```bash
conda run -n deepvcf python data/scripts/cf/cf_negative_sampling.py \
  --positive-path data/processed/cf/datasets/ECO_before_2023.txt \
  --kg-path data/processed/kg/ECO/kg_enhanced.txt \
  --negative-level both \
  --output data/processed/cf/reproduced/ECO_1_6.txt
```

Both negative ratios default to 3. Use `--negative-level gene` or `metabolite`
for one corruption type, and `--help` for filtering, seed, and output options.

KG entry points:

- `scripts/kg/<SPECIES>/build_kg.ipynb`: construct species KGs; run with the
  notebook's directory as the working directory.
- `scripts/kg/SCE/build_yeast9_kg.py`: construct SCE Yeast9 KGs; see its
  [notes](scripts/kg/SCE/README_yeast9.md).
