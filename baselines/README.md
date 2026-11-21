# Mechanistic baseline predictions

Original prediction tables for FSEOF, FVSEOF, and iBridge, grouped by method,
species, and metabolic model.

```text
baselines/
├── README.md
├── FSEOF/<species>/Targets_<model>_c_new/<metabolite>/
│   └── SCE/Yeast9_fluxes_0926/<s_####>/
├── FVSEOF/<species>/<model>/<metabolite>/
└── iBridge/<species>/application_result_<model>_c_new/
```

The Yeast9 collection is an additional SCE FSEOF collection. Standard models
and prediction-file counts, checked on 2026-10-30:

| Species | Model | FSEOF TSV | FVSEOF CSV | iBridge TXT |
| --- | --- | ---: | ---: | ---: |
| ECO | iML1515 | 1,522 | 94 | 2,142 |
| CGL | iCW773 | 788 | 70 | 1,474 |
| SCE | iMM904 | 886 | 44 | 1,264 |
| BSU | iYO844 | 1,080 | 24 | 1,522 |
| PPU | iJN1463 | 1,658 | 10 | 2,678 |
| Total | | 5,934 | 242 | 9,080 |

All populated collections have complete up/down pairs. The additional Yeast9
collection contains 556 pairs (1,112 TSV files), giving 16,368 prediction
files overall. Header-only tables represent empty candidate lists.

## Formats

**FSEOF:** each metabolite directory contains
`<metabolite>_Up_regulation_targets.tsv` and
`<metabolite>_Down_regulation_targets.tsv`. The unnamed first column contains
model gene IDs; `Rank` contains the supplied signed numeric score.

**FVSEOF:** each populated metabolite directory contains
`<metabolite>_up_targets_sorted.csv` and
`<metabolite>_down_targets_sorted.csv`. The unnamed first column contains
model gene IDs, followed by `Rank`. The 135 metabolite directories comprise
121 populated directories and 14 empty directories retained as supplied.
The supplied collection includes prediction tables without generation code
or parameter metadata.

**iBridge:** files are named
`New_reaction_candidate_<up|down>_Candidate_Final_MetScore_DM_<metabolite>.txt`.
The standard tab-separated columns are `Target`, `Gene source`,
`Negative metabolite`, `Positive metabolite`, `Negative score`,
`Positive score`, `Reaction`, `Equation`, `Corr`, and `Cov`.
ECO, CGL, and SCE have 1,514, 786, and 886 files with nonstandard first-line
columns, respectively; some concatenate the header with the first data record.
Original file contents are preserved. The detailed
[format audit](input_audit/format_issues.tsv)
lists the affected paths relative to `baselines/`.

## SCE collections

SCE iBridge tables were copied from
`/home/shengkun/project/iBridge/sce/application_result_iMM904_c_new/` on
2026-10-30. All 1,264 files passed SHA-256 comparison against the source.

The additional SCE FSEOF Yeast9 tables use `s_####` model species identifiers.
Each directory holds an up/down TSV pair; 26 up-regulation tables contain only
the header. These identifiers require mapping through the KG builder's BiGG
annotations and local aliases, with `yeast9_<species>_<compartment>` fallbacks,
before lookup by KG metabolite. The corresponding positive-only CF triples
are at `data/processed/cf/case/yanyan/SCE_FSEOF.txt`. The standard benchmark
runner uses SCE's iMM904 FSEOF tables.

## Evaluation

Evaluation scripts and shared helpers are maintained in
[`reproducibility/external_evaluation/`](../reproducibility/external_evaluation/).
The [file inventory](input_audit/inventory.tsv)
records counts for each input collection.

From the project root, run in `deepvcf`, outside the sandbox on a GPU:

```bash
conda run -n deepvcf python reproducibility/external_evaluation/evaluate_mechanistic_baselines.py --device cuda:0
```

The command defaults to all three methods, five species, and seeds 1–5.
ECO outputs use `results/external_evaluation/<method>/seed_<seed>/`;
other species use
`results/cross_species_transfer/zero_shot/ECO_<species>/seed_<seed>/<method>/`.
