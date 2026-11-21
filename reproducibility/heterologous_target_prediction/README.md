# Heterologous target prediction

Run from the repository root in the `deepvcf` environment, on a GPU outside
the sandbox. The entry point trains complete DeepVCF models through
`reproducibility/training/train_deepvcf.py`, then predicts heterologous
protein targets for the requested ECO metabolites.

```bash
python reproducibility/heterologous_target_prediction/evaluate_deepvcf.py \
    --variants basic_extended \
    --seeds 1 2 3 4 5 6 7 8 9 10 \
    --query-metabolites Metabolite:ptrc_c \
    --device cuda:0
```

- `basic_extended` uses `data/processed/kg/ECO/kg_basic_plus_extend.txt`.
  CF training rows outside this KG are filtered.
- `enhanced_extended` uses `data/processed/kg/ECO/kg_enhanced_plus_extend.txt`.
- Both variants use `data/processed/cf/train/train.txt` by default. DRNs and
  TPNs are trained from scratch without mechanistic pretraining. Each seed
  uses five folds by default.
- `--kg-root`, `--train-path`, and the DRN/TPN options select alternative
  inputs and training settings. `--dry-run` prints the complete plan.

Each run writes to
`results/heterologous_target_prediction/<variant>/seed_<seed>/`:

```text
model.pt
index_mappings/
run.log
summary.json
prediction_summary.json
predictions/Metabolite:ptrc_c.tsv
```

Prediction tables contain scores for both task modifications across the ECO
heterologous protein pool, with per-fold probabilities, ensemble mean, and
ensemble variance. Models and existing query results are reused by default;
`--overwrite` retrains the selected models and regenerates predictions.
