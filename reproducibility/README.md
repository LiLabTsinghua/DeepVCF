# DeepVCF Reproducibility

This directory contains the experiment entry points used to reproduce the
results reported in the DeepVCF paper.

Each experiment has Python entry points and a short description of its
inputs and outputs. Experiment scripts reuse `code/` and the formal datasets
in `data/processed/`. Run GPU experiments outside the sandbox.

## Environment

All paper experiments use the existing Conda environment `deepvcf`:

```bash
conda activate deepvcf
```

Experiment commands may also be run without activating the environment:

```bash
conda run -n deepvcf <command>
```

## Experiments

- `training/`: separate entry points for DRN training, TPN training from
  existing DRNs, and full DeepVCF training from supplied KG and CF data.
- `internal_evaluation/`: internal holdout training, evaluation, and metric aggregation.
- Negative-sampling results are under `results/negative_sampling_effect/`;
  their plotting script is under `figure/negative_sampling_effect/`.
- `external_evaluation/`: DeepVCF external benchmarks using `training/`,
  with additional ML and mechanistic baseline evaluators. Metrics, rank records,
  and checkpoints are grouped under `results/external_evaluation/`.
- `heterologous_target_prediction/`: `evaluate_deepvcf.py` trains complete
  DeepVCF models through `training/` on extended ECO KGs, then predicts
  heterologous enzyme targets for putrescine production.
- `cross_species_transfer/`: one `evaluate_deepvcf.py` entry point for zero-shot
  and few-shot transfer plus ECO external evaluation, with training handled
  by `training/`.
