# Training

- `train_drn.py`: construct a complete model, train its DRNs, and save it with
  initialized TPN weights.
- `train_tpn.py`: load a complete model and train its TPNs with frozen DRNs.
- `train_deepvcf.py`: construct a new model from KG, train DRN followed by TPN,
  and save the complete model.

Run `train_drn.py` from the repository root in the `deepvcf` environment,
on a GPU outside the sandbox. Supply the knowledge graph and the output directory:

```bash
python reproducibility/training/train_drn.py \
  --species ECO \
  --kg ECO=data/processed/kg/ECO/kg_enhanced.txt \
  --output-dir results/drn/ECO/enhanced --seeds 1 2 3 4 5
```

For multiple species, supply their graphs together:

```bash
python reproducibility/training/train_drn.py \
  --species ECO \
  --kg ECO=data/processed/kg/ECO/kg_enhanced.txt \
       SCE=data/processed/kg/SCE/kg_enhanced.txt \
  --output-dir results/drn/cross_species/ECO_SCE/enhanced
```

Each seed writes `seed_<seed>/drn_trained_model.pt`, `index_mappings/`, `summary.json`, and
`run.log`. Choose the output directory to match the downstream experiment.
The default is five seeds and five independent DRNs per checkpoint.
Use `--epochs 0` to save random initialization, and `--help` for training
parameters. `--dry-run` prints the run configuration and output paths.
Existing checkpoints require `--overwrite`; additional seeds can be added
in a separate invocation.

PPI is treated as symmetric when present. `--no-inverse-relations` overrides
the symmetric relation list. Supply prepared KG files for each experiment.

For full DeepVCF training from scratch:

```bash
python reproducibility/training/train_deepvcf.py \
  --species ECO \
  --kg ECO=data/processed/kg/ECO/kg_enhanced.txt \
  --train-path data/processed/cf/train/train.txt \
  --output-dir results/training/ECO --seeds 1 2 3 4 5
```

To train TPNs using existing DRNs:

```bash
python reproducibility/training/train_tpn.py \
  --species ECO \
  --model-checkpoint 'results/drn/ECO/enhanced/seed_{seed}/drn_trained_model.pt' \
  --train-path data/processed/cf/train/train.txt \
  --output-dir results/training/ECO --seeds 1 2 3 4 5
```

A literal checkpoint path requires a single seed. TPN training uses the complete
model checkpoint; KG paths and DRN training parameters are supplied only to
`train_drn.py` or `train_deepvcf.py`. Network architecture, dropout, and fold count
are restored from the checkpoint. TPN optimization and early
stopping use the `--tpn-*` parameters. For training from scratch,
`--hidden-channels` and `--dropout` configure the TPN architecture.

The full-training outputs also include `model.pt`, loadable with
`DeepVCFModel.load_model`. CF data is split into train/validation folds for
early stopping. Evaluation is handled by the individual experiment directories.
Every training command requires `--species`, for example `--species ECO`,
to select the candidate pools explicitly.

Both `train_tpn.py` and `train_deepvcf.py` support four TPN training modes:

| `--mode` | TPN training | Required data |
|---|---|---|
| `default` | CF training | `--train-path` |
| `pretrain_only` | Mechanistic pretraining | `--pretrain-path` |
| `mp` | Mechanistic pretraining, then CF fine-tuning | Both paths |
| `joint_training` | CF folds with mechanistic rows in every training fold | Both paths |

Supply a separate `--output-dir` for each condition. For example:

```bash
python reproducibility/training/train_tpn.py \
  --species ECO \
  --model-checkpoint 'results/drn/ECO/enhanced/seed_{seed}/drn_trained_model.pt' \
  --mode mp \
  --pretrain-path data/processed/cf/mechanistic_pretraining/ECO_FSEOF_1_3.txt \
  --train-path data/processed/cf/train/train.txt \
  --output-dir results/training/ECO_mp --seeds 1 2 3 4 5
```

Mechanistic pretraining defaults to 50 fixed epochs and batch size 1024,
controlled by `--pretrain-epochs` and `--pretrain-batch-size`.
`pretrain_only` uses the mechanistic labels supplied by `--pretrain-path`.

`joint_training` splits CF data normally, removes mechanistic triples that
overlap the full CF dataset, and adds the remaining mechanistic rows only to
each training fold. Validation uses CF rows. `--filter-train-missing` enables
CF-processor coverage filtering, including always-train rows in joint mode.

`train_drn.py` saves the complete model as `drn_trained_model.pt`.
`train_tpn.py` and `train_deepvcf.py` save their final complete model as `model.pt`.
Pretraining also writes `pretrained_model.pt` before CF fine-tuning.
Both use `DeepVCFModel.load_model`.
Every mode saves `model.pt`; `summary.json` records the seed, species, mode,
configuration, and output checkpoint. `mp` additionally
supports `--reset-cls-after-pretrain` to reinitialize the classification layer
before CF fine-tuning.

In `default` and `mp` modes, `--always-train-path` supplies support data that
is added to every CF training fold. Validation still uses only `--train-path`.
Support triples must be disjoint from the main CF data; explicitly select
`--drop-always-train-overlap` to drop overlapping support triples instead.
Each entry point follows the data processing, training, and complete-model
saving steps directly in `main()`.
