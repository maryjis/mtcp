# mtcp
<img width="7040" height="5128" alt="multimodal_2stages" src="https://github.com/user-attachments/assets/2e7abb03-974e-4263-8002-3f74d5600fd8" />

**Joint masked pretraining across histopathology (WSI), transcriptomics (RNA-seq) and DNA methylation.**

`mtcp` implements a multimodal ViT-MAE architecture that learns a shared, process-level
representation of tumor data. Omics modalities are encoded with **GO-guided tokenization**
(biologically interpretable tokens), and the model learns cross-modal interactions through
masked-reconstruction objectives. The same backbone is used for **unimodal** and
**multimodal** training and supports **survival prognosis** and **cancer-type stratification**.

<!-- Replace with your committed figure, e.g. docs/figures/architecture.png -->
<!-- ![mtcp architecture](docs/figures/architecture.png) -->

---

## Table of contents

- [Highlights](#highlights)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Quickstart](#quickstart)
- [Training recipes](#training-recipes)
- [Configuration (Hydra)](#configuration-hydra)
- [Outputs and experiment tracking](#outputs-and-experiment-tracking)
- [Hyperparameter studies and batch runs](#hyperparameter-studies-and-batch-runs)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Highlights

- **One backbone, two regimes.** Train a single modality (RNA-seq or WSI) or fuse several
  (RNA-seq + DNA methylation + WSI) with the same entrypoint.
- **GO-guided omics tokenization.** RNA-seq and DNA methylation share a process-level token
  space, so the full transcriptome is represented without ad-hoc gene preselection.
- **Masked pretraining + downstream heads.** Self-supervised MAE pretraining, then survival
  or stratification fine-tuning.
- **Missing-modality handling** in the multimodal pipeline.
- **Config-driven (Hydra).** Every experiment is a config plus optional command-line overrides.

## Repository structure

```
mtcp/
├── run.py                    # main entrypoint (Hydra) for all experiments
├── run_study.py              # single hyperparameter study
├── run_multistage_study.py   # multi-stage study (e.g. pretrain -> fine-tune)
├── run_schedule.py           # scheduled / sequential runs
├── parallel_run.py           # launch runs in parallel
├── requirements.txt
├── src/                      # models, datasets, training loops, configs
│   └── configs/              # Hydra configs (base/, model/, ...)
├── sources/                  # supporting assets / scripts
├── notebooks/                # exploratory and analysis notebooks
└── README.md
```

> Tip: configuration groups live under `src/configs/` (for example
> `src/configs/base/rna_base.yaml` and `src/configs/model/rna_mae.yaml`).
> Looking through this folder is the fastest way to see every available option.

## Installation

**Prerequisites**

- Python 3.10+
- A CUDA-capable GPU (training is GPU-first)
- **libvips** system library (required by `pyvips` for reading whole-slide images)

```bash
# 1) system dependency for WSI reading (Debian/Ubuntu)
sudo apt-get update && sudo apt-get install -y libvips
#   macOS:        brew install vips
#   conda (any):  conda install -c conda-forge libvips

# 2) clone
git clone https://github.com/maryjis/mtcp.git
cd mtcp

# 3) environment
python -m venv .venv && source .venv/bin/activate
#   or: conda create -n mtcp python=3.10 && conda activate mtcp

# 4) Python dependencies
pip install -r requirements.txt
```

> **PyTorch / CUDA.** `requirements.txt` pins `torch==2.9.1` / `torchvision==0.24.1`.
> If the default wheel does not match your CUDA driver, install the matching build from
> https://pytorch.org first, then run `pip install -r requirements.txt` for the rest.

## Data preparation

The model is trained on multi-omics TCGA cohorts (e.g. `UCEC`, `GBM-LGG`, `LUAD`, `BLCA`,
`BRCA`) across up to three modalities:

| Modality | What it is | Encoder |
|---|---|---|
| `wsi`  | whole-slide histopathology images | ViT / ViT-MAE |
| `rna`  | RNA-seq expression | GO-tokenized transformer |
| `dnam` | DNA methylation | GO-tokenized transformer (shared tokens with RNA) |

Data locations, cohorts and preprocessing are set in the Hydra configs, **not** hard-coded.
To run on your machine:

1. Obtain the per-cohort modality files (e.g. from TCGA).
2. Open the relevant base config (for RNA: `src/configs/base/rna_base.yaml`) and point the
   data-path fields to your local directories.
   `<!-- TODO(maintainer): document the exact expected folder/file layout per modality here -->`
3. Select cohorts at run time with `base.project_ids`, e.g. `base.project_ids='["UCEC"]'`.

## Quickstart

Minimal end-to-end run (RNA-seq survival on a single cohort, no experiment tracking):

```bash
python run.py --config-name unimodal_config \
  base.project_ids='["UCEC"]' \
  base.n_epochs=50 \
  base.device=cuda:0 \
  base.log.logging=False
```

This trains per-fold models and writes them to `outputs/models/` (see
[Outputs](#outputs-and-experiment-tracking)).

## Training recipes

The entrypoint for **all** experiments is `run.py`. Select a scenario with `--config-name`.

### Unimodal — RNA-seq (survival)

```bash
python run.py --config-name unimodal_config
```

Defaults: `base.type=unimodal`, `base.modalities=["rna"]`, `base.strategy=survival`,
with presets from `src/configs/model/rna_mae.yaml` and `src/configs/base/rna_base.yaml`.

Common overrides:

```bash
python run.py --config-name unimodal_config base.n_epochs=50 base.device=cuda:0
python run.py --config-name unimodal_config base.project_ids='["UCEC"]' base.batch_size=32
```

### Unimodal — WSI (several objectives)

```bash
# survival baseline
python run.py --config-name unimodal_config_wsi_base

# MAE pretraining
python run.py --config-name unimodal_config_wsi_mae

# survival with an MAE-pretrained encoder
python run.py --config-name unimodal_config_wsi_mae_surv

# embedding variant
python run.py --config-name unimodal_wsi_embed
```

WSI override examples:

```bash
python run.py --config-name unimodal_config_wsi_mae base.n_epochs=200 base.batch_size=32
python run.py --config-name unimodal_config_wsi_base base.device=cuda:1 base.available_gpus='[1]'
```

### Multimodal (fusion + missing-modality handling)

```bash
python run.py --config-name multimodal_config
# additional presets:
python run.py --config-name multimodal_config_2
python run.py --config-name multimodal_config_3
python run.py --config-name multimodal_config_4
```

Override the fused modalities and budget:

```bash
python run.py --config-name multimodal_config \
  base.n_epochs=100 base.modalities='["rna","dnam","wsi"]'
```

## Configuration (Hydra)

All settings come from a named config (`--config-name`) and can be overridden inline as
`key=value`. The most useful keys:

| Key | Meaning | Example |
|---|---|---|
| `base.type` | pipeline: `unimodal` or `multimodal` | `base.type=multimodal` |
| `base.modalities` | active modalities | `base.modalities='["rna","dnam","wsi"]'` |
| `base.strategy` | objective: `mae`, `survival`, or `boosting_survival` (multimodal) | `base.strategy=mae` |
| `base.project_ids` | TCGA cohorts to use | `base.project_ids='["UCEC"]'` |
| `base.n_epochs` | training epochs | `base.n_epochs=100` |
| `base.batch_size` | batch size | `base.batch_size=32` |
| `base.device` | training device | `base.device=cuda:0` |
| `base.available_gpus` | visible GPUs | `base.available_gpus='[1]'` |
| `base.log.logging` | enable/disable tracking | `base.log.logging=False` |

> Quote list values so the shell does not split them: `base.modalities='["rna","dnam"]'`.

## Outputs and experiment tracking

- **Models.** Per-fold checkpoints are saved to `outputs/models/` with the suffix
  `_split_{fold}.pth`.
- **Metrics.** Printed to the console; optionally logged to **Weights & Biases** or
  **Neptune** when tracking is enabled in the config.
- **Disable tracking.** Add `base.log.logging=False` to any command.

## Hyperparameter studies and batch runs

Beyond `run.py`, several helper entrypoints are provided (Optuna is available via
`hydra-optuna-sweeper`):

- `run_study.py` — run a single hyperparameter study.
- `run_multistage_study.py` — multi-stage studies (e.g. pretrain then fine-tune).
- `run_schedule.py` — run a sequence of experiments.
- `parallel_run.py` — launch multiple runs in parallel.

Inspect each script's arguments/config before launching, since they wrap `run.py` with a
study or scheduling layer.



