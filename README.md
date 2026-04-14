# mtcp
<img width="7040" height="5128" alt="multimodal_2stages" src="https://github.com/user-attachments/assets/2e7abb03-974e-4263-8002-3f74d5600fd8" />

Multimodal Transformers for working with medical data with missing modalities.

## Run Training

The entrypoint for all experiments is `run.py` (Hydra-based config).

### Unimodal run (RNA and WSI)

Use unimodal configs when training only one modality at a time.  
In this project, the most common unimodal variants are `RNA` and `WSI`.

#### 1) RNA (survival)

Base command:

```bash
python run.py --config-name unimodal_config
```

What this config uses by default:
- `base.type=unimodal`
- `base.modalities=["rna"]`
- `base.strategy=survival`
- model/data presets from `src/configs/model/rna_mae.yaml` + `src/configs/base/rna_base.yaml`

Useful overrides:

```bash
python run.py --config-name unimodal_config base.n_epochs=50 base.device=cuda:0
python run.py --config-name unimodal_config base.project_ids='["UCEC"]' base.batch_size=32
python run.py --config-name unimodal_config base.log.logging=False
```

#### 2) WSI

There are several WSI unimodal scenarios depending on the objective:

WSI survival baseline:

```bash
python run.py --config-name unimodal_config_wsi_base
```

WSI MAE pretraining:

```bash
python run.py --config-name unimodal_config_wsi_mae
```

WSI survival with MAE-pretrained encoder:

```bash
python run.py --config-name unimodal_config_wsi_mae_surv
```

WSI embedding variant:

```bash
python run.py --config-name unimodal_wsi_embed
```

Useful WSI overrides:

```bash
python run.py --config-name unimodal_config_wsi_mae base.n_epochs=200 base.batch_size=32
python run.py --config-name unimodal_config_wsi_base base.device=cuda:1 base.available_gpus='[1]'
python run.py --config-name unimodal_config_wsi_mae_surv base.log.logging=False
```

#### Expected outputs

- Fold models are saved to `outputs/models/` with suffix `_split_{fold}.pth`.
- Training/evaluation metrics are printed to console and can be logged to Neptune/W&B (if enabled in config).

### Multimodal run

Use multimodal configs when training a fusion model across multiple modalities (for example RNA + DNAm + WSI), including missing-modality handling.

```bash
python run.py --config-name multimodal_config
```

Other multimodal examples:

```bash
python run.py --config-name multimodal_config_2
python run.py --config-name multimodal_config_3
python run.py --config-name multimodal_config_4
```

### Useful Hydra override examples

You can override config values from the command line:

```bash
python run.py --config-name unimodal_config base.n_epochs=50 base.device=cuda:0
python run.py --config-name multimodal_config base.n_epochs=100 base.modalities='["rna","dnam","wsi"]'
```

### Notes

- `base.type` controls the pipeline: `unimodal` or `multimodal`.
- `base.strategy` controls training objective: `mae`, `survival`, or (for multimodal) `boosting_survival`.
- If you do not want experiment tracking, set `base.log.logging=False`.
