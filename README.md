# mtcp
Multimodal Transformers for working with medical data with missing modalities.

## Run Training

The entrypoint for all experiments is `run.py` (Hydra-based config).

### Unimodal run

Use unimodal configs when training a single modality model (for example RNA or WSI).

```bash
python run.py --config-name unimodal_config
```

Other unimodal examples:

```bash
python run.py --config-name unimodal_config_wsi_base
python run.py --config-name unimodal_config_wsi_mae
python run.py --config-name unimodal_config_wsi_mae_surv
python run.py --config-name unimodal_wsi_embed
```

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
