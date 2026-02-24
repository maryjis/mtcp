# Project structure
- `src/` - source code
  - `configs/` - configuration files
    - `multimodal_config.yaml` - configuration file for multimodal models
    - `unimodal_config.yaml` - configuration file for unimodal models
  - `data/` - data files
  - `unimodal/` - unimodal models, datasets and trainer
    - `clinical/` - clinical model (MAE, survival)
    - `cnv/` - copy number variation model (MAE, survival)
    - `dna/` - DNA model (MAE, survival)
    - `mri/` - MRI model (MAE, survival)
    - `rna/` - RNA model (MAE, survival)
    - `wsi/` - whole slide image model (MAE, survival)
    - `trainer.py` - trainer for unimodal models
  - `multimodal/` - multimodal models and trainer
    - `models.py` - multimodal models (MAE, survival, boosting survival)
    - `trainer.py` - trainer for multimodal models

# How to run
Since environment lives in .venv, be sure that it is active:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Unimodal models
### Train
You should specify unimodal config in run.py and run the following command:
```bash
python run.py
```
## Multimodal models
### Train
You should specify multimodal config in run.py and run the following command:
```bash
python run.py
```

# Logging
Neptune is used for logging. You can find your project in the [Neptune workspace](https://app.neptune.ai/o/dwemer8-workspace/org/mtcp/runs/table?viewId=standard-view) or [Neptune workspace](https://app.neptune.ai/o/almachan2358/org/cancer-mtcp/runs/table?viewId=standard-view)  (if you have access to it).

# Code style
- Add docstrings to all new functions and classes. Docstring should describe the function/class purpose and arguments.
- Do not use global variables in functions, all used variables should be passed as arguments.
- Use f"{}" instead of "{}".format() where possible.
- use logger.info() instead of print() in source code (*.py files) where possible.
