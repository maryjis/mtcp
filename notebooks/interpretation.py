# %%
%load_ext autoreload
%autoreload 2

# %%
import sys
sys.path.append("../")

# %%
import os
import json
import tqdm.auto as tqdm
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from omegaconf import OmegaConf, open_dict
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

import torch

from src.multimodal.trainer import MultiModalMAETrainer
from src.utils import load_splits, seed_everything, DummyExperimentTracker, add_model_paths_to_config

# %%
import logging
logging.getLogger().setLevel(logging.INFO)

# %% [markdown]
# # Model and datasets initialisation

# %%
if not OmegaConf.has_resolver("eval"): OmegaConf.register_new_resolver("eval", eval)

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

initialize(config_path="src/configs", job_name="test_app")  # Specify the config directory
cfg = compose(config_name="interpretation_config")  # Specify the main config file

# Access config values
print(json.dumps(OmegaConf.to_container(cfg), indent=4, ensure_ascii=False))

# %%
requested_device = str(cfg.base.device)
if requested_device.startswith("cuda") and not torch.cuda.is_available():
    device = "cpu"
    with open_dict(cfg):
        cfg.base.device = device
else:
    device = requested_device
print(f"Device: {device}")

# %%
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# %%
trainer_cache_path = f".cache/trainer_{'_'.join(cfg.base.project_ids)}.joblib"
if os.path.exists(trainer_cache_path):
    trainer = joblib.load(trainer_cache_path)
    print(f"Trainer is loaded from {trainer_cache_path}")
else:
    seed_everything(cfg.base.random_seed)

    fold_ind = 0
    print(f"Fold #{fold_ind}")
    cfg.base.save_path = f"outputs/models/{cfg.base.experiment_name}_split_{fold_ind}.pth"
    # with open_dict(cfg):
    #     cfg.base.max_samples_per_split = 64
    if cfg.model.get("is_load_pretrained", False):
        with open_dict(cfg):
            print(f"Model path outputs/models/{cfg.model.pretrained_model_name}_split_{fold_ind}.pth")
            cfg.model.pretrained_model_path = f"outputs/models/{cfg.model.pretrained_model_name}_split_{fold_ind}.pth"
    assert os.path.exists(cfg.model.pretrained_model_path), f"Model path {cfg.model.pretrained_model_path} does not exist"
    
    cfg = add_model_paths_to_config(cfg,fold_ind)
    with open_dict(cfg):
        cfg.data.wsi.random_patch_selection = cfg.model.wsi_model.random_patch_selection
        cfg.data.wsi.max_patches_per_sample = cfg.model.wsi_model.max_patches_per_sample
        print(cfg.data.wsi.random_patch_selection)
        print(cfg.data.wsi.max_patches_per_sample)
        
    splits = load_splits(
        Path(cfg.base.data_path), 
        fold_ind, 
        cfg.base.remove_nan_column, 
        max_samples_per_split=cfg.base.get("max_samples_per_split", None),
        multimodal_intersection_test =cfg.base.get("multimodal_intersection_test", None),
        modalities=cfg.base.modalities,
        project_ids = cfg.base.get("project_ids", None),
    )
    tracker = DummyExperimentTracker(cfg)
    trainer = MultiModalMAETrainer(splits, cfg, tracker, fold_ind)
    joblib.dump(trainer, trainer_cache_path)

# %%
trainer.model.to(device);

# %% [markdown]
# # Inference

# %%
split_name = "test"  # options: "train", "val", "test"
assert split_name in trainer.dataloaders, f"Unknown split_name='{split_name}'. Available splits: {list(trainer.dataloaders.keys())}"
print(f"Using split: {split_name}")

# %% [markdown]
# ## Metrics

# %%
import pandas as pd
from IPython.display import display
from src.interpretation.evaluation import evaluate_test_reconstruction_mse, paired_mse_ttest

baseline_metrics = evaluate_test_reconstruction_mse(
    trainer.model,
    trainer.dataloaders[split_name],
    device=device,
    split_name=split_name,
    zero_rna=False,
)

rna_zero_metrics = evaluate_test_reconstruction_mse(
    trainer.model,
    trainer.dataloaders[split_name],
    device=device,
    split_name=split_name,
    zero_rna=True,
)

rna_permuted_metrics = evaluate_test_reconstruction_mse(
    trainer.model,
    trainer.dataloaders[split_name],
    device=device,
    split_name=split_name,
    permute_rna=True,
)

baseline_mse_losses = baseline_metrics.pop("_mse_losses")
rna_zero_mse_losses = rna_zero_metrics.pop("_mse_losses")
rna_permuted_mse_losses = rna_permuted_metrics.pop("_mse_losses")

baseline_wsi_mse_losses = baseline_metrics.pop("_wsi_mse_losses")
rna_zero_wsi_mse_losses = rna_zero_metrics.pop("_wsi_mse_losses")
rna_permuted_wsi_mse_losses = rna_permuted_metrics.pop("_wsi_mse_losses")

delta_wsi_zeroed_vs_baseline = (
    rna_zero_metrics["mse_wsi_masked_patches"] - baseline_metrics["mse_wsi_masked_patches"]
)
delta_all_zeroed_vs_baseline = (
    rna_zero_metrics["mse_all_modalities"] - baseline_metrics["mse_all_modalities"]
)
delta_wsi_permuted_vs_baseline = (
    rna_permuted_metrics["mse_wsi_masked_patches"] - baseline_metrics["mse_wsi_masked_patches"]
)
delta_all_permuted_vs_baseline = (
    rna_permuted_metrics["mse_all_modalities"] - baseline_metrics["mse_all_modalities"]
)

comparison = {
    "baseline": baseline_metrics,
    "rna_zeroed": rna_zero_metrics,
    "rna_permuted": rna_permuted_metrics,
    "delta_wsi_zeroed_vs_baseline": delta_wsi_zeroed_vs_baseline,
    "delta_all_zeroed_vs_baseline": delta_all_zeroed_vs_baseline,
    "delta_wsi_permuted_vs_baseline": delta_wsi_permuted_vs_baseline,
    "delta_all_permuted_vs_baseline": delta_all_permuted_vs_baseline,
    "stat_test_all_modalities_zeroed": paired_mse_ttest(baseline_mse_losses, rna_zero_mse_losses),
    "stat_test_all_modalities_permuted": paired_mse_ttest(baseline_mse_losses, rna_permuted_mse_losses),
    "stat_test_wsi_masked_patches_zeroed": paired_mse_ttest(baseline_wsi_mse_losses, rna_zero_wsi_mse_losses),
    "stat_test_wsi_masked_patches_permuted": paired_mse_ttest(
        baseline_wsi_mse_losses, rna_permuted_wsi_mse_losses
    ),
}

comparison_df = pd.DataFrame.from_dict(
    {
        "baseline": comparison["baseline"],
        "rna_zeroed": comparison["rna_zeroed"],
        "rna_permuted": comparison["rna_permuted"],
        "delta_zeroed_vs_baseline": {
            "mse_all_modalities": comparison["delta_all_zeroed_vs_baseline"],
            "mse_wsi_masked_patches": comparison["delta_wsi_zeroed_vs_baseline"],
        },
        "delta_permuted_vs_baseline": {
            "mse_all_modalities": comparison["delta_all_permuted_vs_baseline"],
            "mse_wsi_masked_patches": comparison["delta_wsi_permuted_vs_baseline"],
        },
    },
    orient="index",
)

stat_test_df = pd.DataFrame.from_dict(
    {
        "all_modalities_zeroed": comparison["stat_test_all_modalities_zeroed"],
        "all_modalities_permuted": comparison["stat_test_all_modalities_permuted"],
        "wsi_masked_patches_zeroed": comparison["stat_test_wsi_masked_patches_zeroed"],
        "wsi_masked_patches_permuted": comparison["stat_test_wsi_masked_patches_permuted"],
    },
    orient="index",
)

display(comparison_df)
display(stat_test_df)


# %% [markdown]
# ## Visualization

# %%
from src.interpretation.visualization import _collect_rows, _plot_wsi_comparison_grid

rows = _collect_rows(
    trainer=trainer,
    device=device,
    dataloader=trainer.dataloaders[split_name],
    split_name=split_name,
    n_rows=8,
    seed=2026,
)

_plot_wsi_comparison_grid(rows=rows)

# %% [markdown]
# ## Attentions

# %%
from src.interpretation.attention_maps import _collect_attention_heatmaps, _plot_attention_source_comparison

if hasattr(cfg, "model"):
    with open_dict(cfg):
        cfg.model.return_attention = True

for _module in (trainer.model, getattr(trainer.model, "model", None)):
    if _module is None or not hasattr(_module, "cfg"):
        continue
    try:
        _module.cfg.return_attention = True
    except Exception:
        pass


ATTN_SOURCES = ("encoder_fusion", "decoder")
MODALITY_ORDER = ("rna", "dnam", "wsi")
MODALITY_NAMES = {
    "rna": "RNA",
    "dnam": "DNA",
    "wsi": "WSI",
}
MODALITY_COLORS = {
    "rna": "deepskyblue",
    "dnam": "limegreen",
    "wsi": "orange",
}

_core_model = getattr(trainer.model, "model", trainer.model)
rna_cluster_descriptions = list(
    _core_model.encoders["rna"].encoder.embeddings.patch_embeddings.clusters.keys()
)
dnam_cluster_descriptions = list(
    _core_model.encoders["dnam"].encoder.embeddings.patch_embeddings.clusters.keys()
)

def _build_token_index_to_description_map(
    intervals,
    rna_descriptions,
    dnam_descriptions,
):
    token_index_to_description_map = {}
    modality_to_descriptions = {
        "rna": rna_descriptions,
        "dnam": dnam_descriptions,
    }

    for modality, descriptions in modality_to_descriptions.items():
        if modality not in intervals:
            continue
        start_idx, end_idx = intervals[modality]
        n_tokens = max(0, int(end_idx) - int(start_idx))
        for local_token_index, description in enumerate(descriptions[:n_tokens]):
            token_index_to_description_map[int(start_idx) + local_token_index] = str(description)

    return token_index_to_description_map

attention_scores_present, mean_heatmaps_present, token_intervals_present = _collect_attention_heatmaps(
    model=trainer.model,
    dataloader=trainer.dataloaders[split_name],
    device=device,
    split_name=split_name,
    zero_rna=False,
    ATTN_SOURCES=ATTN_SOURCES,
)

attention_scores_zeroed, mean_heatmaps_zeroed, token_intervals_zeroed = _collect_attention_heatmaps(
    model=trainer.model,
    dataloader=trainer.dataloaders[split_name],
    device=device,
    split_name=split_name,
    zero_rna=True,
    ATTN_SOURCES=ATTN_SOURCES,
)

attention_scores_permuted, mean_heatmaps_permuted, token_intervals_permuted = _collect_attention_heatmaps(
    model=trainer.model,
    dataloader=trainer.dataloaders[split_name],
    device=device,
    split_name=split_name,
    permute_rna=True,
    ATTN_SOURCES=ATTN_SOURCES,
)

attention_scores_by_split = {
    "rna_present": attention_scores_present,
    "rna_zeroed": attention_scores_zeroed,
    "rna_permuted": attention_scores_permuted,
}

mean_attention_heatmaps_by_split = {
    "rna_present": mean_heatmaps_present,
    "rna_zeroed": mean_heatmaps_zeroed,
    "rna_permuted": mean_heatmaps_permuted,
}

token_intervals_attention_by_split = {
    "rna_present": token_intervals_present,
    "rna_zeroed": token_intervals_zeroed,
    "rna_permuted": token_intervals_permuted,
}

# %%
for _source in ATTN_SOURCES:
    _intervals = (
        token_intervals_attention_by_split["rna_present"].get(_source)
        or token_intervals_attention_by_split["rna_zeroed"].get(_source)
        or token_intervals_attention_by_split["rna_permuted"].get(_source)
        or {}
    )
    _token_index_to_description = _build_token_index_to_description_map(
        intervals=_intervals,
        rna_descriptions=rna_cluster_descriptions,
        dnam_descriptions=dnam_cluster_descriptions,
    )
    _plot_attention_source_comparison(
        source_name=_source,
        mean_heatmaps_present=mean_attention_heatmaps_by_split["rna_present"].get(_source, []),
        mean_heatmaps_zeroed=mean_attention_heatmaps_by_split["rna_zeroed"].get(_source, []),
        mean_heatmaps_permuted=mean_attention_heatmaps_by_split["rna_permuted"].get(_source, []),
        intervals=_intervals,
        token_index_to_description=_token_index_to_description,
        MODALITY_ORDER=MODALITY_ORDER,
        MODALITY_NAMES=MODALITY_NAMES,
        MODALITY_COLORS=MODALITY_COLORS,
    )
    # plt.tight_layout()
    plt.show()

# %%



