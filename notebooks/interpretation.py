# %%
# %load_ext autoreload
# %autoreload 2

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
# import logging
# logging.getLogger().setLevel(logging.INFO)

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
if os.path.exists(".cache/trainer.joblib"):
    trainer = joblib.load(".cache/trainer.joblib")
else:
    requested_device = str(cfg.base.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        with open_dict(cfg):
            cfg.base.device = device
    else:
        device = requested_device
    print(f"Device: {device}")

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
    joblib.dump(trainer, ".cache/trainer.joblib")

# %%
joblib.dump(trainer, ".cache/trainer.joblib")

# %% [markdown]
# # Inference

# %% [markdown]
# ## Metrics

# %%
try:
    from scipy.stats import ttest_rel
except ImportError:
    ttest_rel = None


def evaluate_test_reconstruction_mse(model, dataloader, device, zero_rna=False):
    model.eval()

    total_mse = 0.0
    total_wsi_mse = 0.0
    total_samples = 0
    total_wsi_samples = 0
    mse_losses = []
    wsi_mse_losses = []

    desc = "test | RNA=0" if zero_rna else "test | RNA=orig"

    with torch.no_grad():
        for data, masks in tqdm.tqdm(dataloader, desc=desc):
            data = {modality: value.to(device) for modality, value in data.items()}
            masks = {modality: value.to(device) for modality, value in masks.items()}

            if zero_rna and "rna" in data:
                data = dict(data)
                data["rna"] = torch.zeros_like(data["rna"])

            outputs = model(data, masks)

            batch_size = next(iter(data.values())).shape[0]
            batch_mse = float(outputs.loss[0].detach().item())
            mse_losses.append(batch_mse)
            total_mse += batch_mse * batch_size
            total_samples += batch_size

            if "wsi" in outputs.loss[1] and "wsi" in masks:
                wsi_count = int(masks["wsi"].sum().item())
                batch_wsi_mse = float(outputs.loss[1]["wsi"].detach().item())
                wsi_mse_losses.append(batch_wsi_mse)
                total_wsi_mse += batch_wsi_mse * wsi_count
                total_wsi_samples += wsi_count

    return {
        "mse_all_modalities": total_mse / total_samples if total_samples > 0 else float("nan"),
        "mse_all_modalities_std": (
            float(torch.tensor(mse_losses, dtype=torch.float64).std(unbiased=True).item())
            if len(mse_losses) > 1
            else float("nan")
        ),
        "mse_wsi_masked_patches": total_wsi_mse / total_wsi_samples if total_wsi_samples > 0 else float("nan"),
        "mse_wsi_masked_patches_std": (
            float(torch.tensor(wsi_mse_losses, dtype=torch.float64).std(unbiased=True).item())
            if len(wsi_mse_losses) > 1
            else float("nan")
        ),
        "num_samples": total_samples,
        "num_samples_with_wsi": total_wsi_samples,
        "_mse_losses": mse_losses,
        "_wsi_mse_losses": wsi_mse_losses,
    }


def paired_mse_ttest(reference_losses, perturbed_losses):
    if ttest_rel is None:
        return {"test": "paired_ttest", "error": "scipy is not installed"}

    if len(reference_losses) != len(perturbed_losses):
        return {
            "test": "paired_ttest",
            "error": f"length mismatch: {len(reference_losses)} vs {len(perturbed_losses)}",
        }

    if len(reference_losses) < 2:
        return {"test": "paired_ttest", "error": "need at least 2 paired observations"}

    stat = ttest_rel(reference_losses, perturbed_losses)
    return {
        "test": "paired_ttest",
        "n_pairs": len(reference_losses),
        "statistic": float(stat.statistic),
        "pvalue": float(stat.pvalue),
    }


# %%
import pandas as pd
from IPython.display import display

baseline_metrics = evaluate_test_reconstruction_mse(
    trainer.model,
    trainer.dataloaders["test"],
    device=device,
    zero_rna=False,
)

rna_zero_metrics = evaluate_test_reconstruction_mse(
    trainer.model,
    trainer.dataloaders["test"],
    device=device,
    zero_rna=True,
)

baseline_mse_losses = baseline_metrics.pop("_mse_losses")
rna_zero_mse_losses = rna_zero_metrics.pop("_mse_losses")
baseline_wsi_mse_losses = baseline_metrics.pop("_wsi_mse_losses")
rna_zero_wsi_mse_losses = rna_zero_metrics.pop("_wsi_mse_losses")

delta_wsi_vs_baseline = (
    rna_zero_metrics["mse_wsi_masked_patches"] - baseline_metrics["mse_wsi_masked_patches"]
)
delta_all_vs_baseline = (
    rna_zero_metrics["mse_all_modalities"] - baseline_metrics["mse_all_modalities"]
)

comparison = {
    "baseline": baseline_metrics,
    "rna_zeroed": rna_zero_metrics,
    "delta_wsi_vs_baseline": delta_wsi_vs_baseline,
    "delta_all_vs_baseline": delta_all_vs_baseline,
    "stat_test_all_modalities": paired_mse_ttest(baseline_mse_losses, rna_zero_mse_losses),
    "stat_test_wsi_masked_patches": paired_mse_ttest(baseline_wsi_mse_losses, rna_zero_wsi_mse_losses),
}

comparison_df = pd.DataFrame.from_dict(
    {
        "baseline": comparison["baseline"],
        "rna_zeroed": comparison["rna_zeroed"],
        "delta_vs_baseline": {
            "mse_all_modalities": comparison["delta_all_vs_baseline"],
            "mse_wsi_masked_patches": comparison["delta_wsi_vs_baseline"],
        },
    },
    orient="index",
)

stat_test_df = pd.DataFrame.from_dict(
    {
        "all_modalities": comparison["stat_test_all_modalities"],
        "wsi_masked_patches": comparison["stat_test_wsi_masked_patches"],
    },
    orient="index",
)

display(comparison_df)
display(stat_test_df)


# %% [markdown]
# ## Visualization

# %%
import numpy as np
import matplotlib.pyplot as plt


def _to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().float()
    if arr.dim() == 4:
        arr = arr[0]
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    arr = arr.permute(1, 2, 0).numpy()
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return arr


def _extract_first_wsi_patch(wsi_sample: torch.Tensor) -> torch.Tensor:
    if wsi_sample.dim() == 4:
        return wsi_sample[0]
    if wsi_sample.dim() == 3:
        return wsi_sample
    raise ValueError(f"Unexpected WSI sample shape: {tuple(wsi_sample.shape)}")


def _select_sample_indices(
    masks_batch: dict[str, torch.Tensor],
    max_count: int,
    required_modalities: tuple[str, ...] = ("wsi", "rna"),
) -> list[int]:
    first_mask = next(iter(masks_batch.values())).bool()
    candidate_mask = torch.ones_like(first_mask)

    for modality in required_modalities:
        if modality in masks_batch:
            candidate_mask = candidate_mask & masks_batch[modality].bool()

    candidate_ids = torch.where(candidate_mask)[0]
    if len(candidate_ids) == 0 and "wsi" in masks_batch:
        candidate_ids = torch.where(masks_batch["wsi"].bool())[0]

    return [int(i) for i in candidate_ids[:max_count].tolist()]


def _get_modality_interval(trainer, modality: str) -> tuple[int, int]:
    start_idx = 1  # skip multimodal CLS token in outputs.mask
    for current_modality in trainer.model.modalities:
        num_patches = trainer.model.get_patches_number(current_modality)
        end_idx = start_idx + num_patches
        if current_modality == modality:
            return start_idx, end_idx
        start_idx = end_idx
    raise KeyError(f"Modality '{modality}' not found in model modalities: {trainer.model.modalities}")


def _run_wsi_reconstruction(
    trainer,
    device: str,
    values_batch: dict[str, torch.Tensor],
    masks_batch: dict[str, torch.Tensor],
    zero_rna: bool = False,
    seed: int = 2026,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_inputs = {k: v.clone().to(device) for k, v in values_batch.items()}
    model_masks = {k: v.clone().to(device) for k, v in masks_batch.items()}

    if zero_rna and "rna" in model_inputs:
        model_inputs["rna"] = torch.zeros_like(model_inputs["rna"])

    trainer.model.eval()
    with torch.no_grad():
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        outputs = trainer.model(model_inputs, model_masks)
        pred_by_modality = trainer.model.split_modalities(outputs.logits)

        pred_wsi = pred_by_modality["wsi"]
        wsi_encoder = trainer.model.model.encoders["wsi"].encoder
        recon_wsi = wsi_encoder.unpatchify(pred_wsi, None)

        wsi_start, wsi_end = _get_modality_interval(trainer=trainer, modality="wsi")
        wsi_patch_mask = outputs.mask[:, wsi_start:wsi_end]

    return recon_wsi, wsi_patch_mask


def _build_masked_wsi_image(
    trainer,
    wsi_sample: torch.Tensor,
    wsi_patch_mask: torch.Tensor,
) -> torch.Tensor:
    wsi_encoder = trainer.model.model.encoders["wsi"].encoder
    wsi_patch = _extract_first_wsi_patch(wsi_sample)

    patchified = wsi_encoder.patchify(wsi_patch.unsqueeze(0))
    patch_mask = wsi_patch_mask.detach().to(patchified.device).float().view(1, -1, 1)
    masked_patchified = patchified * (1.0 - patch_mask)
    masked_image = wsi_encoder.unpatchify(masked_patchified, None)[0]

    return masked_image


def _build_wsi_pixel_mask(
    trainer,
    wsi_patch_mask: torch.Tensor,
) -> torch.Tensor:
    wsi_encoder = trainer.model.model.encoders["wsi"].encoder
    patch_vec_len = (wsi_encoder.config.patch_size ** 2) * wsi_encoder.config.num_channels

    patch_mask = wsi_patch_mask.detach().float().view(1, -1, 1)
    patch_mask = patch_mask.repeat(1, 1, patch_vec_len)

    pixel_mask = wsi_encoder.unpatchify(patch_mask, None)[0].mean(dim=0)
    pixel_mask = (pixel_mask > 0.5).float()
    return pixel_mask


def _build_row_data(
    trainer,
    values_batch: dict[str, torch.Tensor],
    recon_with_rna: torch.Tensor,
    recon_with_zero_rna: torch.Tensor,
    wsi_patch_mask: torch.Tensor,
    sample_idx: int,
    sample_label: str,
) -> dict:
    original_wsi_sample = values_batch["wsi"][sample_idx]
    original_wsi_patch = _extract_first_wsi_patch(original_wsi_sample)

    masked_input_wsi_patch = _build_masked_wsi_image(
        trainer=trainer,
        wsi_sample=original_wsi_sample,
        wsi_patch_mask=wsi_patch_mask[sample_idx],
    )

    recon_present_patch = recon_with_rna[sample_idx]
    recon_zero_patch = recon_with_zero_rna[sample_idx]

    diff_map = (recon_present_patch.detach().cpu().float() - recon_zero_patch.detach().cpu().float()).abs().mean(dim=0).numpy()
    pixel_mask = _build_wsi_pixel_mask(trainer=trainer, wsi_patch_mask=wsi_patch_mask[sample_idx]).cpu().numpy()

    diff_masked = diff_map * pixel_mask
    diff_unmasked = diff_map * (1.0 - pixel_mask)

    return {
        "label": sample_label,
        "img_orig": _to_rgb_image(original_wsi_patch),
        "img_masked": _to_rgb_image(masked_input_wsi_patch),
        "img_rna": _to_rgb_image(recon_present_patch),
        "img_zero": _to_rgb_image(recon_zero_patch),
        "diff_full": diff_map,
        "diff_masked": diff_masked,
        "diff_unmasked": diff_unmasked,
    }


def _collect_rows(
    trainer,
    device: str,
    dataloader,
    n_rows: int = 8,
    seed: int = 2026,
) -> list[dict]:
    rows = []

    for batch_idx, (values_batch, masks_batch) in enumerate(dataloader):
        remaining = n_rows - len(rows)
        if remaining <= 0:
            break

        sample_indices = _select_sample_indices(
            masks_batch=masks_batch,
            max_count=remaining,
            required_modalities=("wsi", "rna"),
        )

        if len(sample_indices) == 0:
            continue

        batch_seed = seed + batch_idx
        recon_with_rna, wsi_patch_mask = _run_wsi_reconstruction(
            trainer=trainer,
            device=device,
            values_batch=values_batch,
            masks_batch=masks_batch,
            zero_rna=False,
            seed=batch_seed,
        )

        recon_with_zero_rna, _ = _run_wsi_reconstruction(
            trainer=trainer,
            device=device,
            values_batch=values_batch,
            masks_batch=masks_batch,
            zero_rna=True,
            seed=batch_seed,
        )

        for sample_idx in sample_indices:
            rows.append(
                _build_row_data(
                    trainer=trainer,
                    values_batch=values_batch,
                    recon_with_rna=recon_with_rna,
                    recon_with_zero_rna=recon_with_zero_rna,
                    wsi_patch_mask=wsi_patch_mask,
                    sample_idx=sample_idx,
                    sample_label=f"batch {batch_idx}, idx {sample_idx}",
                )
            )
            if len(rows) >= n_rows:
                break

    if len(rows) < n_rows:
        raise RuntimeError(
            f"Requested {n_rows} rows, but only found {len(rows)} samples with WSI available. "
            "Increase batch size or ensure enough valid WSI samples in test split."
        )

    return rows


def _plot_wsi_comparison_grid(rows: list[dict]) -> None:
    n_rows = len(rows)
    n_cols = 7

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    vmax = max(float(row["diff_full"].max()) for row in rows)
    vmax = max(vmax, 1e-8)

    titles = [
        "Original WSI patch",
        "Original masked\n(before model)",
        "Recon (RNA present)",
        "Recon (RNA zeroed)",
        "|Present - Zeroed|",
        "|Present - Zeroed|\nmasked patches",
        "|Present - Zeroed|\nunmasked patches",
    ]

    for col, title in enumerate(titles):
        axes[0, col].set_title(title)

    mappable = None
    for row_idx, row in enumerate(rows):
        axes[row_idx, 0].imshow(row["img_orig"])
        axes[row_idx, 1].imshow(row["img_masked"])
        axes[row_idx, 2].imshow(row["img_rna"])
        axes[row_idx, 3].imshow(row["img_zero"])

        axes[row_idx, 4].imshow(row["diff_full"], cmap="magma", vmin=0.0, vmax=vmax)
        axes[row_idx, 5].imshow(row["diff_masked"], cmap="magma", vmin=0.0, vmax=vmax)
        mappable = axes[row_idx, 6].imshow(row["diff_unmasked"], cmap="magma", vmin=0.0, vmax=vmax)

        axes[row_idx, 0].set_ylabel(row["label"], rotation=0, labelpad=52, va="center")

        for col in range(n_cols):
            axes[row_idx, col].axis("off")

    if mappable is not None:
        fig.colorbar(mappable, ax=axes[:, 4:].ravel().tolist(), fraction=0.015, pad=0.01)

    plt.tight_layout()
    plt.show()


rows = _collect_rows(
    trainer=trainer,
    device=device,
    dataloader=trainer.dataloaders["test"],
    n_rows=8,
    seed=2026,
)

_plot_wsi_comparison_grid(rows=rows)



# %% [markdown]
# ## Attentions

# %%
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


def _as_layer_tuple(value):
    if value is None:
        return tuple()
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, tuple):
        return value
    return tuple()


def _sanitize_intervals(intervals):
    result = {}
    if not isinstance(intervals, dict):
        return result

    for key, val in intervals.items():
        if isinstance(val, (list, tuple)) and len(val) == 2:
            start, end = int(val[0]), int(val[1])
            if end > start:
                result[str(key)] = (start, end)

    return result


def _normalize_attention_payload(attentions):
    payload = {
        "encoder_fusion": tuple(),
        "decoder": tuple(),
        "token_intervals": {
            "encoder_fusion": {},
            "decoder": {},
        },
    }

    if attentions is None:
        return payload

    if isinstance(attentions, dict):
        payload["encoder_fusion"] = _as_layer_tuple(attentions.get("encoder_fusion"))
        payload["decoder"] = _as_layer_tuple(attentions.get("decoder"))

        token_intervals = attentions.get("token_intervals", {})
        if isinstance(token_intervals, dict):
            payload["token_intervals"]["encoder_fusion"] = _sanitize_intervals(token_intervals.get("encoder_fusion", {}))
            payload["token_intervals"]["decoder"] = _sanitize_intervals(token_intervals.get("decoder", {}))

        return payload

    payload["decoder"] = _as_layer_tuple(attentions)
    return payload


def _to_sample_attention_matrices(layer_attention: torch.Tensor) -> torch.Tensor:
    att = layer_attention.detach().float().cpu()

    if att.dim() == 4:
        # [B, heads, q_tokens, k_tokens] -> [B, q_tokens, k_tokens]
        att = att.mean(dim=1)
    elif att.dim() == 3:
        # [B, q_tokens, k_tokens]
        pass
    elif att.dim() == 2:
        # [q_tokens, k_tokens] -> [1, q_tokens, k_tokens]
        att = att.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected attention tensor shape: {tuple(att.shape)}")

    return att


def _default_decoder_token_intervals(model):
    intervals = {}
    start = 1  # decoder CLS is index 0
    for modality in model.modalities:
        end = start + model.get_patches_number(modality)
        intervals[modality] = (start, end)
        start = end
    return intervals


def _collect_attention_heatmaps(model, dataloader, device: str, zero_rna: bool = False):
    raw_attention_payloads = []
    layer_sums = {source: [] for source in ATTN_SOURCES}
    layer_counts = {source: [] for source in ATTN_SOURCES}
    token_intervals = {source: {} for source in ATTN_SOURCES}

    desc = "test inference | RNA=0" if zero_rna else "test inference | RNA=orig"
    model.eval()

    with torch.no_grad():
        for values_batch, masks_batch in tqdm.tqdm(dataloader, desc=desc):
            values_batch = {modality: value.to(device) for modality, value in values_batch.items()}
            masks_batch = {modality: value.to(device) for modality, value in masks_batch.items()}

            if zero_rna and "rna" in values_batch:
                values_batch = dict(values_batch)
                values_batch["rna"] = torch.zeros_like(values_batch["rna"])

            outputs = model(values_batch, masks_batch)
            payload = _normalize_attention_payload(outputs.attentions)

            payload_cpu = {
                "encoder_fusion": tuple(att.detach().cpu() for att in payload["encoder_fusion"]),
                "decoder": tuple(att.detach().cpu() for att in payload["decoder"]),
                "token_intervals": {
                    "encoder_fusion": dict(payload["token_intervals"]["encoder_fusion"]),
                    "decoder": dict(payload["token_intervals"]["decoder"]),
                },
            }
            raw_attention_payloads.append(payload_cpu)

            for source in ATTN_SOURCES:
                layers = payload_cpu[source]
                if len(layers) == 0:
                    continue

                if not token_intervals[source]:
                    token_intervals[source] = dict(payload_cpu["token_intervals"].get(source, {}))

                if len(layer_sums[source]) == 0:
                    layer_sums[source] = [None] * len(layers)
                    layer_counts[source] = [0] * len(layers)
                elif len(layers) != len(layer_sums[source]):
                    raise RuntimeError(
                        f"Number of {source} attention layers changed across batches: "
                        f"{len(layer_sums[source])} -> {len(layers)}"
                    )

                for layer_idx, layer_att in enumerate(layers):
                    sample_mats = _to_sample_attention_matrices(layer_att)
                    batch_sum = sample_mats.sum(dim=0).to(torch.float64)

                    if layer_sums[source][layer_idx] is None:
                        layer_sums[source][layer_idx] = batch_sum
                    else:
                        if layer_sums[source][layer_idx].shape != batch_sum.shape:
                            raise RuntimeError(
                                f"Attention shape mismatch in {source}, layer {layer_idx + 1}: "
                                f"{tuple(layer_sums[source][layer_idx].shape)} vs {tuple(batch_sum.shape)}"
                            )
                        layer_sums[source][layer_idx] += batch_sum

                    layer_counts[source][layer_idx] += sample_mats.shape[0]

    if len(raw_attention_payloads) == 0:
        mode = "RNA=0" if zero_rna else "RNA=orig"
        raise RuntimeError(f"No attention payloads were found in outputs.attentions on the test dataset ({mode}).")

    mean_heatmaps = {source: [] for source in ATTN_SOURCES}
    for source in ATTN_SOURCES:
        if len(layer_sums[source]) == 0:
            continue

        mean_heatmaps[source] = [
            (layer_sums[source][i] / layer_counts[source][i]).to(torch.float32)
            for i in range(len(layer_sums[source]))
        ]

    if not token_intervals["decoder"]:
        token_intervals["decoder"] = _default_decoder_token_intervals(model)
    if not token_intervals["encoder_fusion"]:
        token_intervals["encoder_fusion"] = dict(token_intervals["decoder"])

    return raw_attention_payloads, mean_heatmaps, token_intervals


def _format_modality_ranges(intervals):
    parts = ["CLS[0]"]
    for modality in MODALITY_ORDER:
        if modality not in intervals:
            continue
        start, end = intervals[modality]
        parts.append(f"{MODALITY_NAMES.get(modality, modality.upper())}[{start}:{end - 1}]")
    return " | ".join(parts)


def _draw_modality_boundaries(ax, intervals, matrix_size):
    for modality in MODALITY_ORDER:
        if modality not in intervals:
            continue

        color = MODALITY_COLORS.get(modality, "white")
        start, end = intervals[modality]

        start = max(0, min(matrix_size, start))
        end = max(0, min(matrix_size, end))

        for pos in (start, end):
            line_pos = pos - 0.5
            if -0.5 <= line_pos <= matrix_size - 0.5:
                ax.axvline(line_pos, color=color, linestyle="--", linewidth=0.8, alpha=0.95)
                ax.axhline(line_pos, color=color, linestyle="--", linewidth=0.8, alpha=0.95)


def _percentile_bounds(values_1d: np.ndarray, low_q: float = 1.0, high_q: float = 99.0):
    if values_1d.size == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(values_1d, low_q))
    vmax = float(np.percentile(values_1d, high_q))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0

    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    return vmin, vmax


def _plot_attention_source_comparison(
    source_name: str,
    mean_heatmaps_present,
    mean_heatmaps_zeroed,
    intervals,
):
    if len(mean_heatmaps_present) == 0 and len(mean_heatmaps_zeroed) == 0:
        print(f"No attentions found for source='{source_name}'.")
        return

    if len(mean_heatmaps_present) != len(mean_heatmaps_zeroed):
        raise RuntimeError(
            f"Different number of {source_name} layers between RNA-present and RNA-zeroed runs: "
            f"{len(mean_heatmaps_present)} vs {len(mean_heatmaps_zeroed)}"
        )

    num_layers = len(mean_heatmaps_present)
    deltas = [mean_heatmaps_zeroed[i] - mean_heatmaps_present[i] for i in range(num_layers)]

    common_values = torch.cat([x.flatten() for x in (mean_heatmaps_present + mean_heatmaps_zeroed)]).detach().cpu().numpy()
    common_vmin, common_vmax = _percentile_bounds(common_values, low_q=1.0, high_q=99.0)

    delta_values = torch.cat([d.flatten() for d in deltas]).detach().cpu().numpy()
    delta_vmin, delta_vmax = _percentile_bounds(delta_values, low_q=1.0, high_q=99.0)
    if delta_vmax <= 0:
        delta_vmax = 1e-12
    if delta_vmin >= 0:
        delta_vmin = -1e-12

    fig, axes = plt.subplots(num_layers, 3, figsize=(15, 4.6 * num_layers))
    if num_layers == 1:
        axes = np.expand_dims(axes, axis=0)

    # Reserve right margin for colorbars and top margin for header text.
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.06, top=0.88, wspace=0.18, hspace=0.36)

    source_title = "Encoder Fusion" if source_name == "encoder_fusion" else "Decoder"
    interval_text = _format_modality_ranges(intervals)
    fig.suptitle(f"{source_title} attentions\nToken ranges: {interval_text}", fontsize=13, y=0.97)

    im0 = None
    im2 = None

    for layer_idx in range(num_layers):
        heat_present = mean_heatmaps_present[layer_idx].numpy()
        heat_zeroed = mean_heatmaps_zeroed[layer_idx].numpy()
        heat_delta = deltas[layer_idx].numpy()

        matrix_size = heat_present.shape[0]

        im0 = axes[layer_idx, 0].imshow(
            heat_present,
            cmap="viridis",
            vmin=common_vmin,
            vmax=common_vmax,
            aspect="auto",
        )
        im1 = axes[layer_idx, 1].imshow(
            heat_zeroed,
            cmap="viridis",
            vmin=common_vmin,
            vmax=common_vmax,
            aspect="auto",
        )
        im2 = axes[layer_idx, 2].imshow(
            heat_delta,
            cmap="coolwarm",
            vmin=delta_vmin,
            vmax=delta_vmax,
            aspect="auto",
        )

        axes[layer_idx, 0].set_title(f"Layer {layer_idx + 1}: RNA present", pad=8)
        axes[layer_idx, 1].set_title(f"Layer {layer_idx + 1}: RNA zeroed", pad=8)
        axes[layer_idx, 2].set_title(f"Layer {layer_idx + 1}: Zeroed - Present", pad=8)

        for col_idx in range(3):
            ax = axes[layer_idx, col_idx]
            ax.set_ylabel("Query token index")
            ax.set_xlabel("Key token index")
            _draw_modality_boundaries(ax, intervals, matrix_size)

    # Dedicated colorbar axes placed on opposite sides of the heatmap grid.
    cax_main = fig.add_axes([0.03, 0.18, 0.014, 0.62])
    cax_delta = fig.add_axes([0.94, 0.18, 0.014, 0.62])

    if im0 is not None:
        cb_main = fig.colorbar(im0, cax=cax_main)
        cb_main.set_label("Attention score")
    if im2 is not None:
        cb_delta = fig.colorbar(im2, cax=cax_delta)
        cb_delta.set_label("Delta")

    plt.show()

attention_scores_present, mean_heatmaps_present, token_intervals_present = _collect_attention_heatmaps(
    model=trainer.model,
    dataloader=trainer.dataloaders["test"],
    device=device,
    zero_rna=False,
)

attention_scores_zeroed, mean_heatmaps_zeroed, token_intervals_zeroed = _collect_attention_heatmaps(
    model=trainer.model,
    dataloader=trainer.dataloaders["test"],
    device=device,
    zero_rna=True,
)

attention_scores_test = {
    "rna_present": attention_scores_present,
    "rna_zeroed": attention_scores_zeroed,
}

mean_attention_heatmaps_test = {
    "rna_present": mean_heatmaps_present,
    "rna_zeroed": mean_heatmaps_zeroed,
}

token_intervals_attention_test = {
    "rna_present": token_intervals_present,
    "rna_zeroed": token_intervals_zeroed,
}

for _source in ATTN_SOURCES:
    _plot_attention_source_comparison(
        source_name=_source,
        mean_heatmaps_present=mean_attention_heatmaps_test["rna_present"].get(_source, []),
        mean_heatmaps_zeroed=mean_attention_heatmaps_test["rna_zeroed"].get(_source, []),
        intervals=(
            token_intervals_attention_test["rna_present"].get(_source)
            or token_intervals_attention_test["rna_zeroed"].get(_source)
            or {}
        ),
    )

mean_attention_heatmaps_test





# %%
mean_attention_heatmaps_test.keys()

# %%
torch.stack(mean_attention_heatmaps_test['rna_present']['encoder_fusion']).min(), torch.stack(mean_attention_heatmaps_test['rna_present']['encoder_fusion']).max()

# %%
torch.stack(mean_attention_heatmaps_test['rna_present']['decoder']).min(), torch.stack(mean_attention_heatmaps_test['rna_present']['decoder']).max()

# %%
torch.stack(mean_attention_heatmaps_test['rna_zeroed']['encoder_fusion']).min(), torch.stack(mean_attention_heatmaps_test['rna_zeroed']['encoder_fusion']).max()

# %%
torch.stack(mean_attention_heatmaps_test['rna_zeroed']['decoder']).min(), torch.stack(mean_attention_heatmaps_test['rna_zeroed']['decoder']).max()

# %%



