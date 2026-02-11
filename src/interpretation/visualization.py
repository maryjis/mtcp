import numpy as np
import matplotlib.pyplot as plt
import torch

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
    permute_rna: bool = False,
    seed: int = 2026,
) -> tuple[torch.Tensor, torch.Tensor]:
    if zero_rna and permute_rna:
        raise ValueError("Choose only one RNA perturbation mode: zero_rna or permute_rna.")

    model_inputs = {k: v.clone().to(device) for k, v in values_batch.items()}
    model_masks = {k: v.clone().to(device) for k, v in masks_batch.items()}

    trainer.model.eval()
    with torch.no_grad():
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if zero_rna and "rna" in model_inputs:
            model_inputs["rna"] = torch.zeros_like(model_inputs["rna"])
        elif permute_rna and "rna" in model_inputs:
            batch_size = model_inputs["rna"].shape[0]
            permutation = torch.randperm(batch_size, device=model_inputs["rna"].device)
            model_inputs["rna"] = model_inputs["rna"][permutation]

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
    recon_with_permuted_rna: torch.Tensor,
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
    recon_permuted_patch = recon_with_permuted_rna[sample_idx]

    diff_zero_map = (
        (recon_present_patch.detach().cpu().float() - recon_zero_patch.detach().cpu().float())
        # .abs()
        .mean(dim=0)
        .numpy()
    )
    diff_permuted_map = (
        (recon_present_patch.detach().cpu().float() - recon_permuted_patch.detach().cpu().float())
        # .abs()
        .mean(dim=0)
        .numpy()
    )
    pixel_mask = _build_wsi_pixel_mask(trainer=trainer, wsi_patch_mask=wsi_patch_mask[sample_idx]).cpu().numpy()

    diff_zero_masked = diff_zero_map * pixel_mask
    diff_zero_unmasked = diff_zero_map * (1.0 - pixel_mask)
    diff_permuted_masked = diff_permuted_map * pixel_mask
    diff_permuted_unmasked = diff_permuted_map * (1.0 - pixel_mask)

    return {
        "label": sample_label,
        "img_orig": _to_rgb_image(original_wsi_patch),
        "img_masked": _to_rgb_image(masked_input_wsi_patch),
        "img_rna": _to_rgb_image(recon_present_patch),
        "img_zero": _to_rgb_image(recon_zero_patch),
        "img_permuted": _to_rgb_image(recon_permuted_patch),
        "diff_zero_full": diff_zero_map,
        "diff_zero_masked": diff_zero_masked,
        "diff_zero_unmasked": diff_zero_unmasked,
        "diff_permuted_full": diff_permuted_map,
        "diff_permuted_masked": diff_permuted_masked,
        "diff_permuted_unmasked": diff_permuted_unmasked,
    }


def _collect_rows(
    trainer,
    device: str,
    dataloader,
    split_name: str = "test",
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
        recon_with_permuted_rna, _ = _run_wsi_reconstruction(
            trainer=trainer,
            device=device,
            values_batch=values_batch,
            masks_batch=masks_batch,
            permute_rna=True,
            seed=batch_seed,
        )

        for sample_idx in sample_indices:
            rows.append(
                _build_row_data(
                    trainer=trainer,
                    values_batch=values_batch,
                    recon_with_rna=recon_with_rna,
                    recon_with_zero_rna=recon_with_zero_rna,
                    recon_with_permuted_rna=recon_with_permuted_rna,
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
            f"Increase batch size or ensure enough valid WSI samples in '{split_name}' split."
        )

    return rows


def _plot_wsi_comparison_grid(rows: list[dict]) -> None:
    n_rows = len(rows)
    n_cols = 8

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    vmax = max(
        max(float(row["diff_zero_full"].max()), float(row["diff_permuted_full"].max()))
        for row in rows
    )
    vmax = max(vmax, 1e-8)

    titles = [
        "Original WSI patch",
        "Original masked\n(before model)",
        "Recon (RNA present)",
        "Recon (RNA zeroed)",
        "Recon (RNA permuted)",
        "Present - Zeroed",
        "Present - Permuted",
        "Permuted - Present",
        # "|Present - Zeroed|\nmasked patches",
        # "|Present - Zeroed|\nunmasked patches",
        # "|Present - Permuted|\nmasked patches",
        # "|Present - Permuted|\nunmasked patches",
    ]

    for col, title in enumerate(titles):
        axes[0, col].set_title(title)

    mappable = None
    for row_idx, row in enumerate(rows):
        axes[row_idx, 0].imshow(row["img_orig"])
        axes[row_idx, 1].imshow(row["img_masked"])
        axes[row_idx, 2].imshow(row["img_rna"])
        axes[row_idx, 3].imshow(row["img_zero"])
        axes[row_idx, 4].imshow(row["img_permuted"])
        axes[row_idx, 5].imshow(row["diff_zero_full"], cmap="coolwarm", vmin=0.0, vmax=vmax)
        mappable = axes[row_idx, 6].imshow(row["diff_permuted_full"], cmap="coolwarm", vmin=0.0, vmax=vmax)
        axes[row_idx, 7].imshow(-row["diff_permuted_full"], cmap="coolwarm", vmin=0.0, vmax=vmax)
        # axes[row_idx, 6].imshow(row["diff_zero_masked"], cmap="magma", vmin=0.0, vmax=vmax)
        # axes[row_idx, 7].imshow(row["diff_zero_unmasked"], cmap="magma", vmin=0.0, vmax=vmax)
        # axes[row_idx, 9].imshow(row["diff_permuted_masked"], cmap="magma", vmin=0.0, vmax=vmax)
        # mappable = axes[row_idx, 10].imshow(row["diff_permuted_unmasked"], cmap="magma", vmin=0.0, vmax=vmax)

        axes[row_idx, 0].set_ylabel(row["label"], rotation=0, labelpad=52, va="center")

        for col in range(n_cols):
            axes[row_idx, col].axis("off")

    if mappable is not None:
        fig.colorbar(mappable, ax=axes[:, 5:].ravel().tolist(), fraction=0.015, pad=0.01)

    plt.tight_layout()
    plt.show()