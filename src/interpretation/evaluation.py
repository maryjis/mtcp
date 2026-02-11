import torch
import tqdm

try:
    from scipy.stats import ttest_rel
except ImportError:
    ttest_rel = None


def evaluate_test_reconstruction_mse(
    model,
    dataloader,
    device,
    split_name: str = "test",
    zero_rna: bool = False,
    permute_rna: bool = False,
):
    if zero_rna and permute_rna:
        raise ValueError("Choose only one RNA perturbation mode: zero_rna or permute_rna.")

    model.eval()

    total_mse = 0.0
    total_wsi_mse = 0.0
    total_samples = 0
    total_wsi_samples = 0
    mse_losses = []
    wsi_mse_losses = []

    if zero_rna:
        desc = f"{split_name} | RNA=0"
    elif permute_rna:
        desc = f"{split_name} | RNA=perm"
    else:
        desc = f"{split_name} | RNA=orig"

    with torch.no_grad():
        for data, masks in tqdm.tqdm(dataloader, desc=desc):
            data = {modality: value.to(device) for modality, value in data.items()}
            masks = {modality: value.to(device) for modality, value in masks.items()}

            if zero_rna and "rna" in data:
                data = dict(data)
                data["rna"] = torch.zeros_like(data["rna"])
            elif permute_rna and "rna" in data:
                data = dict(data)
                batch_size = data["rna"].shape[0]
                permutation = torch.randperm(batch_size, device=data["rna"].device)
                data["rna"] = data["rna"][permutation]

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
