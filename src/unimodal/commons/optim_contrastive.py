# Standard libraries
import os
from collections import defaultdict
from typing import Tuple

# Third party libraries
import torch
import wandb
import tqdm
import numpy as np
from torch.profiler import profile, ProfilerActivity

# Local libraries
from src.logger import logger

def training_loop_contrastive(
    model: torch.nn.Module,
    epochs: int,
    loss_fn: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    device: torch.device,
    path_to_save: str,
    wandb_logging: bool,
    k: int = 3,
) -> Tuple[torch.nn.Module, dict]:
    metrics = defaultdict(list)
    best_loss = np.infty
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info("-" * 10)
        model.train()
        metrics["lr"].append(optimizer.state_dict()["param_groups"][0]["lr"])
        epoch_loss = 0.0
        top_1, top_k, mean_pos = 0.0, 0.0, 0.0

        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            for batch_data in tqdm.tqdm(train_dl, desc="Training...", total=len(train_dl)):
                if isinstance(batch_data, dict):
                    inputs, inputs_2 = (
                        batch_data["image"].to(device),
                        batch_data["image_2"].to(device),
                    )
                else:
                    inputs, inputs_2 = batch_data[0].to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                outputs_2, _ = model(inputs_2)
                loss, acc_top_1, acc_top_k, acc_mean_pos = loss_fn(outputs, outputs_2)
                top_1 += acc_top_1 * inputs.shape[0]
                top_k += acc_top_k * inputs.shape[0]
                mean_pos += acc_mean_pos * inputs.shape[0]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.shape[0]

        ### debug
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        prof.export_chrome_trace(f"outputs/traces/trace_{epoch}_train.json")

        epoch_loss /= len(train_dl.dataset)
        logger.info(f"Training loss: {epoch_loss:.4f}")

        metrics["train/loss"].append(epoch_loss)
        metrics["train/top_1"].append((top_1 / len(train_dl.dataset)).item())
        metrics[f"train/top_{k}"].append((top_k / len(train_dl.dataset)).item())
        metrics["train/mean_pos"].append((mean_pos / len(train_dl.dataset)).item())
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            top_1, top_k, mean_pos = 0.0, 0.0, 0.0

            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                for batch_data in tqdm.tqdm(valid_dl, desc="Validation..."):
                    if isinstance(batch_data, dict):
                        inputs, inputs_2 = (
                            batch_data["image"].to(device),
                            batch_data["image_2"].to(device),
                        )
                    else:
                        inputs, inputs_2 = batch_data[0].to(device), batch_data[1].to(
                            device
                        )
                    outputs, _ = model(inputs)
                    outputs_2, _ = model(inputs_2)

                    loss, acc_top_1, acc_top_k, acc_mean_pos = loss_fn(outputs, outputs_2)
                    top_1 += acc_top_1 * inputs.shape[0]
                    top_k += acc_top_k * inputs.shape[0]
                    mean_pos += acc_mean_pos * inputs.shape[0]
                    val_loss += loss.item() * inputs.shape[0]

            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
            prof.export_chrome_trace(f"outputs/traces/trace_{epoch}_val.json")

        val_loss /= len(valid_dl.dataset)
        logger.info(f"Validation loss: {val_loss:.4f}")
        metrics["val/top_1"].append((top_1 / len(valid_dl.dataset)).item())
        metrics[f"val/top_{k}"].append((top_k / len(valid_dl.dataset)).item())
        metrics["val/mean_pos"].append((mean_pos / len(valid_dl.dataset)).item())
        metrics["val/loss"].append(val_loss)

        scheduler.step()
        if metrics["val/loss"][-1] < best_loss:
            best_loss = metrics["val/loss"][-1]
            dir_to_save = os.path.join(*path_to_save.split(os.sep)[:-1])
            if not os.path.exists(dir_to_save): os.makedirs(dir_to_save, exist_ok=True)
            torch.save(model.state_dict(), path_to_save)

        if wandb_logging:
            try:
                wandb.log({k: v[-1] for k, v in metrics.items()})

            except Exception as e:
                logger.info(
                    f"Metrics haven't been logged in wandb due to {e}"
                )
                logger.info({k: v[-1] for k, v in metrics.items()})

    return model, metrics
