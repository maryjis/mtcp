# Standard libraries
import argparse
import os

# Third-party libraries
import torch
import wandb
import numpy as np
from imutils import paths

# Local dependencies
from src.utils import seed_everything, seed_worker
from src.logger import logger
from src.unimodal.wsi.transforms import contrastive_wsi_transforms
from src.unimodal.wsi.datasets import PatchDataset
from src.unimodal.wsi.models import ResNetWrapperSimCLR
from src.unimodal.commons.losses import ContrastiveLoss
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

import psutil
import time

import pynvml

def log_resource_usage():
    """Logs CPU, RAM, and GPU usage."""
    cpu_usage = psutil.cpu_percent(interval=0.1)
    ram_usage = psutil.virtual_memory().used / 1e9  # В ГБ
    total_ram = psutil.virtual_memory().total / 1e9  # В ГБ
    gpu_stats = []
    
    if torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            for i in range(torch.cuda.device_count()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                
                gpu_stats.append({
                    "gpu_id": i,
                    "gpu_mem_used": mem_info.used / 1e9,  # В ГБ
                    "gpu_mem_total": mem_info.total / 1e9,  # В ГБ
                    "gpu_util": gpu_util  # Использование в %
                })
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error(f"Failed to retrieve GPU stats: {e}")
    
    logger.info(f"CPU: {cpu_usage:.2f}%, RAM: {ram_usage:.2f}/{total_ram:.2f} GB, GPU: {gpu_stats}")

def training_loop_contrastive_with_saving(
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
        
        # Логирование загрузки ресурсов перед началом эпохи
        log_resource_usage()
        
        model.train()
        metrics["lr"].append(optimizer.state_dict()["param_groups"][0]["lr"])
        epoch_loss = 0.0
        top_1, top_k, mean_pos = 0.0, 0.0, 0.0
        
        start_time = time.time()
        
        for batch_idx, batch_data in enumerate(tqdm.tqdm(train_dl, desc="Training...", total=len(train_dl))):
            if isinstance(batch_data, dict):
                inputs, inputs_2 = batch_data["image"].to(device), batch_data["image_2"].to(device)
            else:
                inputs, inputs_2 = batch_data[0].to(device), batch_data[1].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            outputs_2, _ = model(inputs_2)
            loss, acc_top_1, acc_top_k, acc_mean_pos = loss_fn(outputs, outputs_2)
            loss.backward()
            optimizer.step()
            
            top_1 += acc_top_1 * inputs.shape[0]
            top_k += acc_top_k * inputs.shape[0]
            mean_pos += acc_mean_pos * inputs.shape[0]
            epoch_loss += loss.item() * inputs.shape[0]

            # Логирование каждые 50 батчей
            if batch_idx % 50 == 0:
                log_resource_usage()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1} completed in {elapsed_time:.2f} seconds.")

        epoch_loss /= len(train_dl.dataset)
        logger.info(f"Training loss: {epoch_loss:.4f}")
        
        metrics["train/loss"].append(epoch_loss)
        metrics["train/top_1"].append((top_1 / len(train_dl.dataset)).item())
        metrics[f"train/top_{k}"].append((top_k / len(train_dl.dataset)).item())
        metrics["train/mean_pos"].append((mean_pos / len(train_dl.dataset)).item())
        
        # Validation
        model.eval()
        val_loss = 0.0
        top_1, top_k, mean_pos = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch_data in tqdm.tqdm(valid_dl, desc="Validation..."):
                if isinstance(batch_data, dict):
                    inputs, inputs_2 = batch_data["image"].to(device), batch_data["image_2"].to(device)
                else:
                    inputs, inputs_2 = batch_data[0].to(device), batch_data[1].to(device)
                
                outputs, _ = model(inputs)
                outputs_2, _ = model(inputs_2)
                loss, acc_top_1, acc_top_k, acc_mean_pos = loss_fn(outputs, outputs_2)
                
                top_1 += acc_top_1 * inputs.shape[0]
                top_k += acc_top_k * inputs.shape[0]
                mean_pos += acc_mean_pos * inputs.shape[0]
                val_loss += loss.item() * inputs.shape[0]
        
        val_loss /= len(valid_dl.dataset)
        logger.info(f"Validation loss: {val_loss:.4f}")
        
        metrics["val/top_1"].append((top_1 / len(valid_dl.dataset)).item())
        metrics[f"val/top_{k}"].append((top_k / len(valid_dl.dataset)).item())
        metrics["val/mean_pos"].append((mean_pos / len(valid_dl.dataset)).item())
        metrics["val/loss"].append(val_loss)
        
        scheduler.step()

        # Сохранение модели
        epoch_model_path = path_to_save.replace(".pth", f"_epoch_{epoch + 1}.pth")
        os.makedirs(os.path.dirname(epoch_model_path), exist_ok=True)
        torch.save(model.state_dict(), epoch_model_path)
        logger.info(f"Model saved at {epoch_model_path}")

        if metrics["val/loss"][-1] < best_loss:
            best_loss = metrics["val/loss"][-1]
            torch.save(model.state_dict(), path_to_save)
            logger.info(f"Best model updated at {path_to_save}")

        if wandb_logging:
            try:
                wandb.log({k: v[-1] for k, v in metrics.items()})
            except Exception as e:
                logger.info(f"Metrics haven't been logged in wandb due to {e}")
                logger.info({k: v[-1] for k, v in metrics.items()})

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        nargs="+",
        default=["/mnt/public-datasets/drim/TCGA-GBM_WSI", "/mnt/public-datasets/drim/wsi"],
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--project", type=str, default="cancer_mtcp")
    parser.add_argument("--n_cpus", type=int, default=40)
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1999)
    parser.add_argument("--model_path", type=str, default="/home/a.beliaeva/mtcp/src/outputs/models/main_base/wsi_encoder.pth")
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--entity", type=str, default=None)  # Добавляем entity для wandb
    parser.add_argument("--wandb_project", type=str, default="cancer_mtcp")  # Добавляем проект для wandb
    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)
    
    # Get all filepaths
    filepaths = []
    for path in args.data_path:
        temp_filepaths = list(paths.list_images(path))
        temp_filepaths = [
            filepath for filepath in temp_filepaths if "patches" in filepath
        ]
        # Группируем файлы по папкам и оставляем только первые 100
        grouped_filepaths = {}
        for filepath in temp_filepaths:
            folder = os.path.dirname(filepath)
            if folder not in grouped_filepaths:
                grouped_filepaths[folder] = []
            grouped_filepaths[folder].append(filepath)
        
        for folder, files in grouped_filepaths.items():
            files = sorted(files)  # Сортируем файлы в каждой папке
            filepaths.extend(files[:100])  # Оставляем только первые 10 файлов

    filepaths = np.array(filepaths)


    # Split into train and val
    train_idx = np.random.choice(
        np.arange(len(filepaths)), int(len(filepaths) * 0.75), replace=False
    )
    train_mask = np.zeros(len(filepaths), dtype=bool)
    train_mask[train_idx] = True
    train_filepaths = filepaths[train_idx]
    val_filepaths = filepaths[~train_mask]

    logger.info("Performing contrastive training on TCGA dataset.")
    logger.info("Number of training images : {}.", len(train_filepaths))
    logger.info("Number of validation images : {}.", len(val_filepaths))

    train_dataset = PatchDataset(train_filepaths, transform=contrastive_wsi_transforms)
    val_dataset = PatchDataset(val_filepaths, transform=contrastive_wsi_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        num_workers=4, 
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        num_workers=4, 
    )

    model = ResNetWrapperSimCLR(out_dim=args.out_dim, projection_head=True)

    if args.pretrained_weights:
        if os.path.isfile(args.pretrained_weights):
            logger.info(f"Loading pretrained weights from {args.pretrained_weights}")
            state_dict = torch.load(args.pretrained_weights, weights_only=True)
            # Удалить префикс `module.`, если он есть
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # Загрузить только совпадающие ключи
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.error(f"Pretrained weights not found at {args.pretrained_weights}")
            exit(1)

    logger.info(
        "Number of parameters : {}",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.n_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.n_gpus)))

    model.to(device)
    logger.info("Using {} gpus to train the model", args.n_gpus)

    contrastive_loss = ContrastiveLoss(temperature=args.temperature, k=args.k)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=40, eta_min=5e-6
    )

    # Initialize wandb for logging
    if args.entity:
        wandb.init(
            project=args.wandb_project,
            entity=args.entity,
            config=vars(args),
        )
        logger.info("Wandb logging initialized!")

    logger.info("Training started!")
    _, _ = training_loop_contrastive_with_saving(
        model,
        args.epochs,
        contrastive_loss,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        device,
        args.model_path,  # Save model to the specified path
        wandb_logging=True,  # Enable wandb logging
        k=args.k,
    )

    if args.entity:
        wandb.finish()  # End the wandb run

    logger.info("Training finished!")