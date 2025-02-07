import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf, open_dict
from src.utils import seed_everything, load_splits, agg_fold_metrics
from src.unimodal.trainer import UnimodalSurvivalTrainer, UnimodalMAETrainer
from src.multimodal.trainer import MultiModalMAETrainer, MultiModalSurvivalTrainer
from pathlib import Path
import wandb
import queue
import time
from collections import defaultdict

MAX_FOLDS_PER_GPU = 2  # ✅ Не более 2 фолдов на 1 GPU


def train_fold(fold_ind, cfg, device, log_queue, gpu_usage):
    """
    Обучение одного фолда в отдельном процессе.
    """
    try:
        torch.cuda.set_device(device)
        print(f"Fold #{fold_ind} running on GPU {device}")

        seed_everything(cfg.base.random_seed + fold_ind)

        with open_dict(cfg):
            cfg.base.device = f"cuda:{device}"
            cfg.base.save_path = f"outputs/models/{cfg.base.experiment_name}_split_{fold_ind}.pth"

        if cfg.model.get("is_load_pretrained", False):
            with open_dict(cfg):
                cfg.model.pretrained_model_path = f"outputs/models/{cfg.model.pretrained_model_name}_split_{fold_ind}.pth"

        splits = load_splits(
            Path(cfg.base.data_path), 
            fold_ind, 
            cfg.base.remove_nan_column, 
            max_samples_per_split=cfg.base.get("max_samples_per_split", None)
        )

        # Выбор тренера
        if cfg.base.type == 'unimodal':
            trainer_cls = UnimodalSurvivalTrainer if cfg.base.strategy == "survival" else UnimodalMAETrainer
        elif cfg.base.type == 'multimodal':
            trainer_cls = MultiModalSurvivalTrainer if cfg.base.strategy == "survival" else MultiModalMAETrainer
        else:
            raise NotImplementedError(f"Unknown base type: {cfg.base.type}")

        trainer = trainer_cls(splits, cfg)

        # ✅ **Инициализация W&B в каждом процессе**
        if cfg.base.log.logging:
            wandb.init(
                project=cfg.base.log.wandb_project,
                name=f"{cfg.base.log.wandb_run_name}_fold_{fold_ind}",
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True
            )

        valid_metrics = trainer.train(fold_ind)
        test_metrics = trainer.evaluate(fold_ind)

        # 🔥 **Отправляем метрики в главную очередь**
        log_queue.put(("valid", fold_ind, valid_metrics))
        log_queue.put(("test", fold_ind, test_metrics))
        log_queue.put(("done", fold_ind, None))  # Сигнал завершения процесса

        wandb.finish()  # Завершаем сеанс W&B в процессе

    finally:
        gpu_usage[device] -= 1  # ✅ Освобождаем GPU после завершения


@hydra.main(version_base=None, config_path="src/configs", config_name="unimodal_config_wsi_mae")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    num_folds = cfg.base.splits
    available_gpus = cfg.base.get("available_gpus", [0, 1, 2, 3])  # Доступные GPU

    print(f"Available GPUs: {available_gpus}, running {num_folds} folds dynamically.")

    log_queue = mp.Queue()  # Очередь для сбора метрик
    gpu_usage = {gpu: 0 for gpu in available_gpus}  # ✅ Отслеживаем загрузку GPU (0 - свободен)

    all_valid_metrics = []
    all_test_metrics = []
    finished_folds = 0  # Количество завершённых фолдов
    processes = {}

    # ✅ **Логирование W&B в главном процессе**
    if cfg.base.log.logging:
        wandb.init(
            project=cfg.base.log.wandb_project,
            name=cfg.base.log.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True
        )

    # 🟢 **Запускаем первые фолды**
    for fold_ind in range(min(num_folds, len(available_gpus))):
        best_gpu = min(gpu_usage, key=gpu_usage.get)  # Выбираем самый свободный GPU
        gpu_usage[best_gpu] += 1
        p = mp.Process(target=train_fold, args=(fold_ind, cfg, best_gpu, log_queue, gpu_usage))
        p.start()
        processes[fold_ind] = p

    while finished_folds < num_folds:
        try:
            metric_type, fold_ind, metrics = log_queue.get(timeout=10)  # Ждём данные
            if metric_type == "valid":
                all_valid_metrics.append(metrics)
                wandb.log({f"valid/fold_{fold_ind}/{key}": value for key, value in metrics.items()})
            elif metric_type == "test":
                all_test_metrics.append(metrics)
                wandb.log({f"test/fold_{fold_ind}/{key}": value for key, value in metrics.items()})
            elif metric_type == "done":
                finished_folds += 1
                processes[fold_ind].join()
                del processes[fold_ind]  # Удаляем завершенный процесс из списка

                # 🆕 **Запускаем следующий фолд на самом свободном GPU**
                if finished_folds + len(processes) < num_folds:
                    next_fold = finished_folds + len(processes)
                    best_gpu = min(gpu_usage, key=gpu_usage.get)  # Выбираем самый свободный GPU
                    if gpu_usage[best_gpu] < MAX_FOLDS_PER_GPU:  # ✅ Не больше 2 фолдов на GPU
                        gpu_usage[best_gpu] += 1
                        p = mp.Process(target=train_fold, args=(next_fold, cfg, best_gpu, log_queue, gpu_usage))
                        p.start()
                        processes[next_fold] = p

        except queue.Empty:
            pass  # Просто ждем

    for p in processes.values():
        p.join()

    final_valid_metrics = agg_fold_metrics(all_valid_metrics)
    final_test_metrics = agg_fold_metrics(all_test_metrics)

    if cfg.base.log.logging:
        wandb.summary["final"] = {"valid": final_valid_metrics, "test": final_test_metrics}
        wandb.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run()
