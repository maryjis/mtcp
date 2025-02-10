import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf, open_dict
from src.utils import seed_everything, load_splits, agg_fold_metrics, init_wandb_logging
from src.unimodal.trainer import UnimodalSurvivalTrainer, UnimodalMAETrainer
from src.multimodal.trainer import MultiModalMAETrainer, MultiModalSurvivalTrainer
from pathlib import Path
import wandb
import queue

def train_fold(fold_ind, cfg, device, log_queue):
    """
    Обучение одного фолда в отдельном процессе.
    """
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

    # Добавляем проверку типов перед отправкой в очередь
    def process_metrics(metrics):
        """ Обрабатывает метрики перед записью в очередь """
        if isinstance(metrics, tuple):
            if len(metrics) == 2 and isinstance(metrics[0], dict) and metrics[1] is None:
                return metrics[0]  # Берем только первый элемент (словарь)
            else:
                raise ValueError(f"Unexpected format of metrics: {metrics}")
        return metrics  # Если это не кортеж, возвращаем как есть

    log_queue.put(("valid", fold_ind, process_metrics(valid_metrics)))
    log_queue.put(("test", fold_ind, process_metrics(test_metrics)))
    log_queue.put(("done", fold_ind, None))

    wandb.finish()  # Завершаем сеанс W&B в процессе

@hydra.main(version_base=None, config_path="src/configs", config_name="unimodal_config_wsi_mae")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    num_folds = cfg.base.splits
    available_gpus = cfg.base.get("available_gpus", [0, 1, 2, 3, 4])  # Список доступных GPU

    print(f"Available GPUs: {available_gpus}, running {num_folds} folds in parallel.")

    log_queue = mp.Queue()  # Очередь для сбора метрик
    processes = []

    for fold_ind in range(num_folds):
        device = available_gpus[fold_ind % len(available_gpus)]
        p = mp.Process(target=train_fold, args=(fold_ind, cfg, device, log_queue))
        p.start()
        processes.append(p)

    all_valid_metrics = []
    all_test_metrics = []
    finished_folds = 0  # Счетчик завершенных процессов

    # ✅ **Логирование W&B в главном процессе**
    if cfg.base.log.logging:
        wandb.init(
            project=cfg.base.log.wandb_project,
            name=cfg.base.log.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True
        )

    while finished_folds < num_folds:
        try:
            metric_type, fold_ind, metrics = log_queue.get(timeout=10)  # Ждём данные
            if metrics is not None and isinstance(metrics, tuple):
                metrics = dict(metrics)  # Преобразуем tuple в dict при необходимости
            if metric_type == "valid":
                all_valid_metrics.append(metrics)
                wandb.log({f"valid/fold_{fold_ind}/{key}": value for key, value in metrics.items()})
            elif metric_type == "test":
                all_test_metrics.append(metrics)
                wandb.log({f"test/fold_{fold_ind}/{key}": value for key, value in metrics.items()})
            elif metric_type == "done":
                finished_folds += 1  # Увеличиваем счётчик завершенных процессов
        except queue.Empty:
            pass  # Просто ждем

    for p in processes:
        p.join()

    # **Финальные метрики**
    final_valid_metrics = agg_fold_metrics(all_valid_metrics)
    final_test_metrics = agg_fold_metrics(all_test_metrics)

    # **Финальное логирование**
    if cfg.base.log.logging:
        wandb.summary["final"] = {"valid": final_valid_metrics, "test": final_test_metrics}
        wandb.finish()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # ✅ Исправляем баг с дочерними процессами
    run()