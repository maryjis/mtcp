import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from src.utils import  * 
from src.unimodal.trainer import UnimodalSurvivalTrainer,  UnimodalMAETrainer
from pathlib import Path
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from src.multimodal.trainer import MultiModalMAETrainer, MultiModalSurvivalTrainer
from src.multimodal.ensemble_train import MultimodalBoostingSurvivalTrainer
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Включаем подробный уровень
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("train_debug.log"),  # лог в файл
        logging.StreamHandler()  # лог в stdout
    ]
)

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="src/configs", config_name="multimodal_titan_config")
def run(cfg : DictConfig) -> None:
    if not OmegaConf.has_resolver("eval"): OmegaConf.register_new_resolver("eval", eval) #arithmetic in config params
    logger.info(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.base.random_seed)
    tracker = None
    if cfg.base.log.logging:
        if cfg.base.log.logging_platform == "wandb":
            tracker = WandbExperimentTracker(cfg)
        elif cfg.base.log.logging_platform == "neptune":
            tracker = NeptuneExperimentTracker(cfg)
        else:
            raise NotImplementedError(f"Such logging platform - {cfg.base.log.logging_platform} isn't implemented")
        
    all_valid_metrics, all_test_metrics , all_test_metrics_in_intersection =[], [], []
    for fold_ind in range(cfg.base.splits):
        logger.info(f"Fold #{fold_ind}")
        cfg.base.save_path = f"outputs/models/{cfg.base.experiment_name}_split_{fold_ind}.pth"
        if cfg.model.get("is_load_pretrained", False):
            with open_dict(cfg):
                logger.info(f"Model path outputs/models/{cfg.model.pretrained_model_name}_split_{fold_ind}.pth")
                cfg.model.pretrained_model_path = f"outputs/models/{cfg.model.pretrained_model_name}_split_{fold_ind}.pth"
        
        logger.info("Define splits: ")     
        splits = load_splits(
            Path(cfg.base.data_path), 
            fold_ind, 
            cfg.base.remove_nan_column, 
            max_samples_per_split=cfg.base.get("max_samples_per_split", None),
            multimodal_intersection_test =cfg.base.get("multimodal_intersection_test", None),
            modalities=cfg.base.modalities,
            project_ids = cfg.base.get("project_ids", None),
        )

        if cfg.base.type == 'unimodal':

            if cfg.base.strategy == "survival":
                trainer = UnimodalSurvivalTrainer(splits, cfg, tracker, fold_ind)
            elif cfg.base.strategy == "mae": 
                trainer = UnimodalMAETrainer(splits, cfg, tracker, fold_ind)
            else:
                raise NotImplementedError(f"Such strategy - {cfg.base.strategy} isn't implemented in unimodal approach.")
        elif cfg.base.type == 'multimodal':
              
            cfg = add_model_paths_to_config(cfg,fold_ind)                      
            if cfg.base.strategy == "mae": 
                trainer = MultiModalMAETrainer(splits, cfg, tracker, fold_ind)
            elif cfg.base.strategy == "survival":
                  trainer = MultiModalSurvivalTrainer(splits, cfg, tracker, fold_ind)
            elif cfg.base.strategy == "boosting_survival":
                trainer = MultimodalBoostingSurvivalTrainer(splits, cfg, tracker, fold_ind)
            else:
                raise NotImplementedError(f"Such strategy - {cfg.base.strategy} isn't implemented in multimodal approach.")
        else:
            raise NotImplementedError("Choose from 'multimodal' and 'unimodal' options")

        valid_metrics =trainer.train(fold_ind)
        test_metrics, test_metrics_intersection =trainer.evaluate(fold_ind)
        all_valid_metrics.append(valid_metrics)
        all_test_metrics.append(test_metrics)
        if cfg.base.get("multimodal_intersection_test", None):
            all_test_metrics_in_intersection.append(test_metrics_intersection)
        
    # aggregate valid and test metrics for all folds
    final_valid_metrics = agg_fold_metrics(all_valid_metrics)
    final_test_metrics = agg_fold_metrics(all_test_metrics)
    if cfg.base.get("multimodal_intersection_test", None):
        final_test_metrics_intersection = agg_fold_metrics(all_test_metrics_in_intersection)
    
    if cfg.base.log.logging:
        final_metrics = {"valid": final_valid_metrics, "test": final_test_metrics}
        if cfg.base.get("multimodal_intersection_test", None):
            final_metrics.update({"test_in_intersection": final_test_metrics_intersection})
        tracker.log_summary(final_metrics)
        tracker.finish()
    logger.info(f"Final_test_metrics: {final_test_metrics}")
    return final_test_metrics

if __name__ == "__main__":
    run()
