import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from src.utils import  * 
from src.unimodal.trainer import UnimodalSurvivalTrainer,  UnimodalMAETrainer
from pathlib import Path
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from src.multimodal.trainer import MultiModalMAETrainer, MultiModalSurvivalTrainer

@hydra.main(version_base=None, config_path="src/configs", config_name="unimodal_config_wsi_base")
def run(cfg : DictConfig) -> None:
    if not OmegaConf.has_resolver("eval"): OmegaConf.register_new_resolver("eval", eval) #arithmetic in config params
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.base.random_seed)
    if cfg.base.log.logging:
        init_wandb_logging(cfg)
    all_valid_metrics, all_test_metrics , all_test_metrics_in_intersection =[], [], []
    for fold_ind in range(cfg.base.splits):

        print(f"Fold #{fold_ind}")
        cfg.base.save_path = f"outputs/models/{cfg.base.experiment_name}_split_{fold_ind}.pth"
        if cfg.model.get("is_load_pretrained", False):
            with open_dict(cfg):
                print("Model path", f"outputs/models/{cfg.model.pretrained_model_name}_split_{fold_ind}.pth")
                cfg.model.pretrained_model_path = f"outputs/models/{cfg.model.pretrained_model_name}_split_{fold_ind}.pth"
        
             
        splits = load_splits(
            Path(cfg.base.data_path), 
            fold_ind, 
            cfg.base.remove_nan_column, 
            max_samples_per_split=cfg.base.get("max_samples_per_split", None),
            multimodal_intersection_test =cfg.base.get("multimodal_intersection_test", None),
            modalities=cfg.base.modalities
        )

        if cfg.base.type == 'unimodal':

            # унимодальные (тут мы должны выбрать модальность) или мультимодальный + способ дообучения
            if cfg.base.strategy == "survival":
                trainer = UnimodalSurvivalTrainer(splits, cfg)
            elif cfg.base.strategy == "mae": 
                trainer = UnimodalMAETrainer(splits, cfg)
            else:
                raise NotImplementedError(f"Such strategy - {cfg.base.strategy} isn't implemented in unimodal approach.")
        elif cfg.base.type == 'multimodal':
              
            cfg = add_model_paths_to_config(cfg,fold_ind)                      
            if cfg.base.strategy == "mae": 
                trainer = MultiModalMAETrainer(splits, cfg)
            elif cfg.base.strategy == "survival":
                  trainer = MultiModalSurvivalTrainer(splits, cfg)
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
        wandb.summary["final"] =final_metrics
        wandb.finish()

    return final_test_metrics

if __name__ == "__main__":
    run()