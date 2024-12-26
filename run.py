import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from src.utils import  * 
from src.unimodal.trainer import UnimodalSurvivalTrainer,  UnimodalMAETrainer
from pathlib import Path
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig

@hydra.main(version_base=None, config_path="src/configs", config_name="unimodal_config")
def run(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.base.random_seed)
    if cfg.base.log.logging:
        init_wandb_logging(cfg.base.log)
    all_valid_metrics, all_test_metrics =[], []
    for fold_ind in range(cfg.base.splits):
        print(f"Fold #{fold_ind}")
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

        if cfg.base.type == 'unimodal':
            # унимодальные (тут мы должны выбрать модальность) или мультимодальный + способ дообучения
            if cfg.base.strategy == "survival":
                trainer = UnimodalSurvivalTrainer(splits, cfg)
            elif cfg.base.strategy == "mae": 
                trainer = UnimodalMAETrainer(splits, cfg)
            else:
                raise NotImplementedError(f"Such strategy - {cfg.base.strategy} isn't implemented.")   
        else:
            raise NotImplementedError("Now only unimodal training is implemented.")

        valid_metrics =trainer.train(fold_ind)
        test_metrics =trainer.evaluate(fold_ind)
        all_valid_metrics.append(valid_metrics)
        all_test_metrics.append(test_metrics)
        
    # aggregate valid and test metrics for all folds
    final_valid_metrics = agg_fold_metrics(all_valid_metrics)
    final_test_metrics = agg_fold_metrics(all_test_metrics)
    
    if cfg.base.log.logging:
        wandb.summary["final"] = {"valid": final_valid_metrics, "test": final_test_metrics}
        wandb.finish()


if __name__ == "__main__":
    run()