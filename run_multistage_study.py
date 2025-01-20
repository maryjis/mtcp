import hydra
from omegaconf import DictConfig, OmegaConf
from run import run
import json

@hydra.main(version_base=None, config_path="src/configs", config_name="multistage_optuna_unimodal_config.yaml")
def run_multistage_study(cfg : DictConfig) -> None:
    if cfg.get("debug", False):
        if not OmegaConf.has_resolver("eval"): OmegaConf.register_new_resolver("eval", eval) 
        print(json.dumps(OmegaConf.to_container(cfg, resolve=True), ensure_ascii=True, indent=4))

    run(cfg["pretrain"])
    return run(cfg["task"])["c_index"]["mean"]

if __name__ == "__main__":
    run_multistage_study()