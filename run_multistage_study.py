import hydra
from omegaconf import DictConfig, OmegaConf
from run import run
import json
from functools import reduce

@hydra.main(version_base=None, config_path="src/configs", config_name="multistage_optuna_unimodal_config.yaml")
def run_multistage_study(cfg : DictConfig) -> None:
    if cfg.get("debug", False):
        if not OmegaConf.has_resolver("eval"): OmegaConf.register_new_resolver("eval", eval) 
        print(json.dumps(OmegaConf.to_container(cfg, resolve=True), ensure_ascii=True, indent=4))

    metrics = None
    last_name = None
    for stage_name, stage_cfg in cfg["stages"].items():
        last_name = stage_name
        metrics = run(stage_cfg)
    return reduce(dict.get, cfg["stages"][last_name]["metric"].split("."), metrics)

if __name__ == "__main__":
    run_multistage_study()