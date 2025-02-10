import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from run_multistage_study import run_multistage_study
from run_study import run_study
import json
from functools import reduce
import os
from src.utils import append_config_mode, get_config_mode

def run_schedule(configs_path : str) -> None:
    '''
    1. all configs should be placed in configs_path
    2. config name format: {order}_name_{optional:done}.yaml, for example: 1_num_layers.done.yaml, 2_lr.yaml
    3. configs with type "done" will not be executed, without will be
    '''
    print("START")
    while True:
        configs = sorted(os.listdir(configs_path), key=lambda x: int(x.split("_")[0]))
        done_configs = [c for c in configs if get_config_mode(c) == "done"]
        scheduled_configs = [c for c in configs if c not in done_configs]
        if len(scheduled_configs) == 0:
            break
        next_config = append_config_mode(scheduled_configs[0], "in_progress", base_path=configs_path).split(os.sep)[-1]

        print("----------------------")
        print("SCHEDULE:")
        for c in done_configs: print("DONE:", c)
        print("IN PROGRESS:", next_config)
        for c in scheduled_configs[1:]: print("SCHEDULED:", c)
        print("----------------------")
        
        c = OmegaConf.load(os.path.join(configs_path, next_config))
        if "stages" in c:
            @hydra.main(version_base=None, config_path=configs_path, config_name=next_config)
            def run_multistage_study_wrapper(cfg : DictConfig) -> None:
                return run_multistage_study(cfg)
            run_multistage_study_wrapper()
        else:
            @hydra.main(version_base=None, config_path=configs_path, config_name=next_config)
            def run_wrapper(cfg : DictConfig) -> None:
                return run_study(cfg)
            run_wrapper()
        append_config_mode(next_config, "done", base_path=configs_path)
    print("FINISH")

if __name__ == "__main__":
    run_schedule("src/configs/schedule")