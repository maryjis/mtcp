import hydra
from omegaconf import DictConfig
from run import run

@hydra.main(version_base=None, config_path="src/configs", config_name="multimodal_dna_rna_sceduler")
def run_study(cfg : DictConfig) -> None:
    if cfg.base.strategy == "mae":
        return run(cfg)["mse_loss"]["mean"]
    else:
        return run(cfg)["c_index"]["mean"]

if __name__ == "__main__":
    run_study()