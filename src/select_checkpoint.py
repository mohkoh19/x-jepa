from typing import Any, Dict, Tuple

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.evaluation.checkpoint_selection import run_checkpoint_selection
from src.utils import RankedLogger, extras, prepare_runtime, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def select_checkpoint(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prepare_runtime(cfg)
    payload = run_checkpoint_selection(cfg)
    selected = payload["selected"] or {}
    metrics = {
        "checkpoint_selection/primary_score": selected.get("primary_score"),
        "checkpoint_selection/aggregate_score": selected.get("aggregate_score"),
        "checkpoint_selection/mean_recall": selected.get("mean_recall"),
    }
    return metrics, {"cfg": cfg, "selection": payload}


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="checkpoint_selection/default",
)
def main(cfg: DictConfig) -> None:
    extras(cfg)
    select_checkpoint(cfg)


if __name__ == "__main__":
    main()
