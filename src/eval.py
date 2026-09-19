from typing import Any, Dict, Tuple

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.evaluation.logging import log_evaluation_metrics
from src.utils import (
    RankedLogger,
    build_evaluation_context,
    extras,
    prepare_runtime,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    prepare_runtime(cfg)
    object_dict = build_evaluation_context(cfg)
    datamodule = object_dict["datamodule"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    evaluation_cfg = cfg.get("evaluation") or {}
    if evaluation_cfg.get("log_hparams", False) and trainer.logger:
        from src.utils import maybe_log_hyperparameters

        maybe_log_hyperparameters(object_dict)

    if hasattr(model, "run_evaluation"):
        log.info("Running custom evaluation!")
        metric_dict = model.run_evaluation()
    else:
        lightning_ckpt_path = evaluation_cfg.get("lightning_ckpt_path")

        if evaluation_cfg.get("fit", False):
            log.info("Starting evaluation fit!")
            trainer.fit(
                model=model,
                datamodule=datamodule,
                ckpt_path=lightning_ckpt_path,
            )

        metric_dict = {}
        if evaluation_cfg.get("validate", True):
            log.info("Starting evaluation validation!")
            validate_ckpt_path = None if evaluation_cfg.get("fit", False) else lightning_ckpt_path
            validate_metrics = trainer.validate(
                model=model,
                datamodule=datamodule,
                ckpt_path=validate_ckpt_path,
            )
            if validate_metrics:
                metric_dict.update(validate_metrics[0])

        if evaluation_cfg.get("test", False):
            log.info("Starting evaluation test!")
            test_ckpt_path = None if evaluation_cfg.get("fit", False) else lightning_ckpt_path
            test_metrics = trainer.test(
                model=model,
                datamodule=datamodule,
                ckpt_path=test_ckpt_path,
            )
            if test_metrics:
                metric_dict.update(test_metrics[0])

        if not metric_dict:
            metric_dict = dict(trainer.callback_metrics)

    log_evaluation_metrics(metric_dict, cfg, trainer)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval/default")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
