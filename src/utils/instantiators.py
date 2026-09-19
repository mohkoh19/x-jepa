from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def _instantiate_collection(config: DictConfig, kind: str) -> list:
    objects: list = []

    if not config:
        log.warning("No %s configs found! Skipping...", kind)
        return objects

    if not isinstance(config, DictConfig):
        raise TypeError(f"{kind.title()} config must be a DictConfig!")

    singular = kind[:-1] if kind.endswith("s") else kind
    for _, item_cfg in config.items():
        if isinstance(item_cfg, DictConfig) and "_target_" in item_cfg:
            log.info("Instantiating %s <%s>", singular, item_cfg._target_)
            objects.append(hydra.utils.instantiate(item_cfg))

    return objects


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    return _instantiate_collection(callbacks_cfg, kind="callbacks")


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    return _instantiate_collection(logger_cfg, kind="loggers")
