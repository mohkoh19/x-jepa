"""Shared DataLoader construction for the VL4M pretraining datamodules."""

from __future__ import annotations

import logging

import torchvision.transforms as transforms
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BaseDatamodule(LightningDataModule):
    """Common train/validation DataLoader setup.

    Training always uses a :class:`DistributedSampler` — also in single-process
    runs — so that shuffling is driven by ``sampler.set_epoch``.  Validation
    uses a distributed sampler only when more than one process is active.
    """

    def train_dataloader(self):
        return self._to_dataloader(self.train_set, self.hparams.batch_size, train=True)

    def val_dataloader(self):
        return self._to_dataloader(self.val_set, self.hparams.batch_size, train=False)

    def _default_image_transform(self, crop_size: int = 224):
        return transforms.Compose(
            [
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def _init_transform(self):
        self.train_transform = self.hparams.train_transform or self._default_image_transform()
        self.val_transform = self.hparams.val_transform or self._default_image_transform()

    def _to_dataloader(self, dataset, batch_size: int, *, train: bool):
        world_size = 1
        global_rank = 0
        if getattr(self, "trainer", None) is not None:
            world_size = getattr(self.trainer, "world_size", 1)
            global_rank = getattr(self.trainer, "global_rank", 0)
        is_distributed = world_size > 1

        sampler = None
        if train:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=True,
                drop_last=True,
            )
        elif is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=False,
                drop_last=False,
            )

        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "collate_fn": self.collate_fn,
            "num_workers": self.hparams.num_workers,
            "pin_memory": self.hparams.pin_memory,
            "sampler": sampler,
        }
        if sampler is None:
            dataloader_kwargs["shuffle"] = False
        if self.hparams.num_workers > 0:
            dataloader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2,
            )

        return DataLoader(**dataloader_kwargs)
