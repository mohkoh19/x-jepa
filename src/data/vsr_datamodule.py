from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from datasets import load_dataset
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


class VSRDataset(Dataset):
    """Visual Spatial Reasoning examples backed by COCO image files."""

    def __init__(
        self,
        dataset_name: str = "cambridgeltl/vsr_zeroshot",
        split: str = "test",
        image_root: str = "data/coco/images",
        transform: object = None,
        cache_dir: str | None = None,
        max_samples: int | None = None,
        skip_missing_images: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.image_root = Path(image_root).expanduser().resolve()
        self.transform = transform
        self.dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        if skip_missing_images:
            self.dataset = self.dataset.filter(
                lambda row: self._resolve_image_path(row).exists(),
                desc="Filtering VSR rows with missing COCO images",
            )
        logger.info("Loaded %s VSR samples from %s[%s]", len(self), dataset_name, split)

    def __len__(self) -> int:
        return len(self.dataset)

    def _subdir_from_link(self, image_link: str | None) -> str | None:
        if not image_link:
            return None
        parts = [part for part in Path(urlparse(str(image_link)).path).parts if part]
        if len(parts) >= 2 and parts[-2].endswith(("2014", "2017")):
            return parts[-2]
        return None

    def _resolve_image_path(self, row: dict) -> Path:
        image_ref = Path(str(row["image"]))
        candidates = []
        if image_ref.is_absolute():
            candidates.append(image_ref)
        else:
            candidates.append(self.image_root / image_ref)
            subdir = self._subdir_from_link(row.get("image_link"))
            if subdir:
                candidates.append(self.image_root / subdir / image_ref.name)
            candidates.append(self.image_root / "train2017" / image_ref.name)
            candidates.append(self.image_root / "val2017" / image_ref.name)

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset[idx]
        image_path = self._resolve_image_path(row)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing VSR image {row['image']!r}. Looked under {self.image_root}."
            )

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "id": idx,
            "example_id": f"{self.split}:{idx}",
            "caption": str(row["caption"]),
            "label": int(row["label"]),
            "relation": str(row.get("relation") or "relation"),
            "subj": str(row.get("subj") or ""),
            "obj": str(row.get("obj") or ""),
            "filename": str(row["image"]),
        }


class VSRDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "cambridgeltl/vsr_zeroshot",
        split: str = "test",
        train_split: str | None = None,
        val_split: str | None = None,
        test_split: str | None = None,
        image_root: str = "data/coco/images",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        transform: object = None,
        cache_dir: str | None = None,
        max_samples: int | None = None,
        skip_missing_images: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["transform"], logger=False)
        self.transform = transform

    def _dataset(self, split: str) -> VSRDataset:
        return VSRDataset(
            dataset_name=self.hparams.dataset_name,
            split=split,
            image_root=self.hparams.image_root,
            transform=self.transform,
            cache_dir=self.hparams.cache_dir,
            max_samples=self.hparams.max_samples,
            skip_missing_images=self.hparams.skip_missing_images,
        )

    def setup(self, stage: str | None = None) -> None:
        split = str(self.hparams.split)
        train_split = self.hparams.train_split or split
        val_split = self.hparams.val_split or split
        test_split = self.hparams.test_split or split
        if stage in ("fit", None):
            self.train_set = self._dataset(train_split)
            self.val_set = self._dataset(val_split)
        if stage in ("validate", None):
            self.val_set = self._dataset(val_split)
        if stage in ("test", None):
            self.test_set = self._dataset(test_split)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        dataset = getattr(self, "test_set", None)
        if dataset is None:
            dataset = self.val_set
        return self._make_dataloader(dataset, shuffle=False)

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.train_set, shuffle=True)

    def _make_dataloader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        world_size = 1
        global_rank = 0
        if hasattr(self, "trainer") and self.trainer is not None:
            world_size = getattr(self.trainer, "world_size", 1)
            global_rank = getattr(self.trainer, "global_rank", 0)

        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                self.val_set,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=False,
                drop_last=False,
            )
            shuffle = False
        kwargs = {}
        if self.hparams.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            **kwargs,
        )
