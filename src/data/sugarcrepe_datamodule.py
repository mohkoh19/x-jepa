from __future__ import annotations

import json
import logging
from pathlib import Path

from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)

DEFAULT_SUGARCREPE_FILES = [
    "replace_attribute_train.jsonl",
    "replace_object_train.jsonl",
    "replace_relation_train.jsonl",
    "swap_atribute_train.jsonl",
    "swap_object_train.jsonl",
]

CATEGORY_ALIASES = {
    "swap_atribute": "swap_attribute",
}


class SugarCrepePPDataset(Dataset):
    """SugarCrepe++ image-text triplets stored as local JSONL files."""

    def __init__(
        self,
        annotation_dir: str,
        image_dir: str,
        transform: object = None,
        files: list[str] | None = None,
    ) -> None:
        self.annotation_dir = Path(annotation_dir).expanduser().resolve()
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.transform = transform
        self.files = files or DEFAULT_SUGARCREPE_FILES
        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict]:
        samples = []
        for file_name in self.files:
            path = self.annotation_dir / file_name
            if not path.exists():
                raise FileNotFoundError(f"Missing SugarCrepe++ annotation file: {path}")

            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    category = CATEGORY_ALIASES.get(row["category"], row["category"])
                    samples.append(
                        {
                            "id": row["id"],
                            "filename": row["filename"],
                            "caption": row["caption"],
                            "caption2": row["caption2"],
                            "negative_caption": row["negative_caption"],
                            "category": category,
                            "source_file": file_name,
                            "line_number": line_number,
                        }
                    )

        logger.info("Loaded %s SugarCrepe++ samples from %s", len(samples), self.annotation_dir)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_path = self.image_dir / sample["filename"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing SugarCrepe++ image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "caption": sample["caption"],
            "caption2": sample["caption2"],
            "negative_caption": sample["negative_caption"],
            "category": sample["category"],
            "id": sample["id"],
            "example_id": f"{sample['category']}:{sample['id']}",
            "filename": sample["filename"],
            "source_file": sample["source_file"],
            "line_number": sample["line_number"],
        }


class SugarCrepePPDataModule(LightningDataModule):
    def __init__(
        self,
        annotation_dir: str,
        image_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        transform: object = None,
        files: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["transform"], logger=False)
        self.transform = transform

    def setup(self, stage: str | None = None) -> None:
        if stage in ("validate", "test", "fit", None):
            self.val_set = SugarCrepePPDataset(
                annotation_dir=self.hparams.annotation_dir,
                image_dir=self.hparams.image_dir,
                transform=self.transform,
                files=self.hparams.files,
            )

    def val_dataloader(self) -> DataLoader:
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
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
