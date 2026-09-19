from __future__ import annotations

import csv
import logging
from pathlib import Path

from datasets import load_dataset
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _negative_type(row: dict[str, str]) -> str:
    types = []
    if _parse_bool(row.get("subj_neg")):
        types.append("subject")
    if _parse_bool(row.get("verb_neg")):
        types.append("verb")
    if _parse_bool(row.get("obj_neg")):
        types.append("object")
    return "_".join(types) if types else "unknown"


class SVOProbesDataset(Dataset):
    """SVO-Probes sentence with positive/negative image pairs."""

    def __init__(
        self,
        dataset_name: str = "MichiganNLP/svo_probes",
        split: str = "train",
        annotation_file: str | None = None,
        image_root: str = "data/svo_probes/images",
        transform: object = None,
        cache_dir: str | None = None,
        max_samples: int | None = None,
        skip_missing_images: bool = True,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.annotation_file = (
            Path(annotation_file).expanduser().resolve() if annotation_file else None
        )
        self.image_root = Path(image_root).expanduser().resolve()
        self.transform = transform
        self.cache_dir = cache_dir
        self.skip_missing_images = skip_missing_images
        self.samples = self._load_samples()
        if max_samples is not None:
            self.samples = self.samples[: max(0, max_samples)]
        source = str(self.annotation_file) if self.annotation_file else f"{dataset_name}[{split}]"
        logger.info("Loaded %s SVO-Probes samples from %s", len(self), source)

    def _image_path(self, image_id: str | int) -> Path:
        return self.image_root / f"{int(image_id):06d}.jpg"

    def _iter_rows(self):
        if self.annotation_file is None:
            yield from load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
            )
            return

        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"Missing SVO-Probes annotation file: {self.annotation_file}. "
                "Run `python scripts/download_svo_probes.py --root data/svo_probes` first."
            )
        with self.annotation_file.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)

    def _load_samples(self) -> list[dict]:
        samples = []
        skipped = 0
        for row_idx, row in enumerate(self._iter_rows()):
            pos_path = self._image_path(row["pos_image_id"])
            neg_path = self._image_path(row["neg_image_id"])
            if self.skip_missing_images and (not pos_path.exists() or not neg_path.exists()):
                skipped += 1
                continue
            samples.append(
                {
                    "id": row_idx,
                    "sentence": str(row["sentence"]),
                    "pos_image_id": int(row["pos_image_id"]),
                    "neg_image_id": int(row["neg_image_id"]),
                    "pos_triplet": str(row["pos_triplet"]),
                    "neg_triplet": str(row["neg_triplet"]),
                    "negative_type": _negative_type(row),
                }
            )

        if skipped:
            logger.warning("Skipped %s SVO-Probes rows with missing images.", skipped)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        pos_path = self._image_path(sample["pos_image_id"])
        neg_path = self._image_path(sample["neg_image_id"])
        if not pos_path.exists() or not neg_path.exists():
            raise FileNotFoundError(f"Missing SVO-Probes image pair: {pos_path}, {neg_path}.")

        pos_image = Image.open(pos_path).convert("RGB")
        neg_image = Image.open(neg_path).convert("RGB")
        if self.transform:
            pos_image = self.transform(pos_image)
            neg_image = self.transform(neg_image)

        return {
            "pos_image": pos_image,
            "neg_image": neg_image,
            "id": sample["id"],
            "example_id": f"{self.split}:{sample['id']}",
            "sentence": sample["sentence"],
            "negative_type": sample["negative_type"],
            "negative_type_native": sample["negative_type"],
            "pos_triplet": sample["pos_triplet"],
            "neg_triplet": sample["neg_triplet"],
            "pos_image_id": sample["pos_image_id"],
            "neg_image_id": sample["neg_image_id"],
        }


class SVOProbesDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "MichiganNLP/svo_probes",
        split: str = "train",
        annotation_file: str | None = None,
        image_root: str = "data/svo_probes/images",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        transform: object = None,
        cache_dir: str | None = None,
        max_samples: int | None = None,
        skip_missing_images: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["transform"], logger=False)
        self.transform = transform

    def setup(self, stage: str | None = None) -> None:
        if stage in ("validate", "test", "fit", None):
            self.val_set = SVOProbesDataset(
                dataset_name=self.hparams.dataset_name,
                split=self.hparams.split,
                annotation_file=self.hparams.annotation_file,
                image_root=self.hparams.image_root,
                transform=self.transform,
                cache_dir=self.hparams.cache_dir,
                max_samples=self.hparams.max_samples,
                skip_missing_images=self.hparams.skip_missing_images,
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
