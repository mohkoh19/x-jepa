from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.data.components.captioning import CaptionPreprocessingMixin

log = logging.getLogger(__name__)


class Flickr30KRetrievalDataset(CaptionPreprocessingMixin, Dataset):
    """Flickr30K Karpathy split for retrieval evaluation or contrastive fitting."""

    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        transform: object = None,
        max_words: int = 64,
        split: str = "test",
    ) -> None:
        self.annotation_file = Path(annotation_file).expanduser().resolve()
        self.image_root = Path(image_root).expanduser().resolve()
        self.transform = transform
        self.max_words = max_words
        self.split = split

        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Missing Flickr30K annotation file: {self.annotation_file}")

        with self.annotation_file.open("r", encoding="utf-8") as handle:
            annotations = json.load(handle)
        if not isinstance(annotations, list):
            raise ValueError(f"Expected a JSON list in {self.annotation_file}")

        self.annotations = (
            self._unique_train_annotations(annotations) if split == "train" else annotations
        )
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        if split == "train":
            for idx, row in enumerate(self.annotations):
                self.image.append(row["image"])
                captions = row["caption"]
                if isinstance(captions, list):
                    self.text.extend(self.pre_caption(caption) for caption in captions)
                else:
                    self.text.append(self.pre_caption(captions))
                self.img2txt[idx] = [idx]
                self.txt2img[idx] = idx
        else:
            text_id = 0
            for image_id, row in enumerate(self.annotations):
                captions = row.get("caption")
                if not isinstance(captions, list) or not captions:
                    raise ValueError(
                        f"Missing caption list for image index {image_id} in {self.annotation_file}"
                    )

                self.image.append(row["image"])
                self.img2txt[image_id] = []
                for caption in captions:
                    self.text.append(self.pre_caption(caption))
                    self.img2txt[image_id].append(text_id)
                    self.txt2img[text_id] = image_id
                    text_id += 1

        log.info(
            "Loaded %s Flickr30K %s images and %s captions from %s",
            len(self.image),
            split,
            len(self.text),
            self.annotation_file,
        )

    @staticmethod
    def _unique_train_annotations(annotations: list[dict]) -> list[dict]:
        by_image: dict[str, list[str]] = {}
        for row in annotations:
            image = str(row["image"])
            captions = row.get("caption", [])
            if not isinstance(captions, list):
                captions = [captions]
            by_image.setdefault(image, []).extend(str(caption) for caption in captions)
        return [{"image": image, "caption": captions} for image, captions in by_image.items()]

    def __len__(self) -> int:
        return len(self.image)

    def _image_path(self, relative_path: str) -> Path:
        direct = self.image_root / relative_path
        if direct.exists():
            return direct
        by_name = self.image_root / Path(relative_path).name
        if by_name.exists():
            return by_name
        raise FileNotFoundError(f"Missing Flickr30K image: {direct} or {by_name}")

    def __getitem__(self, idx: int) -> dict:
        row = self.annotations[idx]
        image_path = self._image_path(row["image"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        item = {
            "image": image,
            "image_id": idx,
            "filename": row["image"],
        }
        if self.split == "train":
            captions = row["caption"]
            if isinstance(captions, list):
                item["caption"] = self.pre_caption(random.choice(captions))
            else:
                item["caption"] = self.pre_caption(captions)
        return item


class Flickr30KRetrievalDataModule(LightningDataModule):
    def __init__(
        self,
        annotation_dir: str,
        image_root: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_transform: object = None,
        val_transform: object = None,
        max_words: int = 64,
        train_file: str = "flickr30k_train.json",
        val_file: str = "flickr30k_val.json",
        test_file: str = "flickr30k_test.json",
        eval_split: str = "test",
    ) -> None:
        super().__init__()
        if eval_split not in {"val", "test"}:
            raise ValueError("Flickr30K eval_split must be `val` or `test`.")
        self.save_hyperparameters(ignore=["train_transform", "val_transform"], logger=False)
        self.train_transform = train_transform
        self.val_transform = val_transform

    def _annotation_file(self, split: str) -> str:
        annotation_dir = Path(self.hparams.annotation_dir)
        if split == "train":
            return str(annotation_dir / self.hparams.train_file)
        if split == "val":
            return str(annotation_dir / self.hparams.val_file)
        if split == "test":
            return str(annotation_dir / self.hparams.test_file)
        raise ValueError(f"Unsupported Flickr30K split: {split}")

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_set = Flickr30KRetrievalDataset(
                annotation_file=self._annotation_file("train"),
                image_root=self.hparams.image_root,
                transform=self.train_transform,
                max_words=self.hparams.max_words,
                split="train",
            )
            self.val_set = Flickr30KRetrievalDataset(
                annotation_file=self._annotation_file("val"),
                image_root=self.hparams.image_root,
                transform=self.val_transform,
                max_words=self.hparams.max_words,
                split="val",
            )
        if stage in ("validate", None):
            split = str(self.hparams.eval_split)
            self.val_set = Flickr30KRetrievalDataset(
                annotation_file=self._annotation_file(split),
                image_root=self.hparams.image_root,
                transform=self.val_transform,
                max_words=self.hparams.max_words,
                split=split,
            )
        if stage in ("test", None):
            self.test_set = Flickr30KRetrievalDataset(
                annotation_file=self._annotation_file("test"),
                image_root=self.hparams.image_root,
                transform=self.val_transform,
                max_words=self.hparams.max_words,
                split="test",
            )

    def _distributed_sampler(self, dataset: Dataset):
        world_size = getattr(getattr(self, "trainer", None), "world_size", 1)
        global_rank = getattr(getattr(self, "trainer", None), "global_rank", 0)
        if world_size <= 1:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=False,
            drop_last=False,
        )

    def _loader_kwargs(self) -> dict:
        kwargs = {
            "num_workers": self.hparams.num_workers,
            "pin_memory": self.hparams.pin_memory,
        }
        if self.hparams.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        sampler = self._distributed_sampler(self.val_set)
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            sampler=sampler,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        sampler = self._distributed_sampler(self.test_set)
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            sampler=sampler,
            **self._loader_kwargs(),
        )
