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


class CocoKarpathyRetrievalDataset(CaptionPreprocessingMixin, Dataset):
    """COCO Karpathy retrieval split with five captions per image."""

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
            raise FileNotFoundError(
                f"Missing COCO Karpathy annotation file: {self.annotation_file}"
            )

        with self.annotation_file.open("r", encoding="utf-8") as handle:
            annotations = json.load(handle)
        if not isinstance(annotations, list):
            raise ValueError(f"Expected a JSON list in {self.annotation_file}")

        self.annotations = self._normalise_annotations(annotations)
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

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
            "Loaded %s COCO Karpathy images and %s captions from %s",
            len(self.image),
            len(self.text),
            self.annotation_file,
        )

    def _normalise_annotations(self, annotations: list[dict]) -> list[dict]:
        """Return one record per image with a list of captions.

        The Karpathy val/test files used for retrieval commonly contain one
        JSON object per image with ``caption`` as a list.  Some COCO train
        files instead contain one JSON object per image-caption pair with
        ``caption`` as a string and repeated
        ``image`` values.  Retrieval adapter tuning needs image-unique samples
        with one caption sampled per image, so we canonicalise both formats to
        the same one-image -> many-captions representation.
        """

        grouped: dict[str, dict] = {}
        ordered_images: list[str] = []
        for row_idx, row in enumerate(annotations):
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected annotation object at index {row_idx} in {self.annotation_file}"
                )
            image = row.get("image")
            if not isinstance(image, str) or not image:
                raise ValueError(
                    f"Missing image path for annotation index {row_idx} in {self.annotation_file}"
                )

            raw_captions = row.get("caption")
            if isinstance(raw_captions, str):
                captions = [raw_captions]
            elif isinstance(raw_captions, list):
                captions = [caption for caption in raw_captions if isinstance(caption, str)]
            else:
                captions = []
            if not captions:
                raise ValueError(
                    f"Missing caption string/list for annotation index {row_idx} in {self.annotation_file}"
                )

            if image not in grouped:
                canonical = dict(row)
                canonical["caption"] = []
                grouped[image] = canonical
                ordered_images.append(image)
            grouped[image]["caption"].extend(captions)

        return [grouped[image] for image in ordered_images]

    def __len__(self) -> int:
        return len(self.image)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_root / self.annotations[idx]["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing COCO Karpathy image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "image_id": idx,
            "filename": self.annotations[idx]["image"],
            **(
                {"caption": self.pre_caption(random.choice(self.annotations[idx]["caption"]))}
                if self.split == "train"
                else {}
            ),
        }


class CocoKarpathyRetrievalDataModule(LightningDataModule):
    def __init__(
        self,
        image_root: str,
        annotation_file: str | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        transform: object = None,
        train_transform: object = None,
        val_transform: object = None,
        max_words: int = 64,
        annotation_dir: str | None = None,
        train_file: str = "coco_karpathy_train.json",
        val_file: str = "coco_karpathy_val.json",
        test_file: str = "coco_karpathy_test.json",
        eval_split: str = "test",
    ) -> None:
        super().__init__()
        if eval_split not in {"val", "test"}:
            raise ValueError("COCO Karpathy eval_split must be `val` or `test`.")
        self.save_hyperparameters(
            ignore=["transform", "train_transform", "val_transform"], logger=False
        )
        self.train_transform = train_transform or transform
        self.val_transform = val_transform or transform

    def _annotation_file(self, split: str) -> str:
        if self.hparams.annotation_dir:
            annotation_dir = Path(self.hparams.annotation_dir)
            if split == "train":
                return str(annotation_dir / self.hparams.train_file)
            if split == "val":
                return str(annotation_dir / self.hparams.val_file)
            if split == "test":
                return str(annotation_dir / self.hparams.test_file)
            raise ValueError(f"Unsupported COCO Karpathy split: {split}")
        if self.hparams.annotation_file is None:
            raise ValueError(
                "COCO Karpathy datamodule requires annotation_file or annotation_dir."
            )
        return str(self.hparams.annotation_file)

    def _dataset(self, split: str, transform: object) -> CocoKarpathyRetrievalDataset:
        return CocoKarpathyRetrievalDataset(
            annotation_file=self._annotation_file(split),
            image_root=self.hparams.image_root,
            transform=transform,
            max_words=self.hparams.max_words,
            split=split,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_set = self._dataset("train", self.train_transform)
            self.val_set = self._dataset("val", self.val_transform)
        if stage in ("validate", None):
            self.val_set = self._dataset(str(self.hparams.eval_split), self.val_transform)
        if stage in ("test", None):
            self.test_set = self._dataset("test", self.val_transform)

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
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        world_size = 1
        global_rank = 0
        if hasattr(self, "trainer") and self.trainer is not None:
            world_size = getattr(self.trainer, "world_size", 1)
            global_rank = getattr(self.trainer, "global_rank", 0)

        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                self.test_set,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=False,
                drop_last=False,
            )
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            sampler=sampler,
            **self._loader_kwargs(),
        )
