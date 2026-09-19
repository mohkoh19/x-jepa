from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

log = logging.getLogger(__name__)


def _read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list or JSONL rows in {path}")
    return data


def _parse_label(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return 1
    if normalized in {"false", "0"}:
        return 0
    raise ValueError(f"Unsupported NLVR2 label: {value!r}")


def _candidate_files(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _identifier_image_paths(identifier: str) -> tuple[str, str]:
    parts = identifier.split("-")
    if len(parts) < 4:
        raise ValueError(f"Cannot infer NLVR2 image paths from identifier: {identifier}")
    prefix = "-".join(parts[:3])
    return f"{prefix}-img0.png", f"{prefix}-img1.png"


class NLVR2Dataset(Dataset):
    """NLVR2 rows with two images, one sentence, and a binary label."""

    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        transform: object = None,
        skip_missing_images: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.annotation_file = Path(annotation_file).expanduser().resolve()
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.transform = transform
        self.skip_missing_images = bool(skip_missing_images)
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Missing NLVR2 annotation file: {self.annotation_file}")
        rows = _read_rows(self.annotation_file)
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        if self.skip_missing_images:
            before = len(rows)
            rows = [row for row in rows if self._has_both_images(row)]
            skipped = before - len(rows)
            if skipped:
                log.warning(
                    "Skipped %s NLVR2 examples with missing images from %s",
                    skipped,
                    self.annotation_file,
                )
        self.rows = rows
        log.info("Loaded %s NLVR2 examples from %s", len(self.rows), self.annotation_file)

    def __len__(self) -> int:
        return len(self.rows)

    def _split_image_roots(self, raw_name: str) -> list[Path]:
        def _by_annotation_stem() -> list[str]:
            stem = self.annotation_file.stem.lower()
            if stem.startswith("train"):
                return ["train_img/images/train", "train_img"]
            if stem.startswith(("dev", "val")):
                return ["dev_img/dev", "dev_img"]
            if stem.startswith("test1"):
                return ["test1_img/test1", "test1_img"]
            if stem.startswith("test2"):
                return ["test2_img/test2", "test2_img"]
            if stem.startswith("test"):
                return ["test1_img/test1", "test1_img", "test2_img/test2", "test2_img"]
            return []

        def _by_filename_prefix() -> list[str]:
            prefix = raw_name.split("-", 1)[0].lower()
            if prefix == "train":
                return ["train_img/images/train", "train_img"]
            if prefix in {"dev", "val"}:
                return ["dev_img/dev", "dev_img"]
            if prefix == "test1":
                return ["test1_img/test1", "test1_img"]
            if prefix == "test2":
                return ["test2_img/test2", "test2_img"]
            return []

        roots: list[Path] = []
        seen: set[Path] = set()

        for rel in [*_by_annotation_stem(), *_by_filename_prefix()]:
            candidate = self.image_dir / rel
            if candidate in seen:
                continue
            seen.add(candidate)
            roots.append(candidate)
        return roots

    def _resolve_image(self, raw_path: str, row: dict) -> Path:
        path = Path(raw_path)
        candidates: list[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            split_roots = self._split_image_roots(path.name)
            for root in split_roots:
                candidates.append(root / path)
                candidates.append(root / path.name)
                if row.get("directory") is not None:
                    row_dir = str(row["directory"])
                    candidates.append(root / row_dir / path)
                    candidates.append(root / row_dir / path.name)

            candidates.append(self.image_dir / path)
            candidates.append(self.image_dir / path.name)
            if row.get("directory") is not None:
                candidates.append(self.image_dir / str(row["directory"]) / path.name)

        seen: set[Path] = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)
        for candidate in unique_candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Missing NLVR2 image. Checked: "
            + ", ".join(str(candidate) for candidate in unique_candidates)
        )

    def _has_both_images(self, row: dict) -> bool:
        try:
            left_path, right_path = self._row_image_paths(row)
            self._resolve_image(left_path, row)
            self._resolve_image(right_path, row)
        except (FileNotFoundError, ValueError):
            return False
        return True

    def _row_image_paths(self, row: dict) -> tuple[str, str]:
        if row.get("image_left") and row.get("image_right"):
            return row["image_left"], row["image_right"]
        if row.get("left_image") and row.get("right_image"):
            return row["left_image"], row["right_image"]
        if row.get("identifier"):
            return _identifier_image_paths(row["identifier"])
        raise ValueError("NLVR2 row must contain image_left/image_right or identifier.")

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        left_path, right_path = self._row_image_paths(row)
        left = Image.open(self._resolve_image(left_path, row)).convert("RGB")
        right = Image.open(self._resolve_image(right_path, row)).convert("RGB")
        if self.transform:
            left = self.transform(left)
            right = self.transform(right)

        sentence = row["sentence"]
        identifier = row.get("identifier") or f"{self.annotation_file.stem}-{idx}"
        return {
            "image_left": left,
            "image_right": right,
            "sentence": sentence,
            "label": _parse_label(row["label"]),
            "identifier": identifier,
        }


class NLVR2DataModule(LightningDataModule):
    def __init__(
        self,
        annotation_dir: str,
        image_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_transform: object = None,
        val_transform: object = None,
        train_file: str | list[str] = "train.json",
        val_file: str | list[str] = ("val.json", "dev.json"),
        test_file: str | list[str] = ("test.json", "test1.json"),
        skip_missing_images: bool = False,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["train_transform", "val_transform"], logger=False)
        self.train_transform = train_transform
        self.val_transform = val_transform

    def _resolve_annotation_file(self, names: str | Iterable[str]) -> str:
        annotation_dir = Path(self.hparams.annotation_dir)
        candidates = [annotation_dir / name for name in _candidate_files(names)]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            "Missing NLVR2 annotation file. Checked: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    def _dataset(self, names: str | Iterable[str], transform: object = None) -> NLVR2Dataset:
        return NLVR2Dataset(
            annotation_file=self._resolve_annotation_file(names),
            image_dir=self.hparams.image_dir,
            transform=transform,
            skip_missing_images=bool(self.hparams.skip_missing_images),
            max_samples=self.hparams.max_samples,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_set = self._dataset(self.hparams.train_file, self.train_transform)
            self.val_set = self._dataset(self.hparams.val_file, self.val_transform)
        if stage in ("validate", None):
            self.val_set = self._dataset(self.hparams.val_file, self.val_transform)
        if stage in ("test", None):
            self.test_set = self._dataset(self.hparams.test_file, self.val_transform)

    def _distributed_sampler(self, dataset: Dataset):
        world_size = getattr(getattr(self, "trainer", None), "world_size", 1)
        global_rank = getattr(getattr(self, "trainer", None), "global_rank", 0)
        if world_size <= 1:
            return None
        return DistributedSampler(
            dataset, num_replicas=world_size, rank=global_rank, shuffle=False, drop_last=False
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
